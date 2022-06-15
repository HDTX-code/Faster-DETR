import argparse
import math
import os
import colorsys

import time
import torch
from PIL import Image

from models import build_model
from utils.utils import get_classes, make_coco_transforms


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main(args):
    # num_classes class_names max_map min_loss 初始化
    class_names, num_classes = get_classes(args.class_path)
    # get devices
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # model初始化, 验证与训练保持一致
    model, criterion, post_processors = build_model(args, 21)
    # 载入你自己训练好的模型权重
    assert os.path.exists(args.weights_path), "not found {} file.".format(args.weights_path)
    # 不小心保存的是多卡并行的权重，要去权重key掉前面的字符
    dict_p = torch.load(args.weights_path, map_location='cpu')
    for key in list(dict_p.keys()):
        # 修改参数名
        dict_p[key[7:]] = dict_p.pop(key)
    model.load_state_dict(dict_p)
    # print(model)

    model.to(device)
    model.eval()

    category_index = dict(zip(range(1, len(class_names) + 1), class_names))

    transforms = make_coco_transforms("val")
    DETECTION_THRESHOLD = 0.5

    with torch.no_grad():
        img = Image.open(args.pic_path).convert("RGB")
        w, h = img.size
        orig_target_sizes = torch.tensor([[h, w]], device=device)
        img, _ = transforms(img, target=None)
        img = img.to(device)
        img = img.unsqueeze(0)  # adding batch dimension
        t_start = time_synchronized()
        outputs = model(img)
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))
        results = post_processors['bbox'](outputs, orig_target_sizes)[0]
        indexes = results['scores'] >= DETECTION_THRESHOLD
        scores = results['scores'][indexes].to("cpu").numpy()
        labels = results['labels'][indexes].to("cpu").numpy()
        boxes = results['boxes'][indexes].to("cpu").numpy()

        # Visualize the detection results
        import cv2
        hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        img_det_result = cv2.imread(args.pic_path)
        for i in range(scores.shape[0]):
            x1, y1, x2, y2 = round(float(boxes[i, 0])), round(float(boxes[i, 1])), round(float(boxes[i, 2])), round(
                float(boxes[i, 3]))
            img_det_result = cv2.rectangle(img_det_result, (x1, y1), (x2, y2), colors[labels[i]], 2)
            img = cv2.putText(img_det_result, category_index[labels[i]] + ":" + str(math.floor(scores[i] * 100)) + "%",
                              (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX, 0.3, colors[labels[i]], 1)
        cv2.imwrite(os.path.join(args.save_dir, "test.jpg"), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SAM-DETR: Accelerating DETR Convergence via Semantic-Aligned Matching',
                                     add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=[], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    parser.add_argument('--val', type=str, default=r"weights/val.txt", help="val_txt_path")
    parser.add_argument('--save_dir', type=str, default="weights/loss_20220612222954", help='save_dir')
    parser.add_argument('--class_path', type=str, default=r"weights/voc_classes.txt", help='classes_path')
    parser.add_argument('--weights_path', default='weights/loss_20220612222954/resnet50_FasterDETR_bestMap.pth',
                        type=str,
                        help='training weights')
    parser.add_argument('--pic_path', default=r'/Home/atr2/homefun/zhf/faster_rcnn/faster_rcnn/data/VOCdevkit/VOC2007/JPEGImages/2009_003351.jpg', type=str,
                        help='pic_path')
    parser.add_argument('--device', default='cpu', help='device')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--multiscale', default=False, action='store_true')

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine',),
                        help="Type of positional embedding to use on top of the image features")
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, help="dimension of the FFN in the transformer")
    parser.add_argument('--hidden_dim', default=256, type=int, help="dimension of the transformer")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads in the transformer attention")
    parser.add_argument('--num_queries', default=300, type=int, help="Number of query slots")

    parser.add_argument('--smca', default=False, action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true', help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2.0, type=float, help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5.0, type=float, help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2.0, type=float, help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1.0, type=float)
    parser.add_argument('--dice_loss_coef', default=1.0, type=float)
    parser.add_argument('--cls_loss_coef', default=2.0, type=float)
    parser.add_argument('--bbox_loss_coef', default=5.0, type=float)
    parser.add_argument('--giou_loss_coef', default=2.0, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    args = parser.parse_args()
    main(args)
