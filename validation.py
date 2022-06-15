import argparse
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import build_model
from utils.coco_eval import CocoEvaluator
from utils.coco_utils import get_coco_api_from_dataset
from utils.dataset import FasterDetrDataset
from utils.misc import collate_fn
from utils.utils import get_classes, make_coco_transforms
from torch.nn.parallel import DistributedDataParallel


def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 12, [""] * 12
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
    stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
    stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
    stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
    stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info


def main(args):
    # num_classes class_names max_map min_loss 初始化
    class_names, num_classes = get_classes(args.class_path)
    # num_workers
    num_workers = min(min([os.cpu_count(), args.batch_size if args.batch_size >= 1 else 0, 8]),
                      args.num_workers)  # number of workers
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    with open(args.val, encoding='utf-8') as f:
        val_lines = f.readlines()

    # 验证与训练保持一致
    model, criterion, post_processors = build_model(args, 21)

    category_index = dict(zip(range(1, len(class_names) + 1), class_names))

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    print('Using %g dataloader workers' % num_workers)

    # load validation data set
    dataset_val = FasterDetrDataset(val_lines, transforms=make_coco_transforms("val"), class_names=class_names)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val,
                                 args.batch_size,
                                 sampler=sampler_val,
                                 drop_last=False,
                                 collate_fn=collate_fn,
                                 num_workers=args.num_workers)

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
    postprocessors = post_processors
    base_ds = get_coco_api_from_dataset(dataset_val)
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    with tqdm(total=len(data_loader_val)) as pbar:
        for samples, targets in data_loader_val:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(samples)

            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            if 'segm' in postprocessors.keys():
                target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            if coco_evaluator is not None:
                coco_evaluator.update(res)
            pbar.update(1)

    # gather the stats from all processes
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    coco_eval = coco_evaluator.coco_eval["bbox"]
    voc_map_info_list = []
    for i in range(len(category_index)):
        stats, _ = summarize(coco_eval, catId=i)
        voc_map_info_list.append(" {:15}: {}".format(category_index[i + 1], stats[1]))

    print_voc = "\n".join(voc_map_info_list)
    print(print_voc)

    # 将验证结果保存至txt文件中
    with open(os.path.join(args.save_dir, "record_mAP.txt"), "w") as f:
        record_lines = ["mAP(IoU=0.5) for each category:",
                        print_voc]
        f.write("\n".join(record_lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SAM-DETR: Accelerating DETR Convergence via Semantic-Aligned Matching',
                                     add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=[], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    parser.add_argument('--val', type=str, default=r"weights/val.txt", help="val_txt_path")
    parser.add_argument('--save_dir', type=str, default="weights", help='save_dir')
    parser.add_argument('--class_path', type=str, default=r"weights/voc_classes.txt", help='classes_path')
    parser.add_argument('--weights_path', default='weights/loss_20220612222954/resnet50_FasterDETR_bestMap.pth', type=str,
                        help='training weights')
    parser.add_argument('--device', default='cuda:6', help='device')
    parser.add_argument('--num_workers', type=int, default=24, help="num_workers")

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
