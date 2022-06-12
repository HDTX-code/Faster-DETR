import cv2
import numpy as np

import utils.transforms as T


# ---------------------------------------------------#
#   获得学习率
# ---------------------------------------------------#
def get_lr(optimizer):
    lr_list = []
    for param_group in optimizer.param_groups:
        lr_list.append(param_group['lr'])
    return lr_list


# ---------------------------------------------------#
#   获得类
# ---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


# ---------------------------------------------------#
#   transforms for dataset
# ---------------------------------------------------#
def make_coco_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


# ---------------------------------------------------#
#   show image
# ---------------------------------------------------#
def show_img(image, target):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    boxes = target['boxes']
    for i in range(target['image_id'].shape[0]):
        x1, y1, x2, y2 = round(float(boxes[i, 0])), round(float(boxes[i, 1])), round(float(boxes[i, 2])), round(
            float(boxes[i, 3]))
        img_det_result = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("{}".format(i), img_det_result)

