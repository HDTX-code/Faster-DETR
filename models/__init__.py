from .fast_detr import build


def build_model(args, num_classes):
    return build(args, num_classes)
