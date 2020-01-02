from .hrnet import hrnet
from .resnet import resnet50, resnet101


def build_backbone(opt):
    if opt.backbone == 'resnet101':
        return resnet101()

    elif opt.backbone == 'resnet50':
        return resnet50()

    elif 'hrnet' in opt.backbone:
        return hrnet(opt.backbone)

    else:
        raise NotImplementedError
