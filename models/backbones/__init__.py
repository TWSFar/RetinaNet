from .hrnet import hrnet
from .resnet import resnet50, resnet101
from .res2net import (res2net101, res2next50_32x4d,
                      res2next101_32x8d, se_res2net50)


def build_backbone(backbone):
    if backbone == 'resnet101':
        return resnet101()

    elif backbone == 'resnet50':
        return resnet50()

    elif 'hrnet' in backbone:
        return hrnet(backbone)

    elif backbone == 'res2next101_32x8d':
        return res2next101_32x8d()
    else:
        raise NotImplementedError
