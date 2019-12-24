from .hrnet import hrnet
from .resnet import resnet50, resnet101
from yacs.config import CfgNode as CN


def build_backbone(opt):
    if opt.backbone == 'resnet101':
        return resnet101()

    elif opt.backbone == 'resnet50':
        return resnet50()

    elif 'hrnet' in opt.backbone:
        cfg = CN()
        cfg.NAME = opt.backbone
        cfg.MODEL = CN(new_allowed=True)
        cfg.defrost()
        hrnet_cfg = opt.hrnet_cfg
        cfg.merge_from_file(hrnet_cfg)
        cfg.freeze()

        return hrnet(cfg)

    else:
        raise NotImplementedError
