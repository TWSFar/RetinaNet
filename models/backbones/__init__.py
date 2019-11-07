import os.path as osp
from models.backbones import hrnet, resnet
from yacs.config import CfgNode as CN


def make_backbone(opt):
    if opt.backbone == 'resnet101':
        return resnet.resnet101()

    elif opt.backbone == 'resnet50':
        return resnet.resnet50()

    elif 'hrnet' in opt.backbone:
        cfg = CN()
        cfg.NAME = opt.backbone
        cfg.MODEL = CN(new_allowed=True)
        cfg.defrost()
        hrnet_cfg = opt.hrnet_cfg
        cfg.merge_from_file(hrnet_cfg)
        cfg.freeze()

        return hrnet.hrnet(cfg)

    else:
        raise NotImplementedError
