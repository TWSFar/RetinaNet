from .focal_loss import FocalLoss


def build_loss(opt):
    if opt.loss_cls_type == 'focalloss':
        return FocalLoss()

    else:
        raise NotImplementedError
