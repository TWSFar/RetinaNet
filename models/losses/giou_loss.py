import torch
import torch.nn as nn
from .utils import (weight_reduce_loss, decode, xyxy_2_xywh,
                    bbox_iou)


class GIoULoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(GIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, anchor):
        """
        Args:
            pred: [N_max, 4],  4: dx, dy, dw, dh
            target: [N_max, 4]  4: x1, y1, x2, y2
            anchor: [N_max, 4]  4: x1, y1, x2 ,y2
        Return:
            loss
        """
        assert pred.shape[0] == target.shape[0]
        device = pred.device

        if target.shape[0] > 0:
            variances = torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).to(device)
            anchor = xyxy_2_xywh(anchor)
            pred_box = decode(pred, anchor, variances)

            giou = bbox_iou(pred_box, target, GIoU=True).clamp(min=self.eps)
            loss = -giou.log()

            loss = weight_reduce_loss(loss,
                                      weight=self.loss_weight,
                                      reduction=self.reduction)
        else:
            loss = torch.tensor(0).float().to(device)

        return loss
