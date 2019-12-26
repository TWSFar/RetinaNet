import torch
import torch.nn as nn
from .utils import (weight_reduce_loss, encode,
                    xyxy_2_xywh)


class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
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
        assert self.beta > 0
        assert pred.shape[0] == target.shape[0]

        device = pred.device
        if target.shape[0] > 0:
            anchor = xyxy_2_xywh(anchor)
            variances = torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).to(device)
            gt = encode(target, anchor, variances)

            diff = torch.abs(pred - gt)
            loss = torch.where(diff < self.beta, 0.5 * diff * diff / self.beta,
                               diff - 0.5 * self.beta)

            loss = weight_reduce_loss(loss,
                                      weight=self.loss_weight,
                                      reduction=self.reduction)

        else:
            loss = torch.tensor(0).float().to(device)

        return loss
