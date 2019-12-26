import numpy as np
import torch
import torch.nn as nn
from .utils import (weight_reduce_loss, encode,
                    xyxy_2_xywh)


class BalancedL1Loss(nn.Module):
    """Balanced L1 Loss

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    """

    def __init__(self,
                 alpha=0.5,
                 gamma=1.5,
                 beta=1.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(BalancedL1Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
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
            loss: (tenosr)
        """
        assert self.beta > 0
        assert pred.shape[0] == target.shape[0]

        device = pred.device
        if target.shape[0] > 0:
            anchor = xyxy_2_xywh(anchor)
            variances = torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).to(device)
            gt = encode(target, anchor, variances)

            diff = torch.abs(pred - gt)
            b = np.e**(self.gamma / self.alpha) - 1
            loss = torch.where(
                diff < self.beta, self.alpha / b *
                (b * diff + 1) * torch.log(b * diff / self.beta + 1) - self.alpha * diff,
                self.gamma * diff + self.gamma / b - self.alpha * self.beta)

            loss = weight_reduce_loss(loss,
                                      weight=self.loss_weight,
                                      reduction=self.reduction)

        else:
            loss = torch.tensor(0).float().to(device)

        return loss
