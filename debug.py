import torch.nn as nn
import torch.nn.functional as F


def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target) * focal_weight 
    return loss.mean()

def focal_loss(x, y):
    '''Focal loss.
    Args:
        x: (tensor) sized [N,D].
        y: (tensor) sized [N,].
    Return:
        (tensor) focal loss.
    '''
    alpha = 0.25
    gamma = 2
    t = y
    p = x.sigmoid()
    pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
    w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
    w = w * (1-pt).pow(gamma)
    return F.binary_cross_entropy_with_logits(x, t, w)



import torch
pred = torch.tensor([[0.5, 0.7], [0.3, 0.9]])
target = torch.tensor([[0.0, 1], [0, 1]])
res1 = py_sigmoid_focal_loss(pred, target)
res2 = focal_loss(pred, target)
pass