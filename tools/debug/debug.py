class TT:
    loss_cls = dict(
        type='FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=1.0)

    loss_bbox = dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)


tt = TT()

def fun(use_sigmoid=True,
        type='',
        gamma=2.0,
        alpha=0.25,
        loss_weight=1.0):
    pass

fun(**tt.loss_cls)
pass