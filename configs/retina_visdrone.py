import os
from pprint import pprint
from utils.devices import select_device
user_dir = os.path.expanduser('~')


class Config:
    # data
    dataset = "visdrone"
    root_dir = user_dir + "/data/Visdrone"
    test_dir = user_dir + "/data/Visdrone/VisDrone2019-DET-val/images"
    input_size = (1000, 600)
    norm_cfg = dict(mean=[0.382, 0.383, 0.367], std=[0.164, 0.156, 0.164])
    resume = False
    pre = '/home/twsf/work/RetinaNet/run/retina_visdrone/20200506_01_train/model_best.pth'

    # model
    model = "retina"
    backbone = 'res2next101_32x8d'
    neck = "fpn"
    head = dict(
        type="RetinaHead",
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.11)
    )

    # train
    use_apex = True
    batch_size = 1
    epochs = 50
    workers = 1
    freeze_bn = False

    # optimizer
    adam = False
    lr = 0.0001
    momentum = 0.9
    decay = 0.0001
    steps = [0.7, 0.9]
    gamma = 0.3
    grad_clip = 35

    # eval
    eval_type = "cocoeval"  # [cocoeval, voceval]
    nms = dict(
        type="GreedyNms",  # SoftNms
        pst_thd=0.05,
        nms_thd=0.5,
        n_pre_nms=4000
    )

    # visual
    print_freq = 50
    plot_every = 200  # every n batch plot
    saver_freq = 1

    seed = 1

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        self.device, self.gpu_id = select_device()

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items()
                if not k.startswith('_')}


opt = Config()
