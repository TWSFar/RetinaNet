import os
from pprint import pprint
from utils import select_device
user_dir = os.path.expanduser('~')


class Config:
    # data
    dataset = "visdrone_chip"
    root_dir = user_dir + "/data/Visdrone/detect_voc/"
    resize_type = "letterbox"  # [regular, irregular, letterbox]
    min_size = 1024
    max_size = 1024
    mean = [0.382, 0.383, 0.367]
    std = [0.164, 0.156, 0.164]
    resume = False
    pre = None

    # model
    backbone = 'resnet50'
    neck = "fpn"
    if 'hrnet' in backbone:
        hrnet_cfg = user_dir + '/work/RetinaNet/lib/hrnet_config/hrnet_w48.yaml'

    # train
    batch_size = 4
    epochs = 40
    workers = 1

    # param for optimizer
    adam = True
    lr = 0.0002
    momentum = 0.9
    decay = 5*1e-4
    steps = [0.8, 0.9]
    gamma = 0.3

    # eval
    eval_type = "default"
    # parameters
    pst_thd = 0.2
    nms_thd = 0.5
    n_pre_nms = 20000
    # nms: greedy_nms, soft_nms
    nms_type = 'greedy_nms'

    # loss
    loss_cls = dict(
        type='FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        reduction='sum',
        loss_weight=1.0)
    loss_reg = dict(
        type='SmoothL1Loss',
        beta=0.11,
        loss_weight=1.0)

    # visual
    visualize = True
    print_freq = 50
    plot_every = 200  # every n batch plot
    saver_freq = 1

    seed = 2

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
