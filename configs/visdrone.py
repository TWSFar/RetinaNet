import os
import time
import numpy as np
from pprint import pprint
from utils.devices import select_device
user_dir = os.path.expanduser('~')


class Config:
    # data
    dataset = "visdrone"
    root_dir = user_dir + "/data/Visdrone"
    resume = False
    resize_type = "irregular"  # [regular, irregular, letterbox]
    min_size = 608
    max_size = 1024
    pre = None

    # model
    backbone = 'resnet50'
    hrnet_cfg = user_dir + '/work/RetinaNet/lib/hrnet_config/hrnet_w48.yaml'

    # train
    batch_size = 2
    epochs = 70
    workers = 1

    # param for optimizer
    adam = True
    lr = 0.0002
    momentum = 0.9
    decay = 5*1e-4
    steps = [0.8, 0.9]
    scales = 0.3

    # eval
    eval_type = "default"  # [cocoeval, default]
    # parameters
    pst_thd = 0.2
    nms_thd = 0.5
    n_pre_nms = 20000
    # nms: greedy_nms, soft_nms
    nms_type = 'greedy_nms'

    # loss
    giou_loss = False

    # visual
    visualize = True
    print_freq = 10
    plot_every = 50  # every n batch plot
    saver_freq = 1

    seed = int(time.time())

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