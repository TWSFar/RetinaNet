import os
import time
from mypath import Path
from pprint import pprint
from utils.devices import select_device
user_dir = os.path.expanduser('~')


class Config:
    # data
    dataset = "visdrone"
    root_dir = user_dir + "/work/RetinaNet/data/VisDrone"
    resume = True
    resize_type = "letterbox"  # [regular, irregular, letterbox]
    min_size = 1024
    max_size = 1024
    pre = '/home/twsf/work/RetinaNet/run/visdrone_chip/20192311_115838/model_best.pth.tar'

    # model
    backbone = 'resnet50'
    if 'hrnet' in backbone:
        hrnet_cfg = user_dir + '/work/RetinaNet/lib/hrnet_config/hrnet_w48.yaml'

    # train
    batch_size = 2
    epochs = 3
    workers = 1

    # param for optimizer
    adam = True
    lr = 0.0002
    momentum = 0.9
    decay = 5*1e-4
    steps = [0.8, 0.9]
    gamma = 0.3

    # eval
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
    print_freq = 50
    plot_every = 200  # every n batch plot
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