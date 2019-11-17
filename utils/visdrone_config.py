import os
import time
import torch
import numpy as np
from mypath import Path
from pprint import pprint
user_dir = os.path.expanduser('~')


def select_device(force_cpu=False):
    cuda = False if force_cpu else torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    ng = 0
    if not cuda:
        print('Using CPU\n')
    if cuda:
        c = 1024 ** 2
        ng = torch.cuda.device_count()
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        for i in range(ng):
            print('Using CUDA device{} _CudaDeviceProperties(name={}, total_memory={}MB'.\
                  format(i, x[i].name, round(x[i].total_memory/c)))
        print('')
    return device, np.arange(0, ng).tolist()


class Config:
    # data
    dataset = "visdrone"
    root_dir = Path.db_root_dir(dataset)
    resume = False
    min_size = 608
    max_size = 1024
    pre = user_dir + '/work/RetinaNet/run/visdrone/experiment_4/checkpoint.path.tar'

    # model
    backbone = 'resnet50'
    hrnet_cfg = user_dir + '/work/RetinaNet/lib/hrnet_config/hrnet_w48.yaml'

    # train
    batch_size = 2
    start_epoch = 0
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
    # parameters
    pst_thd = 0.05
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

    seed = time.time()

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
