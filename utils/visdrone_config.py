import time
import torch
import numpy as np
import os.path as osp
from mypath import Path
from pprint import pprint


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
    pre = '/home/twsf/work/RetinaNet/run/visdrone/experiment_0/checkpoint.path.tar'

    # train
    batch_size = 3
    start_epoch = 0
    epochs = 70
    workers = 1

    # param for optimizer
    lr = 0.0002
    momentum = 0.9
    decay = 5*1e-4
    steps = [0.8, 0.9]
    scales = 0.3

    # parameters
    pre_pst_thd = 0.001
    post_pst_thd = 0.001
    nms_thd = 0.5
    n_pre_nms = 6000

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
