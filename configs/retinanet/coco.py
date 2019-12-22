import time
from pprint import pprint
from utils.devices import select_device


class Config:
    # data
    dataset = "coco"
    root_dir = user_dir + "/work/RetinaNet/data/COCO"
    resume = False
    resize_type = "letterbox"  # [regular, irregular, letterbox]
    min_size = 608
    max_size = 1024
    pre = ''

    # train
    batch_size = 12
    epochs = 50
    workers = 3

    # param for optimizer
    lr = 1e-4
    momentum = 0.995
    decay = 5*1e-4
    steps = [0.8, 0.9]
    scales = 0.3

    # parameters
    pre_pst_thd = 0.05
    post_pst_thd = 0.05
    nms_thd = 0.6
    n_pre_nms = 6000

    # loss
    giou_loss = True

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