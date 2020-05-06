import os
import json
import time
import torch
import shutil
import logging
import numpy as np
import os.path as osp


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class Saver(object):

    def __init__(self, opt, mode='train'):
        self.opt = opt
        self.directory = osp.join('run', opt.model + '_' + opt.dataset)
        self.experiment_name = time.strftime("%Y%m%d_%H") + '_' + mode
        self.experiment_dir = osp.join(self.directory, self.experiment_name)
        self.logfile = osp.join(self.experiment_dir, 'train.log')
        if not osp.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        logging.basicConfig(
                    format='[%(asctime)s %(levelname)s] - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.DEBUG)
        f_handler = logging.FileHandler(self.logfile, mode='a')
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(f_handler)
        with open(self.logfile, 'w') as f:
            for key, val in self.opt._state_dict().items():
                f.write(key + ': ' + str(val) + '\n')

    def save_checkpoint(self, state, is_best, filename='last.pth'):
        ''' Saver checkpoint to disk '''
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(osp.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write('epoch {}: {}'.format(state['epoch'], best_pred))
            shutil.copyfile(filename, os.path.join(self.experiment_dir, 'model_best.pth'))

    def save_coco_eval_result(self, stats):
        self.logger.info(("bbox_mAP: {:.3f}, bbox_mAP_50: {:.3f}, bbox_mAP_75: {:.3f}, "
                          "bbox_mAP_s: {:.3f}, bbox_mAP_m: {:.3f}, bbox_mAP_l: {:.3f}").format(*stats))

    def save_test_result(self, results):
        with open(os.path.join(self.experiment_dir, 'results.json'), "w") as f:
            json.dump(results, f, cls=MyEncoder, indent=4)
            print("results json saved.")

    def backup_result(self):
        backup_root = osp.join(osp.expanduser('~'), "cache")
        if not osp.exists(backup_root):
            os.mkdir(backup_root)
        backup_dir = osp.join(backup_root, self.experiment_name)
        assert not osp.exists(backup_dir), "experiment has already backup"
        os.mkdir(backup_dir)
        for file in os.listdir(self.experiment_dir):
            source_file = osp.join(self.experiment_dir, file)
            if osp.isfile(source_file):
                shutil.copy(source_file, backup_dir)
