import os
import fire
import json
import time
import collections
import numpy as np

# from models_demo import model_demo
from configs.visdrone_config import opt
from models.retinanet import RetinaNet
from dataloaders import make_data_loader
from models.utils.functions import PostProcess
from utils.visualization import TensorboardSummary
from utils.saver import Saver
from utils.timer import Timer

import torch
import torch.optim as optim

from pycocotools.cocoeval import COCOeval

import multiprocessing
multiprocessing.set_start_method('spawn', True)


class Trainer(object):
    def __init__(self):
        # Define Saver
        self.saver = Saver(opt)

        # visualize
        if opt.visualize:
            self.summary = TensorboardSummary(self.saver.experiment_dir)
            self.writer = self.summary.create_summary()

        # Define Dataloader
        # train dataset
        self.train_dataset, self.train_loader = make_data_loader(opt, train=True)
        self.num_bt_tr = len(self.train_loader)
        self.num_classes = self.train_dataset.num_classes

        # val dataset
        self.val_dataset, self.val_loader = make_data_loader(opt, train=False)
        self.num_bt_val = len(self.val_loader)

        # Define Network
        # initilize the network here.
        self.model = RetinaNet(opt, self.num_classes)
        # self.model = model_demo.resnet50(num_classes=10, pretrained=False)
        self.model = self.model.to(opt.device)

        # contain nms for val
        self.post_pro = PostProcess(opt)

        # Define Optimizer
        if opt.adam:
            self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.decay)

        # Define lr scherduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=3, verbose=True)

        # Resuming Checkpoint
        self.best_pred = 0.0
        self.start_epoch = opt.start_epoch
        if opt.resume:
            if os.path.isfile(opt.pre):
                print("=> loading checkpoint '{}'".format(opt.pre))
                checkpoint = torch.load(opt.pre)
                opt.start_epoch = checkpoint['epoch']
                self.best_pred = checkpoint['best_pred']
                self.model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(opt.pre, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(opt.pre))

        # Using mul gpu
        if len(opt.gpu_id) > 1:
            print("Using multiple gpu")
            self.model = torch.nn.DataParallel(self.model,
                                               device_ids=opt.gpu_id)

        self.loss_hist = collections.deque(maxlen=500)
        self.timer = Timer(opt.epochs, self.num_bt_tr, self.num_bt_val)
        self.step_time = collections.deque(maxlen=opt.print_freq)

    def training(self, epoch):
        self.model.train()
        if len(opt.gpu_id) > 1:
            self.model.module.freeze_bn()
        else:
            self.model.freeze_bn()
        epoch_loss = []
        for iter_num, data in enumerate(self.train_loader):
            try:
                temp_time = time.time()
                self.optimizer.zero_grad()
                imgs = data['img'].to(opt.device)
                target = data['annot'].to(opt.device)

                cls_loss, loc_loss = self.model([imgs, target])

                cls_loss = cls_loss.mean()
                loc_loss = loc_loss.mean()
                loss = cls_loss + loc_loss

                if bool(loss == 0):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()
                self.loss_hist.append(float(loss))
                epoch_loss.append(float(loss))

                # visualize
                global_step = iter_num + self.num_bt_tr * epoch + 1
                self.writer.add_scalar('train/cls_loss_epoch', cls_loss.cpu().item(), global_step)
                self.writer.add_scalar('train/loc_loss_epoch', loc_loss.cpu().item(), global_step)

                batch_time = time.time() - temp_time
                eta = self.timer.eta(global_step, batch_time)
                self.step_time.append(batch_time)
                if global_step % opt.print_freq == 0:
                    printline = ("Epoch: [{}][{}/{}]  "
                                 "lr: {}, eta: {}, time: {:1.3f}, "
                                 "loss_cls: {:1.5f}, "
                                 "loss_bbox: {:1.5f}, "
                                 "Running loss: {:1.5f}").format(
                                    epoch, iter_num + 1, self.num_bt_tr,
                                    self.optimizer.param_groups[0]['lr'],
                                    eta, np.sum(self.step_time),
                                    float(cls_loss), float(loc_loss),
                                    np.mean(self.loss_hist))
                    print(printline)
                    self.saver.save_experiment_log(printline)

                del cls_loss
                del loc_loss

            except Exception as e:
                print(e)
                continue

        self.scheduler.step(np.mean(epoch_loss))

    def validate(self, epoch):
        self.model.eval()
        # start collecting results
        with torch.no_grad():
            results = []
            image_ids = []
            for ii, data in enumerate(self.val_loader):
                scale = data['scale']
                index = data['index']
                img = data['img'].to(opt.device).float()
                target = data['annot']

                # run network
                scores, labels, boxes = self.model(img)

                scores_bt, labels_bt, boxes_bt = self.post_pro(scores, labels, boxes)

                # visualize
                global_step = ii + self.num_bt_val * epoch
                if ii % opt.plot_every == 0:
                    output = []
                    for k in range(len(boxes_bt)):
                        output.append(torch.cat((
                            boxes_bt[k],
                            labels_bt[k].unsqueeze(1).float(),
                            scores_bt[k].unsqueeze(1)),
                            dim=1))
                    self.summary.visualize_image(
                        self.writer,
                        img, target, output,
                        self.val_dataset.labels,
                        global_step)

                # save json
                for jj in range(len(boxes_bt)):
                    boxes = boxes_bt[jj]
                    scores = scores_bt[jj]
                    labels = labels_bt[jj]
                    # correct boxes for image scale
                    boxes = boxes / scale[jj]

                    if boxes.shape[0] > 0:
                        # change to (x, y, w, h) (MS COCO standard)
                        boxes[:, 2] -= boxes[:, 0]
                        boxes[:, 3] -= boxes[:, 1]

                        # compute predicted labels and scores
                        # for box, score, label in zip(boxes[0], scores[0], labels[0]):
                        for box_id in range(boxes.shape[0]):
                            score = float(scores[box_id])
                            label = int(labels[box_id])
                            box = boxes[box_id, :]

                            # append detection for each positively labeled class
                            image_result = {
                                'image_id': self.val_dataset.image_ids[index[jj]],
                                'category_id': self.val_dataset.label_to_coco_label(label),
                                'score': float(score),
                                'bbox': box.tolist(),
                            }

                            # append detection to results
                            results.append(image_result)

                # append image to list of processed images
                for idx in index:
                    image_ids.append(self.val_dataset.image_ids[idx])

                # print progress
                print('{}/{}'.format(ii, len(self.val_loader)), end='\r')

            if not len(results):
                return 0

            # write output
            json.dump(results, open('run/{}/{}_bbox_results.json'.format(
                opt.dataset, self.val_dataset.set_name), 'w'), indent=4)

            # load results in COCO evaluation tool
            coco_true = self.val_dataset.coco
            coco_pred = coco_true.loadRes('run/{}/{}_bbox_results.json'.format(
                opt.dataset, self.val_dataset.set_name))

            # run COCO evaluation
            coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
            coco_eval.params.imgIds = image_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            # save result
            stats = coco_eval.stats
            self.saver.save_coco_eval_result(stats=stats, epoch=epoch)

            # visualize
            self.writer.add_scalar('val/AP50', stats[1], epoch)

            # according AP50
            return stats[1]


def eval(**kwargs):
    opt._parse(kwargs)
    trainer = Trainer()
    print('Num evaluating images: {}'.format(len(trainer.val_dataset)))

    trainer.validate(trainer.start_epoch)


def train(**kwargs):
    opt._parse(kwargs)
    trainer = Trainer()

    print('Num training images: {}'.format(len(trainer.train_dataset)))

    for epoch in range(opt.start_epoch, opt.epochs):
        # train
        trainer.training(epoch)

        # val
        val_time = time.time()
        ap50 = trainer.validate(epoch)
        trainer.timer.set_val_eta(epoch, time.time() - val_time)

        is_best = ap50 > trainer.best_pred
        trainer.best_pred = max(ap50, trainer.best_pred)
        if (epoch % opt.saver_freq == 0) or is_best:
            trainer.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': trainer.model.module.state_dict() if len(opt.gpu_id) > 1
                else trainer.model.state_dict(),
                'best_pred': trainer.best_pred,
                'optimizer': trainer.optimizer.state_dict(),
            }, is_best)


if __name__ == '__main__':
    # train()
    fire.Fire()
