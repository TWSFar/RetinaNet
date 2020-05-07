import os
import fire
import time
import collections
import numpy as np

# from configs.retina_visdrone import opt
from configs.retina_visdrone import opt

from dataloaders import make_data_loader
from models import Model
from models.utils import PostProcess, VOC_eval, COCO_eval, parse_losses
from utils import TensorboardSummary, Saver, Timer

import torch
import torch.optim as optim
import multiprocessing
from apex import amp
multiprocessing.set_start_method('spawn', True)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)


class Trainer(object):
    def __init__(self, mode):
        # Define Saver
        self.saver = Saver(opt, mode)
        self.logger = self.saver.logger

        # visualize
        self.summary = TensorboardSummary(self.saver.experiment_dir, opt)
        self.writer = self.summary.writer

        # Define Dataloader
        # train dataset
        self.train_dataset, self.train_loader = make_data_loader(opt, train=True)
        self.nbatch_train = len(self.train_loader)
        self.num_classes = self.train_dataset.num_classes

        # val dataset
        self.val_dataset, self.val_loader = make_data_loader(opt, train=False)
        self.nbatch_val = len(self.val_loader)

        # Define Network
        # initilize the network here.
        self.model = Model(opt, self.num_classes)
        self.model = self.model.to(opt.device)

        # Detection post process(NMS...)
        self.post_pro = PostProcess(**opt.nms)

        # Define Optimizer
        if opt.adam:
            self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.decay)

        # Apex
        if opt.use_apex:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')

        # Resuming Checkpoint
        self.best_pred = 0.0
        self.start_epoch = 0
        if opt.resume:
            if os.path.isfile(opt.pre):
                print("=> loading checkpoint '{}'".format(opt.pre))
                checkpoint = torch.load(opt.pre)
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_pred = checkpoint['best_pred']
                self.model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(opt.pre, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(opt.pre))

        # Define lr scherduler
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, patience=3, verbose=True)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[round(opt.epochs * x) for x in opt.steps],
            gamma=opt.gamma)
        self.scheduler.last_epoch = self.start_epoch - 1

        # Using mul gpu
        if len(opt.gpu_id) > 1:
            self.logger.info("Using multiple gpu")
            self.model = torch.nn.DataParallel(self.model,
                                               device_ids=opt.gpu_id)

        # metrics
        if opt.eval_type == 'cocoeval':
            self.eval = COCO_eval(self.val_dataset.coco)
        else:
            self.eval = VOC_eval(self.num_classes)

        self.loss_hist = collections.deque(maxlen=500)
        self.timer = Timer(opt.epochs, self.nbatch_train, self.nbatch_val)
        self.step_time = collections.deque(maxlen=opt.print_freq)

    def training(self, epoch):
        self.model.train()
        epoch_loss = []
        last_time = time.time()
        for iter_num, data in enumerate(self.train_loader):
            # if iter_num >= 0: break
            try:
                self.optimizer.zero_grad()
                inputs = data['img'].to(opt.device)
                targets = data['annot'].to(opt.device)

                losses = self.model(inputs, targets)
                loss, log_vars = parse_losses(losses)

                if bool(loss == 0):
                    continue
                if opt.use_apex:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), opt.grad_clip)
                self.optimizer.step()
                self.loss_hist.append(float(loss.cpu().item()))
                epoch_loss.append(float(loss.cpu().item()))

                # visualize
                global_step = iter_num + self.nbatch_train * epoch + 1
                loss_logs = ""
                for _key, _value in log_vars.items():
                    loss_logs += "{}: {:.4f}  ".format(_key, _value)
                    self.writer.add_scalar('train/{}'.format(_key),
                                           _value,
                                           global_step)

                batch_time = time.time() - last_time
                last_time = time.time()
                eta = self.timer.eta(global_step, batch_time)
                self.step_time.append(batch_time)
                if global_step % opt.print_freq == 0:
                    printline = ("Epoch: [{}][{}/{}]  "
                                 "lr: {}  eta: {}  time: {:1.1f}  "
                                 "{}"
                                 "Running loss: {:1.5f}").format(
                                    epoch, iter_num + 1, self.nbatch_train,
                                    self.optimizer.param_groups[0]['lr'],
                                    eta, np.sum(self.step_time),
                                    loss_logs,
                                    np.mean(self.loss_hist))
                    self.logger.info(printline)

            except Exception as e:
                print(e)
                continue

        # self.scheduler.step(np.mean(epoch_loss))
        self.scheduler.step()

    def validate(self, epoch):
        self.model.eval()
        # start collecting results
        with torch.no_grad():
            # results = []
            # image_ids = []
            for ii, data in enumerate(self.val_loader):
                if ii > 0: break
                scale = data['scale']
                index = data['index']
                inputs = data['img'].to(opt.device)
                targets = data['annot']

                # run network
                scores, labels, boxes = self.model(inputs)

                scores_bt, labels_bt, boxes_bt = self.post_pro(
                    scores, labels, boxes, inputs.shape[-2:])

                outputs = []
                for k in range(len(boxes_bt)):
                    outputs.append(torch.cat((
                        boxes_bt[k].clone(),
                        labels_bt[k].clone().unsqueeze(1).float(),
                        scores_bt[k].clone().unsqueeze(1)),
                        dim=1))

                # visualize
                global_step = ii + self.nbatch_val * epoch
                if global_step % opt.plot_every == 0:
                    self.summary.visualize_image(
                        inputs, targets, outputs,
                        self.val_dataset.labels,
                        global_step)

                # eval
                if opt.eval_type == "voceval":
                    self.eval.statistics(outputs, targets, iou_thresh=0.5)

                elif opt.eval_type == "cocoeval":
                    self.eval.statistics(outputs, scale, index)

                print('{}/{}'.format(ii, len(self.val_loader)), end='\r')

            if opt.eval_type == "voceval":
                stats, ap_class = self.eval.metric()
                for key, value in stats.items():
                    self.writer.add_scalar('val/{}'.format(key), value.mean(), epoch)
                self.saver.save_voc_eval_result(stats, ap_class, self.val_dataset.labels)
                return stats['AP']

            elif opt.eval_type == "cocoeval":
                stats = self.eval.metirc()
                self.saver.save_coco_eval_result(stats)
                self.writer.add_scalar('val/mAP', stats[0], epoch)
                return stats[0]

            else:
                raise NotImplementedError


def val(**kwargs):
    opt._parse(kwargs)
    evaler = Trainer("val")
    print('Num evaluating images: {}'.format(len(evaler.val_dataset)))

    for i in range(2):
        evaler.validate(evaler.start_epoch)


def train(**kwargs):
    start_time = time.time()
    opt._parse(kwargs)
    trainer = Trainer("train")

    trainer.logger.info('Num training images: {}'.format(len(trainer.train_dataset)))

    for epoch in range(trainer.start_epoch, opt.epochs):
        # train
        trainer.training(epoch)

        # val
        val_time = time.time()
        mAP = trainer.validate(epoch)
        trainer.timer.set_val_eta(epoch, time.time() - val_time)

        is_best = mAP > trainer.best_pred
        trainer.best_pred = max(mAP, trainer.best_pred)
        if (epoch % opt.saver_freq == 0 and epoch != 0) or is_best:
            trainer.saver.save_checkpoint({
                'epoch': epoch,
                'state_dict': trainer.model.module.state_dict() if len(opt.gpu_id) > 1
                else trainer.model.state_dict(),
                'best_pred': trainer.best_pred,
                'optimizer': trainer.optimizer.state_dict(),
            }, is_best)

    all_time = trainer.timer.second2hour(time.time()-start_time)
    trainer.logger.info("experiment: " + trainer.saver.experiment_name)
    trainer.logger.info("Train done!, Sum time: {}, Best result: {}".format(all_time, trainer.best_pred))

    # cache result
    print("Backup result...")
    trainer.saver.backup_result()
    print("Done!")


if __name__ == '__main__':
    # train()
    fire.Fire(val)
