import os
import os.path as osp
import numpy as np
import cv2
import matplotlib.pyplot as plt

from dataloaders.transform import Letterbox, Normalizer
from models import Model
from configs.retina_visdrone_chip import opt

import torch

import multiprocessing
multiprocessing.set_start_method('spawn', True)

classes = {}


def draw_bboxes(img, bboxes):
    for bbox in bboxes:
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        id = int(bbox[4])
        cls = classes[id]

        if len(bbox) == 6:
            cls = cls + '{:.2}'.format(bbox[5])

        # plot
        t_size = cv2.getTextSize(cls, cv2.FONT_HERSHEY_COMPLEX, 0.4, 1)[0]
        c1 = (x1, y1 - t_size[1]-1)
        c2 = (x1 + t_size[0], y1)
        cv2.rectangle(img, c1, c2, color=(0, 0, 255), thickness=-1)
        cv2.putText(img, cls, (x1, y1-1), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

    return img


def test(**kwargs):
    opt._parse(kwargs)
    test_path = "/home/twsf/work/RetinaNet/data/demo"
    imgs_path = os.listdir(test_path)
    resize = Letterbox(input_size=(opt.min_size, opt.max_size))
    normalize = Normalizer(mean=opt.mean, std=opt.std)

    # Define Network
    # initilize the network here.
    model = Model(opt, num_classes=10)
    model = model.to(opt.device)
    post_pro = PostProcess(**opt.nms)

    if os.path.isfile(opt.pre):
        print("=> loading checkpoint '{}'".format(opt.pre))
        checkpoint = torch.load(opt.pre)

        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(opt.pre, checkpoint['epoch']))
    else:
        raise FileNotFoundError

    model.eval()
    with torch.no_grad():
        for img_path in imgs_path:
            # data read and transforms
            img_path = osp.join(test_path, img_path)
            img = cv2.imread(img_path)[:, :, ::-1]
            sample = {'img': img, 'annot': None}
            sample = normalize(sample)
            sample = resize(sample)
            input = sample['img'].unsqueeze(0).to(opt.device).permute(0, 3, 1, 2)

            # predict
            scores, labels, boxes = model(imgs)
            scores_bt, labels_bt, boxes_bt = post_pro(
                    scores, labels, boxes, imgs.shape[-2:])

            # draw
            boxes = boxes / sample['scale']
            output = torch.cat((boxes, labels.float().unsqueeze(1), scores.unsqueeze(1)), dim=1)
            output = output.cpu().numpy()
            img = draw_bboxes(img, output)

            # show
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 1, 1).imshow(img)
            plt.show()


if __name__ == '__main__':
    # test()
    fire.Fire(test)
