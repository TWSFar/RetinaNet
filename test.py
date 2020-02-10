import os
import cv2
import fire
import numpy as np
from tqdm import tqdm
import os.path as osp
import matplotlib.pyplot as plt

from configs.retina_visdrone_chip import opt

from models import Model
from models.utils import PostProcess, re_resize
from utils import Saver, plot_img
from dataloaders.transform import Letterbox, Normalizer

import torch

import multiprocessing
multiprocessing.set_start_method('spawn', True)

classes = ('pedestrian', 'person', 'bicycle', 'car', 'van',
           'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
show = True


def test(**kwargs):
    opt._parse(kwargs)
    saver = Saver(opt, "test")

    imgs_name = os.listdir(opt.test_dir)
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

    results = []
    model.eval()
    with torch.no_grad():
        for ii, img_name in enumerate(tqdm(imgs_name)):
            if ii >= 3: break;
            # data read and transforms
            img_path = osp.join(opt.test_dir, img_name)
            img = cv2.imread(img_path)[:, :, ::-1]
            sample = {'img': img, 'annot': None}
            sample = normalize(sample)
            sample = resize(sample)
            input = sample['img'].unsqueeze(0).to(opt.device).permute(0, 3, 1, 2)

            # predict
            scores, labels, boxes = model(input)
            scores_bt, labels_bt, boxes_bt = post_pro(
                    scores, labels, boxes, img.shape[-2:])

            for box, label, score in zip(boxes_bt[0], labels_bt[0], scores_bt[0]):
                box[2:] = box[2:] - box[:2]
                results.append({"image_id": img_name,
                                "category_id": label.numpy(),
                                "bbox": box[:4].numpy(),
                                "score": score.numpy()})

            if show:
                # draw
                boxes = re_resize(boxes_bt[0], sample['scale'], opt.resize_type)
                labels = labels_bt[0].float().view(-1, 1)
                scores = scores_bt[0].float().view(-1, 1)
                output = torch.cat((boxes, labels, scores), dim=1)
                output = output.numpy()
                img = plot_img(img, output, classes)

                plt.figure(figsize=(10, 10))
                plt.subplot(1, 1, 1).imshow(img)
                plt.show()

        saver.save_test_result(results)


if __name__ == '__main__':
    # test()
    fire.Fire(test)
