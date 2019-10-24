import os
import os.path as osp
import numpy as np
import cv2

from dataloaders.transform import IrRegularResizer
from models.retinanet import RetinaNet, Normalizer, UnNormalizer
from utils.config import opt

import torch

import multiprocessing
multiprocessing.set_start_method('spawn', True)


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def test():
    test_path = ""
    imgs_path = osp.listdir(test_path)
    resize = IrRegularResizer()
    normalize = Normalizer()
    unnormalize = UnNormalizer()

    # Define Network
    # initilize the network here.
    model = RetinaNet(opt, num_classes=80)
    model = model.to(opt.device)

    if os.path.isfile(opt.pre):
        print("=> loading checkpoint '{}'".format(opt.pre))
        checkpoint = torch.load(opt.pre)

        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(opt.pre, checkpoint['epoch']))
    else:
        pass
        # raise FileNotFoundError

    model.eval()
    with torch.no_grad():
        for img_path in imgs_path:
            img = cv2.imread(img_path)[:, :, ::-1]
            sample = {'img': img, 'annot': None}
            sample = normalize(sample)
            sample = resize(sample)
            input = sample['img'].unsqueeze(0).to(opt.device)

            scores, classification, transformed_anchors = model(input)
            idxs = np.where(scores > 0.5)
            img = np.array(255 * unnormalize(sample['img'][0, :, :, :])).copy()
            img[img < 0] = 0
            img[img > 255] = 255
            img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                # label_name = dataset_val.labels[int(classification[idxs[0][j]])]
                # draw_caption(img, (x1, y1, x2, y2), label_name)

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

        cv2.imshow('img', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    test()
    # fire.Fire(train)
