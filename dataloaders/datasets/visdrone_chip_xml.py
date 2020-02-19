import os
import os.path as osp
import cv2
import sys
import random
import numpy as np
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))
from dataloaders import transform as tsf

INSTANCES_SET = 'ImageSets/Main/{}.txt'
IMG_ROOT = 'JPEGImages'
ANNO_ROOT = 'Annotations'


class VisdroneDataset(Dataset):
    """voc dataset."""
    classes = ('pedestrian', 'person', 'bicycle', 'car', 'van',
               'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')

    def __init__(self, opt, set_name='train', train=True):
        """
        Args:
            root_dir (string): voc directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.opt = opt
        self.root_dir = opt.root_dir

        self.anno_dir = osp.join(self.root_dir, ANNO_ROOT)
        self.img_dir = osp.join(self.root_dir, IMG_ROOT)
        self.set_name = set_name
        self.train = train

        self.image_ids = self.load_image_set_index(set_name)

        self.labels = self.classes

        self.min_size = opt.min_size
        self.max_size = opt.max_size
        self.input_size = (self.min_size, self.max_size)
        self.resize = self.resizes(opt.resize_type)

        if self.train:
            self.transform = transforms.Compose([
                tsf.Normalizer(opt.mean, opt.std),
                tsf.Augmenter(),
                self.resize
            ])
        else:
            self.transform = transforms.Compose([
                tsf.Normalizer(opt.mean, opt.std),
                self.resize
            ])

    def resizes(self, resize_type):
        if resize_type == 'irregular':
            return tsf.IrRegularResizer(self.min_size, self.max_size)
        elif resize_type == 'regular':
            return tsf.RegularResizer(self.input_size)
        elif resize_type == "letterbox":
            return tsf.Letterbox(self.input_size, train=self.train)
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img, annot = self.load_image_and_annot(idx)
        sample = {'img': img, 'annot': annot}
        sample = self.transform(sample)
        sample['index'] = idx  # it is very import for val

        # show image and labels
        # show_image(sample['img'].numpy(), sample['annot'].numpy())

        return sample

    def load_image_set_index(self, imgset):
        image_ids = []
        image_set_file = osp.join(
            self.root_dir, INSTANCES_SET.format(imgset))
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            for line in f.readlines():
                image_ids.append(line.strip())
        return image_ids

    def load_image_and_annot(self, index):
        anno_file = osp.join(self.anno_dir, self.image_ids(index)+'{}.xml')
        tree = ET.parse(anno_file)
        img_name = tree.find('filename').text
        # read img and BGR to RGB before normalize
        img_path = osp.join(self.img_dir, img_name)
        img = cv2.imread(img_path)[:, :, ::-1]

        objs = tree.findall('object')
        annot = np.zeros((len(objs), 5), dtype=np.float32)
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        for i, obj in enumerate(objs):
            for j, key in enumerate(pts):
                objs[i, j] = float(obj.find(key).text) - 1
            annot[i, 4] = float(obj.find('name').text)

        return img.astype(np.float32)

    @property
    def num_classes(self):
        return 10

    @staticmethod
    def collater(data):
        imgs = [s['img'] for s in data]
        annots = [s['annot'] for s in data]
        scales = [s['scale'] for s in data]
        index = [s['index'] for s in data]

        widths = [int(s.shape[0]) for s in imgs]
        heights = [int(s.shape[1]) for s in imgs]
        batch_size = len(imgs)

        max_width = np.array(widths).max()
        max_height = np.array(heights).max()

        padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

        for i in range(batch_size):
            img = imgs[i]
            padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

        max_num_annots = max(annot.shape[0] for annot in annots)

        if max_num_annots > 0:
            annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
            for idx, annot in enumerate(annots):
                # print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
        else:
            annot_padded = torch.ones((len(annots), 1, 5)) * -1

        padded_imgs = padded_imgs.permute(0, 3, 1, 2)

        return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales, "index": index}


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]


def show_image(img, labels):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1).imshow(img[:, :, ::-1])
    plt.plot(labels[:, [0, 2, 2, 0, 0]].T, labels[:, [1, 1, 3, 3, 1]].T, '-')
    plt.show()
    pass


# if __name__ == '__main__':
#     from torch.utils.data import DataLoader
#     from configs.visdrone_chip import opt
#     dataset = VisdroneDataset(opt)
#     print(dataset.labels)
#     sample = dataset.__getitem__(0)
#     sampler = AspectRatioBasedSampler(dataset, batch_size=2, drop_last=False)
#     dl = DataLoader(dataset, batch_sampler=sampler, collate_fn=dataset.collater)
#     for i, sp in enumerate(dl):
#         pass
