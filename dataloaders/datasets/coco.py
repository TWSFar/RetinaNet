import os
import os.path as osp
import cv2
import sys
import numpy as np
from pycocotools.coco import COCO

import torch
from torch.utils.data import Dataset
from torchvision import transforms
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))
from dataloaders import transform as tsf

INSTANCES_SET = 'instances_{}.json'


class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, opt, set_name='train2017', train=True):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.opt = opt
        self.root_dir = opt.root_dir
        self.anno_dir = osp.join(self.root_dir, 'annotations')
        self.set_name = set_name
        self.train = train

        self.coco = COCO(osp.join(self.anno_dir, INSTANCES_SET.format(self.set_name)))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

        self.train_tsf = transforms.Compose([
            tsf.Normalizer(),
            tsf.Augmenter(),
            self.resize_type()
        ])

        self.test_tsf = transforms.Compose([
            tsf.Normalizer(),
            self.resize_type()
        ])

    def resize_type(self):
        if self.opt.batch_size == 1:
            return tsf.IrRegularResizer()
        elif self.opt.input_size[0] == self.opt.input_size[1]:
            return tsf.Letterbox(self.opt.input_size, train=self.train)
        else:
            return tsf.RegularResizer(self.opt.input_size)

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        self.labels = {}
        for i, c in enumerate(categories):
            self.coco_labels[i] = c['id']
            self.coco_labels_inverse[c['id']] = i
            self.classes[c['name']] = i
            self.labels[i] = c['name']

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.train:
            sample = self.train_tsf(sample)
        else:
            sample = self.test_tsf(sample)

        img = sample['img'].permute(2, 0, 1)

        return img, sample['annot'], sample['scale']

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        # read img and BGR to RGB before normalize
        img = cv2.imread(path)[:, :, ::-1]
        return img.astype(np.float32)

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 80

    @staticmethod
    def collate_fn(batch):
        images, targets, scale = list(zip(*batch))

        if len(images) > 1:
            gt = []
            for i, l in enumerate(targets):
                id = torch.tensor([[i]], dtype=torch.double).repeat(len(l), 1)
                gt.extend(torch.cat((l, id), 1).tolist())

        return torch.stack(images, 0), torch.tensor(gt), scale


def show_image(img, labels):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1).imshow(img[:, :, ::-1])
    plt.plot(labels[:, [1, 3, 3, 1, 1]].T, labels[:, [0, 0, 2, 2, 0]].T, '-')
    plt.show()
    pass


if __name__ == '__main__':
    from easydict import EasyDict
    from torch.utils.data import DataLoader
    opt = EasyDict()
    opt.root_dir = '/home/twsf/work/RetinaNet/data/COCO'
    opt.batch_size = 2
    opt.input_size = (846, 608)
    opt.min_side = 608
    opt.max_size = 1024
    dataset = CocoDataset(opt)
    sample = dataset.__getitem__(0)
    dl = DataLoader(dataset, batch_size=opt.batch_size, collate_fn=dataset.collate_fn)
    for i, (img, gt, s) in enumerate(dl):
        img = img[0].permute(1, 2, 0).numpy()
        gt = gt[:2, :4].numpy()
        show_image(img, gt)