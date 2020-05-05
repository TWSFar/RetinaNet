import cv2
import mmcv
import random
import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision.transforms import ColorJitter


# ------Scale change------
class Resizer(object):
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        image, scale_factor = mmcv.imrescale(image, self.input_size, return_scale=True)

        # Pad
        H, W, C = image.shape
        pad_w = 32 - W % 32 if W % 32 != 0 else 0
        pad_h = 32 - H % 32 if H % 32 != 0 else 0
        new_image = np.zeros((H + pad_h, W + pad_w, C)).astype(image.dtype)
        new_image[:H, :W, :] = image

        if annots is not None and len(annots) > 0:
            annots[:, :4] = annots[:, :4] * scale_factor

        return {'img': new_image, 'annot': annots, 'scale': scale_factor}


class Letterbox(object):
    """
    resize a rectangular image to a padded square
    """
    def __init__(self, input_size=(608, 608), train=True):
        self.input_size = input_size
        self.train = train

    def __call__(self, sample):
        assert self.input_size[0] == self.input_size[1], "input size is not square"
        image, annots = sample['img'], sample['annot']

        shape = image.shape[:2]  # shape = [height, width]
        ratio = float(self.input_size[0]) / max(shape)  # ratio  = old / new

        if self.train:
            dw = random.randint(0, max(shape) - shape[1])
            dh = random.randint(0, max(shape) - shape[0])
            left, right = dw, max(shape) - shape[1] - dw
            top, bottom = dh, max(shape) - shape[0] - dh
        else:
            dw = (max(shape) - shape[1]) / 2  # width padding
            dh = (max(shape) - shape[0]) / 2  # height padding
            left, right = round(dw - 0.1), round(dw + 0.1)
            top, bottom = round(dh - 0.1), round(dh + 0.1)

        image = cv2.copyMakeBorder(image, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT)  # padded square

        image = cv2.resize(image, (self.input_size[0], self.input_size[1]))

        H, W, C = image.shape
        pad_w = 32 - W % 32 if W % 32 != 0 else 0
        pad_h = 32 - H % 32 if H % 32 != 0 else 0
        new_image = np.zeros((H + pad_h, W + pad_w, C)).astype(image.dtype)
        new_image[:H, :W, :] = image

        if annots is not None and len(annots) > 0:
            annots[:, 0] = ratio * (annots[:, 0] + left)
            annots[:, 1] = ratio * (annots[:, 1] + top)
            annots[:, 2] = ratio * (annots[:, 2] + left)
            annots[:, 3] = ratio * (annots[:, 3] + top)

        return {'img': new_image, 'annot': annots, 'scale': (ratio, left, top)}


class RandomCrop(object):
    """Random crop the image & bboxes & masks.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, results):
        img = results['img']
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        # crop the image
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            results[key] = bboxes

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        # filter out the gt bboxes that are completely cropped
        if 'gt_bboxes' in results:
            gt_bboxes = results['gt_bboxes']
            valid_inds = (gt_bboxes[:, 2] > gt_bboxes[:, 0]) & (
                gt_bboxes[:, 3] > gt_bboxes[:, 1])
            # if no gt bbox remains after cropping, just skip this image
            if not np.any(valid_inds):
                return None
            results['gt_bboxes'] = gt_bboxes[valid_inds, :]
            if 'gt_labels' in results:
                results['gt_labels'] = results['gt_labels'][valid_inds]

            # filter and crop the masks
            if 'gt_masks' in results:
                valid_gt_masks = []
                for i in np.where(valid_inds)[0]:
                    gt_mask = results['gt_masks'][i][crop_y1:crop_y2,
                                                     crop_x1:crop_x2]
                    valid_gt_masks.append(gt_mask)

                if valid_gt_masks:
                    results['gt_masks'] = np.stack(valid_gt_masks)
                else:
                    results['gt_masks'] = np.empty(
                        (0, ) + results['img_shape'], dtype=np.uint8)

        return results


# ------Light changes------
class RandomColorJeter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.CJ = ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample):
        img = Image.fromarray(sample['img'].astype(np.uint8))
        sample['img'] = self.CJ(img)
        sample['img'] = np.array(sample['img'])

        return sample


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = Image.fromarray(sample['img'])
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            sample['img'] = np.array(sample['img'])

        return sample


# ------Augmenter------
class RandomHorizontalFlip(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):

        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            H, W, C = image.shape

            if annots is not None:
                x1 = annots[:, 0].copy()
                x2 = annots[:, 2].copy()

                annots[:, 0] = W - x2
                annots[:, 2] = W - x1

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        if sample['img'].max() > 1:
            sample['img'] = sample['img'] / 255.
        sample['img'] = (sample['img'].astype(np.float32)-self.mean) / self.std

        return sample


class UnNormalizer(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


# ------Occlusion------
# No


# ------Tools------
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sample['img'] = torch.from_numpy(sample['img'])
        if sample['annot'] is not None:
            sample['annot'] = torch.from_numpy(sample['annot'])

        return sample


def show_image(img, label):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1).imshow(img)
    plt.plot(label[:, [0, 2, 2, 0, 0]].T, label[:, [1, 1, 3, 3, 1]].T, '-')
    plt.savefig('test.png')
    plt.close()
    pass


if __name__ == "__main__":
    model = Normalizer()
    import cv2
    img = cv2.imread('/home/twsf/work/RetinaNet/test.png')
    label = np.array([[0, 0, 100, 100]])
    tsf = [Resizer((1000, 600)), RandomGaussianBlur(), Letterbox(), RandomHorizontalFlip(), Normalizer(), ToTensor()]
    sample = {"img": img, "annot": label}
    show_image(img, label)
    for t in tsf:
        sample = t(sample)
        show_image(img, label)
    pass
