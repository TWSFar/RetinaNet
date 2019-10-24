import os
import cv2
import numpy as np
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.transform import UnNormalizer
# from dataloaders.utils import decode_seg_map_sequence


def plot_img(img, bboxes, id2name):
    for bbox in bboxes:
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        id = int(bbox[4])
        label = id2name[id]

        if len(bbox) == 6:
            label = label + '{:.2}'.format(bbox[5])

        # plot
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.4, 1)[0]
        c1 = (x1, y1 - t_size[1]-1)
        c2 = (x1 + t_size[0], y1)
        cv2.rectangle(img, c1, c2, color=(0, 0, 1), thickness=-1)
        cv2.putText(img, label, (x1, y1-1), cv2.FONT_HERSHEY_COMPLEX, 0.4, (1, 1, 1), 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 1), thickness=2)

    return img


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, image, target, output, id2name, global_step):
        # image transform((3, x, y) -> (x, y, 3) -> numpy -> BGR)
        unnormalizer = UnNormalizer()
        image = unnormalizer(image.cpu())
        image = image.permute(1, 2, 0).numpy()

        # gt bbox transform
        target = target.cpu().numpy()
        gt_image = plot_img(image.copy(), target, id2name)
        gt_image = torch.from_numpy(gt_image).permute(2, 0, 1)

        # predict bbox transform
        output = output.cpu().numpy()
        p_image = plot_img(image.copy(), output, id2name)
        p_image = torch.from_numpy(p_image).permute(2, 0, 1)

        # target
        grid_target = make_grid(gt_image.clone().data, nrow=1, normalize=False)
        writer.add_image('Groundtruth density', grid_target, global_step)

        # output
        grid_output = make_grid(p_image.clone().data, nrow=1, normalize=False)
        writer.add_image('Predicted density', grid_output, global_step)
