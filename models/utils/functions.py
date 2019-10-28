import torch
import torch.nn as nn
import numpy as np
from models.utils.nms.nms_gpu import nms


class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std = std

    def forward(self, boxes, deltas):
        device = boxes.device
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):

        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes


class PostProcess(object):
    def __init__(self, pre_pst_thd, post_pst_thd, nms_thd):
        self.pre_pst_thd = pre_pst_thd
        self.post_pst_thd = post_pst_thd
        self.nms_thd = nms_thd
        self.scr = torch.zeros(0)
        self.lab = torch.zeros(0)
        self.box = torch.zeros(0, 4)

    # """ method 1: all category use nms gather
    def __call__(self, scores, labels, boxes):
        scores_list = []
        labels_list = []
        boxes_list = []
        for index in range(len(scores)):
            scores_over_thresh = (scores[index] > self.pre_pst_thd)
            if scores_over_thresh.sum() == 0:
                scores_list.append(self.scr)
                labels_list.append(self.lab)
                boxes_list.append(self.box)
                continue

            scr = scores[index, scores_over_thresh]
            lab = labels[index, scores_over_thresh]
            box = boxes[index, scores_over_thresh]
            bboxes = torch.cat((box, scr.unsqueeze(1)), dim=1)

            nms_idx = nms(bboxes.cpu().numpy(), self.nms_thd)
            scr = scr[nms_idx]
            lab = lab[nms_idx]
            box = box[nms_idx]
            post_idx = (scr > self.post_pst_thd)

            if post_idx.sum() == 0:
                scores_list.append(self.scr)
                labels_list.append(self.lab)
                boxes_list.append(self.box)
                continue

            scores_list.append(scr[post_idx].cpu())
            labels_list.append(lab[post_idx].cpu())
            boxes_list.append(box[post_idx].cpu())

        return scores_list, labels_list, boxes_list

    """ method 1: per category use nms along
    def __call__(self, scores, labels, boxes):
        scores_list = []
        labels_list = []
        boxes_list = []
        for index in range(len(scores)):
            scores_over_thresh = (scores[index] > self.pre_pst_thd)
            if scores_over_thresh.sum() == 0:
                scores_list.append(self.scr)
                labels_list.append(self.lab)
                boxes_list.append(self.box)
                continue

            scr = scores[index, scores_over_thresh]
            lab = labels[index, scores_over_thresh]
            box = boxes[index, scores_over_thresh]
            bboxes = torch.cat((box, scr.unsqueeze(1)), dim=1)

            nms_classes = self.lab.type_as(lab)
            nms_scores = self.scr.type_as(scr)
            nms_bboxes = self.box.type_as(box)
            for c in lab.unique():
                idx = lab == c
                b = bboxes[idx]
                c = lab[idx]
                s = scr[idx]
                nms_idx = nms(b.cpu().numpy(), self.nms_thd)
                nms_scores = torch.cat((nms_scores, s[nms_idx]), dim=0)
                nms_classes = torch.cat((nms_classes, c[nms_idx]), dim=0)
                nms_bboxes = torch.cat((nms_bboxes, b[nms_idx, :4]), dim=0)

            post_idx = (nms_scores > self.post_pst_thd)

            if post_idx.sum() == 0:
                scores_list.append(self.scr)
                labels_list.append(self.lab)
                boxes_list.append(self.box)
                continue

            scores_list.append(nms_scores[post_idx].cpu())
            labels_list.append(nms_classes[post_idx].cpu())
            boxes_list.append(nms_bboxes[post_idx].cpu())

        return scores_list, labels_list, boxes_list
    """
