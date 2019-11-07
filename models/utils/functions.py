import torch
import torch.nn as nn
import numpy as np
from models.utils.nms.nms_gpu import nms, soft_nms


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
    def __init__(self, opt):
        self.pst_thd = opt.pst_thd
        self.n_pre_nms = opt.n_pre_nms
        self.nms_thd = opt.nms_thd
        self.nms_type = opt.nms_type
        self.scr = torch.zeros(0)
        self.lab = torch.zeros(0)
        self.box = torch.zeros(0, 4)

    """ method 1: all category use nms gather
    def __call__(self, scores, labels, boxes):
        scores_list = []
        labels_list = []
        boxes_list = []
        for index in range(len(scores)):
            scores_over_thresh = (scores[index] > self.pst_thd)
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

            if nms_idx.sum() == 0:
                scores_list.append(self.scr)
                labels_list.append(self.lab)
                boxes_list.append(self.box)
                continue

            scores_list.append(scr.cpu())
            labels_list.append(lab.cpu())
            boxes_list.append(box.cpu())

        return scores_list, labels_list, boxes_list
    """

    # """ method 2: per category use nms along
    def __call__(self, scores_bt, labels_bt, boxes_bt):
        scores_list = []
        labels_list = []
        boxes_list = []
        desort_idx = scores_bt.argsort(dim=1, descending=True)
        # olny use the first n which scores are the largest
        desort_idx = desort_idx[:, :self.n_pre_nms]
        for index in range(len(scores_bt)):
            scores = scores_bt[index, desort_idx[index]]
            labels = labels_bt[index, desort_idx[index]]
            boxes = boxes_bt[index, desort_idx[index]]
            scores_over_thresh = (scores > self.pst_thd)
            if scores_over_thresh.sum() == 0:
                scores_list.append(self.scr)
                labels_list.append(self.lab)
                boxes_list.append(self.box)
                continue

            scr = scores[scores_over_thresh]
            lab = labels[scores_over_thresh]
            box = boxes[scores_over_thresh]
            bboxes = torch.cat((box, scr.unsqueeze(1)), dim=1)

            nms_classes = self.lab.type_as(lab)
            nms_scores = self.scr.type_as(scr)
            nms_bboxes = self.box.type_as(box)
            for c in lab.unique():
                idx = lab == c
                b = bboxes[idx]
                c = lab[idx]
                s = scr[idx]
                if self.nms_type == 'soft_nms':
                    nms_idx = soft_nms(b.cpu().numpy(), method=0, threshold=self.pst_thd, Nt=self.nms_thd)
                else:
                    nms_idx = nms(b.cpu().numpy(), self.nms_thd)
                nms_scores = torch.cat((nms_scores, s[nms_idx]), dim=0)
                nms_classes = torch.cat((nms_classes, c[nms_idx]), dim=0)
                nms_bboxes = torch.cat((nms_bboxes, b[nms_idx, :4]), dim=0)

            if len(nms_bboxes) == 0:
                scores_list.append(self.scr)
                labels_list.append(self.lab)
                boxes_list.append(self.box)
                continue

            scores_list.append(nms_scores.cpu())
            labels_list.append(nms_classes.cpu())
            boxes_list.append(nms_bboxes.cpu())

        return scores_list, labels_list, boxes_list
    # """
