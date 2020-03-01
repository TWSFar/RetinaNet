import os
import sys
import torch
import numpy as np
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
from models.utils import DefaultEval
ANNOTATION_DIR = ""
PREDICTION_DIR = ""
IGNORE_IOU = 0.9
CLASSES = ('pedestrian', 'person', 'bicycle', 'car', 'van',
           'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')


class DET_toolkit(object):
    def __init__(self, DET_toolkit):
        self.gt_list = os.listdir(ANNOTATION_DIR)
        self.pred_list = os.listdir(PREDICTION_DIR)
        self.ignore_iou = IGNORE_IOU
        self.classes = CLASSES
        assert len(self.gt_list) == len(self.pred_list), \
            "groundTruth file's numbers not equal prediction file"

    def overlap(self, box1, box2):
        """ (box1 cup box2) / box2
        Args:
            box1: [xmin, ymin, xmax, ymax]
            box2: [xmin, ymin, xmax, ymax]
        """
        matric = np.array([box1, box2])
        u_xmin = max(matric[:,0])
        u_ymin = max(matric[:,1])
        u_xmax = min(matric[:,2])
        u_ymax = min(matric[:,3])
        u_w = u_xmax - u_xmin
        u_h = u_ymax - u_ymin
        if u_w <= 0 or u_h <= 0:
            return False
        u_area = u_w * u_h
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        if u_area / box2_area < self.ignore_iou:
            return False
        else:
            return True

    def delete_region(self, pred_bbox, gt_bbox):
        ignore_region = gt_bbox[gt_bbox[:, 4] == 0]
        idx = np.ones(len(pred_bbox)).astype(np.int32)
        for item, box1 in enumerate(pred_bbox):
            for box2 in ignore_region:
                if self.overlap(box1, box2):
                    idx[item] = 0
        pred_bbox = pred_bbox[idx]
        return pred_bbox, gt_bbox[gt_bbox[:, 4] == 1]

    def load_anno(self, anno_path):
        assert osp.isfile(anno_path), \
               "{} not exsit".format(anno_path)
        box_all = []
        with open(anno_path, 'r') as f:
            data = [x.strip().split(',')[:8] for x in f.readlines()]
            annos = np.array(data)

        bboxes = annos[:, :6].astype(np.float32)
        for bbox in bboxes:
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            box_all.append(bbox.tolist())

        return np.array(box_all).astype(np.float32)

    def __call__(self):
        def_eval = DefaultEval()
        for idx, filename in enumerate(self.pred_list):
            pred_bbox = self.load_gtanno(
                osp.join(self.pred_list, filename))
            gt_bbox = self.load_gtanno(
                osp.join(self.gt_list, filename))
            pred_bbox, gt_bbox = self.delete_ignore(pred_bbox, gt_bbox)

            def_eval.statistics(torch.tensor(pred_bbox[[0, 1, 2, 3, 5, 4]]),
                                torch.tensor(gt_bbox[[0, 1, 2, 3, 5]]))

        # Compute statistics
        stats = [np.concatenate(x, 0) for x in list(zip(*def_eval.stats))]
        # number of targets per class
        nt = np.bincount(stats[3].astype(np.int64), minlength=len(self.classes))
        if len(stats):
            p, r, ap, f1, ap_class = def_eval.ap_per_class(*stats)
            mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()

        # Print and Write results
        title = ('%20s' + '%10'*5) % ( 'Class', 'Targets', 'P', 'R', 'mAP', 'F1')
        print(title)
        printline = '%20s' + '%10.3g' * 5
        pf = printline % ('all', nt.sum(), mp, mr, map, mf1)  # print format
        print(pf)
        if len(self.classes) > 1 and len(stats):
            for i, c in enumerate(ap_class):
                pf = printline % (self.classes[c], nt[c], p[i], r[i], ap[i], f1[i])
                print(pf)


if __name__ == '__main__':
    det = DET_toolkit()
    det()
