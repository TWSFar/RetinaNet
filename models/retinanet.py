import math
import torch
import torch.nn as nn
import sys
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
from models.classification import ClassificationModel
from models.regression import RegressionModel
from models.backbones.resnet import resnet50, resnet101
from models.functions import losses
from models.functions.anchors import Anchors
from models.functions.utils import BBoxTransform, ClipBoxes
from models.functions.nms.nms_gpu import nms


class RetinaNet(nn.Module):

    def __init__(self, opt, num_classes=80):
        self.opt = opt
        super(RetinaNet, self).__init__()
        self.pst_thd = self.opt.pst_thd  # positive threshold
        self.nms_thd = self.opt.nms_thd
        self.backbone = resnet50()
        # self.backbone = resnet101()
        self.regressionModel = RegressionModel(num_features_in=256)
        self.classificationModel = ClassificationModel(num_features_in=256,
                                                       num_classes=num_classes)
        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.focalLoss = losses.FocalLoss()

        prior = 0.01
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0-prior)/prior))
        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()

    def freeze_bn(self):
        """Freeeze BarchNorm layers."""
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        device = img_batch.device

        features = self.backbone(img_batch)

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch).to(regression.device)

        if self.training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            scores = torch.max(classification, dim=2, keepdim=True)[0]

            scores_over_thresh = (scores > self.pst_thd)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just return
                return torch.zeros(0).to(device), torch.zeros(0).to(device), torch.zeros(0, 4).to(device)

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]
            bbox = torch.cat([transformed_anchors, scores], dim=2)[0, :, :]

            anchors_nms_idx = nms(bbox.cpu().numpy(), self.nms_thd)

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            return nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]


if __name__ == "__main__":
    from utils.config import opt
    model = RetinaNet(opt)
    model = model.cuda()
    model.eval()

    for i in range(100):
        with torch.no_grad():
            input = torch.ones(1, 3, 320, 320).cuda()
            out1, out2, out2 = model(input)
    pass
