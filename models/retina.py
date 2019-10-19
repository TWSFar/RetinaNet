import math
import torch
import torch.nn as nn
from models.classification import ClassificationModel
from models.regression import RegressionModel
from models.backbones.resnet import resnet50, resnet101
from models.anchors import Anchors
from models.utils import BBoxTransform, ClipBoxes
from models import losses


class Retina(nn.Module):
    def __init__(self, num_classes):
        super(Retina, self).__init__()
        self.backbone = resnet50()
        # self.backbone = resnet101()
        self.regression = RegressionModel(num_features_in=256)
        self.classification = ClassificationModel(num_features_in=256,
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

            scores_over_thresh = (scores > 0.05)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just return
                return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], 0.5)

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]
