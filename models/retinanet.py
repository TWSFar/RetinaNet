import math
import torch
import torch.nn as nn

from models import backbones
from models.classification import ClassificationModel
from models.regression import RegressionModel
from models.utils import losses
from models.utils.anchors import Anchors
from models.utils.functions import BBoxTransform, ClipBoxes


class RetinaNet(nn.Module):

    def __init__(self, opt, num_classes=80):
        self.opt = opt
        super(RetinaNet, self).__init__()
        self.nms_thd = self.opt.nms_thd
        self.backbone = backbones.build_backbone(opt)
        self.regressionModel = RegressionModel(num_features_in=256)
        self.classificationModel = ClassificationModel(num_features_in=256,
                                                       num_classes=num_classes)
        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.focalLoss = losses.FocalLoss(opt.giou_loss)

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

            scores, class_id = torch.max(classification, dim=2, keepdim=True)

            return scores.squeeze(2), class_id.squeeze(2), transformed_anchors


if __name__ == "__main__":
    from utils.visdrone_config import opt
    model = RetinaNet(opt)
    model = model.cuda()
    model.eval()

    for i in range(100):
        with torch.no_grad():
            input = torch.ones(1, 3, 320, 320).cuda()
            out1, out2, out2 = model(input)
    pass
