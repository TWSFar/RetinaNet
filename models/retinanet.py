import torch
import torch.nn as nn
from models import backbones, necks, heads, losses
from models.utils import Anchors, BBoxTransform, ClipBoxes


class RetinaNet(nn.Module):
    def __init__(self, opt, num_classes=80):
        self.opt = opt
        super(RetinaNet, self).__init__()
        self.backbone = backbones.build_backbone(opt)
        self.neck = necks.build_neck(neck=opt.neck,
                                     in_planes=self.backbone.out_planes,
                                     out_plane=256)
        self.reg_head = heads.RegressionModel(num_features_in=256)
        self.cls_head = heads.ClassificationModel(num_features_in=256,
                                                  num_classes=num_classes)
        self.loss = losses.build_loss(opt)

        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()

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
        features = self.neck(features)
        regression = torch.cat([self.reg_head(feature) for feature in features], dim=1)
        classification = torch.cat([self.cls_head(feature) for feature in features], dim=1)
        anchors = self.anchors(img_batch.shape[2:]).to(regression.device)

        if self.training:
            return self.loss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            scores, class_id = torch.max(classification, dim=2, keepdim=True)

            return scores.squeeze(2), class_id.squeeze(2), transformed_anchors


if __name__ == "__main__":
    from configs.visdrone_chip import opt
    model = RetinaNet(opt)
    model = model.cuda()
    model.eval()

    for i in range(100):
        with torch.no_grad():
            input = torch.ones(1, 3, 320, 320).cuda()
            out1, out2, out3 = model(input)
    pass
