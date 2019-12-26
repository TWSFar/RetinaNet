import torch
import torch.nn as nn
from models import backbones, necks, heads, losses
from models.utils import (Anchors, BBoxTransform,
                          ClipBoxes, iou_cpu)
# debug
# from models.losses.debug import FocalLoss


class RetinaNet(nn.Module):
    def __init__(self, opt, num_classes=80):
        self.opt = opt
        self.num_classes = num_classes
        super(RetinaNet, self).__init__()
        self.backbone = backbones.build_backbone(opt)
        self.neck = necks.build_neck(neck=opt.neck,
                                     in_planes=self.backbone.out_planes,
                                     out_plane=256)
        self.reg_head = heads.RegressionModel(num_features_in=256)
        self.cls_head = heads.ClassificationModel(num_features_in=256,
                                                  num_classes=self.num_classes)
        # self.loss = FocalLoss()
        self.cls_loss = losses.build_loss(opt.loss_cls)
        self.reg_loss = losses.build_loss(opt.loss_reg)

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

        batch_size = img_batch.shape[0]
        device = img_batch.device
        tensor_zero = torch.tensor(0).float().to(device)

        features = self.backbone(img_batch)
        features = self.neck(features)
        pred_cls = torch.cat([self.cls_head(feature) for feature in features], dim=1)
        pred_reg = torch.cat([self.reg_head(feature) for feature in features], dim=1)
        anchors = self.anchors(img_batch.shape[2:]).to(device)

        if self.training:
            # return self.loss(pred_cls, pred_reg, anchors[0], annotations)
            loss_cls, loss_reg = [], []

            for bi in range(batch_size):
                annotation = annotations[bi]
                annotation = annotation[annotation[:, 4] != -1]
                if annotation.shape[0] == 0:
                    loss_cls.append(tensor_zero)
                    loss_reg.append(tensor_zero)
                    continue

                target_cls, target_bbox, pst_idx = self._encode(anchors[0], annotation)
                if pst_idx.sum() == 0:
                    loss_cls.append(tensor_zero)
                    loss_reg.append(tensor_zero)
                    continue

                loss_cls_bi = self.cls_loss(pred_cls[bi], target_cls)
                loss_reg_bi = self.reg_loss(pred_reg[bi, pst_idx],
                                            target_bbox,
                                            anchors[0, pst_idx])
                loss_cls.append(loss_cls_bi.sum()/torch.clamp(pst_idx.sum().float(), min=1.0))
                loss_reg.append(loss_reg_bi.mean())

            return torch.stack(loss_cls).mean(dim=0, keepdim=True), \
                torch.stack(loss_reg).mean(dim=0, keepdim=True)

        else:
            transformed_anchors = self.regressBoxes(anchors, pred_reg)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            scores, class_id = torch.max(pred_cls.sigmoid(), dim=2, keepdim=True)

            return scores.squeeze(2), class_id.squeeze(2), transformed_anchors

    def _encode(self, anchors, annotation):
        device = anchors.device
        targets = torch.ones(anchors.shape[0], self.num_classes) * -1
        targets = targets.to(device)

        # num_anchors x num_annotations
        IoU = iou_cpu(anchors, annotation[:, :4])
        IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1

        assigned_annotations = annotation[IoU_argmax, :]

        positive_indices = torch.ge(IoU_max, 0.5)

        targets[torch.lt(IoU_max, 0.4), :] = 0
        targets[positive_indices, :] = 0
        targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

        return targets, assigned_annotations[positive_indices, :4], positive_indices


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
