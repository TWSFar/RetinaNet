import torch


def calc_giou(box1, box2):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box1 = box1.t()
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    # x, y, w, h = box1
    b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
    b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
    b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

    iou = inter_area / union_area  # iou
    c_x1, c_x2 = torch.min(b1_x1, b2_x1), torch.max(b1_x2, b2_x2)
    c_y1, c_y2 = torch.min(b1_y1, b2_y1), torch.max(b1_y2, b2_y2)
    c_area = (c_x2 - c_x1) * (c_y2 - c_y1)  # convex area

    giou = iou - (c_area - union_area) / c_area  # GIoU

    return giou


    if self.giou_loss:
        predicts = predicts * torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).to(device)
        pred_ctr_x = predicts[:, 0] * anchor_widths_pi + anchor_ctr_x_pi
        pred_ctr_y = predicts[:, 1] * anchor_heights_pi + anchor_ctr_y_pi
        pred_widths = torch.exp(predicts[:, 2]) * anchor_widths_pi
        pred_heights = torch.exp(predicts[:, 3]) * anchor_heights_pi
        pred_bbox = torch.stack((pred_ctr_x, pred_ctr_y, pred_widths, pred_heights)).t()
        gt_bbox = torch.stack((gt_ctr_x, gt_ctr_y, gt_widths, gt_heights)).t()
        giou = calc_giou(pred_bbox, gt_bbox)
        regression_loss = 1.0 - giou