from models.utils.external.nms_gpu import nms
from models.utils.external.iou_gpu import iou
from models.utils import calc_iou
import numpy as np
import torch

m = np.array([[681.2814, 239.3609, 776.8018, 419.2532, 0.9],
              [680.2755, 281.5127, 775.8295, 368.8236, 0.9],
              [650.7606, 224.9314, 742.4230, 398.7491, 0.9],
              [647.6743, 265.5477, 744.6954, 353.3916, 0.9],
              [706.0413, 294.2812, 808.4049, 386.3803, 0.9]], dtype=np.float32)

n = np.array([[681.2814, 239.3609, 776.8018, 419.2532, 0.9],
              [680.2755, 281.5127, 775.8295, 368.8236, 0.9],
              [650.7606, 22.9314, 742.4230, 39.7491, 0.9],
              [64.6743, 265.5477, 744.6954, 353.3916, 0.9]], dtype=np.float32)

area = (m[:, 2] - m[:, 0]) * (m[:, 3] - m[:, 1])

nms_res = nms(m, thresh=0.4)

iou_gpu = iou(m, n)
iou_cpu = calc_iou(torch.tensor(m), torch.tensor(n))
pass

# test time
m = np.random.rand(2000, 5).astype(np.float32)
n = np.random.rand(3000, 5).astype(np.float32)

import time
temp1 = time.time()
iou_gpu = iou(m, n)
temp2 = time.time()
iou_cpu = calc_iou(torch.tensor(m).cuda(), torch.tensor(n).cuda())
temp3 = time.time()

print(temp2 - temp1)
print(temp3 - temp2)
