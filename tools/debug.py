import mmcv
import torch
import numpy as np
input_size = (1024, 512)
image = torch.rand(1000, 400, 3).numpy().astype(np.uint8)
img, scale = mmcv.imrescale(image, input_size, return_scale=True)
pass