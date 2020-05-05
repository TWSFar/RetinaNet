def show_image(img, label):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1).imshow(img)
    plt.plot(label[:, [0, 2, 2, 0, 0]].T, label[:, [1, 1, 3, 3, 1]].T, '-')
    plt.savefig('test.png')
    plt.close()
    pass

import mmcv
import cv2
filename = '00344343.png'
import time
temp = time.time()
for i in range(10000):
    img2 = cv2.imread(filename)
temp1 = time.time()
print(temp1 - temp)

for i in range(10000):
    img = mmcv.imread(filename)

temp = time.time()-temp1
print(temp)
pass

# import matplotlib.pyplot as plt
# mean=[0.382, 0.383, 0.367]
# std=[0.164, 0.156, 0.164]
# img2 = self.val_dataset.load_image(index[0]) / 255
# img = inputs[0].cpu().permute(1, 2, 0).numpy() * np.array(std) + np.array(mean)
# label = boxes_bt[0].numpy()
# plt.figure(figsize=(10, 10))
# plt.subplot(1, 1, 1).imshow(img)
# plt.plot(label[:, [0, 2, 2, 0, 0]].T, label[:, [1, 1, 3, 3, 1]].T, '-')
# plt.savefig('test.png')
# plt.close()
# pass
