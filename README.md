# RetinaNet
## Introduction
> 本模型是本人用于小目标检测的实验, 在visdrone, dota, T100数据集上做实验, 这个模型benchmark是RetinaNet. 我做小目标检测的insight主要分为两个部分, 其一是提升目标特征的数量(详见CRGNet), 另一部分是提升网络传输目标特征的性能以及专门针对小目标的结构尝试. 模型的大概结构如下:

1. 数据增强
    > 这一部分现阶段主要是光线变化、几何变化. 之后会考虑mixup, 但是不会考虑遮挡问题, 因为这对于小目标检测没有帮助.

2. backbone
    > 这一部分注重的是模型对于目标的特征的传递能力, 也就是减少特征在传递过程中的损失. 可以添加任意的backbone进行尝试, 目标认为有效的backbone为res2net, hrnet.

3. 特征融合
    > 这一步分是对于小目标检测的关键点之一, 通过深浅层的融合提升分类和定位能力, 目前的融合方式很多, 都可以做尝试.

4. Head
    > head部分的改进基本上是anchor free 和 based anchor之间的选择, 本模型复现了fcos, 但是由于目前没时间调参, 现在的精度还很低, 当然可能存在bug问题.

5. loss函数
    1. 回归:   
    giou
    ciou
    iou
    balanced l1 loss
    smooth l1 loss
    2. 分类
    focal loss

6. 后处理
    > 目标检测的后处理基本上就是NMS, 这里尝试了softNMS, 但是结果不好,
    还有一些想法, 比如YOLOV4中用的DIOUNMS, 后续会尝试


## train
    1. sh setup.sh
    2. 配置configs里面对应的超参数
    3. python train.py

## result
    ... 


## 更新

### v5.1.0 (07/05/2020)

**Highlights**
- 2020-05-07 加入双精度混合训练
- 优化val部分, 提供vocavel和cocoeval


## Contact
This repo is currently maintained by Duan Cheng zhen(twsfcz@163.com)