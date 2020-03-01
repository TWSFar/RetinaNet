import os
import cv2
import pdb
import json
import shutil
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import cnms, plot_img
classes = ('pedestrian', 'person', 'bicycle', 'car', 'van',
           'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')


def parse_args():
    parser = argparse.ArgumentParser(description='VisDrone submit')
    parser.add_argument('--split', type=str, default='val', help='split')
    parser.add_argument('--result_file', type=str,
                        default="/home/twsf/work/RetinaNet/run/retina_visdrone_chip_xml/20200301_170102_test/results.json")
    parser.add_argument('--loc_dir', type=str,
                        default='/home/twsf/data/Visdrone/region_chip/Locations/')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--img_dir', type=str, help="show image path",
                        default="/home/twsf/data/Visdrone/VisDrone2019-DET-val/images/")
    args = parser.parse_args()
    print('# parametes list:')
    for (key, value) in args.__dict__.items():
        print(key, '=', value)
    print('')
    return args


def Combine():
    args = parse_args()

    loc_file = osp.join(args.loc_dir, args.split + "_chip.json")

    with open(args.result_file, 'r') as f:
        results = json.load(f)
    with open(loc_file, 'r') as f:
        chip_loc = json.load(f)

    # if osp.isfile(args.anno_file):
    #     with open(args.anno_file, 'r') as f:
    #         annos = json.load(f)

    detecions = dict()
    for det in tqdm(results):
        img_id = det['image_id']
        cls_id = det['category_id'] + 1
        bbox = det['bbox']
        score = det['score']
        loc = chip_loc[img_id]
        bbox = [bbox[0] + loc[0], bbox[1] + loc[1], bbox[2], bbox[3]]

        img_name = '_'.join(img_id.split('_')[:-1]) + osp.splitext(img_id)[1]
        if img_name in detecions:
            detecions[img_name].append(bbox + [score, cls_id])
        else:
            detecions[img_name] = [bbox + [score, cls_id]]

    output_dir = 'DET_results-%s' % args.split
    if osp.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    for img_name, det in tqdm(detecions.items()):
        det = cnms(det)
        txt_name = osp.splitext(img_name)[0] + '.txt'
        with open(osp.join(output_dir, txt_name), 'w') as f:
            for bbox in det:
                bbox = [str(x) for x in (list(bbox[0:5]) + [int(bbox[5])] + [-1, -1])]
                f.write(','.join(bbox) + '\n')

        if args.show:
            img_path = osp.join(args.img_dir, img_name)
            img = cv2.imread(img_path)[:, :, ::-1]
            bboxes = det[:, [0, 1, 2, 3, 5, 4]]
            bboxes[:, 4] -= 1
            bboxes[:, 2:4] = bboxes[:, :2] + bboxes[:, 2:4]
            img = plot_img(img, bboxes, classes)

            plt.figure(figsize=(10, 10))
            plt.subplot(1, 1, 1).imshow(img)
            plt.show()


if __name__ == '__main__':
    Combine()
