#!/usr/bin/env python

import os
import sys
import cv2
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from cyvlfeat.hog import hog, hog_render

LABELS_ROOT = 'data/wider_face_split'
LABELS_FILE = 'face_train.npz'
LABELS_VAR = 'bounding_boxes'

TRAIN_IMAGES_PATH = 'data/TrainImages'


def get_max_size_bounding_box(bbx):
    max_bbx, img_max = (0, 0), None
    for key in bbx:
        print(key)
        bbx_img = bbx[key]
        if len(bbx_img.shape) == 1:
            max_img_bbx = bbx_img[2:]
        else:
            max_img_bbx = np.amax(bbx_img[:, 2:], axis=0)
        if np.any(max_img_bbx > max_bbx):
            max_bbx, img_max = max_img_bbx, key
    return max_bbx, img_max


def get_dataset_bounding_boxes(bbx, path):
    pos = []
    for dirpath, dirs, files in os.walk(path):
        for file in files:
            basename, _ = osp.splitext(file)
            img_path = osp.join(dirpath, file)
            img_bbx = bbx[basename]


def main():
    bbx = np.load(osp.join(LABELS_ROOT, LABELS_FILE))[LABELS_VAR]
    bbx = bbx.item()
    max_dim = get_max_size_bounding_box(bbx)
    dataset_bbx = get_dataset_bounding_boxes(TRAIN_IMAGES_PATH)


if __name__ == '__main__':
    main()
