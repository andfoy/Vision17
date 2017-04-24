#!/usr/bin/env python

import os
import sys
import cv2
import numpy as np
import progressbar
import os.path as osp
from misc import hog_features
import matplotlib.pyplot as plt
from scipy.misc import imresize
import matplotlib.image as mpimg
from cyvlfeat.hog import hog, hog_render

LABELS_ROOT = 'data/wider_face_split'
LABELS_FILE = 'face_train.npz'
LABELS_VAR = 'bounding_boxes'

TRAIN_IMAGES_PATH = 'data/TrainImages'
CROPPED_IMAGES_PATH = 'data/TrainCrops'

HOG_SIZE_CELL = 8


def get_max_size_bounding_box(bbx):
    max_bbx, img_max = np.array([0, 0]), None
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


def get_mean_size_bounding_box(bbx):
    mean_bbx, count = np.array([0, 0]), 0
    for key in bbx:
        print(key)
        bbx_img = bbx[key]
        if len(bbx_img.shape) == 1:
            bbx_img = bbx_img.reshape(1, 4)
        bbx_img_sum = np.sum(bbx_img[:, 2:], axis=0)
        mean_bbx += bbx_img_sum
        count += bbx_img.shape[0]
    mean_bbx = mean_bbx / count
    return np.ceil(mean_bbx)


def get_dataset_bounding_boxes(bbx, path, dim):
    pos = []
    count = 0
    dim_xy = dim / HOG_SIZE_CELL
    hog_dim = (int(dim_xy[0]), int(dim_xy[1]), 31)
    print(hog_dim)
    mean_template = np.zeros(hog_dim)
    for dirpath, dirs, files in os.walk(path):
        bar = progressbar.ProgressBar(redirect_stdout=True)
        for file in bar(files):
            basename, _ = osp.splitext(file)
            img_path = osp.join(dirpath, file)
            print(img_path)
            img_bbx = bbx[basename]
            if len(img_bbx.shape) == 1:
                img_bbx = img_bbx.reshape(1, len(img_bbx))
            img = mpimg.imread(img_path)
            # print(img_bbx.shape)
            for i in range(0, img_bbx.shape[0]):
                x, y, w, h = img_bbx[i, :]
                # print(img.shape, (x, y, w, h))
                img_cropped = img[y:y + h, x: x + w]
                try:
                    res = cv2.resize(img_cropped, tuple(np.int64(dim)),
                                     interpolation=cv2.INTER_CUBIC)
                except Exception:
                    continue
                res = np.transpose(res, [1, 0, 2])
                # res = imresize(img_cropped, tuple(np.int64(dim)))
                # print(res.shape)
                # hog_feat = hog(res, HOG_SIZE_CELL)
                hog_feat = hog_features(res)
                # print(hog_feat.shape)
                mean_template += hog_feat
                pos.append(hog_feat)
                count += 1
    return pos, mean_template / count


def get_cropped_image_dims(path):
    mean_shape = np.zeros((1, 2))
    max_shape = (0, 0)
    min_shape = (np.inf, np.inf)
    num_crops = 0
    bar = progressbar.ProgressBar(redirect_stdout=True)
    for dirpath, dirs, files in bar(os.walk(path)):
        for file in files:
            img_path = osp.join(dirpath, file)
            img = mpimg.imread(img_path)
            mean_shape += np.array(img.shape[0:2])
            max_shape = max(max_shape, img.shape[0:2])
            min_shape = min(min_shape, img.shape[0:2])
            num_crops += 1
    return min_shape, mean_shape / num_crops, max_shape


def get_mean_hog(path, dim):
    pos = []
    count = 0
    dim_xy = dim / HOG_SIZE_CELL
    hog_dim = (int(dim_xy[:, 0]), int(dim_xy[:, 1]), 31)
    # print(hog_dim)
    mean_template = np.zeros(hog_dim)
    bar = progressbar.ProgressBar(redirect_stdout=True)
    for dirpath, dirs, files in bar(os.walk(path)):
        for file in files:
            img_path = osp.join(dirpath, file)
            img = mpimg.imread(img_path)
            res = cv2.resize(img, tuple(np.int64(dim)), interpolation=cv2.INTER_CUBIC)
            res = np.transpose(res, [1, 0, 2])
            hog_feat = hog_features(res)
            mean_template += hog_feat
            pos.append(hog_feat)
            count += 1
    return pos, mean_template / count


def main():
    bbx = np.load(osp.join(LABELS_ROOT, LABELS_FILE))[LABELS_VAR]
    bbx = bbx.item()
    _, mean_dim, _ = get_cropped_image_dims(CROPPED_IMAGES_PATH)
    mean_dim = np.ceil(mean_dim)
    print("\nCalculating HOG over positive examples")
    print(mean_dim)
    pos, mean_hog = get_mean_hog(CROPPED_IMAGES_PATH, mean_dim)
    np.save('hog_mean.npy', mean_hog)

    # print(min_dim, mean_dim, max_dim)
    """
    mean_dim = get_mean_size_bounding_box(bbx)
    dim = np.ceil(128 * mean_dim / mean_dim[1])
    print(dim)
    print("\nCalculating HOG over positive examples")
    dataset_bbx, mean_template = get_dataset_bounding_boxes(bbx,
                                                            TRAIN_IMAGES_PATH,
                                                            dim)
    np.save('hog_mean.npy', mean_template)
    """


if __name__ == '__main__':
    main()
