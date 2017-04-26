#!/usr/bin/env python

import os
import sys
import cv2
import numpy as np
import progressbar
import os.path as osp
from sklearn import svm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.misc import imresize, imsave
from cyvlfeat.hog import hog, hog_render
from misc import (hog_features, collect_uniform_integers,
                  ind2sub)

NEGATIVE_PATH = 'data/negatives'
LABELS_ROOT = 'data/wider_face_split'
LABELS_FILE = 'face_train.npz'
LABELS_VAR = 'bounding_boxes'

TRAIN_IMAGES_PATH = 'data/TrainImages'
CROPPED_IMAGES_PATH = 'data/TrainCrops'

HOG_SIZE_CELL = 8
HARD_NEG_ITER = 7

MIN_SCALE = -1
MAX_SCALE = 3
NUM_OCTAVE_DIV = 3


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
    mean_template = np.zeros(hog_dim)
    bar = progressbar.ProgressBar(redirect_stdout=True)
    for dirpath, dirs, files in bar(os.walk(path)):
        for file in files:
            img_path = osp.join(dirpath, file)
            img = mpimg.imread(img_path)
            res = cv2.resize(img, tuple(np.int64(dim[0])),
                             interpolation=cv2.INTER_CUBIC)
            res = np.transpose(res, [1, 0, 2])
            hog_feat = hog_features(res)
            mean_template += hog_feat
            pos.append(hog_feat)
            count += 1
    return pos, mean_template / count


def extract_negatives(path, shape=(400, 300)):
    filename = 'negative{0}.jpg'
    neg_seq = 0
    patch_h, patch_w = shape
    for dirpath, dirs, files in os.walk(path):
        bar = progressbar.ProgressBar(redirect_stdout=True)
        for file in bar(files):
            img_path = osp.join(dirpath, file)
            print(img_path)

            img = mpimg.imread(img_path)
            max_width = img.shape[1] - patch_h
            max_height = img.shape[0] - patch_w
            for i in range(0, 5):
                y = np.random.randint(0, max_height)
                x = np.random.randint(0, max_width)
                patch = img[y:y + patch_h, x:x + patch_w]
                file_path = osp.join(NEGATIVE_PATH, filename.format(neg_seq))
                neg_seq += 1
                imsave(file_path, patch)


def collect_negatives(path, model):
    neg = []
    model_height, model_width, _ = model.shape
    for dirpath, dirs, files in os.walk(path):
        bar = progressbar.ProgressBar(redirect_stdout=True)
        for file in bar(files):
            img_path = osp.join(dirpath, file)
            print(img_path)

            img = mpimg.imread(img_path)
            hog_feat = hog_features(img)
            width = hog_feat.shape[1] - model_width + 1
            height = hog_feat.shape[0] - model_height + 1

            idx = collect_uniform_integers(0, width * height, 10)
            # print(idx.shape)
            for i in idx:
                hx, hy = ind2sub((height, width), i)
                hx = int(hx)
                hy = int(hy)
                # print(hx, hy)
                # sx = hx + np.arange(0, model_width)
                # sy = hy + np.arange(0, model_height)
                # neg.append(hog_feat[np.int64(sy), np.int64(sx), :])
                neg.append(hog_feat[hy:hy + model_height,
                                    hx:hx + model_width])
    return neg


def detect(img, model, hog_cell_size, scales):
    model_height, model_width, _ = model.shape
    hog_f = []
    detections = []
    scores = []
    for s in scales:
        img_rescaled = cv2.resize(img, None, fx=1.0 / s, fy=1.0 / s,
                                  interpolation=cv2.INTER_CUBIC)
        if min(*img_rescaled.shape[0:2]) < 128:
            break

        hog_f.append(hog_features(img_rescaled))
        score = cv2.filter2D(hog_f[-1], -1, model)
        score = np.sum(score, axis=-1)
        hy, hx = ind2sub(score.shape, np.arange(0, np.prod(score.shape)))
        x = (hx - 1) * HOG_SIZE_CELL * s
        y = (hy - 1) * HOG_SIZE_CELL * s
        detections.append(np.vstack((x - 0.5, y - 0.5,
                                     x + HOG_SIZE_CELL *
                                     model_width * s - 0.5,
                                     y + HOG_SIZE_CELL *
                                     model_height * s - 0.5)))
        scores.append(score.ravel())
    detections = np.vstack(detections).T
    scores = np.hstack(scores)
    sorted_idx = np.argsort(scores)[:1000]
    scores = scores[sorted_idx]
    detections = detections[:, sorted_idx]
    return detections, scores, hog_f


def eval_model(test_path, test_bbx, model):
    neg = []
    scales = 2**(np.linspace(MIN_SCALE,
                             MAX_SCALE,
                             NUM_OCTAVE_DIV * (MAX_SCALE - MIN_SCALE + 1)))
    for dirpath, dirs, files in os.walk(test_path):
        for file in files:
            img_path = os.join(dirpath, file)
            img = mpimg.imread(img_path)
            detections, scores, hog_f = detect(img, model, HOG_SIZE_CELL,
                                               scales)

def hard_negative_mining(pos, neg):
    for i in range(HARD_NEG_ITER):
        num_pos = len(pos)
        num_neg = len(neg)
        pos_labels = np.ones(num_pos)
        neg_labels = np.zeros(num_neg)
        C = 1.0
        # lambda_ = 0.5
        lambda_ = 1.0 / (C * (num_pos + num_neg))
        # lambda_ = 1.0 / (C * (numPos + numNeg))

        pos_shape = pos.shape
        neg_shape = neg.shape
        unrolled_pos = np.reshape(np.prod(pos_shape[0:3]),
                                  pos_shape[-1])
        unrolled_neg = np.reshape(np.prod(neg_shape[0:3]),
                                  neg_shape[-1])
        inputs = np.hstack((unrolled_pos, unrolled_neg))
        labels = np.hstack(pos_labels, neg_labels)
        model = svm.LinearSVC(C=lambda_)
        model.fit(inputs, labels)


def main():
    try:
        os.mkdir(NEGATIVE_PATH)
    except Exception:
        pass
    extract_negatives(TRAIN_IMAGES_PATH)
    # bbx = np.load(osp.join(LABELS_ROOT, LABELS_FILE))[LABELS_VAR]
    # bbx = bbx.item()
    # _, mean_dim, _ = get_cropped_image_dims(CROPPED_IMAGES_PATH)
    # mean_dim = np.ceil(mean_dim)
    # print("\nCalculating HOG over positive examples")
    # print(mean_dim)
    # pos, mean_hog = get_mean_hog(CROPPED_IMAGES_PATH, mean_dim)
    # pos = np.stack(pos, axis=-1)
    # np.save('hog_mean.npy', mean_hog)
    # print("Collecting negative examples...")
    # neg = collect_negatives(TRAIN_IMAGES_PATH, mean_hog)
    # neg = np.stack(neg, axis=-1)
    # model = hard_negative_mining(pos, neg)

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
