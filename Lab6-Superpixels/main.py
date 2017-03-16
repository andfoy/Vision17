#! /usr/bin/env python

import cv2
import glob
import utils
import argparse
import functools
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.ion()

PLUS_SEP = '+'

COLOR_SPACES = {'rgb': lambda i: i,
                'hsv': lambda i: cv2.cvtColor(i, cv2.COLOR_RGB2HSV),
                'lab': lambda i: cv2.cvtColor(i, cv2.COLOR_RGB2LAB)}

GRAY_CONV = {'rgb': lambda i: cv2.cvtColor(i, cv2.COLOR_RGB2GRAY),
             'hsv': lambda i: np.rollaxis(i[..., :3], axis=-1)[-1],
             'lab': lambda i: np.rollaxis(i[..., :3], axis=-1)[0]}

CLUSTERING_METHODS = {'k-means': utils.k_means,
                      'gmm': utils.gmm,
                      'hierarchical': utils.hierarchical,
                      'watershed': utils.watershed}


def find_contours(func):
    if not func:
        return functools.partial(find_contours)

    @functools.wraps(func)
    def threshold_image(*args, **kwargs):
        seg = func(*args, **kwargs)
        method = args[2]
        if method == 'watershed':
            contours = (seg == -1)
        else:
            contours = cv2.Canny(np.uint8(seg), np.min(seg), np.max(seg)) / 255
        return seg, contours
    return threshold_image


@find_contours
def segment_by_clustering(rgb_image, feature_space,
                          clustering_method, number_of_clusters):
    xy = False
    if PLUS_SEP in feature_space:
        xy = True
        feature_space, _ = feature_space.split(PLUS_SEP)

    img_conv = COLOR_SPACES[feature_space](rgb_image)

    if clustering_method != 'watershed':
        c1, c2, c3 = np.rollaxis(img_conv[..., :3], axis=-1)

        data = np.vstack([x.flatten() for x in (c1, c2, c3)])
        if xy:
            x, y = np.indices(c1.shape)
            data = np.vstack((data, x.flatten(), y.flatten()))
        data = data.astype(np.float64)
        params = (data, number_of_clusters, c1.shape)
    else:
        blur = cv2.GaussianBlur(img_conv, (3, 3), 0)
        gray = GRAY_CONV[feature_space](blur)
        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        data = np.sqrt(dx**2 + dy**2)
        params = (img_conv, data, number_of_clusters)

    seg = CLUSTERING_METHODS[clustering_method](*params)
    return seg


def abs_diff(x, y):
    return np.linalg.norm(x - y) / np.linalg.norm(x + y)


def evaluate_images(path):
    images = sorted(glob.glob(osp.join(path, '*.jpg')))
    masks = sorted(glob.glob(osp.join(path, '*.npz')))
    for im_path, mask_path in zip(images, masks):
        print("Processing: %s" % (im_path))
        img = mpimg.imread(im_path)
        h, w = img.shape[:2]
        ground_truth = np.load(mask_path)['ground_truth']
        for level in ground_truth:
            num_seg = len(np.unique(level['segmentation']))
            print("Number of segmentations: %d" % (num_seg))
            for method in CLUSTERING_METHODS:
                print("Evaluating: %s" % (method))
                spaces = list(COLOR_SPACES.keys())
                if method != 'watershed':
                    spaces += [x + '+xy' for x in COLOR_SPACES.keys()]
                elif method == 'hierarchical':
                    img = cv2.resize(img, (w // 2, h // 2),
                                     interpolation=cv2.INTER_AREA)
                for space in spaces:
                    print("Color Space: %s" % (space))
                    seg, cnt = segment_by_clustering(img, space,
                                                     method, num_seg)
                    score = 1 - abs_diff(cnt, level['boundaries'])
                    print("%s: %s, %s, %d: %g %" %
                          (im_path, method, space, level, score * 100))


parser = argparse.ArgumentParser(description='Evaluate different clustering '
                                 'methods on image segmentation tasks.')

parser.add_argument('path', metavar='path',
                    help='Path that contains input data and mask files')

if __name__ == '__main__':
    args = parser.parse_args()
    path = args.path
    evaluate_images(path)