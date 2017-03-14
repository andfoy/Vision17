# !/usr/bin/env python

import cv2
import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animate

plt.ion()

PLUS_SEP = '+'

COLOR_SPACES = {'rgb': lambda i: i,
                'hsv': lambda i: cv2.cvtColor(i, cv2.COLOR_RGB2HSV),
                'lab': lambda i: cv2.cvtColor(i, cv2.COLOR_RGB2LAB)}

GRAY_CONV = {'rgb': lambda i: cv2.cvtColor(i, cv2.COLOR_RGB2GRAY),
             'hsv': lambda i: np.rollaxis(i[..., :3], axis=-1)[-1],
             'lab': lambda i: np.rollaxis(i[..., :3], axis=-1)[0]}

FEATURE_FUNCS = {'k-means': utils.k_means,
                 'gmm': utils.gmm,
                 'hierarchical': utils.hierarchical,
                 'watershed': utils.watershed}


def segment_by_clustering(rgb_image, feature_space,
                          clustering_method, number_of_clusters):
    xy = False
    if PLUS_SEP in feature_space:
        xy = True
        feature_space, _ = feature_space.split(PLUS_SEP)

    # rgb_image = rgb_image / np.max(rgb_image)
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

    seg = FEATURE_FUNCS[clustering_method](*params)
    return seg
