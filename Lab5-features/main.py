#! /usr/bin/env/python

import glob
import numpy as np
import os.path as osp
from textons import lib_textons
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.ion()

EXT = 'jpg'
CLASS_FILE = 'data/names.txt'
TRAIN_PATH = 'data/train'

with open(CLASS_FILE, 'r') as fp:
    lines = fp.readlines()

lines = [x.rstrip().split('\t') for x in lines if len(x.rstrip()) > 1]
CLASSES = dict(lines)


def subsample_images(n=4):
    """
    Subsample images from original test set.

    Parameters
    ----------
    n: int
        Number of images to subsample per class.

    Returns
    -------
    files: array_like
        List of the filenames of the images sampled
        for each category.
    """
    files = []
    for cat in CLASSES:
        regex = osp.join(TRAIN_PATH, cat + '*.jpg')
        files = glob.glob(regex)
        perm = np.random.permutation(files)
        files += perm[0:n].tolist()
    return files


def compute_texton_set(files, fb, k, debug=False):
    """
    Compute texton dictionary.

    Given a set of images, compute the texton global dictionary defined over
    the input set according to the activation response of each image after
    covolving with a set of filters. The texton grouping is subject to the
    number of texton categories desired.

    Parameters
    ----------
    files: list
        List of the filenames of the images sampled for each category.
    fb: array_like
        Multidimensional matrix that contains the set of filters to be applied.
    k: int
        Number of texton groups to be discovered by clustering.

    Returns
    -------
    textons: array_like
        Array of size (k, d) that contains the centroids assigned to each group
        of textons.
    """
    stack = []
    for file in files:
        img = mpimg.imread(file)
        if len(img.shape) == 3:
            img = lib_textons.rgb2gray(img)
        stack.append(img / 255)
    stack = np.hstack(stack)

    print("Convolving image set with filter bank...\n")
    filter_responses = lib_textons.fb_run(fb, stack)
    if debug:
        np.savez('filter_responses', resp=filter_responses)
    print('Running K-Means...\n')
    textons = lib_textons.compute_textons(filter_responses, k)
    return textons


def compute_texton_histogram(file, fb, textons, k):
    """
    Computes texton histogram of an image based on pixel wise assignment
    of texton categories.

    Parameters
    ----------
    files: str
        Image file to be represented via an histogram over
        texton features.
    fb: array_like
        Multidimensional matrix that contains the set of filters to be applied.
    textons: array_like
        Multidimensional matrix that contains the centroids assigned to each
        group of textons.
    k: int
        Number of texton categories.

    Returns
    -------
    hist: array_like
        Histogram representation of the image in the texton feature space.
    """
    img = mpimg.imread(file)
    if len(img.shape) == 3:
        img = lib_textons.rgb2gray(img)
    img = img / 255

    activation = lib_textons.fb_run(fb, img)
    texton_map = lib_textons.assign_textons(activation, textons.T)
    texton_map = (texton_map.T).ravel()
    hist = np.histogram(texton_map, np.arange(0, k + 1))[0] / len(texton_map)
    return hist


def load_data_set(fb, textons, k):
    labels = np.zeros((len(CLASSES), 30 * len(CLASSES)))
    inputs = None
    i = 0
    for cat in CLASSES:
        label = int(cat.split('T')[1])
        regex = osp.join(TRAIN_PATH, cat + '*.jpg')
        imgs = glob.glob(regex)
        for img in imgs:
            hist = compute_texton_histogram(img, fb, textons, k)
            hist = hist.reshape(len(hist), 1)
            if inputs is None:
                inputs = hist
            else:
                inputs = np.hstack((inputs, hist))
            labels[:, i] = label - 1
            i += 1
    return inputs, labels


def main():
    num_orient = 20
    start_sigma = 0.1
    num_scales = 7
    scaling = np.sqrt(2)
    elong = 2
    k = 256
    n = 10
    load = False

    print("Creating filter bank...\n")
    fb = lib_textons.fb_create(num_orient, start_sigma, num_scales,
                               scaling, elong)
    if not load:
        print("Subsampling images...\n")
        files = subsample_images(n)
        print("Computing textons...\n")
        textons = compute_texton_set(files, fb, k, True)
        np.savez('textons.npz', textons=textons)
    else:
        textons = np.load('textons.npz')['textons']

    inputs, labels = load_data_set(fb, textons, k)
    print(labels)


if __name__ == '__main__':
    main()
