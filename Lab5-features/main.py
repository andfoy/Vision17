#! /usr/bin/env/python

import sys
import glob
import numpy as np
import progressbar
import os.path as osp
from sklearn import ensemble
from sklearn import neighbors
from textons import lib_textons
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics.pairwise import chi2_kernel

if sys.version_info.major == 3:
    import pickle
else:
    import cPickle as pickle

plt.ion()

EXT = 'jpg'
CLASS_FILE = 'data/names.txt'
TRAIN_PATH = 'data/train'
TEST_PATH = 'data/test'

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
    """
    Load image data set and represent it by using texton histograms.

    Parameters
    ----------
    fb: array_like
        Multidimensional matrix that contains the set of filters to be applied.
    textons: array_like
        Multidimensional matrix that contains the centroids assigned to each
        group of textons.
    k: int
        Number of texton categories.

    Returns
    -------
    inputs: array_like
        Matrix that contains each image represented by its texton histogram
        (Stored columnwise)
    labels: array_like
        Binary groundtruth that contains each image category.
    """
    labels = np.zeros((len(CLASSES), 30 * len(CLASSES)))
    inputs = None
    i = 0
    bar = progressbar.Bar()
    for cat in bar(CLASSES):
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


def process_test_set(fb, textons, k):
    """
    Load image test set and represent it by using texton histograms.

    Parameters
    ----------
    fb: array_like
        Multidimensional matrix that contains the set of filters to be applied.
    textons: array_like
        Multidimensional matrix that contains the centroids assigned to each
        group of textons.
    k: int
        Number of texton categories.

    Returns
    -------
    inputs: array_like
        Matrix that contains each image represented by its texton histogram
        (Stored columnwise)
    labels: array_like
        Binary groundtruth that contains each image category.
    """
    test = None
    labels = []
    for cat in CLASSES:
        label = int(cat.split('T')[1])
        regex = osp.join(TEST_PATH, cat + '*.jpg')
        imgs = glob.glob(regex)
        for img in imgs:
            print(img)
            hist = compute_texton_histogram(img, fb, textons, k)
            hist = hist.reshape(len(hist), 1)
            if test is None:
                test = hist
            else:
                test = np.hstack((test, hist))
            labels.append(label)
    return test, labels


def classify_knn(inputs, labels, N=15):
    """
    Train a K-Nearest Neighbors classifier.

    Parameters
    ----------
    inputs: array_like
        Matrix that contains each image represented by its texton histogram
        (Stored columnwise)
    labels: array_like
        Binary groundtruth that contains each image category.

    Returns
    -------
    KNN: sklearn.neighbors.KNeighborsClassifier
        Trained KNN model.
    """
    KNN = neighbors.KNeighborsClassifier(n_neighbors=5,
                                         metric=chi2_kernel,
                                         n_jobs=-1)
    KNN.fit(inputs, labels)
    return KNN


def classify_forest(inputs, labels):
    """
    Train a Random forest classifier.

    Parameters
    ----------
    inputs: array_like
        Matrix that contains each image represented by its texton histogram
        (Stored columnwise)
    labels: array_like
        Binary groundtruth that contains each image category.

    Returns
    -------
    forest: sklearn.ensemble.RandomForestClassifier
        Trained random forest classifier.
    """
    forest = ensemble.RandomForestClassifier(n_jobs=-1)
    forest.fit(inputs, labels)
    return forest


def main():
    num_orient = 20
    start_sigma = 0.1
    num_scales = 6
    scaling = np.sqrt(2)
    elong = 2
    k = 256
    n = 10
    load = True
    process_dataset = True

    if not load:
        print("Creating filter bank...\n")
        fb = lib_textons.fb_create(num_orient, start_sigma, num_scales,
                                   scaling, elong)
        print("Subsampling images...\n")
        files = subsample_images(n)
        print("Computing textons...\n")
        textons = compute_texton_set(files, fb, k, True)
        np.savez('params.npz', textons=textons, fb=fb, k=k)
    else:
        print("Loading previously saved params...")
        file_load = np.load('params.npz')
        textons = file_load['textons']
        fb = file_load['fb']
        k = file_load['k']

    if not process_dataset:
        print("Loading train set....")
        inputs, labels = load_data_set(fb, textons, k)
        test, test_labels = process_test_set(fb, textons, k)
        np.savez('dataset.npz', inputs=inputs, labels=labels,
                 test=test, test_labels=test_labels)
    else:
        file_load = np.load('dataset.npz')
        inputs = file_load['inputs']
        labels = file_load['labels']
        test = file_load['test']
        test_labels = file_load['test_labels']

    print("Training KNN Model....")
    model = classify_knn(inputs.T, labels)
    with open('KNN_model.pkl', 'wb') as fp:
        pickle.dump(model, fp)

    pred = model.predict(test.T)
    norm = np.linalg.norm
    acc = norm(pred - test_labels) / norm(pred + test_labels)
    print("KNN Accuracy: %g%" % (acc * 100))

    print("Training Random Forest...")
    forest = classify_forest(inputs.T, labels)
    with open('forest_model.pkl', 'wb') as fp:
        pickle.dump(forest, fp)

    pred2 = forest.predict(test.T)
    acc2 = norm(pred2 - test_labels) / norm(pred2 + test_labels)
    print("Random forest Accuracy: %g%" % (acc2 * 100))


if __name__ == '__main__':
    main()
