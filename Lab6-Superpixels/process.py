# !/usr/bin/env python

import os
import glob
import progressbar
import numpy as np
import os.path as osp
import scipy.io as sio

GROUND_TRUTH = 'groundTruth'
FOLDER = 'data/Lab5Images'


def main():
    bar = progressbar.ProgressBar()
    files = glob.glob(FOLDER + '/*.mat')
    for file in bar(files):
        data = []
        segmentations = sio.loadmat(file)[GROUND_TRUTH][0]
        for seg in segmentations:
            info = {}
            seg = seg[0][0]
            info['segmentation'], info['boundaries'] = seg
            data.append(info)
        filename, _ = osp.splitext(osp.basename(file))
        np.savez(FOLDER + '/' + filename, ground_truth=data)
        os.remove(file)


if __name__ == '__main__':
    main()
