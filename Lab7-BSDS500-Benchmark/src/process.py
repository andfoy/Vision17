# !/usr/bin/env python

import os
import progressbar
import numpy as np
import os.path as osp
import scipy.io as sio

GROUND_TRUTH = 'groundTruth'
FOLDER = 'BSR/BSDS500/data/groundTruth'


def main():
    for path, dirs, files in os.walk(FOLDER):
        bar = progressbar.ProgressBar(redirect_stdout=True)
        for f in bar(files):
            file = os.path.join(path, f)
            print(file)
            data = []
            segmentations = sio.loadmat(file)[GROUND_TRUTH][0]
            for seg in segmentations:
                info = {}
                seg = seg[0][0]
                info['segmentation'], info['boundaries'] = seg
                data.append(info)
            filename, _ = osp.splitext(osp.basename(file))
            np.savez(osp.join(osp.dirname(file), filename), ground_truth=data)
            os.remove(file)


if __name__ == '__main__':
    main()
