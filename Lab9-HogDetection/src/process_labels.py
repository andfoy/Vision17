#!/usr/bin/env python

import os
import numpy as np

from spyder.utils.iofuncs import load_matlab, get_matlab_value

LABELS_ROOT = 'data/wider_face_split'

if __name__ == '__main__':
    file = os.path.join(LABELS_ROOT, 'wider_face_train.mat')
    out_file = os.path.join(LABELS_ROOT, 'face_train.npz')
    values, _ = load_matlab(file)
    if not isinstance(list, values['file_list']):
        values['file_list'] = get_matlab_value(values['file_list'][0])
    if not isinstance(list, values['face_bbx_list']):
        values['face_bbx_list'] = get_matlab_value(values['face_bbx_list'][0])
    bounding_boxes = {}
    for file_group, bbx_group in zip(values['file_list'],
                                     values['face_bbx_list']):
        bounding_boxes.update(zip(file_group, bbx_group))
    np.savez(out_file, bounding_boxes=bounding_boxes)
