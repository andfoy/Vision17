import os
cimport cython
import numpy as np
cimport numpy as np
import matplotlib.image as mpimg


DTYPE = np.float64
INT32_DTYPE = np.int32
INT64_DTYPE = np.int64
UINT8_DTYPE = np.uint8
ctypedef np.float64_t DTYPE_t
ctypedef np.uint8_t UINT8_DTYPE_t
ctypedef np.int64_t INT64_DTYPE_t
ctypedef np.int32_t INT32_DTYPE_t
ctypedef np.int_t INT_DTYPE_t


@cython.boundscheck(False)
def get_dataset_bounding_boxes(dict bbx, str path):
    cdef list pos = []
    cdef np.ndarray[INT32_DTYPE_t, ndim=2] img_bbx
    for dirpath, dirs, files in os.walk(path):
        for file in files:
            basename, _ = osp.splitext(file)
            img_path = osp.join(dirpath, file)
            img_bbx = bbx[basename]
            img = mpimg.imread(file)
            for 