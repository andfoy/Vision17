
import os
import sys
import cv2
import glob
import shutil
cimport cython
cimport numpy as np
import numpy as np
import os.path as osp
import matplotlib.image as mpimg
from skimage.measure import regionprops
from cython.parallel cimport prange


def all_bench_fast(str img_dir, str gt_dir, str in_dir, str out_dir,
                   int n_thresh=99, int radius=3, bint thinpb=True):
    iids = glob.glob(osp.join(img_dir, '*.jpg'))

    for i in prange(4, num_threads=4, schedule='guided', nogil=True):
        file_name, _ = osp.splitext(osp.basename(iids[i]))
        ev_file_4 = osp.join(out_dir, file_name) + '_ev4.txt'
        in_file = osp.join(in_dir, file_name) + '.npz'
        gt_file = osp.join(gt_dir, file_name) + '.npz'
        ev_file_1 = osp.join(out_dir, file_name) + '_ev1.txt'
        ev_file_2 = osp.join(out_dir, file_name) + '_ev2.txt'
        ev_file_3 = osp.join(out_dir, file_name) + '_ev3.txt'

        evaluation_bdry_image_fast(in_file,gt_file, ev_file1,
                                   n_thresh, radius, thinpb)

        evaluation_reg_image(in_file, gt_file, ev_file_1,
                             ev_file_2, ev_file_3, ev_file_4,
                             n_thresh)
        print(i)

    collect_eval_bdry(out_dir)
    collect_eval_reg(out_dir)
    shutil.rmtree(out_dir)


def evaluation_bdry_image_fast(str in_file, str gt_file, str pr_file,
                               int n_thresh, int radius, bint thinpb):
    _, ext = osp.splitext(osp.basename(in_file))
    thresh_init = False
    if ext == 'npz':
        fh = np.load(in_file)
        if 'ucm2' in fh:
            ucm2 = fh['ucm2']
            idx_y = np.arange(3, ucm2.shape[0], 2)
            idx_x = np.arange(3, ucm2.shape[1], 2)
            pb = np.double(ucm2[idx_y, idx_x])
            del ucm2
        elif 'segs' in fh:
            segs = fh['segs']
            if n_thresh != np.prod(segs.shape):
                n_thresh = np.prod(segs.shape)
            thresh = np.arange(0, n_thresh)
            bmap_func = lambda i: seg2bdry(segs[i], 'image_size')
            thresh_init = True
        else:
            pb = mpimg.imread(in_file)/255.0
        fh.close()

    if not thresh_init:
        thresh = np.linspace(1/(n_thresh+1), 1-1/(n_thresh+1), n_thresh)
        bmap_func = lambda i: pb >= thresh[i]

    f = np.load(gt_file)
    if 'ground_truth' not in f:
        print("Bad ground_truth file")
        sys.exit(-1)

    ground_truth = f['ground_truth']
    f.close()

    human = np.zeros(ground_truth[0]['boundaries'])
    for i in range(0, len(ground_truth)):
        human += ground_truth[i]['boundaries']

    cnt_r = np.zeros(thresh.shape)
    sum_r = np.zeros(thresh.shape)
    cnt_p = np.zeros(thresh.shape)
    sum_p = np.zeros(thresh.shape)

    bmap_old = None
    same_bmp = False

    for t in range(0, n_thresh):
        bmap = bmap_func(t)
        if np.sum(bmap == bmap_old):
            same_bmp = True
        else:
            same_bmp = False
            bmap_old = bmap

        if not same_bmp:
            if thinpb:
                bmap = np.double(bwmorph_thin(bmap))

            match1, match2 = correspond_curves(bmap, human, radius)
            cnt_r[t] = np.sum(match2)
            sum_r[t] = np.sum(human)

            cnt_p[t] = np.sum(match1)
            sum_p[t] = np.sum(bmap)
        else:
            cnt_r[t] = cnt_r[t-1]
            sum_r[t] = sum_r[t-1]

            cnt_p[t] = cnt_p[t-1]
            sum_p[t] = sum_p[t-1]

    with open(pr_file, 'w') as fp:
        for i in range(0, len(thresh)):
            fp.write('%10g %10g %10g %10g %10g\n', (thresh[i], cnt_r[i],
                                                    sum_r[i],
                                                    cnt_p[i], sum_p[i]))

    return thresh, cnt_r, sum_r, cnt_p, sum_p


def correspond_curves(bmap1, bmap2, radius):
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2, radius*2))
    BW1 = bmap1.astype(np.uint8)
    BW2 = bmap2 > 0

    BW1d = cv2.dilate(BW1, kernel, iterations=1)
    BW2d = cv2.dilate(BW2, kernel, iterations=1)

    match1 = np.logical_and(BW1, BW2d).astype(np.double64)
    match2 = bmap2 * np.logical_and(BW1d, BW2)
    return match1, match2


def seg2bdry(seg, fmt='double_size'):
    if fmt != 'image_size' and fmt != 'double_size':
        print("Possible values for fmt are: image_size and double_size")
        sys.exit(-1)

    tx, ty, nch = seg.shape

    if nch != 1:
        print('seg must be a scalar image')

    bdry = np.zeros((2*tx + 1, 2*ty + 1))
    edgels_v = seg[:-1, :] != seg[1:, :]
    edgels_v[-1:, :] = 0
    edgels_h = seg[:, :-1] != seg[:, 1:]
    edgels_h[:, -1:] = 0

    bdry[2:2:, 1:2:] = edgels_v
    bdry[1:2:, 2:2:] = edgels_h
    bdry[2:2:-1, 2:2,-1] = np.max(np.max(edgels_h[0:-1, 0:-1], edgels_h[1:, 0:-1]),
                                  np.max(edgels_v[0:-1, 0:-1], edgels_v[0:-1, 1:]))

    bdry[0, :] = bdry[1, :]
    bdry[:, 0] = bdry[:, 1]
    bdry[-1, :] = bdry[-2, :]
    bdry[:, -1] = bdry[:, -2]

    if fmt == 'image_size':
        bdry = bdry[2:2:, 2:2:]

    return bdry


def evaluation_reg_image(in_file, gt_file, ev_file_2, ev_file_3,
                         ev_file_4, n_thresh=99):
    f = np.load(in_file)
    if 'ucm2' in f:
        ucm = f['ucm2']
        thresh = np.linspace(1.0/(n_thresh + 1), 1-1.0/(n_thresh+1), n_thresh)
    elif 'segs' in f:
        segs = f['segs']
        if n_thresh != np.prod(segs.shape):
            n_thresh = np.prod(segs.shape)
        thresh = np.arange(0, n_thresh)
    f.close()

    f = np.load(gt_file)
    nsegs = len(f['ground_truth'])
    f.close()
    if nsegs == 0:
        print("Bad ground_truth file!")
        sys.exit(-1)

    regions_GT = []
    total_gt = 0
    for s in range(0, nsegs):
        ground_truth[s]['segmentation'] = ground_truth[s]['segmentation'].astype(np.double64)
        props = regionprops(ground_truth[s]['segmentation'])
        regions_tmp = props.area
        regions_GT.append(regions_tmp)
        total_gt += np.max(ground_truth[s]['segmentation'])

    cnt_R = np.zeros(thresh.shape);
    sum_R = np.zeros(thresh.shape);
    cnt_P = np.zeros(thresh.shape);
    sum_P = np.zeros(thresh.shape);
    sum_RI = np.zeros(thresh.shape);
    sum_VOI = np.zeros(thresh.shape);

    


def collect_eval_bdry(pb_dir):
    fname = osp.join(pb_dir, 'eval_bdry.txt')


G123_LUT = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
       0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
       1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1,
       0, 0, 0], dtype=np.bool)

G123P_LUT = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,
       0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0], dtype=np.bool)

def bwmorph_thin(image, n_iter=None):
    """
    Perform morphological thinning of a binary image

    Parameters
    ----------
    image : binary (M, N) ndarray
        The image to be thinned.

    n_iter : int, number of iterations, optional
        Regardless of the value of this parameter, the thinned image
        is returned immediately if an iteration produces no change.
        If this parameter is specified it thus sets an upper bound on
        the number of iterations performed.

    Returns
    -------
    out : ndarray of bools
        Thinned image.

    See also
    --------
    skeletonize

    Notes
    -----
    This algorithm [1]_ works by making multiple passes over the image,
    removing pixels matching a set of criteria designed to thin
    connected regions while preserving eight-connected components and
    2 x 2 squares [2]_. In each of the two sub-iterations the algorithm
    correlates the intermediate skeleton image with a neighborhood mask,
    then looks up each neighborhood in a lookup table indicating whether
    the central pixel should be deleted in that sub-iteration.

    References
    ----------
    .. [1] Z. Guo and R. W. Hall, "Parallel thinning with
           two-subiteration algorithms," Comm. ACM, vol. 32, no. 3,
           pp. 359-373, 1989.
    .. [2] Lam, L., Seong-Whan Lee, and Ching Y. Suen, "Thinning
           Methodologies-A Comprehensive Survey," IEEE Transactions on
           Pattern Analysis and Machine Intelligence, Vol 14, No. 9,
           September 1992, p. 879

    Examples
    --------
    >>> square = np.zeros((7, 7), dtype=np.uint8)
    >>> square[1:-1, 2:-2] = 1
    >>> square[0,1] =  1
    >>> square
    array([[0, 1, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    >>> skel = bwmorph_thin(square)
    >>> skel.astype(np.uint8)
    array([[0, 1, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    """
    # check parameters
    if n_iter is None:
        n = -1
    elif n_iter <= 0:
        raise ValueError('n_iter must be > 0')
    else:
        n = n_iter

    # check that we have a 2d binary image, and convert it
    # to uint8
    skel = np.array(image).astype(np.uint8)

    if skel.ndim != 2:
        raise ValueError('2D array required')
    if not np.all(np.in1d(image.flat,(0,1))):
        raise ValueError('Image contains values other than 0 and 1')

    # neighborhood mask
    mask = np.array([[ 8,  4,  2],
                     [16,  0,  1],
                     [32, 64,128]],dtype=np.uint8)

    # iterate either 1) indefinitely or 2) up to iteration limit
    while n != 0:
        before = np.sum(skel) # count points before thinning
        # for each subiteration
        for lut in [G123_LUT, G123P_LUT]:
            # correlate image with neighborhood mask
            N = ndi.correlate(skel, mask, mode='constant')
            # take deletion decision from this subiteration's LUT
            D = np.take(lut, N)
            # perform deletion
            skel[D] = 0

        after = np.sum(skel) # coint points after thinning

        if before == after:
            # iteration had no effect: finish
            break

        # count down to iteration limit (or endlessly negative)
        n -= 1

    return skel.astype(np.bool)

