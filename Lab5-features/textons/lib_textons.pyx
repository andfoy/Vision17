
cimport cython
import numpy as np
cimport numpy as np
from kmc2 import kmc2
import scipy.signal as scs
import matplotlib.pyplot as plt
# from scipy.cluster.vq import kmeans
from sklearn.cluster import MiniBatchKMeans

DTYPE = np.float64
INT_DYPE = np.int64
OBJ_DTYPE = np.object
LONG_DTYPE = np.long
ctypedef np.float64_t DTYPE_t
ctypedef np.int_t INT_DTYPE_t
ctypedef np.long_t LONG_DTYPE_t

cdef inline DTYPE_t float_max(DTYPE_t a, DTYPE_t b): return a if a >= b else b
cdef inline DTYPE_t float_min(DTYPE_t a, DTYPE_t b): return a if a <= b else b

def rgb2gray(np.ndarray[DTYPE_t, ndim=3] rgb):
    cdef np.ndarray[DTYPE_t, ndim=2] r, g, b
    r, g, b = np.rollaxis(rgb[..., :3], axis=-1)
    return 0.299 * r + 0.587 * g + 0.114 * b

@cython.boundscheck(False)
def fb_create(int num_orient, DTYPE_t start_sigma, int num_scales,
              DTYPE_t scaling,  int elong):
    cdef int support = 3
    cdef np.ndarray fb = np.empty((2 * num_orient, num_scales), dtype='object')
    cdef int scale
    cdef DTYPE_t theta, sigma
    cdef np.ndarray[DTYPE_t, ndim=1] sig = np.array([elong, 1], dtype=DTYPE)
    for scale in range(0, num_scales):
        sigma = start_sigma * scaling**scale
        for orient in range(0, num_orient):
            theta = orient / num_orient * np.pi
            fb[2 * orient - 1, scale] = oe_filter(sigma * sig,
                                                  support, theta, 2, False, False)
            fb[2 * orient, scale] = oe_filter(sigma * sig,
                                              support, theta, 2, True, False)
    return fb


@cython.boundscheck(False)
def isum(np.ndarray[DTYPE_t, ndim=2] x, np.ndarray[LONG_DTYPE_t, ndim=2] idx, int nbins):
    cdef np.ndarray[DTYPE_t, ndim=1] acc = np.zeros(nbins)
    cdef np.ndarray[LONG_DTYPE_t, ndim=1] idx_2 = (idx.T).ravel()
    cdef np.ndarray[DTYPE_t, ndim=1] x_2 = (x.T).ravel()
    cdef int i
    for i in range(0, len(x_2)):
        if idx_2[i] < 1:
            continue
        if idx_2[i] > nbins:
            continue
        acc[idx_2[i] - 1] += x_2[i]
    return acc

@cython.boundscheck(False)
def oe_filter(np.ndarray[DTYPE_t, ndim=1] sigma, int support, DTYPE_t theta,
              int deriv, bint hil, bint vis):
    cdef DTYPE_t hsz = np.max(np.ceil(support * sigma))
    cdef DTYPE_t sz = 2 * hsz + 1

    cdef DTYPE_t maxsamples = 1000.0
    cdef DTYPE_t maxrate = 10.0
    cdef DTYPE_t frate = 10.0

    cdef DTYPE_t rate = float_min(maxrate, float_max(1, np.floor(maxsamples / sz)))
    cdef DTYPE_t samples = sz * rate

    cdef DTYPE_t r = np.floor(sz / 2) + 0.5 * (1 - 1 / rate)
    cdef np.ndarray[DTYPE_t, ndim=1] dom = np.linspace(-r, r, samples)
    cdef np.ndarray[DTYPE_t, ndim=2] sx
    cdef np.ndarray[DTYPE_t, ndim=2] sy
    [sx, sy] = np.meshgrid(dom, dom)

    cdef np.ndarray[DTYPE_t, ndim=2] mx = np.round(sx)
    cdef np.ndarray[DTYPE_t, ndim=2] my = np.round(sy)
    cdef np.ndarray[DTYPE_t, ndim=2] membership = (mx + hsz + 1) + (my + hsz) * sz

    cdef np.ndarray[DTYPE_t, ndim=2] su = sx * np.sin(theta) + sy * np.cos(theta)
    cdef np.ndarray[DTYPE_t, ndim=2] sv = sx * np.cos(theta) - sy * np.sin(theta)

    if vis:
        plt.figure()
        plt.plot(sx, sy, '.')
        plt.plot(mx, my, 'o')
        plt.plot(su, sv, 'x')
        plt.show()

    cdef DTYPE_t R = r * np.sqrt(2) * 1.01  # radius of domain, enlarged by >sqrt(2)
    cdef DTYPE_t fsamples = np.ceil(R * rate * frate)  # number of samples
    fsamples += (fsamples + 1 % 2)  # must be odd
    cdef np.ndarray[DTYPE_t, ndim=1] fdom = np.linspace(-R, R, fsamples)  # domain for function evaluation
    cdef DTYPE_t gap = 2 * R / (fsamples - 1)  # distance between samples

    # The function is a Gaussian in the x direction...
    cdef np.ndarray[DTYPE_t, ndim=1] fx = np.exp(-fdom**2 / (2 * sigma[0]**2))
    # .. and a Gaussian derivative in the y direction...
    cdef np.ndarray[DTYPE_t, ndim=1] fy = np.exp(-fdom**2 / (2 * sigma[1]**2))

    if deriv == 1:
        fy = fy * (-fdom / (sigma[1]**2))
    elif deriv == 2:
        fy = fy * (fdom**2 / (sigma[1]**2) - 1)

    if hil:
        fy = np.imag(scs.hilbert(fy))

    cdef np.ndarray[DTYPE_t, ndim=2] xi = np.round(su / gap) + np.floor(fsamples / 2) + 1
    cdef np.ndarray[DTYPE_t, ndim=2] yi = np.round(sv / gap) + np.floor(fsamples / 2) + 1
    cdef np.ndarray[DTYPE_t, ndim=2] f = fx[np.int64(xi)] * fy[np.int64(yi)]
    cdef np.ndarray[DTYPE_t, ndim=1] f_sum = isum(f, np.int64(membership), int(sz**2))
    f = f_sum.reshape(int(sz), int(sz))

    if deriv > 0:
        f -= np.mean(f)

    cdef DTYPE_t sumf = np.sum(np.abs(f))
    if sumf > 0:
        f = f / sumf

    return f


@cython.boundscheck(False)
def fb_run(np.ndarray fb, np.ndarray[DTYPE_t, ndim=2] im):
    cdef np.ndarray fb_r = (fb.T).ravel()
    cdef DTYPE_t maxsz = 0
    for x in fb_r:
        maxsz = float_max(maxsz, max(x.shape))

    cdef int r = int(np.floor(maxsz / 2))
    cdef np.ndarray[DTYPE_t, ndim=2] impad = np.lib.pad(im, (r, r), 'symmetric')
    cdef np.ndarray fim = np.empty((len(fb_r),), dtype='object')
    cdef int i
    cdef np.ndarray[DTYPE_t, ndim=2] f
    for i, f in enumerate(fb_r):
        if f.shape[0] < 50:
            fim[i] = scs.convolve2d(impad, f, 'same')
        else:
            fim[i] = scs.fftconvolve(impad, f, mode='same')
        fim[i] = fim[i][r + 1:-r + 1, r + 1:-r + 1]
    return fim

@cython.boundscheck(False)
def compute_textons(np.ndarray fim, int k):
    cdef int d = len(fim)
    cdef int n = np.prod(fim[0].shape)
    cdef np.ndarray[DTYPE_t, ndim=2] data = np.zeros((d, n))
    cdef np.ndarray[DTYPE_t, ndim=2] textons
    cdef int i
    for i in range(0, d):
        data[i, :] = fim[i].ravel()
    # textons, _ = kmeans(data.T, k)
    cdef np.ndarray[DTYPE_t, ndim=2] seeding = kmc2(data.T, k)
    model = MiniBatchKMeans(k, init=seeding).fit(data.T)
    textons = model.cluster_centers_
    return textons

@cython.boundscheck(False)
def dist_sqr(np.ndarray[DTYPE_t, ndim=2] x, np.ndarray[DTYPE_t, ndim=2] y):
    assert x.shape[0] == y.shape[0]
    cdef int d, n, m
    d, n = x.shape[0], x.shape[1]
    d, m = y.shape[0], y.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] z = np.dot(x.T, y)
    cdef np.ndarray[DTYPE_t, ndim=1] x2 = np.sum(x**2, axis=0).T
    cdef np.ndarray[DTYPE_t, ndim=1] y2 = np.sum(y**2, axis=0)
    cdef int i
    for i in range(0, m):
        z[:, i] = x2 + y2[i] - 2 * z[:, i]
    return z

@cython.boundscheck(False)
def assign_textons(np.ndarray fim, np.ndarray[DTYPE_t, ndim=2] textons):
    cdef int d = len(fim)
    cdef int n = np.prod(fim[0].shape)
    cdef np.ndarray[DTYPE_t, ndim=2] data = np.zeros((d, n))
    cdef int i
    for i in range(0, d):
        data[i, :] = fim[i].ravel()
    cdef np.ndarray[DTYPE_t, ndim=2] d2 = dist_sqr(data, textons)
    cdef np.ndarray[INT_DTYPE_t, ndim=1] map = np.argmin(d2, axis=1)
    w, h = fim[0].shape
    cdef np.ndarray[INT_DTYPE_t, ndim=2] map_r = map.reshape(w, h)
    return map_r
