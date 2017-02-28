
import numpy as np
import scipy.signal as scs
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans


def rgb2gray(rgb):
    r, g, b = np.rollaxis(rgb[..., :3], axis=-1)
    return 0.299 * r + 0.587 * g + 0.114 * b


def fb_create(num_orient=8, start_sigma=1.0, num_scales=2,
              scaling=np.sqrt(2), elong=2):
    support = 3
    fb = np.empty((2 * num_orient, num_scales), dtype='object')
    for scale in range(0, num_scales):
        sigma = start_sigma * scaling**scale
        for orient in range(0, num_orient):
            theta = orient / num_orient * np.pi
            fb[2 * orient - 1, scale] = oe_filter(sigma * np.array([elong, 1]),
                                                  support, theta, 2, False)
            fb[2 * orient, scale] = oe_filter(sigma * np.array([elong, 1]),
                                              support, theta, 2, True)
    return fb


def isum(x, idx, nbins):
    acc = np.zeros(nbins)
    idx = (idx.T).ravel()
    x = (x.T).ravel()
    for i in range(0, len(x)):
        if idx[i] < 1:
            continue
        if idx[i] > nbins:
            continue
        acc[idx[i] - 1] += x[i]
    return acc


def oe_filter(sigma, support, theta, deriv, hil, vis=False):
    hsz = np.max(np.ceil(support * sigma))
    sz = 2 * hsz + 1

    maxsamples = 1000.0
    maxrate = 10.0
    frate = 10.0

    rate = min(maxrate, max(1, np.floor(maxsamples / sz)))
    samples = sz * rate

    r = np.floor(sz / 2) + 0.5 * (1 - 1 / rate)
    dom = np.linspace(-r, r, samples)
    [sx, sy] = np.meshgrid(dom, dom)

    mx = np.round(sx)
    my = np.round(sy)
    membership = (mx + hsz + 1) + (my + hsz) * sz

    su = sx * np.sin(theta) + sy * np.cos(theta)
    sv = sx * np.cos(theta) - sy * np.sin(theta)

    if vis:
        plt.figure()
        plt.plot(sx, sy, '.')
        plt.plot(mx, my, 'o')
        plt.plot(su, sv, 'x')
        plt.show()

    R = r * np.sqrt(2) * 1.01  # radius of domain, enlarged by >sqrt(2)
    fsamples = np.ceil(R * rate * frate)  # number of samples
    fsamples += (fsamples + 1 % 2)  # must be odd
    fdom = np.linspace(-R, R, fsamples)  # domain for function evaluation
    gap = 2 * R / (fsamples - 1)  # distance between samples

    # print(sigma)
    # The function is a Gaussian in the x direction...
    fx = np.exp(-fdom**2 / (2 * sigma[0]**2))
    # .. and a Gaussian derivative in the y direction...
    fy = np.exp(-fdom**2 / (2 * sigma[1]**2))

    if deriv == 1:
        fy = fy * (-fdom / (sigma[1]**2))
    elif deriv == 2:
        fy = fy * (fdom**2 / (sigma[1]**2) - 1)

    if hil:
        fy = np.imag(scs.hilbert(fy))

    xi = np.round(su / gap) + np.floor(fsamples / 2) + 1
    yi = np.round(sv / gap) + np.floor(fsamples / 2) + 1
    f = fx[np.int32(xi)] * fy[np.int32(yi)]
    f = isum(f, np.int32(membership), int(sz**2))
    f = f.reshape(int(sz), int(sz))

    if deriv > 0:
        f -= np.mean(f)

    sumf = np.sum(np.abs(f))
    if sumf > 0:
        f = f / sumf

    return f


def pad_reflect(im, r):
    impad = np.zeros(np.array(im.shape) + 2 * r)
    return impad


def fb_run(fb, im):
    fb_r = (fb.T).ravel()
    maxsz = 0
    for x in fb_r:
        maxsz = max(maxsz, max(x.shape))

    r = int(np.floor(maxsz / 2))
    impad = np.lib.pad(im, (r, r), 'symmetric')
    fim = np.empty(fb_r.shape, dtype='object')
    for i, f in enumerate(fb_r):
        if f.shape[0] < 50:
            fim[i] = scs.convolve2d(impad, f, 'same')
        else:
            fim[i] = scs.fftconvolve(impad, f, mode='same')
        fim[i] = fim[i][r + 1:-r + 1, r + 1:-r + 1]
    return fim


def compute_textons(fim, k):
    d = len(fim)
    n = np.prod(fim[0].shape)
    data = np.zeros((d, n))
    for i in range(0, d):
        data[i, :] = fim[i].ravel()
    textons, _ = kmeans(data.T, k)
    return textons


def dist_sqr(x, y):
    assert x.shape[0] == y.shape[0]
    d, n = x.shape
    d, m = y.shape
    z = np.dot(x.T, y)
    x2 = np.sum(x**2, axis=0).T
    y2 = np.sum(y**2, axis=0)
    for i in range(0, m):
        z[:, i] = x2 + y2[i] - 2 * z[:, i]
    return z


def assign_textons(fim, textons):
    d = len(fim)
    n = np.prod(fim[0].shape)
    data = np.zeros(d, n)
    for i in range(0, d):
        data[i, :] = fim[i].ravel()
    d2 = dist_sqr(data, textons)
    map = np.min(d2, axis=1)
    w, h = fim[0].shape
    map = map.reshape(w, h)
    return map
