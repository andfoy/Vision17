
import numpy as np
import scipy.signal as scs
import matplotlib.pyplot as plt


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
    # print(idx)
    # print(x.shape)
    idx = (idx.T).ravel()
    x = (x.T).ravel()
    for i in range(0, len(x)):
        if idx[i] < 1:
            continue
        if idx[i] > nbins:
            continue
        acc[idx[i] - 1] += x[i]
    return acc


def oe_filter(sigma, support, theta, deriv, hil, vis=True):
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


def assign_textons(fim, textons):
    d = 0
    print(d)
