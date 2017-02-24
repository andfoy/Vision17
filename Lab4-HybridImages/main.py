#! /usr/bin/env python

import os
import sys
import cv2
import glob
import argparse
import numpy as np
import os.path as osp
import numpy.fft as fft
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=14.0)
plt.rc('legend', fontsize=16.0)
plt.rc('font', weight='normal')

OUTPUT_FOLDER = 'results'

parser = argparse.ArgumentParser(description='Synthetize hybrid image from two input images')
parser.add_argument('files', metavar='file', nargs=2,
                    help='Input images to be merged')
parser.add_argument('--lowpass',
                    default=0.5,
                    help='Cutoff frequency for the low-pass filter')
parser.add_argument('--highpass',
                    default=1,
                    help='Cutoff frequency for the high-pass filter')
parser.add_argument('--lker_size',
                    metavar="dim",
                    nargs=2,
                    default=(21, 21),
                    help='Low-pass filter kernel size')
parser.add_argument('--hker_size',
                    metavar="dim",
                    nargs=2,
                    default=(11, 11),
                    help='High-pass filter kernel size')
parser.add_argument('--save',
                    default='',
                    help='File output filename')
parser.add_argument('--silent',
                    action="store_true",
                    default=False,
                    help="Do not display informative plots")

def butterworth(shape, cutoff, n):
    """
    Compute the frequency response of a 2D Butterworth filter of order n.

    Implementation note: The output response has no phase, to prevent any image distortion

    Parameters
    ----------
    shape: tuple
        Kernel size.
    cutoff: float
        Cutoff frequency for the filter.
    n: int
        Filter order.

    Returns
    -------
    Z: array_like
        Frequency magnitude response of the filter.
    """
    x = np.linspace(-shape[0]/2.0, shape[0]/2.0, shape[0])
    y = np.linspace(-shape[1]/2.0, shape[1]/2.0, shape[1])
    X,Y = np.meshgrid(x,y)
    Z = 1.0/(1 + ((X/cutoff)**2 + (Y/cutoff)**2)**n)
    return Z

def gaussian_filter(shape, sigma):
    x = np.linspace(-shape[0]/2.0, shape[0]/2.0, shape[0])
    y = np.linspace(-shape[1]/2.0, shape[1]/2.0, shape[1])
    X,Y = np.meshgrid(x,y)
    return np.exp(-(X**2 + Y**2))

def pyramid_built_up(img, n):
    """
    Given an input image, build an image that contains n levels of the Gaussian Pyramid.

    Parameters
    ----------
    img: array_like
        Input image.
    n: int
        Number of pyramid levels to display.
    """
    composite = img
    last_lvl, cur_lvl = img, cv2.pyrDown(img)
    n -= 1
    for lvl in range(n):
        H, W, C = composite.shape
        mask = np.zeros((H, W + cur_lvl.shape[1], C))
        mask[-H:, 0:W] = composite
        mask[-cur_lvl.shape[0]:, W:W + cur_lvl.shape[1]] = cur_lvl
        composite = mask
        last_lvl, cur_lvl = cur_lvl, cv2.pyrDown(cur_lvl)
    return composite

if __name__ == '__main__':
    args = parser.parse_args()

    print('\nPress Ctrl+C to stop process\n')
    path1, path2 = args.files
    lp_freq = float(args.lowpass)
    hp_freq = float(args.highpass)
    save_path = args.save
    silent = args.silent
    size_1 = tuple((int(x) for x in args.lker_size))
    size_2 = tuple((int(x) for x in args.hker_size))
    order = 100.0 # Filter order

    print("Image 1: %s" % (path1))
    print("Image 2: %s" % (path2))
    print("Low-pass cutoff: %g" % (lp_freq))
    print("High-pass cutoff: %g" % (hp_freq))
    print("Low-pass Kernel Size: (%d, %d)" % (size_1))
    print("High-pass Kernel Size: (%d, %d)" % (size_2))
    print("Display plots: "+str(not silent))
    print("Image result path: %s" % (save_path))

    img1 = mpimg.imread(path1) # Load First Image
    img2 = mpimg.imread(path2) # Load Second Image

    lp1 = butterworth(size_1, lp_freq, order) # Low pass frequency response
    lp1_t = np.abs(fft.ifft2(lp1)) # Time domain representation
    lp2 = butterworth(size_2, hp_freq, order) # High pass frequency response
    lp2_t = np.abs(fft.ifft2(lp2)) # Time domain representation

    if not silent:
        img1_name = osp.basename(osp.splitext(path1)[0])
        img2_name = osp.basename(osp.splitext(path2)[0])
        try:
            os.mkdir(OUTPUT_FOLDER)
        except FileExistsError:
            pass
        prefix = '%s_%s' % (img1_name, img2_name)
        folder = osp.join(OUTPUT_FOLDER, prefix)
        try:
            os.mkdir(folder)
        except FileExistsError:
            pass

        print("Displaying Filter Frequency Responses...")
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(121)
        plt.imshow(lp1, cmap='gray')
        plt.title(r'Low pass frequency response $|H_1(j\omega_x, j\omega_y)|$')
        plt.xlabel(r'$j\omega_x$')
        plt.ylabel(r'$j\omega_y$')
        ax = fig.add_subplot(122)
        plt.imshow(1-lp2, cmap='gray')
        plt.title(r'High pass frequency response $|H_2(j\omega_x, j\omega_y)|$')
        plt.xlabel(r'$j\omega_x$')
        plt.ylabel(r'$j\omega_y$')
        plt.savefig(osp.join(folder, 'filters_%s.pdf' % (prefix)) , bbox_inches='tight')

    lp_img = cv2.GaussianBlur(img1, size_1, lp_freq) # Apply Gaussian filter (LP)
    hp_img = img2 - cv2.GaussianBlur(img2, size_2, hp_freq) # Apply Gaussian filter (HP)
    # lp_img = cv2.filter2D(img1, -1, lp1_t) # Convolve first image with low pass
    # hp_img = img2 - cv2.filter2D(img2, -1, lp1_t) # Convolve second image with high-pass

    synth = lp_img + hp_img # Build Hybrid Image

    if len(save_path) > 0:
        print("Saving hybrid image...")
        mpimg.imsave(save_path, synth)

    if not silent:
        print("Displaying hybrid image...")
        plt.figure()
        plt.imshow(synth)
        plt.title('Hybrid Image Output', fontsize=14)
        plt.savefig(osp.join(folder, 'result_%s.pdf' % (prefix)) , bbox_inches='tight')

        print("Displaying Gaussian Pyramid...")
        lvls = 5
        pyr = pyramid_built_up(synth, lvls)
        plt.figure()
        plt.imshow(1-pyr)
        plt.title('Hybrid Image Gaussian Pyramid', fontsize=14)
        plt.savefig(osp.join(folder, 'pyr_%s.pdf' % (prefix)) , bbox_inches='tight')
        plt.show()

