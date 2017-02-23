#! /usr/bin/env python

import os
import sys
import cv2
import glob
import argparse
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=14.0)
plt.rc('legend', fontsize=16.0)
plt.rc('font', weight='normal')


parser = argparse.ArgumentParser(description='Synthetize hybrid image from two input images')
parser.add_argument('files', metavar='file', nargs=2,
                    help='Input images to be merged')
parser.add_argument('--lowpass',
                    default=0.5,
                    help='Cutoff frequency for the low-pass filter')
parser.add_argument('--highpass',
                    default=1,
                    help='Cutoff frequency for the high-pass filter')
parser.add_argument('--save',
                    default='',
                    help='Image output filename')
parser.add_argument('--silent',
                    action="store_true",
                    default=False,
                    help="Do not display informative plots")

def butterworth(shape, cutoff, n):
    x = np.linspace(-shape[0]/2.0, shape[0]/2.0, shape[0])
    y = np.linspace(-shape[1]/2.0, shape[1]/2.0, shape[1])
    X,Y = np.meshgrid(x,y)
    Z = 1.0/(1 + ((X/cutoff)**2 + (Y/cutoff)**2)**n)
    return Z

def pyramid_built_up(img, n):
    composite = img
    last_lvl, cur_lvl = img, cv2.pyrDown(img)
    n -= 1
    for lvl in range(n):
        H, W, C = composite.shape
        mask = np.zeros((H, W + cur_lvl.shape[1], C))
        # print(composite.shape)
        # print(cur_lvl.shape)
        # print(mask.shape)
        mask[-H:, 0:W] = composite
        mask[-cur_lvl.shape[0]:, W:W + cur_lvl.shape[1]] = cur_lvl
        composite = mask
        last_lvl, cur_lvl = cur_lvl, cv2.pyrDown(cur_lvl)
    return composite

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    path1, path2 = args.files
    lp_freq = float(args.lowpass)
    hp_freq = float(args.highpass)
    save_path = args.save
    silent = args.silent

    img1 = mpimg.imread(path1)
    img2 = mpimg.imread(path2)

    order = 50.0
    size_1 = (21, 21)
    size_2 = (11, 11)
    lp1 = butterworth(size_1, lp_freq, order)
    lp1_t = np.abs(fft.ifft2(lp1))
    lp2 = butterworth(size_2, hp_freq, order)
    lp2_t = np.abs(fft.ifft2(lp2))

    if not silent:
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

    lp_img = cv2.filter2D(img1, -1, lp1_t)
    hp_img = img2 - cv2.filter2D(img2, -1, lp1_t)

    synth = lp_img + hp_img

    if len(save_path) > 0:
        print("Saving hybrid image...")
        mpimg.imsave(save_path, synth)

    if not silent:
        print("Displaying hybrid image...")
        plt.figure()
        plt.imshow(synth)
        plt.title('Hybrid Image Output', fontsize=14)
        # plt.show()

    print("Displaying Gaussian Pyramid...")
    print('\nPress Ctrl+C to stop process')
    lvls = 5
    pyr = pyramid_built_up(synth, lvls)
    # B, G, R = np.rollaxis(pyr[...,:3], axis = -1)
    plt.figure()
    plt.imshow(1-pyr)
    plt.title('Hybrid Image Gaussian Pyramid', fontsize=14)
    plt.show()


