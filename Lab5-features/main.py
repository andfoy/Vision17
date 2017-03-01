
import numpy as np
from textons import lib_textons
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.ion()
fb = lib_textons.fb_create(8, 1.0, 2, np.sqrt(2), 2)

img1 = lib_textons.rgb2gray(mpimg.imread('img/person1.bmp') / 255)
img2 = lib_textons.rgb2gray(mpimg.imread('img/goat1.bmp') / 255)
k = 16 * 8

filter_responses = lib_textons.fb_run(fb, np.hstack((img1, img2)))
textons = lib_textons.compute_textons(filter_responses, k)

im_test1 = lib_textons.rgb2gray(mpimg.imread('img/person2.bmp') / 255)
im_test2 = lib_textons.rgb2gray(mpimg.imread('img/goat2.bmp') / 255)

act_1 = lib_textons.fb_run(fb, img1)
tmap_base1 = lib_textons.assign_textons(act_1, textons.T)
tmap_base1 = (tmap_base1.T).ravel()

act_2 = lib_textons.fb_run(fb, img2)
tmap_base2 = lib_textons.assign_textons(act_2, textons.T)
tmap_base2 = (tmap_base2.T).ravel()

act_3 = lib_textons.fb_run(fb, im_test1)
tmap_test1 = lib_textons.assign_textons(act_3, textons.T)
tmap_test1 = (tmap_test1.T).ravel()

act_4 = lib_textons.fb_run(fb, im_test2)
tmap_test2 = lib_textons.assign_textons(act_4, textons.T)
tmap_test2 = (tmap_test2.T).ravel()

D = np.norm(np.histogram(tmap_base1, np.arange(0, k + 1)) / len(tmap_base1) -
            np.histogram(tmap_test1, np.arange(0, k + 1)) / len(tmap_test1))
print(D)
