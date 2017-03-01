
import numpy as np
from textons import lib_textons
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.ion()
fb = lib_textons.fb_create(8, 1.0, 2, np.sqrt(2), 2)
img1 = mpimg.imread('img/person1.bmp')
img2 = mpimg.imread('img/goat1.bmp')
