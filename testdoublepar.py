import astra
import numpy as np
import pylab
import time
import sys
from futhark import backprojection
from futhark import forwardprojection
from sysmatjh import intersections
from futhark import mette
from skimage.transform import rotate
from skimage.draw import random_shapes
from matplotlib import pyplot
import pandas as pd
from functools import partial
import timeit
import math
import pickle
import matplotlib.pyplot as plt
plt.switch_backend('agg')

###############################################################################
#Getting parameters for algorithms
###############################################################################
def get_phantom(size):
    return random_shapes((size, size), min_shapes=5, max_shapes=10, multichannel=False, random_seed=0)[0]

def get_rays(size):
    startvalue = (size-1)/2.0
    return np.linspace((-(1.0)*startvalue), startvalue, size).astype(np.float32)

def sinogram(image, theta):
    sinogram = np.zeros((len(theta), max(image.shape)))
    for i in range(0, len(theta)):
        rotated_image = rotate(image, theta[i], resize=False)
        sinogram[i] = sum(rotated_image)
    return sinogram.astype(np.float32)


def futhark_FP_doubleparallel(angles, rays, size, phantom):
    proj = mette.mette()
    return proj.main(angles, rays, size, phantom.flatten().astype(np.float32))

###############################################################################
#Time algorithms, and plot the results
###############################################################################
def main(argv):

    size = 64
    phantom = get_phantom(size)
    rays = get_rays(size)
    angles = np.linspace(0,np.pi,50,False).astype(np.float32)
    angles_deg = np.linspace(0,180,50,False).astype(np.float32)
    singram = sinogram(phantom,angles_deg)
    pylab.imsave("sinogramtest_fut_par.png", futhark_FP_doubleparallel(angles, rays, size, phantom.flatten().astype(np.float32)).get().reshape((len(angles),len(rays))))
    pylab.imsave("sinogramtest_compare.png", singram.reshape((len(angles),len(rays))))



if __name__ == '__main__':
    main(sys.argv)
