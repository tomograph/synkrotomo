import math
from skimage.transform import rotate
from skimage.draw import random_shapes
import numpy as np
import scipy.io
import scipy.misc

def get_angles(size, degrees = False):
    num_angles = math.ceil(size*math.pi/2)
    if degrees:
        return np.linspace(0, 180, num_angles,False)
    return np.linspace(0, np.pi, num_angles,False)

def get_rays(size):
    numrays = np.sqrt(2*(size**2))
    startvalue = (numrays-1)/2.0
    return np.linspace((-(1.0)*startvalue), startvalue, numrays).astype(np.float32)

def savesinogram(filename, data, numrays, numangles):
    reshaped = data.reshape((numangles,numrays))
    scipy.misc.toimage(reshaped).save(filename)

def savebackprojection(filename, data, size):
    reshaped = data.reshape((size,size))
    scipy.misc.toimage(reshaped).save(filename)

def get_phantom(size):
    return random_shapes((size, size), min_shapes=5, max_shapes=10, multichannel=False, random_seed=0)[0]/255
