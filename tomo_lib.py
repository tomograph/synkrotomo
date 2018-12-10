import math
from skimage.transform import rotate
from skimage.draw import random_shapes
import numpy as np
import scipy.io
import scipy.misc

def get_angles(size, degrees=True):
    num_angles = math.ceil(size*math.pi/2)
    if degrees:
        return np.linspace(0, 22.5, num_angles, False)
    else:
        return np.linspace(0,np.pi/8, num_angles,False)

def sinogram(image, theta):
    sinogram = np.zeros((len(theta), max(image.shape)))
    for i in range(0, len(theta)):
        rotated_image = rotate(image, theta[i], resize=False)
        sinogram[i] = sum(rotated_image)
    return sinogram

def get_rays(size):
    startvalue = (size-1)/2.0
    return np.linspace((-(1.0)*startvalue), startvalue, size).astype(np.float32)

def savesinogram(filename, data, numrays, numangles):
    reshaped = data.reshape((numangles,numrays))
    scipy.misc.toimage(reshaped).save(filename)

def savebackprojection(filename, data, size):
    max = np.amax(data)
    scipy.misc.toimage(data.reshape((size,size))).save(filename)

def get_phantom(size):
    return random_shapes((size, size), min_shapes=5, max_shapes=10, multichannel=False, random_seed=0)[0]/255
