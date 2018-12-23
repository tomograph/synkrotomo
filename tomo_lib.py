import math
from skimage.transform import rotate
from skimage.draw import random_shapes
import numpy as np
import scipy.io
import scipy.misc
import astra

def get_sinogram(phantom, rays, angles):
    proj_geom =astra.create_proj_geom('parallel', 1.0, len(rays), angles)
    vol_geom = astra.create_vol_geom(phantom.shape[0])
    # Create projection data
    proj_id = astra.data2d.create('-sino', proj_geom, 0)

    # Create a data object for the reconstruction
    volume_id = astra.data2d.create('-vol', vol_geom, phantom)

    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict("FP_CUDA")
    cfg['VolumeDataId'] = volume_id
    cfg['ProjectionDataId'] = proj_id
    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    # Get the result
    result = astra.data2d.get(proj_id)

    # Clean up. Note that GPU memory is tied up in the algorithm object,
    # and main RAM in the data objects.
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(volume_id)
    astra.data2d.delete(proj_id)

    return result

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
