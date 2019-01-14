import math
from skimage.transform import rotate
from skimage.draw import random_shapes
from skimage.util import random_noise
import numpy as np
import scipy.io
import scipy.misc
import astra
from scipy import ndimage

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

def get_sinogram3D(phantom, rays, angles):
    vol_geom = astra.create_vol_geom(phantom.shape)
    proj_geom =astra.create_proj_geom('parallel3d', 1.0, 1.0, len(rays), len(rays), angles)
    proj_id, proj_data = astra.create_sino3d_gpu(phantom, proj_geom, vol_geom)

    # Create a data object for the reconstruction
    volume_id = astra.data3d.create('-vol', vol_geom, phantom)

    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict("FP_CUDA")
    cfg['VolumeDataId'] = volume_id
    cfg['ProjectionDataId'] = proj_id
    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    # Get the result
    result = astra.data3d.get(proj_id)

    # Clean up. Note that GPU memory is tied up in the algorithm object,
    # and main RAM in the data objects.
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(volume_id)
    astra.data3d.delete(proj_id)

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

def add_noise(image):
    return random_noise(image, mode='poisson', seed=None, clip=True)


def get_phantom(size):
    full_image = random_shapes((size, size), min_shapes=5, max_shapes=10, multichannel=False, random_seed=0, allow_overlap=True)[0]/255
    #Fit within unit disk
    r = size/2;
    y,x = np.ogrid[-r: r, -r: r]
    return np.where(x**2+y**2 <= r**2, full_image,0.0);

def get_phantom3D(size):
    #hollow cube
    cube = np.zeros((size,size,size))
    cube[size/8:size-size/8,size/8:size-size/8,size/8:size-size/8] = 1
    cube[size/4:size-size/4,size/4:size-size/4,size/4:size-size/4] = 0
    return cube;

def save_3D_slices(volume, directory, size, num_slices):
    if not os.path.exists(directory):
        os.makedirs(directory)
    max = np.amax(volume)
    for i in np.nditer(np.linspace(0.,size,num_slices,endpoint=False)):
        i = int(i)
        scipy.misc.toimage(volume[i,:,:], cmin=0, cmax=max).save(directory+"reconstructionx"+ format(i,'03d') + ".png")
        scipy.misc.toimage(volume[:,i,:], cmin=0, cmax=max).save(directory+"reconstructiony"+ format(i,'03d') + ".png")
        scipy.misc.toimage(volume[:,:,i], cmin=0, cmax=max).save(directory+"reconstructionz"+ format(i,'03d') + ".png")
