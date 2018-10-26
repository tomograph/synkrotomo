import astra
import numpy as np
import pylab
import time
import sys
from futhark import backprojection
from futhark import forwardprojection
from skimage.transform import rotate

def sinogram(image, theta):
    #sinogram = radon(image, theta, circle=False)
    sinogram = np.zeros((len(theta), max(image.shape)))
    for i in range(0, len(theta)):
        rotated_image = rotate(image, theta[i], resize=False)
        sinogram[i] = sum(rotated_image)
    return sinogram

def test_astra_BP(sino, proj_geom, vol_geom):
    # Create projection data
    proj_id = astra.data2d.create('-sino', proj_geom, sino)

    # Create a data object for the reconstruction
    rec_id = astra.data2d.create('-vol', vol_geom)

    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict("BP_CUDA")
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id

    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)

    start = time.time()
    astra.algorithm.run(alg_id)
    end = time.time()

    elapsed = (end - start)*1000
    print("The astra BP algorithm took: %d milliseconds" % (elapsed))

    # Get the result
    result = astra.data2d.get(rec_id)

    # Clean up. Note that GPU memory is tied up in the algorithm object,
    # and main RAM in the data objects.
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(proj_id)

    return result, elapsed

def test_astra_FP(phantom, proj_geom, vol_geom):
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

    start = time.time()
    astra.algorithm.run(alg_id)
    end = time.time()

    elapsed = (end - start)*1000
    print("The astra FP algorithm took: %d milliseconds" % (elapsed))

    # Get the result
    result = astra.data2d.get(proj_id)

    # Clean up. Note that GPU memory is tied up in the algorithm object,
    # and main RAM in the data objects.
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(volume_id)
    astra.data2d.delete(proj_id)

    return result, elapsed

def test_futhark_FP(phantom, size, angles):
    startvalue = (size-1)/2.0
    rays = np.linspace((-(1.0)*startvalue), startvalue, size)
    proj = forwardprojection.forwardprojection()
    start = time.time()
    fp = proj.main(rays.astype(np.float32), angles.astype(np.float32), phantom.flatten().astype(np.float32))
    end = time.time()
    elapsed = (end - start)*1000
    print("The futhark FP algorithm took: %d milliseconds" % (elapsed))
    return np.array(fp.get()).reshape((len(angles),len(rays))), elapsed

def test_futhark_BP(sino, size, angles):
    startvalue = (size-1)/2.0
    rays = np.linspace((-(1.0)*startvalue), startvalue, size)
    proj = backprojection.backprojection()
    start = time.time()
    bp = proj.main(rays.astype(np.float32), angles.astype(np.float32), sino.flatten().astype(np.float32), size)
    end = time.time()
    elapsed = (end - start)*1000
    print("The futhark BP algorithm took: %d milliseconds" % (elapsed))
    return np.array(bp.get()).reshape((size,size)), elapsed

def main(argv):

    size = 512
    num_angles = 100
    #square phantom
    phantom = np.zeros((size, size))
    phantom[133:142, 148:187] = 1 ;
    pylab.imsave("phantom.png",phantom)

    angles_deg = np.linspace(0, 180, num_angles, False)
    s = sinogram(phantom,angles_deg)

    #Test futhark functions
    bp_futhark, elapsed_futbp = test_futhark_BP(s,size,angles_deg)
    fp_futhark, elapsed_futbp = test_futhark_FP(phantom,size,angles_deg)

    #Test astra functions
    angles_rad = np.linspace(0, np.pi, num_angles,False)
    vol_geom = astra.create_vol_geom(size)
    proj_geom = astra.create_proj_geom('parallel', 1.0, size, angles_rad)
    bp_astra, elapsed_abp = test_astra_BP(s, proj_geom, vol_geom)
    fp_astra, elapsed_afp = test_astra_FP(phantom, proj_geom, vol_geom)



    #Save images
    pylab.imsave("bp_astra.png",bp_astra)
    pylab.imsave("fp_astra.png",fp_astra)
    pylab.imsave("fp_futhark.png",fp_futhark)
    pylab.imsave("bp_futhark.png",bp_futhark)


if __name__ == '__main__':
    main(sys.argv)
