import astra
import numpy as np
import pylab
import time
import sys


def test_astra_algorithm(name, phantom, proj_geom, vol_geom, iterations):
    # Create projection data
    proj_id, proj_data = astra.create_sino3d_gpu(phantom, proj_geom, vol_geom)

    # Create a data object for the reconstruction
    rec_id = astra.data3d.create('-vol', vol_geom)

    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict(name)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id

    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)

    start = time.time()
    astra.algorithm.run(alg_id,iterations)
    end = time.time()

    elapsed = (end - start)*1000
    print("The ASTRA algorithm: %s took: %d milliseconds for %d iterations" % (name, elapsed, iterations))

    # Get the result
    result = astra.data3d.get(rec_id)

    # Clean up. Note that GPU memory is tied up in the algorithm object,
    # and main RAM in the data objects.
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(proj_id)

    return result, elapsed

def test_astra_FP(phantom, proj_geom, vol_geom):
    # Create projection data
    proj_id = astra.data3d.create('-sino', proj_geom, 0)

    # Create a data object for the reconstruction
    volume_id = astra.data3d.create('-vol', vol_geom, phantom)

    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict("FP3D_CUDA")
    cfg['VolumeDataId'] = volume_id
    cfg['ProjectionDataId'] = proj_id

    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)

    start = time.time()
    astra.algorithm.run(alg_id)
    end = time.time()

    elapsed = (end - start)*1000
    print("The FP algorithm took: %d milliseconds" % (elapsed))

    # Get the result
    result = astra.data3d.get(proj_id)

    # Clean up. Note that GPU memory is tied up in the algorithm object,
    # and main RAM in the data objects.
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(volume_id)
    astra.data3d.delete(proj_id)

    return result, elapsed

def hollow_cube_phantom(size):
    # Create a simple hollow cube phantom
    cube = np.zeros((size,size,size))
    cube[17:size-15,17:size-15,17:size-15] = 1
    cube[33:size-31,33:size-31,33:size-31] = 0
    return cube


def main(argv):
    size = 256
    iterations = 200

    vol_geom = astra.create_vol_geom(size, size, size)

    angles = np.linspace(0, np.pi, 180,False)
    proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, size, size, angles)
    cube = hollow_cube_phantom(size)

    reconstruction, elapsed = test_astra_algorithm('SIRT3D_CUDA', cube, proj_geom, vol_geom, iterations)
    bp, elapsed = test_astra_algorithm('BP3D_CUDA', cube, proj_geom, vol_geom, 1)
    fp, elapsed = test_astra_FP(cube, proj_geom, vol_geom)

    #Save slice of the recosntruction
    pylab.imsave("reconstruction.png",reconstruction[:,:,size/2])
    #Save slice of the bp
    pylab.imsave("bp.png",bp[:,:,size/2])
    #Save slice of the fp
    pylab.imsave("fp.png",fp[size/2,:,:])



if __name__ == '__main__':
    main(sys.argv)
