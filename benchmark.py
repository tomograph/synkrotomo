import astra
import numpy as np
import pylab
import time

def test_astra_algorithm(name, phantom, proj_geom, vol_geom, iterations):
    # Create projection data from this
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
    elapsed = end - start
    print("The ASTRA algorithm: %s took: %d seconds" % (name, elapsed))

    # Get the result
    rec = astra.data3d.get(rec_id)

    # Clean up. Note that GPU memory is tied up in the algorithm object,
    # and main RAM in the data objects.
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(proj_id)

    return rec, elapsed

vol_geom = astra.create_vol_geom(128, 128, 128)

angles = np.linspace(0, np.pi, 180,False)
proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, 128, 192, angles)

# Create a simple hollow cube phantom
cube = np.zeros((128,128,128))
cube[17:113,17:113,17:113] = 1
cube[33:97,33:97,33:97] = 0

test_astra_algorithm('SIRT3D_CUDA', cube, proj_geom, vol_geom, 1000)
