from futhark import SIRT3D
import numpy as np
import tomo_lib
import dataprint_lib
import sys
import astra
import paralleltomo
from scipy.sparse import csr_matrix

def rescale(values):
    minimum = np.min(values)
    maximum = np.max(values)
    return (values - minimum)/(maximum-minimum)

def astra_BP(projgeom, sinogram, volgeom):
    # Create projection data
    proj_id = astra.data2d.create('-sino', projgeom, sinogram)

    # Create a data object for the reconstruction
    rec_id = astra.data2d.create('-vol', volgeom)

    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict("BP_CUDA")
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    # Get the result
    result = astra.data2d.get(rec_id)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(proj_id)

    return result

def astra_reconstruction(projgeom, sinogram, volgeom, algorithm = "SIRT_CUDA"):
    # Create projection data
    proj_id = astra.data2d.create('-sino', projgeom, sinogram)

    # Create a data object for the reconstruction
    rec_id = astra.data2d.create('-vol', volgeom)

    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict(algorithm)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, 200)
    # Get the result
    result = astra.data2d.get(rec_id)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(proj_id)

    return result

def astra_projectionmatrix(proj_geom, vol_geom):
    # For CPU-based algorithms, a "projector" object specifies the projection
    # model used. In this case, we use the "line" model.
    proj_id = astra.create_projector('line', proj_geom, vol_geom)

    # Generate the projection matrix for this projection model.
    # This creates a matrix W where entry w_{i,j} corresponds to the
    # contribution of volume element j to detector element i.
    matrix_id = astra.projector.matrix(proj_id)

    # Get the projection matrix as a Scipy sparse matrix.
    W = astra.matrix.get(matrix_id)
    astra.projector.delete(proj_id)
    astra.matrix.delete(matrix_id)

    return W

def main(argv):
    size = 128
    theta_rad = tomo_lib.get_angles(size)
    rays = tomo_lib.get_rays(size)
    phantom = tomo_lib.get_phantom3D(size)
    phantom = phantom.flatten().astype(np.float32)
    sinogram = rescale(tomo_lib.get_sinogram3D(phantom.reshape(size,size,size), rays,theta_rad))

    sirt = SIRT3D.SIRT3D()
    sirtresult = sirt.main(theta_rad.astype(np.float32), rays.astype(np.float32), np.zeros(size*size).flatten().astype(np.float32), sinogram.flatten().astype(np.float32), 200, size).get()
    sirtresult = rescale(sirtresult)
    tomo_lib.save_3D_slices(sirtresult.reshape(size,size,size),"/output", size, 5);


if __name__ == '__main__':
    main(sys.argv)
