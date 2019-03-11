from futhark import SIRT
from futhark import backprojection
from futhark import forwardprojection
import numpy as np
import tomo_lib
import dataprint_lib
import sys
import astra
import scipy
import math

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
    size = 256
    theta_rad = tomo_lib.get_angles(size).astype(np.float32)
    rays = tomo_lib.get_rays(size)
    phantom = tomo_lib.get_phantom(size)
    phantom = phantom.flatten().astype(np.float32)
    sinogram = tomo_lib.get_sinogram(phantom.reshape(size,size), rays,theta_rad)
    rhozero = -180.5
    deltarho = 1.0
    numrhos = len(rays)
    emptyimage = np.zeros(size*size).flatten().astype(np.float32)

    proj_geom =astra.create_proj_geom('parallel', deltarho, numrhos, theta_rad)
    vol_geom = astra.create_vol_geom(size,)
    astra_bp = astra_BP(proj_geom, sinogram, vol_geom)
    scipy.misc.toimage(astra_bp).save('astra_bp.png')
    fp = forwardprojection.forwardprojection()
    fpresult = fp.main(theta_rad.astype(np.float32), rhozero, deltarho, numrhos, phantom.flatten().astype(np.float32)).get()
    tomo_lib.savesinogram("fp.png",fpresult, numrhos, len(theta_rad))
    tomo_lib.savesinogram("astra_fp.png", sinogram.flatten(), numrhos, len(theta_rad))
    bp = backprojection.backprojection()
    bpresult = bp.main(theta_rad, rhozero, deltarho, size, sinogram.flatten().astype(np.float32)).get()
    tomo_lib.savebackprojection("bp.png",bpresult, size)
    tomo_lib.savebackprojection("phantom.png",phantom, size)
    print(len(theta_rad))
    print((len(sinogram.flatten())/len(theta_rad)))
    print(sinogram.shape)
    print(numrhos)
    print(rhozero)
    print(deltarho)

    sirt = SIRT.SIRT()
    sirtresult = sirt.main(theta_rad, rhozero, deltarho, emptyimage, sinogram.flatten().astype(np.float32), 200).get()
    tomo_lib.savebackprojection("sirt.png",sirtresult, size)

    num_angles = math.ceil(size*math.pi/4)
    theta_rad = np.linspace(-np.pi/4, np.pi/4,  num_angles,False)
    sinogram = tomo_lib.get_sinogram(phantom.reshape(size,size), rays,theta_rad)
    proj_geom =astra.create_proj_geom('parallel', deltarho, numrhos, theta_rad)
    astra_sirt_steep = astra_reconstruction(proj_geom, sinogram, vol_geom)

    theta_rad = np.linspace(np.pi/4, 3*np.pi/4,  num_angles,False)
    sinogram = tomo_lib.get_sinogram(phantom.reshape(size,size), rays,theta_rad)
    proj_geom =astra.create_proj_geom('parallel', deltarho, numrhos, theta_rad)
    astra_sirt_flat = astra_reconstruction(proj_geom, sinogram, vol_geom)
    astra_sirt = astra_sirt_steep.flatten() + astra_sirt_flat.flatten()
    tomo_lib.savebackprojection("astra_sirt.png",astra_sirt, size)
if __name__ == '__main__':
    main(sys.argv)
