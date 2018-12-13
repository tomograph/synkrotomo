from futhark import SIRT
from futhark import backprojection
from futhark import forwardprojection
import numpy as np
import tomo_lib
import dataprint_lib
import sys
import astra

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

def main(argv):
    size = 256
    theta_deg = tomo_lib.get_angles(size)
    theta_rad = tomo_lib.get_angles(size, degrees=False)
    rays = tomo_lib.get_rays(size)
    phantom = tomo_lib.get_phantom(size)
    sinogram = tomo_lib.sinogram(phantom, theta_deg)
    sinogram = rescale(sinogram)
    phantom = rescale(phantom.flatten().astype(np.float32))


    forward = forwardprojection.forwardprojection()
    fpresult = forward.main(theta_rad.astype(np.float32), rays.astype(np.float32), phantom, sinogram.flatten().astype(np.float32), 1).get()
    fpresult = rescale(fpresult)
    tomo_lib.savesinogram("output//forwardprojection.png",fpresult, len(rays), len(theta_rad))

    back = backprojection.backprojection()
    bpresult = back.main(theta_rad.astype(np.float32), rays.astype(np.float32), phantom, fpresult.flatten().astype(np.float32), 1).get()
    bpresult = rescale(bpresult)
    tomo_lib.savebackprojection("output//backprojection.png",bpresult, size)

    tomo_lib.savebackprojection("output//original.png",phantom, size)

    sirt = SIRT.SIRT()
    sirtresult = sirt.main(theta_rad.astype(np.float32), rays.astype(np.float32), np.zeros(size*size).flatten().astype(np.float32), fpresult.flatten().astype(np.float32), 200).get()
    sirtresult = rescale(sirtresult)
    tomo_lib.savebackprojection("output//sirt.png",sirtresult, size)

    proj_geom =astra.create_proj_geom('parallel', 1.0, size, theta_rad)
    vol_geom = astra.create_vol_geom(size)
    astrabp = astra_BP(proj_geom, fpresult.reshape(len(theta_rad),(len(rays))), vol_geom)
    tomo_lib.savebackprojection("output//astrabp.png",astrabp, size)


if __name__ == '__main__':
    main(sys.argv)
