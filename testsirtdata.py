from futhark import SIRT
from futhark import backprojection
from futhark import forwardprojection
import numpy as np
import tomo_lib
import dataprint_lib
import sys
# import astra

def rescale(values):
    minimum = np.min(values)
    maximum = np.max(values)
    return (values - minimum)/(maximum-minimum)
#
# def astra_BP(projgeom, sinogram, volgeom):
#     # Create projection data
#     proj_id = astra.data2d.create('-sino', projgeom, sinogram)
#
#     # Create a data object for the reconstruction
#     rec_id = astra.data2d.create('-vol', volgeom)
#
#     # Set up the parameters for a reconstruction algorithm using the GPU
#     cfg = astra.astra_dict("BP_CUDA")
#     cfg['ReconstructionDataId'] = rec_id
#     cfg['ProjectionDataId'] = proj_id
#     # Create the algorithm object from the configuration structure
#     alg_id = astra.algorithm.create(cfg)
#     astra.algorithm.run(alg_id)
#     # Get the result
#     result = astra.data2d.get(rec_id)
#
#     astra.algorithm.delete(alg_id)
#     astra.data2d.delete(rec_id)
#     astra.data2d.delete(proj_id)
#
#     return result

def main(argv):
    # size = 256
    sizes = [64,128,256,512,1024,2048,4096]
    for i in sizes:
        filename = "data//bpinputf32rad"+str(i)
        theta_deg = tomo_lib.get_angles(i)
        theta_rad = tomo_lib.get_angles(i, degrees=False)
        rays = tomo_lib.get_rays(i)
        phantom = tomo_lib.get_phantom(i)
        sinogram = tomo_lib.sinogram(phantom, theta_deg)
        sinogram = rescale(sinogram)
        phantom = rescale(phantom.flatten().astype(np.float32))


    # forward = forwardprojection.forwardprojection()
    # result = forward.main(theta_rad.astype(np.float32), rays.astype(np.float32), phantom, sinogram.flatten().astype(np.float32)).get()
    # result = rescale(result)
    # tomo_lib.savesinogram("output//forwardprojection.png",result, len(rays), len(theta_rad))

    # back = backprojection.backprojection()
    # filename = "testdata"
        f = open(filename,"w+")
        f.write(dataprint_lib.print_f32array(theta_rad)+" "+dataprint_lib.print_f32array(rays)+" "+dataprint_lib.print_f32array(phantom)+" "+dataprint_lib.print_f32array(sinogram.flatten()))
    # print(theta_rad.astype(np.float32), rays.astype(np.float32), phantom, sinogram.flatten().astype(np.float32))
    # result = rescale(result)
    # tomo_lib.savebackprojection("output//backprojection.png",result, size)
    #
    # tomo_lib.savebackprojection("output//original.png",phantom, size)
    #
    # sirt = SIRT.SIRT()
    # result = sirt.main(theta_rad.astype(np.float32), rays.astype(np.float32), np.zeros(size*size).flatten().astype(np.float32), sinogram.flatten().astype(np.float32), 1).get()
    # result = rescale(result)
    # tomo_lib.savebackprojection("output//sirt.png",result, size)

    # proj_geom =astra.create_proj_geom('parallel', 1.0, size, theta_rad)
    # vol_geom = astra.create_vol_geom(size)
    # astrabp = astra_BP(proj_geom, sinogram, vol_geom)
    # tomo_lib.savebackprojection("output//astrabp.png",astrabp, size)


if __name__ == '__main__':
    main(sys.argv)
