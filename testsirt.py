from futhark import SIRT
from futhark import backprojection
from futhark import forwardprojection
import numpy as np
import tomo_lib
import dataprint_lib
import sys

def rescale(values):
    minimum = np.min(values)
    maximum = np.max(values)
    return (values - minimum)/(maximum-minimum)

def main(argv):
    size = 64
    theta_deg = tomo_lib.get_angles(size)
    theta_rad = tomo_lib.get_angles(size, degrees=False)
    rays = tomo_lib.get_rays(size)
    phantom = tomo_lib.get_phantom(size)
    sinogram = tomo_lib.sinogram(phantom, theta_deg)
    sinogram = rescale(sinogram)
    phantom = rescale(phantom.flatten().astype(np.float32))


    forward = forwardprojection.forwardprojection()
    result = forward.main(theta_rad.astype(np.float32), rays.astype(np.float32), phantom, sinogram.flatten().astype(np.float32)).get()
    result = rescale(result)
    tomo_lib.savesinogram("output//forwardprojection.png",result, len(rays), len(theta_rad))

    back = backprojection.backprojection()
    result = back.main(theta_rad.astype(np.float32), rays.astype(np.float32), phantom, sinogram.flatten().astype(np.float32)).get()
    result = rescale(result)
    tomo_lib.savebackprojection("output//backprojection.png",result, size)

    tomo_lib.savebackprojection("output//original.png",phantom, size)

    sirt = SIRT.SIRT()
    result = sirt.main(theta_rad.astype(np.float32), rays.astype(np.float32), np.zeros(size*size).flatten().astype(np.float32), sinogram.flatten().astype(np.float32), 10).get()
    result = rescale(result)
    tomo_lib.savebackprojection("output//sirt.png",result, size)


if __name__ == '__main__':
    main(sys.argv)
