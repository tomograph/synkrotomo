from futhark import projdiff
import numpy as np
import tomo_lib
import dataprint_lib
import sys

def rescale(values):
    minimum = np.min(values)
    maximum = np.max(values)
    return (values - minimum)/(maximum-minimum)

def main(argv):
    size = 128
    theta_deg = tomo_lib.get_angles(size)
    theta_rad = tomo_lib.get_angles(size, degrees=False)
    rays = tomo_lib.get_rays(size)
    phantom = tomo_lib.get_phantom(size)
    sinogram = tomo_lib.sinogram(phantom, theta_deg)
    sinogram = rescale(sinogram)
    tomo_lib.savesinogram("output//sinogram_standard.png",tomo_lib.sinogram(phantom, theta_deg), numrays=len(rays), numangles=len(theta_deg))

    pd = projdiff.projdiff()
    result = pd.main( theta_rad.astype(np.float32), rays.astype(np.float32), size, phantom.flatten().astype(np.float32), 32).get()
    result = rescale(result)
    np.savetxt("projdifftest.txt", np.array(result), delimiter="," )
    tomo_lib.savesinogram("output//sinogram_projdiff.png",result, numrays=len(rays), numangles=len(theta_rad))


if __name__ == '__main__':
    main(sys.argv)
