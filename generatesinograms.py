from futhark import forwardprojection_doubleparallel
from futhark import forwardprojection_jh
from futhark import forwardprojection_map
import numpy as np
import tomo_lib
import dataprint_lib
import sys

def main(argv):
    size = 32
    theta_deg = tomo_lib.get_angles(size)
    theta_rad = tomo_lib.get_angles(size, degrees=False)
    rays = tomo_lib.get_rays(size)
    phantom = tomo_lib.get_phantom(size)
    tomo_lib.savesinogram("output//sinogram_standard.png",tomo_lib.sinogram(phantom, theta_deg), numrays=len(rays), numangles=len(theta_deg))

    fp_doubleparallel = forwardprojection_doubleparallel.forwardprojection_doubleparallel()
    dp_result = fp_doubleparallel.main( theta_rad.astype(np.float32), rays.astype(np.float32), phantom.flatten().astype(np.float32), 16).get()
    tomo_lib.savesinogram("output//sinogram_dp.png",dp_result, numrays=len(rays), numangles=len(theta_rad))

    fp_jh = forwardprojection_jh.forwardprojection_jh()
    jh_result = fp_jh.main( theta_rad.astype(np.float32), rays.astype(np.float32), phantom.flatten().astype(np.float32), 16).get()
    tomo_lib.savesinogram("output//sinogram_jh.png",jh_result, numrays=len(rays), numangles=len(theta_rad))

    fp_map = forwardprojection_map.forwardprojection_map()
    map_result = fp_map.main( theta_rad.astype(np.float32), rays.astype(np.float32), phantom.flatten().astype(np.float32), 16).get()
    tomo_lib.savesinogram("output//sinogram_map.png",map_result, numrays=len(rays), numangles=len(theta_rad))


if __name__ == '__main__':
    main(sys.argv)
