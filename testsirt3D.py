from futhark import SIRT3D
import numpy as np
import tomo_lib
import dataprint_lib
import sys
import astra
import os

def main(argv):
    size = 64
    theta_rad = tomo_lib.get_angles(size)
    rays = tomo_lib.get_rays(size)
    phantom = tomo_lib.get_phantom3D(size)
    tomo_lib.save_3D_slices(phantom,"output//phantom3d", size, 10);
    phantom = phantom.flatten().astype(np.float32)
    sinogram = tomo_lib.get_sinogram3D(phantom.reshape(size,size,size), rays,theta_rad)

    sirt = SIRT3D.SIRT3D()
    sirtresult = sirt.main(theta_rad.astype(np.float32), rays.astype(np.float32), sinogram.flatten().astype(np.float32), 200, size).get()
    tomo_lib.save_3D_slices(sirtresult.reshape(size,size,size),"output//sirt3d", size, 10);


if __name__ == '__main__':
    main(sys.argv)
