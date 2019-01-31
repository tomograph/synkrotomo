from futhark import SIRT
import numpy as np
import tomo_lib
import sys

def main(argv):
    size = 256
    theta_rad = tomo_lib.get_angles(size)
    rays = tomo_lib.get_rays(size)
    phantom = tomo_lib.get_phantom(size)
    #Rescales to [0,1]
    phantom = tomo_lib.rescale(phantom.flatten().astype(np.float32))

    sirt = SIRT.SIRT()
    sirtresult = sirt.main(theta_rad.astype(np.float32), rays.astype(np.float32), np.zeros(size*size).flatten().astype(np.float32), sinogram.flatten().astype(np.float32), 200).get()
    tomo_lib.savebackprojection("//output//samples//futhark_sirt.png",sirtresult, size)

    #savebackprojection essentially just saves size x size image from data
    tomo_lib.savebackprojection("//output//samples//futhark_sirt_phantom.png",phantom, size)

if __name__ == '__main__':
    main(sys.argv)
