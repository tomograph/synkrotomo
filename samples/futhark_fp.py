from futhark import forwardprojection
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

    forward = forwardprojection.forwardprojection()
    fpresult = forward.main(theta_rad.astype(np.float32), rays.astype(np.float32), phantom).get()
    tomo_lib.savesinogram("//output//samples//futhark_fp.png",fpresult, len(rays), len(theta_rad))

    #savebackprojection essentially just saves size x size image from data
    tomo_lib.savebackprojection("//output//samples//futhark_bp_phantom.png",phantom, size)

if __name__ == '__main__':
    main(sys.argv)
