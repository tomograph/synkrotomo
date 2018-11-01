import numpy as np
import tomo_lib
import sys

def main(argv):
    size = 64
    theta_deg = tomo_lib.get_angles(size)
    theta_rad = tomo_lib.get_angles(size, degrees=False)
    rays = tomo_lib.get_rays(size)
    phantom = tomo_lib.get_phantom(size)
    sinogram = tomo_lib.sinogram(phantom, theta_deg)
    #generate you backprojection
    tomo_lib.savebackprojection("output//backprojection.png",backprojection)

if __name__ == '__main__':
    main(sys.argv)
