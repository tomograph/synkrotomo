import numpy as np
import tomo_lib
import sys
from futhark import backprojection_map
def main(argv):
    size = 64
    theta_deg = tomo_lib.get_angles(size)
    theta_rad = tomo_lib.get_angles(size, degrees=False)
    rays = tomo_lib.get_rays(size)
    phantom = tomo_lib.get_phantom(size)
    sinogram = tomo_lib.sinogram(phantom, theta_deg)
    #generate you backprojection
    back = backprojection_map.backprojection_map()
    backproj = back.main(rays.astype(np.float32), theta_rad.astype(np.float32), sinogram.flatten().astype(np.float32), size, 32).get()
    tomo_lib.savebackprojection("output//backprojection.png", backproj, size)

if __name__ == '__main__':
    main(sys.argv)
