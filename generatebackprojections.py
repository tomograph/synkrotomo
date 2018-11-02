import numpy as np
import tomo_lib
import sys
from futhark import backprojection_map
from futhark import backprojection_jh
from futhark import backprojection_doubleparallel
#from futhark import backprojection_semiflat


def main(argv):
    size = 256
    theta_deg = tomo_lib.get_angles(size)
    theta_rad = tomo_lib.get_angles(size, degrees=False)
    rays = tomo_lib.get_rays(size)
    phantom = tomo_lib.get_phantom(size)
    sinogram = tomo_lib.sinogram(phantom, theta_deg)

    back = backprojection_map.backprojection_map()
    backproj = back.main(theta_rad.astype(np.float32), rays.astype(np.float32), sinogram.flatten().astype(np.float32), size, 32).get()
    tomo_lib.savebackprojection("output//backprojection_map.png", backproj, size)

    back = backprojection_doubleparallel.backprojection_doubleparallel()
    backproj = back.main(theta_rad.astype(np.float32), rays.astype(np.float32), sinogram.flatten().astype(np.float32), size, 32).get()
    tomo_lib.savebackprojection("output//backprojection_doubleparallel.png", backproj, size)

    #generate you backprojection
    back = backprojection_jh.backprojection_jh()
    backproj = back.main(theta_rad.astype(np.float32), rays.astype(np.float32), sinogram.flatten().astype(np.float32), size, 32).get()
    tomo_lib.savebackprojection("output//backprojection_jh.png", backproj, size)

    # back = backprojection_semiflat.backprojection_semiflat()
    # backproj = back.main(theta_rad.astype(np.float32), rays.astype(np.float32), sinogram.flatten().astype(np.float32), size, 32).get()
    # tomo_lib.savebackprojection("output//backprojection_semiflat.png", backproj, size)



if __name__ == '__main__':
    main(sys.argv)
