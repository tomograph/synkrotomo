from futhark import SIRT3D
import numpy as np
import tomo_lib
import dataprint_lib
import sys
import astra
import paralleltomo
from scipy.sparse import csr_matrix

def main(argv):
    size = 128
    theta_rad = tomo_lib.get_angles(size)
    rays = tomo_lib.get_rays(size)
    phantom = tomo_lib.get_phantom3D(size)
    phantom = phantom.flatten().astype(np.float32)
    sinogram = tomo_lib.get_sinogram3D(phantom.reshape(size,size,size), rays,theta_rad)

    sirt = SIRT3D.SIRT3D()
    sirtresult = sirt.main(theta_rad.astype(np.float32), rays.astype(np.float32), np.zeros(size*size*size).flatten().astype(np.float32), sinogram.flatten().astype(np.float32), 200, size).get()
    sirtresult = rescale(sirtresult)
    tomo_lib.save_3D_slices(sirtresult.reshape(size,size,size),"/output", size, 5);


if __name__ == '__main__':
    main(sys.argv)
