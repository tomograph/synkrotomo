from futhark import SIRT
from futhark import backprojection_test
from futhark import backprojection
from futhark import forwardprojection
import numpy as np
import tomo_lib
import dataprint_lib
import sys
from scipy.sparse import csr_matrix
import scipy.io
import scipy.misc
from scipy import ndimage

def rescale(values):
    minimum = np.min(values)
    maximum = np.max(values)
    return (values - minimum)/(maximum-minimum)

def savebackprojection(filename, data, size):
    reshaped = np.flip(np.flip(data,0).reshape((size,size)),1)
    scipy.misc.toimage(reshaped).save(filename)

def main(argv):
    size = 256
    theta_rad = tomo_lib.get_angles(size)
    rays = tomo_lib.get_rays(size)
    phantom = tomo_lib.get_phantom(size)
    phantom = rescale(phantom.flatten().astype(np.float32))
    sinogram = rescale(tomo_lib.get_sinogram(phantom.reshape(size,size), rays,theta_rad))

    forward = forwardprojection.forwardprojection()
    fpresult = forward.main(theta_rad.astype(np.float32), rays.astype(np.float32), phantom).get()
    fpresult = rescale(fpresult)
    tomo_lib.savesinogram("output//forwardprojection.png",fpresult, len(rays), len(theta_rad))

    back = backprojection_test.backprojection_test()
    bpresult = back.main(theta_rad.astype(np.float32), rays.astype(np.float32), size, sinogram.flatten().astype(np.float32)).get()
    bpresult = rescale(bpresult)
    savebackprojection("output//backprojectiontest.png",bpresult, size)

    back = backprojection.backprojection()
    bpresult = back.main(theta_rad.astype(np.float32), rays.astype(np.float32), size, sinogram.flatten().astype(np.float32)).get()
    bpresult = rescale(bpresult)
    savebackprojection("output//backprojection.png",bpresult, size)

if __name__ == '__main__':
    main(sys.argv)
