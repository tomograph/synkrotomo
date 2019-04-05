###################################################################################
### Tomopy sample from https://tomopy.readthedocs.io/en/latest/ipynb/astra.html####
###################################################################################
import tomopy
import dxchange
import matplotlib.pyplot as plt
from futhark import SIRT
#from futhark import backprojection
import numpy as np
import scipy
import sys
import argparse
import re
import os
import dataprint_lib
#import astra
# import numpy as np
# import re
import time
import pandas
from datetime import datetime
# import numpy
import pyopencl.array as pycl_array

def string_to_array(str):
    #remove leading and trailing brackets
    str = str.strip('[')
    str = str.strip(']')
    #substitute all f32 with nothing
    str = re.sub('f32', '', str)
    #read from string into array now that it's just comma separated numbers
    return np.fromstring( str, dtype=np.float, sep=',' )

def string_to_array(str):
    #remove leading and trailing brackets
    str = str.strip('[')
    str = str.strip(']')
    #substitute all f32 with nothing
    str = re.sub('f32', '', str)
    #read from string into array now that it's just comma separated numbers
    return np.fromstring( str, dtype=np.float, sep=',' )

def string_to_float(str):
    str = re.sub('f32', '', str)
    str = re.sub('i32', '', str)
    return float(str)

def string_to_int(str):
    str = re.sub('i32', '', str)
    str = re.sub('f32', '', str)
    return int(str)


def data_generator(filename):
    # root = os.path.expanduser(directory)
    # pattern = re.compile(identifier)
    # data = []
    # for filename in os.listdir(root):
    with open(filename) as f:
        content = f.readlines()
        angles, rhozero, deltarho, initialimg, sinogram, iterations = [str for str in content[0].split(" ")]
        angles = string_to_array(angles)
        rhozero = string_to_float(rhozero)
        deltarho = string_to_float(deltarho)
        initialimg = string_to_array(initialimg)
        sinogram = string_to_array(sinogram)
        iterations = string_to_int(iterations)

        return angles, rhozero, deltarho, initialimg, sinogram, iterations

def save_slice(file, slice):
    with open(file, "a") as f:
        f.write(slice)

def sirt(inname, outdir):
    theta, rhozero, deltarho, size, sinogram, iterations = data_generator(inname)
    print("\n")
    print (inname)
    sirt = SIRT.SIRT()

    # size = len(initialimg)**(1/3)

    # ssq = size*size
    # print (len(sinogram)**(1/3))
    # numangs = len(theta)
    # numrhos
    finalimage = np.empty(dtype=np.float32)
    for i in range(size):
        # (1) copy data from host to device
        # start = time.time()
        theta_gpu = pycl_array.to_device(sirt.queue, theta.astype(np.float32))
        img_gpu = pycl_array.to_device(sirt.queue, np.zeros(size*size).flatten().astype(np.float32))
        # img_gpu = pycl_array.to_device(sirt.queue, initialimg[i*ssq;(i+1)*ssq].astype(np.float32))
        sinogram_gpu = pycl_array.to_device(sirt.queue, sinogram[i*r*a:(i+1)*r*a].astype(np.float32))
        # end = time.time()
        # print("- runtime for data transfer (host->device):\t{}".format(end-start))

        # (2) execute kernel
        # start = time.time()
        result = sirt.main(theta_gpu, rhozero, deltarho, img_gpu, sinogram_gpu, iter)
        # end = time.time()
        # print("- runtime for kernel execution:\t\t\t{}".format(end-start))

        # (3) copy data back from device to host
        # start = time.time()
        finalimage = np.append(finalimage, result.get())
        # end = time.time()


def main(argv):
    indir = os.path.expanduser(argv[1])
    outdir = os.path.expanduser(argv[2])

    sirt(indir, outdir)


if __name__ == '__main__':
    main(sys.argv)
