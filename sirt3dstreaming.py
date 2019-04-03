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


def tooth(outdir, iter):

    root = os.path.expanduser("~/synkrotomo/futhark/data")

    inname = os.path.join(root, "tooth.in")
    theta, rhozero, deltarho, initialimg, sinogram, iterations = data_generator(inname)
    print("\n")
    print ("tooth")
    sirt = SIRT.SIRT()
    # (1) copy data from host to device
    start = time.time()

    size = 640

    theta_gpu = pycl_array.to_device(sirt.queue, theta.astype(np.float32))
    img_gpu = pycl_array.to_device(sirt.queue, initialimg.astype(np.float32))
    sinogram_gpu = pycl_array.to_device(sirt.queue, sinogram.astype(np.float32))
    end = time.time()
    print("- runtime for data transfer (host->device):\t{}".format(end-start))

    # (2) execute kernel
    start = time.time()
    result = sirt.main(theta_gpu, rhozero, deltarho, img_gpu, sinogram_gpu, iter)
    end = time.time()
    print("- runtime for kernel execution:\t\t\t{}".format(end-start))

    outname = "sirt_pyopencl.png"
    # (3) copy data back from device to host
    start = time.time()
    res = result.get()
    end = time.time()
    print("- runtime for data transfer (device->host):\t{}".format(end-start))

    # reshaped = res.reshape((size,size))
    # plt.imsave(os.path.join(outdir, outname), reshaped, cmap='Greys_r')

def time_sirt(indir, outdir, iter):
    sizessirt = [64,128,256,512,1024,1500,2000,2048,2500,3000,3500,4000,4096]
    # ["sirtinputf32rad64", "sirtinputf32rad128", "sirtinputf32rad256", "sirtinputf32rad512", "sirtinputf32rad1024", "sirtinputf32rad1500", "sirtinputf32rad2000", "sirtinputf32rad2048", "sirtinputf32rad2500", "sirtinputf32rad3000", "sirtinputf32rad3500", "sirtinputf32rad4000", "sirtinputf32rad4096"]

    for size in sizessirt:
        name = "sirtinputf32rad" + str(size)
        inname = os.path.join(indir, name)
        theta, rhozero, deltarho, initialimg, sinogram, iterations = data_generator(inname)
        print("\n")
        print (name)
        sirt = SIRT.SIRT()
        
        # (1) copy data from host to device
        start = time.time()


        theta_gpu = pycl_array.to_device(sirt.queue, theta.astype(np.float32))
        img_gpu = pycl_array.to_device(sirt.queue, initialimg.astype(np.float32))
        sinogram_gpu = pycl_array.to_device(sirt.queue, sinogram.astype(np.float32))
        end = time.time()
        print("- runtime for data transfer (host->device):\t{}".format(end-start))

        # (2) execute kernel
        start = time.time()
        result = sirt.main(theta_gpu, rhozero, deltarho, img_gpu, sinogram_gpu, iter)
        end = time.time()
        print("- runtime for kernel execution:\t\t\t{}".format(end-start))

        # (3) copy data back from device to host
        start = time.time()
        res = result.get()
        end = time.time()
        # print("- runtime for data transfer (device->host):\t{}".format(end-start))

        # reshaped = res.reshape((size,size))
        # plt.imsave(os.path.join(outdir, outname), reshaped, cmap='Greys_r')

def main(argv):
    indir = os.path.expanduser(argv[1])
    outdir = os.path.expanduser(argv[2])

    print ("iter 1")
    tooth(outdir, 1)
    time_sirt(indir, outdir, 1)
    print ("iter 200")
    tooth(outdir, 200)
    time_sirt(indir, outdir, 200)


if __name__ == '__main__':
    main(sys.argv)
