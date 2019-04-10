###################################################################################
### python code for tomographic reconstruction in Futhark.
####
###############################################################################

# import tomopy
# import dxchange
from futhark import SIRT
# from futhark import SIRT3Dslices
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import argparse
import re
import os
import dataread3d_lib
#import astra
# import re
import pyopencl.array as pycl_array

def data_generator2D(f):
    content = dataread3d_lib.clean_str(f.readline())
    angles, rhozero, deltarho, initialimg, sino, iterations = [str for str in content.split(" ")]
    angles = np.fromstring( angles, dtype=np.float32, sep=',' )
    rhozero = float(rhozero)
    deltarho = float(deltarho)
    initialimg = np.fromstring( initialimg, dtype=np.float32, sep=',' )
    sino = np.fromstring( sino, dtype=np.float32, sep=',' )
    iterations = int(iterations)
    return angles, rhozero, deltarho, initialimg, sino, iterations


def poc(inname, numSlices, outname1, outname2):
    f = open(inname, "r")
    theta, rhozero, deltarho, size, iterations = dataread3d_lib.data_generator(f)
    slicelen, sinogram = dataread3d_lib.get_sin_slices(f, numSlices)
    # the sinogram has length numangles * numrhos * size, each slice is numangles * numrhos
    # slicelen = int(len(sinogram)/numSlices)

    # initialize data
    finalimage = np.empty(0, dtype=np.float32)
    initialimg = np.zeros(size*size)

    start = time.time()

    # first slice
    sirt1 = SIRT.SIRT()

    sinoslice = sinogram[:slicelen]
    sinogram = sinogram[slicelen:]
    # sinslice = f.readline()

    # transfer data to first slice
    theta_gpu = pycl_array.to_device(sirt1.queue, theta.astype(np.float32))
    img_gpu = pycl_array.to_device(sirt1.queue, initialimg.astype(np.float32))
    sinogram_gpu = pycl_array.to_device(sirt1.queue, sinoslice.astype(np.float32)) # notice the slice

    # for one less than number of slices, the last is calculated after the loop
    for i in range(size-1):

        sinoslice = sinogram[:slicelen]
        sinogram = sinogram[slicelen:]
        if len(sinogram) == 0:
            s, sinogram = dataread3d_lib.get_sin_slices(f, numSlices)

        # alternate buffers
        if i % 2 == 0:
            # calc even slice
            result1 = sirt1.main(theta_gpu, rhozero, deltarho, img_gpu, sinogram_gpu, iterations)
            if i > 0: # if the uneven slice has been calculated
                # get the result
                finalimage = np.append(finalimage, result2.get())
            # initialize uneven slice
            sirt2 = SIRT.SIRT()
            theta_gpu = pycl_array.to_device(sirt2.queue, theta.astype(np.float32))
            # consider using the result just gotten if initialimg is zero
            img_gpu = pycl_array.to_device(sirt2.queue, initialimg.astype(np.float32))
            sinogram_gpu = pycl_array.to_device(sirt2.queue, sinoslice.astype(np.float32)) # notice slicing for next iteration
        else:
            # calc uneven slice
            result2 = sirt2.main(theta_gpu, rhozero, deltarho, img_gpu, sinogram_gpu, iterations)
            # get even result
            finalimage = np.append(finalimage, result1.get())
            # init even slice
            sirt1 = SIRT.SIRT()
            theta_gpu = pycl_array.to_device(sirt1.queue, theta.astype(np.float32))
            img_gpu = pycl_array.to_device(sirt1.queue, initialimg.astype(np.float32))
            sinogram_gpu = pycl_array.to_device(sirt1.queue, sinoslice.astype(np.float32)) # notice slicing for next iteration

    # do remaining calculation, then get the previous result
    if (size-2) % 2 == 0:
        result = sirt2.main(theta_gpu, rhozero, deltarho, img_gpu, sinogram_gpu, iterations)
        finalimage = np.append(finalimage, result1.get())
    else:
        result = sirt1.main(theta_gpu, rhozero, deltarho, img_gpu, sinogram_gpu, iterations)
        finalimage = np.append(finalimage, result2.get())

    # get the final result
    finalimage = np.append(finalimage, result.get())

    end = time.time()
    print("- runtime for pysliced overlapping execution:\t\t\t{}".format(end-start))
    f.close()
    # Save an image to show it works as expected
    reshaped = finalimage[(int(size/2)-1)*size*size : (int(size/2))*size*size].reshape((size,size))
    plt.imsave(outname1, reshaped, cmap='Greys_r')
    reshaped = finalimage[(int(size/2))*size*size : (int(size/2)+1)*size*size].reshape((size,size))
    plt.imsave(outname2, reshaped, cmap='Greys_r')
    # return finalimage


def benchmark(inname, size):
        f = open(inname, "r")
        theta, rhozero, deltarho, size, iterations = dataread3d_lib.data_generator(f)
        slicelen, sinogram = dataread3d_lib.get_sin_slices(f, numSlices)
        # the sinogram has length numangles * numrhos * size, each slice is numangles * numrhos
        # slicelen = int(len(sinogram)/numSlices)

        # initialize data
        finalimage = np.empty(0, dtype=np.float32)
        initialimg = np.zeros(size*size)

        start = time.time()

        # first slice
        sirt1 = SIRT.SIRT()

        sinoslice = sinogram[:slicelen]
        sinogram = sinogram[slicelen:]
        # sinslice = f.readline()

        # transfer data to first slice
        theta_gpu = pycl_array.to_device(sirt1.queue, theta.astype(np.float32))
        img_gpu = pycl_array.to_device(sirt1.queue, initialimg.astype(np.float32))
        sinogram_gpu = pycl_array.to_device(sirt1.queue, sinoslice.astype(np.float32)) # notice the slice

        # for one less than number of slices, the last is calculated after the loop
        for i in range(size-1):

            sinoslice = sinogram[:slicelen]
            sinogram = sinogram[slicelen:]
            if len(sinogram) == 0:
                s, sinogram = dataread3d_lib.get_sin_slices(f, numSlices)

            # alternate buffers
            if i % 2 == 0:
                # calc even slice
                result1 = sirt1.main(theta_gpu, rhozero, deltarho, img_gpu, sinogram_gpu, iterations)
                if i > 0: # if the uneven slice has been calculated
                    # get the result
                    finalimage = np.append(finalimage, result2.get())
                # initialize uneven slice
                sirt2 = SIRT.SIRT()
                theta_gpu = pycl_array.to_device(sirt2.queue, theta.astype(np.float32))
                # consider using the result just gotten if initialimg is zero
                img_gpu = pycl_array.to_device(sirt2.queue, initialimg.astype(np.float32))
                sinogram_gpu = pycl_array.to_device(sirt2.queue, sinoslice.astype(np.float32)) # notice slicing for next iteration
            else:
                # calc uneven slice
                result2 = sirt2.main(theta_gpu, rhozero, deltarho, img_gpu, sinogram_gpu, iterations)
                # get even result
                finalimage = np.append(finalimage, result1.get())
                # init even slice
                sirt1 = SIRT.SIRT()
                theta_gpu = pycl_array.to_device(sirt1.queue, theta.astype(np.float32))
                img_gpu = pycl_array.to_device(sirt1.queue, initialimg.astype(np.float32))
                sinogram_gpu = pycl_array.to_device(sirt1.queue, sinoslice.astype(np.float32)) # notice slicing for next iteration

        # do remaining calculation, then get the previous result
        if (size-2) % 2 == 0:
            result = sirt2.main(theta_gpu, rhozero, deltarho, img_gpu, sinogram_gpu, iterations)
            finalimage = np.append(finalimage, result1.get())
        else:
            result = sirt1.main(theta_gpu, rhozero, deltarho, img_gpu, sinogram_gpu, iterations)
            finalimage = np.append(finalimage, result2.get())

        # get the final result
        finalimage = np.append(finalimage, result.get())

        end = time.time()
        print("- runtime for pysliced overlapping execution:\t\t\t{}".format(end-start))
        f.close()


def overlap(theta, rhozero, deltarho, initialimg, sino, iterations, runs):
    finalimage = np.empty(0, dtype=np.float32)

    start = time.time()

    # first slice
    sirt1 = SIRT.SIRT()

    # sinslice = f.readline()

    # transfer data to first slice

    ############## Try making the names different to see if speedup
    theta_gpu1 = pycl_array.to_device(sirt1.queue, theta.astype(np.float32))
    img_gpu1 = pycl_array.to_device(sirt1.queue, initialimg.astype(np.float32))
    sinogram_gpu1 = pycl_array.to_device(sirt1.queue, sino.astype(np.float32)) # notice the slice

    # for one less than number of slices, the last is calculated after the loop
    for i in range(runs-1):

        # alternate buffers
        if i % 2 == 0:
            # calc even slice
            result1 = sirt1.main(theta_gpu1, rhozero, deltarho, img_gpu1, sinogram_gpu1, iterations)
            if i > 0: # if the uneven slice has been calculated
                # get the result
                finalimage = result2.get()
            # initialize uneven slice
            sirt2 = SIRT.SIRT()
            theta_gpu2 = pycl_array.to_device(sirt2.queue, theta.astype(np.float32))
            # consider using the result just gotten if initialimg is zero
            img_gpu2 = pycl_array.to_device(sirt2.queue, initialimg.astype(np.float32))
            sinogram_gpu2 = pycl_array.to_device(sirt2.queue, sino.astype(np.float32)) # notice slicing for next iteration
        else:
            # calc uneven slice
            result2 = sirt2.main(theta_gpu2, rhozero, deltarho, img_gpu2, sinogram_gpu2, iterations)
            # get even result
            finalimage =  result1.get()
            # init even slice
            sirt1 = SIRT.SIRT()
            theta_gpu1 = pycl_array.to_device(sirt1.queue, theta.astype(np.float32))
            img_gpu1 = pycl_array.to_device(sirt1.queue, initialimg.astype(np.float32))
            sinogram_gpu1 = pycl_array.to_device(sirt1.queue, sino.astype(np.float32)) # notice slicing for next iteration

    # do remaining calculation, then get the previous result
    if (runs-2) % 2 == 0:
        result = sirt2.main(theta_gpu2, rhozero, deltarho, img_gpu2, sinogram_gpu2, iterations)
        finalimage =  result1.get()
    else:
        result = sirt1.main(theta_gpu1, rhozero, deltarho, img_gpu1, sinogram_gpu1, iterations)
        finalimage =  result2.get()

    # get the final result
    finalimage =  result.get()

    end = time.time()
    print("- runtime for execution:\t\t\t{}".format(end-start))


def noOverlap(theta, rhozero, deltarho, initialimg, sino, iterations, runs):
    start = time.time()

    for i in range(runs):
        sirt = SIRT.SIRT()

        theta_gpu = pycl_array.to_device(sirt.queue, theta.astype(np.float32))
        img_gpu = pycl_array.to_device(sirt.queue, initialimg.astype(np.float32))
        sinogram_gpu = pycl_array.to_device(sirt.queue, sino.astype(np.float32))
        result = sirt.main(theta_gpu, rhozero, deltarho, img_gpu, sinogram_gpu, iterations)
        res = result.get()


    end = time.time()
    print("- runtime for execution:\t\t\t{}".format(end-start))

def proof(indir, outdir, sizes):
    for size in sizes:
        if size == 640:
            name = "toothpoc"
            inname = os.path.join(indir, name)
            outname1 = os.path.join(outdir, name + "1.png")
            outname2 = os.path.join(outdir, name + "2.png")
        else:
            name = "3dpoc" + str(size)
            inname = os.path.join(indir, name)
            outname1 = os.path.join(outdir, name + "1.png")
            outname2 = os.path.join(outdir, name + "2.png")
        print("\n")
        print (name)
        poc(inname, size/2, outname1, outname2)


def over(indir):
    f = open(os.path.join(indir, "sirtinputf32rad128"))
    angles, rhozero, deltarho, initialimg, sino, iterations = data_generator2D(f)
    f.close()
    print ("warmup 10")
    noOverlap(angles, rhozero, deltarho, initialimg, sino, iterations, 10)
    print ("non overlapping 10")
    noOverlap(angles, rhozero, deltarho, initialimg, sino, iterations, 10)
    print ("non overlapping 20")
    noOverlap(angles, rhozero, deltarho, initialimg, sino, iterations, 20)
    print ("overlapping 10")
    overlap(angles, rhozero, deltarho, initialimg, sino, iterations, 10)
    print ("overlapping 20")
    overlap(angles, rhozero, deltarho, initialimg, sino, iterations, 20)



def timing(indir, sizes):
    for size in sizessirt:
        name = "sirt3Dinputf32rad" + str(size)
        inname = os.path.join(indir, name)
        print("\n")
        print (name)
        benchmark(inname, min(size, 256))



def main(argv):
    parser = argparse.ArgumentParser(description="3D SIRT recontruction using data streaming to Futhark, benchmarking and proof of correctness and overlapping")
    parser.add_argument('-id', '--inputdirectory', help="Directory containing input data", default='~/synkrotomo/futhark/data')
    parser.add_argument('-od', '--outputdirectory', help="Directory in which to store the output data", default='~/synkrotomo/output')
    parser.add_argument('-s', '--sizes', nargs='*', help="The sizes to run the recontruction for, 640 is the tooth dataset", type=int, default=128)
    parser.add_argument('-p', '--proof', help="tells the code to perform a reconstruction", action='store_true')
    parser.add_argument('-t', '--time', help="benchmark the sizes, 640, i.e the tooth dataset is not working here", action='store_true')
    parser.add_argument('-o', '--overlap', help="proof that overlapping streaming works,", action='store_true')

    args, unknown = parser.parse_known_args(argv)
    indir = os.path.expanduser(args.inputdirectory)
    outdir = os.path.expanduser(args.outputdirectory)
    sizes = args.sizes
    if args.proof:
        recon(indir, outdir, sizes)
    if args.time:
        timing(indir, sizes)
    if args.overlap:
        over(indir)



if __name__ == '__main__':
    main(sys.argv)
