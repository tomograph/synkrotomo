import astra
import numpy as np
import pylab
import time
import sys
from futhark import backprojection
from futhark import forwardprojection
from skimage.transform import rotate
from skimage.draw import random_shapes
from matplotlib import pyplot
import pandas as pd
from functools import partial
import timeit
import math
import pickle
import matplotlib.pyplot as plt
plt.switch_backend('agg')

############################################################
#Plotting function timings from https://codereview.stackexchange.com/questions/165245/plot-timings-for-a-range-of-inputs
############################################################
def plot_times(functions, inputs, repeats=3, n_tests=1):
    timings = get_timings(functions, inputs, repeats=3, n_tests=1)
    results = aggregate_results(timings)
    fig, ax = plot_results(results)

    return fig, ax, results

def get_timings(functions, inputs, repeats, n_tests):
    for func in functions:
        result = pd.DataFrame(index = np.array([inputs[i].size for i in range(0,len(inputs))]), columns = range(repeats),
            data=(timeit.Timer(partial(func, i)).repeat(repeat=repeats, number=n_tests) for i in inputs))
        yield func, result

def aggregate_results(timings):
    empty_multiindex = pd.MultiIndex(levels=[[],[]], labels=[[],[]], names=['func', 'result'])
    aggregated_results = pd.DataFrame(columns=empty_multiindex)

    for func, timing in timings:
        for measurement in timing:
            aggregated_results[func.__name__, measurement] = timing[measurement]
        aggregated_results[func.__name__, 'avg'] = timing.mean(axis=1)
        aggregated_results[func.__name__, 'yerr'] = timing.std(axis=1)

    return aggregated_results

def plot_results(results):
    fig, ax = plt.subplots()
    x = results.index
    for func in results.columns.levels[0]:
        y = results[func, 'avg']
        yerr = results[func, 'yerr']
        ax.errorbar(x, y, yerr=yerr, fmt='-o', label=func)

    ax.set_xlabel('Input')
    ax.set_ylabel('Time [s]')
    ax.legend()
    return fig, ax

###############################################################################
#Getting parameters for algorithms
###############################################################################
def get_phantom(size):
    return random_shapes((size, size), min_shapes=5, max_shapes=10, multichannel=False, random_seed=0)[0]

def get_angles(size, degrees=True):
    num_angles = math.ceil(size*math.pi/2)
    if degrees:
        return np.linspace(0, 180, num_angles, False).astype(np.float32)
    else:
        return np.linspace(0,np.pi, num_angles,False)

def get_rays(size):
    startvalue = (size-1)/2.0
    return np.linspace((-(1.0)*startvalue), startvalue, size).astype(np.float32)

def sinogram(image, theta):
    sinogram = np.zeros((len(theta), max(image.shape)))
    for i in range(0, len(theta)):
        rotated_image = rotate(image, theta[i], resize=False)
        sinogram[i] = sum(rotated_image)
    return sinogram.astype(np.float32)

class Config:
    def __init__(self, size):
        angles_rad = get_angles(size, False)
        self.proj_geom =astra.create_proj_geom('parallel', 1.0, size, angles_rad)
        self.vol_geom = astra.create_vol_geom(size)
        self.phantom = get_phantom(size)
        self.angles = get_angles(size)
        self.sinogram = sinogram(self.phantom, self.angles)
        self.rays = get_rays(size)
        self.size = size


###############################################################################
#algorithms
##############################################################################
def astra_BP(cfg):
    # Create projection data
    proj_id = astra.data2d.create('-sino', cfg.proj_geom, cfg.sinogram)

    # Create a data object for the reconstruction
    rec_id = astra.data2d.create('-vol', cfg.vol_geom)

    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict("BP_CUDA")
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    # Get the result
    result = astra.data2d.get(rec_id)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(proj_id)

    return result


def astra_FP(cfg):
    # Create projection data
    proj_id = astra.data2d.create('-sino', cfg.proj_geom, 0)

    # Create a data object for the reconstruction
    volume_id = astra.data2d.create('-vol', cfg.vol_geom, cfg.phantom)

    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict("FP_CUDA")
    cfg['VolumeDataId'] = volume_id
    cfg['ProjectionDataId'] = proj_id
    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    # Get the result
    result = astra.data2d.get(proj_id)

    # Clean up. Note that GPU memory is tied up in the algorithm object,
    # and main RAM in the data objects.
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(volume_id)
    astra.data2d.delete(proj_id)

    return result

def futhark_FP(cfg):
    proj = forwardprojection.forwardprojection()
    return proj.main(cfg.rays, cfg.angles, cfg.phantom.flatten().astype(np.float32))

def futhark_BP(cfg):
    proj = backprojection.backprojection()
    return proj.main(cfg.rays, cfg.angles, cfg.sinogram, cfg.size)

###############################################################################
#Time algorithms, and plot the results
###############################################################################
def main(argv):
    sizes = [128,256]#,512,1024,2048,4096]

    configs = np.array([Config(sizes[i]) for i in range(0, len(sizes))])
    pickle.dump(configs, open("configs.p", "wb"))
    #configs = pickle.load(open("configs.p", "rb"))
    BPs = [astra_BP, futhark_BP]
    FPs = [astra_FP, futhark_FP]

    figBP, axBP, resultsBP = plot_times(BPs, configs)
    pickle.dump(resultsBP, open("resultsBP.p", "wb"))
    figBP.savefig("BPplot.png")
    figFP, axFP, resultsFP = plot_times(FPs, configs)
    pickle.dump(resultsFP, open("resultsFP.p", "wb"))
    figFP.savefig("FPplot.png")



if __name__ == '__main__':
    main(sys.argv)
