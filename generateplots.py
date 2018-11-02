import pandas as pd
from matplotlib import pyplot
import sys
import matplotlib.pyplot as plt
import numpy as np

def plot_times(filenames, sizes, plotname, stepsize=-1):
    fig, ax = plt.subplots()
    for filenm in filenames:
        timings = get_time(filenm,stepsize)
        plt.plot(sizes[:len(timings)], timings, "-o", label=get_name(filenm))
    ax.set_xlabel('Input')
    ax.set_ylabel('Time [us]')
    ax.legend()
    fig.savefig(plotname)

def get_name(filenm):
    searchstrstart = '\\'
    searchstrend = '_benchmark'
    start = filenm.find(searchstrstart)
    end = filenm.find(searchstrend)
    return filenm[start+1:end]

def get_time(filename, stepsize=-1):
    searchstr = 'us'
    times = []
    with open (filename, 'rt') as in_file:  # Open file  for reading of text data.
        for line in in_file:
            if stepsize == -1 or line.find("_"+str(stepsize)) != -1:
                endoftime = line.find(searchstr)
                if endoftime != -1:
                    times.append(float(line[:endoftime].split()[-1]))
    print(stepsize)
    print(times)
    return times


def main(argv):
    # filesMatrix = ["output\\projectionmatrix_jh_benchmark", "output\\projectionmatrix_map_benchmark", "output\\projectionmatrix_doubleparallel_benchmark"]
    # sizes = [64,128,256,512,1024,2048,4096]
    # plot_times(filesMatrix, sizes, "report\\images\\resultsMatrixPlot.png")

    for stepsize in [32,64,128,256]:
        filesMatrix = ["output\\forwardprojection_jh_benchmark", "output\\forwardprojection_map_benchmark", "output\\forwardprojection_doubleparallel_benchmark", "output\\forwardprojection_semiflat_benchmark", "output\\forwardprojection_dpintegrated_benchmark"]
        sizes = [64,128,256,512,1024,2048,4096]
        plot_times(filesMatrix, sizes, "report\\images\\forwardprojection"+"_"+str(stepsize)+".png", stepsize=stepsize)

if __name__ == '__main__':
    main(sys.argv)
