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
    searchstr = '(avg.'
    times = []
    with open (filename, 'rt') as in_file:  # Open file  for reading of text data.
        for line in in_file:
            if stepsize == -1 or line.find("_"+str(stepsize)) != -1:
                endoftime = line.find(searchstr)
                if endoftime != -1:
                    times.append(float(line[:endoftime-4].split()[-1]))
    print(stepsize)
    print(times)
    return times

def main(argv):
    sizes = [64,128,256,512]
    plot_times(["output\\sirt_benchmark"], sizes, "presentation\\MaP\\images\\sirt_bench.png")
    plot_times(["output\\bp_benchmark"], sizes, "presentation\\MaP\\images\\bp_bench.png")
    sizes = [64,128,256,512,1024,2048]
    plot_times(["output\\fp_benchmark"], sizes, "presentation\\MaP\\images\\fp_bench.png")

if __name__ == '__main__':
    main(sys.argv)
