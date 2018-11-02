import pandas as pd
from matplotlib import pyplot

def get_time(filename):
    searchstr = 'us'
    times = []
    with open (filename, 'rt') as in_file:  # Open file  for reading of text data.
        for line in in_file:
            endoftime = line.find(searchstr)
            if endoftime != -1:
                times.append(float(line[:endoftime].split()[-1]))


def get_timings(functions, sizes, repeats, n_tests):
    for func in functions:
        result = pd.DataFrame(index = sizes), columns = range(repeats),
            data=(get_time(func))
        yield func, result

def plot_results(results):
    fig, ax = plt.subplots()
    x = results.index
    for func in results.columns.levels[0]:
        y = results[func, 'avg']
        ax.errorbar(x, y, fmt='-o', label=func)

    ax.set_xlabel('Input')
    ax.set_ylabel('Time [s]')
    ax.legend()
    return fig, ax

def aggregate_results(timings):
    empty_multiindex = pd.MultiIndex(levels=[[],[]], labels=[[],[]], names=['func', 'result'])
    aggregated_results = pd.DataFrame(columns=empty_multiindex)

    for func, timing in timings:
        for measurement in timing:
            aggregated_results[func, measurement] = timing[measurement]
        aggregated_results[func, 'avg'] = timing.mean(axis=1)

    return aggregated_results

def main(argv):
    filesMatrix = ["output\\projectionmatrix_jh_benchmark", "output\\projectionmatrix_map_benchmark", "output\\projectionmatrix_doubleparallel_benchmark"]
    sizes = [64,128,256,512,1024,2048,4096]
    figMatrix, axMatrix, resultsMatrix= plot_times(filesMatrix, sizes)
    pickle.dump(resultsBP, open("resultsBP.p", "wb"))
    figBP.savefig("BPplot.png")

if __name__ == '__main__':
    main(sys.argv)
