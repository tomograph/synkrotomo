import pandas as pd
from matplotlib import pyplot

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
