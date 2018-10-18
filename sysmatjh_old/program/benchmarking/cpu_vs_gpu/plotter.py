import matplotlib.pyplot as plt
import numpy as np
import json
import subprocess
import sys

# Open configuration file
with open('benches.json') as f:
  configs = json.load(f)

# Label the axes and title the plot
data_sets = sys.argv[1:]
plt.title(configs[data_sets[0]]['title'])
plt.xlabel(configs[data_sets[0]]['label_x'])
plt.ylabel(configs[data_sets[0]]['label_y'])

# Loop the datasets supplied as arguments
for arg in data_sets:
  config = configs[arg]

  # Read data and plot it
  with open('results/' + arg + '.json') as f:
    data = json.load(f)

  # Get the datapoints and plot them
  runs = data['temp.fut']['datasets']
  ys   = [np.average(runs[key]['runtimes'])/1000 for key in runs]
  plt.plot(config['x_ticks'], ys, zorder=1, label=config['label'])  
  plt.scatter(config['x_ticks'], ys, zorder=2)

# Save the figure
plt.legend()
plt.savefig('figures/' + "_".join(data_sets) + '.png', figsize=(8, 6), dpi=160)