import matplotlib.pyplot as plt
import numpy as np
import json

xs = [i*200 for i in range(1, 11)]

with open('../cpu_vs_gpu/results/linecountGPU-fly.json') as f:
  data = json.load(f)

runs  = data['temp.fut']['datasets']
times = [np.average(runs[key]['runtimes'])/1000 for key in runs]
plt.plot(xs, times, zorder=1, label='GPU compute time')  
plt.scatter(xs, times, zorder=2)

with open('./results.txt') as f:
  lines = f.readlines()

ys = [ int(l.split(',')[1]) for l in lines ]
plt.plot(xs, ys, zorder=1, label='Disk load time')
plt.scatter(xs, ys, zorder=2)

plt.title('Loading times of previously saved results')
plt.xlabel('line count')
plt.ylabel('time (ms)')

plt.legend()
# plt.show()
plt.savefig('loadtime.png')