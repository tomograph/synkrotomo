from numpy import *
import math
import matplotlib.pyplot as plt

r = linspace(10000000000,100000000000,100)
naive =( 2048*2048*3216*2048)/r+1
incremental = (2048*3216*2048)/r + 2048

plt.plot(r,naive,'r')
plt.plot(r,incremental,'g')
plt.show()
