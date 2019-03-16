import sys
import numpy as np
import tomopy
import re
import matplotlib.pyplot as plt

def string_to_array(str):
    #remove leading and trailing brackets
    str = str.strip('[')
    str = str.strip(']')
    #substitute all f32 with nothing
    str = re.sub('f32', '', str)
    #read from string into array now that it's just comma separated numbers
    return np.fromstring( str, dtype=np.float, sep=',' )

data = string_to_array(sys.stdin.read())
size = int(np.sqrt(len(data)))
reshaped = data.reshape((size,size))
#recon = tomopy.circ_mask(reshaped, axis=0, ratio=0.95)
plt.imsave("sirt.png", reshaped, cmap='Greys_r')
