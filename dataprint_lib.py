import math
from skimage.transform import rotate
from skimage.draw import random_shapes
import numpy as np
import scipy.io
import scipy.misc

def print_f32array(arr):
    result="["
    for i in arr:
        result+=str(i)+"f32,"
    result = result[:-1]
    return result+"]"

def print_f64array(arr):
    result="["
    for i in arr:
        result+=str(i)+"f64,"
    result = result[:-1]
    return result+"]"

def print_f32matrix(mat):
    result="["
    for r in mat:
        result+="["
        for c in r:
            result+=str(c)+"f32,"
        result = result[:-1]
        result+="],"
    result = result[:-1]
    return result+"]"
