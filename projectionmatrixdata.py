import numpy as np
import sys
import math

def get_angles(size,degrees=False):
    num_angles = math.ceil(size*math.pi/2)
    if degrees:
        return np.linspace(0, 180, num_angles, False)
    return np.linspace(0,np.pi, num_angles,False)

def get_rays(size):
    startvalue = (size-1)/2.0
    return np.linspace((-(1.0)*startvalue), startvalue, size)

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

def main(argv):
    size = 256
    print(("Grid has size "+str(size)))
    angles = get_angles(size)
    rays = get_rays(size)

    # f = open("data/matrixf32rad","w+")
    # f.write((print_f32array(angles)+" "+print_f32array(rays)+" "+str(size)))

    # angles = get_angles(size,True)
    # f = open("data/matrixf64deg","w+")
    # f.write((print_f64array(angles)+" "+print_f64array(rays)+" "+str(size)))

    sizes = [128,256]
    for i in sizes:
        filename = "data//matrixinputf32rad"+str(i)
        f = open(filename,"w+")
        angles = get_angles(size)
        rays = get_rays(size)
        f.write((print_f32array(angles)+" "+print_f32array(rays)+" "+str(size)))

if __name__ == '__main__':
    main(sys.argv)
