import numpy as np
import sys
import math
import tomo_lib
import dataprint_lib

def main(argv):
    sizes = [64,128,256,512,1024,2048,4096]
    iterations = 200
    for i in sizes:
        filename = "data//sirtinputf32rad"+str(i)
        f = open(filename,"w+")
        angles = tomo_lib.get_angles(i, False)
        rays = tomo_lib.get_rays(i)
        initialimg = np.zeros(i*i)
        phantom = tomo_lib.get_phantom(i)
        sino = tomo_lib.sinogram(phantom,tomo_lib.get_angles(i, True)).flatten()
        f.write(dataprint_lib.print_f32array(angles)+" "+dataprint_lib.print_f32array(rays)+" "+dataprint_lib.print_f32matrix(initialimg)+" "+dataprint_lib.print_f32array(sino)+" "+str(iterations))

if __name__ == '__main__':
    main(sys.argv)
