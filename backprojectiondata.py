import numpy as np
import sys
import math
import tomo_lib
import dataprint_lib

def main(argv):
    sizes = [64,128,256,512,1024,2048,4096]
    stepsizes = [32,64,128,256]
    for i in sizes:
        for stepsize in stepsizes:
            filename = "data//bpinputf32rad"+str(i)+"_"+str(stepsize)
            f = open(filename,"w+")
            angles = tomo_lib.get_angles(i, False)
            rays = tomo_lib.get_rays(i)
            phantom = tomo_lib.get_phantom(i)
            sino = tomo_lib.sinogram(phantom,tomo_lib.get_angles(i)).flatten()
            f.write(dataprint_lib.print_f32array(angles)+" "+dataprint_lib.print_f32array(rays)+" "+dataprint_lib.print_f32array(sino)+" "+str(i)+" "+str(stepsize))

if __name__ == '__main__':
    main(sys.argv)
