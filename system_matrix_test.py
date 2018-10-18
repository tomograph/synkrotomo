#Test the system matrix implementations from bachelor projects
from sysmatf import paralleltomo as francois
#from sysmatas.bachelorproject.python.alg import fut_algorithm as anderandsteven
from sysmatjh import intersections
import numpy as np

def test(size, angles):
    #Francois version - naive non-parallel
    A_1 = francois.line_paralleltomo((size/2,size), (size/2,size), angles=angles, thresh=1e-8, angle_labels=False)

    #Jakob and Herluf version - parallel per ray
    lib_hook = intersections.intersections()
    A_2 = lib_hook.main(size/2, 1.0, size, angles[0], angles[-1], (angles[-1]-angles[0])/len(angles))

   # A_3 = anderandsteven.line_parallel_grid_intersections((size, size/2), (1, size/2), angles)

    print(A_1.shape)
    print(A_2[0].shape)
    #print(A_3.shape)


test(4, np.linspace(0,180, 8, endpoint=False))
