#Test the system matrix implementations from bachelor projects
from sysmatf import paralleltomo as francois
#from sysmatas.bachelorproject.python.alg import fut_algorithm as anderandsteven
from sysmatjh import intersections
import scipy.sparse as sp
import numpy as np

def test(size, angles):
    #Francois version - naive non-parallel
    #
    A_1 = francois.line_paralleltomo((size/2,size), (size/2,size), angles=angles, thresh=1e-8, angle_labels=False)

    #Jakob and Herluf version - parallel per ray
    lib_hook = intersections.intersections()

    #grid_size     = size
    #delta         = 1.0
    #line_count    = size/2
    #scan_start    = angles[0]
    #scan_end      = 180
    #scan_step     = 22.5

    #evenly distributed rays through center of each pixel...
    startvalue = (size-1)/2.0
    rays = np.linspace((-(1.0)*startvalue), startvalue, size)
    print(rays)
    print(angles)
    #A_2 = lib_hook.main(grid_size, delta, line_count, scan_start, scan_end, scan_step)
    A_2 = lib_hook.main(angles.astype(np.float32), rays.astype(np.float32), size)
   # A_3 = anderandsteven.line_parallel_grid_intersections((size, size/2), (1, size/2), angles)

    lnz = 0  # number of nonzero elements
    indptr = [0]  # list used to store matrix nonzeros row entries
    indices = []  # list used to store column indices of nonzero rows
    data = []  # the same but for the actual data
    #run through angles, rays
    for i in range(0, A_2[0].shape[0]):
        #run through indexes within angle
        for j in range(0, A_2[0].shape[1]):
            #run through pixels
            #for k in range(0, A_2[0].shape[2]):
            #if non zero value
            if A_2[0][i][j] != -1:
                lnz = lnz +1;
                indices.append(A_2[1][i][j].get())
                data.append(A_2[0][i][j].get())
        indptr.append(lnz)

    # "standard" compressed sparse row matrix construction
    data = np.array(data)
    indices = np.array(indices)
    indptr = np.array(indptr)
    #A_2 = sp.csr_matrix((data, indices, indptr), shape=(32,16))
    print(A_1)
    print(A_2)
    #print(data)
    #print(indices)
    #print(indptr)

    #A_2 = sp.csr_matrix((data, indices, indptr), shape=(32,16))

    #print(A_1-A_2)


test(4, np.linspace(0,180, 8, endpoint=False))
