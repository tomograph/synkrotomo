#Test the system matrix implementations from bachelor projects
from sysmatf import paralleltomo as francois
#from sysmatas.bachelorproject.python.alg import fut_algorithm as anderandsteven
from sysmatjh import intersections
from sysmatjh import intersections_jhversion
from sysmatjh import entrypoints
import scipy.sparse as sp
import numpy as np
import time
import sys

def radonmatrixjh(angles, rays, size):
    #Jakob and Herluf version - parallel per ray
    sysmatjh = intersections.intersections()

    start = time.time()
    A_2 = sysmatjh.main(angles.astype(np.float32), rays.astype(np.float32), size)
    end = time.time()
    elapsed = (end - start)*1000
    print("The elapsed time of the new intersections %d. ms" % (elapsed))

    lnz = 0  # number of nonzero elements
    indptr = [0]  # list used to store matrix nonzeros row entries
    indices = []  # list used to store column indices of nonzero rows
    data = []  # the same but for the actual data
    #run through angles, rays
    for i in range(0, A_2[0].shape[0]):
        #run through rays
        for j in range(0, A_2[0].shape[1]):
            #run through values
            for k in range(0, A_2[0].shape[2]):
                #if non zero value
                if A_2[1][i][j][k] != -1:
                    lnz = lnz +1;
                    indices.append(A_2[1][i][j][k].get())
                    data.append(A_2[0][i][j][k].get())
            indptr.append(lnz)

    # "standard" compressed sparse row matrix construction
    data = np.array(data)
    indices = np.array(indices)
    indptr = np.array(indptr)
    return sp.csr_matrix((data, indices, indptr), shape=(len(rays)*len(angles),size**2))

def test(size, angles):
    #Francois version - naive non-parallel
    #
    #evenly distributed rays through center of each pixel...
    startvalue = (size-1)/2.0
    rays = np.linspace((-(1.0)*startvalue), startvalue, size)

    start = time.time()
    A_1 = francois.line_paralleltomo(angles, rays, size, thresh=1e-8, angle_labels=False)
    end = time.time()
    elapsed1 = (end - start)*1000

    start = time.time()
    A_2 = radonmatrixjh(angles, rays, size)
    end = time.time()
    elapsed2 = (end - start)*1000

    difference = (A_1 - A_2)
    mse = (difference.dot(difference.transpose())).mean(axis=None)

    print("The mse between francois and jh version is %d. Francois version took %d ms, jh version took %d ms" % (mse, elapsed1, elapsed2))

def timeentrypoints(size, angles):
    ep = entrypoints.entrypoints()
    startvalue = (size-1)/2.0
    rays = np.linspace((-(1.0)*startvalue), startvalue, size)
    start = time.time()
    entryps = ep.main(angles.astype(np.float32), rays.astype(np.float32), size)
    end = time.time()
    elapsed = (end - start)*1000

    print("The elapsed time of the entrypoints calculation was %d. ms" % (elapsed))

def timeintersectionsjh():
     a = intersections_jhversion.intersections_jhversion()
     start = time.time()
     A = a.main(2000,1,1000,0,168,12)
     end = time.time()
     elapsed = (end - start)*1000

     print("The elapsed time of the jh original version %d. ms" % (elapsed))

def timeintersections():
    size = 2000
    startvalue = (size-1)/2.0
    rays = np.linspace((-(1.0)*startvalue), startvalue, size)
    a = intersections.intersections()
    start = time.time()
    A = a.main(np.linspace(0,180, 15, endpoint=False).astype(np.float32), rays.astype(np.float32), size)
    end = time.time()
    elapsed = (end - start)*1000

    print("The elapsed time of the new version %d. ms" % (elapsed))


def main(argv):
    #test(4, np.linspace(0,180, 8, endpoint=False))
    #timeentrypoints(512, np.linspace(0,180, 30, endpoint=False))
    timeintersections()


if __name__ == '__main__':
    main(sys.argv)
