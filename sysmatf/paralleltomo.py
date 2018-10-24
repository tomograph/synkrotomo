# coding=utf-8
"""
Project: PyCT
File: paralleltomo.py

Description:
Generate projection matrix for parallel tomography, 2D

Author: François Lauze, University of Copenhagen
Date: September 2015 - July 2016

"""

import numpy as np
import scipy.sparse as sp
from PyCTUtils import *




def line_paralleltomo(angles, rays, size, thresh=1e-8, angle_labels=False):
    # type: (tuple, tuple, numpy.ndarray, float) -> scipy.sparse.csr_matrix
    """
    Construct the projection matrix (discrete Radon transform) for
    parallel beam tomography based on intersection of lines and grid.

    Only dealing with square grids.

    :param detector: tuple
        (half extent, number of rays)
    :param grid: tuple
        (half extent, number of segments per side)
    :param angles: list-like
        angles for the different rotations
    :param thresh: float
        positive value used in some equality testings
    :param angle_labels: bool
        if true, a array which length is the number of rows of the matrix
        which is returned, each entry indicates angle / view number for
        the corresponding row.
    :return: scipy.sparse.csr_matrix
        the projection matrix stored in a compressed sparse rows
         matrix.
    """

    # first extract detector parameters, build detector grid
    S = rays

    # extract grid parameters and build grid
    ghe, gn = size/2, size
    x, hg = np.linspace(-ghe, ghe, gn + 1, retstep=True)
    y = np.linspace(-ghe, ghe, gn + 1)
    N = len(x) - 1

    # objects used to build the sparse matrix
    lA = len(angles) * len(S)  # number of rows

    cA = N ** 2  # number of columns

    lnz = 0  # number of nonzero elements
    indptr = [0]  # list used to store matrix nonzeros row entries
    indices = []  # list used to store column indices of nonzero rows
    data = []  # the same but for the actual data

    # row counter, for debugging purposes
    row = -1

    # loop over all angles
    for i in xrange(len(angles)):
        theta = angles[i]
        ct = cosd(theta)
        st = sind(theta)

        # loop over all offsets / rays
        for s in S:
            row += 1
            if i > 30:
                print("Exited because i is greater than 30")
                exit()
            # rays that match top or right boundary of the grid are ignored
            if ((abs(ct) < thresh) and (abs(s - ghe) < thresh)) or \
                    (abs(st) < thresh and (abs(s - ghe) < thresh)):
                indptr.append(lnz)
                continue
            # 0 and 90 degrees lead to division by 0
            with np.errstate(divide='ignore', invalid='ignore'):
                # (s * cos(theta) - x) / sin(theta)
                tx = (s * ct - x) / st
                ytx = tx * ct + s * st
                ty = (y - s * st) / ct
                xty = -ty * st + s * ct



            # collect all the intersection point and then
            # prune to get only points inside the grid. They
            # are sorted by "natural" parametrization of line.
            t = np.hstack((tx, ty))
            nx = np.hstack((x, xty))
            ny = np.hstack((ytx, y))

            perm = np.argsort(t)
            t = t[perm]
            nx = nx[perm]
            ny = ny[perm]


            # comparisons with +/- Inf will cause a warning, ignore it
            with np.errstate(invalid='ignore'):
                inx = np.where(np.logical_and(nx >= x[0], nx <= x[-1]))
                t = t[inx]
                nx = nx[inx]
                ny = ny[inx]
                iny = np.where(np.logical_and(ny >= y[0], ny <= y[-1]))
                t = t[iny]
                nx = nx[iny]
                ny = ny[iny]

            # next, check that some points are not there
            # twice and remove repeated points
            c = np.vstack((nx, ny))
            if c.shape[1] == 0:  # empty intersection
                indptr.append(lnz)
                continue

            # line segments length must be >= thres to ensure that endpoints are
            # properly distinct. Note that I need to add index to the last point
            # of the list to avoid loosing one intersection point

            segs = c[:, 1:] - c[:, :-1]

            norep = list(np.where(np.linalg.norm(segs, axis=0) >= thresh)[0]) + [c.shape[1] - 1]
            c = c[:,norep]
            if c.shape[1] == 0: # empty list of point. I don't believe it can happen...
                indptr.append(lnz)
                continue

            # extract segment lengths and midpoints.
            # midpoints are then used to find
            # pixel/grid-square indices.
            segs = c[:, 1:] - c[:, :-1]
            midpts = 0.5*(c[:, 1:] + c[:, :-1])
            d = np.linalg.norm(segs, axis=0)
            midpts += ghe
            midpts /= hg
            midpts = np.floor(midpts).astype(int)
            indx = midpts[0,:] + N*midpts[1,:]

            perm = np.argsort(indx)
            indx = indx[perm]
            d = d[perm]

            # add them to entries of the sparse matrix
            lnz += len(d)
            indices += list(indx)
            data += list(d)
            indptr.append(lnz)

            #print "At row %d, got:" %row
            #print "     lnz  = ", lnz
            #print "     idx  = ", indx
            #print "     data = ", d

    # "standard" compressed sparse row matrix construction
    data = np.array(data)
    indices = np.array(indices)
    indptr = np.array(indptr)
    #print "Dump matrix content:"
    #print indptr
    #print data
    #print indices

    A = sp.csr_matrix((data, indices, indptr), shape=(int(lA), int(cA)))

    # row angle labels, to make hopefully easy the extraction of the
    # submatrix corresponding to a given angle.
    if angle_labels:
        row_labels = np.zeros(int(lA), dtype=int)
        for i in xrange(len(angles)):
            row_labels[i*int(dn):(i+1)*int(dn)] = i
        return A, row_labels
    else:
        return A

#As = line_paralleltomo((250,500), (250,500), angles=np.arange(0.0, 90.0), thresh=1e-8, angle_labels=False)
#print(As)

# def lpt_complexity(detector, grid, theta):
#     """
#     For each angle in theta, checks how will matrix lines 'overlap'
#     for line parallel tomographic projection matrix. This in particular has implications
#     in the calculation of the pseudo-inverse of each projection submatrix corresponding
#     to a given angle.

#     :param detector: tuple
#         (half extent, number of rays)
#     :param grid: tuple
#      (half extent, number of segments per side)
#     :param theta: numpy array
#         angles for the different rotations, in degrees
#     :return: numpy array with values
#         0: lines are orthogonal
#         int i: current line c overlap with lines up to c+/- i
#     """

#     ghe, gn = grid
#     k = 2.0*ghe/gn
#     dhe, dn = detector
#     h = 2.0*dhe/(dn-1.0)
#     # k*f is the minimum distance for the detector elements that ensure
#     # that two consecutive lines in the projection submatrix at that angle
#     # are orthogonal.
#     f = 1.0/np.max(abs(np.vstack((cosd(theta), sind(theta)))),axis=0)
#     return (np.ceil(k*f/h)-1).astype(int)





















# if __name__ == "__main__":

#     import  matplotlib.pyplot as plt
#     import matplotlib.cm as cm

#     s2 = np.sqrt(2.0)

#     N = 100
#     # we need to set some sizes
#     detector = (14.14, int(N*s2)+1)
#     grid = (10.0, N)
#     angles = np.arange(0, 180.0)
#     """
#     nangles = len(angles)
#     ln = detector[1]

#     X = Shepp_Logan(N)
#     lX = np.reshape(X, X.size)


#     # compute "brut" projection matrix
#     A = line_paralleltomo(detector, grid, angles)
#     lS = A*lX
#     S = np.reshape(lS, [nangles, ln]).T
#     S /= S.max()
#     fig, ax = plt.subplots(1, 2)
#     ax[0].set_title('No mask')
#     ax[0].imshow(S, vmin=0, vmax=1)
#     ax[0].set_xlabel(r'$\theta$', fontsize=24)
#     ax[0].set_ylabel(r'$x$', fontsize=24)

#     # define mask
#     disk_centers = np.array([[5.0, 5.0, -5.0, -5.0],[5.0, -5.0, -5.0, 5.0]])
#     disks = (1.25, disk_centers)

#     # refine projection matrix by removing different types of null rows
#     rA, nz_rows = p3_compress_matrix(A, detector, angles, disks)
#     idx = np.where(nz_rows)

#     # undo in some sense removal of rows
#     lS = rA*lX
#     S = np.zeros(nz_rows.size)
#     S[idx] = lS
#     S /= S.max()
#     S.shape = (nangles, ln)
#     ax[1].set_title("Mask")
#     ax[1].imshow(S.T, vmin=0, vmax=1)
#     ax[1].set_xlabel(r'$\theta$', fontsize=24)
#     ax[1].set_ylabel(r'$x$', fontsize=24)



#     plt.show()
#     """
#     print lpt_complexity(detector, grid, angles)
