# coding=utf-8
"""
Project: PyCT
File: PyCTUtils.py

Description:
A few utility routines for PyCT

Author: François Lauze, University of Copenhagen
Date: September 2015 - July 2016

"""


import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
#from matplotlib import cm


def cosd(x):
    """
    Returns the cos of x in degrees.
    :param x: float
        angle in degrees
    :return:
        cosine of x
    """
    return np.cos(np.pi*x/180.0)


def sind(x):
    """
    Returns the sin of x in degrees.
    :param x: float
        angle in degrees
    :return:
        sine of x
    """
    return np.sin(np.pi*x/180.0)


def delete_rows_csr(A, nz_rows):
    # type: (scipy.sparse.csr_matrix, list) -> scipy.sparse.csr_matrix
    """
    Delete rows from a sparse matrix. Especially useful for Henning's
    missing data problem.

    :param A: compressed sparse row matrix
    :param nz_rows: boolean list of rows, True: keep, False: remove
    :return: A new, smaller csr matrix
    """
    keep_rows = np.array(nz_rows, dtype=bool)
    return A[keep_rows]


def non_empty_rows(A):
    # type: scipy.sparse.csr_matrix -> numpy.ndarray
    """
    returns a boolean mask for non empty / empty rows
    :param A: sparse matrix
    :return: boolean array, true indicates that row with
        corresponding index is non empty
    """
    d = np.diff(A.indptr).astype(int)
    idx = np.where(d == 0)
    mask = np.ones(A.shape[0], dtype = bool)
    mask[idx] = False
    return mask

def non_zero_rows(A, thres=1e-8):
    # type: (scipy.sparse.csr_matrix, float) -> numpy.ndarray
    """
    Returns a mask indicating whether a given line is null up to a threshold
    :param A: sparse matrix
    :param thres: float
    :return: boolean array, true indicates that row with
        corresponding index is non zero
    """
    m = A.shape[0]
    mask = np.ones(m, dtype=bool)
    data = A.data
    indptr = A.indptr
    for i in xrange(m):
        if (data[indptr[i]:indptr[i+1]]**2).sum() < thres:
            mask[i] = False
    return mask


def remove_empty_rows_from_csr(A):
    # type: scipy.sparse.csr_matrix -> (scipy.sparse.csr_matrix, numpy.ndarray)
    """
    remove empty rows of the matrix, creating a smaller one.
    :param A: compressed sparse rows matrix to be compressed (...)
    :return: (B,mask): B = the smaller matrix with empty rows removed
        and mask = the boolean array indicating which rows were kept
    """
    mask = non_empty_rows(A)
    return A[mask], mask



def olines_pinv(A):
    """
    Computes pseudoinverse of a Matrix A (n,m), m > n
    with non-zeros orthogonal lines (thus onto)
    :param A: sparse matrix to be inverted
    :return: pseudo-inverse of A
    """
    m,n = A.shape
    # A*A.T is diagonal
    B = A.T.copy()
    for i in xrange(n):
        d = np.sum(np.asarray(A[i,:].todense())**2)
        B[:,i] /= d
    return B


def ocols_pinv(A):
    """
    Computes pseudoinverse of a matrix A (n,m), m < n
    with non-zero orthogonal columns (thus into)
    :param A: sparse matrix to be inverted
    :return: pseudoinverse of A
    """
    m,n = A.shape
    # this time A.T*A is diagonal
    B = A.T.copy()
    for i in xrange(m):
        d = np.sum(np.asarray(A[:,i].todense())**2)
        B[i,:] /= d
    return B




def thomas(a, b, c, d):
    """
    Thomas tridiagonal solver for linear system
    Ax = d
    :param a: numpy array
        lower diagonal of A
    :param b: numpy array:
        main diagonal of A
    :param c: numpy array
        upper diagonal of A
    :param d:
        second member
    :return:
        x solving Ax = d. This is done by modifying d in-place
        and the modified d is returned (not really necessary?)
    """
    # Forward step
    n = len(b)
    c[0] /= b[0]
    d[0] /= b[0]

    for i in range(1,n-1):
    #    if i == n-1:
    #       print "bug"
        beta = b[i] - a[i-1]*c[i-1]
        c[i] /= beta
        d[i] = (d[i] - a[i-1]*d[i-1])/beta
    beta = b[-1] - a[-1]*c[-1]
    d[-1] = (d[-1] - a[-1]*d[-2])/beta

    # backward step
    for i in xrange(n-2,-1,-1):
        d[i] -= c[i]*d[i+1]

    return d





def solve_pinv(A,b, complexity, damping = 0.0):
    """
    Computes pinv(A)*b assuming that A has independent lines
    and that A.A^T is diagonal (complexity=0), tridiagonal (complexity=1),
    or more (complexity > 1). It is assumed moreover that the diagonal
    elements of A.A^T are > 0

    :param A: Matrix of system
    :param b: Second member
    :param complexity: complexity of A*A^T
    :damping: diagonal value, corresponding to a Tichonov
        regularization weight of the problem.
    :return: least-square solution of Ax = b
    """
    m, n = A.shape
    if complexity == 0:
        # A.A^T is diagonal,
        for i in xrange(m):
            d = np.sum(np.asarray(A[i,:].todense())**2) + damping
            b[i,:] /= d
        return A.T*b
    elif complexity == 1:
        # A.A^T is tridiagonal
        # lower, main and upper diagonals
        lod = np.zeros(m-1)
        mnd = np.zeros(m)
        upd = np.zeros(m-1)

        for i in xrange(m-1):
            d = np.sum(np.asarray(A[i, :].todense()) ** 2)
            e = np.sum(np.asarray(A[i, :].todense())*np.asarray(A[i+1, :].todense()))
            lod[i] = e
            mnd[i] = d
            upd[i] = e
        mnd[m-1] = np.sum(np.asarray(A[m-1, :].todense()) ** 2)
        mnd += damping

        y = thomas(lod, mnd, upd, b)
        return A.T*y
    else:
        # generic least-squares solver
        return spla.lsqr(A, b)




def detector_mask(N, d, dn, fg=1.0, bg = 0.5, bool_res=False):
    """
    Returns a disk map for a NxN image illustrating
    what the detector sees.
    :param: N: in
        image array side size
    :param: d: float
        grid half extent in some distance units.
    :param dn: float
        detector half extent in same distance units.
    :param; fg: float
        when bool_res == False, value of the mask foreground
    :param; bg: float
        when bool_res == False, value of the mask background
    :param: bool_res: bool
        if true, no float conversion, returns the boolean mask
    :return: disk mask of the region covered by the detector,
    """
    x, y = np.mgrid[0:N, 0:N]
    r = N*0.5*dn/d
    c = (N-1.0)/2.0
    z = (x-c)**2 + (y-c)**2 <= r**2
    if not bool_res:
        z = z.astype(float)
        z = (fg - bg)*z + bg
    return z



def plot_sinogram(s, ax=None, cmap=None):
    """
    Plot a sinogram, with angles on the horizontal axis and
    detector measurements on the vertical axis.
    Parameters:
    -----------
    ax: axes object
        where the sinogram will be show
    s: numpy array
        2D array representing the sinogram. columns: detector,
        lines: angles.
    Returns:
    --------
        None

    """
    if ax == None:
        fig = plt.figure()
        ax = fig.gca()

    dN, lAngles = s.shape
    #print s.shape
    r = lAngles/4
    angleticks = [0, r, 2*r, 3*r, lAngles-1]
    anglelabels = [r'0',r'$\pi/4$',r'$\pi/2$',r'$3\pi/4$', r'$\pi$']
    xticks = [0, dN/2, dN-1]
    xlabels = [str((dN/2)), '0', str(-(dN/2))]
    ax.imshow(s, aspect=lAngles*1.0/dN, cmap=cmap)
    ax.xaxis.set_ticks(angleticks)
    ax.xaxis.set_ticklabels(anglelabels)
    ax.yaxis.set_ticks(xticks)
    ax.yaxis.set_ticklabels(xlabels)




##########################################
# The classical phantom, kind of modified?

def Shepp_Logan(n):
    """
    Create a (modified) Shepp-Logan phantom.
    Code adapted from DTU AIRTools.

    :param n: Image size
    :return: a (N,N) array / image.
    """
    #       A      a      b    x0      y0    phi
    #    --------------------------------------
    e = [[  1,   .69,   .92,    0,      0,   0],
         [-.8, .6624, .8740,    0, -.0184,   0],
         [-.2, .1100, .3100,  .22,      0, -18],
         [-.2, .1600, .4100, -.22,      0,  18],
         [ .1, .2100, .2500,    0,    .35,   0],
         [ .1, .0460, .0460,    0,     .1,   0],
         [ .1, .0460, .0460,    0,    -.1,   0],
         [ .1, .0460, .0230, -.08,  -.605,   0],
         [ .1, .0230, .0230,    0,  -.606,   0],
         [ .1, .0230, .0460,  .06,  -.605,   0]]
    e = np.array(e)
    xn = np.linspace(-1.0, 1.0, num=n)
    Xn = np.tile(xn, (n, 1))
    Yn = np.rot90(Xn)
    X = np.zeros((n, n))

    for i in range(e.shape[0]):
        A, a2, b2, x0, y0, phi = e[i,:]
        a2 *= a2
        b2 *= b2
        x = Xn - x0
        y = Yn - y0
        idx = np.where(((x*cosd(phi) + y*sind(phi))**2)/a2 + ((y*cosd(phi) - x*sind(phi))**2)/b2 <= 1)
        X[idx] += X[idx] + A

    np.place(X, X < 0.0, 0.0)
    return X




def compresslist(blist, keep):
    """
    Compress a block list, by removing indices
    marked as False in keep, and renumber the indices so
    that the match the amount of True entries in keep.

    Indices start at 0, last index value is blist[-1]-1,
    i.e., indices should be all integers from 0 to blist[-1]-1.
    This must be conserved after compression.
    Each block is represented by the sublist blist[i]:blist[i+1].


    I need to explain better what I do, for my own sake!

    Arguments:
    ----------
    blist: list of index endpoints that mark a block
    keep: boolean array ov values that mark whether
        an index is kept or not.

    Returns:
    --------
    cblist: compressed list of blocks, with updated endpoints
    kept_block: a list of the kept blocks
    """

    # Number of blocks
    N = len(blist) - 1
    # number of indices
    n = blist[-1]
    if len(keep) != n:
        raise ValueError('dimensions mismatch.')

    kept_block = []
    cblist = [0]
    newstart = 0
    for i in range(N):
        block_keep = keep[blist[i]:blist[i+1]]
        kept = reduce(lambda x,y:x+y, block_keep, 0)
        if kept == 0:
            continue
        newstart += kept
        cblist.append(newstart)
        kept_block.append(i)

    return cblist, kept_block



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    X = Shepp_Logan(400)
    plt.imshow(X)
    plt.show()
