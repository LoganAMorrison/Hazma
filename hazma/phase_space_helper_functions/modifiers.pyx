import numpy as np
cimport numpy as np
import cython
import warnings


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def normalize_weights(np.ndarray pts, int num_ps_pts, int num_fsp):
    """
    Sums all the events weights and normalizes the each weight.
    """
    cdef int i
    cdef double tot_weight = 0.0

    for i in range(num_ps_pts):
        tot_weight += pts[i, 4 * num_fsp]

    for i in range(num_ps_pts):
        pts[i, 4 * num_fsp] = pts[i, 4 * num_fsp] / tot_weight

    return pts


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray split_point(np.ndarray l, int num_fsp):
    """
    Returns a list of four momentum from a flattened list.
    """
    cdef int i, j
    kList = np.empty((num_fsp, 4), dtype=np.float64)
    for i in range(num_fsp):
        for j in range(4):
            kList[i, j] = l[4 * i + j]
    return kList


@cython.boundscheck(False)
@cython.wraparound(False)
def apply_matrix_elem(np.ndarray pts, int num_ps_pts, int num_fsp,
                      mat_elem_sqrd=lambda klist: 1.0):
    """
    Applies the matrix element squared to the weights.
    """
    cdef int i

    for i in range(num_ps_pts):
        mat_elem2 = mat_elem_sqrd(split_point(pts[i], num_fsp))
        if mat_elem2 < 0:
            warnings.warn('Negative matrix element squared encountered...')
        pts[i, 4 * num_fsp] = pts[i, 4 * num_fsp] * mat_elem2
    return pts
