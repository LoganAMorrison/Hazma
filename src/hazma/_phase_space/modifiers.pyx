import numpy as np
cimport numpy as np
import cython
import warnings

ctypedef np.float64_t DBL_T


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DBL_T, ndim=2] split_point(np.ndarray[DBL_T, ndim=1] l,
                                           int num_fsp):
    """
    Returns a list of four momentum from a flattened list.
    """
    cdef int i, j
    cdef np.ndarray[DBL_T, ndim=2] kList

    kList = np.empty((num_fsp, 4), dtype=np.float64)

    for i in range(num_fsp):
        for j in range(4):
            kList[i, j] = l[4 * i + j]
    return kList



@cython.boundscheck(False)
@cython.wraparound(False)
def apply_matrix_elem(np.ndarray[DBL_T, ndim=2] pts, int num_ps_pts,
                      int num_fsp, mat_elem_sqrd):
    """
    Applies the matrix element squared to the weights.
    """
    cdef int i
    cdef double mat_elem2

    for i in range(num_ps_pts):
        mat_elem2 = mat_elem_sqrd(split_point(pts[i], num_fsp))
        if mat_elem2 < 0:
            warnings.warn('Negative matrix element squared encountered...')
        pts[i, 4 * num_fsp] = pts[i, 4 * num_fsp] * mat_elem2
    return pts
