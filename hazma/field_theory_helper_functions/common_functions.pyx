"""
Module containing common field theory functions.

@author : Logan Morrison and Adam Coogan
@date : January 2018
"""

import numpy as np
cimport numpy as np
import cython

cdef np.ndarray metric_diag = np.array([1.0, -1.0, -1.0, -1.0])


@cython.boundscheck(False)
@cython.wraparound(False)
def minkowski_dot(np.ndarray[double, ndim=1] fv1,
                  np.ndarray[double, ndim=1] fv2):
    """
    Returns the dot product of two four vectors using the west coast metric.
    """
    return np.sum(metric_diag[:] * fv1[:] * fv2[:])



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cross_section_prefactor(double m1, double m2, double cme):
    """
    """
    cdef double E1 = (cme**2 + m1**2 - m2**2) / (2 * cme)
    cdef double E2 = (cme**2 + m2**2 - m1**2) / (2 * cme)

    cdef double p = (cme**2 - m1**2 - m2**2) / (2 * cme)

    cdef double v1 = p / E1
    cdef double v2 = p / E2

    cdef double vrel = v1 + v2

    return 1.0 / (2.0 * E1) / (2.0 * E2) / vrel
