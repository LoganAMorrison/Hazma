"""
Module containing common field theory functions.

@author : Logan Morrison and Adam Coogan
@date : January 2018
"""

import numpy as np
cimport numpy as np
import cython
from libc.math cimport M_PI, sqrt, cos, sin
from libcpp.vector cimport vector


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double c_minkowski_dot(const vector[double] &fv1,const vector[double] &fv2):
    return fv1[0] * fv2[0] - fv1[1] * fv2[1] - fv1[2] * fv2[2] - fv1[3] * fv2[3]


@cython.boundscheck(False)
@cython.wraparound(False)
def minkowski_dot(
    np.ndarray[double, ndim=1] fv1,
    np.ndarray[double, ndim=1] fv2
):
    """
    Returns the dot product of two four vectors using the west coast metric.

    Paramaters
    ----------
    fv1 : numpy.ndarray[double, ndim=1]
        First four vector.
    fv2 : numpy.ndarray[double, ndim=1]
        Second four vector.

    Returns
    -------
    dot_product : double
        Returns fv1 * fv2.
    """
    return c_minkowski_dot(fv1, fv2)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double c_cross_section_prefactor(double m1, double m2, double cme):
    cdef double E1 = (cme**2 + m1**2 - m2**2) / (2. * cme)
    cdef double E2 = (cme**2 + m2**2 - m1**2) / (2. * cme)

    cdef double p = sqrt((m1 - m2 - cme) * (m1 + m2 - cme) *
                         (m1 - m2 + cme) * (m1 + m2 + cme)) / (2. * cme)

    cdef double v1 = p / E1
    cdef double v2 = p / E2

    cdef double vrel = v1 + v2

    return 1.0 / (2.0 * E1) / (2.0 * E2) / vrel


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cross_section_prefactor(double m1, double m2, double cme):
    """
    Returns the prefactor of the 2-N relativistic cross section.

    Paramaters
    ----------
    m1 : double
        Mass of particle one.
    m2 : double
        Mass of particle two.
    cme : double
        Center of mass energy.

    Returns
    -------
    prefactor : double
        Returns 1 / ((2 E1) (2 E2) |v1 - v2|)
    """
    return c_cross_section_prefactor(m1, m2, cme)
