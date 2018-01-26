"""
Module for integrating over thee body phase space.

@author : Logan Morrison and Adam Coogan
@data : January 2018
"""

import numpy as np
cimport numpy as np
import cython
from scipy.integrate import quad


@cython.cdivision(True)
def s_to_E1(double s, double m1, double M):
    return (M**2 + m1**2 - s) / (2 * M)


@cython.cdivision(True)
def E1_to_s(E1, m1, M):
    return M**2 - 2 * M * E1 + m1**2


@cython.cdivision(True)
def t_to_E2(double t, double m2, double M):
    return (M**2 + m2**2 - t) / (2 * M)


@cython.cdivision(True)
def u_to_st(double m1, double m2, double m3, double M, double s, double t):
    return M**2 + m1**2 + m2**2 + m3**2 - s - t


@cython.cdivision(True)
def phase_space_prefactor(double M):
    return 1.0 / (16.0 * M**2 * (2 * np.pi)**3)


@cython.cdivision(True)
def t_lim1(double s, double m1, double m2, double m3, double M):
    """
    Returns the value of the first boundary curve evaluated at the mandelstam
    variable s.

    Parameters
    ----------
    s : double
        Mandelstam variable associated with m1.
    m1 : double
        First particle's mass.
    m2 : double
        Second particle's mass.
    m3 : double
        Third particle's mass.
    M : double
        Center of mass energy.

    Returns
    -------
    The first limiting curve of the three-body phase space.
    """
    return -((M - m1) * (M + m1) * (m2 - m3) * (m2 + m3) -
             (M**2 + m1**2 + m2**2 + m3**2) * s + s**2 +
             np.sqrt(M**4 + (m1**2 - s)**2 - 2 * M**2 * (m1**2 + s)) *
             np.sqrt(m2**4 + (m3**2 - s)**2 - 2 * m2**2 * (m3**2 + s))) / \
        (2. * s)


@cython.cdivision(True)
def t_lim2(double s, double m1, double m2, double m3, double M):
    """
    Returns the value of the second boundary curve evaluated at the mandelstam
    variable s.

    Parameters
    ----------
    s : double
        Mandelstam variable associated with m1.
    m1 : double
        First particle's mass.
    m2 : double
        Second particle's mass.
    m3 : double
        Third particle's mass.
    M : double
        Center of mass energy.

    Returns
    -------
    The seconds limiting curve of the three-body phase space.
    """
    return (-((M - m1) * (M + m1) * (m2 - m3) * (m2 + m3)) +
            (M**2 + m1**2 + m2**2 + m3**2) * s - s**2 +
            np.sqrt(M**4 + (m1**2 - s)**2 - 2 * M**2 * (m1**2 + s)) *
            np.sqrt(m2**4 + (m3**2 - s)**2 - 2 * m2**2 * (m3**2 + s))) /\
        (2. * s)


@cython.cdivision(True)
def s_max(double m1, double m2, double m3, double M):
    """
    Returns the maximum value of the mandelstam variable s.

    Parameters
    ----------
    s : double
        Mandelstam variable associated with m1.
    m1 : double
        First particle's mass.
    m2 : double
        Second particle's mass.
    m3 : double
        Third particle's mass.
    M : double
        Center of mass energy.

    Returns
    -------
    The maximum value that s = M - 2 E1*m1 + m1^2 can have.
    """
    return (M - m1)**2


@cython.cdivision(True)
def s_min(double m1, double m2, double m3, double M):
    """
    Returns the minimum value of the mandelstam variable s.

    Parameters
    ----------
    s : double
        Mandelstam variable associated with m1.
    m1 : double
        First particle's mass.
    m2 : double
        Second particle's mass.
    m3 : double
        Third particle's mass.
    M : double
        Center of mass energy.

    Returns
    -------
    The minimum value that s = M - 2 E1*m1 + m1^2 can have.
    """
    return (m2 + m3)**2


@cython.cdivision(True)
def t_integral(double s, double m1, double m2, double m3, double M,
               mat_elem_sqrd):
    """
    Returns the integral over the mandelstam variable t.
    """
    cdef double t_1 = t_lim1(s, m1, m2, m3, M)
    cdef double t_2 = t_lim2(s, m1, m2, m3, M)

    cdef double t_max = 0.0
    cdef double t_min = 0.0

    if t_1 > t_2:
        t_max = t_1
        t_min = t_2
    else:
        t_max = t_2
        t_min = t_1

    return quad(mat_elem_sqrd, t_min, t_max)


@cython.cdivision(True)
def s_integral(double m1, double m2, double m3, double M, mat_elem_sqrd):
    """
    Returns the integral over the mandelstam variable s.
    """
    cdef double smax = s_max(m1, m2, m3, M)
    cdef double smin = s_min(m1, m2, m3, M)

    def integrand(s):
        return t_integral(s, m1, m2, m3, M, mat_elem_sqrd)

    return quad(integrand, s_min, s_max)
