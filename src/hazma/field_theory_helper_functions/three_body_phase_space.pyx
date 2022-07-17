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
    """
    Returns the energy of particle 1 using the mandelstam variable s.

    Parameters
    ----------
    s : double
        Mandelstam variable associated with m1, defined as (P-p1)^2, where P = p1 + p2 + p3.
    m1 : double
        Mass of particle 1.
    M : double
        Center of mass energy, or sqrt((p1 + p2 + p3)^2).

    Returns
    -------
    E1 : double
        Energy of particle 1.
    """
    return (M**2 + m1**2 - s) / (2 * M)


@cython.cdivision(True)
def E1_to_s(E1, m1, M):
    """
    Returns the mandelstam variable s of particle 1 using the energy.

    Parameters
    ----------
    E! : double
        Energy of particle 1.
    m1 : double
        Mass of particle 1.
    M : double
        Center of mass energy, or sqrt((p1 + p2 + p3)^2).

    Returns
    -------
    s : double
        Mandelstam variable associated with m1, defined as (P-p1)^2, where P = p1 + p2 + p3.
    """
    return M**2 - 2 * M * E1 + m1**2


@cython.cdivision(True)
def t_to_E2(double t, double m2, double M):
    """
    Returns the energy of particle 2 using the mandelstam variable t.

    Parameters
    ----------
    t : double
        Mandelstam variable associated with m2, defined as (P-p2)^2, where P = p1 + p2 + p3.
    m2 : double
        Mass of particle 2.
    M : double
        Center of mass energy, or sqrt((p1 + p2 + p3)^2).

    Returns
    -------
    E2 : double
        Energy of particle 2.
    """
    return (M**2 + m2**2 - t) / (2 * M)


@cython.cdivision(True)
def st_to_u(double m1, double m2, double m3, double M, double s, double t):
    """
    Returns the value of the mandelstam variable u using the mandelstam variables s and t.

    Parameters
    ----------
    m1 : double
        Mass of particle 1.
    m2 : double
        Mass of particle 2.
    m3 : double
        Mass of particle 3.
    M : double
        Center of mass energy, or sqrt((p1 + p2 + p3)^2).
    s : double
        Mandelstam variable associated with m1, defined as (P-p1)^2, where P = p1 + p2 + p3.
    t : double
        Mandelstam variable associated with m2, defined as (P-p2)^2, where P = p1 + p2 + p3.

    Returns
    -------
    u : double
        Mandelstam variable associated with m3, defined as (P-p3)^2, where P = p1 + p2 + p3.
    """
    return M**2 + m1**2 + m2**2 + m3**2 - s - t


@cython.cdivision(True)
def phase_space_prefactor(double M):
    r"""
    Returns the normalization of the differential phase space volume is terms
    of s and t

    Parameters
    ----------
    M : double
        Center of mass energy, or sqrt((p1 + p2 + p3)^2).

    Returns
    -------
    prefactor : double
        The prefactor of ds*dt. That is, the factor, :math:`C` that equates
        :math:`d\phi_3 = C dt st`

    """
    return 1.0 / (16.0 * M**2 * (2. * np.pi)**3)


@cython.cdivision(True)
def t_lim1(double s, double m1, double m2, double m3, double M):
    """
    Returns the value of the first boundary curve evaluated at the mandelstam
    variable s.

    Parameters
    ----------
    s : double
        Mandelstam variable associated with m1, defined as (P-p1)^2, where P = p1 + p2 + p3.
    m1 : double
        Mass of particle 1.
    m2 : double
        Mass of particle 2.
    m3 : double
        Mass of particle 3.
    M : double
        Center of mass energy, or sqrt((p1 + p2 + p3)^2).

    Returns
    -------
    The first limiting curve of the three-body phase space.
    """
    return -((M - m1) * (M + m1) * (m2 - m3) * (m2 + m3) -
             (M**2 + m1**2 + m2**2 + m3**2) * s + s**2 +
             np.sqrt(M**4 + (m1**2 - s)**2 - 2 * M**2 *
                  (m1**2 + s)) * np.sqrt(m2**4 + (m3**2 - s)**2 - 2 * m2**2 *
                                      (m3**2 + s))) / (2. * s)


@cython.cdivision(True)
def t_lim2(double s, double m1, double m2, double m3, double M):
    """
    Returns the value of the second boundary curve evaluated at the mandelstam
    variable s.

    Parameters
    ----------
    s : double
        Mandelstam variable associated with m1, defined as (P-p1)^2, where P = p1 + p2 + p3.
    m1 : double
        Mass of particle 1.
    m2 : double
        Mass of particle 2.
    m3 : double
        Mass of particle 3.
    M : double
        Center of mass energy, or sqrt((p1 + p2 + p3)^2).

    Returns
    -------
    The seconds limiting curve of the three-body phase space.
    """
    return ((-M + m1) * (M + m1) * (m2 - m3) * (m2 + m3) +
            (M**2 + m1**2 + m2**2 + m3**2) * s - s**2 +
            np.sqrt(M**4 + (m1**2 - s)**2 - 2 * M**2 * (m1**2 + s)) * np.sqrt(
                m2**4 + (m3**2 - s)**2 - 2 * m2**2 * (m3**2 + s))) / (2. * s)


@cython.cdivision(True)
def s_max(double m1, double m2, double m3, double M):
    """
    Returns the maximum value of the mandelstam variable s.

    Parameters
    ----------
    m1 : double
        Mass of particle 1.
    m2 : double
        Mass of particle 2.
    m3 : double
        Mass of particle 3.
    M : double
        Center of mass energy, or sqrt((p1 + p2 + p3)^2).

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
    m1 : double
        Mass of particle 1.
    m2 : double
        Mass of particle 2.
    m3 : double
        Mass of particle 3.
    M : double
        Center of mass energy, or sqrt((p1 + p2 + p3)^2).

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

    Parameters
    ----------
    s : double
        Mandelstam variable associated with m1, defined as (P-p1)^2, where P = p1 + p2 + p3.
    m1 : double
        Mass of particle 1.
    m2 : double
        Mass of particle 2.
    m3 : double
        Mass of particle 3.
    M : double
        Center of mass energy, or sqrt((p1 + p2 + p3)^2).
    mat_elem_sqrd : double(*func)(double t)
        Function for the matrix element squared. Argument is the mandelstam variable t associated with m2, defined as (P-p2)^2, where P = p1 + p2 + p3.

    Returns
    -------
    t_int : double
        Value of the integral over mandelstam varible t given the mandelstam varible s.
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
def three_body_phase_space_integral(double m1, double m2, double m3, double M,
                                    mat_elem_sqrd):
    """
    Returns the integral over the mandelstam variable s.

    Parameters
    ----------
    m1 : double
        Mass of particle 1.
    m2 : double
        Mass of particle 2.
    m3 : double
        Mass of particle 3.
    M : double
        Center of mass energy, or sqrt((p1 + p2 + p3)^2).
    mat_elem_sqrd : double(*func)(double s, double t)
        Function for the matrix element squared. Argument is the mandelstam variable s associated with m1, defined as (P-p1)^2, where P = p1 + p2 + p3.

    Returns
    -------
    t_int : double
        Value of the integral over mandelstam varible s.
    """
    cdef double smax = s_max(m1, m2, m3, M)
    cdef double smin = s_min(m1, m2, m3, M)

    def integrand(s):
        """
        Integrand for s integral
        """
        def mat_elem_sqrd_t(t):
            """
            Integrand for t integral
            """
            return mat_elem_sqrd(s, t)

        return t_integral(s, m1, m2, m3, M, mat_elem_sqrd_t)[0]

    return quad(integrand, smin, smax)
