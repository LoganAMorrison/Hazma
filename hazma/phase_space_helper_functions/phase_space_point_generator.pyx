"""
Module for generating a relativistic phase space point.

* Author - Logan A. Morrison and Adam Coogan.
* Date - December 2017
"""

import numpy as np
cimport numpy as np
from libc.math cimport log, M_PI, sqrt, tgamma, fabs, pow, cos, sin
from libcpp cimport bool
import cython
import time


cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937()
        mt19937(unsigned int seed)

    cdef cppclass uniform_real_distribution[T]:
        uniform_real_distribution()
        uniform_real_distribution(T a, T b)
        T operator()(mt19937 gen)

cdef mt19937 rng = mt19937(round(time.time()))

cdef uniform_real_distribution[double] uniform \
    = uniform_real_distribution[double](0., 1.)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double __f_xi(double xi, np.ndarray masses, double cme, np.ndarray ps):
    """
    Function whose zero is the scaling factor to correct masses of massless
    four momenta.

    Parameters
    ----------
    xi : double
        Scaling factor.
    masses : numpy.ndarray
        List of masses of the final state particles.
    cme : double
        Center of mass energy of the process.
    ps : numpy.ndarray
        List of the four momentum of the final state particles.

    Returns
    -------
    f(xi) : double
        Value of function evaluated at xi.
    """
    cdef int i
    cdef double val = 0.0
    cdef int num_fsp = len(masses)

    for i in range(num_fsp):
        val = val + sqrt(pow(masses[i], 2.) + pow(xi, 2.) * pow(ps[4 * i], 2.))

    return val - cme

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double __df_xi(double xi, np.ndarray masses, double cme, np.ndarray ps):
    """
    Derivative of the func_xi.

    Parameters
    ----------
    xi : double
        Scaling factor.
    masses : numpy.ndarray
        List of masses of the final state particles.
    cme : double
        Center of mass energy of the process.
    ps : numpy.ndarray
        List of the four momentum of the final state particles.

    Returns
    -------
    df/dxi(xi) : double
        Value of derivative of f(xi) evaluated at xi.
    """
    cdef int i
    cdef double val = 0.0
    cdef int num_fsp = len(masses)

    for i in range(num_fsp):
        denom = sqrt(pow(masses[i], 2.) + pow(xi, 2.) * pow(ps[4 * i], 2.))
        val = val + xi * pow(ps[4 * i], 2.) / denom

    return val


@cython.cdivision(True)
cdef double __find_root(np.ndarray masses, double cme, np.ndarray ps,
                        double tol=10**-4.0, int max_iter=50):
    """
    Function for finding the scaling parameter to turn massless four-vectors
    the correct set of masses.

    Parameters
    ----------
    xi : double
        Scaling factor.
    masses : numpy.ndarray
        List of masses of the final state particles.
    cme : double
        Center of mass energy of the process.
    ps : numpy.ndarray
        List of the four momentum of the final state particles.

    Returns
    -------
    xi : double
        The scaling factor.
    """
    cdef double mass_sum = 0.0
    cdef double xi0, xi1, xi2
    cdef int i, iter_count
    cdef int num_fsp = len(masses)
    cdef bool isDone

    for i in range(num_fsp):
        mass_sum = mass_sum + masses[i]

    xi0 = sqrt(1.0 - (mass_sum / cme)**2)

    isDone = False
    iter_count = 0
    xi2 = xi0
    while isDone is False:
        if iter_count > max_iter:
            break
        xi1 = xi2
        xi2 = xi1 - __f_xi(xi1, masses, cme, ps) / __df_xi(xi1, masses, cme, ps)
        if fabs(xi2-xi1) < tol:
            isDone=True
        iter_count = iter_count + 1

    return xi2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double __get_mass(np.ndarray fv):
    """
    Computes mass of a four-vector.

    Parameters
    ----------
    fv : np.ndarray
        Four-vector to compute mass of.

    Returns
    -------
    mass : double
        Mass of four-vector.
    """
    return sqrt(pow(fv[0], 2.) - \
        pow(fv[1], 2.) - pow(fv[2], 2.) - pow(fv[3], 2.))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[double, ndim=1] __generate_qs(
    np.ndarray[double, ndim=1] masses, double cme):
    """
    Computes isotropic, random four-vectors with energies, q_0, distributed
    according to q_0 * exp(-q_0).

    Parameters
    ----------
    masses : numpy.ndarray
        List of masses of the final state particles.
    cme : double
        Center of mass energy of the process.
    rands : numpy.ndarray
        List of the random numbers uniformly distributed on (0,1).

    Returns
    -------
    qs : np.ndarray
        List of the massless four-momenta.
    """
    cdef int i
    cdef double rho_1, rho_2, rho_3, rho_4
    cdef double c, phi
    cdef double q_e, q_x, q_y, q_z
    cdef int num_fsp = len(masses)

    cdef np.ndarray[double, ndim=1] qs = \
        np.zeros(num_fsp * 4 + 1, dtype=np.float64)

    for i in range(num_fsp):
        rho_1 = uniform(rng)
        rho_2 = uniform(rng)
        rho_3 = uniform(rng)
        rho_4 = uniform(rng)

        c = 2.0 * rho_1 - 1.0
        phi = 2.0 * M_PI * rho_2

        q_e = -log(rho_3 * rho_4)
        q_x = q_e * sqrt(1.0 - pow(c, 2.)) * cos(phi)
        q_y = q_e * sqrt(1.0 - pow(c, 2.)) * sin(phi)
        q_z = q_e * c

        qs[4 * i + 0] = q_e
        qs[4 * i + 1] = q_x
        qs[4 * i + 2] = q_y
        qs[4 * i + 3] = q_z

    return qs


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[double, ndim=1] __generate_ps(
    np.ndarray[double, ndim=1] masses, double cme,
    np.ndarray[double, ndim=1] qs):
    """
    Generates a list of four-momentum with correct center of mass energy from
    isotropic, random four-momentum with energies, q_0,  distributed according
    to q_0 * exp(-q_0).

    Parameters
    ----------
    masses : numpy.ndarray
        List of masses of the final state particles.
    cme : double
        Center of mass energy of the process.
    qs : numpy.ndarray
        List of the random four-momenta.

    Returns
    -------
    ps : np.ndarray
        List of the massless four-momenta with correct center of mass energy.
    """
    cdef int i
    cdef int num_fsp = len(masses)
    cdef double mass_Q
    cdef double b_x, b_y, b_z
    cdef double x, gamma, a
    cdef double qi_e, qi_x, qi_y, qi_z
    cdef double b_dot_qi
    cdef double pi_e, pi_x, pi_y, pi_z

    cdef np.ndarray[double, ndim=1] sum_qs = np.zeros(4, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] ps = \
        np.zeros(4 * num_fsp + 1, dtype=np.float64)

    for i in range(num_fsp):
        sum_qs[0] = sum_qs[0] + qs[4 * i + 0]
        sum_qs[1] = sum_qs[1] + qs[4 * i + 1]
        sum_qs[2] = sum_qs[2] + qs[4 * i + 2]
        sum_qs[3] = sum_qs[3] + qs[4 * i + 3]

    mass_Q = __get_mass(sum_qs)

    b_x = -sum_qs[1] / mass_Q
    b_y = -sum_qs[2] / mass_Q
    b_z = -sum_qs[3] / mass_Q
    x = cme / mass_Q
    gamma = sum_qs[0] / mass_Q
    a = 1.0 / (1.0 + gamma)

    for i in range(num_fsp):
        qi_e = qs[4 * i + 0]
        qi_x = qs[4 * i + 1]
        qi_y = qs[4 * i + 2]
        qi_z = qs[4 * i + 3]

        b_dot_qi = b_x * qi_x + b_y * qi_y + b_z * qi_z

        pi_e = x * (gamma * qi_e + b_dot_qi)
        pi_x = x * (qi_x + b_x * qi_e + a * b_dot_qi * b_x)
        pi_y = x * (qi_y + b_y * qi_e + a * b_dot_qi * b_y)
        pi_z = x * (qi_z + b_z * qi_e + a * b_dot_qi * b_z)

        ps[4 * i + 0] = pi_e
        ps[4 * i + 1] = pi_x
        ps[4 * i + 2] = pi_y
        ps[4 * i + 3] = pi_z


    ps[4 * num_fsp] = (M_PI / 2.0)**(num_fsp - 1.0) \
        * cme**(2.0 * num_fsp - 4.0) \
        / tgamma(num_fsp) \
        / tgamma(num_fsp - 1) \
        * (2.0 * M_PI)**(4.0 - 3.0 * num_fsp)

    return ps


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[double, ndim=1] __generate_ks(
    np.ndarray[double, ndim=1] masses, double cme,
    np.ndarray[double, ndim=1] ps):
    """
    Generates a list of four-momentum with correct masses from massless
    four-momenta.

    Parameters
    ----------
    masses : numpy.ndarray
        List of masses of the final state particles.
    cme : double
        Center of mass energy of the process.
    ps : numpy.ndarray
        List of the massless four-momenta.

    Returns
    -------
    ks : np.ndarray
        List of the massive four-momenta with correct masses.
    """
    cdef int i
    cdef double xi
    cdef double k_e, k_x, k_y, k_z

    cdef double term1 = 0.0
    cdef double term2 = 0.0
    cdef double term3 = 1.0
    cdef double modulus

    cdef int num_fsp = len(masses)
    cdef np.ndarray[double, ndim=1] ks = \
        np.zeros(4 * num_fsp + 1, dtype=np.float64)

    xi = __find_root(masses, cme, ps)

    for i in range(num_fsp):
        k_e = sqrt(masses[i]**2 + xi**2 * ps[4 * i + 0]**2)
        k_x = xi * ps[4 * i + 1]
        k_y = xi * ps[4 * i + 2]
        k_z = xi * ps[4 * i + 3]

        ks[4 * i + 0] = k_e
        ks[4 * i + 1] = k_x
        ks[4 * i + 2] = k_y
        ks[4 * i + 3] = k_z

        modulus = sqrt(k_x**2.0 + k_y**2.0 + k_z**2.0)

        term1 += modulus / cme
        term2 += modulus**2 / ks[4 * i + 0]
        term3 = term3 * modulus / ks[4 * i + 0]

    term1 = term1**(2.0 * num_fsp - 3.0)
    term2 = term2**(-1.0)

    ks[4 * num_fsp] = ps[4 * num_fsp] * term1 * term2 * term3 * cme

    return ks


def generate_point(np.ndarray[double, ndim=1] masses, double cme):
    """
    Generate a single relativistic phase space point.

    Parameters
    ----------
    masses : numpy.ndarray
        List of masses of the final state particles.
    cme : double
        Center of mass energy of the process.
    rands : numpy.ndarray
        List of random numbers.

    Returns
    -------
    phase_space_point : numpy.ndarray
        List of four momenta and a event weight. The returned numpy array is of
        the form {ke1, kx1, ky1, kz1, ..., keN, kxN, kyN, kzN, weight}.
    """
    cdef int num_fsp = len(masses)
    cdef np.ndarray[double, ndim=1] qs = \
        np.zeros(num_fsp * 4 + 1, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] ps = \
        np.zeros(num_fsp * 4 + 1, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] ks = \
        np.zeros(num_fsp * 4 + 1, dtype=np.float64)

    qs = __generate_qs(masses, cme)
    ps = __generate_ps(masses, cme, qs)
    ks = __generate_ks(masses, cme, ps)

    return ks
