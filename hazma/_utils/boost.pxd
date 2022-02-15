import cython
from libc.math cimport sqrt

ctypedef double(*boost_integrand)(double, void*)

@cython.cdivision(True)
cdef inline double boost_gamma(double energy, double mass):
    """
    Compute the gamma boost factor.

    Parameters
    ----------
    energy: double
        Energy of the particle
    mass: double
        Mass of the particle
    """
    return energy / mass

@cython.cdivision(True)
cdef inline double boost_beta(double energy, double mass): 
    """
    Compute the velocity of a particle given its energy and mass.

    Parameters
    ----------
    energy: double
        Energy of the particle
    mass: double
        Mass of the particle
    """
    return sqrt(1.0 - (mass / energy) ** 2)

cdef double boost_jac(double, double, double, double, double)

cdef double boost_eng(double, double, double, double, double)

cdef double boost_delta_function(double, double, double, double)

