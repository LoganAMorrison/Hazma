import cython
from libc.math cimport sqrt

cdef inline double kallen_lambda(double a, double b, double c):
    """
    Compute the Kallen-Lambda function (triangle function).
    """
    return a * a + b * b + c * c - 2.0 * a * b - 2.0 * a * c - 2.0 * b * c

@cython.cdivision(True)
cdef inline double two_body_three_momentum(double cme, double m1, double m2):
    """
    Compute the momentum of shared by a two-body final state.
    """
    return sqrt(kallen_lambda(cme * cme, m1 * m1, m2 * m2)) / (2 * cme)

@cython.cdivision(True)
cdef inline double two_body_energy(double q, double m1, double m2):
    """
    Compute the energy of particle 1 in the center-of-mass frame from a process
    of the form X -> 1 + 2, given by:
        E1 = (q^2 + m1^2 - m2^2) / (2 * q)
    where `q` is the center-of-mass energy, `m1` is the mass of particle 1 and
    `m2` is the mass of the second particle.
    
    Parameters
    ----------
    q: double
        Center-of-mass energy
    m1: double
        Mass of particle 1
    m2: double
        Mass of particle 2
    """
    return (q * q + m1 * m1 - m2 * m2) / (2.0 * q)

