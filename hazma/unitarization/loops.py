import numpy as np

from .parameters import fpi
from cmath import sqrt, log, pi

from .parameters import pion_mass_chiral_limit as mPI
from .parameters import kaon_mass_chiral_limit as mK


MPI = complex(mPI)
MK = complex(mK)
FPI = complex(fpi)
LAM = complex(1.1 * 10**3)  # cut-off scale taken to be 1.1 GeV
Q_MAX = sqrt(LAM**2 - MK**2)


def __bubble_loop_mat_elements(cme, m1, m2, q_max=Q_MAX):
    r"""Returns value of bubble loop for m1, m2 running in loop"""
    s = complex(cme**2, 0.0)

    delta = complex(m1**2 - m2**2)
    nu = sqrt((s - complex(m1 - m2)**2) * (s - complex(m1 + m2)**2))
    cut_fac1 = sqrt(complex(1. + m1**2 / q_max**2))
    cut_fac2 = sqrt(complex(1. + m2**2 / q_max**2))

    log1 = log(complex(m1**2 / m2**2))
    log2 = log(s - delta + nu * cut_fac1) - log(-s + delta + nu * cut_fac1)
    log3 = log(s - delta + nu * cut_fac2) - log(-s + delta + nu * cut_fac2)
    log4 = log(1. + cut_fac1) - log(1. + cut_fac2)
    log5 = log(1. + cut_fac1) + log(1. + cut_fac2)
    log6 = log(m1**2 * m2**2 / q_max**4)

    sum_logs = complex(-delta / s * log1 + nu / s * (log2 + log3) +
                       2. * delta / s * log4 - 2. * log5 + log6)

    return sum_logs / 32. / pi**2


def bubble_loop(cme, mass, q_max=Q_MAX):
    r"""
    Returns value of bubble loop given a hard momentum cut-off.

    Parameters
    ----------
    cme : float
        Center of mass energy.
    mass : float
        Mass of particle running in loop.

    Notes
    -----
    The bubble integral is
        G_{ii} = i\int_{0}^{\infty}\frac{d^4q}{(2\pi)^4}
            \frac{1}{q^2-m_{i}^2}\frac{1}{(P-q)^2-m_{i}^2}
    """
    cutFact = sqrt(complex(1. + mass**2 / q_max**2))
    if abs(cme) == 0.0:
        return complex((1. - cutFact * log(((1. + cutFact) * q_max) / mass)) /
                       (8. * cutFact * pi**2))
    return __bubble_loop_mat_elements(cme, mass, mass, q_max=q_max)


def loop_matrix(cme, q_max=Q_MAX):
    """
    Returns a matrix of the bubble loop integrals.

    Parameters
    ----------
    cme : float
        Center of mass energy.
    q_max : optional
        Hard momentum cut-off. Taken to be 1.1 GeV by default.

    Returns
    -------
    loop_mat : numpy.matrix
        Matrix of the bubble loop integrals. The 'i, j' component is the result
        of a bubble loop with a mass 'i' and mass 'j' running in the loop.
        'i=0' and 'i=1' correspond to the pion and kaon respectively.

    Notes
    -----
    The bubble loop is given by
        G_{ij} = i\int_{0}^{\infty}\frac{d^4q}{(2\pi)^4}
            \frac{1}{q^2-m_{i}^2}\frac{1}{(P-q)^2-m_{j}^2}
    """
    g11 = __bubble_loop_mat_elements(cme, MPI, MPI, q_max=q_max)
    g12 = __bubble_loop_mat_elements(cme, MPI, MK, q_max=q_max)
    g22 = __bubble_loop_mat_elements(cme, MK, MK, q_max=q_max)

    return np.matrix([[g11, g12], [g12, g22]], dtype=complex)
