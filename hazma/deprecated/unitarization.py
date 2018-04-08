"""
Module for computing unitarized meson-meson scattering amplitudes.
"""

import numpy as np

from .parameters import fpi
from cmath import sqrt, log, pi, phase

from .meson_matrix_elements.leading_order import \
    partial_wave_pipi_to_pipi_LO_I
from .meson_matrix_elements.next_leading_order import \
    partial_wave_pipi_to_pipi_NLO_I

from .parameters import pion_mass_chiral_limit as mPI
from .parameters import kaon_mass_chiral_limit as mK


MPI = complex(mPI)
MK = complex(mK)
FPI = complex(fpi)
LAM = complex(1.1 * 10**3)  # cut-off scale taken to be 1.1 GeV
Q_MAX = sqrt(LAM**2 - MK**2)


# #######################
# Bethe Salpeter Equation
# #######################


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


def __amp_matrix(cme):
    """Matrix of tree level amplitudes. Note: 1 = pion, 2 = kaon."""
    mand_s = complex(cme**2, 0.0)

    m11 = complex((2 * mand_s - MPI**2) / 2. / FPI**2)
    m12 = complex((sqrt(3) / 4. / fpi**2) * mand_s)
    m22 = complex((3. / 4. / fpi**2) * mand_s)

    return np.matrix([[m11, m12], [m12, m22]], dtype=complex)


def __unit_mat(cme, q_max):
    """Returns matrix of unitarized I=0 amplitudes."""
    iden = np.identity(2, dtype=complex)
    gmat_diag = np.diag(np.diag(loop_matrix(cme)))
    mmat = __amp_matrix(cme)
    inv = np.matrix(iden + mmat * gmat_diag, dtype=complex).getI()

    return inv * mmat


def amp_bethe_salpeter_kk_to_kk(cmes, q_max=Q_MAX):
    """
    Returns the unitarized matrix element for kk -> kk in the zero isospin
    channel.

    This is computed by resumming an infinite number of bubble
    intermediate states. The intermediate states used were pions and kaons
    in the I = 0 channel.

    Parameters
    ----------
    cmes: array-like or float
        Invariant mass of the kaons.

    Returns
    -------
    Mu: array-like or float
        Unitarized matrix element for kk -> kk in the zero isospin
        channel.
    """

    if hasattr(cmes, "__len__"):
        return np.array([__unit_mat(cme, q_max)[1, 1] for cme in cmes])

    return __unit_mat(cmes, q_max)[1, 1]


def amp_bethe_salpeter_pipi_to_kk(cmes, q_max=Q_MAX):
    """
    Returns the unitarized matrix element for pipi -> kk in the zero isospin
    channel.

    This is computed by resumming an infinite number of bubble
    intermediate states. The intermediate states used were pions and kaons
    in the I = 0 channel.

    Parameters
    ----------
    cmes: array-like or float
        Invariant mass of the kaons.

    Returns
    -------
    Mu: array-like or float
        Unitarized matrix element for pipi -> kk in the zero isospin
        channel.
    """

    if hasattr(cmes, "__len__"):
        return np.array([__unit_mat(cme, q_max)[0, 1] for cme in cmes])

    return __unit_mat(cmes, q_max)[0, 1]


def amp_bethe_salpeter_pipi_to_pipi(cmes, q_max=Q_MAX):
    """
    Returns the unitarized matrix element for pipi -> pipi in the zero isospin
    channel.

    This is computed by resumming an infinite number of bubble
    intermediate states. The intermediate states used were pions and kaons
    in the I = 0 channel.

    Parameters
    ----------
    cmes: array-like or float
        Invariant mass of the pions.

    Returns
    -------
    Mu: complex array-like or complex
        Unitarized matrix element for pipi -> pipi in the zero isospin
        channel.
    """

    if hasattr(cmes, "__len__"):
        return np.array([__unit_mat(cme, q_max)[0, 0] for cme in cmes])

    return __unit_mat(cmes, q_max)[0, 0]


def __phase_shift(cme, t, deg=False):
    s = complex(cme**2, 0.0)

    p2 = complex(sqrt(s - 4 * MPI**2) / 2.)

    z = complex(1.0 + 2.0j * p2 * t / (8. * pi * sqrt(s)))

    theta = phase(z)

    if z.imag < 0.:
        theta = theta + 2. * pi

    if deg is True:
        theta = theta * 180. / pi

    return theta / 2.0


def phase_shift(cme, t, deg=False):
    """
    Computes the phase shift of an amplitude.

    Parameters
    ----------
    cme : float
        Center of mass energy of process.
    t : complex
        The amplitude of the process

    Returns
    -------
    delta : float
        Phase shift.
    """
    if hasattr(cme, '__len__'):
        return np.array([__phase_shift(cme[i], t[i], deg=deg)
                         for i in range(len(cme))])
    else:
        return __phase_shift(cme, t, deg=deg)


def __find_branch_index(angles, trigger=250):
    """
    Returns index where phases encounter a branch cut.
    """
    angles2 = 2 * angles
    index = 1

    while True:
        angle1 = angles2[index]
        angle2 = angles2[index - 1]
        if abs(angle1 - angle2) > trigger:
            break
        elif (index + 1 == len(angles)):
            index = None
            break
        else:
            index = index + 1

    return index


def fix_phases(phases, trigger=250):
    """
    Returns an corrected phases taking into account branch cuts.

    Parameters
    ----------
    phases : array-like
        List of phases.
    trigger : float, optional {250}
        Phase jump to trigger on.

    Returns
    -------
    phases : array-like
        Corrected list of phases.
    """
    index = __find_branch_index(phases, trigger)
    if index is not None:
        phases[index::] = phases[index::] + 180
    return phases


# ########################
# Inverse Amplitude Method
# ########################


def __amp_iam_pipi_to_pipi(cme):
    """
    Unitarized pion scattering squared matrix element in the isopin I = 0
    channel.
    """
    s = complex(cme**2)

    amp_lo = partial_wave_pipi_to_pipi_LO_I(s, ell=0, iso=0)
    amp_nlo = partial_wave_pipi_to_pipi_NLO_I(s, ell=0, iso=0)

    return amp_lo**2 / (amp_lo - amp_nlo)


def amp_inverse_amplitude_pipi_to_pipi(cmes):
    """
    Unitarized pion scattering amplitude in the isopin I = 0 channel.

    Unitarization was computed using the inverse amplitude method(IAM) with
    only pion contributions.

    Parameters
    ----------
    cme: double
        Invariant mass of the two charged pions.
    """
    if hasattr(cmes, "__len__"):
        return np.array([__amp_iam_pipi_to_pipi(cme) for cme in cmes])
    else:
        return __amp_iam_pipi_to_pipi(cmes)


def msqrd_inverse_amplitude_pipi_to_pipi(cmes):
    """
    Unitarized pion scattering sqrd amplitude in the isopin I = 0 channel.

    Unitarization was computed using the inverse amplitude method(IAM) with
    only pion contributions.

    Parameters
    ----------
    cme: double
        Invariant mass of the two charged pions.
    """
    if hasattr(cmes, "__len__"):
        return np.array([abs(amp_inverse_amplitude_pipi_to_pipi(cme))
                         for cme in cmes])
    else:
        return abs(amp_inverse_amplitude_pipi_to_pipi(cmes))
