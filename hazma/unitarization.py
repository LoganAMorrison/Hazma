"""
Module for computing unitarized meson-meson scattering amplitudes.
"""

import numpy as np
from .parameters import charged_pion_mass as mpi
from .parameters import charged_kaon_mass as mk
from .parameters import neutral_pion_mass as mpi0
from .parameters import neutral_kaon_mass as mk0
from .parameters import fpi
from cmath import sqrt, log, pi, phase

from .matrix_elements.meson_meson_lo import partial_wave_pipi_to_pipi_LO_I
from .matrix_elements.meson_meson_nlo import partial_wave_pipi_to_pipi_NLO_I


MPI = complex((mpi + mpi0) / 2.)
MK = complex((mk + mk0) / 2.)
FPI = complex(fpi)
LAM = complex(1.1 * 10**3)  # cut-off scale taken to be 1.1 GeV
Q_MAX = sqrt(LAM**2 - MK**2)


# #######################
# Bethe Salpeter Equation
# #######################

def bubble_loop(cme, mass, q_max=Q_MAX):
    r"""
    Returns value of bubble loop

    Returns
        G_{ii} = i\int_{0}^{\infty}\frac{d^4q}{(2\pi)^4}
            \frac{1}{q^2-m_{i}^2}\frac{1}{(P-q)^2-m_{i}^2}

    Notes
    -----
    The integral is regulated with a finite momentum cutoff equal to 1.1 GeV.
    """
    mand_s = complex(cme**2, 0.0)

    sig = sqrt(1. - 4. * mass**2 / mand_s)
    cut_fac = sqrt(1. + mass**2 / q_max**2)

    return (sig * log((sig * cut_fac + 1) / (sig * cut_fac - 1)) -
            2. * log(q_max / mass * (1. + cut_fac))) / 16. / pi**2


def __loop_matrix(cme, q_max, i):
    """
    Vector of bubble integrals.
    Note that 1 stands from pion and 2 for kaon.
    """
    if i == 1:
        complex(bubble_loop(cme, MK, q_max))
    if i == 2:
        complex(bubble_loop(cme, MK, q_max))


def __amp_matrix(cme, i, j):
    """
    Matrix of tree level amplitudes.
    Note that 1 stands from pion and 2 for kaon.
    """
    mand_s = complex(cme**2, 0.0)
    if i == 1:
        if j == 1:
            complex((2 * mand_s - MPI**2) / 2. / FPI**2)
        if j == 2:
            complex((sqrt(3) / 4. / fpi**2) * mand_s)
    if i == 2:
        if j == 1:
            complex((sqrt(3) / 4. / fpi**2) * mand_s)
        if j == 2:
            complex((3. / 4. / fpi**2) * mand_s)


def __detM(cme):
    """determinant of matrix of amplitudes."""
    eng = complex(cme)
    return complex(__amp_matrix(eng, 1, 1) * __amp_matrix(eng, 2, 2) -
                   __amp_matrix(eng, 1, 2)**2)


def __delta(cme, q_max):
    """1 + G22 M22 + G11 (M11 + G22 det(M))"""
    eng = complex(cme)
    return 1. + __loop_matrix(eng, eng, 2) * __amp_matrix(eng, 2, 2) + \
        __loop_matrix(eng, eng, 1) * \
        (__amp_matrix(eng, 1, 1) + __loop_matrix(eng, eng, 2) * __detM(eng))


def __amp_bs_pipi_to_pipi(cme, q_max):
    """bse pipi->pipi scattering"""
    eng = complex(cme)

    retval = (__amp_matrix(eng, 1, 1) +
              __loop_matrix(eng, eng, 2) * __detM(eng)) / __delta(eng, q_max)

    return complex(retval)


def __amp_bs_pipi_to_kk(cme, q_max):
    """bse pipi->kk scattering"""
    eng = complex(cme)

    return complex(__amp_matrix(eng, 1, 2) / __delta(eng, q_max))


def __amp_bs_kk_to_kk(cme, q_max):
    """bse kk->kk scattering"""
    eng = complex(cme)

    return complex((__amp_matrix(eng, 2, 2) +
                    __loop_matrix(eng, eng, 1) * __detM(eng)) /
                   __delta(eng, q_max))


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
        return np.array([__amp_bs_kk_to_kk(cme, q_max) for cme in cmes])

    return __amp_bs_kk_to_kk(cmes, q_max)


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
        return np.array([__amp_bs_pipi_to_kk(cme, q_max) for cme in cmes])

    return __amp_bs_pipi_to_kk(cmes, q_max)


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
        return np.array([__amp_bs_pipi_to_pipi(cme, q_max) for cme in cmes])

    return __amp_bs_pipi_to_pipi(cmes, q_max)


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

    amp_lo = partial_wave_pipi_to_pipi_LO_I(s, l=0, iso=0)
    amp_nlo = partial_wave_pipi_to_pipi_NLO_I(s, l=0, iso=0)

    return amp_lo / (amp_lo - amp_nlo)


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
