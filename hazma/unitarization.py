"""
Module for computing unitarized meson-meson scattering amplitudes.
"""

import os
import numpy as np
from .parameters import charged_pion_mass as mpi
from .parameters import charged_kaon_mass as mk
from .parameters import neutral_pion_mass as mpi0
from .parameters import neutral_kaon_mass as mk0
from .parameters import fpi
from cmath import sqrt, log, pi, phase

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


def pion_bubble(cme, q_max):
    r"""
    Returns pion bubble loop factor.

    Returns
    -------
    G22 : complex
        G_{22} = i\int_{0}^{\infty}\frac{d^4q}{(2\pi)^4}
            \frac{1}{q^2-m_{\pi}^2}\frac{1}{(P-q)^2-m_{\pi}^2}

    Notes
    -----
    The integral is regulated with a finite momentum cutoff equal to 1.1 GeV.

    This code is used to determine branch for bubble:

    mand_s = complex(cme**2, 0.0)

    mom1 = complex(sqrt(mand_s - 4. * MK**2) / 2.)
    mom2 = complex(sqrt(mand_s - 4. * MPI**2) / 2.)

    if mom1.imag > 0. and mom2.imag > 0.:
        return g22
    if mom1.imag > 0. and mom2.imag < 0.:
        return g22
    if mom1.imag < 0. and mom2.imag < 0.:
        return g22 - 2.0j * g22.imag
    if mom1.imag < 0. and mom2.imag > 0.:
        return g22 - 2.0j * g22.imag
    """

    return complex(bubble_loop(cme, MPI, q_max=q_max))


def kaon_bubble(cme, q_max):
    r"""
    Returns kaon bubble loop factor.

    Returns
    -------
    G11 : complex
        G_{11} = i\int_{0}^{\infty}\frac{d^4q}{(2\pi)^4}
            \frac{1}{q^2-m_{K}^2}\frac{1}{(P-q)^2-m_{K}^2}

    Notes
    -----
    The integral is regulated with a finite momentum cutoff equal to 1.1 GeV.

    mand_s = complex(cme**2, 0.0)

    mom1 = complex(sqrt(mand_s - 4. * MK**2) / 2.)
    mom2 = complex(sqrt(mand_s - 4. * MPI**2) / 2.)

    if mom1.imag > 0. and mom2.imag > 0.:
        return g11
    if mom1.imag > 0. and mom2.imag < 0.:
        return g11
    if mom1.imag < 0. and mom2.imag < 0.:
        return g11 - 2.0j * g11.imag
    if mom1.imag < 0. and mom2.imag > 0.:
        return g11 - 2.0j * g11.imag
    """

    return complex(bubble_loop(cme, MK, q_max))


def amp_pipi_to_pipi_lo(cme):
    """
    Returns leading-order
    """
    mand_s = complex(cme**2, 0.0)
    return complex((1. / 2. / fpi**2) * (2. * mand_s - MPI**2))


def __M12(cme):
    mand_s = complex(cme**2, 0.0)
    return complex((sqrt(3) / 4. / fpi**2) * mand_s)


def __M22(cme):
    mand_s = complex(cme**2, 0.0)
    return complex((3. / 4. / fpi**2) * mand_s)


def __detM(cme):
    eng = complex(cme)
    return complex(amp_pipi_to_pipi_lo(eng) * __M22(eng) - __M12(eng)**2)


def __delta(cme, q_max):
    eng = complex(cme)
    return 1. + kaon_bubble(eng, q_max) * __M22(eng) + \
        pion_bubble(eng, q_max) * \
        (amp_pipi_to_pipi_lo(eng) + kaon_bubble(eng, q_max) * __detM(eng))


def __amp_bs_pipi_to_pipi(cme, q_max):

    eng = complex(cme)

    retval = (amp_pipi_to_pipi_lo(eng) +
              kaon_bubble(eng, q_max) * __detM(eng)) / __delta(eng, q_max)

    return complex(retval)


def __amp_bs_pipi_to_kk(cme, q_max):

    eng = complex(cme)

    retval = __M12(eng) / __delta(eng, q_max)

    return complex(retval)


def __amp_bs_kk_to_kk(cme, q_max):

    eng = complex(cme)

    retval = (__M22(eng) + pion_bubble(eng, q_max) * __detM(eng)) \
        / __delta(eng, q_max)

    return complex(retval)


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

# Import data for IAM

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "unitarization_data",
                         "final_state_int_unitarizated.dat")

unitarizated_data = np.loadtxt(DATA_PATH, delimiter=',', dtype=complex)
unitarizated_data_x = np.real(unitarizated_data[:, 0])
unitarizated_data_y = unitarizated_data[:, 1]


def amp_pipi_to_pipi_I0(cme):
    """
    Lowest order pion scattering squared matrix element in the isospin I = 0
    channel.

    Parameters
    ----------
    cme: double
        Invariant mass of the two charged pions.
    """
    return (2 * cme**2 - mpi0**2) / (32. * fpi**2 * pi)


def amp_inverse_amplitude_pipi_to_pipi(cme):
    """
    Unitarized pion scattering squared matrix element in the isopin I = 0
    channel.

    Unitarization was computed using the inverse amplitude method(IAM) with
    only pion contributions.

    Parameters
    ----------
    cme: double
        Invariant mass of the two charged pions.
    """
    return np.interp(cme, unitarizated_data_x, unitarizated_data_y)


def ratio_pipi_to_pipi_unitarized_tree(cme):
    """
    Returns the unitarized squared matrix element for: math: `\pi\pi\to\pi\pi`
    divided by the leading order, ununitarized squared matrix element for
    : math: `\pi\pi\to\pi\pi`.

    This was computed using the inverse amplitude method(IAM) with only

    Parameters
    ----------
    cme: double
        Invariant mass of the two charged pions.

    Results
    -------
    __unit_matrix_elem_sqrd: double
        The unitarized matrix element for: math: `\pi\pi\to\pi\pi`, | t_u | ^2,
        divided by the un - unitarized squared matrix element for
        : math: `\pi\pi\to\pi\pi`, | t | ^2; | t_u | ^2 / |t | ^2.
    """
    return amp_inverse_amplitude_pipi_to_pipi(cme) / \
        amp_pipi_to_pipi_I0(cme)


def msqrd_bethe_salpeter_pipi_to_pipi(Q):
    """
    Returns the square of the unitarized pipi -> pipi amplitude.

    Parameters
    ----------
    Q: float
        Invariant mass of the pions.

    Returns
    -------
    Mu2: float
        Unitarized squared matrix element for pipi -> pipi in the zero isospin
        channel.
    """
    return np.abs(amp_bethe_salpeter_pipi_to_pipi(Q))**2
