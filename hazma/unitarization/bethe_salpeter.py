"""
Module for computing unitarized meson-meson scattering amplitudes.
"""

import numpy as np
from cmath import sqrt, pi, phase

from ..parameters import fpi
from ..parameters import pion_mass_chiral_limit as mPI
from ..parameters import kaon_mass_chiral_limit as mK

from .loops import loop_matrix


MPI = complex(mPI)
MK = complex(mK)
FPI = complex(fpi)
LAM = complex(1.1 * 10**3)  # cut-off scale taken to be 1.1 GeV
Q_MAX = sqrt(LAM**2 - MK**2)


# #######################
# Bethe Salpeter Equation
# #######################

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


def amp_kk_to_kk_bse(cmes, q_max=Q_MAX):
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


def amp_pipi_to_kk_bse(cmes, q_max=Q_MAX):
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


def amp_pipi_to_pipi_bse(cmes, q_max=Q_MAX):
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
