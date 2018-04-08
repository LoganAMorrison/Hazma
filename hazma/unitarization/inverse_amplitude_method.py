import numpy as np
from cmath import sqrt

from .lo_amplitudes import partial_wave_pipi_to_pipi_LO_I
from nlo_amplitudes import partial_wave_pipi_to_pipi_NLO_I

from ..parameters import fpi
from ..parameters import pion_mass_chiral_limit as mPI
from ..parameters import kaon_mass_chiral_limit as mK


MPI = complex(mPI)
MK = complex(mK)
FPI = complex(fpi)
LAM = complex(1.1 * 10**3)  # cut-off scale taken to be 1.1 GeV
Q_MAX = sqrt(LAM**2 - MK**2)


def __amp_pipi_to_pipi_iam(cme):
    """
    Unitarized pion scattering squared matrix element in the isopin I = 0
    channel.
    """
    s = complex(cme**2)

    amp_lo = partial_wave_pipi_to_pipi_LO_I(s, ell=0, iso=0)
    amp_nlo = partial_wave_pipi_to_pipi_NLO_I(s, ell=0, iso=0)

    return amp_lo**2 / (amp_lo - amp_nlo)


def amp_pipi_to_pipi_iam(cmes):
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
        return np.array([__amp_pipi_to_pipi_iam(cme) for cme in cmes])
    else:
        return __amp_pipi_to_pipi_iam(cmes)


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
        return np.array([abs(amp_pipi_to_pipi_iam(cme))
                         for cme in cmes])
    else:
        return abs(amp_pipi_to_pipi_iam(cmes))
