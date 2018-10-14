"""Module for computing fsr spectrum from a vector mediator.

@author - Logan Morrison and Adam Coogan.
@data - December 2017

"""
import numpy as np

from ..parameters import alpha_em


def __dnde_xx_to_v_to_ffg(egam, Q, mf):
    """Return the fsr spectra for fermions from decay of vector mediator.

    Computes the final state radiaton spectrum value dNdE from a vector
    mediator given a gamma ray energy of `egam`, center of mass energy `Q`
    and final state fermion mass `mf`.

    Paramaters
    ----------
    egam : float
        Gamma ray energy.
    Q: float
        Center of mass energy of mass of off-shell vector mediator.
    mf : float
        Mass of the final state fermion.

    Returns
    -------
    spec_val : float
        Spectrum value dNdE from vector mediator.
    """
    e, m = egam / Q, mf / Q

    if 0 < e and e < 0.5 * (1.0 - 4 * m**2):
        return -(alpha_em *
                 (4 * np.sqrt(1 - 2 * e - 4 * m**2) *
                  (1 - 2 * m**2 + 2 * e * (-1 + e + 2 * m**2)) +
                  np.sqrt(1 - 2 * e) * (1 + 2 * (-1 + e) * e -
                                        4 * e * m**2 - 4 * m**4) *
                  (np.log(1 - 2 * e) - 4 *
                   np.log(np.sqrt(1 - 2 * e) +
                          np.sqrt(1 - 2 * e - 4 * m**2)) +
                   2 * np.log((np.sqrt(1 - 2 * e) -
                               np.sqrt(1 - 2 * e - 4 * m**2)) *
                              (1 - np.sqrt(1 + (4 * m**2) /
                                           (-1 + 2 * e))))))) / \
            (2. * e * (1 + 2 * m**2) *
             np.sqrt((-1 + 2 * e) * (-1 + 4 * m**2)) * np.pi * Q)

    else:
        return 0.0


def dnde_xx_to_v_to_ffg(egam, Q, mf):
    """Return the fsr spectra for fermions from decay of vector mediator.

    Computes the final state radiaton spectrum value dNdE from a vector
    mediator given a gamma ray energy of `egam`, center of mass energy `Q`
    and final state fermion mass `mf`.

    Paramaters
    ----------
    egam : float
        Gamma ray energy.
    Q: float
        Center of mass energy of mass of off-shell vector mediator.
    mf : float
        Mass of the final state fermion.

    Returns
    -------
    spec_val : float
        Spectrum value dNdE from vector mediator.
    """
    if hasattr(egam, '__len__'):
        return np.array([__dnde_xx_to_v_to_ffg(e, Q, mf) for e in egam])
    else:
        return __dnde_xx_to_v_to_ffg(egam, Q, mf)
