"""Module for computing fsr spectrum from a pseudo-scalar mediator.

@author - Logan Morrison and Adam Coogan.
@data - December 2017

"""

import numpy as np

from cmath import sqrt, log, pi

from hazma.parameters import alpha_em
from hazma.utils import RealArray

from ._proto import PseudoScalarMediatorBase


def __dnde_xx_to_p_to_ffg(photon_energies: RealArray, Q: float, mf: float):
    """
    Returns the fsr spectra for fermions from decay of pseudo-scalar
    mediator.

    Computes the final state radiaton spectrum value dNdE from a
    pseudo-scalar mediator given a gamma ray energy of `eng_gam`,
    center of mass
    energy `cme` and final state fermion mass `mass_f`.

    Paramaters
    ----------
    eng_gam : float
        Gamma ray energy.
    cme: float
        Center of mass energy of mass of off-shell pseudo-scalar mediator.
    mass_f : float
        Mass of the final state fermion.

    Returns
    -------
    spec_val : float
        Spectrum value dNdE from pseudo-scalar mediator.
    """
    m = mf / Q
    ee = photon_energies / Q
    s = Q**2 - 2.0 * Q * photon_energies
    mask = (4.0 * mf**2 <= s) & (s <= Q**2)
    dnde = np.zeros_like(photon_energies)

    e = ee[mask]
    dnde[mask] = (
        2.0
        * alpha_em
        * (
            -sqrt((-1.0 + 2.0 * e) * (-1.0 + 2.0 * e + 4.0 * m**2))
            + (2.0 + 4.0 * (-1.0 + e) * e) * log(m)
            + m**2
            * (
                2.0 * log(sqrt(1.0 - 2.0 * e) - sqrt(1.0 - 2.0 * e - 4.0 * m**2))
                - log(
                    2.0
                    * (
                        1.0
                        - 2.0 * e
                        - 2.0 * m**2
                        + sqrt((-1.0 + 2.0 * e) * (-1.0 + 2.0 * e + 4.0 * m**2))
                    )
                )
            )
            + (1.0 + 2.0 * (-1.0 + e) * e)
            * log(
                -2.0
                / (
                    -1.0
                    + 2.0 * e
                    + 2.0 * m**2
                    + sqrt((-1.0 + 2.0 * e) * (-1.0 + 2.0 * e + 4.0 * m**2))
                )
            )
        )
    ) / (e * sqrt(1.0 - 4.0 * m**2) * pi * Q)

    return dnde


def dnde_xx_to_p_to_ffg(photon_energies, Q, mf):
    """Returns the fsr spectra for fermions from decay of pseudo-scalar
    mediator.

    Computes the final state radiaton spectrum value dNdE from a
    pseudo-scalar mediator given a gamma ray energy of `eng_gam`,
    center of mass energy `cme` and final state fermion mass `mass_f`.

    Paramaters
    ----------
    eng_gam : float or array-like
        Gamma ray energy.
    cme: float
        Center of mass energy of mass of off-shell pseudo-scalar mediator.
    mass_f : float
        Mass of the final state fermion.

    Returns
    -------
    spec_val : float
        Spectrum value dNdE from pseudo-scalar mediator.
    """
    scalar = np.isscalar(photon_energies)
    e = np.atleast_1d(photon_energies)
    dnde = __dnde_xx_to_p_to_ffg(e, Q, mf)
    return dnde[0] if scalar else dnde
