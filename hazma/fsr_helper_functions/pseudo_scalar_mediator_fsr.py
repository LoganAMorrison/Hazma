"""Module for computing fsr spectrum from a pseudo-scalar mediator.

@author - Logan Morrison and Adam Coogan.
@data - December 2017

"""
import numpy as np

alpha = 1.0 / 137.0


def dnde_xx_to_p_to_ffg(eng_gam, cme, mass_f):
    """Returns the fsr spectra for fermions from decay of pseudo-scalar
    mediator.

    Computes the final state radiaton spectrum value dNdE from a pseudo-scalar
    mediator given a gamma ray energy of `eng_gam`, center of mass energy `cme`
    and final state fermion mass `mass_f`.

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
    val = 0.0

    e, m = eng_gam / cme, mass_f / cme

    if 0 < e and e < 0.5 * (1.0 - 2 * m**2):

        pre_factor = alpha / \
            (e * np.pi * np.sqrt((1 - 4 * m**2) * cme**2))

        terms = np.array([
            -2 * np.sqrt(1 - 2 * e) * np.sqrt(1 - 2 * e - 4 * m**2),
            2 * (1 + 2 * (-1 + e) * e - 2 * m**2) *
            np.arctanh(np.sqrt(1 - 2 * e - 4 * m**2) /
                       np.sqrt(1 - 2 * e)),
            (1 + 2 * (-1 + e) * e - 2 * m**2) *
            np.log(1 + np.sqrt(1 - 2 * e - 4 * m**2) /
                   np.sqrt(1 - 2 * e)),
            -((1 + 2 * (-1 + e) * e - 2 * m**2) *
              np.log(1 - np.sqrt(1 - 2 * e - 4 * m**2) /
                     np.sqrt(1 - 2 * e)))
        ])

        val = np.real(pre_factor * np.sum(terms))

    return val
