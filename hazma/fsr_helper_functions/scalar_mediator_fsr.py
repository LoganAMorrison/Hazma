"""
Generate histograms of the energies of particles.

* Author - Logan A. Morrison and Adam Coogan
* Date - December 2017
"""
import numpy as np

alpha = 1.0 / 137.0


def fermion(eng_gam, cme, mass_f):
    """
    Return the fsr spectra for fermions from decay of scalar mediator.

    Computes the final state radiaton spectrum value dNdE from a scalar
    mediator given a gamma ray energy of `eng_gam`, center of mass energy `cme`
    and final state fermion mass `mass_f`.

    Paramaters
    ----------
    eng_gam : float
        Gamma ray energy.
    cme: float
        Center of mass energy of mass of off-shell scalar mediator.
    mass_f : float
        Mass of the final state fermion.

    Returns
    -------
    spec_val : float
        Spectrum value dNdE from scalar mediator.
    """
    val = 0.0

    if 0 < eng_gam and eng_gam < (cme**2 - 2 * mass_f**2) / (2 * cme):
        e, m = eng_gam / cme, mass_f / cme

        prefac = (4 * alpha) / (e * (1 - 4 * m**2)**1.5 * np.pi * cme)

        terms = np.array([
            2 * (-1 + 4 * m**2) *
            np.sqrt((1 - 2 * e) * (1 - 2 * e - 4 * m**2)),
            2 * (1 + 2 * (-1 + e) * e - 6 * m**2 + 8 * e * m**2 + 8 *
                 m**4) * np.arctanh(np.sqrt(1 - (4 * m**2) / (1 - 2 * e))),
            (1 + 2 * (-1 + e) * e - 6 * m**2 + 8 * e * m**2 + 8 * m**4) *
            np.log(1 + np.sqrt(1 - (4 * m**2) / (1 - 2 * e))),
            (-1 - 2 * (-1 + e) * e + 6 * m**2 - 8 * e * m**2 - 8 * m**4) *
            np.log(1 - np.sqrt(1 - (4 * m**2) / (1 - 2 * e)))
        ])

        val = np.real(prefac * np.sum(terms))

    return val
