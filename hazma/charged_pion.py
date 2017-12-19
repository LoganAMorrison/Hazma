"""
Module for computing gamma ray spectra from a charged pion.

@author - Logan Morrison and Adam Coogan
@date - December 2017
"""
from decay_helper_functions import decay_charged_pion


def decay_spectra(eng_gam, eng_pi):
    """
    Compute dNdE from charged pion decay.

    Compute dNdE from decay pi -> mu nu -> e nu nu g in the laborartory frame
    given a gamma ray engergy of `eng_gam` and muon energy of `eng_pi`.

    Paramaters
        eng_gam (float or numpy.ndarray) :
            Gamma ray energy(ies) in laboratory frame.
        eng_pi (float) :
            Charged pion energy in laboratory frame.

    Returns
        spec (np.ndarray) :
            List of gamma ray spectrum values, dNdE, evaluated at `eng_gams`
            given charged pion energy `eng_pi`.

    Examples
        Calculate spectrum for single gamma ray energy

        >>> from hazma import charged_pion
        >>> eng_gam, eng_pi = 200., 1000.
        >>> spec = charged_pion.decay_spectra(eng_gam, eng_pi)

        Calculate spectrum for array of gamma ray energies

        >>> from hazma import charged_pion
        >>> import numpy as np
        >>> eng_gams = np.logspace(0.0, 3.0, num=200, dtype=float)
        >>> eng_pi = 1000.
        >>> spec = charged_pion.decay_spectra(eng_gams, eng_pi)
    """
    # charged_pi = decay_charged_pion.ChargedPion()
    if hasattr(eng_gam, "__len__"):
        return decay_charged_pion.Spectrum(eng_gam, eng_pi)
    return decay_charged_pion.SpectrumPoint(eng_gam, eng_pi)


def fsr(eng_gam, cme, mediator='scalar'):
    """
    NOT YET IMPLEMENTED!
    """
    raise ValueError('FSR spectrum for charged pion is not yet available')
