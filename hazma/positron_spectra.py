"""
Module for computing positron spectra.

@author: Logan Morrison and Adam Coogan
@date: May 2018

"""

from positron_helper_functions import positron_muon
from positron_helper_functions import positron_charged_pion


def muon(eng_p, eng_mu):
    """
    Returns the positron spectrum from a muon.

    Parameters
    ----------
    eng_p : float or numpy.array
        Energy of the positron.
    eng_mu : float or array-like
        Energy of the muon.

    Returns
    -------
    dnde : float or numpy.array
        The value of the spectrum given a positron energy `ee`
        and muon energy `emu`
    """
    if hasattr(eng_p, "__len__"):
        return positron_muon.Spectrum(eng_p, eng_mu)
    return positron_muon.SpectrumPoint(eng_p, eng_mu)


def charged_pion(eng_p, eng_pi):
    """
    Returns the positron spectrum from a muon.

    Parameters
    ----------
    eng_p : float or numpy.array
        Energy of the positron.
    eng_pi : float or numpy.array
        Energy of the charged pion.

    Returns
    -------
    dnde : float or numpy.array
        The value of the spectrum given a positron energy(ies) `eng_p`
        and charged pion energy `eng_pi`
    """
    if hasattr(eng_p, "__len__"):
        return positron_charged_pion.Spectrum(eng_p, eng_pi)
    return positron_charged_pion.SpectrumPoint(eng_p, eng_pi)
