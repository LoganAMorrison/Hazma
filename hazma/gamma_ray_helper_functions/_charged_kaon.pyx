from ..decay_helper_functions cimport decay_charged_kaon
import numpy as np
cimport numpy as np


cdef double decay_spectra_point(double eng_gam, double eng_k):
    """
    Compute dNdE from charged kaon decay.

    Paramaters
        eng_gam (float) :
            Gamma ray energy in laboratory frame.
        eng_k (float) :
            Charged kaon energy in laboratory frame.

    Returns
        spec (np.ndarray) :
            List of gamma ray spectrum values, dNdE, evaluated at `eng_gams`
            given muon energy `eng_mu`.
    """
    return decay_charged_kaon.CSpectrumPoint(eng_gam, eng_k)


cdef np.ndarray decay_spectra(np.ndarray eng_gam, double eng_k):
    """
    Compute dNdE from charged kaon decay.

    Compute dNdE from decay of charged kaon through K -> X in the laboratory
    frame given a gamma ray engergy of `eng_gam` and charged kaon energy of
    `eng_k`. The decay modes impemendted are
        * k -> mu  + nu
        * k -> pi  + pi0
        * k -> pi  + pi  + pi
        * k -> pi0 + e   + nu
        * k -> pi0 + mu  + nu
        * k -> pi  + pi0 + pi0

    Paramaters
        eng_gam (float or numpy.ndarray) :
            Gamma ray energy(ies) in laboratory frame.
        eng_k (float) :
            Charged kaon energy in laboratory frame.

    Returns
        spec (np.ndarray) :
            List of gamma ray spectrum values, dNdE, evaluated at `eng_gams`
            given muon energy `eng_mu`.
    """
    return decay_charged_kaon.CSpectrum(eng_gam, eng_k)


def fsr(eng_gam, cme, mediator='scalar'):
    """
    NOT YET IMPLEMENTED!
    """
    raise ValueError('FSR spectrum for charged pion is not yet available')
