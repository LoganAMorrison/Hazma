from ..decay_helper_functions cimport decay_long_kaon
import numpy as np
cimport numpy as np


cdef double decay_spectra_point(double eng_gam, double eng_pi):
    """
    Compute dNdE from long kaon decay.

    Compute dNdE from decay of charged kaon through K -> X in the laboratory
    frame given a gamma ray engergy of `eng_gam` and long kaon energy of
    `eng_k`. The decay modes impemendted are
        * kl    -> pi  + e   + nu
        * kl    -> pi  + mu  + nu
        * kl    -> pi0 + pi0  + pi0
        * kl    -> pi  + pi  + pi0

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
    return decay_long_kaon.CSpectrumPoint(eng_gam, eng_k)


cdef np.ndarray decay_spectra(np.ndarray eng_gam, double eng_k):
    """
    Compute dNdE from long kaon decay.

    Compute dNdE from decay of charged kaon through K -> X in the laboratory
    frame given a gamma ray engergy of `eng_gam` and long kaon energy of
    `eng_k`. The decay modes impemendted are
        * kl    -> pi  + e   + nu
        * kl    -> pi  + mu  + nu
        * kl    -> pi0 + pi0  + pi0
        * kl    -> pi  + pi  + pi0

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
    return decay_long_kaon.CSpectrum(eng_gam, eng_k)


def fsr(eng_gam, cme, mediator='scalar'):
    """
    NOT YET IMPLEMENTED!
    """
    raise ValueError('FSR spectrum for charged pion is not yet available')
