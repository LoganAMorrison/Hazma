from ..decay_helper_functions cimport decay_charged_pion
import numpy as np
cimport numpy as np


cdef double decay_spectra_point(double eng_gam, double eng_pi):
    """
    Compute dNdE from charged pion decay.

    Paramaters
        eng_gam (float or numpy.ndarray) :
            Gamma ray energy(ies) in laboratory frame.
        eng_pi (float) :
            Charged pion energy in laboratory frame.
    """
    return decay_charged_pion.CSpectrumPoint(eng_gam, eng_pi)


cdef np.ndarray decay_spectra(np.ndarray eng_gam, double eng_pi):
    """
    Compute dNdE from charged pion decay.

    Paramaters
        eng_gam (float or numpy.ndarray) :
            Gamma ray energy(ies) in laboratory frame.
        eng_pi (float) :
            Charged pion energy in laboratory frame.
    """
    return decay_charged_pion.CSpectrum(eng_gam, eng_pi)

def fsr(eng_gam, cme, mediator='scalar'):
    """
    NOT YET IMPLEMENTED!
    """
    raise ValueError('FSR spectrum for charged pion is not yet available')
