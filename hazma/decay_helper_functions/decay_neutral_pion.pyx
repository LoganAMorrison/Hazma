from libc.math cimport sqrt
import numpy as np
cimport numpy as np
import cython

include "parameters.pxd"

"""
Module for computing the decay spectrum from neutral pion.
"""

@cython.cdivision(True)
cdef double CSpectrumPoint(double eng_gam, double eng_pi):
    """
    Returns decay spectrum for pi0 -> g g.
    """
    if eng_pi < MASS_PI0:
        return 0.0

    cdef float beta = sqrt(1.0 - (MASS_PI0 / eng_pi)**2)
    cdef float ret_val = 0.0

    if eng_pi * (1 - beta) / 2.0 <= eng_gam <= eng_pi * (1 + beta) / 2.0:
        ret_val = BR_PI0_TO_GG * 2.0 / (eng_pi * beta)

    return ret_val

@cython.cdivision(True)
cdef np.ndarray CSpectrum(np.ndarray eng_gam, double eng_pi):
    """
    Returns decay spectrum for pi0 -> g g.
    """

    cdef int num_pts = len(eng_gam)
    cdef np.ndarray spec = np.zeros(num_pts, dtype=np.float64)
    cdef int i

    for i in range(num_pts):
        spec[i] = CSpectrumPoint(eng_gam[i], eng_pi)

    return spec

@cython.cdivision(True)
def SpectrumPoint(double eng_gam, double eng_pi):
    """
    Returns decay spectrum for pi0 -> g g.
    """
    return CSpectrumPoint(eng_gam, eng_pi)

@cython.cdivision(True)
def Spectrum(np.ndarray eng_gam, double eng_pi):
    """
    Returns decay spectrum for pi0 -> g g.
    """
    return CSpectrum(eng_gam, eng_pi)
