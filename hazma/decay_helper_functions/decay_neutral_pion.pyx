from libc.math cimport sqrt
import numpy as np
cimport numpy as np
import cython
include "common.pxd"


"""
Module for computing the decay spectrum from neutral pion.
"""

@cython.cdivision(True)
cdef double c_neutral_pion_decay_spectrum_point(double eng_gam, double eng_pi):
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
cdef np.ndarray[np.float64_t,ndim=1] c_neutral_pion_decay_spectrum_array(np.ndarray[np.float64_t,ndim=1] egams, double epi):
    """
    Returns decay spectrum for pi0 -> g g.
    """
    cdef int npts = egams.shape[0]
    cdef np.ndarray[np.float64_t,ndim=1] spec = np.zeros_like(egams)
    for i in range(npts):
        spec[i] = c_neutral_pion_decay_spectrum_point(egams[i], epi)
    return spec


@cython.cdivision(True)
@cython.boundscheck(True)
@cython.wraparound(False)
def neutral_pion_decay_spectrum(egams, epi):
    """
    Compute the photon spectrum dN/dE from the decay of a neutral pion.

    Paramaters
    ----------
    egam: float or array-like
        Photon energy.
    epi: float 
        Energy of the pion.
    """
    if hasattr(egams, '__len__'):
        energies = np.array(egams)
        assert len(energies.shape) == 1, "Photon energies must be 0 or 1-dimensional."
        return c_neutral_pion_decay_spectrum_array(energies, epi)
    else:
        return c_neutral_pion_decay_spectrum_point(egams, epi)
