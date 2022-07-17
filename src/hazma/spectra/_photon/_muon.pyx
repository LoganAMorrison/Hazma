"""
Module for computing the photon spectrum from radiative muon decay.
The radiative spectrum of the muon was taken from: arXiv:hep-ph/9909265
"Muon Decay and Physics Beyond the Standard Model".
"""
import cython
from libc.math cimport exp, log, M_PI, log10, sqrt
from libc.float cimport DBL_EPSILON

import numpy as np
cimport numpy as np
from scipy.integrate import quad
from scipy.special.cython_special cimport spence

include "../../_utils/constants.pxd"



# ===================================================================
# ---- Pure Cython API functions ------------------------------------
# ===================================================================

# Computes the muon decay spectrum dN/dE into photons for a muon at 
# rest.

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double dnde_photon_muon_rest_frame(double egam):
    cdef double y
    cdef double r
    cdef double pre
    cdef double ym
    cdef double poly1
    cdef double poly2

    # Rescaled variables
    y = 2 * egam / MASS_MU
    r = (MASS_E / MASS_MU)**2 

    if y <= 0.0 or y >= 1.0 - MASS_E / MASS_MU:
        return 0.0

    pre = ALPHA_EM / (3.0 * M_PI * y * MASS_MU) 

    ym = 1.0 - y
    poly1 = -102.0 + 46.0 * y - 101.0 * y**2 + 55.0 * y**3
    poly2 = 3.0 - 5.0 * y + 6.0 * y**2 - 6.0 * y**3 + 2.0 * y**4

    return 2.0 * pre * (poly1 * ym / 12.0 + poly2 * log(ym / r))

# Computes the muon decay spectrum dN/dE into photons for a boosted
# muon.

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double dnde_photon_muon_point(double egam, double emu):
    cdef double gamma
    cdef double beta
    cdef double y
    cdef double r
    cdef double x
    cdef double xp
    cdef double xm
    cdef double wp
    cdef double wm
    cdef double result

    if emu < MASS_MU:
        return 0.0

    # If we are sufficiently close to the muon rest-frame, use the 
    # rest-frame result.
    if emu - MASS_MU < DBL_EPSILON:
        return dnde_photon_muon_rest_frame(egam)

    gamma = emu / MASS_MU
    beta = sqrt(1.0 - (MASS_MU / emu)**2)

    # Rescaled variables
    y = 2 * egam / MASS_MU
    r = (MASS_E / MASS_MU)**2 
    # Re-rescaled variables
    x = y * gamma

    # Bounds check
    if x < 0.0 or x >= (1.0 - r) / (1.0 - beta):
        return 0.0

    # Upper bound on 'angular' variable (w = 1 - beta * ctheta)
    if x < (1.0 - r) / (1.0 + beta):
        wp = 1.0 + beta
    else:
        wp = (1.0 - r) / x
    wm = 1.0 - beta

    xp = x * wp
    xm = x * wm

    # Polynomial contribution
    result = (
            (xm - xp) * (102.0 + xm * xp * (
                191.0 + 21.0 * xm**2 + xm * (-92.0 + 21.0 * xp) + xp * (-92.0 + 21.0 * xp))
            ) / (12 * xm * xp * beta)
    )
    # Logarithmic contributions
    result += (9 + xm * (18 + xm * (-18 + (9 - 2 * xm) * xm))) * log((1 - xm) / r) / (3 * xm * beta)
    result += (-9 + xp * (-18 + xp * (18 + xp * (-9 + 2 * xp)))) * log((1 - xp) / r) / (3 * xp * beta)
    result += 5 / beta * (log((1 - xm) / r) * log(xm) - log((1 - xp) / r) * log(xp))
    result += 4 / (3 * beta) * (4 * log((1 - xp) / (1 - xm)) + 7 * log(xp / xm))
    # PolyLog terms
    result += 5 / beta * (spence(xm) - spence(xp))
    
    return result * ALPHA_EM / (3 * M_PI * emu)


# Same as the above function but over an array of arguments.

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t,ndim=1] dnde_photon_muon_array(double[:] energies, double muon_eng):
    cdef int npts = energies.shape[0]
    cdef np.ndarray[np.float64_t,ndim=1] spec = np.zeros_like(energies)

    for i in range(npts):
        spec[i] = dnde_photon_muon_point(energies[i], muon_eng)

    return spec


# ===================================================================
# ---- Python API ---------------------------------------------------
# ===================================================================

def dnde_photon(egam, emu):
    """
    Compute the photon spectrum dN/dE from the decay of a muon into an electron,
    two neutrinos and a photon.
    Paramaters
    ----------
    egam: float or array-like
        Photon energy.
    emu: float 
        Energy of the muon.
    """
    if hasattr(egam, '__len__'):
        energies = np.array(egam)
        assert len(energies.shape) == 1, "Photon energies must be 0 or 1-dimensional."
        return dnde_photon_muon_array(energies, emu)
    else:
        return dnde_photon_muon_point(egam, emu)
