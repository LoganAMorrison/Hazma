"""
Module for computing the neutrino spectrum from a muon decay.
"""

import numpy as np
cimport numpy as np
import cython
from libc.math cimport log, sqrt, fmin
from libc.float cimport DBL_EPSILON
from hazma.spectra._neutrino._neutrino cimport NeutrinoSpectrumPoint, new_neutrino_spectrum_point

include "../../_utils/constants.pxd"

# ===================================================================
# ---- Pure Cython API functions ------------------------------------
# ===================================================================

DEF R = MASS_E / MASS_MU
DEF R2 = R * R
DEF R4 = R2 * R2
DEF R6 = R4 * R2
# 1 / (1 - 8 r^2 + 8 r^6 - r^8 - 12 r^2 ln(r^2))
DEF R_FACTOR = 1.0001870858234163 

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef NeutrinoSpectrumPoint c_muon_decay_spectrum_point_rest(double enu):
    """
    Compute the muon decay spectrum into a single neutrino (either an
    electron or muon neutrino) from a muon at rest.

    Parameters
    ----------
    enu: double
        Energy of the neutrino.

    Returns
    -------
    dnde: double
        Spectrum given a muon at rest and neutrino with energy `enu`.
    """
    cdef double pre
    cdef double dndxe
    cdef double dndxm
    cdef double common
    cdef double x
    cdef double xm
    cdef NeutrinoSpectrumPoint result = new_neutrino_spectrum_point()

    pre = 2.0 / MASS_MU
    x = pre * enu

    if x <= 0.0 or x >= 1 - R**2:
        return result

    xm = 1.0 - x
    common = R_FACTOR * x**2 * (1.0 - R**2 - x)**2 / xm

    dndxe = 12.0 * common
    dndxm = 2.0 * common * (3.0 + R2 * (3.0 - x) - 5.0 * x + 2.0 * x**2) / xm**2

    result.electron = pre * dndxe
    result.muon = pre * dndxm

    return result


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef NeutrinoSpectrumPoint c_muon_decay_spectrum_point(double enu, double emu):
    """
    Compute the boosted muon decay spectrum into a single neutrino (either an
    electron or muon neutrino) given the neutrino energy and muon energy.

    Parameters
    ----------
    enu: double
        Energy of the neutrino.
    emu: double
        Energy of the muon.

    Returns
    -------
    dnde: double
        Spectrum given a boosted muon with energy `emu` and neutrino with
        energy `enu`.
    """
    cdef:
        double e_to_x
        double pre
        double xm, xp
        double xmm, xpm
        double xmax_rf
        double x
        double gam, beta
        NeutrinoSpectrumPoint result = new_neutrino_spectrum_point()

    if emu < MASS_MU:
        return result

    # If we are sufficiently close to the muon rest-frame, use the 
    # rest-frame result.
    if emu - MASS_MU < DBL_EPSILON:
        return c_muon_decay_spectrum_point_rest(enu)

    # dN/dE = (2 / Q) * dN/dx
    e_to_x = 2.0 / emu

    x = e_to_x * enu
    gam = emu / MASS_MU
    beta = sqrt(1.0 - (MASS_MU / emu)**2)
    pre = R_FACTOR * e_to_x / (2.0 * beta)
   
    # Maximum x in the muon rest-frame
    xmax_rf = 1 - R**2
    
    # Bounds on the neutrino energy
    if x <= 0.0 or (1.0 + beta) * xmax_rf <= x:
        return result

    # Upper and lower bounds on energy integral
    xm = gam ** 2 * x * (1.0 - beta)
    xp = fmin(xmax_rf, gam ** 2 * x * (1.0 + beta))

    xmm = 1.0 - xm
    xpm = 1.0 - xp
    
    result.electron = 2 * pre * (
        (xm - xp)
        * (
            -3.0 * (xm + xp)
            + 2 * (3 * R4 + xm ** 2 + xm * xp + xp ** 2 + 3 * R2 * (xm + xp))
        )
        - 6 * R4 * log(xpm / xmm)
    )

    result.muon = pre * (
        3 * R2 * (xm - xp) * (xm + xp)
        + (xm ** 2 * (-9.0 + 4.0 * xm) + (9.0 - 4 * xp) * xp ** 2) / 3.0
        + R6 * ((-2.0 * xm) / xmm ** 2 + (2.0 * xp) / xpm ** 2)
        + 6 * R4 * (1.0 / xmm - 1.0 / xpm)
        + 2 * R4 * (-3 + R2) * log(xpm / xmm)
    )

    return result


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t,ndim=2] c_muon_decay_spectrum_array(double[:] energies, double emu):
    """
    Compute the boosted muon decay spectrum into a single neutrino (either an
    electron or muon neutrino) given array of neutrino energies and muon energy.

    Parameters
    ----------
    enu: np.ndarray
        Energies of the neutrino.
    emu: double
        Energy of the muon.

    Returns
    -------
    dnde: np.ndarray
        Spectrum given a boosted muon with energy `emu` and neutrino with
        energies `energies`.
    """
    cdef int npts = energies.shape[0]
    spec = np.zeros((3, npts), dtype=np.float64)
    cdef double[:,:] spec_view = spec
    cdef NeutrinoSpectrumPoint res

    for i in range(npts):
        res = c_muon_decay_spectrum_point(energies[i], emu)
        spec_view[0][i] = res.electron
        spec_view[1][i] = res.muon
        spec_view[2][i] = res.tau

    return spec


# ===================================================================
# ---- Python API ---------------------------------------------------
# ===================================================================

def dnde_neutrino_muon(egam, double emu):
    """
    Compute the neutrino spectrum dN/dE from the decay of a muon into an electron,
    and two neutrinos.

    Paramaters
    ----------
    egam: float or array-like
        Photon energy.
    emu: float 
        Energy of the muon.
    """
    cdef NeutrinoSpectrumPoint res

    if hasattr(egam, '__len__'):
        energies = np.array(egam)
        assert len(energies.shape) == 1, "Photon energies must be 0 or 1-dimensional."
        return c_muon_decay_spectrum_array(energies, emu)
    else:
        res = c_muon_decay_spectrum_point(egam, emu)
        return (res.electron, res.muon, res.tau)
