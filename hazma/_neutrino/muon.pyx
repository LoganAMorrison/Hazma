"""
Module for computing the neutrino spectrum from a muon decay.
"""

import numpy as np
cimport numpy as np
import cython
from libc.math cimport log, sqrt
from libc.float cimport DBL_EPSILON
from hazma._neutrino.neutrino cimport NeutrinoSpectrumPoint, new_neutrino_spectrum_point
include "../_utils/constants.pxd"  


# ===================================================================
# ---- Pure Cython API functions ------------------------------------
# ===================================================================

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
    cdef double num
    cdef double den
    cdef double dnde
    cdef double y
    cdef double r
    cdef NeutrinoSpectrumPoint result = new_neutrino_spectrum_point()

    y = 2.0 * enu / MASS_MU
    r = MASS_E / MASS_MU

    if y <= 0.0 or y >= 1 - r**2:
        return result

    num = 24.0 * y ** 2 * (-1.0 + r ** 2 + y) ** 2
    den = (
        MASS_MU
        * (-1.0 + y)
        * (-1.0 + 8.0 * r ** 2 - 8.0 * r ** 6 + r ** 8 + 12.0 * r ** 4 * log(r ** 2))
    )
    
    dnde = num / den

    result.electron = dnde
    result.muon = dnde

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
    cdef double num
    cdef double den
    cdef double dnde
    cdef double y
    cdef double r
    cdef double wp
    cdef double g
    cdef NeutrinoSpectrumPoint result = new_neutrino_spectrum_point()


    # If we are sufficiently close to the muon rest-frame, use the 
    # rest-frame result.
    if emu - MASS_MU < DBL_EPSILON:
        return c_muon_decay_spectrum_point_rest(enu)

    y = 2.0 * enu / MASS_MU
    r = MASS_E / MASS_MU
    g = emu / MASS_MU
    b = sqrt(1.0 - (MASS_MU / emu)**2) 
    
    # Bounds on the neutrino energy
    if y <= 0.0 or y >= (1 - r**2) * (1.0 + b):
        return result

    # Upper bound on the angular integration variable depends on the
    # neutrino energy.
    if y > (1.0 - r**2) * (1.0 - b):
        wp = (1.0 - r**2) / (g**2 * y)
    else:
        wp = 1.0 + b

    num = g ** 2 * (-1.0 + b + wp) * y * (
        6.0 * r ** 4
        + 6.0 * g ** 2 * r ** 2 * (1.0 - b + wp) * y
        + g ** 2
        * y
        * (
            -3.0
            + 3.0 * b
            - 2.0 * y
            + (2.0 * (2.0 + wp) * y) / (1.0 + b)
            + wp * (-3.0 + 2.0 * g ** 2 * wp * y)
        )
    ) + 6.0 * r ** 4 * log(((1.0 + b) * (-1.0 + g ** 2 * wp * y)) / (-1.0 - b + y))

    den = b * (-1.0 + 8.0 * r ** 2 - 8.0 * r ** 6 + r ** 8 + 12.0 * r ** 4 * log(r ** 2))

    dnde = (2.0 / MASS_MU) * num / den

    result.electron = dnde
    result.muon = dnde

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

def muon_neutrino_spectrum(egam, double emu):
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
