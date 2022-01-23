"""
Module for computing the photon spectrum from radiative short kaon decay.

Description:
    The short kaon has many decay modes:

        ks -> pi + pi
        ks -> pi0 + pi0
        ks -> pi + pi + g

    For the the two-body final states, the sum of the decay spectra are
    computed given the known energies of the final state particles in the
    kaon's rest frame. The spectrum is then boosted into the lab frame.
"""

import numpy as np
cimport numpy as np
from scipy.integrate import quad
from libc.math cimport sqrt
import cython
import os
import sys
from .get_path import get_dir_path
include "common.pxd"


data_path_total = os.path.join(get_dir_path(),
                               "interpolation_data",
                               "skaon",
                               "short_kaon_interp_total.dat")
data_path_00 = os.path.join(get_dir_path(),
                            "interpolation_data",
                            "skaon",
                            "short_kaon_interp_00.dat")
data_path_pm = os.path.join(get_dir_path(),
                            "interpolation_data",
                            "skaon",
                            "short_kaon_interp_pm.dat")

data_path_pmg = os.path.join(get_dir_path(),
                             "interpolation_data",
                             "skaon",
                             "short_kaon_interp_pmg.dat")

__e_gams_total, __spec_total = np.loadtxt(data_path_total, delimiter=',').T
__e_gams_00, __spec_00 = np.loadtxt(data_path_00, delimiter=',').T
__e_gams_pm, __spec_pm = np.loadtxt(data_path_pm, delimiter=',').T
__e_gams_pmg, __spec_pmg = np.loadtxt(data_path_pmg, delimiter=',').T


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double __interp_spec(double eng_gam, int bitflags):
    cdef double ret = 0.0
    if bitflags & 1:
        ret += np.interp(eng_gam, __e_gams_00, __spec_00)
    if bitflags & 2:
        ret += np.interp(eng_gam, __e_gams_pm, __spec_pm)
    if bitflags & 4:
        ret += np.interp(eng_gam, __e_gams_pmg, __spec_pmg)
    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double __integrand(double cl, double eng_gam, double eng_k, str mode):
    cdef double gamma_k = eng_k / MASS_K
    cdef double beta_k = sqrt(1.0 - (MASS_K / eng_k)**2)
    cdef double eng_gam_k_rf = eng_gam * gamma_k * (1.0 - beta_k * cl)
    cdef double pre_factor = 1.0 / (2.0 * gamma_k * (1.0 - beta_k * cl))
    return pre_factor * __interp_spec(eng_gam_k_rf, mode)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double c_short_kaon_decay_spectrum_point(double eng_gam, double eng_k, int mode):
    if eng_k < MASS_K0:
        return 0.0
    return quad(__integrand, -1.0, 1.0, points=[-1.0, 1.0], args=(eng_gam, eng_k, mode), epsabs=1e-10, epsrel=1e-4)[0]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[np.float64_t,ndim=1] c_short_kaon_decay_spectrum_array(np.ndarray[np.float64_t,ndim=1] eng_gams, double eng_k, int mode):
    cdef int npts = eng_gams.shape[0]
    cdef np.ndarray[np.float64_t,ndim=1] spec = np.zeros_like(eng_gams)
    for i in range(npts):
        spec[i] = c_short_kaon_decay_spectrum_point(eng_gams[i], eng_k, mode)
    return spec


# ===================================================================
# ---- Python API ---------------------------------------------------
# ===================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def short_kaon_decay_spectrum(egam, ek, modes=["00","pm","pmg"]):
    """
    Compute the photon spectrum dN/dE from the decay of a short kaon.

    Paramaters
    ----------
    egam: float or array-like
        Photon energy.
    ek: float 
        Energy of the kaon.
    modes: List[str]
        List of strings representing the modes to include. The availible modes are:
        "00","pm", and "pmg".
    """
    cdef int bitflags = 0 

    if "00" in modes:
        bitflags += 1
    if "pm" in modes:
        bitflags += 2
    if "pmg" in modes:
        bitflags += 4
    
    if bitflags == 0:
        raise ValueError("Invalid modes specified.") 

    if hasattr(egam, '__len__'):
        energies = np.array(egam)
        assert len(energies.shape) == 1, "Photon energies must be 0 or 1-dimensional."
        return c_short_kaon_decay_spectrum_array(energies, ek, bitflags)
    else:
        return c_short_kaon_decay_spectrum_point(egam, ek, bitflags)
