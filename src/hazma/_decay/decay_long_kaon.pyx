"""
Module for computing the photon spectrum from radiative long kaon decay.

Description:
    The charged kaon has many decay modes:

    kl    -> pi  + e   + nu
    kl    -> pi  + mu  + nu
    kl    -> pi0 + pi0  + pi0
    kl    -> pi  + pi  + pi0

    For the three-body final state, the energies of the final state
    particles are computed using RAMBO. The spectra for each final state
    particle is computed are each point in phases space in the charged kaon's rest frame and then spectra are summed over. The spectra is then boosted into the lab frame.
"""

import numpy as np
cimport numpy as np
from scipy.integrate import quad
from libc.math cimport sqrt
import cython
import os
import sys
from .get_path import get_dir_path
import warnings

include "common.pxd"


# ===================================================================
# ---- Cython API ---------------------------------------------------
# ===================================================================

data_path_total = os.path.join(get_dir_path(),
                               "interpolation_data",
                               "lkaon",
                               "long_kaon_interp_total.dat")
data_path_000 = os.path.join(get_dir_path(),
                              "interpolation_data",
                              "lkaon",
                              "long_kaon_interp_000.dat")
data_path_penu = os.path.join(get_dir_path(),
                               "interpolation_data",
                               "lkaon",
                               "long_kaon_interp_penu.dat")
data_path_penug = os.path.join(get_dir_path(),
                             "interpolation_data",
                             "lkaon",
                             "long_kaon_interp_penug.dat")
data_path_pm0 = os.path.join(get_dir_path(),
                              "interpolation_data",
                              "lkaon",
                              "long_kaon_interp_pm0.dat")
data_path_pm0g = os.path.join(get_dir_path(),
                              "interpolation_data",
                              "lkaon",
                              "long_kaon_interp_pm0g.dat")
data_path_pmunu = os.path.join(get_dir_path(),
                            "interpolation_data",
                            "lkaon",
                            "long_kaon_interp_pmunu.dat")
data_path_pmunug = os.path.join(get_dir_path(),
                             "interpolation_data",
                             "lkaon",
                             "long_kaon_interp_pmunug.dat")


__e_gams_total, __spec_total = np.loadtxt(data_path_total, delimiter=',').T
__e_gams_000, __spec_000 = np.loadtxt(data_path_000, delimiter=',').T
__e_gams_penu, __spec_penu = np.loadtxt(data_path_penu, delimiter=',').T
__e_gams_penug, __spec_penug = np.loadtxt(data_path_penug, delimiter=',').T
__e_gams_pm0, __spec_pm0 = np.loadtxt(data_path_pm0, delimiter=',').T
__e_gams_pm0g, __spec_pm0g = np.loadtxt(data_path_pm0g, delimiter=',').T
__e_gams_pmunu, __spec_pmunu = np.loadtxt(data_path_pmunu, delimiter=',').T
__e_gams_pmunug, __spec_pmunug = np.loadtxt(data_path_pmunug, delimiter=',').T


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double __interp_spec(double eng_gam, int bitflags):
    ret = 0.0
    if bitflags & 1:
        ret += np.interp(eng_gam, __e_gams_000, __spec_000)
    if bitflags & 2:
        ret += np.interp(eng_gam, __e_gams_penu, __spec_penu)
    if bitflags & 4:
        ret += np.interp(eng_gam, __e_gams_penug, __spec_penug)
    if bitflags & 8:
        ret += np.interp(eng_gam, __e_gams_pm0, __spec_pm0)
    if bitflags & 16:
        ret += np.interp(eng_gam, __e_gams_pm0g, __spec_pm0g)
    if bitflags & 32:
        ret += np.interp(eng_gam, __e_gams_pmunu, __spec_pmunu)
    if bitflags & 64:
        ret += np.interp(eng_gam, __e_gams_pmunug, __spec_pmunug)
    return ret

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double __integrand(double cl, double eng_gam, double eng_k, int mode):
    cdef double gamma_k = eng_k / MASS_K
    cdef double beta_k = sqrt(1.0 - (MASS_K / eng_k)**2)
    cdef double eng_gam_k_rf = eng_gam * gamma_k * (1.0 - beta_k * cl)

    cdef double ret_val
    cdef double pre_factor = 1.0 / (2.0 * gamma_k * (1.0 - beta_k * cl))

    return pre_factor * __interp_spec(eng_gam_k_rf, mode)


cdef double c_long_kaon_decay_spectrum_point(double eng_gam, double eng_k, int mode):
    if eng_k < MASS_K0:
        return 0.0
    return quad(__integrand, -1.0, 1.0, points=[-1.0, 1.0], args=(eng_gam, eng_k, mode), epsabs=1e-10, epsrel=1e-4)[0]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t,ndim=1] c_long_kaon_decay_spectrum_array(np.ndarray[np.float64_t, ndim=1] eng_gams, double eng_k, int mode):
  cdef int npts = eng_gams.shape[0]
  cdef np.ndarray[np.float64_t,ndim=1] spec = np.zeros_like(eng_gams)

  for i in range(npts):
      spec[i] = c_long_kaon_decay_spectrum_point(eng_gams[i], eng_k, mode)

  return spec


# ===================================================================
# ---- Python API ---------------------------------------------------
# ===================================================================

def long_kaon_decay_spectrum(egam, ek, modes=["000","penu","penug","pm0","pm0g","pmunu","pmunug"]):
    """
    Compute the photon spectrum dN/dE from the decay of a long kaon.

    Paramaters
    ----------
    egam: float or array-like
        Photon energy.
    ek: float 
        Energy of the kaon.
    modes: List[str]
        List of strings representing the modes to include. The availible modes are:
        "000","penu","penug","pm0","pm0g","pmunu" and "pmunug".
    """
    cdef int bitflags = 0 

    if "000" in modes:
        bitflags += 1
    if "penu" in modes:
        bitflags += 2
    if "penug" in modes:
        bitflags += 4
    if "pm0" in modes:
        bitflags += 8
    if "pm0g" in modes:
        bitflags += 16
    if "pmunu" in modes:
        bitflags += 32
    if "pmunug" in modes:
        bitflags += 64
    
    if bitflags == 0:
        raise ValueError("Invalid modes specified.") 


    if hasattr(egam, '__len__'):
        energies = np.array(egam)
        assert len(energies.shape) == 1, "Photon energies must be 0 or 1-dimensional."
        return c_long_kaon_decay_spectrum_array(energies, ek, bitflags)
    else:
        return c_long_kaon_decay_spectrum_point(egam, ek, bitflags)
