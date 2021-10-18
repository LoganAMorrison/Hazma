"""
Module for computing the photon spectrum from radiative kaon decay.

Description:
    The charged kaon has many decay modes:

        k -> mu  + nu
        k -> pi  + pi0
        k -> pi  + pi  + pi
        k -> pi0 + e   + nu
        k -> pi0 + mu  + nu
        k -> pi  + pi0 + pi0

    For the the two-body final states, the sum of the decay spectra are
    computed given the known energies of the final state particles in the
    kaon's rest frame. The spectrum is then boosted into the lab frame.

    For the three-body final state, the energies of the final state
    particles are computed using RAMBO. The spectra for each final state
    particle is computed are each point in phases space in the charged kaon's rest frame and then spectra are summed over. The spectra is then boosted into the lab frame.
"""

cimport decay_muon
cimport decay_charged_pion
cimport decay_neutral_pion
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
                               "ckaon",
                               "charged_kaon_interp_total.dat")
data_path_0enu = os.path.join(get_dir_path(),
                              "interpolation_data",
                              "ckaon",
                              "charged_kaon_interp_0enu.dat")
data_path_0munu = os.path.join(get_dir_path(),
                               "interpolation_data",
                               "ckaon",
                               "charged_kaon_interp_0munu.dat")
data_path_00p = os.path.join(get_dir_path(),
                             "interpolation_data",
                             "ckaon",
                             "charged_kaon_interp_00p.dat")
data_path_mmug = os.path.join(get_dir_path(),
                              "interpolation_data",
                              "ckaon",
                              "charged_kaon_interp_mmug.dat")
data_path_munu = os.path.join(get_dir_path(),
                              "interpolation_data",
                              "ckaon",
                              "charged_kaon_interp_munu.dat")
data_path_p0 = os.path.join(get_dir_path(),
                            "interpolation_data",
                            "ckaon",
                            "charged_kaon_interp_p0.dat")
data_path_p0g = os.path.join(get_dir_path(),
                             "interpolation_data",
                             "ckaon",
                             "charged_kaon_interp_p0g.dat")
data_path_ppm = os.path.join(get_dir_path(),
                             "interpolation_data",
                             "ckaon",
                             "charged_kaon_interp_ppm.dat")

__e_gams_total, __spec_total = np.loadtxt(data_path_total, delimiter=',').T
__e_gams_0enu, __spec_0enu = np.loadtxt(data_path_0enu, delimiter=',').T
__e_gams_0munu, __spec_0munu = np.loadtxt(data_path_0munu, delimiter=',').T
__e_gams_00p, __spec_00p = np.loadtxt(data_path_00p, delimiter=',').T
__e_gams_mmug, __spec_mmug = np.loadtxt(data_path_mmug, delimiter=',').T
__e_gams_munu, __spec_munu = np.loadtxt(data_path_munu, delimiter=',').T
__e_gams_p0, __spec_p0 = np.loadtxt(data_path_p0, delimiter=',').T
__e_gams_p0g, __spec_p0g = np.loadtxt(data_path_p0g, delimiter=',').T
__e_gams_ppm, __spec_ppm = np.loadtxt(data_path_ppm, delimiter=',').T


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double __interp_spec(double eng_gam, int bitflags):
    cdef double ret = 0.0
    if bitflags & 1:
        ret += np.interp(eng_gam, __e_gams_0enu, __spec_0enu)
    if bitflags & 2:
        ret += np.interp(eng_gam, __e_gams_0munu, __spec_0munu)
    if bitflags & 4:
        ret += np.interp(eng_gam, __e_gams_00p, __spec_00p)
    if bitflags & 8:
        ret += np.interp(eng_gam, __e_gams_mmug, __spec_mmug)
    if bitflags & 16:
        ret += np.interp(eng_gam, __e_gams_munu, __spec_munu)
    if bitflags & 32:
        ret += np.interp(eng_gam, __e_gams_p0, __spec_p0)
    if bitflags & 64:
        ret += np.interp(eng_gam, __e_gams_p0g, __spec_p0g)
    if bitflags & 128:
        ret += np.interp(eng_gam, __e_gams_ppm, __spec_ppm)
    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double __integrand(double cl, double eng_gam, double eng_k, int mode):
    cdef double gamma_k = eng_k / MASS_K
    cdef double beta_k = sqrt(1.0 - (MASS_K / eng_k)**2)
    cdef double eng_gam_k_rf = eng_gam * gamma_k * (1.0 - beta_k * cl)
    cdef double pre_factor = 1.0 / (2.0 * gamma_k * (1.0 - beta_k * cl))
    return  __interp_spec(eng_gam_k_rf, mode)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double c_charged_kaon_decay_spectrum_point(double eng_gam, double eng_k, int mode):
    if eng_k < MASS_K:
        return 0.0
    return quad(__integrand, -1.0, 1.0, points=[-1.0, 1.0], args=(eng_gam, eng_k, mode), epsabs=1e-10, epsrel=1e-4)[0]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[np.float64_t,ndim=1] c_charged_kaon_decay_spectrum_array(np.ndarray[np.float64_t,ndim=1] eng_gams, double eng_k, int mode):
    cdef int npts = eng_gams.shape[0]
    cdef np.ndarray[np.float64_t,ndim=1] spec = np.zeros_like(eng_gams)
    for i in range(npts):
        spec[i] = c_charged_kaon_decay_spectrum_point(eng_gams[i], eng_k, mode)
    return spec


# ===================================================================
# ---- Python API ---------------------------------------------------
# ===================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def charged_kaon_decay_spectrum(egam, ek, modes=["0enu", "0munu", "00p","mmug","munu","p0","p0g","ppm"]):
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
        "0enu", "0munu", "00p","mmug","munu","p0","p0g" and "ppm".
    """
    cdef int bitflags = 0 

    if "0enu" in modes:
        bitflags += 1
    if "0munu" in modes:
        bitflags += 2
    if "00p" in modes:
        bitflags += 4
    if "mmug" in modes:
        bitflags += 8
    if "munu" in modes:
        bitflags += 16
    if "p0" in modes:
        bitflags += 32
    if "p0g" in modes:
        bitflags += 64
    if "ppm" in modes:
        bitflags += 128
    
    if bitflags == 0:
        raise ValueError("Invalid modes specified.") 

    if hasattr(egam, '__len__'):
        energies = np.array(egam)
        assert len(energies.shape) == 1, "Photon energies must be 0 or 1-dimensional."
        return c_charged_kaon_decay_spectrum_array(energies, ek, bitflags)
    else:
        return c_charged_kaon_decay_spectrum_point(egam, ek, bitflags)


