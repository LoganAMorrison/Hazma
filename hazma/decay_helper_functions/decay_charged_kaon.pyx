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
include "parameters.pxd"
import warnings

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

cdef double __interp_spec(double eng_gam, str mode):
    """
    Intepolation function for the charged kaon data.

    Parameters
    ----------
    eng_gam : double
        Energy of the photon.
    mode : str {"total"}
        Optional. The decay mode to use.
    """
    if mode == "total":
        return np.interp(eng_gam, __e_gams_total, __spec_total)
    if mode == "0enu":
        return np.interp(eng_gam, __e_gams_0enu, __spec_0enu)
    if mode == "0munu":
        return np.interp(eng_gam, __e_gams_0munu, __spec_0munu)
    if mode == "00p":
        return np.interp(eng_gam, __e_gams_00p, __spec_00p)
    if mode == "mmug":
        return np.interp(eng_gam, __e_gams_mmug, __spec_mmug)
    if mode == "munu":
        return np.interp(eng_gam, __e_gams_munu, __spec_munu)
    if mode == "p0":
        return np.interp(eng_gam, __e_gams_p0, __spec_p0)
    if mode == "p0g":
        return np.interp(eng_gam, __e_gams_p0g, __spec_p0g)
    if mode == "ppm":
        return np.interp(eng_gam, __e_gams_ppm, __spec_ppm)



@cython.cdivision(True)
cdef double __integrand(double cl, double eng_gam, double eng_k, str mode):
    """
    Integrand for K -> X, where X is a any final state. The X's
    used are
        mu + nu
        pi  + pi0
        pi + pi + pi
        pi0 + mu + nu.
    When the ChargedKaon object is instatiated, the energies of the FSP are
    computed using RAMBO and energy distributions are formed. All the
    energies from the energy distributions are summed over against their
    weights.

    Keyword arguments::
        cl: Angle of photon w.r.t. charged kaon in lab frame.
        eng_gam: Energy of photon in laboratory frame.
        eng_k: Energy of kaon in laboratory frame.
    """
    cdef double gamma_k = eng_k / MASS_K
    cdef double beta_k = sqrt(1.0 - (MASS_K / eng_k)**2)
    cdef double eng_gam_k_rf = eng_gam * gamma_k * (1.0 - beta_k * cl)

    cdef double pre_factor \
        = 1.0 / (2.0 * gamma_k * (1.0 - beta_k * cl))

    return  __interp_spec(eng_gam_k_rf, mode)


cdef double CSpectrumPoint(double eng_gam, double eng_k, str mode):
    """
    Returns the radiative spectrum value from charged kaon at
    a single gamma ray energy.

    Keyword arguments::
        eng_gam: Energy of photon is laboratory frame.
        eng_k: Energy of charged kaon in laboratory frame.
    """
    message = 'Energy of kaon cannot be less than the kaon mass. Returning 0.'
    if eng_k < MASS_K:
        # raise warnings.warn(message, RuntimeWarning)
        return 0.0

    cdef double result = 0.0

    result = quad(__integrand, -1.0, 1.0, points=[-1.0, 1.0], \
                  args=(eng_gam, eng_k, mode), epsabs=10**-10., \
                  epsrel=10**-4.)[0]

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray CSpectrum(np.ndarray[np.float64_t, ndim=1] eng_gams,
                          double eng_k, str mode):
    """
    Returns the radiative spectrum dNde from charged kaon for a
    list of gamma ray energies.

    Keyword arguments::
        eng_gams: List of energies of photon in laboratory frame.
        eng_k: Energy of charged kaon in laboratory frame.
    """

    cdef int numpts = len(eng_gams)

    cdef np.ndarray spec = np.zeros(numpts, dtype=np.float64)

    cdef int i = 0

    for i in range(numpts):
        spec[i] = CSpectrumPoint(eng_gams[i], eng_k, mode)

    return spec

def SpectrumPoint(double eng_gam, double eng_k, str mode):
    """
    Returns the radiative spectrum value from charged kaon at
    a single gamma ray energy.

    Keyword arguments::
        eng_gam: Energy of photon is laboratory frame.
        eng_k: Energy of charged kaon in laboratory frame.
    """

    return CSpectrumPoint(eng_gam, eng_k, mode)

@cython.boundscheck(False)
@cython.wraparound(False)
def Spectrum(np.ndarray[np.float64_t, ndim=1] eng_gams, double eng_k,
             str mode):
    """
    Returns the radiative spectrum dNde from charged kaon for a
    list of gamma ray energies.

    Keyword arguments::
        eng_gams: List of energies of photon in laboratory frame.
        eng_k: Energy of charged kaon in laboratory frame.
    """
    return CSpectrum(eng_gams, eng_k, mode)
