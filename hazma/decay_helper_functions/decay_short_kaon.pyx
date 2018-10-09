"""
TODO: Update documentation.
"""
from hazma.decay_helper_functions cimport decay_charged_pion
from hazma.decay_helper_functions cimport decay_neutral_pion
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

cdef double __interp_spec(double eng_gam, str mode):
    """
    Intepolation function for the short kaon data.

    Parameters
    ----------
    eng_gam : double
        Energy of the photon.
    mode : str {"total"}
        String specifying which decay mode to use.
    """
    if mode == "total":
        return np.interp(eng_gam, __e_gams_total, __spec_total)
    if mode == "00":
        return np.interp(eng_gam, __e_gams_00, __spec_00)
    if mode == "pm":
        return np.interp(eng_gam, __e_gams_pm, __spec_pm)
    if mode == "pmg":
        return np.interp(eng_gam, __e_gams_pmg, __spec_pmg)


@cython.cdivision(True)
cdef double __integrand(double cl, double eng_gam, double eng_k, str mode):
    """
    Integrand for K_S -> X, where X is a any final state.

    The X's used are "pi + pi", "pi0 + pi0" and "pi + pi + g".

    Parameters
    ----------
    cl: float
        Angle of photon w.r.t. charged kaon in lab frame.
    eng_gam: float
        Energy of photon in laboratory frame.
    eng_k: float
        Energy of kaon in laboratory frame.
    mode : str {"total"}
        String specifying which decay mode to use.
    """
    cdef double gamma_k = eng_k / MASS_K
    cdef double beta_k = sqrt(1.0 - (MASS_K / eng_k)**2)
    cdef double eng_gam_k_rf = eng_gam * gamma_k * (1.0 - beta_k * cl)

    cdef int i, j
    cdef double ret_val
    cdef double pre_factor \
        = 1.0 / (2.0 * gamma_k * (1.0 - beta_k * cl))

    return pre_factor * __interp_spec(eng_gam_k_rf, mode)


cdef double CSpectrumPoint(double eng_gam, double eng_k, str mode):
    """
    Returns the radiative spectrum value from charged kaon at
    a single gamma ray energy.

    Parameters
    ----------
    eng_gam : float
        Energy of photon is laboratory frame.
    eng_k : float
        Energy of charged kaon in laboratory frame.
    mode : str {"total"}
        String specifying which decay mode to use.
    """
    message = 'Energy of kaon cannot be less than the kaon mass. Returning 0.'
    if eng_k < MASS_K0:
        # raise warnings.warn(message, RuntimeWarning)
        return 0.0

    cdef double result = 0.0

    return quad(__integrand, -1.0, 1.0, points=[-1.0, 1.0], \
                  args=(eng_gam, eng_k, mode), epsabs=10**-10., \
                  epsrel=10**-4.)[0]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray CSpectrum(np.ndarray[np.float64_t, ndim=1] eng_gams,
                          double eng_k, str mode):
    """
    Returns the radiative spectrum dNde from charged kaon for a
    list of gamma ray energies.

    Parameters
    ----------
    eng_gams : numpy.ndarray
        List of energies of photon in laboratory frame.
    eng_k : float
        Energy of charged kaon in laboratory frame.
    mode : str {"total"}
        String specifying which decay mode to use.
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

    Parameters
    ----------
    eng_gam : float
        Energy of photon is laboratory frame.
    eng_k : float
        Energy of charged kaon in laboratory frame.
    mode : str {"total"}
        String specifying which decay mode to use.
    """
    return CSpectrumPoint(eng_gam, eng_k, mode)


@cython.boundscheck(False)
@cython.wraparound(False)
def Spectrum(np.ndarray[np.float64_t, ndim=1] eng_gams, double eng_k,
             str mode):
    """
    Returns the radiative spectrum dNde from charged kaon for a
    list of gamma ray energies.

    Parameters
    ----------
    eng_gams : numpy.ndarray
        List of energies of photon in laboratory frame.
    eng_k :
        Energy of charged kaon in laboratory frame.
    mode : str {"total"}
        String specifying which decay mode to use.
    """
    return CSpectrum(eng_gams, eng_k, mode)
