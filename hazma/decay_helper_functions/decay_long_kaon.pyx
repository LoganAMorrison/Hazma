from hazma.decay_helper_functions cimport decay_muon
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
    if mode == "000":
        return np.interp(eng_gam, __e_gams_000, __spec_000)
    if mode == "penu":
        return np.interp(eng_gam, __e_gams_penu, __spec_penu)
    if mode == "penug":
        return np.interp(eng_gam, __e_gams_penug, __spec_penug)
    if mode == "pm0":
        return np.interp(eng_gam, __e_gams_pm0, __spec_pm0)
    if mode == "pm0g":
        return np.interp(eng_gam, __e_gams_pm0g, __spec_pm0g)
    if mode == "pmunu":
        return np.interp(eng_gam, __e_gams_pmunu, __spec_pmunu)
    if mode == "pmunug":
        return np.interp(eng_gam, __e_gams_pmunug, __spec_pmunug)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double __integrand(double cl, double eng_gam, double eng_k, str mode):
    """
    Integrand for K -> X, where X is a three body final state. The X's
    used are
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

    cdef double ret_val
    cdef double pre_factor \
        = 1.0 / (2.0 * gamma_k * (1.0 - beta_k * cl))

    return pre_factor * __interp_spec(eng_gam_k_rf, mode)


cdef double CSpectrumPoint(double eng_gam, double eng_k, str mode):
    """
    Returns the radiative spectrum value from charged kaon at
    a single gamma ray energy.

    Keyword arguments::
        eng_gam: Energy of photon is laboratory frame.
        eng_k: Energy of charged kaon in laboratory frame.
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
def Spectrum(np.ndarray[np.float64_t, ndim=1] eng_gams, double eng_k, str mode):
    """
    Returns the radiative spectrum dNde from charged kaon for a
    list of gamma ray energies.

    Keyword arguments::
        eng_gams: List of energies of photon in laboratory frame.
        eng_k: Energy of charged kaon in laboratory frame.
    """
    return CSpectrum(eng_gams, eng_k, mode)
