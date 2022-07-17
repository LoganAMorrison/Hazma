"""
Module for computing the photon spectrum from radiative kaon decays.

Description:
    The charged kaon has many decay modes:

        k -> mu  + nu
        k -> pi  + pi0
        k -> pi  + pi  + pi
        k -> pi0 + e   + nu
        k -> pi0 + mu  + nu
        k -> pi  + pi0 + pi0

    For the the two-body final states, the sum of the decay spectra are computed
    given the known energies of the final state particles in the kaon's rest
    frame. The spectrum is then boosted into the lab frame.  For the three-body
    final state, the energies of the final state particles are computed using
    RAMBO. The spectra for each final state particle is computed are each point
    in phases space in the charged kaon's rest frame and then spectra are summed
    over. The spectra is then boosted into the lab frame.

    The charged kaon has many decay modes:

        kl    -> pi  + e   + nu
        kl    -> pi  + mu  + nu
        kl    -> pi0 + pi0  + pi0
        kl    -> pi  + pi  + pi0

    For the three-body final state, the energies of the final state particles
    are computed using RAMBO. The spectra for each final state particle is
    computed are each point in phases space in the charged kaon's rest frame and
    then spectra are summed over. The spectra is then boosted into the lab
    frame.

    The short kaon has many decay modes:

        ks -> pi + pi
        ks -> pi0 + pi0
        ks -> pi + pi + g

    For the the two-body final states, the sum of the decay spectra are
    computed given the known energies of the final state particles in the
    kaon's rest frame. The spectrum is then boosted into the lab frame.
"""
import os
import sys

import cython

import numpy as np
cimport numpy as np
from scipy.integrate import quad
from libc.math cimport sqrt
from libc.float cimport DBL_EPSILON

from hazma.spectra._photon.path import DATA_DIR
from hazma._utils.boost cimport boost_beta, boost_gamma, boost_delta_function
from hazma._utils.boost cimport boost_integrate_linear_interp

include "../../_utils/constants.pxd"

# ============================================================================
# ---- Data Loading ----------------------------------------------------------
# ============================================================================

# Format: energy, 1:mu_nu, 2:pi_pi0, 3:pi_pi_pi, 4:pi0_e_nu, 5:pi0_mu_nu, 
#                 6:pi_pi0_pi0, 7:e_nu
charged_kaon_data = np.loadtxt(
    DATA_DIR.joinpath("charged_kaon_photon.csv"),
    delimiter=","
).T
charged_kaon_data_energies = charged_kaon_data[0]
# charged_kaon_data_mu_nu = charged_kaon_data[1]
# charged_kaon_data_pi_pi0 = charged_kaon_data[2]
# charged_kaon_data_pi_pi_pi = charged_kaon_data[3]
# charged_kaon_data_pi0_e_nu = charged_kaon_data[4]
# charged_kaon_data_pi0_mu_nu = charged_kaon_data[5]
# charged_kaon_data_pi_pi0_pi0 = charged_kaon_data[6]
# charged_kaon_data_e_nu = charged_kaon_data[7]
charged_kaon_data_dnde = np.sum(charged_kaon_data[1:], axis=0)
charged_kaon_data_emin = charged_kaon_data_energies[0]
charged_kaon_data_emax = charged_kaon_data_energies[-1]

# format: energy, pi_pi, pi0_pi0
# missing: a_a
short_kaon_data = np.loadtxt(
    DATA_DIR.joinpath("short_kaon_photon.csv"),
    delimiter=","
).T
short_kaon_data_energies = short_kaon_data[0]
# short_kaon_data_pi_pi = short_kaon_data[1]
# short_kaon_data_pi0_pi0 = short_kaon_data[2]
short_kaon_data_dnde = np.sum(short_kaon_data[1:], axis=0)
short_kaon_data_emin = short_kaon_data_energies[0]
short_kaon_data_emax = short_kaon_data_energies[-1]

# format: energy, pi0_pi0_pi0, pi_pi_pi0, pi_e_nu, pi_mu_nu, pi_pi, pi0_pi0
# missing: a_a
long_kaon_data = np.loadtxt(
    DATA_DIR.joinpath("long_kaon_photon.csv"),
    delimiter=","
).T
long_kaon_data_energies = long_kaon_data[0]
# long_kaon_data_pi0_pi0_pi0 = long_kaon_data[1]
# long_kaon_data_pi_pi_pi0 = long_kaon_data[2]
# long_kaon_data_pi_e_nu = long_kaon_data[3]
# long_kaon_data_pi_mu_nu = long_kaon_data[4]
# long_kaon_data_pi_pi = long_kaon_data[5]
# long_kaon_data_pi0_pi0 = long_kaon_data[6]
long_kaon_data_dnde = np.sum(long_kaon_data[1:], axis=0)
long_kaon_data_emin = long_kaon_data_energies[0]
long_kaon_data_emax = long_kaon_data_energies[-1]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double kaon_interp_spec(double photon_energy, double[:] energies, double[:] dnde, double emin, double emax):
    cdef double ret = 0.0

    if photon_energy > emax:
        return ret

    if photon_energy < emin:
        return dnde[0] * emin / photon_energy
    else:
        return np.interp(photon_energy, energies, dnde)

# ============================================================================
# ---- Charged Kaon ----------------------------------------------------------
# ============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dnde_photon_charged_kaon_rest_frame(double photon_energy):
    return kaon_interp_spec(
        photon_energy,
        charged_kaon_data_energies,
        charged_kaon_data_dnde,
        charged_kaon_data_emin,
        charged_kaon_data_emax,
    )

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef double integrand_charged_kaon(double photon_energy):
#     return dnde_photon_charged_kaon_rest_frame(photon_energy) / photon_energy


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef double dnde_photon_charged_kaon_point(double photon_energy, double kaon_energy):
#     cdef double gamma
#     cdef double beta
#     cdef double pre
#     cdef double emin
#     cdef double emax
#     cdef double res

#     if kaon_energy < MASS_K:
#         return 0.0

#     if kaon_energy - MASS_K < DBL_EPSILON:
#         return dnde_photon_charged_kaon_rest_frame(photon_energy)

#     gamma = boost_gamma(kaon_energy, MASS_K)
#     beta = boost_beta(kaon_energy, MASS_K)
#     pre = 0.5 / (gamma * beta)
#     emin = gamma * photon_energy * (1.0 - beta)
#     emax = gamma * photon_energy * (1.0 + beta)
#     res = 0.0

#     res = quad(integrand_charged_kaon, emin, emax, epsabs=1e-10, epsrel=1e-4)[0]
#     return pre * res 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dnde_photon_charged_kaon_point(double photon_energy, double kaon_energy):
    cdef double gamma
    cdef double beta
    cdef double res

    if kaon_energy < MASS_K:
        return 0.0

    if kaon_energy - MASS_K < DBL_EPSILON:
        return dnde_photon_charged_kaon_rest_frame(photon_energy)

    gamma = boost_gamma(kaon_energy, MASS_K)
    beta = boost_beta(kaon_energy, MASS_K)
    res = boost_integrate_linear_interp(photon_energy, beta, charged_kaon_data_energies, charged_kaon_data_dnde)
    return res 


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[np.float64_t,ndim=1] dnde_photon_charged_kaon_array(double[:] photon_energy, double kaon_energy):
    cdef int npts = photon_energy.shape[0]
    cdef np.ndarray[np.float64_t,ndim=1] spec = np.zeros_like(photon_energy)
    for i in range(npts):
        spec[i] = dnde_photon_charged_kaon_point(photon_energy[i], kaon_energy)
    return spec


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def dnde_photon_charged_kaon(photon_energy, kaon_energy):
    """
    Compute the photon spectrum dN/dE from the decay of a short kaon.
    Paramaters
    ----------
    egam: float or array-like
        Photon energy.
    ek: float 
        Energy of the kaon.
    """
    if hasattr(photon_energy, '__len__'):
        energies = np.array(photon_energy)
        assert len(energies.shape) == 1, "Photon energies must be 0 or 1-dimensional."
        return dnde_photon_charged_kaon_array(energies, kaon_energy)
    else:
        return dnde_photon_charged_kaon_point(photon_energy, kaon_energy)


# ============================================================================
# ---- Long Kaon -------------------------------------------------------------
# ============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dnde_photon_long_kaon_rest_frame(double photon_energy):
    return kaon_interp_spec(
        photon_energy,
        long_kaon_data_energies,
        long_kaon_data_dnde,
        long_kaon_data_emin,
        long_kaon_data_emax,
    )


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef double integrand_long_kaon(double photon_energy):
#     return dnde_photon_long_kaon_rest_frame(photon_energy) / photon_energy


# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef double dnde_photon_long_kaon_point(double photon_energy, double kaon_energy):
#     cdef double gamma
#     cdef double beta
#     cdef double pre
#     cdef double emin
#     cdef double emax
#     cdef double res

#     if kaon_energy < MASS_K0:
#         return 0.0

#     if kaon_energy - MASS_K0 < DBL_EPSILON:
#         return dnde_photon_long_kaon_rest_frame(photon_energy)

#     gamma = boost_gamma(kaon_energy, MASS_K0)
#     beta = boost_beta(kaon_energy, MASS_K0)
#     pre = 0.5 / (gamma * beta)
#     emin = gamma * photon_energy * (1.0 - beta)
#     emax = gamma * photon_energy * (1.0 + beta)
#     res = 0.0

#     res = pre * quad(integrand_long_kaon, emin, emax, epsabs=1e-10, epsrel=1e-4)[0]
#     res += 2 * BR_KL_TO_A_A * boost_delta_function(MASS_K0 / 2.0, photon_energy, 0.0, beta)
#     return res 

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double dnde_photon_long_kaon_point(double photon_energy, double kaon_energy):
    cdef double gamma
    cdef double beta
    cdef double res

    if kaon_energy < MASS_K0:
        return 0.0

    if kaon_energy - MASS_K0 < DBL_EPSILON:
        return dnde_photon_long_kaon_rest_frame(photon_energy)

    gamma = boost_gamma(kaon_energy, MASS_K0)
    beta = boost_beta(kaon_energy, MASS_K0)
    res = boost_integrate_linear_interp(photon_energy, beta, long_kaon_data_energies, long_kaon_data_dnde)
    res += 2 * BR_KL_TO_A_A * boost_delta_function(MASS_K0 / 2.0, photon_energy, 0.0, beta)
    return res 


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t,ndim=1] dnde_photon_long_kaon_array(double[:] photon_energy, double kaon_energy):
  cdef int npts = photon_energy.shape[0]
  cdef np.ndarray[np.float64_t,ndim=1] spec = np.zeros_like(photon_energy)

  for i in range(npts):
      spec[i] = dnde_photon_long_kaon_point(photon_energy[i], kaon_energy)

  return spec


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def dnde_photon_long_kaon(photon_energy, kaon_energy):
    """
    Compute the photon spectrum dN/dE from the decay of a long kaon.
    Paramaters
    ----------
    egam: float or array-like
        Photon energy.
    ek: float 
        Energy of the kaon.
    """

    if hasattr(photon_energy, '__len__'):
        energies = np.array(photon_energy)
        assert len(energies.shape) == 1, "Photon energies must be 0 or 1-dimensional."
        return dnde_photon_long_kaon_array(energies, kaon_energy)
    else:
        return dnde_photon_long_kaon_point(photon_energy, kaon_energy)


# ============================================================================
# ---- Short Kaon ------------------------------------------------------------
# ============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dnde_photon_short_kaon_rest_frame(double photon_energy):
    return kaon_interp_spec(
        photon_energy,
        short_kaon_data_energies,
        short_kaon_data_dnde,
        short_kaon_data_emin,
        short_kaon_data_emax,
    )

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef double integrand_short_kaon(double photon_energy):
#     return dnde_photon_short_kaon_rest_frame(photon_energy) / photon_energy


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef double dnde_photon_short_kaon_point(double photon_energy, double kaon_energy):
#     cdef double gamma
#     cdef double beta
#     cdef double pre
#     cdef double emin
#     cdef double emax
#     cdef double res

#     if kaon_energy < MASS_K0:
#         return 0.0

#     if kaon_energy - MASS_K0 < DBL_EPSILON:
#         return dnde_photon_short_kaon_rest_frame(photon_energy)

#     gamma = boost_gamma(kaon_energy, MASS_K0)
#     beta = boost_beta(kaon_energy, MASS_K0)
#     pre = 0.5 / (gamma * beta)
#     emin = gamma * photon_energy * (1.0 - beta)
#     emax = gamma * photon_energy * (1.0 + beta)
#     res = 0.0

#     res = pre * quad(integrand_short_kaon, emin, emax, epsabs=1e-10, epsrel=1e-4)[0]
#     res += 2 * BR_KS_TO_A_A * boost_delta_function(MASS_K0 / 2.0, photon_energy, 0.0, beta)

#     return res 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dnde_photon_short_kaon_point(double photon_energy, double kaon_energy):
    cdef double gamma
    cdef double beta
    cdef double res

    if kaon_energy < MASS_K0:
        return 0.0

    if kaon_energy - MASS_K0 < DBL_EPSILON:
        return dnde_photon_short_kaon_rest_frame(photon_energy)

    gamma = boost_gamma(kaon_energy, MASS_K0)
    beta = boost_beta(kaon_energy, MASS_K0)
    res = boost_integrate_linear_interp(photon_energy, beta, short_kaon_data_energies, short_kaon_data_dnde)
    res += 2 * BR_KS_TO_A_A * boost_delta_function(MASS_K0 / 2.0, photon_energy, 0.0, beta)
    return res 


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[np.float64_t,ndim=1] dnde_photon_short_kaon_array(double[:] photon_energy, double kaon_energy):
    cdef int npts = photon_energy.shape[0]
    cdef np.ndarray[np.float64_t,ndim=1] spec = np.zeros_like(photon_energy)
    for i in range(npts):
        spec[i] = dnde_photon_short_kaon_point(photon_energy[i], kaon_energy)
    return spec


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def dnde_photon_short_kaon(photon_energy, kaon_energy):
    """
    Compute the photon spectrum dN/dE from the decay of a short kaon.
    Paramaters
    ----------
    egam: float or array-like
        Photon energy.
    ek: float 
        Energy of the kaon.
    """

    if hasattr(photon_energy, '__len__'):
        energies = np.array(photon_energy)
        assert len(energies.shape) == 1, "Photon energies must be 0 or 1-dimensional."
        return dnde_photon_short_kaon_array(energies, kaon_energy)
    else:
        return dnde_photon_short_kaon_point(photon_energy, kaon_energy)
