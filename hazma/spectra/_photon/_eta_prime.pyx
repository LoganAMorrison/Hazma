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

include "../../_utils/constants.pxd"

# ============================================================================
# ---- Data Loading ----------------------------------------------------------
# ============================================================================

# Format: energy, pi_pi_eta, rho0_a, pi0_pi0_eta, omega_a, a_a, pi_rho
eta_prime_data = np.loadtxt(
    DATA_DIR.joinpath("eta_prime_photon.csv"),
    delimiter=","
).T
eta_prime_data_energies = eta_prime_data[0]
eta_prime_data_dnde = np.sum(eta_prime_data[1:], axis=0)
eta_prime_data_emin = eta_prime_data_energies[0]
eta_prime_data_emax = eta_prime_data_energies[-1]

# ============================================================================
# ---- Charged Kaon ----------------------------------------------------------
# ============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dnde_photon_eta_prime_rest_frame(double photon_energy):
    cdef double ret = 0.0

    if photon_energy > eta_prime_data_emax:
        return ret

    if photon_energy < eta_prime_data_emin:
        return eta_prime_data_dnde[0] * eta_prime_data_emin / photon_energy
    else:
        return np.interp(photon_energy, eta_prime_data_energies, eta_prime_data_dnde)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double integrand_eta_prime(double photon_energy):
    return dnde_photon_eta_prime_rest_frame(photon_energy) / photon_energy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dnde_photon_eta_prime_point(double photon_energy, double eta_prime_energy):
    cdef double gamma
    cdef double beta
    cdef double pre
    cdef double emin
    cdef double emax
    cdef double res
    cdef double eng_a_w_to_pi0_a

    if eta_prime_energy < MASS_ETAP:
        return 0.0

    if eta_prime_energy - MASS_ETAP < DBL_EPSILON:
        return dnde_photon_eta_prime_rest_frame(photon_energy)

    gamma = boost_gamma(eta_prime_energy, MASS_ETAP)
    beta = boost_beta(eta_prime_energy, MASS_ETAP)
    pre = 0.5 / (gamma * beta)
    emin = gamma * photon_energy * (1.0 - beta)
    emax = gamma * photon_energy * (1.0 + beta)
    res = 0.0

    res = quad(integrand_eta_prime, emin, emax, epsabs=1e-10, epsrel=1e-4)[0]

    res +=  BR_ETAP_TO_A_A * boost_delta_function(MASS_ETAP / 2.0, photon_energy, 0.0, beta)

    return pre * res 


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[np.float64_t,ndim=1] dnde_photon_eta_prime_array(double[:] photon_energy, double eta_prime_energy):
    cdef int npts = photon_energy.shape[0]
    cdef np.ndarray[np.float64_t,ndim=1] spec = np.zeros_like(photon_energy)
    for i in range(npts):
        spec[i] = dnde_photon_eta_prime_point(photon_energy[i], eta_prime_energy)
    return spec


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def dnde_photon_eta_prime(photon_energy, eta_prime_energy):
    if hasattr(photon_energy, '__len__'):
        energies = np.array(photon_energy)
        assert len(energies.shape) == 1, "Photon energies must be 0 or 1-dimensional."
        return dnde_photon_eta_prime_array(energies, eta_prime_energy)
    else:
        return dnde_photon_eta_prime_point(photon_energy, eta_prime_energy)

