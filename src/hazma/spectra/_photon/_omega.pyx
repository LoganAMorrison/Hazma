import os
import sys

import cython

import numpy as np
cimport numpy as np
# from scipy.integrate import quad
from libc.math cimport sqrt
from libc.float cimport DBL_EPSILON

from hazma.spectra._photon.path import DATA_DIR
from hazma._utils.boost cimport boost_beta, boost_gamma, boost_delta_function
from hazma._utils.boost cimport boost_integrate_linear_interp

include "../../_utils/constants.pxd"

# ============================================================================
# ---- Data Loading ----------------------------------------------------------
# ============================================================================

# Format: energy, pi_pi_pi0, pi0_a, pi_pi, eta_a, e_e, mu_mu
omega_data = np.loadtxt(
    DATA_DIR.joinpath("omega_photon.csv"),
    delimiter=","
).T
omega_data_energies = omega_data[0]
omega_data_dnde = np.sum(omega_data[1:], axis=0)
omega_data_emin = omega_data_energies[0]
omega_data_emax = omega_data_energies[-1]

# ============================================================================
# ---- Charged Kaon ----------------------------------------------------------
# ============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dnde_photon_omega_rest_frame(double photon_energy):
    cdef double ret = 0.0

    if photon_energy > omega_data_emax:
        return ret

    if photon_energy < omega_data_emin:
        return omega_data_dnde[0] * omega_data_emin / photon_energy
    else:
        return np.interp(photon_energy, omega_data_energies, omega_data_dnde)


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef double integrand_omega(double photon_energy):
#     return dnde_photon_omega_rest_frame(photon_energy) / photon_energy


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef double dnde_photon_omega_point(double photon_energy, double omega_energy):
#     cdef double gamma
#     cdef double beta
#     cdef double pre
#     cdef double emin
#     cdef double emax
#     cdef double res
#     cdef double eng_a_w_to_pi0_a

#     if omega_energy < MASS_OMEGA:
#         return 0.0

#     if omega_energy - MASS_OMEGA < DBL_EPSILON:
#         return dnde_photon_omega_rest_frame(photon_energy)

#     gamma = boost_gamma(omega_energy, MASS_OMEGA)
#     beta = boost_beta(omega_energy, MASS_OMEGA)
#     pre = 0.5 / (gamma * beta)
#     emin = gamma * photon_energy * (1.0 - beta)
#     emax = gamma * photon_energy * (1.0 + beta)
#     res = 0.0

#     res = quad(integrand_omega, emin, emax, epsabs=1e-10, epsrel=1e-4)[0]

#     eng_a_w_to_pi0_a = (MASS_OMEGA**2 - MASS_PI0**2) / (2 * MASS_OMEGA)
#     eng_a_w_to_eta_a = (MASS_OMEGA**2 - MASS_ETA**2) / (2 * MASS_OMEGA)
#     res +=  BR_OMEGA_TO_PI0_A * boost_delta_function(eng_a_w_to_pi0_a, photon_energy, 0.0, beta)
#     res +=  BR_OMEGA_TO_ETA_A * boost_delta_function(eng_a_w_to_eta_a, photon_energy, 0.0, beta)

#     return pre * res 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dnde_photon_omega_point(double photon_energy, double omega_energy):
    cdef double gamma
    cdef double beta
    cdef double res
    cdef double eng_a_w_to_pi0_a

    if omega_energy < MASS_OMEGA:
        return 0.0

    if omega_energy - MASS_OMEGA < DBL_EPSILON:
        return dnde_photon_omega_rest_frame(photon_energy)

    gamma = boost_gamma(omega_energy, MASS_OMEGA)
    beta = boost_beta(omega_energy, MASS_OMEGA)
    res = boost_integrate_linear_interp(photon_energy, beta, omega_data_energies, omega_data_dnde)

    eng_a_w_to_pi0_a = (MASS_OMEGA**2 - MASS_PI0**2) / (2 * MASS_OMEGA)
    eng_a_w_to_eta_a = (MASS_OMEGA**2 - MASS_ETA**2) / (2 * MASS_OMEGA)
    res += BR_OMEGA_TO_PI0_A * boost_delta_function(eng_a_w_to_pi0_a, photon_energy, 0.0, beta)
    res += BR_OMEGA_TO_ETA_A * boost_delta_function(eng_a_w_to_eta_a, photon_energy, 0.0, beta)

    return res 


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[np.float64_t,ndim=1] dnde_photon_omega_array(double[:] photon_energy, double omega_energy):
    cdef int npts = photon_energy.shape[0]
    cdef np.ndarray[np.float64_t,ndim=1] spec = np.zeros_like(photon_energy)
    for i in range(npts):
        spec[i] = dnde_photon_omega_point(photon_energy[i], omega_energy)
    return spec


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def dnde_photon_omega(photon_energy, omega_energy):
    if hasattr(photon_energy, '__len__'):
        energies = np.array(photon_energy)
        assert len(energies.shape) == 1, "Photon energies must be 0 or 1-dimensional."
        return dnde_photon_omega_array(energies, omega_energy)
    else:
        return dnde_photon_omega_point(photon_energy, omega_energy)

