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

phi_data = np.loadtxt(
    DATA_DIR.joinpath("phi_photon.csv"),
    delimiter=","
).T
phi_data_energies = phi_data[0]
phi_data_dnde = np.sum(phi_data[1:], axis=0)
phi_data_emin = phi_data_energies[0]
phi_data_emax = phi_data_energies[-1]

# ============================================================================
# ---- Charged Kaon ----------------------------------------------------------
# ============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dnde_photon_phi_rest_frame(double photon_energy):
    cdef double ret = 0.0

    if photon_energy > phi_data_emax:
        return ret

    if photon_energy < phi_data_emin:
        return phi_data_dnde[0] * phi_data_emin / photon_energy
    else:
        return np.interp(photon_energy, phi_data_energies, phi_data_dnde)


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef double integrand_phi(double photon_energy):
#     return dnde_photon_phi_rest_frame(photon_energy) / photon_energy


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef double dnde_photon_phi_point(double photon_energy, double phi_energy):
#     cdef double gamma
#     cdef double bphi
#     cdef double pre
#     cdef double emin
#     cdef double emax
#     cdef double res
#     cdef double eng_eta

#     if phi_energy < MASS_PHI:
#         return 0.0

#     if phi_energy - MASS_PHI < DBL_EPSILON:
#         return dnde_photon_phi_rest_frame(photon_energy)

#     gamma = boost_gamma(phi_energy, MASS_PHI)
#     beta = boost_beta(phi_energy, MASS_PHI)
#     pre = 0.5 / (gamma * beta)
#     emin = gamma * photon_energy * (1.0 - beta)
#     emax = gamma * photon_energy * (1.0 + beta)
#     res = 0.0

#     res = quad(integrand_phi, emin, emax, epsabs=1e-10, epsrel=1e-4)[0]

#     eng_eta = (MASS_PHI**2 + MASS_ETA**2) / (2 * MASS_PHI)
#     res += BR_PHI_TO_ETA_A * boost_delta_function(eng_eta, photon_energy, 0.0, beta)
#     eng_eta = (MASS_PHI**2 + MASS_ETAP**2) / (2 * MASS_PHI)
#     res += BR_PHI_TO_ETAP_A * boost_delta_function(eng_eta, photon_energy, 0.0, beta)

#     return pre * res 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dnde_photon_phi_point(double photon_energy, double phi_energy):
    cdef double gamma
    cdef double bphi
    cdef double res
    cdef double eng_eta

    if phi_energy < MASS_PHI:
        return 0.0

    if phi_energy - MASS_PHI < DBL_EPSILON:
        return dnde_photon_phi_rest_frame(photon_energy)

    gamma = boost_gamma(phi_energy, MASS_PHI)
    beta = boost_beta(phi_energy, MASS_PHI)
    
    res = boost_integrate_linear_interp(photon_energy, beta, phi_data_energies, phi_data_dnde)

    eng_eta = (MASS_PHI**2 + MASS_ETA**2) / (2 * MASS_PHI)
    res += BR_PHI_TO_ETA_A * boost_delta_function(eng_eta, photon_energy, 0.0, beta)
    eng_eta = (MASS_PHI**2 + MASS_ETAP**2) / (2 * MASS_PHI)
    res += BR_PHI_TO_ETAP_A * boost_delta_function(eng_eta, photon_energy, 0.0, beta)

    return res 


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[np.float64_t,ndim=1] dnde_photon_phi_array(double[:] photon_energy, double phi_energy):
    cdef int npts = photon_energy.shape[0]
    cdef np.ndarray[np.float64_t,ndim=1] spec = np.zeros_like(photon_energy)
    for i in range(npts):
        spec[i] = dnde_photon_phi_point(photon_energy[i], phi_energy)
    return spec


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def dnde_photon_phi(photon_energy, phi_energy):
    if hasattr(photon_energy, '__len__'):
        energies = np.array(photon_energy)
        assert len(energies.shape) == 1, "Photon energies must be 0 or 1-dimensional."
        return dnde_photon_phi_array(energies, phi_energy)
    else:
        return dnde_photon_phi_point(photon_energy, phi_energy)

