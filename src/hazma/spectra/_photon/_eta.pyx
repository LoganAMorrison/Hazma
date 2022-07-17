import cython

import numpy as np
cimport numpy as np
from scipy.integrate import quad
from libc.float cimport DBL_EPSILON

from hazma.spectra._photon.path import DATA_DIR
from hazma._utils.boost cimport boost_beta, boost_gamma, boost_delta_function
from hazma._utils.boost cimport boost_integrate_linear_interp

include "../../_utils/constants.pxd"

# ============================================================================
# ---- Data Loading ----------------------------------------------------------
# ============================================================================

# Format: energy, 1:'a a', 2:'pi0 pi0 pi0', 3:'pi pi pi0', 4:'pi pi a', 5:'mu mu'
eta_data = np.loadtxt(
    DATA_DIR.joinpath("eta_photon.csv"),
    delimiter=","
).T
eta_data_energies = eta_data[0]
eta_data_dnde = np.sum(eta_data[1:], axis=0)
eta_data_emin = eta_data_energies[0]
eta_data_emax = eta_data_energies[-1]

# ============================================================================
# ---- Charged Kaon ----------------------------------------------------------
# ============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dnde_photon_eta_rest_frame(double photon_energy):
    cdef double ret = 0.0

    if photon_energy > eta_data_emax:
        return ret

    if photon_energy < eta_data_emin:
        return eta_data_dnde[0] * eta_data_emin / photon_energy
    else:
        return np.interp(photon_energy, eta_data_energies, eta_data_dnde)


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef double integrand_eta(double photon_energy):
#     return dnde_photon_eta_rest_frame(photon_energy) / photon_energy


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef double dnde_photon_eta_point(double photon_energy, double eta_energy):
#     cdef double gamma
#     cdef double beta
#     cdef double pre
#     cdef double emin
#     cdef double emax
#     cdef double res

#     if eta_energy < MASS_ETA:
#         return 0.0

#     if eta_energy - MASS_ETA < DBL_EPSILON:
#         return dnde_photon_eta_rest_frame(photon_energy)

#     gamma = boost_gamma(eta_energy, MASS_ETA)
#     beta = boost_beta(eta_energy, MASS_ETA)
#     pre = 0.5 / (gamma * beta)
#     emin = gamma * photon_energy * (1.0 - beta)
#     emax = gamma * photon_energy * (1.0 + beta)
#     res = 0.0

#     res = pre * quad(integrand_eta, emin, emax, epsabs=1e-10, epsrel=1e-4)[0]
#     res += 2.0 * BR_ETA_TO_A_A * boost_delta_function(MASS_ETA / 2.0, photon_energy, 0.0, beta)
#     return res 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dnde_photon_eta_point(double photon_energy, double eta_energy):
    cdef double gamma
    cdef double beta
    cdef double res

    if eta_energy < MASS_ETA:
        return 0.0

    if eta_energy - MASS_ETA < DBL_EPSILON:
        return dnde_photon_eta_rest_frame(photon_energy)

    beta = boost_beta(eta_energy, MASS_ETA)
    gamma = boost_gamma(eta_energy, MASS_ETA)
    res = boost_integrate_linear_interp(photon_energy, beta, eta_data_energies, eta_data_dnde)
    res += 2.0 * BR_ETA_TO_A_A * boost_delta_function(MASS_ETA / 2.0, photon_energy, 0.0, beta)
    return res 


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[np.float64_t,ndim=1] dnde_photon_eta_array(double[:] photon_energy, double eta_energy):
    cdef int npts = photon_energy.shape[0]
    cdef np.ndarray[np.float64_t,ndim=1] spec = np.zeros_like(photon_energy)
    for i in range(npts):
        spec[i] = dnde_photon_eta_point(photon_energy[i], eta_energy)
    return spec


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def dnde_photon_eta(photon_energy, eta_energy):
    if hasattr(photon_energy, '__len__'):
        energies = np.array(photon_energy)
        assert len(energies.shape) == 1, "Photon energies must be 0 or 1-dimensional."
        return dnde_photon_eta_array(energies, eta_energy)
    else:
        return dnde_photon_eta_point(photon_energy, eta_energy)

