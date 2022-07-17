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

charged_kaon_data = np.loadtxt(
    DATA_DIR.joinpath("charged_kaon_positron.csv"),
    delimiter=","
).T
charged_kaon_data_energies = charged_kaon_data[0]
charged_kaon_data_dnde = np.sum(charged_kaon_data[1:], axis=0)
charged_kaon_data_emin = charged_kaon_data_energies[0]
charged_kaon_data_emax = charged_kaon_data_energies[-1]


short_kaon_data = np.loadtxt(
    DATA_DIR.joinpath("short_kaon_positron.csv"),
    delimiter=","
).T
short_kaon_data_energies = short_kaon_data[0]
short_kaon_data_dnde = np.sum(short_data[1:], axis=0)
short_kaon_data_emin = short_kaon_data_energies[0]
short_kaon_data_emax = short_kaon_data_energies[-1]


long_kaon_data = np.loadtxt(
    DATA_DIR.joinpath("long_kaon_positron.csv"),
    delimiter=","
).T
long_kaon_data_energies = long_kaon_data[0]
long_kaon_data_dnde = np.sum(long_data[1:], axis=0)
long_kaon_data_emin = long_kaon_data_energies[0]
long_kaon_data_emax = long_kaon_data_energies[-1]

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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double integrand_eta(double photon_energy):
    return dnde_photon_eta_rest_frame(photon_energy) / photon_energy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dnde_photon_eta_point(double photon_energy, double eta_energy):
    cdef double gamma
    cdef double beta
    cdef double pre
    cdef double emin
    cdef double emax
    cdef double res

    if eta_energy < MASS_ETA:
        return 0.0

    if eta_energy - MASS_ETA < DBL_EPSILON:
        return dnde_photon_eta_rest_frame(photon_energy)

    gamma = boost_gamma(eta_energy, MASS_ETA)
    beta = boost_beta(eta_energy, MASS_ETA)
    pre = 0.5 / (gamma * beta)
    emin = gamma * photon_energy * (1.0 - beta)
    emax = gamma * photon_energy * (1.0 + beta)
    res = 0.0

    res = quad(integrand_eta, emin, emax, epsabs=1e-10, epsrel=1e-4)[0]
    res += 2.0 * BR_ETA_TO_A_A * boost_delta_function(MASS_ETA / 2.0, photon_energy, 0.0, beta)
    return pre * res 


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

