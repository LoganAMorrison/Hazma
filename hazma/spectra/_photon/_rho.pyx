from libc.float cimport DBL_EPSILON
import cython

import numpy as np
cimport numpy as np
from scipy.integrate import quad

from hazma._utils.boost cimport boost_beta, boost_gamma
from hazma._utils.kinematics cimport two_body_energy
from hazma.spectra._photon._pion cimport dnde_photon_neutral_pion_point
from hazma.spectra._photon._pion cimport dnde_photon_charged_pion_point

include "../../_utils/constants.pxd"

DEF MPI = MASS_PI
DEF MPI0 = MASS_PI0
DEF MRHO = MASS_RHO

# ============================================================================
# ---- Neutral Rho -----------------------------------------------------------
# ============================================================================

@cython.cdivision(True)
cdef double integrand_neutral_rho(double e):
    cdef epi = MRHO / 2.0
    return 2 * dnde_photon_charged_pion_point(e, epi) / e


@cython.cdivision(True)
cdef double dnde_photon_neutral_rho_point(double e, double erho):
    cdef:
        beta
        gamma
        emin
        emax
        pre

    if erho < MRHO:
        return 0.0

    # If we are sufficiently close to the rho rest-frame, use the 
    # rest-frame result.
    if erho - MRHO < DBL_EPSILON:
        return integrand_neutral_rho(e)

    beta = boost_beta(erho, MRHO)
    gamma = boost_gamma(erho, MRHO)
    emin = gamma * e * (1 - beta)
    emax = gamma * e * (1 + beta)
    pre = 0.5 / (beta * gamma)

    return pre * quad(integrand_neutral_rho, emin, emax, epsabs=1e-10, epsrel=1e-5)[0]


@cython.boundscheck(True)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t,ndim=1] dnde_photon_neutral_rho_array(double[:] egams, double erho):
    cdef int npts = egams.shape[0]
    cdef np.ndarray[np.float64_t,ndim=1] spec = np.zeros_like(egams)

    for i in range(npts):
        spec[i] = dnde_photon_neutral_rho_point(egams[i], erho)
    return spec


@cython.boundscheck(True)
@cython.wraparound(False)
def dnde_photon_neutral_rho(photon_energy, rho_energy):
    """
    Compute the photon spectrum dN/dE from the decay of a neutral rho meson.
    Paramaters
    ----------
    photon_energy: float or array-like
        Photon energy.
    rho_energy: float 
        Energy of the neutral rho meson.
    """
    if hasattr(photon_energy, '__len__'):
        energies = np.array(photon_energy)
        assert len(energies.shape) == 1, "Photon energies must be 0 or 1-dimensional."
        return dnde_photon_neutral_rho_array(energies, rho_energy)
    else:
        return dnde_photon_neutral_rho_point(photon_energy, rho_energy)


# ============================================================================
# ---- Charged Rho -----------------------------------------------------------
# ============================================================================

@cython.cdivision(True)
cdef double integrand_charged_rho(double e):
    cdef epi = two_body_energy(MRHO, MPI, MPI0)
    cdef epi0 = two_body_energy(MRHO, MPI0, MPI)
    return (
        dnde_photon_charged_pion_point(e, epi) 
        + dnde_photon_neutral_pion_point(e, epi0)
    ) / e


@cython.cdivision(True)
cdef double dnde_photon_charged_rho_point(double e, double erho):
    cdef:
        beta
        gamma
        emin
        emax
        pre

    if erho < MRHO:
        return 0.0

    # If we are sufficiently close to the rho rest-frame, use the 
    # rest-frame result.
    if erho - MRHO < DBL_EPSILON:
        return integrand_charged_rho(e)

    beta = boost_beta(erho, MRHO)
    gamma = boost_gamma(erho, MRHO)
    emin = gamma * e * (1 - beta)
    emax = gamma * e * (1 + beta)
    pre = 0.5 / (beta * gamma)

    return pre * quad(integrand_charged_rho, emin, emax, epsabs=1e-10, epsrel=1e-5)[0]


@cython.boundscheck(True)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t,ndim=1] dnde_photon_charged_rho_array(double[:] egams, double erho):
    cdef int npts = egams.shape[0]
    cdef np.ndarray[np.float64_t,ndim=1] spec = np.zeros_like(egams)

    for i in range(npts):
        spec[i] = dnde_photon_charged_rho_point(egams[i], erho)
    return spec


@cython.boundscheck(True)
@cython.wraparound(False)
def dnde_photon_charged_rho(photon_energy, rho_energy):
    """
    Compute the photon spectrum dN/dE from the decay of a charged rho meson.
    Paramaters
    ----------
    photon_energy: float or array-like
        Photon energy.
    rho_energy: float 
        Energy of the neutral rho meson.
    """
    if hasattr(photon_energy, '__len__'):
        energies = np.array(photon_energy)
        assert len(energies.shape) == 1, "Photon energies must be 0 or 1-dimensional."
        return dnde_photon_charged_rho_array(energies, rho_energy)
    else:
        return dnde_photon_charged_rho_point(photon_energy, rho_energy)