from hazma._decay.decay_charged_pion cimport c_charged_pion_decay_spectrum_point
from hazma._decay.decay_neutral_pion cimport c_neutral_pion_decay_spectrum_point
from libc.float cimport DBL_EPSILON
import numpy as np
cimport numpy as np
from scipy.integrate import quad
import cython
from hazma._utils.boost cimport boost_beta, boost_gamma
include "../_utils/kinematics.pxd"
include "../_utils/constants.pxd"

DEF MPI = MASS_PI
DEF MPI0 = MASS_PI0
DEF MRHO = MASS_RHO


@cython.cdivision(True)
cdef double neutral_rho_integrand(double e):
    cdef epi = two_body_energy(MRHO, MPI, MPI)
    return 2 * c_charged_pion_decay_spectrum_point(e, epi, 1 + 2 + 4) / e

@cython.cdivision(True)
cdef double charged_rho_integrand(double e):
    cdef epi = two_body_energy(MRHO, MPI, MPI0)
    cdef epi0 = two_body_energy(MRHO, MPI0, MPI)
    return (
        c_charged_pion_decay_spectrum_point(e, epi, 1 + 2 + 4) 
        + c_neutral_pion_decay_spectrum_point(e, epi0)
    ) / e


@cython.cdivision(True)
cdef double c_neutral_rho_decay_spectrum_point(double e, double erho):
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
        return neutral_rho_integrand(e)

    beta = boost_beta(erho, MRHO)
    gamma = boost_gamma(erho, MRHO)
    emin = gamma * e * (1 - beta)
    emax = gamma * e * (1 + beta)
    pre = 0.5 / (beta * gamma)

    return pre * quad(neutral_rho_integrand, emin, emax, epsabs=1e-10, epsrel=1e-5)[0]


@cython.cdivision(True)
cdef double c_charged_rho_decay_spectrum_point(double e, double erho):
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
        return charged_rho_integrand(e)

    beta = boost_beta(erho, MRHO)
    gamma = boost_gamma(erho, MRHO)
    emin = gamma * e * (1 - beta)
    emax = gamma * e * (1 + beta)
    pre = 0.5 / (beta * gamma)

    return pre * quad(charged_rho_integrand, emin, emax, epsabs=1e-10, epsrel=1e-5)[0]


@cython.boundscheck(True)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t,ndim=1] c_neutral_rho_decay_spectrum_array(np.ndarray[np.float64_t,ndim=1] egams, double erho):
    cdef int npts = egams.shape[0]
    cdef np.ndarray[np.float64_t,ndim=1] spec = np.zeros_like(egams)

    for i in range(npts):
        spec[i] = c_neutral_rho_decay_spectrum_point(egams[i], erho)
    return spec


@cython.boundscheck(True)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t,ndim=1] c_charged_rho_decay_spectrum_array(np.ndarray[np.float64_t,ndim=1] egams, double erho):
    cdef int npts = egams.shape[0]
    cdef np.ndarray[np.float64_t,ndim=1] spec = np.zeros_like(egams)

    for i in range(npts):
        spec[i] = c_charged_rho_decay_spectrum_point(egams[i], erho)
    return spec


@cython.boundscheck(True)
@cython.wraparound(False)
def neutral_rho_decay_spectrum(photon_energy, rho_energy):
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
        return c_neutral_rho_decay_spectrum_array(energies, rho_energy)
    else:
        return c_neutral_rho_decay_spectrum_point(photon_energy, rho_energy)


@cython.boundscheck(True)
@cython.wraparound(False)
def charged_rho_decay_spectrum(photon_energy, rho_energy):
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
        return c_charged_rho_decay_spectrum_array(energies, rho_energy)
    else:
        return c_charged_rho_decay_spectrum_point(photon_energy, rho_energy)