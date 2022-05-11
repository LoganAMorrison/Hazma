from libc.math cimport sqrt, pow, log10, fmin, fmax
import cython

from scipy.integrate import quad
import numpy as np
cimport numpy as np

from hazma.spectra._positron._muon cimport dnde_positron_muon_point
from hazma._utils.boost cimport boost_beta, boost_gamma, boost_delta_function

include "../../_utils/constants.pxd"

DEF mmu = MASS_MU
DEF mpi = MASS_PI
DEF me = MASS_E

DEF eng_mu_pi_rf = 0.5 * (mpi * mpi + mmu * mmu) / mpi
DEF eng_e_pi_rf = 0.5 * (mpi * mpi + me * me) / mpi
DEF gamma_mu = eng_mu_pi_rf / mmu

cdef double beta_mu = sqrt(1.0 - pow(mmu / eng_mu_pi_rf, 2))

cdef double emax_mu_rf = (me * me + mmu * mmu) / (2.0 * mmu)
cdef double emax_pi_rf = gamma_mu * emax_mu_rf * (1.0 + beta_mu * sqrt(1.0 - (me / emax_mu_rf)**2))

# cdef double eng_p_max_mu_rf = (me * me + mmu * mmu) / (2.0 * mmu)
# cdef np.ndarray eng_ps_mu = np.logspace(log10(me), log10(eng_p_max_pi_rf), num=500, dtype=np.float64)
# cdef np.ndarray __muspec = muspec(eng_ps_mu, eng_mu_pi_rf)


# @cython.boundscheck(True)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef double __muon_spectrum(double eng_p):
#     """Returns the muon spectrum in the pion rest frame."""
#     return np.interp(eng_p, eng_ps_mu, __muspec)


# @cython.boundscheck(True)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef double __integrand(double cl, double eng_p, double eng_pi):
#     """Returns the integrand of the boost integral at a given angle."""
#     if eng_p < me:
#         return 0.0
#     cdef double p = sqrt(eng_p * eng_p - me * me)
#     cdef double gamma = eng_pi / mpi
#     cdef double beta = sqrt(1.0 - pow(mpi / eng_pi, 2))
#     cdef double jac = p / (2. * sqrt((1 + pow(beta * cl, 2)) * eng_p * eng_p -
#                                      (1 + beta * beta * (-1 + cl * cl)) *
#                                      me * me -
#                                      2 * beta * cl * eng_p * p) * gamma)
#     cdef double eng_p_pi_rf = gamma * (eng_p - p * beta * cl)
#     return BR_PI_TO_MU_NUMU * jac * __muon_spectrum(eng_p_pi_rf)


@cython.boundscheck(True)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dnde_positron_charged_pion_integrand(double e):
    return dnde_positron_muon_point(e, eng_mu_pi_rf) / sqrt(e**2 - me**2)


@cython.boundscheck(True)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dnde_positron_charged_pion_point(double e, double epi):
    cdef double gamma
    cdef double beta
    cdef double emin
    cdef double emax
    cdef double e0
    cdef double delta
    cdef double dnde_mu
    cdef double dnde_e

    if epi < mpi or e < me:
        return 0.0

    gamma = boost_gamma(epi, mpi)
    beta = boost_beta(epi, mpi)

    emin = fmax(gamma * (e - beta * sqrt(e**2 - me**2)), me) 
    emax = fmin(gamma * (e + beta * sqrt(e**2 - me**2)), emax_pi_rf)

    dnde_mu = BR_PI_TO_MU_NUMU * quad(
        dnde_positron_charged_pion_integrand, emin, emax, epsabs=1e-10, epsrel=1e-4
    )[0]
    dnde_mu = dnde_mu / (2 * beta * gamma)

    dnde_e = BR_PI_TO_E_NUE * boost_delta_function(eng_e_pi_rf, e, me, beta)

    return dnde_mu + dnde_e

@cython.boundscheck(True)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[np.float64_t,ndim=1] dnde_positron_charged_pion_array(double[:] eng_ps, double eng_pi):
    """Returns the positron spectrum from a charged pion for many positron energies."""
    cdef int npts = eng_ps.shape[0]
    cdef np.ndarray[np.float64_t,ndim=1] spec = np.zeros_like(eng_ps)
    for i in range(npts):
        spec[i] = dnde_positron_charged_pion_point(eng_ps[i], eng_pi)
    return spec


# ===================================================================
# ---- Python API ---------------------------------------------------
# ===================================================================

def dnde_positron_charged_pion(epos, epi):
    """
    Compute the positron spectrum dN/dE from the decay of a charged pion.
    Paramaters
    ----------
    epos: float or array-like
        Positron energy.
    epi: float 
        Energy of the pion.
    """
    if hasattr(epos, '__len__'):
        energies = np.array(epos)
        assert len(energies.shape) == 1, "Positron energies must be 0 or 1-dimensional."
        return dnde_positron_charged_pion_array(energies, epi)
    else:
        return dnde_positron_charged_pion_point(epos, epi)
