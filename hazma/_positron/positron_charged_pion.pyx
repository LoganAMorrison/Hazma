from hazma._positron.positron_muon cimport c_muon_positron_spectrum_array as muspec
from scipy.integrate import quad
from libc.math cimport sqrt, pow, log10
import numpy as np
cimport numpy as np
import cython

include "../_decay/common.pxd"

cdef double mmu = MASS_MU
cdef double mpi = MASS_PI
cdef double me = MASS_E

cdef double eng_mu_pi_rf = (mpi * mpi + mmu * mmu) / (2.0 * mpi)

cdef double beta_mu = sqrt(1.0 - pow(mmu / eng_mu_pi_rf, 2))
cdef double gamma_mu = eng_mu_pi_rf / mmu

cdef double eng_p_max_mu_rf = (me * me + mmu * mmu) / (2.0 * mmu)
cdef double eng_p_max_pi_rf = eng_p_max_mu_rf * gamma_mu * (1.0 + beta_mu)

cdef np.ndarray eng_ps_mu = np.logspace(log10(me), log10(eng_p_max_pi_rf), num=500, dtype=np.float64)
cdef np.ndarray __muspec = muspec(eng_ps_mu, eng_mu_pi_rf)


@cython.boundscheck(True)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double __muon_spectrum(double eng_p):
    """Returns the muon spectrum in the pion rest frame."""
    return np.interp(eng_p, eng_ps_mu, __muspec)


@cython.boundscheck(True)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double __integrand(double cl, double eng_p, double eng_pi):
    """Returns the integrand of the boost integral at a given angle."""
    if eng_p < me:
        return 0.0
    cdef double p = sqrt(eng_p * eng_p - me * me)
    cdef double gamma = eng_pi / mpi
    cdef double beta = sqrt(1.0 - pow(mpi / eng_pi, 2))
    cdef double jac = p / (2. * sqrt((1 + pow(beta * cl, 2)) * eng_p * eng_p -
                                     (1 + beta * beta * (-1 + cl * cl)) *
                                     me * me -
                                     2 * beta * cl * eng_p * p) * gamma)
    cdef double eng_p_pi_rf = gamma * (eng_p - p * beta * cl)
    return BR_PI_TO_MUNU * jac * __muon_spectrum(eng_p_pi_rf)


@cython.boundscheck(True)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double c_charged_pion_positron_spectrum_point(double eng_p, double eng_pi):
    if eng_pi < mpi:
        return 0.0

    return quad(__integrand, -1.0, 1.0, points=[-1.0, 1.0],
                args=(eng_p, eng_pi), epsabs=1e-10, epsrel=1e-4)[0]

@cython.boundscheck(True)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[np.float64_t,ndim=1] c_charged_pion_positron_spectrum_array(np.ndarray[np.float64_t,ndim=1] eng_ps, double eng_pi):
    """Returns the positron spectrum from a charged pion for many positron energies."""
    cdef int npts = eng_ps.shape[0]
    cdef np.ndarray[np.float64_t,ndim=1] spec = np.zeros_like(eng_ps)
    for i in range(npts):
        spec[i] = c_charged_pion_positron_spectrum_point(eng_ps[i], eng_pi)
    return spec


# ===================================================================
# ---- Python API ---------------------------------------------------
# ===================================================================

def charged_pion_positron_spectrum(epos, epi):
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
        return c_charged_pion_positron_spectrum_array(energies, epi)
    else:
        return c_charged_pion_positron_spectrum_point(epos, epi)
