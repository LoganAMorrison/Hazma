from hazma.positron_helper_functions.positron_muon cimport CSpectrum as muspec
from scipy.integrate import quad
from libc.math cimport sqrt, pow, log10
import numpy as np
cimport numpy as np
import cython

include "parameters.pxd"

cdef double mmu = MASS_MU
cdef double mpi = MASS_PI
cdef double me = MASS_E

cdef double eng_mu_pi_rf = (mpi * mpi + mmu * mmu) / (2.0 * mpi)

cdef double beta_mu = sqrt(1.0 - pow(mmu / eng_mu_pi_rf, 2))
cdef double gamma_mu = eng_mu_pi_rf / mmu

cdef double eng_p_max_mu_rf = (me * me + mmu * mmu) / (2.0 * mmu)
cdef double eng_p_max_pi_rf = eng_p_max_mu_rf * gamma_mu * (1.0 + beta_mu)

# TODO: Really?? 10000 pts?
eng_ps_mu = np.logspace(log10(me), log10(eng_p_max_pi_rf), num=500, dtype=np.float64)

cdef np.ndarray __muspec = muspec(eng_ps_mu, eng_mu_pi_rf)

cdef double __muon_spectrum(double eng_p):
    """
    Returns the muon spectrum in the pion rest frame.

    Parameters
    ----------
    eng_p : double
        Energy of the positron.

    Returns
    -------
    dnde_mu : double
        Value of muon positron spectrum at an positron energy of `eng_p`
    """
    return np.interp(eng_p, eng_ps_mu, __muspec)

@cython.cdivision(True)
cdef double __integrand(double cl, double eng_p, double eng_pi):
    """
    Returns the integrand of the boost integral at a given angle.

    Parameters
    ----------
    eng_p : double
        Energy of the positron.
    eng_pi : double
        Energy of the pion.

    Returns
    -------
    integrand : double
        Integrand of the boost integral at angle `cl`, positron energy `eng_p`
        and pion energy `eng_pi`.
    """
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

cdef double CSpectrumPoint(double eng_p, double eng_pi):
    """
    Cythonized version of SpectrumPoint.

    Returns the positron spectrum from a charged pion for a single positron
    energy.

    Parameters
    ----------
    eng_p : double
        Energy of the positron.
    eng_pi : double
        Energy of the pion.

    Returns
    -------
    dnde_cpi : double
        Positron spectrum from a charged pion given positron energy `eng_p`
        and pion energy `eng_pi`.
    """
    if eng_pi < mpi:
        return 0.0

    return quad(__integrand, -1.0, 1.0, points=[-1.0, 1.0],
                args=(eng_p, eng_pi), epsabs=1e-10, epsrel=1e-4)[0]

@cython.boundscheck(True)
@cython.wraparound(False)
cdef np.ndarray CSpectrum(np.ndarray eng_ps, double eng_pi):
    """
    Cythonized version of Spectrum.

    Returns the positron spectrum from a charged pion for many positron
    energies.

    Parameters
    ----------
    eng_ps : numpy.array
        Energies of the positron.
    eng_pi : double
        Energy of the pion.

    Returns
    -------
    dnde_cpi : double
        Positron spectrum from a charged pion given positron energies `engs_p`
        and pion energy `eng_pi`.
    """
    cdef int num_pts = len(eng_ps)
    cdef np.ndarray spec = np.zeros(num_pts, dtype=np.float64)
    cdef int i = 0

    for i in range(num_pts):
        spec[i] = CSpectrumPoint(eng_ps[i], eng_pi)

    return spec

def SpectrumPoint(double eng_p, double eng_pi):
    """
    Returns the positron spectrum from a charged pion for a single positron
    energy.

    Parameters
    ----------
    eng_p : double
        Energy of the positron.
    eng_pi : double
        Energy of the pion.

    Returns
    -------
    dnde_cpi : double
        Positron spectrum from a charged pion given positron energy `eng_p`
        and pion energy `eng_pi`.
    """
    return CSpectrum(eng_p, eng_pi)

@cython.boundscheck(True)
@cython.wraparound(False)
def Spectrum(np.ndarray eng_ps, double eng_pi):
    """
    Returns the positron spectrum from a charged pion for many positron
    energies.

    Parameters
    ----------
    eng_ps : numpy.array
        Energies of the positron.
    eng_pi : double
        Energy of the pion.

    Returns
    -------
    dnde_cpi : double
        Positron spectrum from a charged pion given positron energies `engs_p`
        and pion energy `eng_pi`.
    """
    return CSpectrum(eng_ps, eng_pi)
