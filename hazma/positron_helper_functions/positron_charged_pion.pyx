from hazma.positron_helper_functions.positron_muon cimport CSpectrum as muspec
from scipy.integrate import quad
from libc.math cimport exp, log, M_PI, log10, sqrt, abs, pow
import numpy as np
cimport numpy as np
import cython
include "parameters.pxd"

cdef double mmu = MASS_MU
cdef double mpi = MASS_PI
cdef double me = MASS_E

cdef double eng_mu_pi_rf = (mpi**2.0 + mmu**2.0) / (2.0 * mpi)

cdef double beta_mu = sqrt(1.0 - (mmu / eng_mu_pi_rf)**2)
cdef double gamma_mu = eng_mu_pi_rf / mmu

cdef double eng_p_max_mu_rf = (me**2 + mmu**2) / (2.0 * mmu)
cdef double eng_p_max_pi_rf = eng_p_max_mu_rf * gamma_mu * (1.0 + beta_mu)

eng_ps_mu = np.linspace(0.0, eng_p_max_pi_rf, num=10000, dtype=np.float64)

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
    cdef double gamma = eng_pi / mpi
    cdef double beta = sqrt(1.0 - (mpi / eng_pi)**2)

    cdef double eng_p_pi_rf = eng_p * gamma * (1.0 - beta * cl)

    cdef double pre_factor = BR_PI_TO_MUNU \
        / (2.0 * gamma * abs(1.0 - beta * cl))

    return pre_factor * __muon_spectrum(eng_p_pi_rf)


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

    return quad(__integrand, -1.0, 1.0, points=[-1.0, 1.0], \
                args=(eng_p, eng_pi), epsabs=10**-10., epsrel=10**-4.)[0]


@cython.boundscheck(True)
@cython.wraparound(False)
cdef np.ndarray CSpectrum(np.ndarray engs_p, double eng_pi):
    """
    Cythonized version of Spectrum.

    Returns the positron spectrum from a charged pion for many positron
    energies.

    Parameters
    ----------
    eng_p : numpy.array
        Energy of the positron.
    eng_pi : double
        Energy of the pion.

    Returns
    -------
    dnde_cpi : double
        Positron spectrum from a charged pion given positron energies `engs_p`
        and pion energy `eng_pi`.
    """
    cdef int numpts = len(engs_p)

    cdef np.ndarray spec = np.zeros(numpts, dtype=np.float64)

    cdef int i = 0

    for i in range(numpts):
        spec[i] = CSpectrumPoint(engs_p[i], eng_pi)

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
def Spectrum(np.ndarray engs_p, double eng_pi):
    """
    Returns the positron spectrum from a charged pion for many positron
    energies.

    Parameters
    ----------
    eng_p : numpy.array
        Energy of the positron.
    eng_pi : double
        Energy of the pion.

    Returns
    -------
    dnde_cpi : double
        Positron spectrum from a charged pion given positron energies `engs_p`
        and pion energy `eng_pi`.
    """
    return CSpectrum(engs_p, eng_pi)
