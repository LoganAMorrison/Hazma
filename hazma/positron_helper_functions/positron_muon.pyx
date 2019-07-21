import numpy as np
cimport numpy as np
import cython
from libc.math cimport sqrt

include "parameters.pxd"

from scipy.integrate import quad

cdef double mmu = MASS_MU
cdef double me = MASS_E

@cython.cdivision(True)
cdef double __spectrum_rf(double eng_p):
    """
    Returns the positron spectrum from a muon in the muon rest frame.

    Parameters
    ----------
    eng_p : float
        Energy of the positron.

    Returns
    -------
    dnde : float
        The value of the spectrum given a positron energy `eng_p`.
    """
    cdef double r = me / mmu
    cdef double s = me * me - 2. * eng_p * mmu + mmu * mmu
    cdef double smax = (mmu - me) * (mmu - me)
    cdef double smin = 0.0
    if s <= smin or smax <= s:
        return 0.0

    return 2 * mmu * (2 * (pow(mmu, 4) * pow(-1 + r * r, 2) + mmu * mmu *
                           (1 + r * r) * s - 2 * s * s) *
                      np.sqrt(pow(mmu, 4) * pow(-1 + r * r, 2) -
                              2 * mmu**2 * (1 + r * r) * s + s * s)) / pow(mmu, 8)

@cython.cdivision(True)
cdef double __integrand(double cl, double eng_p, double eng_mu):
    """
    Returns the integrand for the boost integral used to compute the postitron
    spectrum from the muon.

    Parameters
    ----------
    cl : double
        Angle the electron makes with the z-axis.
    eng_p : double
        Energy of the positron.
    eng_mu : double
        Energy of the muon.

    Returns
    -------
    integrand : double
        Integral for the boost integral.
    """
    if eng_p < me:
        return 0.0
    cdef double p = sqrt(eng_p * eng_p - me * me)
    cdef double gamma = eng_mu / mmu
    cdef double beta = sqrt(1.0 - pow(mmu / eng_mu, 2))
    cdef double emurf = gamma * (eng_p - p * beta * cl)
    cdef double jac = p / (2. * sqrt((1 + pow(beta * cl, 2)) * eng_p * eng_p -
                                     (1 + beta * beta * (-1 + cl * cl)) *
                                     me * me -
                                     2 * beta * cl * eng_p * p) * gamma)

    return __spectrum_rf(emurf) * jac

@cython.cdivision(True)
cdef double CSpectrumPoint(double eng_p, double eng_mu):
    """
    Returns the positron spectrum at a single electron energy from the muon
    given an arbitrary muon energy.

    Parameters
    ----------
    eng_p : double
        Energy of the positron.
    eng_mu : double
        Energy of the muon.

    Returns
    -------
    dnde : double
        Positron spectrum of the muon given an electron energy `eng_p` and a muon
        energy `eng_mu`.
    """
    if eng_mu < mmu:
        return 0.0

    return quad(__integrand, -1., 1., points=[-1.0, 1.0],
                args=(eng_p, eng_mu), epsabs=1e-10, epsrel=1e-4)[0]

@cython.boundscheck(True)
@cython.wraparound(False)
cdef np.ndarray CSpectrum(np.ndarray engs_p, double eng_mu):
    """
    Returns the positron spectrum at many electron energies from the muon
    given an arbitrary muon energy.

    Parameters
    ----------
    engs_p : numpy.array
        Energy of the positron.
    eng_mu : double
        Energy of the muon.

    Returns
    -------
    dnde : numpy.array
        Positron spectrum of the muon given electron energies `engs_p` and a
        muon energy `eng_mu`.
    """
    cdef int numpts = len(engs_p)

    cdef np.ndarray spec = np.zeros(numpts, dtype=np.float64)

    cdef int i = 0

    for i in range(numpts):
        spec[i] = CSpectrumPoint(engs_p[i], eng_mu)

    return spec

def SpectrumPoint(double eng_p, double eng_mu):
    """
    Returns the positron spectrum at a single electron energy from the muon
    given an arbitrary muon energy.

    Parameters
    ----------
    eng_p : double
        Energy of the positron.
    eng_mu : double
        Energy of the muon.

    Returns
    -------
    dnde : double
        Positron spectrum of the muon given an electron energy `eng_p` and a muon
        energy `eng_mu`.
    """
    return CSpectrumPoint(eng_p, eng_mu)

@cython.boundscheck(True)
@cython.wraparound(False)
def Spectrum(np.ndarray engs_p, double eng_mu):
    """
    Returns the positron spectrum at many electron energies from the muon
    given an arbitrary muon energy.

    Parameters
    ----------
    engs_p : numpy.array
        Energy of the positron.
    eng_mu : double
        Energy of the muon.

    Returns
    -------
    dnde : numpy.array
        Positron spectrum of the muon given electron energies `engs_p` and a
        muon energy `eng_mu`.
    """
    return CSpectrum(engs_p, eng_mu)
