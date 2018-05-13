import numpy as np
cimport numpy as np
import cython
from libc.math cimport M_PI, sqrt, abs
include "parameters.pxd"

from scipy.integrate import quad

cdef double mmu = MASS_MU
cdef double me = MASS_E


@cython.cdivision(True)
cdef double __spectrum_rf(double ee):
    """
    Returns the positron spectrum from a muon in the muon rest frame.

    Parameters
    ----------
    ee : float
        Energy of the positron.

    Returns
    -------
    dnde : float
        The value of the spectrum given a positron energy `ee`.
    """
    cdef double r = me / mmu
    cdef double s = me**2 - 2. * ee * mmu + mmu**2
    cdef double smax = mmu**2 * (1. - r)**2
    cdef double smin = 0.
    cdef double dnds = 0.0
    if s < smin or smax < s:
        return dnds
    dnds = (2 * (mmu**4 * (-1 + r**2)**2 + mmu**2 *
                 (1 + r**2) * s - 2 * s**2) *
            np.sqrt(mmu**4 * (-1 + r**2)**2 -
                    2 * mmu**2 * (1 + r**2) * s + s**2)) / mmu**8
    return 2 * mmu * dnds


@cython.cdivision(True)
cdef double __integrand(double cl, double ee, double emu):
    """
    Returns the integrand for the boost integral used to compute the postitron
    spectrum from the muon.

    Parameters
    ----------
    cl : double
        Angle the electron makes with the z-axis.
    ee : double
        Energy of the positron.
    emu : double
        Energy of the muon.

    Returns
    -------
    integrand : double
        Integral for the boost integral.
    """
    cdef double gamma = emu / mmu
    cdef double beta = sqrt(1. - 1. / gamma**2)
    cdef double emurf = gamma * emu * (1. - cl * beta)
    cdef double jac = 1. / (2. * gamma * abs(1. - beta * cl))

    return __spectrum_rf(emurf) * jac


@cython.cdivision(True)
cdef double CSpectrumPoint(double ee, double emu):
    """
    Returns the positron spectrum at a single electron energy from the muon
    given an arbitrary muon energy.

    Parameters
    ----------
    ee : double
        Energy of the positron.
    emu : double
        Energy of the muon.

    Returns
    -------
    dnde : double
        Positron spectrum of the muon given an electron energy `ee` and a muon
        energy `emu`.
    """
    if emu < mmu:
        return 0.0

    cdef double result = 0.0

    return quad(__integrand, -1., 1., points=[-1.0, 1.0], \
                args=(ee, emu), epsabs=10**-10., epsrel=10**-4.)[0]


@cython.boundscheck(True)
@cython.wraparound(False)
cdef np.ndarray CSpectrum(np.ndarray ee, double emu):
    """
    Returns the positron spectrum at many electron energies from the muon
    given an arbitrary muon energy.

    Parameters
    ----------
    ee : numpy.array
        Energy of the positron.
    emu : double
        Energy of the muon.

    Returns
    -------
    dnde : numpy.array
        Positron spectrum of the muon given electron energies `ee` and a muon
        energy `emu`.
    """
    cdef int numpts = len(ee)

    cdef np.ndarray spec = np.zeros(numpts, dtype=np.float64)

    cdef int i = 0

    for i in range(numpts):
        spec[i] = Cdnde_muon_point(ee[i], emu)

    return spec


def SpectrumPoint(double ee, double emu):
    """
    Returns the positron spectrum at a single electron energy from the muon
    given an arbitrary muon energy.

    Parameters
    ----------
    ee : double
        Energy of the positron.
    emu : double
        Energy of the muon.

    Returns
    -------
    dnde : double
        Positron spectrum of the muon given an electron energy `ee` and a muon
        energy `emu`.
    """
    return CSpectrumPoint(ee, emu)


@cython.boundscheck(True)
@cython.wraparound(False)
def SpectrumPoint(np.ndarray ee, double emu):
    """
    Returns the positron spectrum at many electron energies from the muon
    given an arbitrary muon energy.

    Parameters
    ----------
    ee : numpy.array
        Energy of the positron.
    emu : double
        Energy of the muon.

    Returns
    -------
    dnde : numpy.array
        Positron spectrum of the muon given electron energies `ee` and a muon
        energy `emu`.
    """
    return CSpectrum(ee, emu)
