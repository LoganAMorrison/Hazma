import numpy as np
cimport numpy as np
from scipy.integrate import quad
from libc.math cimport exp, log, M_PI, log10, sqrt
from scipy.special.cython_special cimport spence
import cython
from functools import partial
include "parameters.pxd"
import warnings


"""
Module for computing the photon spectrum from radiative muon decay.

The radiative spectrum of the muon was taken from: arXiv:hep-ph/9909265
"Muon Decay and Physics Beyond the Standard Model".
"""

@cython.cdivision(True)
cdef double __j_plus(double y):
    """
    Form factor in differential branching fraction of radiative muon decay.
    See p.18, eqn (54) of arXiv:hep-ph/9909265.

    Keyword arguments::
        y -- 2 * (photon energy) / (muon mass)
    """
    cdef double yconj = 1.0 - y
    cdef double r = RATIO_E_MU_MASS_SQ
    cdef double preFactor = ALPHA_EM * yconj / 6.0 / M_PI
    cdef double term1 = 3.0 * log(yconj / r) - (17.0 / 2.0)
    cdef double term2 = -3.0 * log(yconj / r) + 7.0
    cdef double term3 = 2.0 * log(yconj / r) - (13.0 / 3.0)
    return preFactor * (term1 + term2 * yconj + term3 * yconj**2.0)

@cython.cdivision(True)
cdef double __j_minus(double y):
    """
    Form factor in differential branching fraction of radiative muon decay.
    See p.18, eqn (55) of arXiv:hep-ph/9909265.

    Keyword arguments::
        y -- 2 * (photon energy) / (muon mass)
    """
    cdef double yconj = 1.0 - y
    cdef double r = RATIO_E_MU_MASS_SQ
    cdef double preFactor = ALPHA_EM * yconj**2.0 / 6.0 / M_PI
    cdef double term1 = 3.0 * log(yconj / r) - (93.0 / 12.0)
    cdef double term2 = -4.0 * log(yconj / r) + (29.0 / 3.0)
    cdef double term3 = 2.0 * log(yconj / r) - (55.0 / 12.0)
    return preFactor * (term1 + term2 * yconj + term3 * yconj**2.0)

@cython.cdivision(True)
cdef double __dBdy(double y):
    """
    Differential branching fraction from: mu -> e nu nu gam.
    See p.18, eqn (56) of arXiv:hep-ph/9909265.

    Keyword arguments::
        y -- 2 * (photon energy) / (muon mass)
    """
    cdef double result = 0.0
    if 0.0 <= y and y <= 1.0 - RATIO_E_MU_MASS_SQ:
        result = (2.0 / y) * (__j_plus(y) + __j_minus(y))
    return result

@cython.cdivision(True)
cdef double __gamma(double eng, double mass):
    """
    Returns special relativity boost factor gamma.

    Keyword arguments::
        eng -- energy of particle.
        mass -- mass of particle.
    """
    return eng / mass

@cython.cdivision(True)
cdef double __beta(double eng, double mass):
    """
    Returns velocity in natural units.

    Keyword arguments::
        eng -- energy of particle.
        mass -- mass of particle.
    """
    return sqrt(1.0 - (mass / eng)**2.0)

@cython.cdivision(True)
cdef double __integrand(double cl, double engGam, double engMu):
    """
    Compute integrand of dN_{\gamma}/dE_{\gamma} from mu -> e nu nu gamma
    in laboratory frame. The integration variable is cl - the angle between
    gamma ray and muon.

    Keyword arguments::
        cl -- Angle between gamma ray and muon in laboratory frame.
        engGam -- Gamma ray energy in laboratory frame.
        engMu -- Muon energy in laboratory frame.
    """
    cdef double beta = __beta(engMu, MASS_MU)
    cdef double gamma = __gamma(engMu, MASS_MU)

    cdef double engGamMuRF = gamma * engGam * (1.0 - beta * cl)

    return __dBdy((2.0 / MASS_MU) * engGamMuRF) \
        / (engMu * (1.0 - cl * beta))


cdef double CSpectrumPoint(double eng_gam, double eng_mu):
    """
    Compute dN_{\gamma}/dE_{\gamma} from mu -> e nu nu gamma in the
    laborartory frame.

    Keyword arguments::
        eng_gam (float) -- Gamma ray energy in laboratory frame.
        eng_mu (float) -- Muon energy in laboratory frame.
    """
    message = 'Energy of muon cannot be less than the muon mass. Returning 0.'
    if eng_mu < MASS_MU:
        # raise warnings.warn(message, RuntimeWarning)
        return 0.0

    cdef double result = 0.0

    cdef double beta = __beta(eng_mu, MASS_MU)
    cdef double gamma = __gamma(eng_mu, MASS_MU)

    cdef double eng_gam_max = 0.5 * (MASS_MU - MASS_E**2.0 / MASS_MU) \
        * gamma * (1.0 + beta)

    if 0 <= eng_gam and eng_gam <= eng_gam_max:
        result = quad(__integrand, -1.0, 1.0, args=(eng_gam, eng_mu), \
            points=[-1.0, 1.0], epsabs=10**-10., epsrel=10**-4.)[0]


    return result


cdef double cspectrum_point_analytic(double egam, double emu):
    """
    Compute dN_{\gamma}/dE_{\gamma} from mu -> e nu nu gamma in the
    laborartory frame.

    Paramaters
    ----------
    egam: double 
        Gamma ray energy in laboratory frame.
    emu: double
        Muon energy in laboratory frame.
    """
    cdef double gamma
    cdef double beta
    cdef double y
    cdef double r
    cdef double x
    cdef double xp
    cdef double xm
    cdef double wp
    cdef double wm
    cdef double result

    gamma = __gamma(emu, MASS_MU)
    beta = __beta(emu, MASS_MU)

    # Rescaled variables
    y = 2 * egam / MASS_MU
    r = (MASS_E / MASS_MU)**2
    # Re-rescaled variables
    x = y * gamma

    # Bounds check
    if x < 0.0 or x >= (1.0 - r) / (1.0 - beta):
        return 0.0

    # Upper bound on 'angular' variable (w = 1 - beta * ctheta)
    if x < (1.0 - r) / (1.0 + beta):
        wp = 1.0 + beta
    else:
        wp = (1.0 - r) / x
    wm = 1.0 - beta

    xp = x * wp
    xm = x * wm

    # Polynomial contribution
    result = (
            (xm - xp) * (102.0 + xm * xp * (
                191.0 + 21.0 * xm**2 + xm * (-92.0 + 21.0 * xp) + xp * (-92.0 + 21.0 * xp))
            ) / (12 * xm * xp * beta)
    )
    # Logarithmic contributions
    result +=
        (9 + xm * (18 + xm * (-18 + (9 - 2 * xm) * xm))) * log((1 - xm) / r) / (3 * xm * beta)
    result +=
        (-9 + xp * (-18 + xp * (18 + xp * (-9 + 2 * xp)))) * log((1 - xp) / r) /
        (3 * xp * beta)
    result += 5 / beta * (log((1 - xm) / r) * log(xm) - log((1 - xp) / r) * log(xp))
    result += 4 / (3 * beta) * (4 * log((1 - xp) / (1 - xm)) + 7 * log(xp / xm))
    # PolyLog terms
    result += 5 / beta * (spence(xm) - spence(xp))
    
    return result * ALPHA_EM / (3 * M_PI * emu)


@cython.cdivision(True)
cdef np.ndarray CSpectrum(np.ndarray eng_gams, double eng_mu):
    """
    Compute dN/dE from mu -> e nu nu gamma in the laborartory frame.

    Paramaters
    ----------
    eng_gams : np.ndarray
        List of gamma ray energies in laboratory frame.
    eng_mu : float
        Muon energy in laboratory frame.

    Returns
    -------
    spec : np.ndarray
        List of gamma ray spectrum values, dNdE, evaluated at `eng_gams`
        given muon energy `eng_mu`.
    """
    cdef int numpts = len(eng_gams)

    cdef np.ndarray spec = np.zeros(numpts, dtype=np.float64)

    for i in range(numpts):
        spec[i] = CSpectrumPoint(eng_gams[i], eng_mu)

    return spec

def SpectrumPoint(double eng_gam, double eng_mu):
    """
    Compute dN_{\gamma}/dE_{\gamma} from mu -> e nu nu gamma in the
    laborartory frame.

    Keyword arguments::
        eng_gam (float) -- Gamma ray energy in laboratory frame.
        eng_mu (float) -- Muon energy in laboratory frame.
    """
    return CSpectrumPoint(eng_gam, eng_mu)



@cython.cdivision(True)
def Spectrum(np.ndarray eng_gams, double eng_mu):
    """
    Compute dN/dE from mu -> e nu nu gamma in the laborartory frame.

    Paramaters
    ----------
    eng_gams : np.ndarray
        List of gamma ray energies in laboratory frame.
    eng_mu : float
        Muon energy in laboratory frame.

    Returns
    -------
    spec : np.ndarray
        List of gamma ray spectrum values, dNdE, evaluated at `eng_gams`
        given muon energy `eng_mu`.
    """
    return CSpectrum(eng_gams, eng_mu)
