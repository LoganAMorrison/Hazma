from decay_muon cimport CSpectrum  as muspec
import numpy as np
cimport numpy as np
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline
from libc.math cimport exp, log, M_PI, log10, sqrt, abs, pow
import cython
from functools import partial
include "parameters.pxd"

import warnings

__eng_gam_max_mu_rf = (MASS_MU**2.0 - MASS_E**2.0) / (2.0 * MASS_MU)
__eng_mu_pi_rf = (MASS_PI**2.0 + MASS_MU**2.0) / (2.0 * MASS_PI)

__eng_gams_mu = np.logspace(-5.5, 3.0, num=10000, dtype=np.float64)

__mu_spec = muspec(__eng_gams_mu, __eng_mu_pi_rf)

__mu_spec2 = InterpolatedUnivariateSpline(__eng_gams_mu, __mu_spec, k=1)

cdef double __muon_spectrum(double eng_gam):
    return np.interp(eng_gam, __eng_gams_mu, __mu_spec)

cdef double fpi = DECAY_CONST_PI / np.sqrt(2)
cdef double mpi = MASS_PI
cdef double me = MASS_E
cdef double mmu = MASS_MU


@cython.cdivision(True)
cdef double __dnde_pi_to_lnug(double egam, double ml):
    cdef double r = pow(ml / mpi, 2.0)
    if 0.0 <= egam and egam <= mpi * (1. - r) / 2. :
        return (ALPHA_EM  * (2 * (2 * egam + mpi * (-1 + r)) *
                 (3 * fpi**2 * mpi * (-2 * egam + mpi) *
                  (egam**2 + 2 * egam * mpi * (-1 + r) -
                   mpi**2 * (-1 + r)) * r -
                  3 * A_PI * egam**2 * fpi * (2 * egam - mpi) * mpi * r *
                  (-4 * egam + mpi + mpi * r) + A_PI**2 * egam**4 *
                  (2 * egam + mpi * (-1 + r)) *
                  (-4 * egam + mpi * (2 + r)) -
                  6 * egam**3 * fpi * (2 * egam - mpi) *
                  mpi * r * V_PI + egam**4 *
                  (2 * egam + mpi * (-1 + r)) *
                  (-4 * egam + mpi * (2 + r)) * V_PI**2) +
                 3 * fpi * mpi * (-2 * egam + mpi)**2 * r *
                 (-2 * egam**2 * fpi - 2 * egam * fpi * mpi * (-1 + r) +
                  4 * A_PI * egam**2 * (egam - mpi * r) + fpi * mpi**2 *
                  (-1 + r**2) - 4 * egam**3 * V_PI) *
                 (2 * np.log(mpi) -
                  np.log(mpi * (-2 * egam + mpi)) + np.log(r)))) / \
            (1536. * egam * fpi**2 * mpi**3 * (-2 * egam + mpi)**2 * np.pi**4 *
             (-1 + r)**2 * r)
    else :
        return 0.0



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


cdef double __eng_gam_max(double eng_pi):
    """
    Returns the maximum allowed gamma ray energy from a charged pion decay.

    Keyword arguments::
        eng_pi: Energy of pion in laboratory frame.

    More details:
        This is computed using the fact that in the mu restframe, the
        Maximum allowed value is
            eng_gam_max_mu_rf = (pow(mass_mu,2.0) - pow(mass_e,2.0))
                            / (2.0 * mass_mu).
        Then, boosting into the pion rest frame, then to the mu rest
        frame, we get the maximum allowed energy in the lab frame.
    """
    cdef double betaPi = __beta(eng_pi, MASS_PI)
    cdef double gammaPi = __gamma(eng_pi, MASS_PI)

    cdef double betaMu = __beta(__eng_mu_pi_rf, MASS_MU)
    cdef double gammaMu = __gamma(__eng_mu_pi_rf, MASS_MU)

    return __eng_gam_max_mu_rf * gammaPi * gammaMu * \
        (1.0 + betaPi) * (1.0 + betaMu)


@cython.cdivision(True)
cdef double __integrand(double cl, double eng_gam, double eng_pi):
    """
    Returns the integrand of the differential radiative decay spectrum for
    the charged pion.

    Keyword arguments::
        cl: Angle of photon w.r.t. charged pion in lab frame.
        engPi: Energy of photon in laboratory frame.
        engPi: Energy of pion in laboratory frame.
    """
    cdef double betaPi = __beta(eng_pi, MASS_PI)
    cdef double gammaPi = __gamma(eng_pi, MASS_PI)

    cdef double engGamPiRF = eng_gam * gammaPi * (1.0 - betaPi * cl)

    cdef double preFactor = BR_PI_TO_MUNU \
        / (2.0 * gammaPi * abs(1.0 - betaPi * cl))
    cdef double preFactorE = BR_PI_TO_ENU \
        / (2.0 * gammaPi * abs(1.0 - betaPi * cl))

    return preFactor * (__muon_spectrum(engGamPiRF) +
                        __dnde_pi_to_lnug(engGamPiRF, mmu)) + \
            preFactorE * __dnde_pi_to_lnug(engGamPiRF, me)


cdef double CSpectrumPoint(double eng_gam, double eng_pi):
    """
    Returns the radiative spectrum value from charged pion given a gamma
    ray energy eng_gam and charged pion energy eng_pi. When the
    ChargedPion object is instatiated, an interplating function for the
    mu spectrum is computed.

    Keyword arguments::
        eng_gam: Energy of photon is laboratory frame.
        eng_pi: Energy of charged pion in laboratory frame.
    """
    message = 'Energy of pion cannot be less than the pion mass. Returning 0.'
    if eng_pi < MASS_PI:
        # raise warnings.warn(message, RuntimeWarning)
        return 0.0

    cdef double result = 0.0

    if 0.0 <= eng_gam and eng_gam <= __eng_gam_max(eng_pi):
        result = quad(__integrand, -1.0, 1.0, points=[-1.0, 1.0], \
                      args=(eng_gam, eng_pi), epsabs=10**-10., \
                      epsrel=10**-4.)[0]

    if 0.0 <= eng_gam and eng_gam <= (mpi**2 - mmu**2) / 2.0 / mpi:
        result = result + __dnde_pi_to_lnug(eng_gam, me)
        result = result + __dnde_pi_to_lnug(eng_gam, mmu)

    return result


@cython.boundscheck(True)
@cython.wraparound(False)
cdef np.ndarray CSpectrum(np.ndarray eng_gams, double eng_pi):
    """
    Returns the radiative spectrum dNde from charged pion given a gamma
    ray energies eng_gams and charged pion energy eng_pi.
    When the ChargedPion object is instatiated, an interplating function
    for the mu spectrum is computed.

    Keyword arguments::
        eng_gams: Gamma ray energies to evaluate spectrum.
        eng_pi: Energy of charged pion in laboratory frame.
    """
    cdef int numpts = len(eng_gams)

    cdef np.ndarray spec = np.zeros(numpts, dtype=np.float64)

    cdef int i = 0

    for i in range(numpts):
        spec[i] = CSpectrumPoint(eng_gams[i], eng_pi)

    return spec


def SpectrumPoint(double eng_gam, double eng_pi):
    """
    Returns the radiative spectrum value from charged pion given a gamma
    ray energy eng_gam and charged pion energy eng_pi. When the
    ChargedPion object is instatiated, an interplating function for the
    mu spectrum is computed.

    Keyword arguments::
        eng_gam: Energy of photon is laboratory frame.
        eng_pi: Energy of charged pion in laboratory frame.
    """
    return CSpectrum(eng_gam, eng_pi)


@cython.boundscheck(True)
@cython.wraparound(False)
def Spectrum(np.ndarray eng_gams, double eng_pi):
    """
    Returns the radiative spectrum dNde from charged pion given a gamma
    ray energies eng_gams and charged pion energy eng_pi.
    When the ChargedPion object is instatiated, an interplating function
    for the mu spectrum is computed.

    Keyword arguments::
        eng_gams: Gamma ray energies to evaluate spectrum.
        eng_pi: Energy of charged pion in laboratory frame.
    """
    return CSpectrum(eng_gams, eng_pi)
