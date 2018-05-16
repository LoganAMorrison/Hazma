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
        return (ALPHA_EM *
            (-2 * (2 * egam + mpi * (-1 + r)) *
             (3 * fpi**2 * (2 * egam - mpi) * mpi * r *
              ((egam - mpi)**2 +
               (2 * egam - mpi) * mpi * r) - 3 * egam**2 * fpi *
              (2 * egam - mpi) * mpi * r *
              (4 * A_PI * egam - A_PI * mpi *
              (1 + r) - 2 * egam * V_PI) + egam**4 *
              (2 * egam + mpi * (-1 + r)) * (4 * egam - mpi * (2 + r)) *
              (A_PI**2 + V_PI**2)) + 3 * fpi * mpi * (-2 * egam + mpi)**2 * r *
             (2 * egam**2 * fpi + 2 * egam * fpi * mpi *
              (-1 + r) - 4 * A_PI * egam**2 * (egam - mpi * r) - fpi * mpi**2 *
              (-1 + r**2) + 4 * egam**3 * V_PI) *
             (log(-2 * egam + mpi) - log(mpi * r)))) / \
        (6. * egam * fpi**2 * (
            1 - (2 * egam) / mpi)**2 * mpi**4 * M_PI * (-1 + r)**2 * r)
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
cdef double __integrand(double cl, double eng_gam, double eng_pi, str mode):
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

    cdef double betaMu = __beta(__eng_mu_pi_rf, MASS_MU)
    cdef double gammaMu = __gamma(__eng_mu_pi_rf, MASS_MU)

    cdef double engGamPiRF = eng_gam * gammaPi * (1.0 - betaPi * cl)

    cdef double jac = 1. / (2.0 * gammaPi * abs(1.0 - betaPi * cl))

    cdef double dnde_munu = 0.
    cdef double dnde_munug = 0.
    cdef double dnde_enug = 0.

    __eng_gam_max_pi_rf = __eng_gam_max_mu_rf * gammaMu * (1.0 + betaMu)

    if 0. < eng_gam and eng_gam < __eng_gam_max_pi_rf:
        dnde_munu = BR_PI_TO_MUNU * jac * __muon_spectrum(engGamPiRF)

    dnde_munug = BR_PI_TO_MUNU * jac * __dnde_pi_to_lnug(engGamPiRF, mmu)
    dnde_enug = 0.000123 * jac * __dnde_pi_to_lnug(engGamPiRF, me)

    if mode == "total":
        return dnde_munu + dnde_munug + dnde_enug
    if mode == "munu":
        return dnde_munu
    if mode == "munug":
        return dnde_munug
    if mode == "enug":
        return dnde_enug

cdef double CSpectrumPoint(double eng_gam, double eng_pi, str mode):
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
        return 0.0

    return quad(__integrand, -1.0, 1.0, points=[-1.0, 1.0], \
                  args=(eng_gam, eng_pi, mode), epsabs=10**-10., \
                  epsrel=10**-5.)[0]


@cython.boundscheck(True)
@cython.wraparound(False)
cdef np.ndarray CSpectrum(np.ndarray eng_gams, double eng_pi, str mode):
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
        spec[i] = CSpectrumPoint(eng_gams[i], eng_pi, mode)

    return spec


def SpectrumPoint(double eng_gam, double eng_pi, str mode):
    """
    Returns the radiative spectrum value from charged pion given a gamma
    ray energy eng_gam and charged pion energy eng_pi. When the
    ChargedPion object is instatiated, an interplating function for the
    mu spectrum is computed.

    Keyword arguments::
        eng_gam: Energy of photon is laboratory frame.
        eng_pi: Energy of charged pion in laboratory frame.
    """
    return CSpectrumPoint(eng_gam, eng_pi, mode)


@cython.boundscheck(True)
@cython.wraparound(False)
def Spectrum(np.ndarray eng_gams, double eng_pi, str mode):
    """
    Returns the radiative spectrum dNde from charged pion given a gamma
    ray energies eng_gams and charged pion energy eng_pi.
    When the ChargedPion object is instatiated, an interplating function
    for the mu spectrum is computed.

    Keyword arguments::
        eng_gams: Gamma ray energies to evaluate spectrum.
        eng_pi: Energy of charged pion in laboratory frame.
    """
    return CSpectrum(eng_gams, eng_pi, mode)
