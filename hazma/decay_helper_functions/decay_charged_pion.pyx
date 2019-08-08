from hazma.decay_helper_functions.decay_muon cimport CSpectrum as muspec
from hazma.decay_helper_functions.decay_muon cimport CSpectrumPoint as muspecpt
import numpy as np
cimport numpy as np
from scipy.integrate import quad
from libc.math cimport exp, log, M_PI, log10, sqrt, abs, pow
import cython
include "parameters.pxd"

import warnings

cdef double eng_gam_max_mu_rf = (MASS_MU**2.0 - MASS_E**2.0) / (2.0 * MASS_MU)
cdef double eng_mu_pi_rf = (MASS_PI**2.0 + MASS_MU**2.0) / (2.0 * MASS_PI)
cdef double fpi = DECAY_CONST_PI / sqrt(2.)  # ~92 MeV
cdef double mpi = MASS_PI
cdef double me = MASS_E
cdef double mmu = MASS_MU

cdef np.ndarray eng_gams_mu
cdef np.ndarray mu_spec

# TODO: load from a file
eng_gams_mu = np.logspace(-5.5, 3.0, num=10000, dtype=np.float64)
mu_spec = muspec(eng_gams_mu, eng_mu_pi_rf)


cdef double muon_spectrum(double eng_gam):
    """
    Returns the interpolated muon spectrum.
    """
    return np.interp(eng_gam, eng_gams_mu, mu_spec)


@cython.cdivision(True)
cdef double dnde_pi_to_lnug(double egam, double ml):
    """
    Returns dnde from pi-> l nu g.
    """
    cdef double x = 2 * egam / mpi
    cdef double r = (ml / mpi) * (ml / mpi)

    if 0.0 <= x and x <= (1 - r):
        return __dnde_pi_to_lnug(x, r)
    else :
        return 0.0


@cython.cdivision(True)
cdef double __dnde_pi_to_lnug(double x, double r):
    """
    Helper function for computing dnde from pi-> l nu g.
    """
    # Account for energy-dependence of vector form factor
    cdef double F_V = F_V_PI * (1 + F_V_PI_SLOPE * (1 - x))

    # Numerator terms with no log
    cdef double f = (r + x - 1) * (
        mpi*mpi * x*x*x*x * (F_A_PI*F_A_PI + F_V*F_V) * (r*r - r*x + r - 2 * (x-1)*(x-1))
        - 12 * sqrt(2) * fpi * mpi * r * (x-1) * x*x * (F_A_PI * (r - 2*x + 1) + F_V * x)
        - 24 * fpi*fpi * r * (x-1) * (4*r*(x-1) + (x-2)*(x-2)))

    # Numerator terms with log
    cdef double g = 12 * sqrt(2) * fpi * r * (x-1)*(x-1) * log(r / (1-x)) * (
        mpi * x*x * (F_A_PI * (x - 2*r) - F_V * x)
        + sqrt(2) * fpi * (2*r*r - 2*r*x - x*x + 2*x - 2))

    return ALPHA_EM * (f + g) / (24 * M_PI * mpi * fpi*fpi * (r-1)*(r-1)
                                 * (x-1)*(x-1) * r * x)


@cython.cdivision(True)
cdef double gamma(double eng, double mass):
    """
    Returns special relativity boost factor gamma.

    Keyword arguments::
        eng -- energy of particle.
        mass -- mass of particle.
    """
    return eng / mass


@cython.cdivision(True)
cdef double beta(double eng, double mass):
    """
    Returns velocity in natural units.

    Keyword arguments::
        eng -- energy of particle.
        mass -- mass of particle.
    """
    return sqrt(1.0 - (mass / eng)**2.0)


cdef double eng_gam_max(double eng_pi):
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
    cdef double beta_pi = beta(eng_pi, MASS_PI)
    cdef double gamma_pi = gamma(eng_pi, MASS_PI)

    cdef double beta_mu = beta(eng_mu_pi_rf, MASS_MU)
    cdef double gamma_mu = gamma(eng_mu_pi_rf, MASS_MU)

    return eng_gam_max_mu_rf * gamma_pi * gamma_mu * \
        (1.0 + beta_pi) * (1.0 + beta_mu)


@cython.cdivision(True)
cdef double integrand(double cl, double eng_gam, double eng_pi, str mode):
    """
    Returns the integrand of the differential radiative decay spectrum for
    the charged pion.

    Keyword arguments::
        cl: Angle of photon w.r.t. charged pion in lab frame.
        engPi: Energy of photon in laboratory frame.
        engPi: Energy of pion in laboratory frame.
    """
    cdef double beta_pi = beta(eng_pi, MASS_PI)
    cdef double gamma_pi = gamma(eng_pi, MASS_PI)

    cdef double beta_mu = beta(eng_mu_pi_rf, MASS_MU)
    cdef double gamma_mu = gamma(eng_mu_pi_rf, MASS_MU)

    cdef double eng_gam_pi_rF = eng_gam * gamma_pi * (1.0 - beta_pi * cl)

    cdef double jac = 1. / (2.0 * gamma_pi * abs(1.0 - beta_pi * cl))

    cdef double dnde_munu = 0.
    cdef double dnde_munug = 0.
    cdef double dnde_enug = 0.

    __eng_gam_max_pi_rf = eng_gam_max_mu_rf * gamma_mu * (1.0 + beta_mu)

    if 0. < eng_gam_pi_rF and eng_gam_pi_rF < __eng_gam_max_pi_rf:
        dnde_munu = BR_PI_TO_MUNU * jac * muon_spectrum(eng_gam_pi_rF)

    dnde_munug = BR_PI_TO_MUNU * jac * dnde_pi_to_lnug(eng_gam_pi_rF, mmu)
    dnde_enug = BR_PI_TO_ENU * jac * dnde_pi_to_lnug(eng_gam_pi_rF, me)

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

    return quad(integrand, -1.0, 1.0, points=[-1.0, 1.0], \
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
