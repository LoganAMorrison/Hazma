import decay_muon
import decay_charged_pion
import decay_neutral_pion
from ..phases_space_generator cimport rambo
import numpy as np
cimport numpy as np
from scipy.integrate import quad
from libc.math cimport exp, log, M_PI, log10, sqrt, fmax
import cython
include "parameters.pxd"

"""
Module for computing the photon spectrum from radiative kaon decay.

Description:
    The charged kaon has many decay modes:

        k -> mu  + nu
        k -> pi  + pi0
        k -> pi  + pi  + pi
        k -> pi0 + e   + nu
        k -> pi0 + mu  + nu
        k -> pi  + pi0 + pi0

    For the the two-body final states, the sum of the decay spectra are
    computed given the known energies of the final state particles in the
    kaon's rest frame. The spectrum is then boosted into the lab frame.

    For the three-body final state, the energies of the final state
    particles are computed using RAMBO. The spectra for each final state
    particle is computed are each point in phases space in the charged kaon's rest frame and then spectra are summed over. The spectra is then boosted into the lab frame.

    Attributes:
        __num_ps_pts   : Number of phase space points to use for RAMBO in
                         creating energies distributions for three-body
                         final states. Set to 1000 by default.

        __num_bins     : Number of bins to use for energies distributions of
                         three-body final states (i.e. the number of
                         energies to use.) Set to 10 by default.

        __msPiPiPi     : Array of masses for the three charged pion final
                         state.

        __msPi0MuNu    : Array of masses for the pi0 + mu + nu final state.

        __probsPiPiPi  : Array storing the energies and probabilities of the
                         three charged pion final state.

        __probsPi0MuNu : Array storing the energies and probabilities of the
                        pi0 + mu + nu final state.

        __ram          : Rambo object used to create energy distributions.

        __muon         : Muon object to compute muon decay and FSR spectra.

        __neuPion      : Neutal pion object to compute decay and FSR
                         spectra.

        __chrgpi       : Charged pion object to compute decay and FSR
                        spectra.

        __funcsPiPiPi  : List of functions to compute decay and FSR spectrum
                        from the three charged pion final state.

        __funcsPi0MuNu : List of functions to compute decay and FSR spectrum
                        from the pi0 + mu + nu final state.
"""
cdef int __num_ps_pts
cdef int __num_bins

cdef np.ndarray __msPiPiPi
cdef np.ndarray __msPi0MuNu

cdef np.ndarray __probsPiPiPi
cdef np.ndarray __probsPi0MuNu

cdef np.ndarray __funcsPiPiPi
cdef np.ndarray __funcsPi0MuNu

cdef rambo.Rambo __ram

__num_ps_pts = 1000
__num_bins = 10

__msPiPiPi = np.ndarray([MASS_PI, MASS_PI, MASS_PI], dtype=np.float64)
__msPi0MuNu = np.ndarray([MASS_PI0, MASS_MU, 0.0], dtype=np.float64)

__probsPiPiPi = np.zeros((3, 2, __num_bins), dtype=np.float64)
__probsPi0MuNu = np.zeros((3, 2, __num_bins), dtype=np.float64)

__ram = rambo.Rambo()

__probsPiPiPi = __ram.generate_energy_histogram(__num_ps_pts, __msPiPiPi,
                                                MASS_K, __num_bins)

__probsPi0MuNu = __ram.generate_energy_histogram(__num_ps_pts, __msPi0MuNu,
                                                 MASS_K, __num_bins)

__funcsPiPiPi = np.array([decay_charged_pion.SpectrumPoint, \
                          decay_charged_pion.SpectrumPoint, \
                          decay_charged_pion.SpectrumPoint])

__funcsPi0MuNu = np.array([decay_muon.SpectrumPoint,\
                           decay_neutral_pion.SpectrumPoint])


@cython.cdivision(True)
cdef double __integrand2(double cl, double eng_gam, double eng_k):
    """
    Integrand for K -> X, where X is a two body final state. The X's
    used are
        mu + nu
        pi  + pi0
    Keyword arguments::
        cl: Angle of photon w.r.t. charged kaon in lab frame.
        eng_gam: Energy of photon in laboratory frame.
        eng_k: Energy of kaon in laboratory frame.
    """

    cdef double gamma_k = eng_k / MASS_K
    cdef double beta_k = sqrt(1.0 - (MASS_K / eng_k)**2)
    cdef double eng_gam_k_rf = eng_gam * gamma_k * (1.0 - beta_k * cl)

    cdef int i, j
    cdef double ret_val = 0.0
    cdef double pre_factor \
        = 1.0 / (2.0 * gamma_k * (1.0 - beta_k * cl))

    eng_mu = (MASS_K**2 + MASS_MU**2) / (2.0 * MASS_K)
    eng_pi = (MASS_K**2 + MASS_PI**2 - MASS_PI0**2) / (2.0 * MASS_K)
    eng_pi0 = (MASS_K**2 - MASS_PI**2 + MASS_PI0**2) / (2.0 * MASS_K)

    ret_val += BR_K_TO_MUNU * \
        decay_muon.SpectrumPoint(eng_gam_k_rf, eng_mu)
    ret_val += BR_K_TO_PIPI0 * \
        decay_charged_pion.SpectrumPoint(eng_gam_k_rf, eng_pi)
    ret_val += BR_K_TO_PIPI0 * \
        decay_neutral_pion.SpectrumPoint(eng_gam_k_rf, eng_pi0)

    return pre_factor * ret_val

@cython.cdivision(True)
cdef double __integrand3(double cl, double eng_gam, double eng_k):
    """
    Integrand for K -> X, where X is a three body final state. The X's
    used are
        pi + pi + pi
        pi0 + mu + nu.
    When the ChargedKaon object is instatiated, the energies of the FSP are
    computed using RAMBO and energy distributions are formed. All the
    energies from the energy distributions are summed over against their
    weights.

    Keyword arguments::
        cl: Angle of photon w.r.t. charged kaon in lab frame.
        eng_gam: Energy of photon in laboratory frame.
        eng_k: Energy of kaon in laboratory frame.
    """

    cdef double gamma_k = eng_k / MASS_K
    cdef double beta_k = sqrt(1.0 - (MASS_K / eng_k)**2)
    cdef double eng_gam_k_rf = eng_gam * gamma_k * (1.0 - beta_k * cl)

    cdef int i, j
    cdef double ret_val = 0.0
    cdef double eng, weight
    cdef double pre_factor \
        = 1.0 / (2.0 * gamma_k * (1.0 - beta_k * cl))

    cdef int lenPiPiPi = int(len(__funcsPiPiPi))
    cdef int lenPi0MuNu = int(len(__funcsPi0MuNu))

    for i in range(3):
        for j in range(__num_bins):
            # K -> pi + pi + pi
            if i < lenPiPiPi:
                eng = __probsPiPiPi[i, 0, j]
                weight = __probsPiPiPi[i, 1, j]
                ret_val += BR_K_TO_3PI * weight \
                    * __funcsPiPiPi[i](eng_gam_k_rf, eng)
            # K -> pi0 + mu + nu
            if i < lenPi0MuNu:
                eng = __probsPi0MuNu[i, 0, j]
                weight = __probsPi0MuNu[i, 1, j]
                ret_val += BR_K_TO_PI0MUNU * weight \
                    * __funcsPi0MuNu[i](eng_gam_k_rf, eng)

    return pre_factor * ret_val


cdef double __integrand(double cl, double eng_gam, double eng_k):
    """
    Integrand for K -> X, where X is a any final state. The X's
    used are
        mu + nu
        pi  + pi0
        pi + pi + pi
        pi0 + mu + nu.
    When the ChargedKaon object is instatiated, the energies of the FSP are
    computed using RAMBO and energy distributions are formed. All the
    energies from the energy distributions are summed over against their
    weights.

    Keyword arguments::
        cl: Angle of photon w.r.t. charged kaon in lab frame.
        eng_gam: Energy of photon in laboratory frame.
        eng_k: Energy of kaon in laboratory frame.
    """

    cdef double ret_val = 0.0

    ret_val += __integrand2(cl, eng_gam, eng_k)
    ret_val += __integrand3(cl, eng_gam, eng_k)

    return ret_val


def SpectrumPoint(double eng_gam, double eng_k):
    """
    Returns the radiative spectrum value from charged kaon at
    a single gamma ray energy.

    Keyword arguments::
        eng_gam: Energy of photon is laboratory frame.
        eng_k: Energy of charged kaon in laboratory frame.
    """
    cdef double result = 0.0

    return quad(__integrand, -1.0, 1.0, points=[-1.0, 1.0], \
                  args=(eng_gam, eng_k), epsabs=10**-10., \
                  epsrel=10**-4.)[0]


def Spectrum(np.ndarray eng_gams, double eng_k):
    """
    Returns the radiative spectrum dNde from charged kaon for a
    list of gamma ray energies.

    Keyword arguments::
        eng_gams: List of energies of photon in laboratory frame.
        eng_k: Energy of charged kaon in laboratory frame.
    """
    cdef double result = 0.0

    cdef int numpts = len(eng_gams)

    cdef np.ndarray spec = np.zeros(numpts, dtype=np.float64)

    cdef int i = 0

    for i in range(numpts):
        spec[i] = quad(__integrand, -1.0, 1.0, points=[-1.0, 1.0], \
                       args=(eng_gams[i], eng_k), epsabs=10**-10., \
                       epsrel=10**-4.)[0]

    return spec
