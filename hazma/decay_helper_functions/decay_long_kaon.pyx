cimport decay_muon
cimport decay_charged_pion
cimport decay_neutral_pion
from ..phases_space_generator cimport rambo
import numpy as np
cimport numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp2d
from libc.math cimport exp, log, M_PI, log10, sqrt, fmax
import cython
include "parameters.pxd"

"""
Module for computing the photon spectrum from radiative long kaon decay.

Description:
    The charged kaon has many decay modes:

    kl    -> pi  + e   + nu
    kl    -> pi  + mu  + nu
    kl    -> pi0 + pi0  + pi0
    kl    -> pi  + pi  + pi0

    For the three-body final state, the energies of the final state
    particles are computed using RAMBO. The spectra for each final state
    particle is computed are each point in phases space in the charged kaon's rest frame and then spectra are summed over. The spectra is then boosted into the lab frame.
"""

""" Distributions for 3-body final states """
# Number of phase space points to use for RAMBO in creating energies
# distributions for three-body final states. Set to 1000 by default.
__num_ps_pts = 1000
# Number of bins to use for energies distributions of three-body final
# states (i.e. the number of energies to use.) Set to 10 by default.
__num_bins = 10
# Array of masses for the pi  + e   + nu final state.
__msPIENU = np.array([MASS_PI, MASS_E, 0.0])
# Array of masses for the pi  + mu  + nu final
__msPIMUNU = np.array([MASS_PI, MASS_MU, 0.0])
# Array of masses for the pi0 + pi0  + pi0 final
__ms3PI0 = np.array([MASS_PI0, MASS_PI0, MASS_PI0])
# Array of masses for the pi  + pi  + pi0 final
__ms2PIPI0 = np.array([MASS_PI, MASS_PI, MASS_PI0])
# Array storing the energies and probabilities of the 3-body final states
__probsPIENU = np.zeros((3, 2, __num_bins), dtype=np.float64)
__probsPIMUNU = np.zeros((3, 2, __num_bins), dtype=np.float64)
__probs3PI0 = np.zeros((3, 2, __num_bins), dtype=np.float64)
__probs2PIPI0 = np.zeros((3, 2, __num_bins), dtype=np.float64)
# Rambo object used to create energy distributions.
__ram = rambo.Rambo()
# Call rambo to generate energ distributions for pi  + e   + nu final state.
__probsPIENU = __ram.generate_energy_histogram(__num_ps_pts, __msPIENU,
                                               MASS_K, __num_bins)
# Call rambo to generate energ distributions for pi  + mu  + nu final state.
__probsPIMUNU = __ram.generate_energy_histogram(__num_ps_pts, __msPIMUNU,
                                               MASS_K, __num_bins)
# Call rambo to generate energ distributions for pi0 + pi0  + pi0 final state.
__probs3PI0 = __ram.generate_energy_histogram(__num_ps_pts, __ms3PI0,
                                              MASS_K, __num_bins)
# Call rambo to generate energ distributions for pi  + pi  + pi0  final state.
__probs2PIPI0 = __ram.generate_energy_histogram(__num_ps_pts, __ms2PIPI0,
                                                MASS_K, __num_bins)

""" Interpolating spectrum functions """
# Gamma ray energies for interpolating functions. Need a very low lower bound
# in order to no pass outside interpolation bounds when called from kaon decay.
__eng_gams_interp = np.logspace(-5.5, 3.0, num=10000, dtype=np.float64)

__spec_PIENU = np.zeros(10000, dtype=np.float64)
__spec_PIMUNU = np.zeros(10000, dtype=np.float64)
__spec_3PI0 = np.zeros(10000, dtype=np.float64)
__spec_2PIPI0 = np.zeros(10000, dtype=np.float64)

cdef int k
for k in range(__num_bins):
    __spec_PIENU += __probsPIENU[0, 0, k] * \
        decay_charged_pion.CSpectrum(__eng_gams_interp, __probsPIENU[0, 1, k])
    __spec_PIENU += __probsPIENU[1, 0, k] * \
        decay_muon.CSpectrum(__eng_gams_interp, __probsPIENU[1, 1, k])

    __spec_PIMUNU += __probsPIMUNU[0, 0, k] * \
        decay_charged_pion.CSpectrum(__eng_gams_interp, __probsPIMUNU[0, 1, k])
    __spec_PIMUNU += __probsPIMUNU[1, 0, k] * \
        decay_muon.CSpectrum(__eng_gams_interp, __probsPIMUNU[1, 1, k])

    __spec_3PI0 += __probs3PI0[0, 0, k] * \
        decay_neutral_pion.CSpectrum(__eng_gams_interp, __probs3PI0[0, 1, k])
    __spec_3PI0 += __probs3PI0[1, 0, k] * \
        decay_neutral_pion.CSpectrum(__eng_gams_interp, __probs3PI0[1, 1, k])
    __spec_3PI0 += __probs3PI0[2, 0, k] * \
        decay_neutral_pion.CSpectrum(__eng_gams_interp, __probs3PI0[2, 1, k])

    __spec_2PIPI0 += __probs2PIPI0[0, 0, k] * \
        decay_charged_pion.CSpectrum(__eng_gams_interp, __probs2PIPI0[0, 1, k])
    __spec_2PIPI0 += __probs2PIPI0[1, 0, k] * \
        decay_charged_pion.CSpectrum(__eng_gams_interp, __probs2PIPI0[1, 1, k])
    __spec_2PIPI0 += __probs2PIPI0[2, 0, k] * \
        decay_neutral_pion.CSpectrum(__eng_gams_interp, __probs2PIPI0[2, 1, k])


cdef double __interp_PIENU(double eng_gam):
    return np.interp(eng_gam, __eng_gams_interp, __spec_PIENU)

cdef double __interp_PIMUNU(double eng_gam):
    return np.interp(eng_gam, __eng_gams_interp, __spec_PIMUNU)

cdef double __interp_3PI0(double eng_gam):
    return np.interp(eng_gam, __eng_gams_interp, __spec_3PI0)

cdef double __interp_2PIPI0(double eng_gam):
    return np.interp(eng_gam, __eng_gams_interp, __spec_2PIPI0)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double __integrand(double cl, double eng_gam, double eng_k):
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

    ret_val += BR_KL_TO_PIENU * __interp_PIENU(eng_gam_k_rf)
    ret_val += BR_KL_TO_PIMUNU * __interp_PIMUNU(eng_gam_k_rf)
    ret_val += BR_KL_TO_3PI0 * __interp_3PI0(eng_gam_k_rf)
    ret_val += BR_KL_TO_2PIPI0 * __interp_2PIPI0(eng_gam_k_rf)

    return pre_factor * ret_val


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

@cython.boundscheck(False)
@cython.wraparound(False)
def Spectrum(np.ndarray[np.float64_t, ndim=1] eng_gams, double eng_k):
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
                       args=(eng_gams[i], eng_k), epsabs=0.0, \
                       epsrel=10**-4.)[0]

    return spec
