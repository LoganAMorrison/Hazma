cimport _decay_muon
cimport _decay_charged_pion
cimport _decay_neutral_pion
from . import electron
from ..rambo cimport rambo
import numpy as np
cimport numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from libc.math cimport exp, log, M_PI, log10, sqrt
import cython
from functools import partial
include "parameters.pxd"



cdef int __num_ps_pts = 1000
cdef int __num_bins = 10

# Mass arrays
cdef np.ndarray __msPiPiPi = np.array([MASS_PI, MASS_PI, MASS_PI], \
    dtype=np.float64)

cdef np.ndarray __msPi0MuNu = np.array([MASS_PI0, MASS_MU, 0.0],\
    dtype=np.float64)

# Energy probability distributions
cdef np.ndarray __probsPiPiPi = np.zeros((3, 2, __num_bins), dtype=float)
cdef np.ndarray __probsPi0MuNu = np.zeros((3, 2, __num_bins), dtype=float)


cdef rambo.Rambo __ram = rambo.Rambo()

__probsPiPiPi \
    = __ram.generate_energy_histogram(__num_ps_pts, __msPiPiPi, MASS_K, __num_bins)

__probsPi0MuNu \
    = __ram.generate_energy_histogram(__num_ps_pts, __msPi0MuNu, MASS_K, __num_bins)


cdef _decay_muon.Muon __muon = _decay_muon.Muon()
cdef _decay_neutral_pion.NeutralPion __neuPion \
    = _decay_neutral_pion.NeutralPion()
cdef _decay_charged_pion.ChargedPion __chrgpi \
    = _decay_charged_pion.ChargedPion()


# K -> pi + pi + pi
__funcsPiPiPi = [__chrgpi.SpectrumPoint, __chrgpi.SpectrumPoint, \
               __chrgpi.SpectrumPoint]

# K -> pi0 + mu + nu
__funcsPi0MuNu = [__muon.SpectrumPoint, __neuPion.decay_spectra_point]

cdef class ChargedKaon:
    """
    Class for computing the photon spectrum from radiative kaon decay.
    """

    def __init__(self):
        pass


    """ Integrand for K -> mu + mu """
    cdef float __integrandMuNu(self, float cl, float eng_gam, float eng_k):

        cdef float gamma_k = eng_k / MASS_K
        cdef float beta_k = sqrt(1.0 - (MASS_K / eng_k)**2)
        cdef float eng_gam_k_rf = eng_gam * gamma_k * (1.0 - beta_k * cl)

        cdef int i, j
        cdef float ret_val = 0.0
        cdef float eng, weight
        cdef float pre_factor \
            = BR_K_TO_MUNU / (2.0 * gamma_k * (1.0 - beta_k * cl))

        eng_mu = (MASS_K**2 + MASS_MU**2) / (2.0 * MASS_K)
        ret_val = __muon.SpectrumPoint(eng_gam, eng_mu)

        return pre_factor * ret_val


    """ Integrand for K -> pi + pi + pi and K -> pi0 + mu + nu """
    cdef float __integrand3(self, float cl, float eng_gam, float eng_k):

        cdef float gamma_k = eng_k / MASS_K
        cdef float beta_k = sqrt(1.0 - (MASS_K / eng_k)**2)
        cdef float eng_gam_k_rf = eng_gam * gamma_k * (1.0 - beta_k * cl)

        cdef int i, j
        cdef float ret_val = 0.0
        cdef float eng, weight
        cdef float pre_factor \
            = 1.0 / (2.0 * gamma_k * (1.0 - beta_k * cl))

        cdef int size = max(len(__funcsPiPiPi), len(__funcsPi0MuNu))

        for i in range(size):
            for j in range(__num_bins):
                # K -> pi + pi + pi
                if i < len(__funcsPiPiPi):
                    eng = __probsPiPiPi[i, 0, j]
                    weight = __probsPiPiPi[i, 1, j]
                    ret_val += BR_K_TO_3PI * weight \
                        * __funcsPiPiPi[i](eng_gam, eng)
                # K -> pi0 + mu + nu
                if i < len(__funcsPi0MuNu):
                    eng = __probsPi0MuNu[i, 0, j]
                    weight = __probsPi0MuNu[i, 1, j]
                    ret_val += BR_K_TO_PI0MUNU * weight \
                        * __funcsPi0MuNu[i](eng_gam, eng)

        return pre_factor * ret_val


    """ Integrand for K -> sum_{X} X """
    cdef float __integrand(self, float cl, float eng_gam, float eng_k):

        cdef float ret_val = 0.0

        ret_val += self.__integrandMuNu(cl, eng_gam, eng_k)
        ret_val += self.__integrand3(cl, eng_gam, eng_k)

        return ret_val


    def SpectrumPoint(self, float eng_gam, float eng_k):
        """
        Returns the radiative spectrum value from charged kaon.
        """
        cdef float result = 0.0

        integrand = partial(self.__integrand, self)

        return quad(integrand, -1.0, 1.0, args=(eng_gam, eng_k))[0]


    def Spectrum(self, np.ndarray eng_gams, float eng_k):
        """
        Returns the radiative spectrum dNde from charged kaon.

        Keyword arguments::
            engGamMin: Minimum energy of photon is laboratory frame.
            engGamMax: Minimum energy of photon is laboratory frame.
            engPi: Energy of charged pion in laboratory frame.
        """
        cdef float result = 0.0

        integrand = partial(self.__integrand, self)

        cdef int numpts = len(eng_gams)

        cdef np.ndarray spec = np.zeros(numpts, dtype=np.float32)

        cdef int i = 0

        for i in range(numpts):
            spec[i] = quad(integrand, -1.0, 1.0, args=(eng_gams[i], eng_k))[0]

        return spec






"""
Add these later...

cdef np.ndarray __mssPi0ENu = np.array([MASS_PI0, MASS_E, 0.0],\
    dtype=np.float64)
cdef np.ndarray __msPiPiPi0Pi0 = np.array([MASS_PI, MASS_PI, MASS_PI0, \
    MASS_PI0], dtype=np.float64)

cdef np.ndarray __probsPi0ENu = np.zero((3, 2, __num_bins), dtype=float)
cdef np.ndarray __probsPiPiPi0Pi0 = np.zero((3, 2, __num_bins), dtype=float)

__probsPi0ENu \
    = __ram.generate_energy_histogram(__num_ps_pts, __mssPi0ENu, MASS_K, 25)

__probsPiPiPi0Pi0 \
    = __ram.generate_energy_histogram(__num_ps_pts,__msPiPiPi0Pi0, MASS_K, 25)
"""
