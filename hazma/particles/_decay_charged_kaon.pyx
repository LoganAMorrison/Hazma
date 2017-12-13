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



cdef class ChargedKaon:
    """
    Class for computing the photon spectrum from radiative kaon decay.

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
    """

    def __init__(self):
        pass

    def __cinit__(self):
        """
        Initalized the ChargedKaon object.

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
        self.__num_ps_pts = 1000
        self.__num_bins = 10

        self.__msPiPiPi = np.array([MASS_PI, MASS_PI, MASS_PI], \
            dtype=np.float64)
        self.__msPi0MuNu = np.array([MASS_PI0, MASS_MU, 0.0],\
            dtype=np.float64)

        self.__probsPiPiPi = np.zeros((3, 2, self.__num_bins), dtype=float)
        self.__probsPi0MuNu = np.zeros((3, 2, self.__num_bins), dtype=float)

        self.__ram = rambo.Rambo()

        self.__probsPiPiPi \
            = self.__ram.generate_energy_histogram(self.__num_ps_pts,
                                                   self.__msPiPiPi, MASS_K, self.__num_bins)

        self.__probsPi0MuNu \
            = self.__ram.generate_energy_histogram(self.__num_ps_pts,\
                                                   self.__msPi0MuNu,\
                                                   MASS_K, self.__num_bins)

        self.__muon = _decay_muon.Muon()
        self.__neuPion = _decay_neutral_pion.NeutralPion()
        self.__chrgpi = _decay_charged_pion.ChargedPion()

        self.__funcsPiPiPi = np.array([self.__chrgpi.SpectrumPoint, \
                                       self.__chrgpi.SpectrumPoint, \
                                       self.__chrgpi.SpectrumPoint])

        self.__funcsPi0MuNu = np.array([self.__muon.SpectrumPoint,\
                                        self.__neuPion.SpectrumPoint])

    def __dealloc__(self):
        pass


    cdef float __integrand2(self, float cl, float eng_gam, float eng_k):
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

        cdef float gamma_k = eng_k / MASS_K
        cdef float beta_k = sqrt(1.0 - (MASS_K / eng_k)**2)
        cdef float eng_gam_k_rf = eng_gam * gamma_k * (1.0 - beta_k * cl)

        cdef int i, j
        cdef float ret_val = 0.0
        cdef float pre_factor \
            = 1.0 / (2.0 * gamma_k * (1.0 - beta_k * cl))

        eng_mu = (MASS_K**2 + MASS_MU**2) / (2.0 * MASS_K)
        eng_pi = (MASS_K**2 + MASS_PI**2 - MASS_PI0**2) / (2.0 * MASS_K)
        eng_pi0 = (MASS_K**2 - MASS_PI**2 + MASS_PI0**2) / (2.0 * MASS_K)

        ret_val += BR_K_TO_MUNU * \
            self.__muon.SpectrumPoint(eng_gam_k_rf, eng_mu)
        ret_val += BR_K_TO_PIPI0 * \
            self.__chrgpi.SpectrumPoint(eng_gam_k_rf, eng_pi)
        ret_val += BR_K_TO_PIPI0 * \
            self.__neuPion.SpectrumPoint(eng_gam_k_rf, eng_pi0)

        return pre_factor * ret_val


    cdef float __integrand3(self, float cl, float eng_gam, float eng_k):
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

        cdef float gamma_k = eng_k / MASS_K
        cdef float beta_k = sqrt(1.0 - (MASS_K / eng_k)**2)
        cdef float eng_gam_k_rf = eng_gam * gamma_k * (1.0 - beta_k * cl)

        cdef int i, j
        cdef float ret_val = 0.0
        cdef float eng, weight
        cdef float pre_factor \
            = 1.0 / (2.0 * gamma_k * (1.0 - beta_k * cl))

        cdef int size = max(len(self.__funcsPiPiPi), len(self.__funcsPi0MuNu))

        for i in range(size):
            for j in range(self.__num_bins):
                # K -> pi + pi + pi
                if i < len(self.__funcsPiPiPi):
                    eng = self.__probsPiPiPi[i, 0, j]
                    weight = self.__probsPiPiPi[i, 1, j]
                    ret_val += BR_K_TO_3PI * weight \
                        * self.__funcsPiPiPi[i](eng_gam_k_rf, eng)
                # K -> pi0 + mu + nu
                if i < len(self.__funcsPi0MuNu):
                    eng = self.__probsPi0MuNu[i, 0, j]
                    weight = self.__probsPi0MuNu[i, 1, j]
                    ret_val += BR_K_TO_PI0MUNU * weight \
                        * self.__funcsPi0MuNu[i](eng_gam_k_rf, eng)

        return pre_factor * ret_val


    cdef float __integrand(self, float cl, float eng_gam, float eng_k):
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

        cdef float ret_val = 0.0

        ret_val += self.__integrand2(cl, eng_gam, eng_k)
        ret_val += self.__integrand3(cl, eng_gam, eng_k)

        return ret_val


    def SpectrumPoint(self, float eng_gam, float eng_k):
        """
        Returns the radiative spectrum value from charged kaon at
        a single gamma ray energy.

        Keyword arguments::
            eng_gam: Energy of photon is laboratory frame.
            eng_k: Energy of charged kaon in laboratory frame.
        """
        cdef float result = 0.0

        integrand = partial(self.__integrand, self)

        return quad(integrand, -1.0, 1.0, points=[-1.0, 1.0], \
                      args=(eng_gam, eng_k), epsabs=10**-10., \
                      epsrel=10**-4.)[0]


    def Spectrum(self, np.ndarray eng_gams, float eng_k):
        """
        Returns the radiative spectrum dNde from charged kaon for a
        list of gamma ray energies.

        Keyword arguments::
            eng_gams: List of energies of photon in laboratory frame.
            eng_k: Energy of charged kaon in laboratory frame.
        """
        cdef float result = 0.0

        integrand = partial(self.__integrand, self)

        cdef int numpts = len(eng_gams)

        cdef np.ndarray spec = np.zeros(numpts, dtype=np.float32)

        cdef int i = 0

        for i in range(numpts):
            spec[i] = quad(integrand, -1.0, 1.0, points=[-1.0, 1.0], \
                           args=(eng_gams[i], eng_k), epsabs=10**-10., \
                           epsrel=10**-4.)[0]

        return spec
