cimport _decay_muon
import numpy as np
cimport numpy as np
from scipy.integrate import quad
from libc.math cimport exp, log, M_PI, log10, sqrt, abs
import cython
from functools import partial
include "parameters.pxd"

cdef class ChargedPion:
    """
    Class for computing the photon spectrum from radiative charged Pion decay.
    """

    def __init__(self):
        pass

    def __cinit__(self):
        self.__eng_gam_max_mu_rf \
            = (MASS_MU**2.0 - MASS_E**2.0) / (2.0 * MASS_MU)
        self.__eng_mu_pi_rf = (MASS_PI**2.0 + MASS_MU**2.0) / (2.0 * MASS_PI)

        self.__eng_gams_mu = np.logspace(-5.5, 3.0, num=10000, dtype=float)

        self.__muon = _decay_muon.Muon()

        self.__mu_spec = self.__muon.Spectrum(self.__eng_gams_mu,\
                                              self.__eng_mu_pi_rf)

    def __dealloc__(self):
        pass

    cdef float __muon_spectrum(self, float eng_gam):
        return np.interp(eng_gam, self.__eng_gams_mu, self.__mu_spec)

    @cython.cdivision(True)
    cdef float __gamma(self, float eng, float mass):
        """
        Returns special relativity boost factor gamma.

        Keyword arguments::
            eng -- energy of particle.
            mass -- mass of particle.
        """
        return eng / mass

    @cython.cdivision(True)
    cdef float __beta(self, float eng, float mass):
        """
        Returns velocity in natural units.

        Keyword arguments::
            eng -- energy of particle.
            mass -- mass of particle.
        """
        return sqrt(1.0 - (mass / eng)**2.0)

    cdef float __eng_gam_max(self, float engPi):
        """
        Returns the maximum allowed gamma ray energy from a charged pion decay.

        Keyword arguments::
            engPi: Energy of pion in laboratory frame.

        More details:
            This is computed using the fact that in the mu restframe, the
            Maximum allowed value is
                eng_gam_max_mu_rf = (pow(mass_mu,2.0) - pow(mass_e,2.0))
                                / (2.0 * mass_mu).
            Then, boosting into the pion rest frame, then to the mu rest
            frame, we get the maximum allowed energy in the lab frame.
        """
        cdef float betaPi = self.__beta(engPi, MASS_PI)
        cdef float gammaPi = self.__gamma(engPi, MASS_PI)

        cdef float betaMu = self.__beta(self.__eng_mu_pi_rf, MASS_MU)
        cdef float gammaMu = self.__gamma(self.__eng_mu_pi_rf, MASS_MU)

        return self.__eng_gam_max_mu_rf * gammaPi * gammaMu * \
            (1.0 + betaPi) * (1.0 + betaMu)


    @cython.cdivision(True)
    cdef float __integrand(self, float cl, float eng_gam, float eng_pi):
        """
        Returns the integrand of the differential radiative decay spectrum for
        the charged pion.

        Keyword arguments::
            cl: Angle of photon w.r.t. charged pion in lab frame.
            engPi: Energy of photon in laboratory frame.
            engPi: Energy of pion in laboratory frame.
        """
        cdef float betaPi = self.__beta(eng_pi, MASS_PI)
        cdef float gammaPi = self.__gamma(eng_pi, MASS_PI)

        cdef float engGamPiRF = eng_gam * gammaPi * (1.0 - betaPi * cl)

        cdef float preFactor = BR_PI_TO_MUNU \
            / (2.0 * gammaPi * abs(1.0 - betaPi * cl))

        return preFactor * self.__muon_spectrum(engGamPiRF)

    def SpectrumPoint(self, float eng_gam, float eng_pi):
        """
        Returns the radiative spectrum value from charged pion given a gamma
        ray energy Egam and charged pion energy Epi energy eng_mu. When the
        ChargedPion object is instatiated, an interplating function for the
        mu spectrum is computed.

        Keyword arguments::
            eng_gam: Energy of photon is laboratory frame.
            eng_pi: Energy of charged pion in laboratory frame.
        """
        cdef float result = 0.0

        integrand = partial(self.__integrand, self)

        if 0.0 <= eng_gam and eng_gam <= self.__eng_gam_max(eng_pi):
            result = quad(integrand, -1.0, 1.0, points=[-1.0, 1.0], \
                          args=(eng_gam, eng_pi), epsabs=10**-4., \
                          epsrel=10**-10.)[0]

        return result


    def Spectrum(self, np.ndarray eng_gams, float eng_pi):
        """
        Returns the radiative spectrum dNde from charged pion given a gamma
        ray energy Egam and charged pion energy Epi energy eng_mu. When the
        ChargedPion object is instatiated, an interplating function for the
        mu spectrum is computed.

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
            if 0.0 <= eng_gams[i] and eng_gams[i] <= self.__eng_gam_max(eng_pi):
                spec[i] = quad(integrand, -1.0, 1.0, points=[-1.0, 1.0], \
                               args=(eng_gams[i], eng_pi), epsabs=10**-4., \
                               epsrel=10**-10.)[0]

        return spec
