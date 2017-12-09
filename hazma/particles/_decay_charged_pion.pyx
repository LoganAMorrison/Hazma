from ._decay_muon import Muon
import numpy as np
cimport numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from libc.math cimport exp, log, M_PI, log10, sqrt
import cython
from functools import partial
include "parameters.pxd"


cdef float engGamMaxMuRF = (MASS_MU**2.0 - MASS_E**2.0) / (2.0 * MASS_MU)
cdef float engMuPiRF = (MASS_PI**2.0 + MASS_MU**2.0) / (2.0 * MASS_PI)

mu = Muon()

cdef np.ndarray eng_gams_mu = np.logspace(-5.5, 3.0, num=10000, dtype=float)

mu_spec = mu.Spectrum(eng_gams_mu, engMuPiRF)
__muSpectrum = interp1d(eng_gams_mu, mu_spec, kind='linear')

cdef class ChargedPion:
    """
    Class for computing the photon spectrum from radiative charged Pion decay.
    """

    def __init__(self):
        pass

    @cython.cdivision(True)
    cdef float __Gamma(self, float eng, float mass):
        """
        Returns special relativity boost factor gamma.

        Keyword arguments::
            eng -- energy of particle.
            mass -- mass of particle.
        """
        return eng / mass

    @cython.cdivision(True)
    cdef float __Beta(self, float eng, float mass):
        """
        Returns velocity in natural units.

        Keyword arguments::
            eng -- energy of particle.
            mass -- mass of particle.
        """
        return sqrt(1.0 - (mass / eng)**2.0)

    cdef float __EngGamMax(self, float engPi):
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
        cdef float betaPi = self.__Beta(engPi, MASS_PI)
        cdef float gammaPi = self.__Gamma(engPi, MASS_PI)

        cdef float betaMu = self.__Beta(engMuPiRF, MASS_MU)
        cdef float gammaMu = self.__Gamma(engMuPiRF, MASS_MU)

        return engGamMaxMuRF * gammaPi * gammaMu * \
            (1.0 + betaPi) * (1.0 + betaMu)


    @cython.cdivision(True)
    cdef float __Integrand(self, float cl, float engGam, float engPi):
        """
        Returns the integrand of the differential radiative decay spectrum for
        the charged pion.

        Keyword arguments::
            cl: Angle of photon w.r.t. charged pion in lab frame.
            engPi: Energy of photon in laboratory frame.
            engPi: Energy of pion in laboratory frame.
        """
        cdef float betaPi = self.__Beta(engPi, MASS_PI)
        cdef float gammaPi = self.__Gamma(engPi, MASS_PI)

        cdef float engGamPiRF = engGam * gammaPi * (1.0 - betaPi * cl)

        # cdef float gamBound = engGamMaxMuRF * gammaPi * (1.0 + betaPi)

        cdef float preFactor = BR_PI_TO_MUNU \
            / (2.0 * gammaPi * np.abs(1.0 - betaPi * cl))

        # cdef float result = 0.0

        # if 0.0 < engGamPiRF and engGamPiRF < gamBound:
        #     result = preFactor * __muSpectrum(engGamPiRF)

        return preFactor * __muSpectrum(engGamPiRF)

    def SpectrumPoint(self, float engGam, float engPi):
        """
        Returns the radiative spectrum value from charged pion given a gamma
        ray energy Egam and charged pion energy Epi energy eng_mu. When the
        ChargedPion object is instatiated, an interplating function for the
        mu spectrum is computed.

        Keyword arguments::
            engGam: Energy of photon is laboratory frame.
            engPi: Energy of charged pion in laboratory frame.
        """
        cdef float result = 0.0

        integrand = partial(self.__Integrand, self)

        if 0.0 <= engGam and engGam <= self.__EngGamMax(engPi):
            result = quad(integrand, -1.0, 1.0, args=(engGam, engPi))[0]

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

        integrand = partial(self.__Integrand, self)

        cdef int numpts = len(eng_gams)

        cdef np.ndarray spec = np.zeros(numpts, dtype=np.float32)

        cdef int i = 0

        for i in range(numpts):
            if 0.0 <= eng_gams[i] and eng_gams[i] <= self.__EngGamMax(eng_pi):
                spec[i] = quad(integrand, -1.0, 1.0, args=(eng_gams[i], eng_pi))[0]

        return spec
