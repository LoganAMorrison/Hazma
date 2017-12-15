cimport decay_muon
import numpy as np
cimport numpy as np

cdef class ChargedPion:
    cdef double __gamma(self, double eng, double mass)
    cdef double __beta(self, double eng, double mass)
    cdef double __eng_gam_max(self, double engPi)
    cdef double __integrand(self, double cl, double engGam, double engPi)
    cdef double __eng_gam_max_mu_rf
    cdef double __eng_mu_pi_rf
    cdef double __muon_spectrum(self, double eng_gam)

    cdef decay_muon.Muon __muon
    cdef np.ndarray __eng_gams_mu
    cdef np.ndarray __mu_spec
    #cpdef SpectrumPoint(self, double eng_gam, double eng_pi)
    #cpdef Spectrum(self, np.ndarray[np.float64_t, ndim=1] eng_gams, \
    #               double eng_pi)
