cimport _decay_muon
import numpy as np
cimport numpy as np

cdef class ChargedPion:
    cdef float __gamma(self, float eng, float mass)
    cdef float __beta(self, float eng, float mass)
    cdef float __eng_gam_max(self, float engPi)
    cdef float __integrand(self, float cl, float engGam, float engPi)
    cdef float __eng_gam_max_mu_rf
    cdef float __eng_mu_pi_rf
    cdef float __muon_spectrum(self, float eng_gam)

    cdef _decay_muon.Muon __muon
    cdef np.ndarray __eng_gams_mu
    cdef np.ndarray __mu_spec
