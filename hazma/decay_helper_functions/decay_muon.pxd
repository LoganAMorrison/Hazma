cdef class Muon:
    cdef double __j_plus(self, double y)
    cdef double __j_minus(self, double y)
    cdef double __dBdy(self, double y)
    cdef double __gamma(self, double eng, double mass)
    cdef double __beta(self, double eng, double mass)
    cdef double __integrand(self, double cl, double engGam, double engMu)

    cdef np.ndarray CSpectrum(self, np.ndarray eng_gams, float eng_mu)
