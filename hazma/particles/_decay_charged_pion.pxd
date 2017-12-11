cdef class ChargedPion:
    cdef float __Gamma(self, float eng, float mass)
    cdef float __Beta(self, float eng, float mass)
    cdef float __EngGamMax(self, float engPi)
    cdef float __Integrand(self, float cl, float engGam, float engPi)
    cdef float engGamMaxMuRF
