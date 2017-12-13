cdef class Muon:
    cdef float __j_plus(self, float y)
    cdef float __j_minus(self, float y)
    cdef float __dBdy(self, float y)
    cdef float __gamma(self, float eng, float mass)
    cdef float __beta(self, float eng, float mass)
    cdef float __integrand(self, float cl, float engGam, float engMu)
