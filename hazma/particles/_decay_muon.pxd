cdef class Muon:
    cdef float __JPlus(self, float y)
    cdef float __JMinus(self, float y)
    cdef float __dBdy(self, float y)
    cdef float __Gamma(self, float eng, float mass)
    cdef float __Beta(self, float eng, float mass)
    cdef float __Integrand(self, float cl, float engGam, float engMu)
