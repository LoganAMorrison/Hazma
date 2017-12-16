import numpy as np
cimport numpy as np

cdef double __j_plus(double y)
cdef double __j_minus(double y)
cdef double __dBdy(double y)
cdef double __gamma(double eng, double mass)
cdef double __beta(double eng, double mass)
cdef double __integrand(double cl, double engGam, double engMu)
cdef double CSpectrumPoint(double eng_gam, double eng_mu)
cdef np.ndarray CSpectrum(np.ndarray eng_gams, float eng_mu)
