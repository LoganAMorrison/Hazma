import numpy as np
cimport numpy as np

cdef double __j_plus(double)
cdef double __j_minus(double)
cdef double __dBdy(double)
cdef double __gamma(double, double)
cdef double __beta(double, double)
cdef double __integrand(double, double, double)
cdef double CSpectrumPoint(double, double)
cdef np.ndarray CSpectrum(np.ndarray, double)
