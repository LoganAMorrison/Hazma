import numpy as np
cimport numpy as np

cdef double __interp_spec(double)
cdef np.ndarray __data
cdef np.ndarray __eng_gams
cdef np.ndarray __spec
cdef double __integrand(double, double, double)

cdef double CSpectrumPoint(double, double)
cdef np.ndarray CSpectrum(np.ndarray[np.float64_t, ndim=1], double)
