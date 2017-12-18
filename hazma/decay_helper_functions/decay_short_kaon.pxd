import numpy as np
cimport numpy as np

cdef np.ndarray __spec
cdef double __interp_spec(double)
cdef double __integrand(double, double, double)

cdef double CSpectrumPoint(double, double)
cdef np.ndarray CSpectrum(np.ndarray[np.float64_t, ndim=1], double)
