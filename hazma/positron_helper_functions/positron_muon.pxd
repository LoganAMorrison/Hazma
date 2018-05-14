import numpy as np
cimport numpy as np

cdef double __spectrum_rf(double)
cdef double __integrand(double, double, double)
cdef double CSpectrumPoint(double, double)
cdef np.ndarray CSpectrum(np.ndarray, double)
