import numpy as np
cimport numpy as np

cdef np.ndarray eng_ps_mu

cdef double __muon_spectrum(double)
cdef double __integrand(double, double, double)
cdef double CSpectrumPoint(double, double)
cdef np.ndarray CSpectrum(np.ndarray, double)
