import numpy as np
cimport numpy as np
from decay_muon cimport CSpectrum  as muspec

cdef double muon_spectrum(double)
cdef double gamma(double, double)
cdef double beta(double, double)
cdef double eng_gam_max(double)
cdef double integrand(double, double, double, str)

cdef double CSpectrumPoint(double, double, str)
cdef np.ndarray[np.float64_t, ndim=1] CSpectrum(np.ndarray, double, str)
