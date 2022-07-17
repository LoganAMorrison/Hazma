import numpy as np
cimport numpy as np

cdef double CSpectrumPoint(double eng_gam, double eng_mu)
cdef np.ndarray CSpectrum(np.ndarray eng_gams, double eng_mu)
