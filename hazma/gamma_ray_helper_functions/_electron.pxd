import numpy as np
cimport numpy as np

cdef double decay_spectra_point(double eng_gam, double eng_mu)
cdef np.ndarray decay_spectra(np.ndarray eng_gam, double eng_mu)
