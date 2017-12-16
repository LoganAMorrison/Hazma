import numpy as np
cimport numpy as np

cdef double __eng_gam_max_mu_rf
cdef double__eng_mu_pi_rf
cdef np.ndarray __eng_gams_mu
cdef np.ndarray __mu_spec

cdef double __muon_spectrum(double)
cdef double __gamma(double, double)
cdef double __beta(double, double)
cdef double __eng_gam_max(double)
cdef double __integrand(double, double, double)

cdef double CSpectrumPoint(double, double)
cdef np.ndarray CSpectrum(np.ndarray, double)
