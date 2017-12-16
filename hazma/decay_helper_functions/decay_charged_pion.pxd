import numpy as np
cimport numpy as np

cdef double __eng_gam_max_mu_rf
cdef double__eng_mu_pi_rf
cdef np.ndarray __eng_gams_mu
cdef np.ndarray __mu_spec

cdef double __muon_spectrum(double eng_gam)
cdef double __gamma(double eng, double mass)
cdef double __beta(double eng, double mass)
cdef double __eng_gam_max(double eng_pi)
cdef double __integrand(double cl, double eng_gam, double eng_pi)

cdef double CSpectrumPoint(double eng_gam, double eng_pi)
cdef np.ndarray CSpectrum(np.ndarray eng_gams, double eng_pi)
