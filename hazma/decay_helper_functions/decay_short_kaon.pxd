import numpy as np
cimport numpy as np

cdef np.ndarray __e_gams_total
cdef np.ndarray __e_gams_00
cdef np.ndarray __e_gams_pm
cdef np.ndarray __e_gams_pmg

cdef np.ndarray __spec_total
cdef np.ndarray __spec_00
cdef np.ndarray __spec_pm
cdef np.ndarray __spec_pmg

cdef double __interp_spec(double, str)
cdef double __integrand(double, double, double, str)

cdef double CSpectrumPoint(double, double, str)
cdef np.ndarray CSpectrum(np.ndarray[np.float64_t, ndim=1], double, str)
