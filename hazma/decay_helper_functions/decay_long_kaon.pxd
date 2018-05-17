import numpy as np
cimport numpy as np

cdef np.ndarray __e_gams_total
cdef np.ndarray __e_gams_000
cdef np.ndarray __e_gams_penu
cdef np.ndarray __e_gams_penug
cdef np.ndarray __e_gams_pm0
cdef np.ndarray __e_gams_pm0g
cdef np.ndarray __e_gams_pmunu
cdef np.ndarray __e_gams_pmunug

cdef np.ndarray __spec_total
cdef np.ndarray __spec_000
cdef np.ndarray __spec_penu
cdef np.ndarray __spec_penug
cdef np.ndarray __spec_pm0
cdef np.ndarray __spec_pm0g
cdef np.ndarray __spec_pmunu
cdef np.ndarray __spec_pmunug

cdef double __interp_spec(double, str)
cdef double __integrand(double, double, double, str)

cdef double CSpectrumPoint(double, double, str)
cdef np.ndarray CSpectrum(np.ndarray[np.float64_t, ndim=1], double, str)
