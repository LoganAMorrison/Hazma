import numpy as np
cimport numpy as np

cdef np.ndarray __e_gams_total
cdef np.ndarray __e_gams_0enu
cdef np.ndarray __e_gams_0munu
cdef np.ndarray __e_gams_00p
cdef np.ndarray __e_gams_mmug
cdef np.ndarray __e_gams_munu
cdef np.ndarray __e_gams_p0
cdef np.ndarray __e_gams_p0g
cdef np.ndarray __e_gams_ppm

cdef np.ndarray __spec_total
cdef np.ndarray __spec_0enu
cdef np.ndarray __spec_0munu
cdef np.ndarray __spec_00p
cdef np.ndarray __spec_mmug
cdef np.ndarray __spec_munu
cdef np.ndarray __spec_p0
cdef np.ndarray __spec_p0g
cdef np.ndarray __spec_ppm

cdef double __interp_spec(double, str)

cdef double __integrand(double, double, double, str)

cdef double CSpectrumPoint(double, double, str)

cdef np.ndarray CSpectrum(np.ndarray[np.float64_t, ndim=1], double, str)
