import numpy as np
cimport numpy as np

cdef double dnde_photon_charged_kaon_point(double, double)
cdef np.ndarray[np.float64_t,ndim=1] dnde_photon_charged_kaon_array(double[:], double)

cdef double dnde_photon_long_kaon_point(double, double)
cdef np.ndarray[np.float64_t,ndim=1] dnde_photon_long_kaon_array(double[:], double)

cdef double dnde_photon_short_kaon_point(double, double)
cdef np.ndarray[np.float64_t,ndim=1] dnde_photon_short_kaon_array(double[:], double)