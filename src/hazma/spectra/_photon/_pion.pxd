import numpy as np
cimport numpy as np

cdef double dnde_photon_neutral_pion_point(double, double)
cdef np.ndarray[np.float64_t,ndim=1] dnde_photon_neutral_pion_array(double[:], double)

cdef double dnde_photon_charged_pion_point(double, double)
cdef np.ndarray[np.float64_t,ndim=1] dnde_photon_charged_pion_array(double[:], double)