import numpy as np
cimport numpy as np

cdef double dnde_photon_charged_rho_point(double, double)
cdef double dnde_photon_neutral_rho_point(double, double)

cdef np.ndarray[np.float64_t,ndim=1] dnde_photon_charged_rho_array(double[:], double)
cdef np.ndarray[np.float64_t,ndim=1] dnde_photon_neutral_rho_array(double[:], double)