import numpy as np
cimport numpy as np

cdef double dnde_photon_eta_prime_point(double, double)
cdef np.ndarray[np.float64_t,ndim=1] dnde_photon_eta_prime_array(double[:], double)