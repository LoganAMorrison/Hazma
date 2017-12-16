import numpy as np
cimport numpy as np
from ..phases_space_generator cimport rambo

cdef int __num_ps_pts
cdef int __num_bins

cdef np.ndarray __msPiPiPi
cdef np.ndarray __msPi0MuNu

cdef np.ndarray __probsPiPiPi
cdef np.ndarray __probsPi0MuNu

cdef double __eng_mu_k_rf
cdef double __eng_pi_k_rf
cdef double __eng_pi0_k_rf

cdef np.ndarray __spec_PiPi0
cdef np.ndarray __spec_MuNu
cdef np.ndarray __spec_Pi0MuNu
cdef np.ndarray __spec_PiPiPi

cdef rambo.Rambo __ram

cdef double __integrand2(double, double, double)
cdef double __integrand3(double, double, double)
cdef double __integrand(double, double, double)
