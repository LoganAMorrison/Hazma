import numpy as np
cimport numpy as np
from ..phases_space_generator cimport rambo

cdef int __num_ps_pts
cdef int __num_bins

cdef np.ndarray __msPiPiPi
cdef np.ndarray __msPi0MuNu

cdef np.ndarray __probsPiPiPi
cdef np.ndarray __probsPi0MuNu

cdef np.ndarray __funcsPiPiPi
cdef np.ndarray __funcsPi0MuNu

cdef rambo.Rambo __ram

cdef double __integrand2(double cl, double eng_gam, double eng_k)
cdef double __integrand3(double cl, double eng_gam, double eng_k)
cdef double __integrand(double cl, double eng_gam, double eng_k)
