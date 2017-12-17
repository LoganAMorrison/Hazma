import numpy as np
cimport numpy as np
from ..phases_space_generator cimport rambo

cdef int __num_ps_pts
cdef int __num_bins

cdef np.ndarray __msPIENU
cdef np.ndarray __msPIMUNU
cdef np.ndarray __ms3PI0
cdef np.ndarray __ms2PIPI0

cdef np.ndarray __probsPIENU
cdef np.ndarray __probsPIMUNU
cdef np.ndarray __probs3PI0
cdef np.ndarray __probs2PIPI0

cdef np.ndarray __spec_PIENU
cdef np.ndarray __spec_PIMUNU
cdef np.ndarray __spec_3PI0
cdef np.ndarray __spec_2PIPI0

cdef rambo.Rambo __ram

cdef double __integrand(double, double, double)
