import numpy as np
cimport numpy as np

cdef double c_short_kaon_decay_spectrum_point(double, double, int)
cdef np.ndarray[np.float64_t,ndim=1] c_short_kaon_decay_spectrum_array(np.ndarray[np.float64_t,ndim=1], double, int)
