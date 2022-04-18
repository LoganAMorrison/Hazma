import numpy as np
cimport numpy as np

cdef double dnde_positron_charged_pion_point(double, double)
cdef np.ndarray[np.float64_t,ndim=1] dnde_positron_charged_pion_array(double[:], double)