import numpy as np
cimport numpy as np

cdef class Rambo:
    cdef np.int __num_fsp, __num_phase_space_pts, __event_count
    cdef np.float64_t __cme
    cdef np.float64_t[:] __masses, __weight_array, __randoms
    cdef np.float64_t[:, :] __q_list, __p_list, __k_list
    cdef np.float64_t[:, :, :] __phase_space_array
    cdef np.float64_t __weight

    cdef __initilize(self, np.int num_phase_space_pts, \
                     np.float64_t[:] masses, np.float64_t cme)
    cdef func_xi(self, xi)
    cdef deriv_func_xi(self, xi)
    cdef np.float64_t __find_root(self)
    cdef __get_mass(self, fv)
    cdef __generate_qs(self)
    cdef __generate_ps(self)
    cdef __generate_ks(self)
    cdef __normalize_weights(self)
