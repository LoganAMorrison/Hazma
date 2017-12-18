import numpy as np
cimport numpy as np

from ..phase_space_generator import rambo


cdef rambo.Rambo __ram

cdef int __num_engs
cdef int __num_fsp

cdef np.ndarray __masses
cdef np.ndarray __funcs
cdef np.ndarray __probs
cdef np.ndarray __spec
