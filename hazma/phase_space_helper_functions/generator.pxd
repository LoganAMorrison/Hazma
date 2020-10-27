import numpy as np
cimport numpy as np

cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        void push_back(T&) nogil except+
        size_t size() nogil
        T& operator[](size_t)

cdef vector[vector[double]] c_generate_space(int num_ps_pts, vector[double] masses, double cme, int num_fsp)
