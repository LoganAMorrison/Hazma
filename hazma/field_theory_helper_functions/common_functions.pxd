from libcpp.vector cimport vector

cdef double c_minkowski_dot(const vector[double]&, const vector[double]&)
cdef double c_cross_section_prefactor(double, double, double)
