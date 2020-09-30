from libcpp.vector cimport vector
from libcpp.pair cimport pair

ctypedef double (*msqrd_type_vector)(vector[vector[double]]&, vector[double]&)

cdef pair[vector[double], vector[double]] c_gamma_ray_fsr(
    vector[double] &,
    double,
    vector[double] &,
    vector[double] &,
    double,
    msqrd_type_vector,
    int,
    vector[double]&,
)
