
cdef extern from "<limits.h>":
    const double DBL_EPSILON

cdef struct LVector:
    double e
    double px
    double py
    double pz

cdef struct ScalarWf:
    double complex phi
    LVector p

cdef struct VectorWf:
    double complex eps0
    double complex eps1
    double complex eps2
    double complex eps3
    LVector p

cdef struct DiracWf:
    double complex psi1
    double complex psi2
    double complex psi3
    double complex psi4
    LVector p

cdef scalar_wf(LVector, int)
cdef vector_wf(LVector, double, int, int)
cdef spinor_u(LVector, double, int)
cdef spinor_v(LVector, double, int)
cdef spinor_ubar(LVector, double, int)
cdef spinor_vbar(LVector, double, int)
