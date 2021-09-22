from libc.math cimport sqrt, M_SQRT2, fabs
from wavefunctions cimport LVector, ScalarWf, VectorWf, DiracWf

cdef ScalarWf scalar_wf(LVector p, int final_state):
    cdef ScalarWf wf
    wf.phi = 1.0 + 0.0j
    if final_state == 1:
        wf.p.e = p.e
        wf.p.px = p.px
        wf.p.py = p.py
        wf.p.pz = p.pz
    else:
        wf.p.e = -p.e
        wf.p.px = -p.px
        wf.p.py = -p.py
        wf.p.pz = -p.pz
    return wf



cdef VectorWf vector_wf(LVector p, double mass, int final_state, int spin):
    cdef VectorWf wf
    cdef double e = p.e
    cdef double kx = p.px
    cdef double ky = p.py
    cdef double kz = p.pz

    cdef double kt = sqrt(kx**2 + ky**2)
    cdef double km = sqrt(kx**2 + ky**2 + kz**2)
    cdef int s = -final_state
    cdef double kkz
    cdef double kkt

    cdef double mu

    if spin == 0:
        if mass == 0.0:
           mu = 0.0
        else:
            mu = 1.0 / mass
        wf.eps0 = mu * km
        wf.eps1 = mu * e * kx / km
        wf.eps2 = mu * e * ky / km
        wf.eps3 = mu * e * kz / km
    else:
        kkz = kz / km
        kkt = kt / km

        if kt < DBL_EPSILON:
            wf.eps0 = 0.0j
            wf.eps1 = -spin / M_SQRT2 + 0.0j
            wf.eps2 = -s * 1j / M_SQRT2 + 0.0j
            wf.eps3 = spin * kkt / M_SQRT2 + 0.0j
        else:
            wf.eps0 = 0.0j
            wf.eps1 = (-spin * kx * kkz / kt + 1j * s * ky / kt) / M_SQRT2,
            wf.eps2 = (-spin * ky * kkz / kt - 1j * s * kx / kt) / M_SQRT2,
            wf.eps3 = spin * kkt / M_SQRT2 + 0.0j
    return wf


cdef DiracWf spinor_u(LVector p, double mass, int spin):
    cdef DiracWf wf
    cdef double e = p.e
    cdef double px = p.px
    cdef double py = p.py
    cdef double pz = p.pz

    cdef pm = sqrt(px**2 + py**2 + pz**2)
    cdef wp = sqrt(fabs(px + pm))
    cdef wm = mass / wp
    cdef pmz = abs(pm + p[3])
    cdef den = sqrt(2 * pm * pmz)
    cdef double x1
    cdef double x2

    wf.p.e = p.e
    wf.p.px = p.px
    wf.p.py = p.py
    wf.p.pz = p.pz

    if pmz < DBL_EPSILON:
        x1 = spin + 0.0j
        x2 = 0.0j
    else:
        x1 = (spin * p[1] + p[2] * 1j) / den
        x2 = (pm + p[3] +  0.0j) / den
    if spin == 1:
        wf.psi1 = wm * x2
        wf.psi2 = wm * x1
        wf.psi3 = wp * x2
        wf.psi4 = wp * x1
    else:
        wf.psi1 = wp * x1
        wf.psi2 = wp * x2
        wf.psi3 = wm * x1
        wf.psi4 = wm * x2
    return wf


cdef spinor_v(LVector p, double mass, int spin):
    cdef DiracWf wf
    cdef double e = p.e
    cdef double px = p.px
    cdef double py = p.py
    cdef double pz = p.pz

    cdef pm = sqrt(px**2 + py**2 + pz**2)
    cdef wp = sqrt(fabs(px + pm))
    cdef wm = mass / wp
    cdef pmz = abs(pm + p[3])
    cdef den = sqrt(2 * pm * pmz)
    cdef double x1
    cdef double x2

    wf.p.e = p.e
    wf.p.px = p.px
    wf.p.py = p.py
    wf.p.pz = p.pz

    if pmz < DBL_EPSILON:
        x1 = complex(float(-spin), 0.0)
        x2 = complex(0.0, 0.0)
    else:
        x1 = complex(-spin * p[1], p[2]) / den
        x2 = complex(pm + p[3], 0.0) / den

    if spin == 1:
        wf.psi1 = -wp * x1
        wf.psi2 = -wp * x2
        wf.psi3 = wm * x1
        wf.psi4 = wm * x2
    else:
        wf.psi1 = wm * x2
        wf.psi2 = wm * x1
        wf.psi3 = -wp * x2
        wf.psi4 = -wp * x1
    return wf


cdef DiracWf spinor_ubar(LVector p, double mass, int spin):
    cdef DiracWf wf
    cdef double e = p.e
    cdef double px = p.px
    cdef double py = p.py
    cdef double pz = p.pz

    cdef pm = sqrt(px**2 + py**2 + pz**2)
    cdef wp = sqrt(fabs(px + pm))
    cdef wm = mass / wp
    cdef pmz = abs(pm + p[3])
    cdef den = sqrt(2 * pm * pmz)
    cdef double x1
    cdef double x2

    wf.p.e = p.e
    wf.p.px = p.px
    wf.p.py = p.py
    wf.p.pz = p.pz

    if pmz < DBL_EPSILON:
        x1 = float(spin) + 0.0j
        x2 = 0.0j
    else:
        x1 = complex(spin * p[1], - p[2]) / den
        x2 = complex(pm + p[3], 0.0) / den

    if spin == 1:
        wf.psi1 = wp * x2
        wf.psi2 = wp * x1
        wf.psi3 = wm * x2
        wf.psi4 = wm * x1
    else:
        wf.psi1 = wm * x1
        wf.psi2 = wm * x2
        wf.psi3 = wp * x1
        wf.psi4 = wp * x2
    return wf


cdef DiracWf spinor_vbar(LVector p, double mass, int spin):
    cdef DiracWf wf
    cdef double e = p.e
    cdef double px = p.px
    cdef double py = p.py
    cdef double pz = p.pz

    cdef pm = sqrt(px**2 + py**2 + pz**2)
    cdef wp = sqrt(fabs(px + pm))
    cdef wm = mass / wp
    cdef pmz = abs(pm + p[3])
    cdef den = sqrt(2 * pm * pmz)
    cdef double x1
    cdef double x2

    wf.p.e = p.e
    wf.p.px = p.px
    wf.p.py = p.py
    wf.p.pz = p.pz

    if pmz < DBL_EPSILON:
        x1 = -spin + 0.0j
        x2 = 0.0j
    else:
        x1 = (-spin * p[1]  - p[2] * 1j) / den
        x2 = (pm + p[3] + 0.0j) / den

    if spin == 1:
        wf.psi1 = wm * x1
        wf.psi2 = wm * x2
        wf.psi3 = -wp * x1
        wf.psi4 = -wp * x2
    else:
        wf.psi1 = -wp * x2
        wf.psi2 = -wp * x1
        wf.psi3 = wm * x2
        wf.psi4 = wm * x1
    return wf
