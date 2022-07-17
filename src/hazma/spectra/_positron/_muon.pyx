import cython

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fmin, fmax
from libc.float cimport DBL_EPSILON
from scipy.integrate import quad
from hazma._utils.boost cimport boost_beta, boost_gamma

include "../../_utils/constants.pxd"

DEF R = MASS_E / MASS_MU
DEF R2 = R * R
# 1 / (1 - 8 r^2 + 8 r^6 - r^8 - 12 r^2 ln(r^2))
DEF R_FACTOR = 1.0001870858234163 

# ===================================================================
# ---- Cython API ---------------------------------------------------
# ===================================================================


@cython.boundscheck(True)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dndx_positron_muon_rest_frame(double x):
    if x <= 2 * R or x >= 1.0 + R2:
        return 0.0
    return -2.0 * sqrt(x**2 - 4.0 * R2) * (4.0 * R2 + x * (-3.0 - 3.0 * R2 + 2.0 * x)) / R_FACTOR


@cython.boundscheck(True)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dndx_positron_muon(double x, double beta):
    cdef double gamma2
    cdef double r22
    cdef double xm
    cdef double xp

    if beta < 0.0 or beta > 1.0:
        return 0.0

    if beta < DBL_EPSILON:
        return dndx_positron_muon_rest_frame(x)

    gamma2 = 1.0 / (1.0 - beta**2)
    r22 = 4.0 * R2 * (1.0 - beta**2)

    xm = fmax(gamma2 * (x - beta * sqrt(x**2 - r22)), 2.0 * R)
    xp = fmin(gamma2 * (x + beta * sqrt(x**2 - r22)), 1.0 + R2)

    if xm > xp or xp < xm:
        return 0.0

    return (xm * (8 * R2 + xm * (-3 - 3 * R2 + (4 * xm) / 3.0)) + xp * (
        -8 * R2 + (3 + 3 * R2 - (4 * xp) / 3.0) * xp
    )) / (2 * beta * R_FACTOR)


@cython.boundscheck(True)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dnde_positron_muon_point(double e, double emu):
    cdef double beta
    cdef double pre
    cdef double dndx

    if emu < MASS_MU or e <= MASS_E:
        return 0.0

    if emu - MASS_MU < DBL_EPSILON:
        pre = 2.0 / MASS_MU
        dndx = dndx_positron_muon_rest_frame(pre * e)
    else:
        beta = boost_beta(emu, MASS_MU)
        pre = 2.0 / emu
        dndx = dndx_positron_muon(pre * e, beta)

    return pre * dndx


@cython.boundscheck(True)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[np.float64_t,ndim=1] dnde_positron_muon_array(double[:] engs_p, double eng_mu):
    cdef int npts = engs_p.shape[0]
    cdef np.ndarray[np.float64_t,ndim=1] spec = np.zeros_like(engs_p)
    for i in range(npts):
        spec[i] = dnde_positron_muon_point(engs_p[i], eng_mu)
    return spec



# @cython.boundscheck(True)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef double __spectrum_rf(double eng_p):
#     cdef double r = me / mmu
#     cdef double s = me * me - 2. * eng_p * mmu + mmu * mmu
#     cdef double smax = (mmu - me) * (mmu - me)
#     cdef double smin = 0.0
#     if s <= smin or smax <= s:
#         return 0.0

#     return 2 * mmu * (2 * (pow(mmu, 4) * pow(-1 + r * r, 2) + mmu * mmu *
#                            (1 + r * r) * s - 2 * s * s) *
#                       np.sqrt(pow(mmu, 4) * pow(-1 + r * r, 2) -
#                               2 * mmu**2 * (1 + r * r) * s + s * s)) / pow(mmu, 8)


# @cython.boundscheck(True)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef double __integrand(double cl, double eng_p, double eng_mu):
#     if eng_p < me:
#         return 0.0
#     cdef double p = sqrt(eng_p * eng_p - me * me)
#     cdef double gamma = eng_mu / mmu
#     cdef double beta = sqrt(1.0 - pow(mmu / eng_mu, 2))
#     cdef double emurf = gamma * (eng_p - p * beta * cl)
#     cdef double jac = p / (2. * sqrt((1 + pow(beta * cl, 2)) * eng_p * eng_p -
#                                      (1 + beta * beta * (-1 + cl * cl)) *
#                                      me * me -
#                                      2 * beta * cl * eng_p * p) * gamma)
#     return __spectrum_rf(emurf) * jac


# @cython.boundscheck(True)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef double dnde_positron_muon_point(double eng_p, double eng_mu):
#     if eng_mu < mmu:
#         return 0.0
#     return quad(__integrand, -1., 1., points=[-1.0, 1.0],
#                 args=(eng_p, eng_mu), epsabs=1e-10, epsrel=1e-4)[0]


# @cython.boundscheck(True)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef np.ndarray[np.float64_t,ndim=1] dnde_positron_muon_array(double[:] engs_p, double eng_mu):
#     cdef int npts = engs_p.shape[0]
#     cdef np.ndarray[np.float64_t,ndim=1] spec = np.zeros_like(engs_p)
#     for i in range(npts):
#         spec[i] = dnde_positron_muon_point(engs_p[i], eng_mu)
#     return spec


# ===================================================================
# ---- Python API ---------------------------------------------------
# ===================================================================

def dnde_positron_muon(epos, emu):
    """
    Compute the positron spectrum dN/dE from the decay of a muon.
    Paramaters
    ----------
    epos: float or array-like
        Positron energy.
    emu: float 
        Energy of the muon.
    """
    if hasattr(epos, '__len__'):
        energies = np.array(epos)
        assert len(energies.shape) == 1, "Positron energies must be 0 or 1-dimensional."
        return dnde_positron_muon_array(energies, emu)
    else:
        return dnde_positron_muon_point(epos, emu)
