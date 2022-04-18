import cython

import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from scipy.integrate import quad

include "../../_utils/constants.pxd"

cdef double mmu = MASS_MU
cdef double me = MASS_E

# ===================================================================
# ---- Cython API ---------------------------------------------------
# ===================================================================

@cython.boundscheck(True)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double __spectrum_rf(double eng_p):
    cdef double r = me / mmu
    cdef double s = me * me - 2. * eng_p * mmu + mmu * mmu
    cdef double smax = (mmu - me) * (mmu - me)
    cdef double smin = 0.0
    if s <= smin or smax <= s:
        return 0.0

    return 2 * mmu * (2 * (pow(mmu, 4) * pow(-1 + r * r, 2) + mmu * mmu *
                           (1 + r * r) * s - 2 * s * s) *
                      np.sqrt(pow(mmu, 4) * pow(-1 + r * r, 2) -
                              2 * mmu**2 * (1 + r * r) * s + s * s)) / pow(mmu, 8)


@cython.boundscheck(True)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double __integrand(double cl, double eng_p, double eng_mu):
    if eng_p < me:
        return 0.0
    cdef double p = sqrt(eng_p * eng_p - me * me)
    cdef double gamma = eng_mu / mmu
    cdef double beta = sqrt(1.0 - pow(mmu / eng_mu, 2))
    cdef double emurf = gamma * (eng_p - p * beta * cl)
    cdef double jac = p / (2. * sqrt((1 + pow(beta * cl, 2)) * eng_p * eng_p -
                                     (1 + beta * beta * (-1 + cl * cl)) *
                                     me * me -
                                     2 * beta * cl * eng_p * p) * gamma)
    return __spectrum_rf(emurf) * jac


@cython.boundscheck(True)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dnde_positron_muon_point(double eng_p, double eng_mu):
    if eng_mu < mmu:
        return 0.0
    return quad(__integrand, -1., 1., points=[-1.0, 1.0],
                args=(eng_p, eng_mu), epsabs=1e-10, epsrel=1e-4)[0]


@cython.boundscheck(True)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[np.float64_t,ndim=1] dnde_positron_muon_array(double[:] engs_p, double eng_mu):
    cdef int npts = engs_p.shape[0]
    cdef np.ndarray[np.float64_t,ndim=1] spec = np.zeros_like(engs_p)
    for i in range(npts):
        spec[i] = dnde_positron_muon_point(engs_p[i], eng_mu)
    return spec


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