from ..decay_helper_functions.decay_charged_pion cimport CSpectrum as cspec
from ..decay_helper_functions.decay_neutral_pion cimport CSpectrumPoint as nspec
from ..decay_helper_functions.decay_muon cimport CSpectrumPoint as muspec
include "../decay_helper_functions/parameters.pxd"

import cython
import numpy as np
cimport numpy as np
from scipy.integrate import quad

from libc.math cimport exp, log, M_PI, log10, sqrt, abs, pow

cdef double mmu = MASS_MU
cdef double me = MASS_E
cdef double mpi = MASS_PI
cdef double qe = sqrt(4. * M_PI * ALPHA_EM)
ctypedef np.ndarray ndarray


cdef int num_interp = 100
cdef np.ndarray eng_gams_interp = np.zeros(num_interp, dtype=np.float64)
cdef np.ndarray cp_spec_interp = np.zeros(num_interp, dtype=np.float64)

def set_cp_spec(double ms):
    global eng_gams_interp
    global cp_spec_interp
    eng_gams_interp = np.logspace(-3., np.log10(ms / 2.), num=num_interp)
    cp_spec_interp = cspec(eng_gams_interp, ms / 2., "total")

cdef double __cp_spec(double eng_gam):
    if eng_gam < eng_gams_interp[0]:
        return cp_spec_interp[0] * eng_gams_interp[0] / eng_gam
    return np.interp(eng_gam, eng_gams_interp, cp_spec_interp)


@cython.cdivision(True)
cdef double __dnde_fsr_cp_srf(double egam, double ms):

    cdef double mupi = mpi / ms
    cdef double x = 2. * egam / ms
    cdef double xmin = 0.0
    cdef double xmax = 1 - 4. * mupi**2
    cdef double dynamic
    cdef double coeff
    cdef double result

    if x < xmin or x > xmax:
        return 0.0

    dynamic = (
        -2 * sqrt(1 - x) * sqrt(1 - 4 * mupi**2 - x) +
        (-1 + 2 * mupi**2 + x) * log(
            (1 - x - sqrt(1 - x) * sqrt(1 - 4 * mupi**2 - x))**2 /
            (-1 + x - sqrt(1 - x) * sqrt(1 - 4 * mupi**2 - x))**2)) / x

    coeff = qe**2 / (8. * sqrt(1 - 4 * mupi**2) * M_PI**2)

    result = dynamic * coeff

    return 2 * result / ms


@cython.cdivision(True)
cdef double __dnde_fsr_l_srf(double egam, double ml, double ms):
    cdef double mul = ml / ms
    cdef double x = 2. * egam / ms

    cdef double xmin = 0.0
    cdef double xmax = 1 - 4. * mul**2

    cdef double dynamic
    cdef double coeff
    cdef double result

    if x < xmin or x > xmax:
        return 0.0

    dynamic = (
        4 * (-1 + 4 * mul**2) *
        sqrt(1 - x) * sqrt(1 - 4 * mul**2 - x) +
        (2 - 12 * mul**2 + 16 * mul**4 - 2 * x + 8 * mul**2 * x + x**2) * log(
            (1 - x + sqrt(
                (-1 + x) * (-1 + 4 * mul**2 + x)))**2 / (-1 + x + sqrt(
                    (-1 + x) * (-1 + 4 * mul**2 + x)))**2)) / x

    coeff = qe**2 / (16. * (1 - 4 * mul**2)**1.5 * M_PI**2)

    result = dynamic * coeff

    return 2 * result / ms


@cython.cdivision(True)
cdef double __integrand(double cl, double eng_gam, double eng_s,
                        double ms, np.ndarray[double] pws, str mode):
    """
    Integrand of the boost integralself.

    Parameters
    ----------
    cl : float
        Angle the final state particle make with respect to the z-axis.
    eng_gams : float
        Gamma-ray energy to evaluate spectrum at.
    eng_s : float
        Energy of the scalar mediator.
    params :
        Scalar mediator model parameters.

    Returns
    -------
    integrand : float
        The value of the boost integral.
    """
    cdef double pwee = pws[0]
    cdef double pwmumu = pws[1]
    cdef double pwpi0pi0 = pws[2]
    cdef double pwpipi = pws[3]

    cdef double beta = sqrt(1. - (ms / eng_s)**2)
    cdef double gamma = eng_s / ms

    cdef double jac = 1. / (2. * gamma * abs(1. - beta * cl))

    cdef double eng_gam_srf = eng_gam * gamma * (1. - beta * cl)

    cdef double dnde = 0.0

    if mode == "total":
        dnde = pwee * __dnde_fsr_l_srf(eng_gam_srf, me, ms) + \
            2. * pwpipi * __cp_spec(eng_gam_srf) + \
            pwpipi * __dnde_fsr_cp_srf(eng_gam_srf, ms) + \
            2. * pwpi0pi0 * nspec(eng_gam_srf, ms / 2.) +\
            2. * pwmumu * muspec(eng_gam_srf, ms / 2.) + \
            pwmumu * __dnde_fsr_l_srf(eng_gam_srf, mmu, ms)

    if mode == "e e g":
        dnde = pwee * __dnde_fsr_l_srf(eng_gam_srf, me, ms)
    if mode == "pi pi g":
        dnde = pwpipi * __dnde_fsr_cp_srf(eng_gam_srf, ms)
    if mode == "pi pi":
        dnde = 2. * pwpipi * __cp_spec(eng_gam_srf)
    if mode == "pi0 pi0":
        dnde = 2. * pwpi0pi0 * nspec(eng_gam_srf, ms / 2.)
    if mode == "mu mu g":
        dnde = pwmumu * __dnde_fsr_l_srf(eng_gam_srf, mmu, ms)
    if mode == "mu mu":
        dnde = 2. * pwmumu * muspec(eng_gam_srf, ms / 2.)
    if mode == "g g":
        dnde = 0.0

    return jac * dnde


@cython.cdivision(True)
cdef double __dnde_decay_s(double eng_gam, double eng_s, double ms,
                           np.ndarray[double] pws, str mode):
    """
    Unvectorized dnde_decay_s

    Compute the gamma ray spectrum from the decay of the scalar mediator.

    Parameters
    ----------
    eng_gams : float or array-like
        Float or array of gamma-ray energies to evaluate spectrum at.
    eng_s : float
        Energy of the scalar mediator.
    params :
        Scalar mediator model parameters.

    Returns
    -------
    dnde : float or array-like
        Values of dnde at gamma-ray energies `eng_gams`.
    """
    cdef double lines_contrib = 0.0
    cdef double beta = np.sqrt(1. - pow(ms / eng_s, 2.))
    cdef double gamma = eng_s / ms

    if eng_s < ms:
        return 0.

    cdef double eplus = eng_s * (1. + beta) / 2.0
    cdef double eminus = eng_s * (1. - beta) / 2.0
    cdef double result = 0.0

    if eminus <= eng_gam and eng_gam <= eplus:
        lines_contrib = pws[4] * 2. / (eng_s * beta)

    result = quad(__integrand, -1.0, 1.0, points=[-1.0, 1.0],
                  args=(eng_gam, eng_s, ms, pws, mode), epsabs=10**-10.,
                  epsrel=10**-5.)[0]

    if mode == "g g":
        return lines_contrib

    if mode == "total":
        return result + lines_contrib

    return result


@cython.boundscheck(True)
@cython.wraparound(False)
def cdnde_decay_s_pt(double eng_gam, double eng_s, double ms,
                 np.ndarray[double] pws, str mode):
    """
    Compute the gamma ray spectrum from the decay of the scalar mediator.

    Parameters
    ----------
    eng_gams : float
        Gamma-ray energy to evaluate spectrum at.
    eng_s : float
        Energy of the scalar mediator.
    params :
        Scalar mediator model parameters.

    Returns
    -------
    dnde : float or array-like
        Value of dnde at gamma-ray energy `eng_gam`.
    """
    return __dnde_decay_s(eng_gam, eng_s, ms, pws, mode)


@cython.boundscheck(True)
@cython.wraparound(False)
def dnde_decay_s_pt(double eng_gam, double eng_s, double ms,
                 np.ndarray[double] pws, str mode):
    """
    Compute the gamma ray spectrum from the decay of the scalar mediator.

    Parameters
    ----------
    eng_gams : float
        Gamma-ray energy to evaluate spectrum at.
    eng_s : float
        Energy of the scalar mediator.
    params :
        Scalar mediator model parameters.

    Returns
    -------
    dnde : float or array-like
        Value of dnde at gamma-ray energy `eng_gam`.
    """
    set_cp_spec(ms)
    return cdnde_decay_s_pt(eng_gam, eng_s, ms, pws, mode)


@cython.boundscheck(True)
@cython.wraparound(False)
def dnde_decay_s(np.ndarray[double] eng_gam, double eng_s, double ms,
                   np.ndarray[double] pws, str mode):
    """
    Compute the gamma ray spectrum from the decay of the scalar mediator.

    Parameters
    ----------
    eng_gams : float
        Gamma-ray energy to evaluate spectrum at.
    eng_s : float
        Energy of the scalar mediator.
    params :
        Scalar mediator model parameters.

    Returns
    -------
    dnde : float or array-like
        Value of dnde at gamma-ray energy `eng_gam`.
    """
    set_cp_spec(ms)
    cdef int num_pts = len(eng_gam)
    cdef int i

    spec = np.zeros(num_pts, dtype=np.float64)

    for i in range(num_pts):
        spec[i] = cdnde_decay_s_pt(eng_gam[i], eng_s, ms, pws, mode)

    return spec
