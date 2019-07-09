from hazma.decay_helper_functions.decay_charged_pion \
    cimport CSpectrum as cp_spec
from hazma.decay_helper_functions.decay_neutral_pion \
    cimport CSpectrum as np_spec
from hazma.decay_helper_functions.decay_neutral_pion cimport CSpectrumPoint
from hazma.decay_helper_functions.decay_muon cimport CSpectrum as mu_spec

import cython
import numpy as np
cimport numpy as np
from scipy.integrate import quad

from libc.math cimport exp, log, M_PI, log10, sqrt, abs, pow

include "../decay_helper_functions/parameters.pxd"

cdef double mmu = MASS_MU
cdef double me = MASS_E
cdef double mpi = MASS_PI
cdef double qe = sqrt(4. * M_PI * ALPHA_EM)
ctypedef np.ndarray ndarray

cdef int n_interp_pts = 500

cdef np.ndarray __e_gams = np.zeros((n_interp_pts,), dtype=np.float64)

cdef np.ndarray __spec_cp = np.zeros((n_interp_pts,), dtype=np.float64)
cdef np.ndarray __spec_np = np.zeros((n_interp_pts,), dtype=np.float64)
cdef np.ndarray __spec_mu = np.zeros((n_interp_pts,), dtype=np.float64)

cdef double __set_spectra(double ms):
    global __e_gams
    global __spec_cp
    global __spec_np
    global __spec_mu

    __e_gams = np.logspace(-1., np.log10(ms / 2.), num=n_interp_pts)

    __spec_cp = cp_spec(__e_gams, ms / 2., "total")
    __spec_np = np_spec(__e_gams, ms / 2.)
    __spec_mu = mu_spec(__e_gams, ms / 2.)

cdef double __interp_spec(double eng_gam, str fs):
    """
    Intepolation function for the short kaon data.

    Parameters
    ----------
    eng_gam : double
        Energy of the photon.
    fs : str {"total"}
        String specifying which decay fs to use.
    """
    if fs == "cp":
        if eng_gam < 10**-1:
            return __spec_cp[0] * __e_gams[0] / eng_gam
        return np.interp(eng_gam, __e_gams, __spec_cp)
    if fs == "mu":
        if eng_gam < 10**-1:
            return __spec_mu[0] * __e_gams[0] / eng_gam
        return np.interp(eng_gam, __e_gams, __spec_mu)
    else:
        return 0.0

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
@cython.boundscheck(True)
@cython.wraparound(False)
cdef double __integrand(double cl, double eng_gam, double eng_s,
                        double ms, np.ndarray[double] pws, str fs):
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

    cdef double dnde_ee_f = pwee * __dnde_fsr_l_srf(eng_gam_srf, me, ms)
    cdef double dnde_mu_f = pwmumu * __dnde_fsr_l_srf(eng_gam_srf, mmu, ms)

    cdef double dnde_cp_f = pwpipi * __dnde_fsr_cp_srf(eng_gam_srf, ms)

    cdef double dnde_cp_d = 2. * pwpipi * __interp_spec(eng_gam_srf, "cp")
    cdef double dnde_np_d = 2. * pwpi0pi0 * CSpectrumPoint(eng_gam_srf, ms / 2.)
    cdef double dnde_mu_d = 2. * pwmumu * __interp_spec(eng_gam_srf, "mu")

    dnde = dnde_ee_f + dnde_mu_f + dnde_cp_f + \
           dnde_cp_d + dnde_np_d + dnde_mu_d

    if fs == "total":
        return jac * dnde
    if fs == "e e g":
        return jac * dnde_ee_f
    if fs == "pi pi g":
        return jac * dnde_cp_f
    if fs == "pi pi":
        return jac * dnde_cp_d
    if fs == "pi0 pi0":
        return jac * dnde_np_d
    if fs == "mu mu g":
        return jac * dnde_mu_f
    if fs == "mu mu":
        return jac * dnde_mu_d
    if fs == "g g":
        return 0.0

@cython.boundscheck(True)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double __dnde_decay_s(double eng_gam, double eng_s, double ms,
                           np.ndarray[double] pws, str fs):
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
    cdef double beta = sqrt(1. - pow(ms / eng_s, 2.))
    cdef double gamma = eng_s / ms

    if eng_s < ms:
        return 0.

    cdef double eplus = eng_s * (1. + beta) / 2.0
    cdef double eminus = eng_s * (1. - beta) / 2.0
    cdef double result = 0.0

    if eminus <= eng_gam <= eplus:
        lines_contrib = pws[4] * 1. / (eng_s * beta)

    if fs == "g g":
        return lines_contrib

    if fs == "total":
        result = quad(__integrand, -1.0, 1.0, points=[-1.0, 1.0],
                      args=(eng_gam, eng_s, ms, pws, fs), epsabs=10**-10.,
                      epsrel=10**-5.)[0]

        return result + lines_contrib

    return result

@cython.boundscheck(True)
@cython.wraparound(False)
def dnde_decay_s_pt(double eng_gam, double eng_s, double ms,
                    np.ndarray[double] pws, str fs):
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
    __set_spectra(ms)
    return __dnde_decay_s(eng_gam, eng_s, ms, pws, fs)

@cython.boundscheck(True)
@cython.wraparound(False)
def dnde_decay_s(np.ndarray[double] eng_gam, double eng_s, double ms,
                 np.ndarray[double] pws, str fs):
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
    __set_spectra(ms)
    cdef int num_pts = len(eng_gam)
    cdef int i

    spec = np.zeros(num_pts, dtype=np.float64)

    for i in range(num_pts):
        spec[i] = __dnde_decay_s(eng_gam[i], eng_s, ms, pws, fs)

    return spec
