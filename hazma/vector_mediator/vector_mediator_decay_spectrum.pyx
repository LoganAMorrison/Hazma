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
cdef double mpi0 = MASS_PI0
cdef double qe = sqrt(4. * M_PI * ALPHA_EM)
ctypedef np.ndarray ndarray

cdef int n_interp_pts = 500

cdef np.ndarray __e_gams = np.zeros((n_interp_pts,), dtype=np.float64)

cdef np.ndarray __spec_cp = np.zeros((n_interp_pts,), dtype=np.float64)
cdef np.ndarray __spec_mu = np.zeros((n_interp_pts,), dtype=np.float64)

cdef double __set_spectra(double mv):
    global __e_gams
    global __spec_cp
    global __spec_mu

    __e_gams = np.logspace(-1., np.log10(mv / 2.), num=n_interp_pts)

    __spec_cp = cp_spec(__e_gams, mv / 2., "total")
    __spec_mu = mu_spec(__e_gams, mv / 2.)

cdef double __interp_spec(double eng_gam, str mode):
    """
    Intepolation function for the short kaon data.

    Parameters
    ----------
    eng_gam : double
        Energy of the photon.
    mode : str {"total"}
        String specifying which decay mode to use.
    """
    if mode == "cp":
        if eng_gam < 10**-1:
            return __spec_cp[0] * __e_gams[0] / eng_gam
        return np.interp(eng_gam, __e_gams, __spec_cp)
    if mode == "mu":
        if eng_gam < 10**-1:
            return __spec_mu[0] * __e_gams[0] / eng_gam
        return np.interp(eng_gam, __e_gams, __spec_mu)
    else:
        return 0.0

@cython.cdivision(True)
cdef double __dnde_fsr_cp_vrf(double egam, double mv):
    cdef double mupi = mpi / mv
    cdef double x = 2. * egam / mv
    cdef double xmin = 0.0
    cdef double xmax = 1 - 4. * mupi**2
    cdef double dynamic
    cdef double coeff
    cdef double result

    if x < xmin or x > xmax:
        return 0.0

    coeff = qe**2 / (4. * (1 - 4 * mupi**2)**1.5 * M_PI**2)

    dynamic = ((2 * sqrt(1 - 4 * mupi**2 - x) *
                (-1 - 4 * mupi**2 * (-1 + x) + x + x**2)) / sqrt(1 - x) +
               (-1 + 4 * mupi**2) * (-1 + 2 * mupi**2 + x) *
               log((1 + sqrt(1 - x) * sqrt(1 - 4 * mupi**2 - x) - x)**2 /
                   (-1 + sqrt(1 - x) * sqrt(1 - 4 * mupi**2 - x) + x)**2)) / x

    result = dynamic * coeff

    return 2 * result / mv

@cython.cdivision(True)
cdef double __dnde_fsr_l_vrf(double egam, double ml, double mv):
    cdef double mul = ml / mv
    cdef double x = 2. * egam / mv

    cdef double xmin = 0.0
    cdef double xmax = 1 - 4. * mul**2

    cdef double dynamic
    cdef double coeff
    cdef double result

    if x < xmin or x > xmax:
        return 0.0

    coeff = -qe**2 / (8. * sqrt(1 - 4 * mul**2) * (1 + 2 * mul**2) * M_PI**2)

    dynamic = ((2 * sqrt(1 - 4 * mul**2 - x) *
                (2 - 4 * mul**2 * (-1 + x) - 2 * x + x**2)) / sqrt(1 - x) +
               (2 - 8 * mul**4 - 4 * mul**2 * x + (-2 + x) * x) * log(
                (-1 + sqrt(1 - x) * sqrt(1 - 4 * mul**2 - x) + x)**2 / (
                        1 + sqrt(1 - x) * sqrt(1 - 4 * mul**2 - x) - x)**2)) / x

    result = dynamic * coeff

    return 2 * result / mv

@cython.cdivision(True)
@cython.boundscheck(True)
@cython.wraparound(False)
cdef double __integrand(double cl, double eng_gam, double eng_v,
                        double mv, np.ndarray[double] pws, str mode):
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
    cdef double pwpi0g = pws[2]
    cdef double pwpipi = pws[3]

    cdef double beta = sqrt(1. - (mv / eng_v)**2)
    cdef double gamma = eng_v / mv

    cdef double jac = 1. / (2. * gamma * abs(1. - beta * cl))

    cdef double eng_gam_vrf = eng_gam * gamma * (1. - beta * cl)

    cdef double dnde = 0.0

    cdef double dnde_ee_f = pwee * __dnde_fsr_l_vrf(eng_gam_vrf, me, mv)
    cdef double dnde_mu_f = pwmumu * __dnde_fsr_l_vrf(eng_gam_vrf, mmu, mv)

    cdef double dnde_cp_f = pwpipi * __dnde_fsr_cp_vrf(eng_gam_vrf, mv)

    cdef double dnde_cp_d = 2. * pwpipi * __interp_spec(eng_gam_vrf, "cp")

    # Neutral pion energy is:
    cdef double e_pi0 = 0.5 * (mpi0**2 + mv**2) / mv
    cdef double dnde_np_d = pwpi0g * CSpectrumPoint(eng_gam_vrf, e_pi0)

    cdef double dnde_mu_d = 2. * pwmumu * __interp_spec(eng_gam_vrf, "mu")

    dnde = dnde_ee_f + dnde_mu_f + dnde_cp_f + \
           dnde_cp_d + dnde_np_d + dnde_mu_d

    if mode == "total":
        return jac * dnde
    if mode == "e e g":
        return jac * dnde_ee_f
    if mode == "pi pi g":
        return jac * dnde_cp_f
    if mode == "pi pi":
        return jac * dnde_cp_d
    if mode == "pi0 g":
        return jac * dnde_np_d
    if mode == "mu mu g":
        return jac * dnde_mu_f
    if mode == "mu mu":
        return jac * dnde_mu_d

@cython.boundscheck(True)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double __dnde_decay_v(double eng_gam, double eng_v, double mv,
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
    cdef double beta = sqrt(1. - pow(mv / eng_v, 2.))
    cdef double gamma = eng_v / mv

    if eng_v < mv:
        return 0.

    cdef double eplus = eng_v * (1. + beta) / 2.0
    cdef double eminus = eng_v * (1. - beta) / 2.0
    cdef double result = 0.0

    if eminus <= eng_gam <= eplus:
        lines_contrib = pws[2] / (eng_v * beta)

    result = quad(__integrand, -1.0, 1.0, points=[-1.0, 1.0],
                  args=(eng_gam, eng_v, mv, pws, mode), epsabs=10**-10.,
                  epsrel=10**-5.)[0]

    if mode == "pi0 g" or mode == "total":
        return result + lines_contrib

    else:
        return result

@cython.boundscheck(True)
@cython.wraparound(False)
def dnde_decay_v_pt(double eng_gam, double eng_v, double mv,
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
    __set_spectra(mv)
    return __dnde_decay_v(eng_gam, eng_v, mv, pws, mode)

@cython.boundscheck(True)
@cython.wraparound(False)
def dnde_decay_v(np.ndarray[double] eng_gam, double eng_v, double mv,
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
    __set_spectra(mv)
    cdef int num_pts = len(eng_gam)
    cdef int i

    spec = np.zeros(num_pts, dtype=np.float64)

    for i in range(num_pts):
        spec[i] = __dnde_decay_v(eng_gam[i], eng_v, mv, pws, mode)

    return spec
