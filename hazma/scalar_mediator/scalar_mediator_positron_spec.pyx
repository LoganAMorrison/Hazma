from hazma.positron_helper_functions.positron_muon \
    cimport CSpectrum as mu_spec
from hazma.positron_helper_functions.positron_charged_pion \
    cimport CSpectrum as cp_spec

import cython
import numpy as np
cimport numpy as np
from scipy.integrate import quad

from libc.math cimport M_PI, sqrt, pow, log10

include "../decay_helper_functions/parameters.pxd"

cdef double mmu = MASS_MU
cdef double me = MASS_E
cdef double mpi = MASS_PI
cdef double qe = sqrt(4. * M_PI * ALPHA_EM)
ctypedef np.ndarray ndarray

# Cached values of the mediator mass and partial widths. If the mass of the
# mediator or partial widths change, the spectra need to be recomputed.
cdef double cache_ms = -1.0;
cdef np.ndarray cache_pws = np.array([-1.0, -1.0, -1.0])

# Set up arrays for the interpolating positron spectra for the charged pion
# and the muon.
cdef int n_interp_pts = 500
cdef np.ndarray __e_ps = np.zeros((n_interp_pts,), dtype=np.float64)
cdef np.ndarray __spec_cp = np.zeros((n_interp_pts,), dtype=np.float64)
cdef np.ndarray __spec_mu = np.zeros((n_interp_pts,), dtype=np.float64)

cdef int __recompute_rf_spectra(double ms, np.ndarray[double] pws):
    """
    Determine if we need to recompute the positron spectra for the muon and
    pion spectra in the scalar rest frame.

    Parameters
    ----------
    ms: double
        Mass of the scalar mediator
    pws: np.ndarray
        Relevant partial widths: pws[0] = pw_ee, pws[1] = pw_mumu and
        pws[2] = pw_pipi

    Returns
    -------
    should_recompute:
        True if we need to recompute the spectra.
    """
    global cache_ms
    global cache_pws

    if ms != cache_ms or pws[0] != cache_pws[0] or pws[1] != cache_pws[1] or \
            pws[2] != cache_pws[2]:
        return 1
    return 0

cdef void __set_spectra(double ms):
    """
    Set interpolating functions for charged pion and muon positron spectra to
    speed up functions calls during integration.

    Parameters
    ----------
    ms: double
        Mass of the scalar mediator

    """
    global __e_ps
    global __spec_cp
    global __spec_mu

    __e_ps = np.logspace(log10(me), log10(ms / 2.), num=n_interp_pts)
    __spec_cp = cp_spec(__e_ps, ms / 2.)
    __spec_mu = mu_spec(__e_ps, ms / 2.)

cdef double __interp_spec(double eng_p, str fs):
    """
    Return the positron spectrum from the decay of the scalar mediator into
    either a charged pion or muon for a given electron/positron energy.

    Parameters
    ----------
    eng_p: double
        Energy of the positron/electron
    fs: str
        String specifying the final state: 'cp' for charged pion and 'mu' for
        the muon.

    Returns
    -------
    spectrum: double
        The positron spectrum value from the decay of the scalar mediator into
        pions or muons.

    """
    if fs == "cp":
        return np.interp(eng_p, __e_ps, __spec_cp)
    if fs == "mu":
        return np.interp(eng_p, __e_ps, __spec_mu)
    else:
        return 0.0

@cython.cdivision(True)
@cython.boundscheck(True)
@cython.wraparound(False)
cdef double __integrand(double cl, double eng_p, double eng_s,
                        double ms, np.ndarray[double] pws, str fs):
    """
    Integrand of the boost integral.

    Parameters
    ----------
    cl : float
        Angle the final state particle make with respect to the z-axis.
    eng_p : float
        Gamma-ray energy to evaluate spectrum at.
    eng_s : float
        Energy of the scalar mediator.
    ms : double
        Scalar mediator mass.
    pws: np.ndarray
        Relevant partial widths: pws[0] = pw_ee, pws[1] = pw_mumu and
        pws[2] = pw_pipi
    fs: str
        String specifying the final state to compute spectrum for.

    Returns
    -------
    integrand : float
        The value of the boost integral.
    """
    if eng_p < me:
        return 0.0

    cdef double pwmumu = pws[1]
    cdef double pwpipi = pws[2]

    cdef double p = sqrt(eng_p * eng_p - me * me)
    cdef double gamma = eng_s / ms
    cdef double beta = sqrt(1. - pow(ms / eng_s, 2))
    cdef double eng_p_srf = gamma * (eng_p - p * beta * cl)
    cdef double jac = p / (2. * sqrt((1 + pow(beta * cl, 2)) * eng_p * eng_p -
                                     (1 + beta * beta * (-1 + cl * cl)) *
                                     me * me -
                                     2 * beta * cl * eng_p * p) * gamma)

    cdef double dnde_cp = 0.0
    cdef double dnde_mu = 0.0

    if fs == "total":
        dnde_cp = pwpipi * __interp_spec(eng_p_srf, "cp")
        dnde_mu = pwmumu * __interp_spec(eng_p_srf, "mu")
        return jac * (dnde_cp + dnde_mu)
    if fs == "pi pi":
        dnde_cp = pwpipi * __interp_spec(eng_p_srf, "cp")
        return jac * (dnde_cp + dnde_mu)
    if fs == "mu mu":
        dnde_mu = pwmumu * __interp_spec(eng_p_srf, "mu")
        return jac * (dnde_cp + dnde_mu)
    if fs == "e e":
        return 0.0

@cython.boundscheck(True)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double __dnde_decay_s(double eng_p, double eng_s, double ms,
                           np.ndarray[double] pws, str fs):
    """
    Un-vectorized dnde_decay_s

    Compute the gamma ray spectrum from the decay of the scalar mediator.

    Parameters
    ----------
    eng_p : double
        Positron/electron energy to evaluate spectrum at.
    eng_s : float
        Energy of the scalar mediator.
    ms : double
        Mass of the scalar mediator.
    pws: np.ndarray[double]
        Array of the relevant partial widths: pws[0] = pw_ee,
        pws[1] = pw_mumu and pws[2] = pw_pipi

    Returns
    -------
    dnde : float or array-like
        Values of dnde at positron/electron energy `eng_p`.
    """
    cdef double lines_contrib = 0.0
    cdef double beta = sqrt(1. - pow(ms / eng_s, 2.))
    cdef double gamma = eng_s / ms

    if eng_s < ms:
        return 0.

    cdef double r = sqrt(1.0 - 4.0 * me * me / (ms * ms))
    cdef double eplus = eng_s * (1. + r * beta) / 2.0
    cdef double eminus = eng_s * (1. - r * beta) / 2.0
    cdef double result = 0.0

    if eminus <= eng_p <= eplus:
        lines_contrib = pws[0] * 1. / (eng_s * beta)

    if fs == "e e":
        return lines_contrib

    if fs == "total" or fs == "pi pi" or fs == "mu mu":
        result = quad(__integrand, -1.0, 1.0, points=[-1.0, 1.0],
                      args=(eng_p, eng_s, ms, pws, fs), epsabs=1e-10,
                      epsrel=1e-5)[0]

        return result + lines_contrib

    return result

@cython.boundscheck(True)
@cython.wraparound(False)
def dnde_decay_s_pt(double eng_p, double eng_s, double ms,
                    np.ndarray[double] pws, str fs):
    """
    Compute the gamma ray spectrum from the decay of the scalar mediator.

    Parameters
    ----------
    eng_p : float
        Positron energy to evaluate spectrum at.
    eng_s : float
        Energy of the scalar mediator.
    ms : double
        Mass of the scalar mediator.
    pws: np.ndarray[double]
        Array of the relevant partial widths: pws[0] = pw_ee,
        pws[1] = pw_mumu and pws[2] = pw_pipi
    fs: str
        String specifying the final state to compute spectrum for.

    Returns
    -------
    dnde : float or array-like
        Value of dnde at positron energy `eng_p`.
    """
    if __recompute_rf_spectra(ms, pws) == 1:
        __set_spectra(ms)
    return __dnde_decay_s(eng_p, eng_s, ms, pws, fs)

@cython.boundscheck(True)
@cython.wraparound(False)
def dnde_decay_s(np.ndarray[double] eng_ps, double eng_s, double ms,
                 np.ndarray[double] pws, str fs):
    """
    Compute the gamma ray spectrum from the decay of the scalar mediator.

    Parameters
    ----------
    eng_ps : float
        Positron energy to evaluate spectrum at.
    eng_s : float
        Energy of the scalar mediator.
    ms : double
        Mass of the scalar mediator.
    pws: np.ndarray[double]
        Array of the relevant partial widths: pws[0] = pw_ee,
        pws[1] = pw_mumu and pws[2] = pw_pipi
    fs: str
        String specifying the final state to compute spectrum for.

    Returns
    -------
    dnde : float or array-like
        Value of dnde at positron energy `eng_p`.
    """
    if __recompute_rf_spectra(ms, pws) == 1:
        __set_spectra(ms)
    cdef int num_pts = len(eng_ps)
    cdef int i

    spec = np.zeros(num_pts, dtype=np.float64)

    for i in range(num_pts):
        spec[i] = __dnde_decay_s(eng_ps[i], eng_s, ms, pws, fs)

    return spec
