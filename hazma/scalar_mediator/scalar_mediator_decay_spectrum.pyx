from hazma.spectra._photon._muon cimport dnde_photon_muon_point
from hazma.spectra._photon._pion cimport dnde_photon_charged_pion_point
from hazma.spectra._photon._pion cimport dnde_photon_charged_pion_array
from hazma.spectra._photon._pion cimport dnde_photon_neutral_pion_array
from hazma.spectra._photon._pion cimport dnde_photon_neutral_pion_point

import cython
import numpy as np
cimport numpy as np
from scipy.integrate import quad

from libc.math cimport exp, log, M_PI, log10, sqrt, abs, pow

include "../_decay/parameters.pxd"

cdef int BITFLAG_PP = 1
cdef int BITFLAG_MM = 2
cdef int BITFLAG_P0P0 = 4
cdef int BITFLAG_GG = 8
cdef int BITFLAG_EEG = 16
cdef int BITFLAG_PPG = 32
cdef int BITFLAG_MMG = 64

cdef double mmu = MASS_MU
cdef double me = MASS_E
cdef double mpi = MASS_PI
cdef double qe = sqrt(4. * M_PI * ALPHA_EM)
ctypedef np.ndarray ndarray

cdef int n_interp_pts = 500

cdef np.ndarray e_gams = np.zeros((n_interp_pts,), dtype=np.float64)
cdef np.ndarray spec_cp = np.zeros((n_interp_pts,), dtype=np.float64)

# ===================================================================
# ---- Internal functions -------------------------------------------
# ===================================================================

# Sets a global array so that we don't need to recompute decay spectra 
# for the charged pion unless the scalar mass has changed.

cdef double __set_spectra(double ms):
    global e_gams
    global spec_cp
    e_gams = np.logspace(-1.0, np.log10(ms / 2.0), num=n_interp_pts)
    spec_cp = dnde_photon_charged_pion_array(e_gams, ms / 2.0)


# Use interpolating function to compute charged pion spectrum.

@cython.cdivision(True)
@cython.boundscheck(True)
@cython.wraparound(False)
cdef double interp_spec_cp(double eng_gam):
    if eng_gam < 10**-1:
        return spec_cp[0] * e_gams[0] / eng_gam
    return np.interp(eng_gam, e_gams, spec_cp)


# Compute the FSR off the charged pion.

@cython.cdivision(True)
cdef double dnde_fsr_cp_srf(double egam, double ms):
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
            (-1 + 2 * mupi**2 + x) * 
            log((1 - x - sqrt(1 - x) * sqrt(1 - 4 * mupi**2 - x))**2 /
                (-1 + x - sqrt(1 - x) * sqrt(1 - 4 * mupi**2 - x))**2)
            ) / x

    coeff = qe**2 / (8. * sqrt(1 - 4 * mupi**2) * M_PI**2)
    result = dynamic * coeff
    return 2 * result / ms


# Compute the FSR off either the electron or muon.

@cython.cdivision(True)
cdef double dnde_fsr_l_srf(double egam, double ml, double ms):
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
            (2 - 12 * mul**2 + 16 * mul**4 - 2 * x + 8 * mul**2 * x + x**2) * 
            log((1 - x + 
                sqrt((-1 + x) * (-1 + 4 * mul**2 + x)))**2 / 
                (-1 + x + sqrt((-1 + x) * (-1 + 4 * mul**2 + x)))**2)
            ) / x

    coeff = qe**2 / (16. * (1 - 4 * mul**2)**1.5 * M_PI**2)
    result = dynamic * coeff
    return 2 * result / ms


# Integrand for the scalar mediator decay spectrum.

@cython.cdivision(True)
@cython.boundscheck(True)
@cython.wraparound(False)
cdef double integrand(
    double cl,
    double eng_gam,
    double eng_s,
    double ms,
    np.ndarray[np.float64_t,ndim=1] pws,
    int bitflag
):
    cdef double result = 0.0
    cdef double pwee = pws[0]
    cdef double pwmumu = pws[1]
    cdef double pwpi0pi0 = pws[2]
    cdef double pwpipi = pws[3]

    cdef double beta = sqrt(1. - (ms / eng_s)**2)
    cdef double gamma = eng_s / ms
    cdef double jac = 1. / (2. * gamma * abs(1. - beta * cl))
    cdef double eng_gam_srf = eng_gam * gamma * (1. - beta * cl)

    if bitflag & BITFLAG_EEG:
        result += pwee * dnde_fsr_l_srf(eng_gam_srf, me, ms)
    if bitflag & BITFLAG_PPG:
        result += pwpipi * dnde_fsr_cp_srf(eng_gam_srf, ms)
    if bitflag & BITFLAG_PP:
        result += 2. * pwpipi * interp_spec_cp(eng_gam_srf)
    if bitflag & BITFLAG_P0P0:
        result += 2. * pwpi0pi0 * dnde_photon_neutral_pion_point(eng_gam_srf, ms / 2.0)
    if bitflag & BITFLAG_MMG:
        result += pwmumu * dnde_fsr_l_srf(eng_gam_srf, mmu, ms)
    if bitflag & BITFLAG_MM:
        result += 2. * pwmumu * dnde_photon_muon_point(eng_gam_srf, ms / 2.0)

    return jac * result

# ===================================================================
# ---- Pure Cython API functions ------------------------------------
# ===================================================================

# Compute the photon spectrum from the decay of the scalar-mediator.

@cython.boundscheck(True)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double c_scalar_mediator_decay_spectrum_point(
    double eng_gam,
    double eng_s,
    double ms,
    np.ndarray[np.float64_t,ndim=1] pws, 
    int bitflag
):
    if eng_s < ms:
        return 0.

    cdef double lines_contrib = 0.0
    cdef double beta = sqrt(1. - pow(ms / eng_s, 2.))
    cdef double gamma = eng_s / ms

    cdef double eplus = eng_s * (1. + beta) / 2.0
    cdef double eminus = eng_s * (1. - beta) / 2.0
    cdef double result = 0.0

    result = quad(integrand, -1.0, 1.0, points=[-1.0, 1.0],
                args=(eng_gam, eng_s, ms, pws, bitflag), epsabs=10**-10.,
                epsrel=10**-5.)[0]

    if (bitflag & BITFLAG_GG) and (eminus <= eng_gam <= eplus):
        result += pws[4] * 1. / (eng_s * beta)

    return result


@cython.boundscheck(True)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[np.float64_t,ndim=1] c_scalar_mediator_decay_spectrum_array(
    np.ndarray[np.float64_t,ndim=1] photon_energies, 
    double sm_energy, 
    double sm_mass,
    np.ndarray[np.float64_t,ndim=1] pws, 
    int bitflag
):
    cdef int npts = photon_energies.shape[0]
    cdef np.ndarray[np.float64_t,ndim=1] spec = np.zeros_like(photon_energies)

    for i in range(npts):
        spec[i] = c_scalar_mediator_decay_spectrum_point(photon_energies[i], sm_energy, sm_mass, pws, bitflag)
    return spec



@cython.boundscheck(True)
@cython.wraparound(False)
def scalar_mediator_decay_spectrum(
    photon_energies, 
    sm_energy, 
    sm_mass, 
    partial_widths, 
    modes=["pi pi", "mu mu", "pi0 pi0", "g g", "e e g", "pi pi g", "mu mu g"]
):
    """
    Compute the gamma ray spectrum from the decay of the scalar mediator.

    Parameters
    ----------
    photon_energies : float
        Gamma-ray energy to evaluate spectrum at.
    sm_energy : float
        Energy of the scalar mediator.
    sm_mass: float
        Mass of the scalar mediator.
    partial_widths: List[float]
        Partial widths of the scalar mediator.
    modes: List[str], optional
        List of modes to compute spectrum for. Entries can be:
        "pi pi", "mu mu", "pi0 pi0", "g g", "e e g", "pi pi g", 
        and/or "mu mu g". Default is all of these.

    Returns
    -------
    dnde : float or array-like
        Value of dnde at gamma-ray energy `eng_gam`.
    """
    __set_spectra(sm_mass)
    cdef int bitflag = 0

    if not hasattr(partial_widths, '__len__'):
        raise ValueError("Partial widths must be a list or array.")
    pws = np.array(partial_widths)
    assert len(pws.shape) == 1, "Partial widths must be 1-dimensional."

    if "pi pi" in modes:
        bitflag += BITFLAG_PP
    if "mu mu" in modes:
        bitflag += BITFLAG_MM
    if "pi0 pi0" in modes:
        bitflag += BITFLAG_P0P0
    if "g g" in modes:
        bitflag += BITFLAG_GG
    if "e e g" in modes:
        bitflag += BITFLAG_EEG
    if "pi pi g" in modes:
        bitflag += BITFLAG_PPG
    if "mu mu g" in modes:
        bitflag += BITFLAG_MMG

    if hasattr(photon_energies, '__len__'):
        energies = np.array(photon_energies)
        assert len(energies.shape) == 1, "Photon energies must be 0 or 1-dimensional."
        return c_scalar_mediator_decay_spectrum_array(energies, sm_energy, sm_mass, pws, bitflag)
    return c_scalar_mediator_decay_spectrum_point(photon_energies, sm_energy, sm_mass, pws, bitflag)
