import numpy as np
cimport numpy as np

from scipy.integrate import quad
from libc.math cimport exp, log, M_PI, log10, sqrt, abs, pow, M_SQRT1_2

import cython

from hazma.spectra._photon._muon cimport dnde_photon_muon_point
from hazma._utils.boost cimport boost_beta, boost_gamma

# include "common.pxd"
include "../../_utils/constants.pxd"


DEF ENG_GAM_MAX_MURF = 52.82795006985128
DEF ENG_GAM_MAX_PIRG = 69.78345771948752
DEF ENG_MU_PIRF = 109.77820123634007
cdef double FPI = DECAY_CONST_PI * M_SQRT1_2  # ~92 MeV
DEF MPI = MASS_PI
DEF ME = MASS_E
DEF MMU = MASS_MU
DEF BETA_MU_PIRF = 0.27138337509758564
DEF GAMMA_MU_PIRF = 1.0389919859434902


# ============================================================================
# ---- Charged Pion ----------------------------------------------------------
# ============================================================================

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double dnde_pi_to_lnug(double egam, double ml):
    """
    Returns dnde from pi-> l nu g.
    """
    cdef double x = 2 * egam / MPI
    cdef double r = (ml / MPI) * (ml / MPI)
    cdef double F_V
    cdef double f
    cdef double g

    if x < 0.0 or (1 - r) < x:
        return 0.0

    # Account for energy-dependence of vector form factor
    F_V = F_V_PI * (1 + F_V_PI_SLOPE * (1 - x))

    # Numerator terms with no log
    f = (r + x - 1) * (
        MPI*MPI * x*x*x*x * (F_A_PI*F_A_PI + F_V*F_V) * (r*r - r*x + r - 2 * (x-1)*(x-1))
        - 12 * sqrt(2) * FPI * MPI * r * (x-1) * x*x * (F_A_PI * (r - 2*x + 1) + F_V * x)
        - 24 * FPI*FPI * r * (x-1) * (4*r*(x-1) + (x-2)*(x-2)))

    # Numerator terms with log
    g = 12 * sqrt(2) * FPI * r * (x-1)*(x-1) * log(r / (1-x)) * (
        MPI * x*x * (F_A_PI * (x - 2*r) - F_V * x)
        + sqrt(2) * FPI * (2*r*r - 2*r*x - x*x + 2*x - 2))

    return ALPHA_EM * (f + g) / (24 * M_PI * MPI * FPI*FPI * (r-1)*(r-1)
                                 * (x-1)*(x-1) * r * x)



@cython.cdivision(True)
cdef double eng_gam_max(double eng_pi):
    """
    Returns the maximum allowed gamma ray energy from a charged pion decay.
    Keyword arguments::
        eng_pi: Energy of pion in laboratory frame.
    More details:
        This is computed using the fact that in the mu restframe, the
        Maximum allowed value is
            ENG_GAM_MAX_MURF = (pow(mass_mu,2.0) - pow(mass_e,2.0))
                            / (2.0 * mass_mu).
        Then, boosting into the pion rest frame, then to the mu rest
        frame, we get the maximum allowed energy in the lab frame.
    """
    cdef double beta_pi = boost_beta(eng_pi, MASS_PI)
    cdef double gamma_pi = boost_gamma(eng_pi, MASS_PI)
    return ENG_GAM_MAX_MURF * gamma_pi * GAMMA_MU_PIRF * (1.0 + beta_pi) * (1.0 + BETA_MU_PIRF)


@cython.cdivision(True)
cdef double charged_pion_integrand(double cl, double eng_gam, double eng_pi):
    """
    Returns the charged_pion_integrand of the differential radiative decay spectrum for
    the charged pion.
    Keyword arguments::
        cl: Angle of photon w.r.t. charged pion in lab frame.
        engPi: Energy of photon in laboratory frame.
        engPi: Energy of pion in laboratory frame.
    """
    cdef double beta_pi = boost_beta(eng_pi, MASS_PI)
    cdef double gamma_pi = boost_gamma(eng_pi, MASS_PI)

    cdef double eng_gam_pi_rF = eng_gam * gamma_pi * (1.0 - beta_pi * cl)
    cdef double jac = 1. / (2.0 * gamma_pi * abs(1.0 - beta_pi * cl))

    cdef double result = 0.0

    result += BR_PI_TO_MU_NUMU * jac * dnde_photon_muon_point(eng_gam_pi_rF, ENG_MU_PIRF)
    result += BR_PI_TO_MU_NUMU * jac * dnde_pi_to_lnug(eng_gam_pi_rF, MMU)
    result += BR_PI_TO_E_NUE * jac * dnde_pi_to_lnug(eng_gam_pi_rF, ME)

    return result


cdef double dnde_photon_charged_pion_point(double eng_gam, double eng_pi):
    """
    Returns the radiative spectrum value from charged pion given a gamma
    ray energy eng_gam and charged pion energy eng_pi. When the
    ChargedPion object is instatiated, an interplating function for the
    mu spectrum is computed.
    Keyword arguments::
        eng_gam: Energy of photon is laboratory frame.
        eng_pi: Energy of charged pion in laboratory frame.
    """
    if eng_pi < MASS_PI:
        return 0.0

    return quad(charged_pion_integrand, -1.0, 1.0, points=[-1.0, 1.0], \
                   args=(eng_gam, eng_pi), epsabs=1e-10, \
                   epsrel=1e-5)[0]



@cython.boundscheck(True)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t,ndim=1] dnde_photon_charged_pion_array(double[:] egams, double epi):
    cdef int npts = egams.shape[0]
    cdef np.ndarray[np.float64_t,ndim=1] spec = np.zeros_like(egams)

    for i in range(npts):
        spec[i] = dnde_photon_charged_pion_point(egams[i], epi)
    return spec



@cython.boundscheck(True)
@cython.wraparound(False)
def dnde_photon_charged_pion(photon_energy, pion_energy):
    """
    Compute the photon spectrum dN/dE from the decay of a charged pion.

    Paramaters
    ----------
    photon_energy: float or array-like
        Photon energy.
    pion_energy: float 
        Energy of the pion.
    """
    if hasattr(photon_energy, '__len__'):
        energies = np.array(photon_energy)
        assert len(energies.shape) == 1, "Photon energies must be 0 or 1-dimensional."
        return dnde_photon_charged_pion_array(energies, pion_energy)
    else:
        return dnde_photon_charged_pion_point(photon_energy, pion_energy)


# ============================================================================
# ---- Neutral Pion ----------------------------------------------------------
# ============================================================================


@cython.cdivision(True)
cdef double dnde_photon_neutral_pion_point(double eng_gam, double eng_pi):
    """
    Returns decay spectrum for pi0 -> g g.
    """
    if eng_pi < MASS_PI0:
        return 0.0

    cdef float beta = sqrt(1.0 - (MASS_PI0 / eng_pi)**2)
    cdef float ret_val = 0.0

    if eng_pi * (1 - beta) / 2.0 <= eng_gam <= eng_pi * (1 + beta) / 2.0:
        ret_val = BR_PI0_TO_A_A * 2.0 / (eng_pi * beta)

    return ret_val

@cython.cdivision(True)
cdef np.ndarray[np.float64_t,ndim=1] dnde_photon_neutral_pion_array(double[:] egams, double epi):
    """
    Returns decay spectrum for pi0 -> g g.
    """
    cdef int npts = egams.shape[0]
    cdef np.ndarray[np.float64_t,ndim=1] spec = np.zeros_like(egams)
    for i in range(npts):
        spec[i] = dnde_photon_neutral_pion_point(egams[i], epi)
    return spec


@cython.cdivision(True)
@cython.boundscheck(True)
@cython.wraparound(False)
def dnde_photon_neutral_pion(photon_energy, pion_energy):
    """
    Compute the photon spectrum dN/dE from the decay of a neutral pion.
    Paramaters
    ----------
    photon_energy: float or array-like
        Photon energy.
    pion_energy: float 
        Energy of the pion.
    """
    if hasattr(photon_energy, '__len__'):
        energies = np.array(photon_energy)
        assert len(energies.shape) == 1, "Photon energies must be 0 or 1-dimensional."
        return dnde_photon_neutral_pion_array(energies, pion_energy)
    else:
        return dnde_photon_neutral_pion_point(photon_energy, pion_energy)