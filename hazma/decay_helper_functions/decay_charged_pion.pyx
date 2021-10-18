from hazma.decay_helper_functions.decay_muon cimport c_muon_decay_spectrum_point
import numpy as np
cimport numpy as np
from scipy.integrate import quad
from libc.math cimport exp, log, M_PI, log10, sqrt, abs, pow
import cython
include "common.pxd"


cdef double ENG_GAM_MAX_MURF = 52.82795006985128
cdef double ENG_GAM_MAX_PIRG = 69.78345771948752
cdef double ENG_MU_PIRF = 109.77820123634007
cdef double FPI = DECAY_CONST_PI / sqrt(2.)  # ~92 MeV
cdef double MPI = MASS_PI
cdef double ME = MASS_E
cdef double MMU = MASS_MU
cdef double BETA_MU_PIRF = 0.27138337509758564
cdef double GAMMA_MU_PIRF = 1.0389919859434902


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double __dnde_muon_pirf(double egam):
    return c_muon_decay_spectrum_point(egam, ENG_MU_PIRF)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double __dnde_pi_to_lnug(double x, double r):
    """
    Helper function for computing dnde from pi-> l nu g.
    """
    # Account for energy-dependence of vector form factor
    cdef double F_V = F_V_PI * (1 + F_V_PI_SLOPE * (1 - x))

    # Numerator terms with no log
    cdef double f = (r + x - 1) * (
        MPI*MPI * x*x*x*x * (F_A_PI*F_A_PI + F_V*F_V) * (r*r - r*x + r - 2 * (x-1)*(x-1))
        - 12 * sqrt(2) * FPI * MPI * r * (x-1) * x*x * (F_A_PI * (r - 2*x + 1) + F_V * x)
        - 24 * FPI*FPI * r * (x-1) * (4*r*(x-1) + (x-2)*(x-2)))

    # Numerator terms with log
    cdef double g = 12 * sqrt(2) * FPI * r * (x-1)*(x-1) * log(r / (1-x)) * (
        MPI * x*x * (F_A_PI * (x - 2*r) - F_V * x)
        + sqrt(2) * FPI * (2*r*r - 2*r*x - x*x + 2*x - 2))

    return ALPHA_EM * (f + g) / (24 * M_PI * MPI * FPI*FPI * (r-1)*(r-1)
                                 * (x-1)*(x-1) * r * x)


@cython.cdivision(True)
cdef double dnde_pi_to_lnug(double egam, double ml):
    """
    Returns dnde from pi-> l nu g.
    """
    cdef double x = 2 * egam / MPI
    cdef double r = (ml / MPI) * (ml / MPI)

    if 0.0 <= x and x <= (1 - r):
        return __dnde_pi_to_lnug(x, r)
    else :
        return 0.0


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
    cdef double beta_pi = beta(eng_pi, MASS_PI)
    cdef double gamma_pi = gamma(eng_pi, MASS_PI)
    return ENG_GAM_MAX_MURF * gamma_pi * GAMMA_MU_PIRF * (1.0 + beta_pi) * (1.0 + BETA_MU_PIRF)


@cython.cdivision(True)
cdef double integrand(double cl, double eng_gam, double eng_pi, int bitflag):
    """
    Returns the integrand of the differential radiative decay spectrum for
    the charged pion.

    Keyword arguments::
        cl: Angle of photon w.r.t. charged pion in lab frame.
        engPi: Energy of photon in laboratory frame.
        engPi: Energy of pion in laboratory frame.
    """
    cdef double beta_pi = beta(eng_pi, MASS_PI)
    cdef double gamma_pi = gamma(eng_pi, MASS_PI)

    cdef double eng_gam_pi_rF = eng_gam * gamma_pi * (1.0 - beta_pi * cl)
    cdef double jac = 1. / (2.0 * gamma_pi * abs(1.0 - beta_pi * cl))

    cdef double result = 0.0

    if (bitflag & 1) and (0.0 < eng_gam_pi_rF and eng_gam_pi_rF < ENG_GAM_MAX_PIRG):
        result += BR_PI_TO_MUNU * jac * c_muon_decay_spectrum_point(eng_gam_pi_rF, ENG_MU_PIRF)

    if bitflag & 2:
        result += BR_PI_TO_MUNU * jac * dnde_pi_to_lnug(eng_gam_pi_rF, MMU)

    if bitflag & 4:
        result += BR_PI_TO_ENU * jac * dnde_pi_to_lnug(eng_gam_pi_rF, ME)

    return result

cdef double c_charged_pion_decay_spectrum_point(double eng_gam, double eng_pi, int mode):
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

    return quad(integrand, -1.0, 1.0, points=[-1.0, 1.0], \
                   args=(eng_gam, eng_pi, mode), epsabs=1e-10, \
                   epsrel=1e-5)[0]


@cython.boundscheck(True)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t,ndim=1] c_charged_pion_decay_spectrum_array(np.ndarray[np.float64_t,ndim=1] egams, double epi, int mode):
    cdef int npts = egams.shape[0]
    cdef np.ndarray[np.float64_t,ndim=1] spec = np.zeros_like(egams)

    for i in range(npts):
        spec[i] = c_charged_pion_decay_spectrum_point(egams[i], epi, mode)
    return spec


@cython.boundscheck(True)
@cython.wraparound(False)
def charged_pion_decay_spectrum(egams, epi, modes=["munu", "munug", "enug"]):
    """
    Compute the photon spectrum dN/dE from the decay of a charged pion.

    Paramaters
    ----------
    egam: float or array-like
        Photon energy.
    epi: float 
        Energy of the pion.
    mode: optional, List 
        List of modes to compute spectrum for. Entries can be:
        "munu", "munug" and "enug". Default is all of these.
    """
    cdef int bitflag = 0

    if "munu" in modes:
        bitflag += 1
    if "munug" in modes:
        bitflag += 2
    if "enug" in modes:
        bitflag += 4

    if bitflag == 0:
        raise ValueError("Invalid modes specified.") 
    
    if hasattr(egams, '__len__'):
        energies = np.array(egams)
        assert len(energies.shape) == 1, "Photon energies must be 0 or 1-dimensional."
        return c_charged_pion_decay_spectrum_array(energies, epi, bitflag)
    else:
        return c_charged_pion_decay_spectrum_point(egams, epi, bitflag)
