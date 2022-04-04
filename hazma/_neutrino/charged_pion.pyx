"""
Module for computing the neutrino spectrum from a charged pion decay.

The charged pion decays through:
    π⁺ -> μ⁺ + νμ
    π⁺ -> e⁺ + νe
"""

import numpy as np
cimport numpy as np
import cython
from libc.math cimport log, sqrt, fabs, fmax, fmin
from libc.float cimport DBL_EPSILON
from scipy.integrate import quad
from hazma._neutrino.muon cimport c_muon_decay_spectrum_point 
from hazma._neutrino.neutrino cimport NeutrinoSpectrumPoint, new_neutrino_spectrum_point
from hazma._utils.boost cimport boost_delta_function, boost_gamma, boost_beta
from hazma._utils.kinematics cimport two_body_energy
include "../_utils/constants.pxd"  


# ===================================================================
# ---- Pure Cython API functions ------------------------------------
# ===================================================================

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double c_integrand_dnde_mu_numu(double e1, int gen):
    """
    Compute the integrand of boost integral for pi -> mu + numu.

    Parameters
    ----------
    e1: double
        Neutrino energy in original frame.
    e2: double
        Neutrino energy in boosted frame.
    epi: double
        Energy of the pion.
    gen: int
        If 1, electron-neutrino contribution is returned. If 2,
        muon-neutrino contribution is returned.

    Returns
    -------
    integrand: double
        The integrand for the boost integral.
    """
    cdef:
        double emu
        NeutrinoSpectrumPoint res

    emu = two_body_energy(MASS_PI, MASS_MU, 0.0)
    res = c_muon_decay_spectrum_point(e1, emu)

    if gen == 1:
        return res.electron / e1
    else:
        return res.muon / e1


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef NeutrinoSpectrumPoint c_dnde_mu_numu_point(double enu, double epi):
    """
    Compute the boosted charged pion decay neutrino spectrum from pi -> mu + numu.

    Parameters
    ----------
    enu: double
        Energy of the neutrino.
    epi: double
        Energy of the pion.

    Returns
    -------
    dnde: NeutrinoSpectrumPoint
        Structure containing the electron, muon and tau spectra.
    """
    cdef:
        double emu_rf
        double enu_rf
        double beta
        double gamma
        double delta_e
        double delta_m
        double muon_contrib
        double k
        # double ep, em
        double emin, emax
        double pre
        NeutrinoSpectrumPoint result = new_neutrino_spectrum_point()

    if epi < MASS_PI:
        return result

    if epi - MASS_PI < DBL_EPSILON:
        emu_rf = two_body_energy(MASS_PI, MASS_MU, 0.0)
        result = c_muon_decay_spectrum_point(enu, emu_rf)
        result.electron *= BR_PI_TO_MU_NUMU
        result.muon *= BR_PI_TO_MU_NUMU
    else:
        beta = boost_beta(epi, MASS_PI)
        gamma = 1.0 / sqrt(1.0 - beta ** 2)

        pre = 0.5 / (gamma * beta)

        # Contribution from pi -> nu_e + e
        enu_rf = two_body_energy(MASS_PI, 0.0, MASS_E)
        delta_e = BR_PI_TO_E_NUE * boost_delta_function(enu_rf, enu, 0.0, beta)
        
        # Contribution from pi -> nu_mu + mu
        enu_rf = two_body_energy(MASS_PI, 0.0, MASS_MU)
        delta_m = BR_PI_TO_MU_NUMU * boost_delta_function(enu_rf, enu, 0.0, beta)

        # Contribution from pi -> nu_mu + (mu -> nu_mu + nu_e + e)
        emin = fmax(0.0, enu * gamma * (1.0 - beta))
        emax = enu * gamma * (1.0 + beta)

        muon_contrib_nue = pre * BR_PI_TO_MU_NUMU * quad(
                c_integrand_dnde_mu_numu, emin, emax, args=(1,)
        )[0]
        muon_contrib_numu = pre * BR_PI_TO_MU_NUMU * quad(
                c_integrand_dnde_mu_numu, emin, emax, args=(2,)
        )[0]

        result.electron = delta_e + muon_contrib_nue  # electron-neutrino
        result.muon = delta_m + muon_contrib_numu

    return result


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef NeutrinoSpectrumPoint c_dnde_e_nue_point(double enu, double epi):
    """
    Compute the boosted charged pion decay neutrino spectrum from pi -> e + nue.

    Parameters
    ----------
    enu: double
        Energy of the neutrino.
    epi: double
        Energy of the pion.

    Returns
    -------
    dnde: NeutrinoSpectrumPoint
        Structure containing the electron, muon and tau spectra.
    """
    cdef:
        double beta
        double enu_rf
        NeutrinoSpectrumPoint res = new_neutrino_spectrum_point()

    if epi < MASS_PI:
        return res

    beta = sqrt(1.0 - (MASS_PI / epi) ** 2)
    enu_rf = (MASS_PI**2 - MASS_E**2) / (2.0 * MASS_PI)

    res.electron = BR_PI_TO_E_NUE * boost_delta_function(enu_rf, enu, 0.0, beta)

    return res


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef NeutrinoSpectrumPoint c_charged_pion_decay_spectrum_point(double enu, double epi):
    """
    Compute the boosted charged pion decay spectrum into a single neutrino (either an
    electron or muon neutrino) given the neutrino energy and pion energy.

    Parameters
    ----------
    enu: double
        Energy of the neutrino.
    epi: double
        Energy of the pion.

    Returns
    -------
    dnde: NeutrinoSpectrumPoint
        Structure containing the electron, muon and tau spectra.
    """
    cdef NeutrinoSpectrumPoint mu_nu
    cdef NeutrinoSpectrumPoint e_nu
    cdef NeutrinoSpectrumPoint result = new_neutrino_spectrum_point()

    mu_nu = c_dnde_mu_numu_point(enu, epi)
    e_nu = c_dnde_e_nue_point(enu, epi)

    result.electron = mu_nu.electron + e_nu.electron
    result.muon = mu_nu.muon + e_nu.muon

    return result


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t,ndim=2] c_charged_pion_decay_spectrum_array(double[:] energies, double epi):
    """
    Compute the boosted charged-pion decay spectrum into electron-,muon- and tau-neutrinos
    given array of neutrino energies and pion energy.

    Parameters
    ----------
    enu: np.ndarray
        Energies of the neutrino.
    epi: double
        Energy of the charged pion.

    Returns
    -------
    dnde: np.ndarray
        Two-dimensional numpy array containing the electron, muon and tau 
        neutrino spectra. The electron-neutrino spectrum is contained in 
        `dnde[0]`, the muon-neutrino spectrum in `dnde[1]` and tau-neutrino
        spectrum in `dnde[2]`.
    """
    cdef NeutrinoSpectrumPoint res
    cdef int npts = energies.shape[0]
    spec = np.zeros((3,npts), dtype=np.float64)
    cdef double[:,:] spec_view = spec

    for i in range(npts):
        res = c_charged_pion_decay_spectrum_point(energies[i], epi)
        spec_view[0][i] = res.electron
        spec_view[1][i] = res.muon
        spec_view[2][i] = res.tau

    return spec


# ===================================================================
# ---- Python API ---------------------------------------------------
# ===================================================================

def charged_pion_neutrino_spectrum(egam, double epi):
    """
    Compute the neutrino spectrum dN/dE from the decay of a charged pion into 
    e + nu_e and mu + nu_e.

    Paramaters
    ----------
    egam: float or array-like
        Photon energy.
    epi: float 
        Energy of the charged pion.
    """
    cdef NeutrinoSpectrumPoint res
    if hasattr(egam, '__len__'):
        energies = np.array(egam)
        assert len(energies.shape) == 1, "Neutrino energies must be 0 or 1-dimensional."
        return c_charged_pion_decay_spectrum_array(energies, epi)
    else:
        res = c_charged_pion_decay_spectrum_point(egam, epi)
        return (res.electron, res.muon, res.tau)

