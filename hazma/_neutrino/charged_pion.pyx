"""
Module for computing the neutrino spectrum from a muon decay.
"""

import numpy as np
cimport numpy as np
import cython
from libc.math cimport log, sqrt, fabs
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
cdef double c_integrand_dnde_mu_numu(double z, double enu, double epi):
    """
    Compute the integrand of boost integral for pi -> mu + numu.

    Parameters
    ----------
    z: double
        Angle muon makes with the z-axis.
    enu: double
        Energy of the neutrino.
    epi: double
        Energy of the pion.

    Returns
    -------
    integrand: double
        The integrand for the boost integral.
    """
    cdef double b
    cdef double g
    cdef double er
    cdef double r
    cdef double emu
    cdef double jac
    cdef NeutrinoSpectrumPoint res

    g = boost_gamma(epi, MASS_PI)
    b = boost_beta(epi, MASS_PI)
    emu = two_body_energy(MASS_PI, MASS_MU, 0.0)

    er = g * enu * (1 - b * z)
    jac = 1.0 / (2.0 * g * (1.0 - b * z))

    res = c_muon_decay_spectrum_point(er, emu)

    return res.muon * jac


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
    cdef double emu_rf
    cdef double enu_rf
    cdef double b
    cdef double delta
    cdef double muon_contrib
    cdef NeutrinoSpectrumPoint result = new_neutrino_spectrum_point()

    if epi < MASS_PI:
        return result

    emu_rf = two_body_energy(MASS_PI, MASS_MU, 0.0)
    enu_rf = two_body_energy(MASS_PI, 0.0, MASS_MU)

    if epi - MASS_PI < DBL_EPSILON:
        result = c_muon_decay_spectrum_point(enu, emu_rf)
        result.electron = BR_PI_TO_MU_NUMU * result.electron  # electron-neutrino
        result.muon = BR_PI_TO_MU_NUMU * result.muon
        result.tau = 0.0
        return result

    b = boost_beta(epi, MASS_PI)

    # Contribution from pi -> nu_mu + mu
    delta = boost_delta_function(enu_rf, enu, 0.0, b)

    # Contribution from pi -> nu_mu + (mu -> nu_mu + nu_e + e)
    muon_contrib = quad(
        c_integrand_dnde_mu_numu,
        -1.0,
        1.0,
        points=[-1.0, 1.0],
        args=(enu, epi),
        epsabs=1e-10,
        epsrel=1e-5
    )[0]

    result.electron = BR_PI_TO_MU_NUMU * muon_contrib  # electron-neutrino
    result.muon = BR_PI_TO_MU_NUMU * (delta + muon_contrib)
    result.tau = 0.0

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
    cdef double b
    cdef double enu_rf
    cdef NeutrinoSpectrumPoint res = new_neutrino_spectrum_point()

    if epi < MASS_PI:
        return res

    b = sqrt(1.0 - (MASS_PI / epi) ** 2)
    enu_rf = (MASS_PI**2 - MASS_MU**2) / (2.0 * MASS_PI)

    res.electron = BR_PI_TO_E_NUE * boost_delta_function(enu_rf, enu, 0.0, b)

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
    result.tau = mu_nu.tau + e_nu.tau

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

