"""
This file contains the decay spectra from a right-handed neutrino at rest.
"""
from typing import Callable

import numpy as np

from hazma import spectra
from hazma import parameters
from hazma.form_factors.vector import VectorFormFactorPiPi

from hazma.gamma_ray import gamma_ray_decay

from ._proto import SingleRhNeutrinoModel
from ._widths import invariant_mass_distribution_v_pi_pi
from ._widths import invariant_mass_distribution_l_pi0_pi
from ._widths import invariant_mass_distribution_v_f_f


_lepton_strs = ["e", "mu", "tau"]
_lepton_masses = {
    "e": parameters.electron_mass,
    "mu": parameters.muon_mass,
    "tau": parameters.tau_mass,
}


def dnde_zero(e, _):
    return np.zeros_like(e)


_lepton_decay_spectra = {
    "e": dnde_zero,
    "mu": spectra.dnde_photon_muon,
    "tau": dnde_zero,
}


def _dnde_v_m(
    model: SingleRhNeutrinoModel, photon_energies, mp: float, dnde_m: Callable
):
    """
    Compute the gamma-ray spectrum from the decay of a right-handed
    neutrino into a neutral pion and neutrino.

    Parameters
    ----------
    photon_energies: float or np.array
        Photon energies where the spectrum should be computed.
    """
    single = np.isscalar(photon_energies)
    e = np.atleast_1d(photon_energies).astype(np.float64)
    dnde = np.zeros_like(e)

    mx = model.mx

    if mx > mp:
        mx = model.mx
        ep = (mx**2 + mp**2) / (2 * mx)
        dnde = dnde_m(e, ep)

    if single:
        return dnde[0]

    return dnde


def dnde_v_pi0(model: SingleRhNeutrinoModel, photon_energies):
    """
    Compute the gamma-ray spectrum from the decay of a right-handed
    neutrino into a neutral pion and neutrino.

    Parameters
    ----------
    photon_energies: float or np.array
        Photon energies where the spectrum should be computed.
    """
    mp = parameters.neutral_pion_mass
    dnde_m = spectra.dnde_photon_neutral_pion
    return _dnde_v_m(model, photon_energies, mp, dnde_m)


def dnde_v_eta(model: SingleRhNeutrinoModel, photon_energies):
    """
    Compute the gamma-ray spectrum from the decay of a right-handed
    neutrino into a eta and neutrino.

    Parameters
    ----------
    photon_energies: float or np.array
        Photon energies where the spectrum should be computed.
    """
    mp = parameters.eta_mass
    dnde_m = spectra.dnde_photon_eta
    return _dnde_v_m(model, photon_energies, mp, dnde_m)


def _dnde_l_m(
    model: SingleRhNeutrinoModel, photon_energies, mp: float, dnde_m: Callable
):
    """
    Compute the gamma-ray spectrum from the decay of a right-handed
    neutrino into a charged meson and lepton.

    Parameters
    ----------
    photon_energies: float or np.array
        Photon energies where the spectrum should be computed.
    """
    single = np.isscalar(photon_energies)
    e = np.atleast_1d(photon_energies).astype(np.float64)
    dnde = np.zeros_like(e)

    mx = model.mx
    ml = _lepton_masses[model.flavor]

    if mx > mp + ml:
        mx = model.mx
        ep = (mx**2 + mp**2 - ml**2) / (2 * mx)
        el = (mx**2 - mp**2 + ml**2) / (2 * mx)
        dnde = (
            dnde_m(e, ep)
            + _lepton_decay_spectra[model.flavor](e, el)
            + spectra.dnde_photon_ap_scalar(e, mx**2, mp)
            + spectra.dnde_photon_ap_fermion(e, mx**2, ml)
        )

    if single:
        return dnde[0]

    return dnde


def dnde_l_pi(model: SingleRhNeutrinoModel, photon_energies):
    """
    Compute the gamma-ray spectrum from the decay of a right-handed
    neutrino into a charged pion and lepton.

    Parameters
    ----------
    photon_energies: float or np.array
        Photon energies where the spectrum should be computed.
    """
    mp = parameters.charged_pion_mass
    dnde_m = spectra.dnde_photon_charged_pion
    return _dnde_l_m(model, photon_energies, mp, dnde_m)


def dnde_l_k(model: SingleRhNeutrinoModel, photon_energies):
    """
    Compute the gamma-ray spectrum from the decay of a right-handed
    neutrino into a charged kaon and lepton.

    Parameters
    ----------
    photon_energies: float or np.array
        Photon energies where the spectrum should be computed.
    spectrum_type: str, optional
        String specifying which spectrum component should be computed.
        Options are: "all", "decay" or "fsr"
    """
    mp = parameters.charged_kaon_mass
    dnde_m = spectra.dnde_photon_charged_kaon
    return _dnde_l_m(model, photon_energies, mp, dnde_m)


def dnde_v_l_l(model: SingleRhNeutrinoModel, photon_energies, j, n, m, nbins: int):
    """
    Compute the gamma-ray spectrum from the decay of a right-handed
    neutrino into an active neutrino and two charged leptons.

    Parameters
    ----------
    photon_energies: float or np.array
        Photon energies where the spectrum should be computed.
    """
    return np.zeros_like(photon_energies)


def dnde_l_pi0_pi(
    model: SingleRhNeutrinoModel, photon_energies, ff: VectorFormFactorPiPi, nbins: int
):
    """
    Compute the gamma-ray spectrum from the decay of a right-handed
    neutrino into a charged pion, neutral pion and charged lepton.

    Parameters
    ----------
    photon_energies: float or np.array
        Photon energies where the spectrum should be computed.
    """

    mx = model.mx
    mpi = parameters.charged_pion_mass
    ml = _lepton_masses[model.flavor]

    single = np.isscalar(photon_energies)
    e = np.atleast_1d(photon_energies).astype(np.float64)
    dnde = np.zeros_like(e)

    def dnde_(s):
        epi = 0.5 * np.sqrt(s)
        dec1 = spectra.dnde_photon_charged_pion(e, epi)
        dec2 = spectra.dnde_photon_neutral_pion(e, epi)
        fsrp = spectra.dnde_photon_ap_scalar(e, s, mpi)
        fsrl = spectra.dnde_photon_ap_fermion(e, s, mpi)

        return dec1 + dec2 + fsrp + fsrl

    if mx > 2 * mpi:
        smin = 4 * mpi**2
        smax = mx**2
        ss = np.linspace(smin, smax, nbins)
        dwds = invariant_mass_distribution_l_pi0_pi(model, ss, ml, ff)
        norm = np.trapz(dwds, ss)
        if norm > 0.0:
            dwds = dwds / norm
            dnde = np.sum(np.array([p * dnde_(s) for p, s in zip(dwds, ss)]), axis=0)

    if single:
        return dnde[0]

    return dnde


def dnde_nu_pi_pi(
    model: SingleRhNeutrinoModel,
    photon_energies,
    ff: VectorFormFactorPiPi,
    nbins: int = 30,
):
    """
    Compute the gamma-ray spectrum from the decay of a right-handed
    neutrino into an active neutrino and two charged pions.

    Parameters
    ----------
    photon_energies: float or np.array
        Photon energies where the spectrum should be computed.
    """
    mpi = parameters.charged_pion_mass
    mx = model.mx

    single = np.isscalar(photon_energies)
    e = np.atleast_1d(photon_energies).astype(np.float64)
    dnde = np.zeros_like(e)

    if mx > 2 * mpi:
        smin = 4 * mpi**2
        smax = mx**2
        ss = np.linspace(smin, smax, nbins)
        dwds = invariant_mass_distribution_v_pi_pi(model, ss, ff)
        norm = np.trapz(dwds, ss)
        if norm > 0.0:
            dwds = dwds / norm
            for s, p in zip(ss, dwds):
                epi = 0.5 * np.sqrt(s)
                dec = spectra.dnde_photon_charged_pion(e, epi)
                fsr = spectra.dnde_photon_ap_scalar(e, s, mpi)
                dnde = dnde + 2.0 * p * (dec + fsr)

    if single:
        return dnde[0]

    return dnde
