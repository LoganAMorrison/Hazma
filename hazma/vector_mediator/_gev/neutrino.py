"""Module for generating neutrino spectra from GeV vector bosons."""

# pylint: disable=invalid-name,protected-access,too-many-lines,too-few-public-methods
# pyright: reportUnusedVariable=false

import functools
from typing import Any, Callable, Protocol, TypedDict

import numpy as np

from hazma import spectra
from hazma.form_factors import vector as vff
from hazma.form_factors.vector import (
    VectorFormFactorPi0K0K0,
    VectorFormFactorPi0KpKm,
    VectorFormFactorPi0Pi0Omega,
    VectorFormFactorPiKK0,
    VectorFormFactorPiPiEta,
    VectorFormFactorPiPiEtaPrime,
    VectorFormFactorPiPiOmega,
    VectorFormFactorPiPiPi0,
    VectorFormFactorPiPiPi0Pi0,
    VectorFormFactorPiPiPiPi,
)
from hazma.parameters import charged_kaon_mass as mk
from hazma.parameters import charged_pion_mass as mpi
from hazma.parameters import eta_mass as meta
from hazma.parameters import eta_prime_mass as metap
from hazma.parameters import muon_mass as mmu
from hazma.parameters import neutral_kaon_mass as mk0
from hazma.parameters import neutral_pion_mass as mpi0
from hazma.parameters import omega_mass as momega
from hazma.parameters import phi_mass as mphi
from hazma.phase_space import PhaseSpaceDistribution1D
from hazma.spectra import boost
from hazma.utils import NeutrinoFlavor, RealArray

from .utils import V_V_REST_FRAME_POINTS

NeutrinoDecaySpectrumFn = Callable[[Any, float, NeutrinoFlavor], Any]


class _DecayDndeNeutrino(Protocol):
    mv: float
    _couplings: vff.VectorFormFactorCouplings
    _ff_pi_pi: vff.VectorFormFactorPiPi
    _ff_pi0_pi0: vff.VectorFormFactorPi0Pi0
    _ff_k_k: vff.VectorFormFactorKK
    _ff_k0_k0: vff.VectorFormFactorK0K0
    _ff_eta_gamma: vff.VectorFormFactorEtaGamma
    _ff_eta_omega: vff.VectorFormFactorEtaOmega
    _ff_eta_phi: vff.VectorFormFactorEtaPhi
    _ff_pi0_gamma: vff.VectorFormFactorPi0Gamma
    _ff_pi0_omega: vff.VectorFormFactorPi0Omega
    _ff_pi0_phi: vff.VectorFormFactorPi0Phi

    _ff_pi_pi_pi0: vff.VectorFormFactorPiPiPi0
    _ff_pi_pi_eta: vff.VectorFormFactorPiPiEta
    _ff_pi_pi_etap: vff.VectorFormFactorPiPiEtaPrime
    _ff_pi_pi_omega: vff.VectorFormFactorPiPiOmega
    _ff_pi0_pi0_omega: vff.VectorFormFactorPi0Pi0Omega
    _ff_pi0_k0_k0: vff.VectorFormFactorPi0K0K0
    _ff_pi0_k_k: vff.VectorFormFactorPi0KpKm
    _ff_pi_k_k0: vff.VectorFormFactorPiKK0
    _ff_pi_pi_pi_pi: vff.VectorFormFactorPiPiPiPi
    _ff_pi_pi_pi0_pi0: vff.VectorFormFactorPiPiPi0Pi0

    def partial_widths(self) -> dict[str, float]: ...


def _make_zeros(neutrino_energies):
    dtype = neutrino_energies.dtype
    shape = neutrino_energies.shape
    return np.zeros(shape, dtype=dtype)


def _make_spectrum_n_body_decay(
    neutrino_energies,
    energy_distributions: list[PhaseSpaceDistribution1D],
    dnde_decays,
    flavor: NeutrinoFlavor,
):

    dnde = np.zeros(neutrino_energies.shape)

    for i, dist in enumerate(energy_distributions):
        bins = dist.bin_centers
        probs = dist.probabilities
        dec = np.array([dnde_decays[i](neutrino_energies, e, flavor) for e in bins])
        dnde += np.trapezoid(np.expand_dims(probs, 1) * dec, x=bins, axis=0)

    return dnde


def _dnde_neutrino_two_body(
    neutrino_energies, *, cme, m1, m2, dnde1, dnde2, flavor: NeutrinoFlavor
):
    if cme < m1 + m2:
        return _make_zeros(neutrino_energies)

    e1 = (cme**2 + m1**2 - m2**2) / (2.0 * cme)
    e2 = (cme**2 - m1**2 + m2**2) / (2.0 * cme)
    return dnde1(neutrino_energies, e1, flavor) + dnde2(neutrino_energies, e2, flavor)


def _dnde_neutrino_zero(neutrino_energies, *_):
    return _make_zeros(neutrino_energies)


# =============================================================================
# ---- 2-body Lepton ----------------------------------------------------------
# =============================================================================


# pylint: disable=unused-argument
def dnde_neutrino_e_e(
    _: _DecayDndeNeutrino,
    neutrino_energies,
    cme: float,
    flavor: NeutrinoFlavor,
):
    """Generate the spectrum into e^+,e^-.

    Parameters
    ----------
    neutrino_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.
    flavor: str
        Flavor of neutrino. Can be 'e', 'mu' or 'tau'.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `neutrino_energies`.
    """
    return _make_zeros(neutrino_energies)


def dnde_neutrino_mu_mu(
    _: _DecayDndeNeutrino,
    neutrino_energies,
    cme: float,
    flavor: NeutrinoFlavor,
):
    """Generate the spectrum into mu^+,mu^-.

    Parameters
    ----------
    neutrino_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.
    flavor: str
        Flavor of neutrino. Can be 'e', 'mu' or 'tau'.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `neutrino_energies`.
    """
    if cme < 2 * mmu:
        return _make_zeros(neutrino_energies)

    dec = 2 * spectra.dnde_neutrino_muon(neutrino_energies, cme / 2.0, flavor)
    return dec


# =============================================================================
# ---- 2-body Meson-Meson -----------------------------------------------------
# =============================================================================


def dnde_neutrino_pi_pi(
    _: _DecayDndeNeutrino,
    neutrino_energies,
    cme: float,
    flavor: NeutrinoFlavor,
):
    """Generate the spectrum into pi^+,pi^-.

    Parameters
    ----------
    neutrino_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.
    flavor: str
        Flavor of neutrino. Can be 'e', 'mu' or 'tau'.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `neutrino_energies`.
    """
    if cme < 2 * mpi:
        return _make_zeros(neutrino_energies)

    dec = 2 * spectra.dnde_neutrino_charged_pion(
        neutrino_energies, cme / 2.0, flavor=flavor
    )
    return dec


def dnde_neutrino_k0_k0(
    _: _DecayDndeNeutrino,
    neutrino_energies,
    cme: float,
    flavor: NeutrinoFlavor,
):
    """Generate the spectrum into k^0,k^0.

    Parameters
    ----------
    neutrino_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.
    flavor: str
        Flavor of neutrino. Can be 'e', 'mu' or 'tau'.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `neutrino_energies`.
    """
    if cme < 2 * mk0:
        return _make_zeros(neutrino_energies)

    dec_l = spectra.dnde_neutrino_long_kaon(neutrino_energies, cme / 2.0, flavor=flavor)
    dec_s = spectra.dnde_neutrino_short_kaon(
        neutrino_energies, cme / 2.0, flavor=flavor
    )
    return dec_l + dec_s


def dnde_neutrino_k_k(
    _: _DecayDndeNeutrino,
    neutrino_energies,
    cme: float,
    flavor: NeutrinoFlavor,
):
    """Generate the spectrum into k^+,k^-.

    Parameters
    ----------
    neutrino_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.
    flavor: str
        Flavor of neutrino. Can be 'e', 'mu' or 'tau'.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `neutrino_energies`.
    """
    if cme < 2 * mk:
        return _make_zeros(neutrino_energies)

    dec = spectra.dnde_neutrino_charged_kaon(
        neutrino_energies, cme / 2.0, flavor=flavor
    )
    return dec


# =============================================================================
# ---- 2-body Meson-Photon ----------------------------------------------------
# =============================================================================


def _dnde_neutrino_m_gamma(neutrino_energies, cme, mass, dnde, flavor: NeutrinoFlavor):
    if cme < mass:
        return _make_zeros(neutrino_energies)
    eng = (cme**2 + mass**2) / (2.0 * cme)
    return dnde(neutrino_energies, eng, flavor=flavor)


# pylint: disable=unused-argument
def dnde_neutrino_pi0_gamma(
    _: _DecayDndeNeutrino,
    neutrino_energies,
    cme: float,
    flavor: NeutrinoFlavor,
):
    """Generate the spectrum into pi^0,gamma.

    Parameters
    ----------
    neutrino_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.
    flavor: str
        Flavor of neutrino. Can be 'e', 'mu' or 'tau'.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `neutrino_energies`.
    """
    return _make_zeros(neutrino_energies)


def dnde_neutrino_eta_gamma(
    _: _DecayDndeNeutrino,
    neutrino_energies,
    cme: float,
    flavor: NeutrinoFlavor,
):
    """Generate the spectrum into eta,gamma.

    Parameters
    ----------
    neutrino_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.
    flavor: str
        Flavor of neutrino. Can be 'e', 'mu' or 'tau'.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `neutrino_energies`.
    """
    return _dnde_neutrino_m_gamma(
        neutrino_energies, cme, meta, spectra.dnde_neutrino_eta, flavor=flavor
    )


# =============================================================================
# ---- 2-body Meson-Phi -------------------------------------------------------
# =============================================================================


def _dnde_neutrino_m_phi(neutrino_energies, *, cme, mass, dnde, flavor: NeutrinoFlavor):
    if cme < mass + mphi:
        return _make_zeros(neutrino_energies)
    dnde_phi = spectra.dnde_neutrino_phi
    return _dnde_neutrino_two_body(
        neutrino_energies,
        cme=cme,
        m1=mass,
        m2=mphi,
        dnde1=dnde,
        dnde2=dnde_phi,
        flavor=flavor,
    )


def dnde_neutrino_pi0_phi(
    _: _DecayDndeNeutrino,
    neutrino_energies,
    cme: float,
    flavor: NeutrinoFlavor,
):
    """Generate the spectrum into pi^0,phi.

    Parameters
    ----------
    neutrino_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.
    flavor: str
        Flavor of neutrino. Can be 'e', 'mu' or 'tau'.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `neutrino_energies`.
    """
    return _dnde_neutrino_m_phi(
        neutrino_energies, cme=cme, mass=mpi0, dnde=_dnde_neutrino_zero, flavor=flavor
    )


def dnde_neutrino_eta_phi(
    _: _DecayDndeNeutrino,
    neutrino_energies,
    cme: float,
    flavor: NeutrinoFlavor,
):
    """Generate the spectrum into eta,phi.

    Parameters
    ----------
    neutrino_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.
    flavor: str
        Flavor of neutrino. Can be 'e', 'mu' or 'tau'.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `neutrino_energies`.
    """
    return _dnde_neutrino_m_phi(
        neutrino_energies,
        cme=cme,
        mass=meta,
        dnde=spectra.dnde_neutrino_eta,
        flavor=flavor,
    )


# =============================================================================
# ---- 2-body Meson-Omega -----------------------------------------------------
# =============================================================================


def dnde_neutrino_eta_omega(
    _: _DecayDndeNeutrino,
    neutrino_energies,
    cme: float,
    flavor: NeutrinoFlavor,
):
    """Generate the spectrum into eta,omega.

    Parameters
    ----------
    neutrino_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.
    flavor: str
        Flavor of neutrino. Can be 'e', 'mu' or 'tau'.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `neutrino_energies`.
    """
    if cme < meta + momega:
        return _make_zeros(neutrino_energies)
    dnde1 = spectra.dnde_neutrino_eta
    dnde2 = spectra.dnde_neutrino_omega
    return _dnde_neutrino_two_body(
        neutrino_energies,
        cme=cme,
        m1=meta,
        m2=momega,
        dnde1=dnde1,
        dnde2=dnde2,
        flavor=flavor,
    )


# =============================================================================
# ---- 3-body -----------------------------------------------------------------
# =============================================================================


# pylint: disable=unused-argument
def dnde_neutrino_pi0_pi0_gamma(
    _: _DecayDndeNeutrino,
    neutrino_energies,
    cme: float,
    flavor: NeutrinoFlavor,
):
    """Generate the spectrum into pi^0,pi^0,gamma.

    Parameters
    ----------
    neutrino_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.
    flavor: str
        Flavor of neutrino. Can be 'e', 'mu' or 'tau'.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `neutrino_energies`.
    """
    return _make_zeros(neutrino_energies)


def dnde_neutrino_pi_pi_pi0(
    self: _DecayDndeNeutrino,
    neutrino_energies,
    cme: float,
    flavor: NeutrinoFlavor,
    *,
    npts: int = 1 << 14,
    nbins: int = 25,
):
    """Generate the spectrum into pi^+,pi^-,pi^0.

    Parameters
    ----------
    neutrino_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.
    flavor: str
        Flavor of neutrino. Can be 'e', 'mu' or 'tau'.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `neutrino_energies`.
    """
    if cme < 2 * mpi + mpi0:
        return _make_zeros(neutrino_energies)

    ff: VectorFormFactorPiPiPi0 = self._ff_pi_pi_pi0

    dnde_decays = [
        spectra.dnde_neutrino_charged_pion,
        spectra.dnde_neutrino_charged_pion,
        _dnde_neutrino_zero,
    ]
    dists = ff.energy_distributions(
        q=cme,
        couplings=self._couplings,
        nbins=nbins,
        npts=npts,
    )
    dnde = _make_spectrum_n_body_decay(
        neutrino_energies, dists, dnde_decays, flavor=flavor
    )

    return dnde


def dnde_neutrino_pi_pi_eta(
    self: _DecayDndeNeutrino,
    neutrino_energies,
    cme: float,
    flavor: NeutrinoFlavor,
    *,
    nbins: int = 25,
):
    """Generate the spectrum into pi^+,pi^-,eta.

    Parameters
    ----------
    neutrino_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.
    flavor: str
        Flavor of neutrino. Can be 'e', 'mu' or 'tau'.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `neutrino_energies`.
    """
    if cme < 2 * mpi + meta:
        return _make_zeros(neutrino_energies)

    ff: VectorFormFactorPiPiEta = self._ff_pi_pi_eta

    dnde_decays = [
        spectra.dnde_neutrino_charged_pion,
        spectra.dnde_neutrino_charged_pion,
        spectra.dnde_neutrino_eta,
    ]
    dists = ff.energy_distributions(
        q=cme,
        couplings=self._couplings,
        nbins=nbins,
    )

    dec = _make_spectrum_n_body_decay(neutrino_energies, dists, dnde_decays, flavor)

    return dec


def dnde_neutrino_pi_pi_etap(
    self: _DecayDndeNeutrino,
    neutrino_energies,
    cme: float,
    flavor: NeutrinoFlavor,
    *,
    nbins: int = 25,
):
    """Generate the spectrum into pi^+,pi^-,eta'.

    Parameters
    ----------
    neutrino_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.
    flavor: str
        Flavor of neutrino. Can be 'e', 'mu' or 'tau'.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `neutrino_energies`.
    """
    if cme < 2 * mpi + metap:
        return _make_zeros(neutrino_energies)

    ff: VectorFormFactorPiPiEtaPrime = self._ff_pi_pi_etap

    dnde_decays = [
        spectra.dnde_neutrino_charged_pion,
        spectra.dnde_neutrino_charged_pion,
        spectra.dnde_neutrino_eta_prime,
    ]
    dists = ff.energy_distributions(
        q=cme,
        couplings=self._couplings,
        nbins=nbins,
    )

    dec = _make_spectrum_n_body_decay(neutrino_energies, dists, dnde_decays, flavor)

    return dec


def dnde_neutrino_pi_pi_omega(
    self: _DecayDndeNeutrino,
    neutrino_energies,
    cme: float,
    flavor: NeutrinoFlavor,
    *,
    nbins: int = 25,
):
    """Generate the spectrum into pi^+,pi^-,omega.

    Parameters
    ----------
    neutrino_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.
    flavor: str
        Flavor of neutrino. Can be 'e', 'mu' or 'tau'.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `neutrino_energies`.
    """
    if cme < 2 * mpi + momega:
        return _make_zeros(neutrino_energies)

    ff: VectorFormFactorPiPiOmega = self._ff_pi_pi_omega

    dnde_decays = [
        spectra.dnde_neutrino_charged_pion,
        spectra.dnde_neutrino_charged_pion,
        spectra.dnde_neutrino_omega,
    ]
    dists = ff.energy_distributions(
        q=cme,
        couplings=self._couplings,
        nbins=nbins,
    )

    dec = _make_spectrum_n_body_decay(neutrino_energies, dists, dnde_decays, flavor)

    return dec


def dnde_neutrino_pi0_pi0_omega(
    self: _DecayDndeNeutrino,
    neutrino_energies,
    cme: float,
    flavor: NeutrinoFlavor,
    *,
    nbins: int = 25,
):
    """Generate the spectrum into pi^0,pi^0,omega.

    Parameters
    ----------
    neutrino_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.
    flavor: str
        Flavor of neutrino. Can be 'e', 'mu' or 'tau'.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `neutrino_energies`.
    """
    if cme < 2 * mpi0 + momega:
        return _make_zeros(neutrino_energies)

    ff: VectorFormFactorPi0Pi0Omega = self._ff_pi0_pi0_omega

    dnde_decays = [
        _dnde_neutrino_zero,
        _dnde_neutrino_zero,
        spectra.dnde_neutrino_omega,
    ]
    dists = ff.energy_distributions(
        q=cme,
        couplings=self._couplings,
        nbins=nbins,
    )

    dec = _make_spectrum_n_body_decay(neutrino_energies, dists, dnde_decays, flavor)

    return dec


def dnde_neutrino_pi0_k0_k0(
    self: _DecayDndeNeutrino,
    neutrino_energies,
    cme: float,
    flavor: NeutrinoFlavor,
    *,
    npts: int = 1 << 14,
    nbins: int = 25,
):
    """Generate the spectrum into pi^0,K^0,K^0.

    Parameters
    ----------
    neutrino_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.
    flavor: str
        Flavor of neutrino. Can be 'e', 'mu' or 'tau'.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `neutrino_energies`.
    """
    if cme < mpi0 + 2 * mk0:
        return _make_zeros(neutrino_energies)

    ff: VectorFormFactorPi0K0K0 = self._ff_pi0_k0_k0

    dnde_decays = [
        _dnde_neutrino_zero,
        spectra.dnde_neutrino_long_kaon,
        spectra.dnde_neutrino_short_kaon,
    ]
    dists = ff.energy_distributions(
        q=cme,
        couplings=self._couplings,
        npts=npts,
        nbins=nbins,
    )

    dec = _make_spectrum_n_body_decay(neutrino_energies, dists, dnde_decays, flavor)

    return dec


def dnde_neutrino_pi0_k_k(
    self: _DecayDndeNeutrino,
    neutrino_energies,
    cme: float,
    flavor: NeutrinoFlavor,
    *,
    npts: int = 1 << 14,
    nbins: int = 25,
):
    """Generate the spectrum into pi^0,K^+,K^-.

    Parameters
    ----------
    neutrino_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.
    flavor: str
        Flavor of neutrino. Can be 'e', 'mu' or 'tau'.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `neutrino_energies`.
    """
    if cme < mpi0 + 2 * mk:
        return _make_zeros(neutrino_energies)

    ff: VectorFormFactorPi0KpKm = self._ff_pi0_k_k

    dnde_decays = [
        _dnde_neutrino_zero,
        spectra.dnde_neutrino_charged_kaon,
        spectra.dnde_neutrino_charged_kaon,
    ]
    dists = ff.energy_distributions(
        q=cme,
        couplings=self._couplings,
        npts=npts,
        nbins=nbins,
    )

    dec = _make_spectrum_n_body_decay(neutrino_energies, dists, dnde_decays, flavor)

    return dec


def dnde_neutrino_pi_k_k0(
    self: _DecayDndeNeutrino,
    neutrino_energies,
    cme: float,
    flavor: NeutrinoFlavor,
    *,
    npts: int = 1 << 14,
    nbins: int = 25,
):
    """Generate the spectrum into pi^+,K^-,K^0.

    Parameters
    ----------
    neutrino_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.
    flavor: str
        Flavor of neutrino. Can be 'e', 'mu' or 'tau'.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `neutrino_energies`.
    """
    if cme < mpi + mk + mk0:
        return _make_zeros(neutrino_energies)

    ff: VectorFormFactorPiKK0 = self._ff_pi_k_k0

    dnde_decays = [
        _dnde_neutrino_zero,
        spectra.dnde_neutrino_charged_kaon,
        spectra.dnde_neutrino_charged_kaon,
    ]
    dists = ff.energy_distributions(
        q=cme,
        couplings=self._couplings,
        npts=npts,
        nbins=nbins,
    )

    dec = _make_spectrum_n_body_decay(neutrino_energies, dists, dnde_decays, flavor)

    return dec


# =============================================================================
# ---- 4-body  ----------------------------------------------------------------
# =============================================================================


def dnde_neutrino_pi_pi_pi_pi(
    self: _DecayDndeNeutrino,
    neutrino_energies,
    cme: float,
    flavor: NeutrinoFlavor,
    *,
    npts: int = 1 << 14,
    nbins: int = 25,
):
    """Generate the spectrum into four charged pions.

    Parameters
    ----------
    neutrino_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.
    flavor: str
        Flavor of neutrino. Can be 'e', 'mu' or 'tau'.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `neutrino_energies`.
    """
    if cme < 4 * mpi:
        return _make_zeros(neutrino_energies)

    ff: VectorFormFactorPiPiPiPi = self._ff_pi_pi_pi_pi

    dnde_decays = [
        spectra.dnde_neutrino_charged_kaon,
        spectra.dnde_neutrino_charged_kaon,
        spectra.dnde_neutrino_charged_pion,
        spectra.dnde_neutrino_charged_pion,
    ]
    dists = ff.energy_distributions(
        q=cme,
        couplings=self._couplings,
        npts=npts,
        nbins=nbins,
    )

    dnde = _make_spectrum_n_body_decay(neutrino_energies, dists, dnde_decays, flavor)

    return dnde


def dnde_neutrino_pi_pi_pi0_pi0(
    self: _DecayDndeNeutrino,
    neutrino_energies,
    cme: float,
    flavor: NeutrinoFlavor,
    *,
    npts: int = 1 << 15,
    nbins: int = 30,
):
    """Generate the spectrum into two neutral and two charged pions.

    Parameters
    ----------
    neutrino_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.
    flavor: str
        Flavor of neutrino. Can be 'e', 'mu' or 'tau'.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `neutrino_energies`.
    """
    if cme < 2 * mpi + 2 * mpi0:
        return _make_zeros(neutrino_energies)

    ff: VectorFormFactorPiPiPi0Pi0 = self._ff_pi_pi_pi0_pi0

    dnde_decays = [
        spectra.dnde_neutrino_charged_kaon,
        spectra.dnde_neutrino_charged_kaon,
        _dnde_neutrino_zero,
        _dnde_neutrino_zero,
    ]
    dists = ff.energy_distributions(
        q=cme,
        couplings=self._couplings,
        npts=npts,
        nbins=nbins,
    )

    dnde = _make_spectrum_n_body_decay(neutrino_energies, dists, dnde_decays, flavor)

    return dnde


def _dnde_neutrino_v_v_rest_frame(  # noqa: PLR0913 — the channel signature plus `bfs`
    self: _DecayDndeNeutrino,
    neutrino_energies: RealArray,
    flavor: NeutrinoFlavor,
    bfs: dict[str, float],
    *,
    npts: int,
    nbins: int,
) -> RealArray:
    """Neutrino spectrum of a pair of vector mediators, each at rest.

    The spectrum is twice the sum of the mediator's decay channels, each
    evaluated at a center-of-mass energy equal to the mediator mass and
    weighted by its branching fraction in `bfs`. Like every channel in this
    module it counts neutrinos and anti-neutrinos together. The neutrino
    lines are left to `dnde_neutrino_v_v`.
    """
    kwargs = {"npts": npts, "nbins": nbins}
    kwargs2 = {"nbins": nbins}

    # Factor of 2 for 2 vectors
    args = (neutrino_energies, self.mv, flavor)
    return 2 * (
        bfs["mu mu"] * dnde_neutrino_mu_mu(self, *args)
        + bfs["pi pi"] * dnde_neutrino_pi_pi(self, *args)
        + bfs["k0 k0"] * dnde_neutrino_k0_k0(self, *args)
        + bfs["k k"] * dnde_neutrino_k_k(self, *args)
        + bfs["pi0 gamma"] * dnde_neutrino_pi0_gamma(self, *args)
        + bfs["eta gamma"] * dnde_neutrino_eta_gamma(self, *args)
        + bfs["pi0 phi"] * dnde_neutrino_pi0_phi(self, *args)
        + bfs["eta phi"] * dnde_neutrino_eta_phi(self, *args)
        + bfs["eta omega"] * dnde_neutrino_eta_omega(self, *args)
        + bfs["pi0 pi0 gamma"] * dnde_neutrino_pi0_pi0_gamma(self, *args)
        + bfs["pi pi pi0"] * dnde_neutrino_pi_pi_pi0(self, *args, **kwargs)
        + bfs["pi pi eta"] * dnde_neutrino_pi_pi_eta(self, *args, **kwargs2)
        + bfs["pi pi etap"] * dnde_neutrino_pi_pi_etap(self, *args, **kwargs2)
        + bfs["pi pi omega"] * dnde_neutrino_pi_pi_omega(self, *args, **kwargs2)
        + bfs["pi0 pi0 omega"] * dnde_neutrino_pi0_pi0_omega(self, *args, **kwargs2)
        + bfs["pi0 k0 k0"] * dnde_neutrino_pi0_k0_k0(self, *args, **kwargs)
        + bfs["pi0 k k"] * dnde_neutrino_pi0_k_k(self, *args, **kwargs)
        + bfs["pi k k0"] * dnde_neutrino_pi_k_k0(self, *args, **kwargs)
        + bfs["pi pi pi pi"] * dnde_neutrino_pi_pi_pi_pi(self, *args, **kwargs)
        + bfs["pi pi pi0 pi0"] * dnde_neutrino_pi_pi_pi0_pi0(self, *args, **kwargs)
    )


# The mediator decay that yields a neutrino line of each flavor.
_NEUTRINO_LINE_CHANNEL: dict[str, str] = {"e": "ve ve", "mu": "vm vm", "tau": "vt vt"}


def dnde_neutrino_v_v(  # noqa: PLR0913 — the channel signature
    self: _DecayDndeNeutrino,
    neutrino_energies,
    cme,
    flavor: NeutrinoFlavor,
    *,
    npts: int = 1 << 15,
    nbins: int = 30,
):
    """Generate the spectrum into two vector mediators.

    Each mediator carries `cme / 2` and decays isotropically in its rest
    frame. The rest-frame spectrum is tabulated on `V_V_REST_FRAME_POINTS`
    log-spaced energies from `1e-6 m_V / 2` to `m_V / 2`, which spans its
    support up to a negligible soft tail, and its linear interpolant is
    boosted exactly by `hazma.spectra.boost.dnde_boost_array`. The
    `V -> nu nubar` line of the requested flavor at `m_V / 2` carries two
    neutrinos per decay and is boosted analytically.

    Parameters
    ----------
    neutrino_energies: array
        Array of neutrino energies where the spectrum should be computed,
        in MeV.
    cme: float
        Center-of-mass energy, in MeV.
    flavor: str
        Flavor of neutrino. Can be 'e', 'mu' or 'tau'.
    npts: int, optional
        Number of phase-space points for the n-body channels.
    nbins: int, optional
        Number of energy bins for the n-body channels.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `neutrino_energies`, in MeV^-1.
    """
    gamma = 0.5 * cme / self.mv
    if gamma < 1.0:
        return _make_zeros(neutrino_energies)

    beta = np.sqrt(1.0 - gamma**-2)

    pws = self.partial_widths()
    width = sum(pws.values())
    bfs = {key: val / width for key, val in pws.items()}

    emax = 0.5 * self.mv
    rest_energies = np.geomspace(1e-6 * emax, emax, V_V_REST_FRAME_POINTS)
    dnde = _dnde_neutrino_v_v_rest_frame(
        self, rest_energies, flavor, bfs, npts=npts, nbins=nbins
    )
    boosted = boost.dnde_boost_array(
        dnde, neutrino_energies, beta, rest_energies=rest_energies
    )

    # Two neutrinos per decay, two vectors
    boosted += (
        4
        * bfs[_NEUTRINO_LINE_CHANNEL[flavor]]
        * boost.boost_delta_function(neutrino_energies, emax, 0.0, beta)
    )
    return boosted


NeutrinoSpectrumFunctions = TypedDict(
    "NeutrinoSpectrumFunctions",
    {
        "e e": NeutrinoDecaySpectrumFn,
        "mu mu": NeutrinoDecaySpectrumFn,
        "ve ve": NeutrinoDecaySpectrumFn,
        "vt vt": NeutrinoDecaySpectrumFn,
        "vm vm": NeutrinoDecaySpectrumFn,
        "pi pi": NeutrinoDecaySpectrumFn,
        "k0 k0": NeutrinoDecaySpectrumFn,
        "k k": NeutrinoDecaySpectrumFn,
        "pi0 gamma": NeutrinoDecaySpectrumFn,
        "eta gamma": NeutrinoDecaySpectrumFn,
        "pi0 phi": NeutrinoDecaySpectrumFn,
        "eta phi": NeutrinoDecaySpectrumFn,
        "eta omega": NeutrinoDecaySpectrumFn,
        "pi0 pi0 gamma": NeutrinoDecaySpectrumFn,
        "pi pi pi0": NeutrinoDecaySpectrumFn,
        "pi pi eta": NeutrinoDecaySpectrumFn,
        "pi pi etap": NeutrinoDecaySpectrumFn,
        "pi pi omega": NeutrinoDecaySpectrumFn,
        "pi0 pi0 omega": NeutrinoDecaySpectrumFn,
        "pi0 k0 k0": NeutrinoDecaySpectrumFn,
        "pi0 k k": NeutrinoDecaySpectrumFn,
        "pi k k0": NeutrinoDecaySpectrumFn,
        "pi pi pi pi": NeutrinoDecaySpectrumFn,
        "pi pi pi0 pi0": NeutrinoDecaySpectrumFn,
        "v v": NeutrinoDecaySpectrumFn,
    },
)


def dnde_neutrino_spectrum_fns(self) -> NeutrinoSpectrumFunctions:
    """Return a dictionary containing functions to generate neutrino spectra.


    Returns
    -------
    dnde_fns: array
        Dictionary containing functions to generate neutrino spectra.
    """

    def dnde_zero(e, _: float, flavor: NeutrinoFlavor):
        return _make_zeros(e)

    def wrap(f):
        @functools.wraps(f)
        def fnew(*args, **kwargs):
            return f(self, *args, **kwargs)

        return fnew

    return {
        "e e": wrap(dnde_neutrino_e_e),
        "mu mu": wrap(dnde_neutrino_mu_mu),
        "ve ve": dnde_zero,
        "vt vt": dnde_zero,
        "vm vm": dnde_zero,
        "pi pi": wrap(dnde_neutrino_pi_pi),
        "k0 k0": wrap(dnde_neutrino_k0_k0),
        "k k": wrap(dnde_neutrino_k_k),
        "pi0 gamma": wrap(dnde_neutrino_pi0_gamma),
        "eta gamma": wrap(dnde_neutrino_eta_gamma),
        "pi0 phi": wrap(dnde_neutrino_pi0_phi),
        "eta phi": wrap(dnde_neutrino_eta_phi),
        "eta omega": wrap(dnde_neutrino_eta_omega),
        "pi0 pi0 gamma": wrap(dnde_neutrino_pi0_pi0_gamma),
        "pi pi pi0": wrap(dnde_neutrino_pi_pi_pi0),
        "pi pi eta": wrap(dnde_neutrino_pi_pi_eta),
        "pi pi etap": wrap(dnde_neutrino_pi_pi_etap),
        "pi pi omega": wrap(dnde_neutrino_pi_pi_omega),
        "pi0 pi0 omega": wrap(dnde_neutrino_pi0_pi0_omega),
        "pi0 k0 k0": wrap(dnde_neutrino_pi0_k0_k0),
        "pi0 k k": wrap(dnde_neutrino_pi0_k_k),
        "pi k k0": wrap(dnde_neutrino_pi_k_k0),
        "pi pi pi pi": wrap(dnde_neutrino_pi_pi_pi_pi),
        "pi pi pi0 pi0": wrap(dnde_neutrino_pi_pi_pi0_pi0),
        "v v": wrap(dnde_neutrino_v_v),
    }
