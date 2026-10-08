"""Module for generating postiron spectra from GeV vector bosons."""

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
from hazma.parameters import electron_mass as me
from hazma.parameters import eta_mass as meta
from hazma.parameters import eta_prime_mass as metap
from hazma.parameters import muon_mass as mmu
from hazma.parameters import neutral_kaon_mass as mk0
from hazma.parameters import neutral_pion_mass as mpi0
from hazma.parameters import omega_mass as momega
from hazma.parameters import phi_mass as mphi
from hazma.phase_space import PhaseSpaceDistribution1D
from hazma.spectra import boost
from hazma.utils import RealArray

from .utils import float_zeros_like, v_v_rest_energies

PositronDecaySpectrumFn = Callable[[Any, float], Any]


class _DecayDndePostiron(Protocol):
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


def _make_spectrum_n_body_decay(
    positron_energies,
    energy_distributions: list[PhaseSpaceDistribution1D],
    dnde_decays,
):
    dnde = np.zeros_like(positron_energies)

    for i, dist in enumerate(energy_distributions):
        probs = dist.probabilities
        bins = dist.bin_centers

        dec = np.array([dnde_decays[i](positron_energies, e) for e in bins])
        dnde += np.trapezoid(np.expand_dims(probs, 1) * dec, x=bins, axis=0)

    return dnde


def _dnde_positron_two_body(positron_energies, *, cme, m1, m2, dnde1, dnde2):
    if cme < m1 + m2:
        return np.zeros_like(positron_energies)
    e1 = (cme**2 + m1**2 - m2**2) / (2.0 * cme)
    e2 = (cme**2 - m1**2 + m2**2) / (2.0 * cme)
    return dnde1(positron_energies, e1) + dnde2(positron_energies, e2)


def _dnde_positron_zero(positron_energies, *_):
    return np.zeros_like(positron_energies)


# =============================================================================
# ---- 2-body Lepton ----------------------------------------------------------
# =============================================================================


def dnde_positron_e_e(
    _: _DecayDndePostiron,
    positron_energies,
    cme: float,
):
    """Generate the spectrum into two electrons.

    Parameters
    ----------
    photon_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `photon_energies`.
    """
    return np.zeros_like(positron_energies)


def dnde_positron_mu_mu(
    _: _DecayDndePostiron,
    positron_energies,
    cme: float,
):
    """Generate the spectrum into two muons.

    Parameters
    ----------
    photon_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `photon_energies`.
    """
    if cme < 2 * mmu:
        return np.zeros_like(positron_energies)

    dec = 2 * spectra.dnde_positron_muon(positron_energies, cme / 2.0)
    return dec


# =============================================================================
# ---- 2-body Meson-Meson -----------------------------------------------------
# =============================================================================


def dnde_positron_pi_pi(
    _: _DecayDndePostiron,
    positron_energies,
    cme: float,
):
    """Generate the spectrum into two pions.

    Parameters
    ----------
    photon_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `photon_energies`.
    """
    if cme < 2 * mpi:
        return np.zeros_like(positron_energies)

    dec = 2 * spectra.dnde_positron_charged_pion(positron_energies, cme / 2.0)
    return dec


def dnde_positron_k0_k0(
    _: _DecayDndePostiron,
    positron_energies,
    cme: float,
):
    """Generate the spectrum into two neutral kaons.

    Parameters
    ----------
    photon_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `photon_energies`.
    """
    if cme < 2 * mk0:
        return np.zeros_like(positron_energies)

    dec_l = spectra.dnde_positron_long_kaon(positron_energies, cme / 2.0)
    dec_s = spectra.dnde_positron_short_kaon(positron_energies, cme / 2.0)
    return dec_l + dec_s


def dnde_positron_k_k(
    _: _DecayDndePostiron,
    positron_energies,
    cme: float,
):
    """Generate the spectrum into two charged kaons.

    Parameters
    ----------
    photon_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `photon_energies`.
    """
    if cme < 2 * mk:
        return np.zeros_like(positron_energies)

    dec = 2 * spectra.dnde_positron_charged_kaon(positron_energies, cme / 2.0)
    return dec


# =============================================================================
# ---- 2-body Meson-Photon ----------------------------------------------------
# =============================================================================


def _dnde_positron_m_gamma(positron_energies, cme, mass, dnde):
    if cme < mass:
        return np.zeros_like(positron_energies)
    eng = (cme**2 + mass**2) / (2.0 * cme)
    return dnde(positron_energies, eng)


def dnde_positron_pi0_gamma(
    _: _DecayDndePostiron,
    positron_energies,
    cme: float,
):
    """Generate the spectrum into a neutral pion and photon.

    Parameters
    ----------
    photon_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `photon_energies`.
    """
    return _dnde_positron_zero(positron_energies)


def dnde_positron_eta_gamma(
    _: _DecayDndePostiron,
    positron_energies,
    cme: float,
):
    """Generate the spectrum into an eta and photon.

    Parameters
    ----------
    photon_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `photon_energies`.
    """
    return _dnde_positron_m_gamma(
        positron_energies, cme, meta, spectra.dnde_positron_eta
    )


# =============================================================================
# ---- 2-body Meson-Phi -------------------------------------------------------
# =============================================================================


def _dnde_positron_m_phi(positron_energies, *, cme, mass, dnde):
    if cme < mass + mphi:
        return np.zeros_like(positron_energies)
    dnde_phi = spectra.dnde_positron_phi
    return _dnde_positron_two_body(
        positron_energies, cme=cme, m1=mass, m2=mphi, dnde1=dnde, dnde2=dnde_phi
    )


def dnde_positron_pi0_phi(
    _: _DecayDndePostiron,
    positron_energies,
    cme: float,
):
    """Generate the spectrum into a neutral pion and phi.

    Parameters
    ----------
    photon_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `photon_energies`.
    """
    return _dnde_positron_m_phi(
        positron_energies, cme=cme, mass=mpi0, dnde=_dnde_positron_zero
    )


def dnde_positron_eta_phi(
    _: _DecayDndePostiron,
    positron_energies,
    cme: float,
):
    """Generate the spectrum into an eta and phi.

    Parameters
    ----------
    photon_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `photon_energies`.
    """
    return _dnde_positron_m_phi(
        positron_energies, cme=cme, mass=meta, dnde=spectra.dnde_positron_eta
    )


# =============================================================================
# ---- 2-body Meson-Omega -----------------------------------------------------
# =============================================================================


def dnde_positron_eta_omega(
    _: _DecayDndePostiron,
    positron_energies,
    cme: float,
):
    """Generate the spectrum into an eta and omega.

    Parameters
    ----------
    photon_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `photon_energies`.
    """
    if cme < meta + momega:
        return np.zeros_like(positron_energies)
    dnde1 = spectra.dnde_positron_eta
    dnde2 = spectra.dnde_positron_omega
    return _dnde_positron_two_body(
        positron_energies, cme=cme, m1=meta, m2=momega, dnde1=dnde1, dnde2=dnde2
    )


# =============================================================================
# ---- 3-body -----------------------------------------------------------------
# =============================================================================


def dnde_positron_pi0_pi0_gamma(
    _: _DecayDndePostiron,
    positron_energies,
    cme: float,
):
    """Generate the spectrum into two neutral pions and a photon.

    Parameters
    ----------
    photon_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `photon_energies`.
    """
    return np.zeros_like(positron_energies)


def dnde_positron_pi_pi_pi0(
    self: _DecayDndePostiron,
    positron_energies,
    cme: float,
    *,
    npts: int = 1 << 14,
    nbins: int = 25,
):
    """Generate the spectrum into two charged pions and a neutral pion.

    Parameters
    ----------
    photon_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `photon_energies`.
    """
    if cme < 2 * mpi + mpi0:
        return np.zeros_like(positron_energies)

    ff: VectorFormFactorPiPiPi0 = self._ff_pi_pi_pi0

    dnde_decays = [
        spectra.dnde_positron_charged_pion,
        spectra.dnde_positron_charged_pion,
        _dnde_positron_zero,
    ]
    dists = ff.energy_distributions(
        q=cme,
        couplings=self._couplings,
        nbins=nbins,
        npts=npts,
    )
    dnde = _make_spectrum_n_body_decay(positron_energies, dists, dnde_decays)

    return dnde


def dnde_positron_pi_pi_eta(
    self: _DecayDndePostiron,
    positron_energies,
    cme: float,
    *,
    nbins: int = 25,
):
    """Generate the spectrum into pi^+, pi^-, eta.

    Parameters
    ----------
    photon_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `photon_energies`.
    """
    if cme < 2 * mpi + meta:
        return np.zeros_like(positron_energies)

    ff: VectorFormFactorPiPiEta = self._ff_pi_pi_eta

    dnde_decays = [
        spectra.dnde_positron_charged_pion,
        spectra.dnde_positron_charged_pion,
        spectra.dnde_positron_eta,
    ]
    dists = ff.energy_distributions(
        q=cme,
        couplings=self._couplings,
        nbins=nbins,
    )

    dec = _make_spectrum_n_body_decay(positron_energies, dists, dnde_decays)

    return dec


def dnde_positron_pi_pi_etap(
    self: _DecayDndePostiron,
    positron_energies,
    cme: float,
    *,
    nbins: int = 25,
):
    """Generate the spectrum into pi^+, pi^-, eta'.

    Parameters
    ----------
    photon_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `photon_energies`.
    """
    if cme < 2 * mpi + metap:
        return np.zeros_like(positron_energies)

    ff: VectorFormFactorPiPiEtaPrime = self._ff_pi_pi_etap

    dnde_decays = [
        spectra.dnde_positron_charged_pion,
        spectra.dnde_positron_charged_pion,
        spectra.dnde_positron_eta_prime,
    ]
    dists = ff.energy_distributions(
        q=cme,
        couplings=self._couplings,
        nbins=nbins,
    )

    dec = _make_spectrum_n_body_decay(positron_energies, dists, dnde_decays)

    return dec


def dnde_positron_pi_pi_omega(
    self: _DecayDndePostiron,
    positron_energies,
    cme: float,
    *,
    nbins: int = 25,
):
    """Generate the spectrum into pi^+, pi^-, omega.

    Parameters
    ----------
    photon_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `photon_energies`.
    """
    if cme < 2 * mpi + momega:
        return np.zeros_like(positron_energies)

    ff: VectorFormFactorPiPiOmega = self._ff_pi_pi_omega

    dnde_decays = [
        spectra.dnde_positron_charged_pion,
        spectra.dnde_positron_charged_pion,
        spectra.dnde_positron_omega,
    ]
    dists = ff.energy_distributions(
        q=cme,
        couplings=self._couplings,
        nbins=nbins,
    )

    dec = _make_spectrum_n_body_decay(positron_energies, dists, dnde_decays)

    return dec


def dnde_positron_pi0_pi0_omega(
    self: _DecayDndePostiron,
    positron_energies,
    cme: float,
    *,
    nbins: int = 25,
):
    """Generate the spectrum into pi^0, pi^0, omega.

    Parameters
    ----------
    photon_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `photon_energies`.
    """
    if cme < 2 * mpi0 + momega:
        return np.zeros_like(positron_energies)

    ff: VectorFormFactorPi0Pi0Omega = self._ff_pi0_pi0_omega

    dnde_decays = [
        _dnde_positron_zero,
        _dnde_positron_zero,
        spectra.dnde_positron_omega,
    ]
    dists = ff.energy_distributions(
        q=cme,
        couplings=self._couplings,
        nbins=nbins,
    )

    dec = _make_spectrum_n_body_decay(positron_energies, dists, dnde_decays)

    return dec


def dnde_positron_pi0_k0_k0(
    self: _DecayDndePostiron,
    positron_energies,
    cme: float,
    *,
    npts: int = 1 << 14,
    nbins: int = 25,
):
    """Generate the spectrum into pi^0, K^0, K^0.

    Parameters
    ----------
    photon_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `photon_energies`.
    """
    if cme < mpi0 + 2 * mk0:
        return np.zeros_like(positron_energies)

    ff: VectorFormFactorPi0K0K0 = self._ff_pi0_k0_k0

    dnde_decays = [
        _dnde_positron_zero,
        spectra.dnde_positron_long_kaon,
        spectra.dnde_positron_short_kaon,
    ]
    dists = ff.energy_distributions(
        q=cme,
        couplings=self._couplings,
        npts=npts,
        nbins=nbins,
    )

    dec = _make_spectrum_n_body_decay(positron_energies, dists, dnde_decays)

    return dec


def dnde_positron_pi0_k_k(
    self: _DecayDndePostiron,
    positron_energies,
    cme: float,
    *,
    npts: int = 1 << 14,
    nbins: int = 25,
):
    """Generate the spectrum into pi^0, K^+, K^-.

    Parameters
    ----------
    photon_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `photon_energies`.
    """
    if cme < mpi0 + 2 * mk:
        return np.zeros_like(positron_energies)

    ff: VectorFormFactorPi0KpKm = self._ff_pi0_k_k

    dnde_decays = [
        _dnde_positron_zero,
        spectra.dnde_positron_charged_kaon,
        spectra.dnde_positron_charged_kaon,
    ]
    dists = ff.energy_distributions(
        q=cme,
        couplings=self._couplings,
        npts=npts,
        nbins=nbins,
    )

    dec = _make_spectrum_n_body_decay(positron_energies, dists, dnde_decays)

    return dec


def dnde_positron_pi_k_k0(
    self: _DecayDndePostiron,
    positron_energies,
    cme: float,
    *,
    npts: int = 1 << 14,
    nbins: int = 25,
):
    """Generate the spectrum into pi^+, K^-, K^0.

    Parameters
    ----------
    photon_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `photon_energies`.
    """
    if cme < mpi + mk + mk0:
        return np.zeros_like(positron_energies)

    ff: VectorFormFactorPiKK0 = self._ff_pi_k_k0

    dnde_decays = [
        _dnde_positron_zero,
        spectra.dnde_positron_charged_kaon,
        spectra.dnde_positron_charged_kaon,
    ]
    dists = ff.energy_distributions(
        q=cme,
        couplings=self._couplings,
        npts=npts,
        nbins=nbins,
    )

    dec = _make_spectrum_n_body_decay(positron_energies, dists, dnde_decays)

    return dec


# =============================================================================
# ---- 4-body  ----------------------------------------------------------------
# =============================================================================


def dnde_positron_pi_pi_pi_pi(
    self: _DecayDndePostiron,
    positron_energies,
    cme: float,
    *,
    npts=1 << 14,
    nbins: int = 25,
):
    """Generate the spectrum into pi^+, pi^-, pi^+, pi^-.

    Parameters
    ----------
    photon_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `photon_energies`.
    """
    if cme < 4 * mpi:
        return np.zeros_like(positron_energies)

    ff: VectorFormFactorPiPiPiPi = self._ff_pi_pi_pi_pi

    dnde_decays = [
        spectra.dnde_positron_charged_kaon,
        spectra.dnde_positron_charged_kaon,
        spectra.dnde_positron_charged_pion,
        spectra.dnde_positron_charged_pion,
    ]
    dists = ff.energy_distributions(
        q=cme,
        couplings=self._couplings,
        npts=npts,
        nbins=nbins,
    )

    dnde = _make_spectrum_n_body_decay(positron_energies, dists, dnde_decays)

    return dnde


def dnde_positron_pi_pi_pi0_pi0(
    self: _DecayDndePostiron,
    positron_energies,
    cme: float,
    *,
    npts: int = 1 << 15,
    nbins: int = 30,
):
    """Generate the spectrum into pi^+, pi^-, pi^0, pi^0.

    Parameters
    ----------
    photon_energies: array
        Array of photon energies where the spectrum should be computed.
    cme: float
        Center-of-mass energy.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `photon_energies`.
    """
    if cme < 2 * mpi + 2 * mpi0:
        return np.zeros_like(positron_energies)

    ff: VectorFormFactorPiPiPi0Pi0 = self._ff_pi_pi_pi0_pi0

    dnde_decays = [
        spectra.dnde_positron_charged_kaon,
        spectra.dnde_positron_charged_kaon,
        _dnde_positron_zero,
        _dnde_positron_zero,
    ]
    dists = ff.energy_distributions(
        q=cme,
        couplings=self._couplings,
        npts=npts,
        nbins=nbins,
    )

    dnde = _make_spectrum_n_body_decay(positron_energies, dists, dnde_decays)

    return dnde


def _dnde_positron_v_v_rest_frame(
    self: _DecayDndePostiron,
    positron_energies: RealArray,
    bfs: dict[str, float],
    *,
    npts: int,
    nbins: int,
) -> RealArray:
    """Positron spectrum of a pair of vector mediators, each at rest.

    The spectrum is twice the sum of the mediator's decay channels, each
    evaluated at a center-of-mass energy equal to the mediator mass and
    weighted by its branching fraction in `bfs`. Like every channel in this
    module it counts electrons and positrons together. The `e e` line is
    left to `dnde_positron_v_v`.
    """
    kwargs = {"npts": npts, "nbins": nbins}
    kwargs2 = {"nbins": nbins}

    # Factor of 2 for 2 vectors
    args = (positron_energies, self.mv)
    return 2 * (
        bfs["mu mu"] * dnde_positron_mu_mu(self, *args)
        + bfs["pi pi"] * dnde_positron_pi_pi(self, *args)
        + bfs["k0 k0"] * dnde_positron_k0_k0(self, *args)
        + bfs["k k"] * dnde_positron_k_k(self, *args)
        + bfs["pi0 gamma"] * dnde_positron_pi0_gamma(self, *args)
        + bfs["eta gamma"] * dnde_positron_eta_gamma(self, *args)
        + bfs["pi0 phi"] * dnde_positron_pi0_phi(self, *args)
        + bfs["eta phi"] * dnde_positron_eta_phi(self, *args)
        + bfs["eta omega"] * dnde_positron_eta_omega(self, *args)
        + bfs["pi0 pi0 gamma"] * dnde_positron_pi0_pi0_gamma(self, *args)
        + bfs["pi pi pi0"] * dnde_positron_pi_pi_pi0(self, *args, **kwargs)
        + bfs["pi pi eta"] * dnde_positron_pi_pi_eta(self, *args, **kwargs2)
        + bfs["pi pi etap"] * dnde_positron_pi_pi_etap(self, *args, **kwargs2)
        + bfs["pi pi omega"] * dnde_positron_pi_pi_omega(self, *args, **kwargs2)
        + bfs["pi0 pi0 omega"] * dnde_positron_pi0_pi0_omega(self, *args, **kwargs2)
        + bfs["pi0 k0 k0"] * dnde_positron_pi0_k0_k0(self, *args, **kwargs)
        + bfs["pi0 k k"] * dnde_positron_pi0_k_k(self, *args, **kwargs)
        + bfs["pi k k0"] * dnde_positron_pi_k_k0(self, *args, **kwargs)
        + bfs["pi pi pi pi"] * dnde_positron_pi_pi_pi_pi(self, *args, **kwargs)
        + bfs["pi pi pi0 pi0"] * dnde_positron_pi_pi_pi0_pi0(self, *args, **kwargs)
    )


def dnde_positron_v_v(
    self: _DecayDndePostiron,
    positron_energies,
    cme,
    *,
    npts: int = 1 << 15,
    nbins: int = 30,
):
    """Generate the spectrum into two vector mediators.

    Each mediator carries `cme / 2` and decays isotropically in its rest
    frame. The rest-frame spectrum is tabulated from the electron mass to
    `m_V / 2`, which spans its support, on the grid of
    `utils.v_v_rest_energies`, and its linear interpolant is boosted
    exactly by `hazma.spectra.boost.dnde_boost_array`. The `V -> e+ e-`
    line at `m_V / 2` carries two leptons per decay and is boosted analytically.

    Parameters
    ----------
    positron_energies: array
        Array of positron energies where the spectrum should be computed,
        in MeV.
    cme: float
        Center-of-mass energy, in MeV.
    npts: int, optional
        Number of phase-space points for the n-body channels.
    nbins: int, optional
        Number of energy bins for the n-body channels.

    Returns
    -------
    dnde: array
        Spectrum evaluated at `positron_energies`, in MeV^-1.
    """
    gamma = 0.5 * cme / self.mv
    if gamma < 1.0 or self.mv <= 2.0 * me:
        return float_zeros_like(positron_energies)

    pws = self.partial_widths()
    width = sum(pws.values())
    if width == 0.0:
        # A mediator with no open decay is stable and yields nothing.
        return float_zeros_like(positron_energies)

    beta = np.sqrt(1.0 - gamma**-2)
    bfs = {key: val / width for key, val in pws.items()}

    rest_energies = v_v_rest_energies(self.mv, me)
    dnde = _dnde_positron_v_v_rest_frame(
        self, rest_energies, bfs, npts=npts, nbins=nbins
    )
    boosted = boost.dnde_boost_array(
        dnde, positron_energies, beta, mass=me, rest_energies=rest_energies
    )

    # Two leptons per decay, two vectors
    boosted += (
        4
        * bfs["e e"]
        * boost.boost_delta_function(positron_energies, 0.5 * self.mv, me, beta)
    )
    return boosted


PositronSpectrumFunctions = TypedDict(
    "NeutrinoSpectrumFunctions",
    {
        "e e": PositronDecaySpectrumFn,
        "mu mu": PositronDecaySpectrumFn,
        "ve ve": PositronDecaySpectrumFn,
        "vt vt": PositronDecaySpectrumFn,
        "vm vm": PositronDecaySpectrumFn,
        "pi pi": PositronDecaySpectrumFn,
        "k0 k0": PositronDecaySpectrumFn,
        "k k": PositronDecaySpectrumFn,
        "pi0 gamma": PositronDecaySpectrumFn,
        "eta gamma": PositronDecaySpectrumFn,
        "pi0 phi": PositronDecaySpectrumFn,
        "eta phi": PositronDecaySpectrumFn,
        "eta omega": PositronDecaySpectrumFn,
        "pi0 pi0 gamma": PositronDecaySpectrumFn,
        "pi pi pi0": PositronDecaySpectrumFn,
        "pi pi eta": PositronDecaySpectrumFn,
        "pi pi etap": PositronDecaySpectrumFn,
        "pi pi omega": PositronDecaySpectrumFn,
        "pi0 pi0 omega": PositronDecaySpectrumFn,
        "pi0 k0 k0": PositronDecaySpectrumFn,
        "pi0 k k": PositronDecaySpectrumFn,
        "pi k k0": PositronDecaySpectrumFn,
        "pi pi pi pi": PositronDecaySpectrumFn,
        "pi pi pi0 pi0": PositronDecaySpectrumFn,
        "v v": PositronDecaySpectrumFn,
    },
)


def dnde_positron_spectrum_fns(self) -> PositronSpectrumFunctions:
    """Return a dictionary containing functions to generate positron spectra.


    Returns
    -------
    dnde_fns: array
        Dictionary containing functions to generate positron spectra.
    """

    def dnde_zero(e, _: float):
        return np.zeros_like(e)

    # The channel functions count particles and antiparticles together.
    # Each final state is either its own charge conjugate or summed with
    # its conjugate, so half of each is the positron spectrum the model reports.
    def wrap(f):
        @functools.wraps(f)
        def fnew(*args, **kwargs):
            return 0.5 * f(self, *args, **kwargs)

        return fnew

    return {
        "e e": wrap(dnde_positron_e_e),
        "mu mu": wrap(dnde_positron_mu_mu),
        "ve ve": dnde_zero,
        "vt vt": dnde_zero,
        "vm vm": dnde_zero,
        "pi pi": wrap(dnde_positron_pi_pi),
        "k0 k0": wrap(dnde_positron_k0_k0),
        "k k": wrap(dnde_positron_k_k),
        "pi0 gamma": wrap(dnde_positron_pi0_gamma),
        "eta gamma": wrap(dnde_positron_eta_gamma),
        "pi0 phi": wrap(dnde_positron_pi0_phi),
        "eta phi": wrap(dnde_positron_eta_phi),
        "eta omega": wrap(dnde_positron_eta_omega),
        "pi0 pi0 gamma": wrap(dnde_positron_pi0_pi0_gamma),
        "pi pi pi0": wrap(dnde_positron_pi_pi_pi0),
        "pi pi eta": wrap(dnde_positron_pi_pi_eta),
        "pi pi etap": wrap(dnde_positron_pi_pi_etap),
        "pi pi omega": wrap(dnde_positron_pi_pi_omega),
        "pi0 pi0 omega": wrap(dnde_positron_pi0_pi0_omega),
        "pi0 k0 k0": wrap(dnde_positron_pi0_k0_k0),
        "pi0 k k": wrap(dnde_positron_pi0_k_k),
        "pi k k0": wrap(dnde_positron_pi_k_k0),
        "pi pi pi pi": wrap(dnde_positron_pi_pi_pi_pi),
        "pi pi pi0 pi0": wrap(dnde_positron_pi_pi_pi0_pi0),
        "v v": wrap(dnde_positron_v_v),
    }
