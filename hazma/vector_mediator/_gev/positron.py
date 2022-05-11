import numpy as np

from hazma import spectra
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
from hazma.vector_mediator.form_factors.four_pi import FormFactorPiPiPiPi
from hazma.vector_mediator.form_factors.pi_k_k import (
    FormFactorPi0K0K0,
    FormFactorPi0KpKm,
    FormFactorPiKK0,
)
from hazma.vector_mediator.form_factors.pi_pi_eta import FormFactorPiPiEta
from hazma.vector_mediator.form_factors.pi_pi_etap import FormFactorPiPiEtaP
from hazma.vector_mediator.form_factors.pi_pi_omega import FormFactorPiPiOmega
from hazma.vector_mediator.form_factors.pi_pi_pi0 import FormFactorPiPiPi0


def make_spectrum_n_body_decay(positron_energies, energy_distributions, dnde_decays):
    dnde = np.zeros_like(positron_energies)

    for i, (probs, bins) in enumerate(energy_distributions):
        dec = np.array([dnde_decays[i](positron_energies, e) for e in bins])
        dnde += np.trapz(np.expand_dims(probs, 1) * dec, x=bins, axis=0)

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


def dnde_positron_e_e(self, positron_energies, cme: float):
    return np.zeros_like(positron_energies)


def dnde_positron_mu_mu(self, positron_energies, cme: float):
    if cme < 2 * mmu:
        return np.zeros_like(positron_energies)

    dec = 2 * spectra.dnde_positron_muon(positron_energies, cme / 2.0)
    return dec


# =============================================================================
# ---- 2-body Meson-Meson -----------------------------------------------------
# =============================================================================


def dnde_positron_pi_pi(self, positron_energies, cme: float):
    if cme < 2 * mpi:
        return np.zeros_like(positron_energies)

    dec = 2 * spectra.dnde_positron_charged_pion(positron_energies, cme / 2.0)
    return dec


def dnde_positron_k0_k0(self, positron_energies, cme: float):
    if cme < 2 * mk0:
        return np.zeros_like(positron_energies)

    dec_l = spectra.dnde_positron_long_kaon(positron_energies, cme / 2.0)
    dec_s = spectra.dnde_positron_short_kaon(positron_energies, cme / 2.0)
    return dec_l + dec_s


def dnde_positron_k_k(self, positron_energies, cme: float):
    if cme < 2 * mk:
        return np.zeros_like(positron_energies)

    dec = spectra.dnde_positron_charged_kaon(positron_energies, cme / 2.0)
    return dec


# =============================================================================
# ---- 2-body Meson-Photon ----------------------------------------------------
# =============================================================================


def _dnde_positron_m_gamma(positron_energies, cme, mass, dnde):
    if cme < mass:
        return np.zeros_like(positron_energies)
    eng = (cme**2 + mass**2) / (2.0 * cme)
    return dnde(positron_energies, eng)


def dnde_positron_pi0_gamma(self, positron_energies, cme: float):
    return _dnde_positron_zero(positron_energies)


def dnde_positron_eta_gamma(self, positron_energies, cme: float):
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


def dnde_positron_pi0_phi(self, positron_energies, cme: float):
    return _dnde_positron_m_phi(
        positron_energies, cme=cme, mass=mpi0, dnde=_dnde_positron_zero
    )


def dnde_positron_eta_phi(self, positron_energies, cme: float):
    return _dnde_positron_m_phi(
        positron_energies, cme=cme, mass=meta, dnde=spectra.dnde_positron_eta
    )


# =============================================================================
# ---- 2-body Meson-Omega -----------------------------------------------------
# =============================================================================


def dnde_positron_eta_omega(self, positron_energies, cme: float):
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


def dnde_positron_pi0_pi0_gamma(self, positron_energies, cme: float):
    return np.zeros_like(positron_energies)


def dnde_positron_pi_pi_pi0(
    self, positron_energies, cme: float, *, npts: int = 1 << 14, nbins: int = 25
):
    if cme < 2 * mpi + mpi0:
        return np.zeros_like(positron_energies)

    ff: FormFactorPiPiPi0 = self._ff_pi_pi_pi0

    dnde_decays = [
        spectra.dnde_positron_charged_pion,
        spectra.dnde_positron_charged_pion,
        _dnde_positron_zero,
    ]
    dists = ff.energy_distributions(
        cme=cme,
        gvuu=self.gvuu,
        gvdd=self.gvdd,
        gvss=self.gvss,
        nbins=nbins,
        npts=npts,
    )
    dnde = make_spectrum_n_body_decay(positron_energies, dists, dnde_decays)

    return dnde


def dnde_positron_pi_pi_eta(self, positron_energies, cme: float, *, nbins: int = 25):
    if cme < 2 * mpi + meta:
        return np.zeros_like(positron_energies)

    ff: FormFactorPiPiEta = self._ff_pi_pi_eta

    dnde_decays = [
        spectra.dnde_positron_charged_pion,
        spectra.dnde_positron_charged_pion,
        spectra.dnde_positron_eta,
    ]
    dists = ff.energy_distributions(
        cme=cme, gvuu=self.gvuu, gvdd=self.gvdd, nbins=nbins
    )

    dec = make_spectrum_n_body_decay(positron_energies, dists, dnde_decays)

    return dec


def dnde_positron_pi_pi_etap(self, positron_energies, cme: float, *, nbins: int = 25):
    if cme < 2 * mpi + metap:
        return np.zeros_like(positron_energies)

    ff: FormFactorPiPiEtaP = self._ff_pi_pi_etap

    dnde_decays = [
        spectra.dnde_positron_charged_pion,
        spectra.dnde_positron_charged_pion,
        spectra.dnde_positron_eta_prime,
    ]
    dists = ff.energy_distributions(
        cme=cme, gvuu=self.gvuu, gvdd=self.gvdd, nbins=nbins
    )

    dec = make_spectrum_n_body_decay(positron_energies, dists, dnde_decays)

    return dec


def dnde_positron_pi_pi_omega(self, positron_energies, cme: float, *, nbins=25):
    if cme < 2 * mpi + momega:
        return np.zeros_like(positron_energies)

    ff: FormFactorPiPiOmega = self._ff_pi_pi_omega

    dnde_decays = [
        spectra.dnde_positron_charged_pion,
        spectra.dnde_positron_charged_pion,
        spectra.dnde_positron_omega,
    ]
    dists = ff.energy_distributions(
        cme=cme,
        imode=1,
        nbins=nbins,
    )

    dec = make_spectrum_n_body_decay(positron_energies, dists, dnde_decays)

    return dec


def dnde_positron_pi0_pi0_omega(self, positron_energies, cme: float, *, nbins=25):
    if cme < 2 * mpi0 + momega:
        return np.zeros_like(positron_energies)

    ff: FormFactorPiPiOmega = self._ff_pi_pi_omega

    dnde_decays = [
        _dnde_positron_zero,
        _dnde_positron_zero,
        spectra.dnde_positron_omega,
    ]
    dists = ff.energy_distributions(
        cme=cme,
        imode=0,
        nbins=nbins,
    )

    dec = make_spectrum_n_body_decay(positron_energies, dists, dnde_decays)

    return dec


def dnde_positron_pi0_k0_k0(
    self, positron_energies, cme: float, *, npts: int = 1 << 14, nbins=25
):
    if cme < mpi0 + 2 * mk0:
        return np.zeros_like(positron_energies)

    ff: FormFactorPi0K0K0 = self._ff_pi0_k0_k0

    dnde_decays = [
        _dnde_positron_zero,
        spectra.dnde_positron_long_kaon,
        spectra.dnde_positron_short_kaon,
    ]
    dists = ff.energy_distributions(
        m=cme,
        gvuu=self.gvuu,
        gvdd=self.gvdd,
        gvss=self.gvss,
        npts=npts,
        nbins=nbins,
    )

    dec = make_spectrum_n_body_decay(positron_energies, dists, dnde_decays)

    return dec


def dnde_positron_pi0_k_k(
    self, positron_energies, cme: float, *, npts: int = 1 << 14, nbins=25
):
    if cme < mpi0 + 2 * mk:
        return np.zeros_like(positron_energies)

    ff: FormFactorPi0KpKm = self._ff_pi0_k_k

    dnde_decays = [
        _dnde_positron_zero,
        spectra.dnde_positron_charged_kaon,
        spectra.dnde_positron_charged_kaon,
    ]
    dists = ff.energy_distributions(
        m=cme,
        gvuu=self.gvuu,
        gvdd=self.gvdd,
        gvss=self.gvss,
        npts=npts,
        nbins=nbins,
    )

    dec = make_spectrum_n_body_decay(positron_energies, dists, dnde_decays)

    return dec


def dnde_positron_pi_k_k0(
    self, positron_energies, cme: float, *, npts: int = 1 << 14, nbins=25
):
    if cme < mpi + mk + mk0:
        return np.zeros_like(positron_energies)

    ff: FormFactorPiKK0 = self._ff_pi_k_k0

    dnde_decays = [
        _dnde_positron_zero,
        spectra.dnde_positron_charged_kaon,
        spectra.dnde_positron_charged_kaon,
    ]
    dists = ff.energy_distributions(
        m=cme,
        gvuu=self.gvuu,
        gvdd=self.gvdd,
        gvss=self.gvss,
        npts=npts,
        nbins=nbins,
    )

    dec = make_spectrum_n_body_decay(positron_energies, dists, dnde_decays)

    return dec


# =============================================================================
# ---- 4-body  ----------------------------------------------------------------
# =============================================================================


def dnde_positron_pi_pi_pi_pi(
    self, positron_energies, cme: float, *, npts=1 << 14, nbins=25
):
    if cme < 4 * mpi:
        return np.zeros_like(positron_energies)

    ff: FormFactorPiPiPiPi = self._ff_four_pi

    dnde_decays = [
        spectra.dnde_positron_charged_kaon,
        spectra.dnde_positron_charged_kaon,
        spectra.dnde_positron_charged_pion,
        spectra.dnde_positron_charged_pion,
    ]
    dists = ff.energy_distributions(
        cme=cme,
        gvuu=self.gvuu,
        gvdd=self.gvdd,
        neutral=True,
        npts=npts,
        nbins=nbins,
    )

    dnde = make_spectrum_n_body_decay(positron_energies, dists, dnde_decays)

    return dnde


def dnde_positron_pi_pi_pi0_pi0(
    self, positron_energies, cme: float, *, npts: int = 1 << 15, nbins: int = 30
):
    if cme < 2 * mpi + 2 * mpi0:
        return np.zeros_like(positron_energies)

    ff: FormFactorPiPiPiPi = self._ff_four_pi

    dnde_decays = [
        spectra.dnde_positron_charged_kaon,
        spectra.dnde_positron_charged_kaon,
        _dnde_positron_zero,
        _dnde_positron_zero,
    ]
    dists = ff.energy_distributions(
        cme=cme,
        gvuu=self.gvuu,
        gvdd=self.gvdd,
        neutral=True,
        npts=npts,
        nbins=nbins,
    )

    dnde = make_spectrum_n_body_decay(positron_energies, dists, dnde_decays)

    return dnde
