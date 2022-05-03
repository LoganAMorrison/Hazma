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
from hazma.spectra.altarelli_parisi import dnde_photon_ap_fermion, dnde_photon_ap_scalar
from hazma.spectra.boost import boost_delta_function, double_boost_delta_function
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


def make_spectrum_n_body_decay(photon_energies, energy_distributions, dnde_decays):
    dnde = np.zeros_like(photon_energies)

    for i, (probs, bins) in enumerate(energy_distributions):
        dec = np.array([dnde_decays[i](photon_energies, e) for e in bins])
        dnde += np.trapz(np.expand_dims(probs, 1) * dec, x=bins, axis=0)

    return dnde


def _dnde_photon_two_body(photon_energies, *, cme, m1, m2, dnde1, dnde2):
    if cme < m1 + m2:
        return np.zeros_like(photon_energies)
    e1 = (cme**2 + m1**2 - m2**2) / (2.0 * cme)
    e2 = (cme**2 - m1**2 + m2**2) / (2.0 * cme)
    return dnde1(photon_energies, e1) + dnde2(photon_energies, e2)


# =============================================================================
# ---- 2-body Lepton ----------------------------------------------------------
# =============================================================================


def dnde_photon_e_e(self, photon_energies, cme: float):
    if cme < 2 * me:
        return np.zeros_like(photon_energies)

    return dnde_photon_ap_fermion(photon_energies, cme**2, me, charge=-1.0)


def dnde_photon_mu_mu(self, photon_energies, cme: float):
    if cme < 2 * mmu:
        return np.zeros_like(photon_energies)

    fsr = dnde_photon_ap_fermion(photon_energies, cme**2, mmu, charge=-1.0)
    dec = 2 * spectra.dnde_photon_muon(photon_energies, cme / 2.0)
    return fsr + dec


# =============================================================================
# ---- 2-body Meson-Meson -----------------------------------------------------
# =============================================================================


def dnde_photon_pi_pi(self, photon_energies, cme: float):
    if cme < 2 * mpi:
        return np.zeros_like(photon_energies)

    fsr = dnde_photon_ap_scalar(photon_energies, cme**2, mpi, charge=-1.0)
    dec = 2 * spectra.dnde_photon_charged_pion(photon_energies, cme / 2.0)
    return fsr + dec


def dnde_photon_k0_k0(self, photon_energies, cme: float):
    if cme < 2 * mk0:
        return np.zeros_like(photon_energies)

    dec_l = spectra.dnde_photon_long_kaon(photon_energies, cme / 2.0)
    dec_s = spectra.dnde_photon_short_kaon(photon_energies, cme / 2.0)
    return dec_l + dec_s


def dnde_photon_k_k(self, photon_energies, cme: float):
    if cme < 2 * mk:
        return np.zeros_like(photon_energies)

    fsr = dnde_photon_ap_scalar(photon_energies, cme**2, mk, charge=-1.0)
    dec = spectra.dnde_photon_charged_kaon(photon_energies, cme / 2.0)
    return fsr + dec


# =============================================================================
# ---- 2-body Meson-Photon ----------------------------------------------------
# =============================================================================


def _dnde_photon_m_gamma(photon_energies, cme, mass, dnde):
    if cme < mass:
        return np.zeros_like(photon_energies)
    eng = (cme**2 + mass**2) / (2.0 * cme)
    return dnde(photon_energies, eng)


def dnde_photon_pi0_gamma(self, photon_energies, cme: float):
    return _dnde_photon_m_gamma(
        photon_energies, cme, mpi0, spectra.dnde_photon_neutral_pion
    )


def dnde_photon_eta_gamma(self, photon_energies, cme: float):
    return _dnde_photon_m_gamma(photon_energies, cme, meta, spectra.dnde_photon_eta)


# =============================================================================
# ---- 2-body Meson-Phi -------------------------------------------------------
# =============================================================================


def _dnde_photon_m_phi(photon_energies, *, cme, mass, dnde):
    if cme < mass + mphi:
        return np.zeros_like(photon_energies)
    dnde_phi = spectra.dnde_photon_phi
    return _dnde_photon_two_body(
        photon_energies, cme=cme, m1=mass, m2=mphi, dnde1=dnde, dnde2=dnde_phi
    )


def dnde_photon_pi0_phi(self, photon_energies, cme: float):
    return _dnde_photon_m_phi(
        photon_energies, cme=cme, mass=mpi0, dnde=spectra.dnde_photon_neutral_pion
    )


def dnde_photon_eta_phi(self, photon_energies, cme: float):
    return _dnde_photon_m_phi(
        photon_energies, cme=cme, mass=meta, dnde=spectra.dnde_photon_eta
    )


# =============================================================================
# ---- 2-body Meson-Omega -----------------------------------------------------
# =============================================================================


def dnde_photon_eta_omega(self, photon_energies, cme: float):
    if cme < meta + momega:
        return np.zeros_like(photon_energies)
    dnde1 = spectra.dnde_photon_eta
    dnde2 = spectra.dnde_photon_omega
    return _dnde_photon_two_body(
        photon_energies, cme=cme, m1=meta, m2=momega, dnde1=dnde1, dnde2=dnde2
    )


# =============================================================================
# ---- 3-body -----------------------------------------------------------------
# =============================================================================


def dnde_photon_pi0_pi0_gamma(self, photon_energies, cme: float):
    if cme < mpi0 + momega:
        return np.zeros_like(photon_energies)
    # This final state comes from V -> pi0 + omega with the omega decaying into
    # pi0 + gamma.
    epi1 = (cme**2 + mpi0**2 - momega**2) / (2.0 * cme)
    eomega = (cme**2 - mpi0**2 + momega**2) / (2.0 * cme)

    dnde = spectra.dnde_photon_neutral_pion(photon_energies, epi1)

    # Contributions from omega decay into pi0 + gamma:
    br_omega_to_pi0_a = 8.34e-2

    # Factors to boost from omega rest-frame to V rest-frame.
    gamma2 = eomega / momega
    beta2 = np.sqrt(1.0 - gamma2**-2)
    # Factors to boost from pi0 rest-frame to omega rest-frame.
    gamma1 = (momega**2 + mpi0**2) / (2.0 * momega * mpi0)
    beta1 = np.sqrt(1.0 - gamma1**-2)

    # Final state photon from omega decay: In the omega rest-frame, the
    # spectrum is a delta-function. We perform one boost to bring into the V
    # rest-frame.
    pre = br_omega_to_pi0_a
    e0 = (momega**2 - mpi0**2) / (2 * momega)
    dnde += pre * boost_delta_function(photon_energies, e0=e0, m=0.0, beta=beta2)

    # Double-boosted neutral pion: In the pion rest-frame, the spectra is a
    # delta-function. We then boost this into the omega rest-frame, then again
    # into the V rest-frame.
    br_pi0_to_a_a = 98.823e-2
    pre = 2.0 * br_omega_to_pi0_a * br_pi0_to_a_a
    e0 = mpi0 / 2.0
    dnde += pre * double_boost_delta_function(
        photon_energies, e0=e0, m=0.0, beta1=beta1, beta2=beta2
    )

    return dnde


def dnde_photon_pi_pi_pi0(
    self, photon_energies, cme: float, *, npts: int = 1 << 14, nbins: int = 25
):
    if cme < 2 * mpi + mpi0:
        return np.zeros_like(photon_energies)

    ff: FormFactorPiPiPi0 = self._ff_pi_pi_pi0

    dnde_decays = [
        spectra.dnde_photon_charged_pion,
        spectra.dnde_photon_charged_pion,
        spectra.dnde_photon_neutral_pion,
    ]
    dists = ff.energy_distributions(
        cme=cme,
        gvuu=self.gvuu,
        gvdd=self.gvdd,
        gvss=self.gvss,
        nbins=nbins,
        npts=npts,
    )
    dnde = make_spectrum_n_body_decay(photon_energies, dists, dnde_decays)

    inv_mass_dist = ff.invariant_mass_distributions(
        cme=cme,
        gvuu=self.gvuu,
        gvdd=self.gvdd,
        gvss=self.gvss,
        nbins=nbins,
        pairs=[(0, 1)],
    )

    ps, ms = inv_mass_dist[(0, 1)]
    fsr = np.array(
        [2.0 * dnde_photon_ap_scalar(photon_energies, m**2, mpi) for m in ms]
    )
    dnde += np.trapz(np.expand_dims(ps, 1) * fsr, x=ms, axis=0)

    return dnde


def dnde_photon_pi_pi_eta(self, photon_energies, cme: float, *, nbins: int = 25):
    if cme < 2 * mpi + meta:
        return np.zeros_like(photon_energies)

    ff: FormFactorPiPiEta = self._ff_pi_pi_eta

    dnde_decays = [
        spectra.dnde_photon_charged_pion,
        spectra.dnde_photon_charged_pion,
        spectra.dnde_photon_eta,
    ]
    dists = ff.energy_distributions(
        cme=cme, gvuu=self.gvuu, gvdd=self.gvdd, nbins=nbins
    )

    dec = make_spectrum_n_body_decay(photon_energies, dists, dnde_decays)

    return dec


def dnde_photon_pi_pi_etap(self, photon_energies, cme: float, *, nbins: int = 25):
    if cme < 2 * mpi + metap:
        return np.zeros_like(photon_energies)

    ff: FormFactorPiPiEtaP = self._ff_pi_pi_etap

    dnde_decays = [
        spectra.dnde_photon_charged_pion,
        spectra.dnde_photon_charged_pion,
        spectra.dnde_photon_eta_prime,
    ]
    dists = ff.energy_distributions(
        cme=cme, gvuu=self.gvuu, gvdd=self.gvdd, nbins=nbins
    )

    dec = make_spectrum_n_body_decay(photon_energies, dists, dnde_decays)

    return dec


def dnde_photon_pi_pi_omega(self, photon_energies, cme: float, *, nbins=25):
    if cme < 2 * mpi + momega:
        return np.zeros_like(photon_energies)

    ff: FormFactorPiPiOmega = self._ff_pi_pi_omega

    dnde_decays = [
        spectra.dnde_photon_charged_pion,
        spectra.dnde_photon_charged_pion,
        spectra.dnde_photon_omega,
    ]
    dists = ff.energy_distributions(
        cme=cme,
        imode=1,
        nbins=nbins,
    )

    dec = make_spectrum_n_body_decay(photon_energies, dists, dnde_decays)

    return dec


def dnde_photon_pi0_pi0_omega(self, photon_energies, cme: float, *, nbins=25):
    if cme < 2 * mpi0 + momega:
        return np.zeros_like(photon_energies)

    ff: FormFactorPiPiOmega = self._ff_pi_pi_omega

    dnde_decays = [
        spectra.dnde_photon_neutral_pion,
        spectra.dnde_photon_neutral_pion,
        spectra.dnde_photon_omega,
    ]
    dists = ff.energy_distributions(
        cme=cme,
        imode=0,
        nbins=nbins,
    )

    dec = make_spectrum_n_body_decay(photon_energies, dists, dnde_decays)

    return dec


def dnde_photon_pi0_k0_k0(
    self, photon_energies, cme: float, *, npts: int = 1 << 14, nbins=25
):
    if cme < mpi0 + 2 * mk0:
        return np.zeros_like(photon_energies)

    ff: FormFactorPi0K0K0 = self._ff_pi0_k0_k0

    dnde_decays = [
        spectra.dnde_photon_neutral_pion,
        spectra.dnde_photon_long_kaon,
        spectra.dnde_photon_short_kaon,
    ]
    dists = ff.energy_distributions(
        m=cme,
        gvuu=self.gvuu,
        gvdd=self.gvdd,
        gvss=self.gvss,
        npts=npts,
        nbins=nbins,
    )

    dec = make_spectrum_n_body_decay(photon_energies, dists, dnde_decays)

    return dec


def dnde_photon_pi0_k_k(
    self, photon_energies, cme: float, *, npts: int = 1 << 14, nbins=25
):
    if cme < mpi0 + 2 * mk:
        return np.zeros_like(photon_energies)

    ff: FormFactorPi0KpKm = self._ff_pi0_k_k

    dnde_decays = [
        spectra.dnde_photon_neutral_pion,
        spectra.dnde_photon_charged_kaon,
        spectra.dnde_photon_charged_kaon,
    ]
    dists = ff.energy_distributions(
        m=cme,
        gvuu=self.gvuu,
        gvdd=self.gvdd,
        gvss=self.gvss,
        npts=npts,
        nbins=nbins,
    )

    dec = make_spectrum_n_body_decay(photon_energies, dists, dnde_decays)

    return dec


def dnde_photon_pi_k_k0(
    self, photon_energies, cme: float, *, npts: int = 1 << 14, nbins=25
):
    if cme < mpi + mk + mk0:
        return np.zeros_like(photon_energies)

    ff: FormFactorPiKK0 = self._ff_pi_k_k0

    dnde_decays = [
        spectra.dnde_photon_neutral_pion,
        spectra.dnde_photon_charged_kaon,
        spectra.dnde_photon_charged_kaon,
    ]
    dists = ff.energy_distributions(
        m=cme,
        gvuu=self.gvuu,
        gvdd=self.gvdd,
        gvss=self.gvss,
        npts=npts,
        nbins=nbins,
    )

    dec = make_spectrum_n_body_decay(photon_energies, dists, dnde_decays)

    return dec


# =============================================================================
# ---- 4-body  ----------------------------------------------------------------
# =============================================================================


def dnde_photon_pi_pi_pi_pi(
    self, photon_energies, cme: float, *, npts=1 << 14, nbins=25
):
    if cme < 4 * mpi:
        return np.zeros_like(photon_energies)

    ff: FormFactorPiPiPiPi = self._ff_four_pi

    dnde_decays = [
        spectra.dnde_photon_charged_kaon,
        spectra.dnde_photon_charged_kaon,
        spectra.dnde_photon_charged_pion,
        spectra.dnde_photon_charged_pion,
    ]
    dists = ff.energy_distributions(
        cme=cme,
        gvuu=self.gvuu,
        gvdd=self.gvdd,
        neutral=True,
        npts=npts,
        nbins=nbins,
    )

    dnde = make_spectrum_n_body_decay(photon_energies, dists, dnde_decays)

    pairs = [(0, 1), (1, 2), (2, 3)]
    inv_mass_dist = ff.invariant_mass_distributions(
        cme=cme,
        gvuu=self.gvuu,
        gvdd=self.gvdd,
        neutral=True,
        npts=npts,
        nbins=nbins,
        pairs=pairs,
    )
    for pair in pairs:
        ps, ms = inv_mass_dist[pair]
        fsr = np.array(
            [2.0 * dnde_photon_ap_scalar(photon_energies, m**2, mpi) for m in ms]
        )
        # 0.5 since each particle is counted twice
        dnde += 0.5 * np.trapz(np.expand_dims(ps, 1) * fsr, x=ms, axis=0)

    return dnde


def dnde_photon_pi_pi_pi0_pi0(
    self, photon_energies, cme: float, *, npts: int = 1 << 15, nbins: int = 30
):
    if cme < 2 * mpi + 2 * mpi0:
        return np.zeros_like(photon_energies)

    ff: FormFactorPiPiPiPi = self._ff_four_pi

    dnde_decays = [
        spectra.dnde_photon_charged_kaon,
        spectra.dnde_photon_charged_kaon,
        spectra.dnde_photon_neutral_pion,
        spectra.dnde_photon_neutral_pion,
    ]
    dists = ff.energy_distributions(
        cme=cme,
        gvuu=self.gvuu,
        gvdd=self.gvdd,
        neutral=True,
        npts=npts,
        nbins=nbins,
    )

    dnde = make_spectrum_n_body_decay(photon_energies, dists, dnde_decays)
    inv_mass_dist = ff.invariant_mass_distributions(
        cme=cme,
        gvuu=self.gvuu,
        gvdd=self.gvdd,
        neutral=True,
        npts=npts,
        nbins=nbins,
        pairs=[(0, 1)],
    )

    ps, ms = inv_mass_dist[(0, 1)]
    fsr = np.array([dnde_photon_ap_scalar(photon_energies, m**2, mpi) for m in ms])
    dnde += np.trapz(np.expand_dims(ps, 1) * fsr, x=ms, axis=0)

    return dnde
