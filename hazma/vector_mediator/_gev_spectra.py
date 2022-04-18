import numpy as np

from hazma import spectra
from hazma.spectra.altarelli_parisi import dnde_photon_ap_fermion, dnde_photon_ap_scalar
from hazma.parameters import (
    neutral_pion_mass as mpi0,
    charged_pion_mass as mpi,
    muon_mass as mmu,
    electron_mass as me,
    charged_kaon_mass as mk,
    neutral_kaon_mass as mk0,
    eta_mass as meta,
    omega_mass as momega,
)


def dnde_photon_v_to_e_e(self, photon_energies):
    cme = self.mv
    if cme < 2 * me:
        return np.zeros_like(photon_energies)

    return dnde_photon_ap_fermion(photon_energies, cme**2, me, charge=-1.0)


def dnde_photon_v_to_mu_mu(self, photon_energies):
    cme = self.mv
    if cme < 2 * mmu:
        return np.zeros_like(photon_energies)

    fsr = dnde_photon_ap_fermion(photon_energies, cme**2, mmu, charge=-1.0)
    dec = spectra.dnde_photon_muon(photon_energies, cme / 2.0)
    return fsr + dec


def dnde_photon_v_to_pi_pi(self, photon_energies):
    cme = self.mv
    if cme < 2 * mpi:
        return np.zeros_like(photon_energies)

    fsr = dnde_photon_ap_scalar(photon_energies, cme**2, mpi, charge=-1.0)
    dec = spectra.dnde_photon_charged_pion(photon_energies, cme / 2.0)
    return fsr + dec


def dnde_photon_v_to_k0_k0(self, photon_energies):
    cme = self.mv
    if cme < 2 * mk0:
        return np.zeros_like(photon_energies)

    dec_l = spectra.dnde_photon_long_kaon(photon_energies, cme / 2.0)
    dec_s = spectra.dnde_photon_short_kaon(photon_energies, cme / 2.0)
    return dec_l + dec_s


def dnde_photon_v_to_k_k(self, photon_energies):
    cme = self.mv
    if cme < 2 * mk:
        return np.zeros_like(photon_energies)

    fsr = dnde_photon_ap_scalar(photon_energies, cme**2, mk, charge=-1.0)
    dec = spectra.dnde_photon_charged_kaon(photon_energies, cme / 2.0)
    return fsr + dec


def dnde_photon_v_to_pi0_gamma(self, photon_energies):
    cme = self.mv
    if cme < mpi0:
        return np.zeros_like(photon_energies)

    epi = (cme**2 + mpi0**2) / (2 * cme)
    return spectra.dnde_photon_neutral_pion(photon_energies, epi)


def dnde_photon_v_to_eta_gamma(self, photon_energies):
    cme = self.mv
    if cme < meta:
        return np.zeros_like(photon_energies)

    epi = (cme**2 + meta**2) / (2 * cme)
    return spectra.dnde_photon_eta(photon_energies, epi)


def dnde_photon_v_to_pi0_phi(self, photon_energies):
    pass


def dnde_photon_v_to_eta_phi(self, photon_energies):
    pass


def dnde_photon_v_to_eta_omega(self, photon_energies):
    cme = self.mv
    if cme < meta + momega:
        return np.zeros_like(photon_energies)

    ee = (cme**2 + meta**2 - momega**2) / (2 * cme)
    ew = (cme**2 - meta**2 + momega**2) / (2 * cme)
    dec_e = spectra.dnde_photon_eta(photon_energies, ee)
    dec_w = spectra.dnde_photon_omega(photon_energies, ew)
    return dec_e + dec_w


def dnde_photon_v_to_pi0_pi0_gamma(self, photon_energies):
    pass


def dnde_photon_v_to_pi_pi_pi0(self, photon_energies):
    pass


def dnde_photon_v_to_pi_pi_eta(self, photon_energies):
    pass


def dnde_photon_v_to_pi_pi_etap(self, photon_energies):
    pass


def dnde_photon_v_to_pi_pi_omega(self, photon_energies):
    pass


def dnde_photon_v_to_pi0_pi0_omega(self, photon_energies):
    pass


def dnde_photon_v_to_pi0_k0_k0(self, photon_energies):
    pass


def dnde_photon_v_to_pi0_k_k(self, photon_energies):
    pass


def dnde_photon_v_to_pi_k_k0(self, photon_energies):
    pass


def dnde_photon_v_to_pi_pi_pi_pi(self, photon_energies):
    pass


def dnde_photon_v_to_pi_pi_pi0_pi0(self, photon_energies):
    pass
