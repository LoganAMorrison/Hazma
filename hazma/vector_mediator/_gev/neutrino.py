import functools
from typing import Optional

import numpy as np

from hazma import spectra
from hazma.spectra import boost
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


def make_zeros(neutrino_energies, flavor: Optional[str] = None):
    dtype = neutrino_energies.dtype
    shape = neutrino_energies.shape
    if flavor is None:
        return np.zeros((3,) + shape, dtype=dtype)
    return np.zeros(shape, dtype=dtype)


def make_spectrum_n_body_decay(
    neutrino_energies, energy_distributions, dnde_decays, flavor: Optional[str] = None
):
    dnde = np.zeros_like(neutrino_energies)

    for i, (probs, bins) in enumerate(energy_distributions):
        dec = np.array([dnde_decays[i](neutrino_energies, e, flavor) for e in bins])
        dnde += np.trapz(np.expand_dims(probs, 1) * dec, x=bins, axis=0)

    return dnde


def _dnde_neutrino_two_body(
    neutrino_energies, *, cme, m1, m2, dnde1, dnde2, flavor: Optional[str] = None
):
    if cme < m1 + m2:
        return make_zeros(neutrino_energies, flavor)

    e1 = (cme**2 + m1**2 - m2**2) / (2.0 * cme)
    e2 = (cme**2 - m1**2 + m2**2) / (2.0 * cme)
    return dnde1(neutrino_energies, e1, flavor) + dnde2(neutrino_energies, e2, flavor)


def _dnde_neutrino_zero(neutrino_energies, *_):
    return make_zeros(neutrino_energies, flavor)


# =============================================================================
# ---- 2-body Lepton ----------------------------------------------------------
# =============================================================================


def dnde_neutrino_e_e(self, neutrino_energies, cme: float):
    return make_zeros(neutrino_energies, flavor)


def dnde_neutrino_mu_mu(
    self, neutrino_energies, cme: float, flavor: Optional[str] = None
):
    if cme < 2 * mmu:
        return make_zeros(neutrino_energies, flavor)

    dec = 2 * spectra.dnde_neutrino_muon(neutrino_energies, cme / 2.0, flavor)
    return dec


# =============================================================================
# ---- 2-body Meson-Meson -----------------------------------------------------
# =============================================================================


def dnde_neutrino_pi_pi(
    self, neutrino_energies, cme: float, flavor: Optional[str] = None
):
    if cme < 2 * mpi:
        return make_zeros(neutrino_energies, flavor)

    dec = 2 * spectra.dnde_neutrino_charged_pion(neutrino_energies, cme / 2.0)
    return dec


def dnde_neutrino_k0_k0(
    self, neutrino_energies, cme: float, flavor: Optional[str] = None
):
    if cme < 2 * mk0:
        return make_zeros(neutrino_energies, flavor)

    dec_l = spectra.dnde_neutrino_long_kaon(neutrino_energies, cme / 2.0)
    dec_s = spectra.dnde_neutrino_short_kaon(neutrino_energies, cme / 2.0)
    return dec_l + dec_s


def dnde_neutrino_k_k(
    self, neutrino_energies, cme: float, flavor: Optional[str] = None
):
    if cme < 2 * mk:
        return make_zeros(neutrino_energies, flavor)

    dec = spectra.dnde_neutrino_charged_kaon(neutrino_energies, cme / 2.0)
    return dec


# =============================================================================
# ---- 2-body Meson-Photon ----------------------------------------------------
# =============================================================================


def _dnde_neutrino_m_gamma(
    neutrino_energies, cme, mass, dnde, flavor: Optional[str] = None
):
    if cme < mass:
        return make_zeros(neutrino_energies, flavor)
    eng = (cme**2 + mass**2) / (2.0 * cme)
    return dnde(neutrino_energies, eng)


def dnde_neutrino_pi0_gamma(self, neutrino_energies, cme: float):
    return make_zeros(neutrino_energies, flavor)


def dnde_neutrino_eta_gamma(
    self, neutrino_energies, cme: float, flavor: Optional[str] = None
):
    return _dnde_neutrino_m_gamma(
        neutrino_energies, cme, meta, spectra.dnde_neutrino_eta
    )


# =============================================================================
# ---- 2-body Meson-Phi -------------------------------------------------------
# =============================================================================


def _dnde_neutrino_m_phi(
    neutrino_energies, *, cme, mass, dnde, flavor: Optional[str] = None
):
    if cme < mass + mphi:
        return make_zeros(neutrino_energies, flavor)
    dnde_phi = spectra.dnde_neutrino_phi
    return _dnde_neutrino_two_body(
        neutrino_energies, cme=cme, m1=mass, m2=mphi, dnde1=dnde, dnde2=dnde_phi
    )


def dnde_neutrino_pi0_phi(
    self, neutrino_energies, cme: float, flavor: Optional[str] = None
):
    return _dnde_neutrino_m_phi(
        neutrino_energies, cme=cme, mass=mpi0, dnde=_dnde_neutrino_zero
    )


def dnde_neutrino_eta_phi(
    self, neutrino_energies, cme: float, flavor: Optional[str] = None
):
    return _dnde_neutrino_m_phi(
        neutrino_energies, cme=cme, mass=meta, dnde=spectra.dnde_neutrino_eta
    )


# =============================================================================
# ---- 2-body Meson-Omega -----------------------------------------------------
# =============================================================================


def dnde_neutrino_eta_omega(
    self, neutrino_energies, cme: float, flavor: Optional[str] = None
):
    if cme < meta + momega:
        return make_zeros(neutrino_energies, flavor)
    dnde1 = spectra.dnde_neutrino_eta
    dnde2 = spectra.dnde_neutrino_omega
    return _dnde_neutrino_two_body(
        neutrino_energies, cme=cme, m1=meta, m2=momega, dnde1=dnde1, dnde2=dnde2
    )


# =============================================================================
# ---- 3-body -----------------------------------------------------------------
# =============================================================================


def dnde_neutrino_pi0_pi0_gamma(
    self, neutrino_energies, cme: float, flavor: Optional[str] = None
):
    return make_zeros(neutrino_energies, flavor)


def dnde_neutrino_pi_pi_pi0(
    self,
    neutrino_energies,
    cme: float,
    *,
    npts: int = 1 << 14,
    nbins: int = 25,
    flavor: Optional[str] = None
):
    if cme < 2 * mpi + mpi0:
        return make_zeros(neutrino_energies, flavor)

    ff: FormFactorPiPiPi0 = self._ff_pi_pi_pi0

    dnde_decays = [
        spectra.dnde_neutrino_charged_pion,
        spectra.dnde_neutrino_charged_pion,
        _dnde_neutrino_zero,
    ]
    dists = ff.energy_distributions(
        cme=cme,
        gvuu=self.gvuu,
        gvdd=self.gvdd,
        gvss=self.gvss,
        nbins=nbins,
        npts=npts,
    )
    dnde = make_spectrum_n_body_decay(neutrino_energies, dists, dnde_decays)

    return dnde


def dnde_neutrino_pi_pi_eta(
    self,
    neutrino_energies,
    cme: float,
    *,
    nbins: int = 25,
    flavor: Optional[str] = None
):
    if cme < 2 * mpi + meta:
        return make_zeros(neutrino_energies, flavor)

    ff: FormFactorPiPiEta = self._ff_pi_pi_eta

    dnde_decays = [
        spectra.dnde_neutrino_charged_pion,
        spectra.dnde_neutrino_charged_pion,
        spectra.dnde_neutrino_eta,
    ]
    dists = ff.energy_distributions(
        cme=cme, gvuu=self.gvuu, gvdd=self.gvdd, nbins=nbins
    )

    dec = make_spectrum_n_body_decay(neutrino_energies, dists, dnde_decays)

    return dec


def dnde_neutrino_pi_pi_etap(
    self,
    neutrino_energies,
    cme: float,
    *,
    nbins: int = 25,
    flavor: Optional[str] = None
):
    if cme < 2 * mpi + metap:
        return make_zeros(neutrino_energies, flavor)

    ff: FormFactorPiPiEtaP = self._ff_pi_pi_etap

    dnde_decays = [
        spectra.dnde_neutrino_charged_pion,
        spectra.dnde_neutrino_charged_pion,
        spectra.dnde_neutrino_eta_prime,
    ]
    dists = ff.energy_distributions(
        cme=cme, gvuu=self.gvuu, gvdd=self.gvdd, nbins=nbins
    )

    dec = make_spectrum_n_body_decay(neutrino_energies, dists, dnde_decays)

    return dec


def dnde_neutrino_pi_pi_omega(
    self, neutrino_energies, cme: float, *, nbins=25, flavor: Optional[str] = None
):
    if cme < 2 * mpi + momega:
        return make_zeros(neutrino_energies, flavor)

    ff: FormFactorPiPiOmega = self._ff_pi_pi_omega

    dnde_decays = [
        spectra.dnde_neutrino_charged_pion,
        spectra.dnde_neutrino_charged_pion,
        spectra.dnde_neutrino_omega,
    ]
    dists = ff.energy_distributions(
        cme=cme,
        imode=1,
        nbins=nbins,
    )

    dec = make_spectrum_n_body_decay(neutrino_energies, dists, dnde_decays)

    return dec


def dnde_neutrino_pi0_pi0_omega(
    self, neutrino_energies, cme: float, *, nbins=25, flavor: Optional[str] = None
):
    if cme < 2 * mpi0 + momega:
        return make_zeros(neutrino_energies, flavor)

    ff: FormFactorPiPiOmega = self._ff_pi_pi_omega

    dnde_decays = [
        _dnde_neutrino_zero,
        _dnde_neutrino_zero,
        spectra.dnde_neutrino_omega,
    ]
    dists = ff.energy_distributions(
        cme=cme,
        imode=0,
        nbins=nbins,
    )

    dec = make_spectrum_n_body_decay(neutrino_energies, dists, dnde_decays)

    return dec


def dnde_neutrino_pi0_k0_k0(
    self,
    neutrino_energies,
    cme: float,
    *,
    npts: int = 1 << 14,
    nbins=25,
    flavor: Optional[str] = None
):
    if cme < mpi0 + 2 * mk0:
        return make_zeros(neutrino_energies, flavor)

    ff: FormFactorPi0K0K0 = self._ff_pi0_k0_k0

    dnde_decays = [
        _dnde_neutrino_zero,
        spectra.dnde_neutrino_long_kaon,
        spectra.dnde_neutrino_short_kaon,
    ]
    dists = ff.energy_distributions(
        m=cme,
        gvuu=self.gvuu,
        gvdd=self.gvdd,
        gvss=self.gvss,
        npts=npts,
        nbins=nbins,
    )

    dec = make_spectrum_n_body_decay(neutrino_energies, dists, dnde_decays)

    return dec


def dnde_neutrino_pi0_k_k(
    self,
    neutrino_energies,
    cme: float,
    *,
    npts: int = 1 << 14,
    nbins=25,
    flavor: Optional[str] = None
):
    if cme < mpi0 + 2 * mk:
        return make_zeros(neutrino_energies, flavor)

    ff: FormFactorPi0KpKm = self._ff_pi0_k_k

    dnde_decays = [
        _dnde_neutrino_zero,
        spectra.dnde_neutrino_charged_kaon,
        spectra.dnde_neutrino_charged_kaon,
    ]
    dists = ff.energy_distributions(
        m=cme,
        gvuu=self.gvuu,
        gvdd=self.gvdd,
        gvss=self.gvss,
        npts=npts,
        nbins=nbins,
    )

    dec = make_spectrum_n_body_decay(neutrino_energies, dists, dnde_decays)

    return dec


def dnde_neutrino_pi_k_k0(
    self,
    neutrino_energies,
    cme: float,
    *,
    npts: int = 1 << 14,
    nbins=25,
    flavor: Optional[str] = None
):
    if cme < mpi + mk + mk0:
        return make_zeros(neutrino_energies, flavor)

    ff: FormFactorPiKK0 = self._ff_pi_k_k0

    dnde_decays = [
        _dnde_neutrino_zero,
        spectra.dnde_neutrino_charged_kaon,
        spectra.dnde_neutrino_charged_kaon,
    ]
    dists = ff.energy_distributions(
        m=cme,
        gvuu=self.gvuu,
        gvdd=self.gvdd,
        gvss=self.gvss,
        npts=npts,
        nbins=nbins,
    )

    dec = make_spectrum_n_body_decay(neutrino_energies, dists, dnde_decays)

    return dec


# =============================================================================
# ---- 4-body  ----------------------------------------------------------------
# =============================================================================


def dnde_neutrino_pi_pi_pi_pi(
    self,
    neutrino_energies,
    cme: float,
    *,
    npts=1 << 14,
    nbins=25,
    flavor: Optional[str] = None
):
    if cme < 4 * mpi:
        return make_zeros(neutrino_energies, flavor)

    ff: FormFactorPiPiPiPi = self._ff_four_pi

    dnde_decays = [
        spectra.dnde_neutrino_charged_kaon,
        spectra.dnde_neutrino_charged_kaon,
        spectra.dnde_neutrino_charged_pion,
        spectra.dnde_neutrino_charged_pion,
    ]
    dists = ff.energy_distributions(
        cme=cme,
        gvuu=self.gvuu,
        gvdd=self.gvdd,
        neutral=True,
        npts=npts,
        nbins=nbins,
    )

    dnde = make_spectrum_n_body_decay(neutrino_energies, dists, dnde_decays)

    return dnde


def dnde_neutrino_pi_pi_pi0_pi0(
    self,
    neutrino_energies,
    cme: float,
    *,
    npts: int = 1 << 15,
    nbins: int = 30,
    flavor: Optional[str] = None
):
    if cme < 2 * mpi + 2 * mpi0:
        return make_zeros(neutrino_energies, flavor)

    ff: FormFactorPiPiPiPi = self._ff_four_pi

    dnde_decays = [
        spectra.dnde_neutrino_charged_kaon,
        spectra.dnde_neutrino_charged_kaon,
        _dnde_neutrino_zero,
        _dnde_neutrino_zero,
    ]
    dists = ff.energy_distributions(
        cme=cme,
        gvuu=self.gvuu,
        gvdd=self.gvdd,
        neutral=True,
        npts=npts,
        nbins=nbins,
    )

    dnde = make_spectrum_n_body_decay(neutrino_energies, dists, dnde_decays)

    return dnde


def dnde_neutrino_v_v(
    self,
    neutrino_energies,
    cme,
    *,
    method="quad",
    npts=1 << 15,
    nbins=30,
    flavor: Optional[str] = None
):
    gamma = 2.0 * self.mv / cme
    if gamma < 1.0:
        return make_zeros(neutrino_energies, flavor)

    beta = np.sqrt(1.0 - gamma**-2)

    kwargs = {"npts": npts, "nbins": nbins}
    kwargs2 = {"nbins": nbins}

    def dnde(es):
        args = (es, cme)
        spec = np.zeros_like(es)
        spec += dnde_neutrino_e_e(self, *args)
        spec += dnde_neutrino_mu_mu(self, *args)
        spec += dnde_neutrino_pi_pi(self, *args)
        spec += dnde_neutrino_k0_k0(self, *args)
        spec += dnde_neutrino_k_k(self, *args)
        spec += dnde_neutrino_pi0_gamma(self, *args)
        spec += dnde_neutrino_eta_gamma(self, *args)
        spec += dnde_neutrino_pi0_phi(self, *args)
        spec += dnde_neutrino_eta_phi(self, *args)
        spec += dnde_neutrino_eta_omega(self, *args)
        spec += dnde_neutrino_pi0_pi0_gamma(self, *args)
        spec += dnde_neutrino_pi_pi_pi0(self, *args, **kwargs)
        spec += dnde_neutrino_pi_pi_eta(self, *args, **kwargs2)
        spec += dnde_neutrino_pi_pi_etap(self, *args, **kwargs2)
        spec += dnde_neutrino_pi_pi_omega(self, *args, **kwargs2)
        spec += dnde_neutrino_pi0_pi0_omega(self, *args, **kwargs2)
        spec += dnde_neutrino_pi0_k0_k0(self, *args, **kwargs)
        spec += dnde_neutrino_pi0_k_k(self, *args, **kwargs)
        spec += dnde_neutrino_pi_k_k0(self, *args, **kwargs)
        spec += dnde_neutrino_pi_pi_pi_pi(self, *args, **kwargs)
        spec += dnde_neutrino_pi_pi_pi0_pi0(self, *args, **kwargs)

        return spec

    return boost.make_boost_function(dnde)(
        neutrino_energies, beta, mass=me, method=method
    )


def dnde_neutrino_spectrum_fns(self):
    def dnde_zero(e, _: float, flavor: Optional[str] = None):
        return make_zeros(e, flavor)

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
