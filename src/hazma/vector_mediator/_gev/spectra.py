from typing import Callable, Dict, Union
import functools

import numpy as np
import numpy.typing as npt

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

DndeFn = Callable[[Union[float, npt.NDArray[np.float64]], float], float]


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


def _dnde_photon_v_v_rest_frame(self, photon_energies, *, npts=1 << 15, nbins=30):

    if self.mv < 2 * me:
        return np.zeros_like(photon_energies)

    pws = self.partial_widths()
    pw = sum(pws.values())
    pws = {key: val / pw for key, val in pws.items()}

    kwargs = {"npts": npts, "nbins": nbins}
    kwargs2 = {"nbins": nbins}

    pw_ee = pws["e e"]
    pw_mm = pws["mu mu"]
    pw_pp = pws["pi pi"]
    pw_k0k0 = pws["k0 k0"]
    pw_kk = pws["k k"]
    pw_0a = pws["pi0 gamma"]
    pw_na = pws["eta gamma"]
    pw_0f = pws["pi0 phi"]
    pw_nf = pws["eta phi"]
    pw_nw = pws["eta omega"]
    pw_00a = pws["pi0 pi0 gamma"]
    pw_pp0 = pws["pi pi pi0"]
    pw_ppn = pws["pi pi eta"]
    pw_ppnp = pws["pi pi etap"]
    pw_ppw = pws["pi pi omega"]
    pw_00w = pws["pi0 pi0 omega"]
    pw_0k0k0 = pws["pi0 k0 k0"]
    pw_0kk = pws["pi0 k k"]
    pw_pkk0 = pws["pi k k0"]
    pw_pppp = pws["pi pi pi pi"]
    pw_pp00 = pws["pi pi pi0 pi0"]

    # Factor of 2 for 2 vectors
    args = (photon_energies, self.mv)
    return 2 * (
        pw_ee * dnde_photon_e_e(self, *args)
        + pw_mm * dnde_photon_mu_mu(self, *args)
        + pw_pp * dnde_photon_pi_pi(self, *args)
        + pw_k0k0 * dnde_photon_k0_k0(self, *args)
        + pw_kk * dnde_photon_k_k(self, *args)
        + pw_0a * dnde_photon_pi0_gamma(self, *args)
        + pw_na * dnde_photon_eta_gamma(self, *args)
        + pw_0f * dnde_photon_pi0_phi(self, *args)
        + pw_nf * dnde_photon_eta_phi(self, *args)
        + pw_nw * dnde_photon_eta_omega(self, *args)
        + pw_00a * dnde_photon_pi0_pi0_gamma(self, *args)
        + pw_pp0 * dnde_photon_pi_pi_pi0(self, *args, **kwargs)
        + pw_ppn * dnde_photon_pi_pi_eta(self, *args, **kwargs2)
        + pw_ppnp * dnde_photon_pi_pi_etap(self, *args, **kwargs2)
        + pw_ppw * dnde_photon_pi_pi_omega(self, *args, **kwargs2)
        + pw_00w * dnde_photon_pi0_pi0_omega(self, *args, **kwargs2)
        + pw_0k0k0 * dnde_photon_pi0_k0_k0(self, *args, **kwargs)
        + pw_0kk * dnde_photon_pi0_k_k(self, *args, **kwargs)
        + pw_pkk0 * dnde_photon_pi_k_k0(self, *args, **kwargs)
        + pw_pppp * dnde_photon_pi_pi_pi_pi(self, *args, **kwargs)
        + pw_pp00 * dnde_photon_pi_pi_pi0_pi0(self, *args, **kwargs)
    )


def dnde_photon_v_v(
    self, photon_energies, cme, *, method="simpson", npts=1 << 15, nbins=30
):
    ev = 0.5 * cme
    mv = self.mv
    gamma = ev / mv
    if gamma < 1.0:
        return np.zeros_like(photon_energies)

    beta = np.sqrt(1.0 - gamma**-2)

    # def dnde(es):
    #     return _dnde_photon_v_v_rest_frame(self, es, npts=npts, nbins=nbins)

    # boosted = boost.make_boost_function(dnde)(
    #     photon_energies, beta=beta, mass=0.0, method=method
    # )
    dnde = _dnde_photon_v_v_rest_frame(self, photon_energies, npts=npts, nbins=nbins)
    boosted = boost.dnde_boost_array(dnde, photon_energies, beta)

    # Factor of 2 for 2 vectors
    e0pi = 0.5 * (mv - mpi**2 / mv)
    boosted += 2 * boost_delta_function(photon_energies, e0pi, 0.0, beta)
    e0eta = 0.5 * (mv - meta**2 / mv)
    boosted += 2 * boost_delta_function(photon_energies, e0eta, 0.0, beta)

    return boosted


def dnde_photon_spectrum_fns(
    self,
) -> Dict[str, Callable[[Union[float, npt.NDArray[np.float64]], float], float]]:
    def dnde_zero(e, _: float):
        return np.zeros_like(e)

    def wrap(f):
        @functools.wraps(f)
        def fnew(*args, **kwargs):
            return f(self, *args, **kwargs)

        return fnew

    return {
        "e e": wrap(dnde_photon_e_e),
        "mu mu": wrap(dnde_photon_mu_mu),
        "ve ve": dnde_zero,
        "vt vt": dnde_zero,
        "vm vm": dnde_zero,
        "pi pi": wrap(dnde_photon_pi_pi),
        "k0 k0": wrap(dnde_photon_k0_k0),
        "k k": wrap(dnde_photon_k_k),
        "pi0 gamma": wrap(dnde_photon_pi0_gamma),
        "eta gamma": wrap(dnde_photon_eta_gamma),
        "pi0 phi": wrap(dnde_photon_pi0_phi),
        "eta phi": wrap(dnde_photon_eta_phi),
        "eta omega": wrap(dnde_photon_eta_omega),
        "pi0 pi0 gamma": wrap(dnde_photon_pi0_pi0_gamma),
        "pi pi pi0": wrap(dnde_photon_pi_pi_pi0),
        "pi pi eta": wrap(dnde_photon_pi_pi_eta),
        "pi pi etap": wrap(dnde_photon_pi_pi_etap),
        "pi pi omega": wrap(dnde_photon_pi_pi_omega),
        "pi0 pi0 omega": wrap(dnde_photon_pi0_pi0_omega),
        "pi0 k0 k0": wrap(dnde_photon_pi0_k0_k0),
        "pi0 k k": wrap(dnde_photon_pi0_k_k),
        "pi k k0": wrap(dnde_photon_pi_k_k0),
        "pi pi pi pi": wrap(dnde_photon_pi_pi_pi_pi),
        "pi pi pi0 pi0": wrap(dnde_photon_pi_pi_pi0_pi0),
        "v v": wrap(dnde_photon_v_v),
    }
