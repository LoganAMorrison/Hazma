from typing import Callable

import numpy as np

from hazma.theory import TheoryAnn, TheoryDec
from hazma.parameters import (
    neutral_pion_mass as m_pi0,
    charged_pion_mass as m_pi,
    alpha_em,
    electron_mass as m_e,
    muon_mass as m_mu,
)
from hazma.spectra import (
    dnde_photon_muon as dnde_g_mu,
    dnde_photon_neutral_pion as dnde_g_pi0,
    dnde_photon_charged_pion as dnde_g_pi,
)
from hazma.spectra import (
    dnde_positron_charged_pion as dnde_p_pi,
    dnde_positron_muon as dnde_p_mu,
)
from hazma.utils import RealOrRealArray


class SingleChannelAnn(TheoryAnn):
    def __init__(self, mx, fs, sigma):
        self._mx = mx
        self._fs = fs
        self.sigma = sigma

        self.fs_mass: float = 0.0
        self._spectrum_funcs: Callable[[], dict] = lambda: dict()
        self._gamma_ray_line_energies: Callable[[float], dict] = lambda _: dict()
        self._positron_spectrum_funcs: Callable[[], dict] = lambda: dict()
        self._positron_line_energies: Callable[[float], dict] = lambda _: dict()

        self.setup()

    def __repr__(self):
        args = [
            f"mx={self._mx} MeV",
            f"fs='{self._fs}'",
            f"sigma={self.sigma} MeV^-2",
        ]
        return "SingleChannelAnn(" + ", ".join(args) + ")"

    @property
    def fs(self):
        return self._fs

    @fs.setter
    def fs(self, fs):
        self._fs = fs
        self.setup()

    @property
    def mx(self):
        return self._mx

    @mx.setter
    def mx(self, mx):
        self._mx = mx
        self.setup()

    def annihilation_cross_section_funcs(self):
        def xsec(e_cm):
            if e_cm < 2 * self.mx or e_cm < self.fs_mass:
                return 0.0
            else:
                return self.sigma

        return {self.fs: xsec}

    def list_annihilation_final_states(self):
        return [self.fs]

    def setup(self):
        self.set_fs_mass()
        self.set_spectrum_funcs()
        self.set_gamma_ray_line_energies()
        self.set_positron_spectrum_funcs()
        self.set_positron_line_energies()

    def set_fs_mass(self):
        # Sets kinematic threshold for DM annihilations/decays
        if self.fs == "g g":
            self.fs_mass = 0.0
        elif self.fs == "e e":
            self.fs_mass = 2 * m_e
        elif self.fs == "mu mu":
            self.fs_mass = 2 * m_mu
        elif self.fs == "pi pi":
            self.fs_mass = 2 * m_pi
        elif self.fs == "pi0 pi0":
            self.fs_mass = 2 * m_pi0
        elif self.fs == "pi0 g":
            self.fs_mass = m_pi0

    def __dnde_photon_ee(self, eg: RealOrRealArray, ecm: float) -> RealOrRealArray:
        return self._dnde_ap_fermion(eg, ecm, m_e)

    def __dnde_photon_mumu(self, eg: RealOrRealArray, ecm: float) -> RealOrRealArray:
        return 2 * dnde_g_mu(eg, ecm / 2) + self._dnde_ap_fermion(eg, ecm, m_mu)

    def __dnde_photon_pi0pi0(self, eg: RealOrRealArray, ecm: float) -> RealOrRealArray:
        return 2 * dnde_g_pi0(eg, ecm / 2)

    def __dnde_photon_pi0g(self, eg: RealOrRealArray, ecm: float) -> RealOrRealArray:
        return dnde_g_pi0(eg, (ecm**2 + m_pi0**2) / (2.0 * ecm))

    def __dnde_photon_pipi(self, eg: RealOrRealArray, ecm: float) -> RealOrRealArray:
        return 2 * dnde_g_pi(eg, ecm / 2) + self._dnde_ap_scalar(eg, ecm, m_pi)

    def set_spectrum_funcs(self) -> None:
        """
        Sets gamma ray spectrum functions.
        """
        funcs = dict()

        if self.fs == "e e":
            funcs[self.fs] = lambda eg, ecm: self.__dnde_photon_ee(eg, ecm)
        elif self.fs == "mu mu":
            funcs[self.fs] = lambda eg, ecm: self.__dnde_photon_mumu(eg, ecm)
        elif self.fs == "pi0 pi0":
            funcs[self.fs] = lambda eg, ecm: self.__dnde_photon_pi0pi0(eg, ecm)
        elif self.fs == "pi0 g":
            funcs[self.fs] = lambda eg, ecm: self.__dnde_photon_pi0g(eg, ecm)
        elif self.fs == "pi pi":
            funcs[self.fs] = lambda eg, ecm: self.__dnde_photon_pipi(eg, ecm)
        else:
            funcs[self.fs] = lambda eg, ecm: np.zeros_like(eg)

        self._spectrum_funcs = lambda: funcs

    def set_gamma_ray_line_energies(self) -> None:
        if self.fs == "g g":
            self._gamma_ray_line_energies = lambda e_cm: {"g g": e_cm / 2}

        elif self.fs == "pi0 g":
            self._gamma_ray_line_energies = lambda e_cm: {
                "pi0 g": (e_cm**2 - m_pi0**2) / (2.0 * e_cm)
            }
        else:
            self._gamma_ray_line_energies = lambda _: {}

    def __dnde_positron_mumu(self, ep: RealOrRealArray, ecm: float) -> RealOrRealArray:
        if ecm < self.fs_mass:
            return 0.0
        return dnde_p_mu(ep, ecm / 2.0)

    def __dnde_positron_pipi(self, ep: RealOrRealArray, ecm: float) -> RealOrRealArray:
        if ecm < self.fs_mass:
            return 0.0
        return dnde_p_pi(ep, ecm / 2.0)

    def set_positron_spectrum_funcs(self):
        funcs = dict()

        if self.fs == "mu mu":
            funcs[self.fs] = lambda eg, ecm: self.__dnde_positron_mumu(eg, ecm)
        elif self.fs == "pi pi":
            funcs[self.fs] = lambda eg, ecm: self.__dnde_positron_pipi(eg, ecm)
        else:
            funcs[self.fs] = lambda eg, ecm: np.zeros_like(eg)

        self._positron_spectrum_funcs = lambda: funcs

    def set_positron_line_energies(self):
        if self.fs == "e e":
            self._positron_line_energies = lambda e_cm: {"e e": e_cm / 2.0}
        else:
            self._positron_line_energies = lambda _: {}

    def _dnde_ap_scalar(self, e_g, e_cm, m_scalar):
        def fn(e_g):
            mu = m_scalar / e_cm
            x = 2 * e_g / e_cm
            P_g_scalar = 2 * (1 - x) / x
            res = (
                2
                * alpha_em
                / (np.pi * e_cm)
                * P_g_scalar
                * (np.log((1 - x) / mu**2) - 1)
            )
            if not np.isnan(res) and res >= 0:
                return res
            else:
                return 0

        return np.vectorize(fn)(e_g)

    def _dnde_ap_fermion(self, e_g, e_cm, m_fermion):
        def fn(e_g):
            mu = m_fermion / e_cm
            x = 2 * e_g / e_cm
            P_g_fermion = (1 + (1 - x) ** 2) / x
            res = (
                2
                * alpha_em
                / (np.pi * e_cm)
                * P_g_fermion
                * (np.log((1 - x) / mu**2) - 1)
            )
            if not np.isnan(res) and res >= 0:
                return res
            else:
                return 0

        return np.vectorize(fn)(e_g)


class SingleChannelDec(TheoryDec):
    def __init__(self, mx, fs, width):
        self._mx = mx
        self._fs = fs
        self.width = width

        self._spectrum_funcs: Callable[[], dict] = lambda: dict()
        self._gamma_ray_line_energies: Callable[[], dict] = lambda: dict()
        self._positron_spectrum_funcs: Callable[[], dict] = lambda: dict()
        self._positron_line_energies: Callable[[], dict] = lambda: dict()

        self.setup()

    def __repr__(self):

        args = [
            f"mx={self._mx} MeV",
            f"fs='{self._fs}'",
            f"width={self.width} MeV",
        ]
        return "SingleChannelDec(" + ", ".join(args) + ")"

    @property
    def fs(self):
        return self._fs

    @fs.setter
    def fs(self, fs):
        self._fs = fs
        self.setup()

    @property
    def mx(self):
        return self._mx

    @mx.setter
    def mx(self, mx):
        self._mx = mx
        self.setup()

    def list_decay_final_states(self):
        return [self.fs]

    def _decay_widths(self):
        return {self.fs: self.width}

    def setup(self):
        self.set_fs_mass()
        self.set_spectrum_funcs()
        self.set_gamma_ray_line_energies()
        self.set_positron_spectrum_funcs()
        self.set_positron_line_energies()

    def set_fs_mass(self):
        # Sets kinematic threshold for DM annihilations/decays
        if self.fs == "g g":
            self.fs_mass = 0.0
        elif self.fs == "e e":
            self.fs_mass = 2 * m_e
        elif self.fs == "mu mu":
            self.fs_mass = 2 * m_mu
        elif self.fs == "pi pi":
            self.fs_mass = 2 * m_pi
        elif self.fs == "pi0 pi0":
            self.fs_mass = 2 * m_pi0
        elif self.fs == "pi0 g":
            self.fs_mass = m_pi0

    def __dnde_photon_ee(self, eg):
        return self._dnde_ap_fermion(eg, m_e)

    def __dnde_photon_mumu(self, eg):
        return 2 * dnde_g_mu(eg, self.mx / 2) + self._dnde_ap_fermion(eg, m_mu)

    def __dnde_photon_pi0pi0(self, eg):
        return 2 * dnde_g_pi0(eg, self.mx / 2)

    def __dnde_photon_pi0g(self, eg):
        return dnde_g_pi0(eg, (self.mx**2 + m_pi0**2) / (2.0 * self.mx))

    def __dnde_photon_pipi(self, eg):
        return 2 * dnde_g_pi(eg, self.mx / 2) + self._dnde_ap_scalar(eg, m_pi)

    def set_spectrum_funcs(self):
        """
        Sets gamma ray spectrum functions.
        """
        funcs = dict()

        if self.fs == "e e":
            funcs[self.fs] = lambda eg: self.__dnde_photon_ee(eg)
        elif self.fs == "mu mu":
            funcs[self.fs] = lambda eg: self.__dnde_photon_mumu(eg)
        elif self.fs == "pi0 pi0":
            funcs[self.fs] = lambda eg: self.__dnde_photon_pi0pi0(eg)
        elif self.fs == "pi0 g":
            funcs[self.fs] = lambda eg: self.__dnde_photon_pi0g(eg)
        elif self.fs == "pi pi":
            funcs[self.fs] = lambda eg: self.__dnde_photon_pipi(eg)

        self._spectrum_funcs = lambda: funcs

    def set_gamma_ray_line_energies(self):
        lines = dict()

        if self.fs == "g g":
            lines[self.fs] = self.mx / 2
        elif self.fs == "pi0 g":
            lines[self.fs] = (self.mx**2 - m_pi0**2) / (2.0 * self.mx)

        self._gamma_ray_line_energies = lambda: lines

    def __dnde_positron_mumu(self, ep):
        if self.mx < self.fs_mass:
            return 0.0
        return dnde_p_mu(ep, self.mx / 2.0)

    def __dnde_positron_pipi(self, ep):
        if self.mx < self.fs_mass:
            return 0.0
        return dnde_p_pi(ep, self.mx / 2.0)

    def set_positron_spectrum_funcs(self):
        funcs = dict()

        if self.fs == "mu mu":
            funcs[self.fs] = lambda ep: self.__dnde_positron_mumu(ep)
        elif self.fs == "pi pi":
            funcs[self.fs] = lambda ep: self.__dnde_positron_pipi(ep)

        self._positron_spectrum_funcs = lambda: funcs

    def set_positron_line_energies(self):
        lines = dict()

        if self.fs == "e e":
            lines[self.fs] = self.mx / 2.0

        self._positron_line_energies = lambda: lines

    def _dnde_ap_scalar(self, e_g, m_scalar):
        def fn(e_g):
            mu = m_scalar / self.mx
            x = 2 * e_g / self.mx
            P_g_scalar = 2 * (1 - x) / x
            res = (
                2
                * alpha_em
                / (np.pi * self.mx)
                * P_g_scalar
                * (np.log((1 - x) / mu**2) - 1)
            )
            if not np.isnan(res) and res >= 0:
                return res
            else:
                return 0

        return np.vectorize(fn)(e_g)

    def _dnde_ap_fermion(self, e_g, m_fermion):
        def fn(e_g):
            mu = m_fermion / self.mx
            x = 2 * e_g / self.mx
            P_g_fermion = (1 + (1 - x) ** 2) / x
            res = (
                2
                * alpha_em
                / (np.pi * self.mx)
                * P_g_fermion
                * (np.log((1 - x) / mu**2) - 1)
            )
            if not np.isnan(res) and res >= 0:
                return res
            else:
                return 0

        return np.vectorize(fn)(e_g)
