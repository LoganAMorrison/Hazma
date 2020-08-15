import numpy as np

from hazma.theory import Theory
from hazma.parameters import (
    neutral_pion_mass as m_pi0,
    charged_pion_mass as m_pi,
    alpha_em,
    electron_mass as m_e,
    muon_mass as m_mu,
)
from hazma.decay import (
    muon as dnde_g_mu,
    neutral_pion as dnde_g_pi0,
    charged_pion as dnde_g_pi,
)
from hazma.positron_spectra import charged_pion as dnde_p_pi, muon as dnde_p_mu


# TODO: use AP approximation for FSR


class SingleChannel(Theory):
    def __init__(self, mx, fs, sigma):
        self._mx = mx
        self._fs = fs
        self._sigma = sigma
        self.setup()

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

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        self._sigma = sigma
        self.setup()

    def setup(self):
        self.set_fs_mass()
        self.set_spectrum_funcs()
        self.set_gamma_ray_line_energies()
        self.set_positron_spectrum_funcs()
        self.set_positron_line_energies()

    def set_fs_mass(self):
        # Sets kinematic threshold for DM annihilations
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

    def set_spectrum_funcs(self):
        """
        Sets gamma ray spectrum functions.
        """
        if self.fs == "e e":

            def dnde_g(e_g, e_cm):
                return self._dnde_ap_fermion(e_g, e_cm, m_e)

        elif self.fs == "mu mu":

            def dnde_g(e_g, e_cm):
                return 2 * dnde_g_mu(e_g, e_cm / 2) + self._dnde_ap_fermion(
                    e_g, e_cm, m_mu
                )

        elif self.fs == "pi0 pi0":

            def dnde_g(e_g, e_cm):
                return 2 * dnde_g_pi0(e_g, e_cm / 2)

        elif self.fs == "pi0 g":

            def dnde_g(e_g, e_cm):
                return dnde_g_pi0(e_g, (e_cm ** 2 + m_pi0 ** 2) / (2.0 * e_cm))

        elif self.fs == "pi pi":

            def dnde_g(e_g, e_cm):
                return 2 * dnde_g_pi(e_g, e_cm / 2) + self._dnde_ap_scalar(
                    e_g, e_cm, m_pi
                )

        else:
            # Final state produces no photons
            self._spectrum_funcs = lambda: {}
            return

        self._spectrum_funcs = lambda: {self.fs: dnde_g}

    def set_gamma_ray_line_energies(self):
        if self.fs == "g g":
            self._gamma_ray_line_energies = lambda e_cm: {"g g": e_cm / 2}
        elif self.fs == "pi0 g":
            self._gamma_ray_line_energies = lambda e_cm: {
                "pi0 g": (e_cm ** 2 - m_pi0 ** 2) / (2.0 * e_cm)
            }
        else:
            self._gamma_ray_line_energies = lambda e_cm: {}

    def set_positron_spectrum_funcs(self):
        if self.fs == "mu mu":

            def dnde_p(e_p, e_cm):
                if e_cm < self.fs_mass:
                    return 0.0
                return dnde_p_mu(e_p, e_cm / 2.0)

        elif self.fs == "pi pi":

            def dnde_p(e_p, e_cm):
                if e_cm < self.fs_mass:
                    return 0.0
                return dnde_p_pi(e_p, e_cm / 2.0)

        else:
            # Final state produces no positrons
            self._positron_spectrum_funcs = lambda: {}
            return

        self._positron_spectrum_funcs = lambda: {self.fs: dnde_p}

    def set_positron_line_energies(self):
        if self.fs == "e e":
            self._positron_line_energies = lambda e_cm: {"e e": e_cm / 2.0}
        else:
            self._positron_line_energies = lambda e_cm: {}

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
                * (np.log((1 - x) / mu ** 2) - 1)
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
                * (np.log((1 - x) / mu ** 2) - 1)
            )
            if not np.isnan(res) and res >= 0:
                return res
            else:
                return 0

        return np.vectorize(fn)(e_g)

    def sigma(self, e_cm):
        if e_cm < 2 * self.mx or e_cm < self.fs_mass:
            return 0.0
        else:
            return self._sigma

    def annihilation_cross_section_funcs(self):
        return {self.fs: self.sigma}

    def list_annihilation_final_states(self):
        return [self.fs]
