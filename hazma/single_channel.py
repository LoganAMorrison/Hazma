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
        self.mx = mx
        self.fs = fs
        self.sigma = sigma

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

    def annihilation_cross_section_funcs(self):
        return {self.fs: lambda e_cm: self.sigma}

    def spectrum_funcs(self):
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
                e_pi0 = (e_cm ** 2 + m_pi0 ** 2) / (2.0 * e_cm)
                return dnde_g_pi0(e_g, e_pi0)

        elif self.fs == "pi pi":

            def dnde_g(e_g, e_cm):
                return 2 * dnde_g_pi(e_g, e_cm / 2) + self._dnde_ap_scalar(
                    e_g, e_cm, m_pi
                )

        else:
            return {}

        return {self.fs: dnde_g}

    def gamma_ray_lines(self, e_cm):
        if self.fs == "g g":
            return {"g g": {"energy": e_cm / 2, "bf": 1}}
        elif self.fs == "pi0 g":
            return {
                "pi0 g": {"energy": (e_cm ** 2 - m_pi0 ** 2) / (2.0 * e_cm), "bf": 1}
            }
        else:
            return {}

    def positron_spectrum_funcs(self):
        if self.fs == "mu mu":

            def dnde_p(e_p, e_cm):
                return dnde_p_mu(e_p, e_cm / 2.0)

        elif self.fs == "pi pi":

            def dnde_p(e_p, e_cm):
                return dnde_p_pi(e_p, e_cm / 2.0)

        else:
            return {}

        return {self.fs: dnde_p}

    def positron_lines(self, e_cm):
        if self.fs == "e e":
            return {"e e": {"energy": e_cm / 2.0, "bf": 1}}
        else:
            return {}

    def list_annihilation_final_states(self):
        return [self.fs]
