from typing import Tuple, List
from dataclasses import dataclass

import numpy as np

from hazma.rambo import PhaseSpace
from hazma.utils import RealArray

M_PI = 0.13957061
M_PI0 = 0.1349770
M_RHO = 0.7755
G_RHO = 0.1494
M_RHO1 = 1.459
G_RHO1 = 0.4
M_RHO2 = 1.72
G_RHO2 = 0.25
M_A1 = 1.23
G_A1 = 0.2
M_F0 = 1.35
G_F0 = 0.2
M_OMEGA = 0.78265
G_OMEGA = 0.00849

# TODO: Should the BW propagators with square roots be clipped at zero?
#       (e.g. in a1, f0, BW3)
# TODO: Check minus signs in F0, rho currents


def lnorm_sqr(q):
    return q[0] ** 2 - q[1] ** 2 - q[2] ** 2 - q[3] ** 2


def ldot(p, k):
    return p[0] * k[0] - p[1] * k[1] - p[2] * k[2] - p[3] * k[3]


@dataclass
class FormFactorPiPiPiPi:
    mass_rho_bar_1: float = 1.437
    mass_rho_bar_2: float = 1.738
    mass_rho_bar_3: float = 2.12

    width_rho_bar_1: float = 0.6784824438511003
    width_rho_bar_2: float = 0.8049287553822373
    width_rho_bar_3: float = 0.20919646790795576

    beta_a1_1: float = -0.051871563361440096  # unitless
    beta_a1_2: float = -0.041610293030827125  # unitless
    beta_a1_3: float = -0.0018934309483457441  # unitless

    beta_f0_1: float = 73860.28659732222
    beta_f0_2: float = -26182.725634782986
    beta_f0_3: float = 333.6314358023821

    beta_omega_1: float = -0.36687866443745953
    beta_omega_2: float = 0.036253295280213906
    beta_omega_3: float = -0.004717302695776386

    beta_b_rho: float = -0.145
    beta_t_rho1: float = 0.08
    beta_t_rho2: float = -0.0075

    coupling_f0: float = 124.10534971287902
    coupling_a1: float = -201.79098091602876
    coupling_rho: float = -2.3089567893904537
    coupling_omega: float = -1.5791482789120541

    coupling_rho_gamma: float = 0.1212  # GeV^2
    coupling_rho_pi_pi: float = 5.997
    coupling_omega_pi_rho: float = 42.3  # GeV^-5

    # ============================================================================
    # ---- Propagators -----------------------------------------------------------
    # ============================================================================

    def _breit_wigner(self, s, mass, width):
        """
        Compute the generic Breit-Wigner propagator.
        """
        return mass ** 2 / (mass ** 2 - s - 1j * mass * width)

    def _clip_pos(self, x):
        """
        Return an array with all negative values set to zero.
        """
        return np.clip(x, 0.0, None)

    def _propagator_a1(self, s):
        """
        Compute the Breit-Wigner a1 propagator from ArXiv:0804.0359 Eqn.(A.16).

        Parameters
        ----------
        s: ndarray
            Squared momentum flowing through propagator.
        """
        a1, a0, am1, am2 = 1.623, 10.38, -9.32, 0.65
        c3, c4, c5 = 4.1, -3.3, 5.8
        m2 = M_A1 ** 2

        def g(x):
            y = x - 9 * M_PI ** 2
            return np.where(
                x > 0.838968432668,  # (M_A1 + M_PI) ** 2,
                a1 * x + a0 + am1 / x + am2 / x ** 2,
                c3 * y ** 3 * (1.0 + c4 * y + c5 * y ** 2),
            )

        return self._breit_wigner(s, M_A1, G_A1 * g(s) / g(m2))

    def _propagator_f0(self, s):
        """
        Compute the Breit-Wigner f0 propagator from ArXiv:0804.0359 Eqn.(A.21).

        Parameters
        ----------
        s: ndarray
            Squared momentum flowing through propagator.
        """
        mf2 = M_F0 ** 2
        mu = 4.0 * M_PI ** 2
        width = mf2 / s * (s - mu) / (mf2 - mu)
        # width = G_F0 * np.where(width > 0, 1.0, -1.0) * np.sqrt(np.abs(width))
        width = G_F0 * np.sqrt(self._clip_pos(mf2 / s * (s - mu) / (mf2 - mu)))
        return self._breit_wigner(s, M_F0, width)

    def _propagator_omega(self, s):
        """
        Compute the Breit-Wigner omega propagator from ArXiv:0804.0359 Eqn.(A.24).

        Parameters
        ----------
        s: ndarray
            Squared momentum flowing through propagator.
        """
        return self._breit_wigner(s, M_OMEGA, G_OMEGA)

    def _propagator_rho_g(self, q1, q2, q3, q4):
        """
        Compute the Breit-Wigner-G rho propagator from ArXiv:0804.0359 Eqn.(A.9).
        """
        prop13 = self._propagator_rho_double(lnorm_sqr(q1 + q3))
        prop24 = self._propagator_rho_double(lnorm_sqr(q2 + q4))
        return q1 * prop13 * (prop24 * ldot(q1 + q2 + 3 * q3 + q4, q2 - q4) + 2)

    def _propagator_rho_double(self, s):
        """
        Compute the double Breit-Wigner rho propagator from ArXiv:0804.0359 Eqn.(A.10).

        Parameters
        ----------
        s: ndarray
            Squared momentum flowing through propagator.
        """
        prop1 = self._breit_wigner3(s, M_RHO, G_RHO) / M_RHO ** 2
        prop2 = self._breit_wigner3(s, M_RHO1, G_RHO1) / M_RHO1 ** 2
        return prop1 - prop2

    def _propagator_rho_f(self, s, b1, b2, b3):
        """
        Compute the 4 Breit-Wigner rho propagator from ArXiv:0804.0359 Eqn.(A.11).

        Parameters
        ----------
        s: ndarray
            Squared momentum flowing through propagator.
        b1, b2, b3: float
            Scaling factors of propagators 1, 2, and 3.
        """
        prop0 = self._breit_wigner3(s, M_RHO, G_RHO)
        prop1 = self._breit_wigner3(s, self.mass_rho_bar_1, self.width_rho_bar_1)
        prop2 = self._breit_wigner3(s, self.mass_rho_bar_2, self.width_rho_bar_2)
        prop3 = self._breit_wigner3(s, self.mass_rho_bar_3, self.width_rho_bar_3)

        return (prop0 + b1 * prop1 + b2 * prop2 + b3 * prop3) / (1.0 + b1 + b2 + b3)

    def _breit_wigner3(self, s, mass, width):
        """
        Compute the Breit-Wigner-3 propagator from ArXiv:0804.0359 Eqn.(A.12).

        Parameters
        ----------
        s: ndarray
            Squared momentum flowing through propagator.
        mass: float
            Mass of the resonance.
        width: float
            Width of the resonance.
        """
        m2 = mass ** 2
        mu = 4.0 * M_PI ** 2
        gt = self._clip_pos(m2 / s * ((s - mu) / (m2 - mu)) ** 3)
        # gt = m2 / s * ((s - mu) / (m2 - mu)) ** 3
        gt = np.where(gt > 0.0, 1.0, -1.0) * np.sqrt(np.abs(gt))

        return self._breit_wigner(s, mass, width * gt)

    def _propagator_rho_b(self, s):
        """
        Compute the Breit-Wigner-B rho propagator from ArXiv:0804.0359 Eqn.(A.14).

        Parameters
        ----------
        s: ndarray
            Squared momentum flowing through propagator.
        """
        b1 = self.beta_b_rho
        prop0 = self._breit_wigner3(s, M_RHO, G_RHO)
        prop1 = self._breit_wigner3(s, M_RHO1, G_RHO1)

        return (prop0 + b1 * prop1) / (1.0 + b1)

    def _propagator_rho_t(self, s):
        """
        Compute the Breit-Wigner-T rho propagator from ArXiv:0804.0359 Eqn.(A.19).

        Parameters
        ----------
        s: ndarray
            Squared momentum flowing through propagator.
        """
        b1, b2 = self.beta_t_rho1, self.beta_t_rho2
        prop0 = self._breit_wigner3(s, M_RHO, G_RHO)
        prop1 = self._breit_wigner3(s, M_RHO1, G_RHO1)
        prop2 = self._breit_wigner3(s, M_RHO2, G_RHO2)

        return (prop0 + b1 * prop1 + b2 * prop2) / (1.0 + b1 + b2)

    def _propagator_rho_h(self, s1, s2, s3):
        """
        Compute the Breit-Wigner-H rho propagator from ArXiv:0804.0359 Eqn.(A.23).

        Parameters
        ----------
        s1,s2,s3: ndarray
            Squared momentum flowing through propagators 1, 2, and 3.
        """
        prop1 = self._breit_wigner3(s1, M_RHO, G_RHO)
        prop2 = self._breit_wigner3(s2, M_RHO, G_RHO)
        prop3 = self._breit_wigner3(s3, M_RHO, G_RHO)
        return prop1 + prop2 + prop3

    # ============================================================================
    # ---- Currents --------------------------------------------------------------
    # ============================================================================

    def _current_a1(self, Q, Q2, q1, q2, q3, q4):
        """
        Compute the hadronic current contribution from A1 meson.
        See ArXiv:0804.0359 Eqn.(A.2).
        """
        cur = self._current_a1_t(Q, Q2, q3, q2, q1, q4)
        cur += self._current_a1_t(Q, Q2, q3, q1, q2, q4)
        cur -= self._current_a1_t(Q, Q2, q4, q2, q1, q3)
        cur -= self._current_a1_t(Q, Q2, q4, q1, q2, q3)
        return cur

    def _current_a1_t(self, Q, Q2, q1, q2, q3, q4):
        """
        Components of the hadronic current contribution from A1 meson.
        See ArXiv:0804.0359 Eqn.(A.3).
        """
        qmq1_sqr = lnorm_sqr(Q - q1)

        coupling = self.coupling_a1

        b1, b2, b3 = self.beta_a1_1, self.beta_a1_2, self.beta_a1_3
        prop0 = self._propagator_rho_f(Q2, b1, b2, b3)
        prop1 = self._propagator_a1(qmq1_sqr)
        prop2 = self._propagator_rho_b(lnorm_sqr(q3 + q4))
        prop = prop0 * prop1 * prop2

        qq = (q3 - q4) + q1 * ldot(q2, q3 - q4) / qmq1_sqr
        lorentz = qq - Q * ldot(Q, qq) / Q2

        return coupling * prop * lorentz

    def _current_f0(self, Q, Q2, q1, q2, q3, q4):
        """
        Compute the hadronic current contribution from the omega meson.
        See ArXiv:0804.0359 Eqn.(A.4).
        """
        coupling = self.coupling_f0

        b1, b2, b3 = self.beta_f0_1, self.beta_f0_2, self.beta_f0_3
        prop0 = self._propagator_rho_f(Q2, b1, b2, b3)
        prop1 = self._propagator_rho_t(lnorm_sqr(q3 + q4))
        prop2 = self._propagator_f0(lnorm_sqr(q1 + q2))
        prop = prop0 * prop1 * prop2

        lorentz = (q3 - q4) - Q * ldot(Q, q3 - q4) / Q2

        return coupling * prop * lorentz

    def _current_omega(self, Q, Q2, q1, q2, q3, q4):
        """
        Compute the hadronic current contribution from the omega meson.
        See ArXiv:0804.0359 Eqn.(A.5).
        """
        return self._current_omega_t(Q, Q2, q1, q2, q3, q4) + self._current_omega_t(
            Q, Q2, q2, q1, q3, q4
        )

    def _current_omega_t(self, Q, Q2, q1, q2, q3, q4):
        """
        Components the hadronic current contribution from the omega meson from
        rho -> pi + [omega -> 3pi]. See ArXiv:0804.0359 Eqn.(A.6).
        """
        coupling = (
            2
            * self.coupling_omega
            * self.coupling_omega_pi_rho
            * self.coupling_rho_pi_pi
        )

        b1, b2, b3 = self.beta_omega_1, self.beta_omega_2, self.beta_omega_3
        prop0 = self._propagator_rho_f(Q2, b1, b2, b3)
        prop1 = self._propagator_omega(lnorm_sqr(Q - q1))
        prop2 = self._propagator_rho_h(
            lnorm_sqr(q2 + q3), lnorm_sqr(q2 + q4), lnorm_sqr(q3 + q4)
        )
        prop = prop0 * prop1 * prop2

        lorentz = (
            q2 * (ldot(q1, q4) * ldot(q3, Q) - ldot(q1, q3) * ldot(q4, Q))
            + q3 * (ldot(q1, q2) * ldot(q4, Q) - ldot(q1, q4) * ldot(q2, Q))
            + q4 * (ldot(q1, q3) * ldot(q2, Q) - ldot(q1, q2) * ldot(q3, Q))
        )

        return coupling * prop * lorentz

    def _current_rho(self, Q, Q2, q1, q2, q3, q4):
        """
        Compute the hadronic current contribution from the rho meson.
        See ArXiv:0804.0359 Eqn.(A.8).
        """
        coupling = (
            self.coupling_rho * self.coupling_rho_pi_pi ** 3 * self.coupling_rho_gamma
        )

        prop = self._propagator_rho_double(Q2)

        gnu = (
            self._propagator_rho_g(q1, q2, q3, q4)
            + self._propagator_rho_g(q4, q1, q2, q3)
            - self._propagator_rho_g(q1, q2, q4, q3)
            - self._propagator_rho_g(q3, q1, q2, q4)
            + self._propagator_rho_g(q2, q1, q3, q4)
            + self._propagator_rho_g(q4, q2, q1, q3)
            - self._propagator_rho_g(q2, q1, q4, q3)
            - self._propagator_rho_g(q3, q2, q1, q4)
        )

        lorentz = gnu - Q * ldot(Q, gnu) / Q2

        return coupling * prop * lorentz

    def _current_neutral(
        self,
        Q,
        Q2,
        q1,
        q2,
        q3,
        q4,
        contributions: List[str] = ["a1", "f0", "omega", "rho"],
    ):
        """
        Compute the total hadronic current for <pi^+,pi^-,pi^0,pi^0|J|0>.
        See ArXiv:0804.0359 Eqn.(3) and Eqn.(A.1).

        Parameters
        ----------
        Q: ndarray
            Total momentum of the system.
        Q2: ndarray
            Squared center-of-mass energy.
        q1, q2: ndarray
            Momenta of the neutral pions.
        q3, q4: ndarray
            Momenta of the charged pions.
        """
        result = np.zeros_like(q1, dtype=np.complex128)
        if "a1" in contributions:
            result += self._current_a1(Q, Q2, q1, q2, q3, q4)
        if "f0" in contributions:
            result += self._current_f0(Q, Q2, q1, q2, q3, q4)
        if "omega" in contributions:
            result += self._current_omega(Q, Q2, q1, q2, q3, q4)
        if "rho" in contributions:
            result += self._current_rho(Q, Q2, q1, q2, q3, q4)

        return result  # / np.sqrt(2)

    def _current_charged(
        self,
        Q,
        Q2,
        q1,
        q2,
        q3,
        q4,
        contributions: List[str] = ["a1", "f0", "omega", "rho"],
    ):
        """
        Compute the total hadronic current for <pi^+,pi^+,pi^-,pi^-|J|0>.
        See ArXiv:0804.0359 Eqn.(3) and Eqn.(A.1).

        Parameters
        ----------
        Q: ndarray
            Total momentum of the system.
        Q2: ndarray
            Squared center-of-mass energy.
        q1, q2, q3, q4: ndarray
            Momenta of the charged pions.
        """
        # p1+ == p1p == q1
        # p1- == p1m == q2
        # p2+ == p2p == q3
        # p2- == p2m == q4

        j1 = self._current_neutral(Q, Q2, q3, q4, q1, q2, contributions)
        j2 = self._current_neutral(Q, Q2, q1, q4, q3, q2, contributions)
        j3 = self._current_neutral(Q, Q2, q3, q2, q1, q4, contributions)
        j4 = self._current_neutral(Q, Q2, q1, q2, q3, q4, contributions)

        return j1 + j2 + j3 + j4

    # ============================================================================
    # ---- Widths ----------------------------------------------------------------
    # ============================================================================

    def _width(
        self,
        mass: float,
        momenta: RealArray,
        weights: RealArray,
        neutral: bool,
        contributions: List[str] = ["a1", "f0", "omega", "rho"],
    ) -> Tuple[float, float]:
        """
        Compute the width.
        """
        Q = np.sum(momenta, axis=1)
        Q2 = lnorm_sqr(Q)

        q1 = momenta[:, 0, :]
        q2 = momenta[:, 1, :]
        q3 = momenta[:, 2, :]
        q4 = momenta[:, 3, :]

        if neutral:
            current = self._current_neutral(Q, Q2, q1, q2, q3, q4, contributions)
        else:
            current = self._current_charged(Q, Q2, q1, q2, q3, q4, contributions)

        jj = np.real(ldot(current, np.conj(current)))
        jq = np.abs(ldot(Q, np.conj(current))) ** 2

        wgts = weights * (-jj + jq / mass ** 2)

        if neutral:
            pre = 1.0 / (6.0 * mass) / 2.0
        else:
            pre = 1.0 / (6.0 * mass) / 4.0

        width = pre * np.nanmean(wgts)
        error = pre * np.nanstd(wgts) / np.sqrt(momenta.shape[-1])

        return width, error

    def decay_width(
        self,
        mv: float,
        n: int,
        neutral: bool,
        contributions: List[str] = ["a1", "f0", "omega", "rho"],
    ) -> Tuple[float, float]:
        """
        Compute the decay width into four pions.
        """
        if neutral:
            masses = np.array([M_PI0, M_PI0, M_PI, M_PI])
        else:
            masses = np.array([M_PI, M_PI, M_PI, M_PI])

        if mv < np.sum(masses):
            return 0.0, 0.0

        phase_space = PhaseSpace(mv, masses)
        momenta, weights = phase_space.generate(n)
        return self._width(
            mv, momenta, weights, neutral=neutral, contributions=contributions
        )

    # ============================================================================
    # ---- Cross Sections --------------------------------------------------------
    # ============================================================================

    def _cross_section(
        self,
        cme: float,
        mx: float,
        mv: float,
        widthv: float,
        momenta: RealArray,
        weights: RealArray,
        neutral: bool,
    ) -> Tuple[float, float]:
        """
        Compute the cross section for x + xbar -> 4pi
        """

        Q = np.sum(momenta, axis=1)
        Q2 = lnorm_sqr(Q)

        q1 = momenta[:, 0, :]
        q2 = momenta[:, 1, :]
        q3 = momenta[:, 2, :]
        q4 = momenta[:, 3, :]

        ex = cme / 2
        px = np.sqrt(ex ** 2 - mx ** 2)
        p1 = np.array([ex, 0.0, 0.0, px])
        p2 = np.array([ex, 0.0, 0.0, -px])

        if neutral:
            j = self._current_neutral(Q, Q2, q1, q2, q3, q4)
        else:
            j = self._current_charged(Q, Q2, q1, q2, q3, q4)

        jc = np.conj(j)

        s = cme ** 2
        # Coefficient of (JC.J)
        coeff_jc_j = 2 * mv ** 4 * mx ** 2 - 0.5 * mv ** 4 * s
        # Coefficient of (JC.p1) * (JC.p2)
        coeff_jcp1_jp2 = mv ** 4 - 4 * mv ** 2 * mx ** 2 + 2 * mx ** 2 * s
        # Coefficient of (JC.p2) * (JC.p2)
        coeff_jcp2_jp2 = -4 * mv ** 2 * mx ** 2 + 4 * mx ** 2 * s
        # Coefficient of (JC.p1) * (JC.p1)
        coeff_jcp1_jp1 = -4 * mv ** 2 * mx ** 2 + 4 * mx ** 2 * s
        # Coefficient of (JC.p2) * (JC.p1)
        coeff_jcp2_jp1 = mv ** 4 - 4 * mv ** 2 * mx ** 2 + 4 * mx ** 2 * s

        msqrd_jc_j = np.real(ldot(j, jc))
        msqrd_jcp1_jp2 = ldot(jc, p1) * ldot(j, p2)
        msqrd_jcp2_jp2 = ldot(jc, p2) * ldot(j, p2)
        msqrd_jcp1_jp1 = ldot(jc, p1) * ldot(j, p1)
        msqrd_jcp2_jp1 = ldot(jc, p2) * ldot(j, p1)

        den = mv ** 4 * (mv ** 2 - s) ** 2 + mv ** 6 * widthv ** 2

        msqrd = (
            msqrd_jc_j * coeff_jc_j
            + msqrd_jcp1_jp2 * coeff_jcp1_jp2
            + msqrd_jcp2_jp2 * coeff_jcp2_jp2
            + msqrd_jcp1_jp1 * coeff_jcp1_jp1
            + msqrd_jcp2_jp1 * coeff_jcp2_jp1
        )

        wgts = weights * msqrd / den

        pre = 1.0 / (4.0 * px * cme)
        width = pre * np.nanmean(wgts)
        error = pre * np.nanstd(wgts) / np.sqrt(Q.shape[-1])

        return width, error

    def cross_section(
        self, cme: float, mx: float, mv: float, widthv: float, n: int, neutral: bool
    ) -> Tuple[float, float]:
        """
        Compute the cross-section for x + xbar -> 4pi.
        """
        if neutral:
            accessable = (cme > 2 * mx) and (cme > 2 * M_PI0 + 2 * M_PI)
        else:
            accessable = (cme > 2 * mx) and (cme > 4 * M_PI)

        if not accessable:
            return 0.0, 0.0

        if neutral:
            masses = np.array([M_PI0, M_PI0, M_PI, M_PI])
        else:
            masses = np.array([M_PI, M_PI, M_PI, M_PI])

        phase_space = PhaseSpace(cme, masses)
        momenta, weights = phase_space.generate(n)
        return self._cross_section(cme, mx, mv, widthv, momenta, weights, neutral)
