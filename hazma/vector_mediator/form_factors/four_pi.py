from dataclasses import dataclass
from typing import Tuple

import numpy as np

from hazma.vector_mediator.form_factors import utils
from hazma.vector_mediator.form_factors.utils import (
    MPI_GEV,
    MRHO_GEV,
    RealArray,
)

from hazma.rambo import generate_phase_space


@dataclass
class FormFactorPiPiPiPi:

    mass_rho = 0.7755
    mass_rho1 = 1.459
    mass_rho2 = 1.72
    width_rho = 0.1494
    width_rho1 = 0.4
    width_rho2 = 0.25

    mass_omega = 0.78265
    width_omega = 0.00849

    mass_a1 = 1.23
    width_a1 = 0.2

    g_omega_pi_rho: float = 42.3
    g_rho_pi_pi: float = 5.997
    g_rho_gamma: float = 0.1212

    def mass_squared(self, p1: RealArray):
        s = p1 * p1
        s[:, 1:] *= -1
        return np.sum(s, axis=1)

    def dot(self, p1: RealArray, p2: RealArray):
        s = p1 * p2
        s[:, 1:] *= -1
        return np.sum(s, axis=1)

    def bw3(self, q2: RealArray, mass: float, width: float) -> RealArray:
        w1: RealArray = (
            (q2 - 4.0 * MPI_GEV ** 2) / (mass ** 2 - 4.0 * MPI_GEV ** 2)
        ) ** 3
        w2 = np.clip(0.0, w1)  # type: ignore
        den = mass ** 2 - q2 - 1j * width * mass ** 2 * np.sqrt(w2 / q2)
        return mass ** 2 / den

    def h(self, s1, s2, s3):
        return (
            self.bw3(s1, self.mass_rho, self.width_rho)
            + self.bw3(s2, self.mass_rho, self.width_rho)
            + self.bw3(s3, self.mass_rho, self.width_rho)
        )

    def wa1(self, q2):
        delta = np.clip(0.0, q2 - 9.0 * MPI_GEV ** 2)  # type: ignore
        return np.where(
            q2 > 0.838968432668,
            1.623 * q2 + 10.38 - 9.32 / q2 + 0.65 / q2 ** 2,
            4.1 * delta ** 3 * (1.0 - 3.3 * delta + 5.8 * delta ** 2),
        )

    def breit_wigner_a1(self, q2, mass, width):
        m2 = mass ** 2
        return m2 / (m2 - q2 - 1j * width * mass * self.wa1(q2) / self.wa1(m2))

    def brho(self, q2):
        beta = -0.145
        return (
            1.0
            / (1.0 + beta)
            * (
                self.bw3(q2, self.mass_rho, self.width_rho)
                + beta * self.bw3(q2, self.mass_rho1, self.width_rho1)
            )
        )

    def trho(self, q2):
        beta1 = 0.08
        beta2 = -0.0075
        return (
            1.0
            / (1.0 + beta1 + beta2)
            * (
                self.bw3(q2, self.mass_rho, self.width_rho)
                + beta1 * self.bw3(q2, self.mass_rho1, self.width_rho1)
                + beta2 * self.bw3(q2, self.mass_rho2, self.width_rho2)
            )
        )

    def bf0(self, q2):
        return mf0 ** 2 / complex(
            mf0 ** 2 - q2,
            -gf0
            * mf0 ** 2
            * math.sqrt(
                max(0.0, ((q2 - 4.0 * mpip ** 2) / (mf0 ** 2 - 4.0 * mpip ** 2))) / q2
            ),
        )

    def current_omega(self, p1: RealArray, p2: RealArray, p3: RealArray, p4: RealArray):
        p = p1 + p2 + p3 + p4
        s1 = self.mass_squared(p2 + p3)
        s2 = self.mass_squared(p2 + p4)
        s3 = self.mass_squared(p3 + p4)
        pre = (
            2.0
            * utils.breit_wigner_fw(
                self.mass_squared(p2 + p3 + p4), self.mass_omega, self.width_omega
            )
            * self.g_omega_pi_rho
            * self.g_rho_pi_pi
            * self.h(s1, s2, s3)
        )
        p2Q = self.dot(p2, p)
        p3Q = self.dot(p3, p)
        p4Q = self.dot(p4, p)
        p1p2 = self.dot(p1, p2)
        p1p3 = self.dot(p1, p3)
        p1p4 = self.dot(p1, p4)
        current = (
            p2 * (p1p4 * p3Q - p1p3 * p4Q)
            + p3 * (p1p2 * p4Q - p1p4 * p2Q)
            + p4 * (p1p3 * p2Q - p1p2 * p3Q)
        )
        return -pre * current

    def current_f0(
        self, q2, p1: RealArray, p2: RealArray, p3: RealArray, p4: RealArray
    ):
        m342 = self.mass_squared(p3 + p4)
        m122 = self.mass_squared(p1 + p2)
        q = p1 + p2 + p3 + p4
        current = p3 - p4 - q * self.dot(q, p3 - p4) / q2
        return -self.trho(m342) * self.bf0(m122) * current

    def current_a1(
        self, q2, p1: RealArray, p2: RealArray, p3: RealArray, p4: RealArray
    ):
        p = p1 + p2 + p3 + p4
        msq = self.mass_squared(p - p1)
        pre = self.breit_wigner_a1(msq, self.mass_a1, self.width_a1) * self.brho(
            self.mass_squared(p3 + p4)
        )
        current = (
            p3
            - p4
            + p1 * self.dot(p2, p3 - p4) / msq
            - p
            * (
                self.dot(p, p3 - p4) / q2
                + self.dot(p, p1) * self.dot(p2, p3 - p4) / q2 / msq
            )
        )
        return pre * current

    def bwrr(self, q2):
        return (
            self.bw3(q2, self.mass_rho, self.width_rho) / self.mass_rho ** 2
            - self.bw3(q2, self.mass_rho1, self.width_rho1) / self.mass_rho1 ** 2
        )

    def grho(self, p1, p2, p3, p4):
        pre = self.bwrr(self.mass_squared(p1 + p3)) * (
            self.bwrr(self.mass_squared(p2 + p4))
            * self.dot(p1 + p2 + 3.0 * p3 + p4, p2 - p4)
            + 2.0
        )
        return pre * p1

    # rho contribution
    def current_rho(self, q2, p1, p2, p3, p4):
        p = p1 + p2 + p3 + p4
        c1 = (
            self.grho(p1, p2, p3, p4)
            + self.grho(p4, p1, p2, p3)
            - self.grho(p1, p2, p4, p3)
            - self.grho(p3, p1, p2, p4)
            + self.grho(p2, p1, p3, p4)
            + self.grho(p4, p2, p1, p3)
            - self.grho(p2, p1, p4, p3)
            - self.grho(p3, p2, p1, p4)
        )
        d1 = self.dot(p, c1) / q2
        return (
            -(self.g_rho_pi_pi ** 3) * self.g_rho_gamma * self.bwrr(q2) * (c1 - p * d1)
        )

    def form_factor_rho(self, q2, beta1, beta2, beta3):
        return (
            1.0
            / (1 + beta1 + beta2 + beta3)
            * (
                self.bw3(q2, self.mass_rho, self.width_rho)
                + beta1 * self.bw3(q2, mBar1, gBar1)
                + beta2 * self.bw3(q2, mBar2, gBar2)
                + beta3 * self.bw3(q2, mBar3, gBar3)
            )
        )
