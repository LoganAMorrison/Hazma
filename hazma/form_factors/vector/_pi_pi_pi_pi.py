from dataclasses import dataclass, field
from typing import Tuple, overload, Union

import numpy as np

from hazma.rambo import PhaseSpace
from hazma.utils import RealArray

from ._base import VectorFormFactorPPPP

MPI_GEV = 0.13957061
MPI0_GEV = 0.1349770
MRHO_GEV = 0.7755
GRHO_GEV = 0.1494
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

MPI = MPI_GEV
MPI0 = MPI0_GEV


def lnorm_sqr(q):
    return q[0] ** 2 - q[1] ** 2 - q[2] ** 2 - q[3] ** 2


def ldot(p, k):
    return p[0] * k[0] - p[1] * k[1] - p[2] * k[2] - p[3] * k[3]


@dataclass
class _VectorFormFactorPiPiPiPiBase(VectorFormFactorPPPP):
    _imode: int = field()
    _fsp_masses: Tuple[float, float, float, float] = field(init=False)
    fsp_masses: Tuple[float, float, float, float] = field(init=False)

    mass_rho_bar_1: float = field(default=1.437)
    mass_rho_bar_2: float = field(default=1.738)
    mass_rho_bar_3: float = field(default=2.12)

    width_rho_bar_1: float = field(default=0.6784824438511003)
    width_rho_bar_2: float = field(default=0.8049287553822373)
    width_rho_bar_3: float = field(default=0.20919646790795576)

    beta_a1_1: float = field(default=-0.051871563361440096)  # unitless
    beta_a1_2: float = field(default=-0.041610293030827125)  # unitless
    beta_a1_3: float = field(default=-0.0018934309483457441)  # unitless

    beta_f0_1: float = field(default=73860.28659732222)
    beta_f0_2: float = field(default=-26182.725634782986)
    beta_f0_3: float = field(default=333.6314358023821)

    beta_omega_1: float = field(default=-0.36687866443745953)
    beta_omega_2: float = field(default=0.036253295280213906)
    beta_omega_3: float = field(default=-0.004717302695776386)

    beta_b_rho: float = field(default=-0.145)
    beta_t_rho1: float = field(default=0.08)
    beta_t_rho2: float = field(default=-0.0075)

    coupling_f0: float = field(default=124.10534971287902)
    coupling_a1: float = field(default=-201.79098091602876)
    coupling_rho: float = field(default=-2.3089567893904537)
    coupling_omega: float = field(default=-1.5791482789120541)

    coupling_rho_gamma: float = field(default=0.1212)  # GeV^2
    coupling_rho_pi_pi: float = field(default=5.997)
    coupling_omega_pi_rho: float = field(default=42.3)  # GeV^-5

    def __post_init__(self):
        if self._imode == 0:
            self._fsp_masses = (MPI_GEV, MPI_GEV, MPI0_GEV, MPI0_GEV)
        elif self._imode == 1:
            self._fsp_masses = (MPI_GEV,) * 4
        else:
            raise ValueError(f"Invalid _imode = {self._imode}. Must be 0 or 1.")
        self.fsp_masses = tuple(m * 1e3 for m in self._fsp_masses)

    # ============================================================================
    # ---- Propagators -----------------------------------------------------------
    # ============================================================================

    def _breit_wigner(self, s, mass, width):
        """
        Compute the generic Breit-Wigner propagator.
        """
        return mass**2 / (mass**2 - s - 1j * mass * width)

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
        m2 = M_A1**2

        def g(x):
            y = x - 9 * MPI_GEV**2
            return np.where(
                x > 0.838968432668,  # (M_A1 + MPI_GEV) ** 2,
                a1 * x + a0 + am1 / x + am2 / x**2,
                c3 * y**3 * (1.0 + c4 * y + c5 * y**2),
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
        mf2 = M_F0**2
        mu = 4.0 * MPI_GEV**2
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
        prop1 = self._breit_wigner3(s, MRHO_GEV, GRHO_GEV) / MRHO_GEV**2
        prop2 = self._breit_wigner3(s, M_RHO1, G_RHO1) / M_RHO1**2
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
        prop0 = self._breit_wigner3(s, MRHO_GEV, GRHO_GEV)
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
        m2 = mass**2
        mu = 4.0 * MPI_GEV**2
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
        prop0 = self._breit_wigner3(s, MRHO_GEV, GRHO_GEV)
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
        prop0 = self._breit_wigner3(s, MRHO_GEV, GRHO_GEV)
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
        prop1 = self._breit_wigner3(s1, MRHO_GEV, GRHO_GEV)
        prop2 = self._breit_wigner3(s2, MRHO_GEV, GRHO_GEV)
        prop3 = self._breit_wigner3(s3, MRHO_GEV, GRHO_GEV)
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
            self.coupling_rho * self.coupling_rho_pi_pi**3 * self.coupling_rho_gamma
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

    def _current_neutral(self, Q, Q2, q1, q2, q3, q4):
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
        result += self._current_a1(Q, Q2, q1, q2, q3, q4)
        result += self._current_f0(Q, Q2, q1, q2, q3, q4)
        result += self._current_omega(Q, Q2, q1, q2, q3, q4)
        result += self._current_rho(Q, Q2, q1, q2, q3, q4)

        return result  # / np.sqrt(2)

    def _current_charged(self, Q, Q2, q1, q2, q3, q4):
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

        j1 = self._current_neutral(Q, Q2, q3, q4, q1, q2)
        j2 = self._current_neutral(Q, Q2, q1, q4, q3, q2)
        j3 = self._current_neutral(Q, Q2, q3, q2, q1, q4)
        j4 = self._current_neutral(Q, Q2, q1, q2, q3, q4)

        return j1 + j2 + j3 + j4

    def _form_factor(self, Q, Q2, q1, q2, q3, q4):
        """
        Compute the total hadronic current.

        Parameters
        ----------
        Q: ndarray
            Total momentum of the system.
        Q2: ndarray
            Squared center-of-mass energy.
        q1, q2, q3, q4: ndarray
            Momenta of the charged pions.
        """
        if self._imode == 0:
            return self._current_neutral(Q, Q2, q1, q2, q3, q4)
        return self._current_charged(Q, Q2, q1, q2, q3, q4)

    def form_factor(self, momenta: RealArray):
        """
        Compute the total hadronic current.

        Parameters
        ----------
        momenta: ndarray
            Array containing the four-momenta of the final state pions.
        """
        ps = momenta * 1e-3

        q1 = ps[:, 0]
        q2 = ps[:, 1]
        q3 = ps[:, 2]
        q4 = ps[:, 3]

        q = np.sum(ps, axis=1)
        q2 = np.mean(np.sum(ps[0], axis=0))

        ff = self._form_factor(q, q2, q1, q2, q3, q4)

        return ff

    def _msqrd(self, momenta, gvuu: float, gvdd: float):
        """
        Compute the squared matrix element.

        Parameters
        ----------
        momenta: np.ndarray
            Four-momenta of the final-state pions.
        gvuu: float
            Coupling of vector to up-quarks.
        gvdd: float
            Coupling of vector to down-quarks.
        neutral: bool
            If true, form-factor for pi-pi-pi0-pi0 is returned.

        Returns
        -------
        msqrd: np.ndarray
        """
        Q = np.sum(momenta, axis=1)
        Q2 = lnorm_sqr(Q)

        q1 = momenta[:, 0, :]
        q2 = momenta[:, 1, :]
        q3 = momenta[:, 2, :]
        q4 = momenta[:, 3, :]

        if self._imode == 0:
            current = self._current_neutral(Q, Q2, q1, q2, q3, q4)
        else:
            current = self._current_charged(Q, Q2, q1, q2, q3, q4)

        j = -np.real(ldot(current, np.conj(current)))
        weights = j / (3.0 * Q2)

        c1 = gvuu - gvdd
        if self._imode == 0:
            pre = c1**2 / 2.0
        else:
            pre = c1**2 / 4.0

        return pre * weights

    def _integrated_form_factor(self, *, q: float, gvuu: float, gvdd: float, npts: int):

        if q < np.sum(self._fsp_masses):
            return 0.0, 0.0

        phase_space = PhaseSpace(q, self._fsp_masses)
        momenta, weights = phase_space.generate(npts)

        weights = weights * self._msqrd(momenta, gvuu=gvuu, gvdd=gvdd)

        ff_avg = np.nanmean(weights)
        ff_err = np.nanstd(weights) / np.sqrt(momenta.shape[-1])

        return ff_avg, ff_err

    @overload
    def integrated_form_factor(
        self, *, q: float, gvuu: float, gvdd: float, npts: int
    ) -> float:
        ...

    @overload
    def integrated_form_factor(
        self, *, q: RealArray, gvuu: float, gvdd: float, npts: int
    ) -> RealArray:
        ...

    def integrated_form_factor(
        self, *, q: Union[float, RealArray], gvuu: float, gvdd: float, npts: int
    ) -> Union[float, RealArray]:
        """
        Compute the form-factor integrated over phase-space for a four-pion final-state.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy in units of MeV.
        gvuu: float
            Coupling of vector to up-quarks.
        gvdd: float
            Coupling of vector to down-quarks.
        npts: int
            Number of Monte-Carlo phase-space points to use.

        Returns
        -------
        form_factor: float
            Form factor.
        error: float
            Estimated error.
        """
        scalar = np.isscalar(q)
        qq = np.atleast_1d(q) * 1e-3

        integral = np.array(
            [
                self._integrated_form_factor(q=q_, gvuu=gvuu, gvdd=gvdd, npts=npts)[0]
                for q_ in qq
            ]
        )

        if scalar:
            return integral[0]

        return integral

    # ============================================================================
    # ---- Widths ----------------------------------------------------------------
    # ============================================================================

    @overload
    def width(self, mv: float, *, gvuu: float, gvdd: float, npts: int) -> float:
        ...

    @overload
    def width(self, mv: RealArray, *, gvuu: float, gvdd: float, npts: int) -> RealArray:
        ...

    def width(
        self,
        mv: Union[float, RealArray],
        *,
        gvuu: float,
        gvdd: float,
        npts: int = 10000,
    ) -> Union[float, RealArray]:
        r"""Compute the decay width of a massive vector into 4-pions.

        Parameters
        ----------
        mv: float or array-like
            The mass (masses) of the massive vector.
        gvuu: float
            Coupling of the vector to up-quarks.
        gvdd: float
            Coupling of the vector to down-quarks.
        npts: int, optional
            Number of Monte-Carlo phase-space points to use in integration.

        Returns
        -------
        width: float or array-like
            Decay width of vector into 4-pions.
        """
        return (
            0.5
            * mv
            * self.integrated_form_factor(q=mv, gvuu=gvuu, gvdd=gvdd, npts=npts)
        )

    # ============================================================================
    # ---- Cross Sections --------------------------------------------------------
    # ============================================================================

    @overload
    def cross_section(
        self,
        *,
        q: float,
        mx: float,
        mv: float,
        gvxx: float,
        wv: float,
        gvuu: float,
        gvdd: float,
        npts: int,
    ) -> float:
        ...

    @overload
    def cross_section(
        self,
        *,
        q: RealArray,
        mx: float,
        mv: float,
        gvxx: float,
        wv: float,
        gvuu: float,
        gvdd: float,
        npts: int,
    ) -> RealArray:
        ...

    def cross_section(
        self,
        *,
        q: Union[float, RealArray],
        mx: float,
        mv: float,
        gvxx: float,
        wv: float,
        gvuu: float,
        gvdd: float,
        npts: int = 10_000,
    ) -> Union[float, RealArray]:
        r"""Compute the decay width of a massive vector into 4-pions.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy (energies) in MeV.
        mx: float
            Dark matter mass in MeV.
        mv: float
            Vector mediator mass in MeV.
        gvxx: float
            Coupling of the vector to dark matter.
        wv: float
            Vector mediator width in MeV.
        gvuu: float
            Coupling of the vector to up-quarks.
        gvdd: float
            Coupling of the vector to down-quarks.
        npts: int, optional
            Number of Monte-Carlo phase-space points to use in integration.

        Returns
        -------
        sigma: float or array-like
            Annihilation cross-section of dark matter into 4-pions.
        """
        single = np.isscalar(q)
        qq = np.atleast_1d(q).astype(np.float64)

        s = qq**2
        pre = (
            gvxx**2
            * (s + 2 * mx**2)
            / (np.sqrt(s - 4 * mx**2) * ((s - mv**2) ** 2 + (mv * wv) ** 2))
        )
        pre = pre * 0.5 * qq
        cs = pre * self.integrated_form_factor(q=qq, gvuu=gvuu, gvdd=gvdd, npts=npts)

        if single:
            return cs[0]
        return cs


@dataclass
class VectorFormFactorPiPiPi0Pi0(_VectorFormFactorPiPiPiPiBase):
    r"""Class for computing the form-factors and other quantities related to a
    pi-pi-pi0-pi0 final state.
    """

    _imode: int = 0

    def _current(self, Q, Q2, q1, q2, q3, q4):
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
        result += self._current_a1(Q, Q2, q1, q2, q3, q4)
        result += self._current_f0(Q, Q2, q1, q2, q3, q4)
        result += self._current_omega(Q, Q2, q1, q2, q3, q4)
        result += self._current_rho(Q, Q2, q1, q2, q3, q4)

        return result  # / np.sqrt(2)


@dataclass
class VectorFormFactorPiPiPiPi(_VectorFormFactorPiPiPiPiBase):
    r"""Class for computing the form-factors and other quantities related to a
    four-charged-pion final state.
    """

    _imode: int = 1

    def _current(self, Q, Q2, q1, q2, q3, q4):
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

        j1 = self._current_neutral(Q, Q2, q3, q4, q1, q2)
        j2 = self._current_neutral(Q, Q2, q1, q4, q3, q2)
        j3 = self._current_neutral(Q, Q2, q3, q2, q1, q4)
        j4 = self._current_neutral(Q, Q2, q1, q2, q3, q4)

        return j1 + j2 + j3 + j4
