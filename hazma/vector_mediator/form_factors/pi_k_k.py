from dataclasses import dataclass, field, InitVar, KW_ONLY

import numpy as np
from scipy import integrate

from hazma.rambo import generate_phase_space
from hazma.utils import kallen_lambda
from hazma.vector_mediator.form_factors.utils import (
    MPI_GEV,
    MK_GEV,
    MPI0_GEV,
    MK0_GEV,
    RealArray,
    breit_wigner_fw,
    breit_wigner_pwave,
)


KS_MASS_GEV = 0.8956  # KStar mass
KS_WDITH_GEV = 0.047  # KStar width


def lorentz_structure(s, t, m, m1, m2, m3):
    """
    Returns the Lorentz part of the matrix element.

    Here, s and t are given by:
        s = (P - p1)^2 = (p2 + p3)^2 = invariant mass of 2 and 3
        t = (P - p2)^2 = (p1 + p3)^2 = invariant mass of 1 and 3
    """
    return (
        -(
            (m1 * m2 - m * m3)
            * (m1 * m2 + m * m3)
            * (-(m ** 2) + m1 ** 2 + m2 ** 2 - m3 ** 2)
        )
        - (m - m2) * (m + m2) * (m1 - m3) * (m1 + m3) * s
        - (m - m1) * (m + m1) * (m2 - m3) * (m2 + m3) * t
        + (m ** 2 + m1 ** 2 + m2 ** 2 + m3 ** 2) * s * t
        - s ** 2 * t
        - s * t ** 2
    ) / 12.0


def lorentz_structure_integrated_over_t(s, m, m1, m2, m3):
    """
    Returns the Lorentz part of the matrix element integrated over t.

    Here, s and t are given by:
        s = (P - p1)^2 = (p2 + p3)^2 = invariant mass of 2 and 3
        t = (P - p2)^2 = (p1 + p3)^2 = invariant mass of 1 and 3
    """
    p1 = kallen_lambda(s, m ** 2, m1 ** 2)
    p2 = kallen_lambda(s, m2 ** 2, m3 ** 2)
    return (p1 * p2) ** 1.5 / (72.0 * s ** 2)


def mandelstam_t_bounds(m, m1, m2, m3):
    """
    Returns functions that compute the upper and lower bounds
    on the mandelstam variable t.
    """

    def f(s):
        return (
            (-(m ** 2) + m1 ** 2) * (m2 - m3) * (m2 + m3)
            + (m ** 2 + m1 ** 2 + m2 ** 2 + m3 ** 2) * s
            - s ** 2
        )

    def g(s):
        return np.sqrt(
            kallen_lambda(s, m ** 2, m1 ** 2) * kallen_lambda(s, m2 ** 2, m1 ** 2)
        )

    def lb(s):
        return (f(s) - g(s)) / (2 * s)

    def ub(s):
        return (f(s) + g(s)) / (2 * s)

    return lb, ub


def generate_mandelstam_invariants_st(m, m1, m2, m3, n=10000):
    """
    Generate mandelstam invariants s and t and phase-space weights from a 3-body process
    assuming a flat squared matrix element.

    Parameters
    ----------
    m: float
        Total energy of the process or mass of decaying parent particle.
    m1,m2,m3: float
        Masses of the 3 final state particles.
    n: int, optional
        Number of invariants and weights to generate.

    Returns
    -------
    s,t: np.ndarray
        Mandelstam invariants.
    ws: np.ndarray
        Phase-space weights.
    """
    ps = generate_phase_space([m1, m2, m3], m, num_ps_pts=n)
    ws = ps[:, -1]
    p1 = ps[:, 0:4]
    p2 = ps[:, 4:8]
    p3 = ps[:, 8:12]

    # Computes [(p20+p30)^2, (p2x+p3x)^2, (p2y+p3y)^2, (p2z+p3z)^2]
    s = (p2 + p3) ** 2
    t = (p1 + p3) ** 2

    # Negate 3-momenta components so we get correct scalar products
    # After this, we have:
    #   [(p20+p30)^2, -(p2x+p3x)^2, -(p2y+p3y)^2, -(p2z+p3z)^2]
    s[:, 1:] *= -1
    t[:, 1:] *= -1

    s = np.sum(s, axis=1)
    t = np.sum(t, axis=1)

    return s, t, ws


ISO_SCALAR_MASSES = np.array([1019.461e-3, 1633.4e-3, 1957e-3])
ISO_SCALAR_WIDTHS = np.array([4.249e-3, 218e-3, 267e-3])
ISO_SCALAR_AMPS = np.array([0.0, 0.233, 0.0405])
ISO_SCALAR_PHASES = np.array([0, 1.1e-07, 5.19]) * np.pi / 180.0


ISO_VECTOR_MASSES = np.array([775.26e-3, 1470e-3, 1720e-3])
ISO_VECTOR_WIDTHS = np.array([149.1e-3, 400e-3, 250e-3])
ISO_VECTOR_AMPS = np.array([-2.34, 0.594, -0.0179])
ISO_VECTOR_PHASES = np.array([0, 0.317, 2.57]) * np.pi / 180.0


@dataclass
class FormFactorPiKKBase:
    _: KW_ONLY
    iso_scalar_masses: RealArray = ISO_SCALAR_MASSES
    iso_scalar_widths: RealArray = ISO_SCALAR_WIDTHS
    iso_scalar_amps: InitVar[RealArray] = ISO_SCALAR_AMPS
    iso_scalar_phases: InitVar[RealArray] = ISO_SCALAR_PHASES

    iso_vector_masses: RealArray = ISO_VECTOR_MASSES
    iso_vector_widths: RealArray = ISO_VECTOR_WIDTHS
    iso_vector_amps: InitVar[RealArray] = ISO_VECTOR_AMPS
    iso_vector_phases: InitVar[RealArray] = ISO_VECTOR_PHASES

    g_ks_k_pi: float = 5.37392360229

    __iso_scalar_amps: RealArray = field(init=False)
    __iso_vector_amps: RealArray = field(init=False)

    def __post_init__(
        self, iso_scalar_amps, iso_scalar_phases, iso_vector_amps, iso_vector_phases
    ):
        self.__iso_scalar_amps = iso_scalar_amps * np.exp(1j * iso_scalar_phases)
        self.__iso_vector_amps = iso_vector_amps * np.exp(1j * iso_vector_phases)

    def iso_spin_amplitudes(self, m, gvuu, gvdd, gvss):
        """
        Compute the amplitude coefficients grouped in terms of iso-spin.
        """
        ci1 = gvuu - gvdd
        cs = -3 * gvss

        a0 = np.sum(
            cs
            * self.__iso_scalar_amps
            * breit_wigner_fw(m ** 2, self.iso_scalar_masses, self.iso_scalar_widths)
        )
        a1 = np.sum(
            ci1
            * self.__iso_vector_amps
            * breit_wigner_fw(m ** 2, self.iso_vector_masses, self.iso_vector_widths)
        )
        return (a0, a1)

    def kstar_propagator(self, s, m1, m2):
        """
        Returns the K^* energy-dependent propagator for a K^* transitioning into two
        other particles.

        Parameters
        ----------
        s: Union[float, np.ndarray]
            Squared momentum of K^*.
        m1, m2: float
            Masses of the particles the K^* transitions into.
        """
        return (
            breit_wigner_pwave(s, KS_MASS_GEV, KS_WDITH_GEV, m1, m2) / KS_MASS_GEV ** 2
        )

    def lorentz_structure(self, s, t, m, m1, m2, m3):
        """
        Returns the Lorentz part of the matrix element.

        Here, s and t are given by:
            s = (P - p1)^2 = (p2 + p3)^2 = invariant mass of 2 and 3
            t = (P - p2)^2 = (p1 + p3)^2 = invariant mass of 1 and 3
        """
        return (
            -(
                (m1 * m2 - m * m3)
                * (m1 * m2 + m * m3)
                * (-(m ** 2) + m1 ** 2 + m2 ** 2 - m3 ** 2)
            )
            - (m - m2) * (m + m2) * (m1 - m3) * (m1 + m3) * s
            - (m - m1) * (m + m1) * (m2 - m3) * (m2 + m3) * t
            + (m ** 2 + m1 ** 2 + m2 ** 2 + m3 ** 2) * s * t
            - s ** 2 * t
            - s * t ** 2
        ) / 4.0


class FormFactorPi0K0K0(FormFactorPiKKBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fsp_masses = [MK0_GEV, MK0_GEV, MPI0_GEV]

    def __integrand(self, s, t, m, gvuu, gvdd, gvss):
        a0, a1 = self.iso_spin_amplitudes(m, gvuu, gvdd, gvss)
        m1, m2, m3 = self._fsp_masses

        coeff = (a0 + a1) / np.sqrt(6.0) * self.g_ks_k_pi

        f = coeff * (
            self.kstar_propagator(s, m2, m3) + self.kstar_propagator(t, m1, m3)
        )

        return np.abs(f) ** 2 * self.lorentz_structure(s, t, m, m1, m2, m3)

    def integrated_form_factor(self, m: float, gvuu: float, gvdd: float, gvss: float):
        m1, m2, m3 = self._fsp_masses
        ss, ts, ws = generate_mandelstam_invariants_st(m, m1, m2, m3)
        ws = ws * self.__integrand(ss, ts, m, gvuu, gvdd, gvss)
        return np.average(ws)


class FormFactorPi0KpKm(FormFactorPiKKBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fsp_masses = [MK_GEV, MK_GEV, MPI0_GEV]

    def __integrand(self, s, t, m, gvuu, gvdd, gvss):
        a0, a1 = self.iso_spin_amplitudes(m ** 2, gvuu, gvdd, gvss)
        m1, m2, m3 = self._fsp_masses

        coeff = (a0 - a1) / np.sqrt(6.0) * self.g_ks_k_pi

        f = coeff * (
            self.kstar_propagator(s, m2, m3) + self.kstar_propagator(t, m1, m3)
        )

        return np.abs(f) ** 2 * self.lorentz_structure(s, t, m, m1, m2, m3)

    def integrated_form_factor(self, m: float, gvuu: float, gvdd: float, gvss: float):
        m1, m2, m3 = self._fsp_masses
        ss, ts, ws = generate_mandelstam_invariants_st(m, m1, m2, m3)
        ws = ws * self.__integrand(ss, ts, m, gvuu, gvdd, gvss)
        return np.average(ws)


class FormFactorPiKK0(FormFactorPiKKBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fsp_masses = [MK0_GEV, MK_GEV, MPI_GEV]

    def __integrand(self, s, t, m, gvuu, gvdd, gvss):
        a0, a1 = self.iso_spin_amplitudes(m ** 2, gvuu, gvdd, gvss)
        m1, m2, m3 = self._fsp_masses

        cs = (a0 - a1) / np.sqrt(6.0) * self.g_ks_k_pi
        ct = (a0 + a1) / np.sqrt(6.0) * self.g_ks_k_pi

        f = cs * self.kstar_propagator(s, m2, m3) + ct * self.kstar_propagator(
            t, m1, m3
        )

        return np.abs(f) ** 2 * self.lorentz_structure(s, t, m, m1, m2, m3)

    def integrated_form_factor(self, m: float, gvuu: float, gvdd: float, gvss: float):
        m1, m2, m3 = self._fsp_masses
        ss, ts, ws = generate_mandelstam_invariants_st(m, m1, m2, m3)
        ws = ws * self.__integrand(ss, ts, m, gvuu, gvdd, gvss)
        return np.average(ws)
