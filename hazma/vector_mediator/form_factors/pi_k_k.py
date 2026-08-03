import abc
from dataclasses import KW_ONLY, InitVar, dataclass, field
from typing import Tuple, Union

import numpy as np

from hazma.phase_space import Rambo
from hazma.utils import lnorm_sqr
from hazma.vector_mediator.form_factors import utils
from hazma.vector_mediator.form_factors.utils import RealArray

KS_MASS_GEV = 0.8956  # KStar mass
KS_WIDTH_GEV = 0.047  # KStar width


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
    phase_space = Rambo(m, np.array([m1, m2, m3]))
    ps, ws = phase_space.generate(n)

    p1 = ps[:, 0]
    p2 = ps[:, 1]
    p3 = ps[:, 2]

    s = lnorm_sqr(p2 + p3)
    t = lnorm_sqr(p1 + p3)

    return s, t, ws


ISO_SCALAR_MASSES = np.array([1019.461e-3, 1633.4e-3, 1957e-3])
ISO_SCALAR_WIDTHS = np.array([4.249e-3, 218e-3, 267e-3])
ISO_SCALAR_AMPS = np.array([0.0, 0.233, 0.0405])
ISO_SCALAR_PHASES = np.array([0, 1.1e-07, 5.19])  # * np.pi / 180.0


ISO_VECTOR_MASSES = np.array([775.26e-3, 1470e-3, 1720e-3])
ISO_VECTOR_WIDTHS = np.array([149.1e-3, 400e-3, 250e-3])
ISO_VECTOR_AMPS = np.array([-2.34, 0.594, -0.0179])
ISO_VECTOR_PHASES = np.array([0, 0.317, 2.57])  # * np.pi / 180.0


@dataclass
class FormFactorPiKKBase(abc.ABC):
    fsp_masses: Tuple[float, float, float]

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
        s = m**2

        a0 = np.sum(
            cs
            * self.__iso_scalar_amps
            * utils.breit_wigner_fw(s, self.iso_scalar_masses, self.iso_scalar_widths)
        )
        a1 = np.sum(
            ci1
            * self.__iso_vector_amps
            * utils.breit_wigner_fw(s, self.iso_vector_masses, self.iso_vector_widths)
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
            utils.breit_wigner_pwave(s, KS_MASS_GEV, KS_WIDTH_GEV, m1, m2)
            / KS_MASS_GEV**2
        )

    def lorentz_structure(self, s, t, m):
        """
        Returns the Lorentz part of the matrix element.

        Here, s and t are given by:
            s = (P - p1)^2 = (p2 + p3)^2 = invariant mass of 2 and 3
            t = (P - p2)^2 = (p1 + p3)^2 = invariant mass of 1 and 3
        """
        m1, m2, m3 = self.fsp_masses
        return (
            -(
                (m1 * m2 - m * m3)
                * (m1 * m2 + m * m3)
                * (-(m**2) + m1**2 + m2**2 - m3**2)
            )
            - (m - m2) * (m + m2) * (m1 - m3) * (m1 + m3) * s
            - (m - m1) * (m + m1) * (m2 - m3) * (m2 + m3) * t
            + (m**2 + m1**2 + m2**2 + m3**2) * s * t
            - s**2 * t
            - s * t**2
        ) / 4.0

    @abc.abstractmethod
    def _integrand(self, s, t, m, gvuu, gvdd, gvss) -> float:
        pass

    def _integrated_form_factor(
        self, *, m: float, gvuu: float, gvdd: float, gvss: float, npts: int
    ) -> float:
        if m < sum(self.fsp_masses):
            return 0.0

        m1, m2, m3 = self.fsp_masses
        ss, ts, ws = generate_mandelstam_invariants_st(m, m1, m2, m3, npts)
        ws = ws * self._integrand(ss, ts, m, gvuu, gvdd, gvss)
        return np.average(ws)  # type: ignore

    def integrated_form_factor(
        self, *, m: float, gvuu: float, gvdd: float, gvss: float, npts: int
    ) -> float:
        if m < sum(self.fsp_masses):
            return 0.0

        mgev = m * 1e-3
        integral = self._integrated_form_factor(
            m=mgev, gvuu=gvuu, gvdd=gvdd, gvss=gvss, npts=npts
        )
        # gev^2 -> mev^2
        return integral * 1e6

    def width(self, *, m: float, gvuu: float, gvdd: float, gvss: float, npts: int):
        integral = self.integrated_form_factor(
            m=m, gvuu=gvuu, gvdd=gvdd, gvss=gvss, npts=npts
        )
        return integral / (6.0 * m)

    def energy_distributions(
        self,
        *,
        m: float,
        gvuu: float,
        gvdd: float,
        gvss: float,
        npts: int,
        nbins: int = 25,
    ):
        def _msqrd(momenta):
            s = lnorm_sqr(momenta[:, 1] + momenta[:, 2])
            t = lnorm_sqr(momenta[:, 0] + momenta[:, 2])
            return self._integrand(s, t, m, gvuu, gvdd, gvss)

        m1, m2, m3 = self.fsp_masses
        phase_space = Rambo(m, masses=np.array([m1, m2, m3]), msqrd=_msqrd)
        return phase_space.energy_distributions(n=npts, nbins=nbins)


@dataclass
class FormFactorPi0K0K0(FormFactorPiKKBase):
    fsp_masses: Tuple[float, float, float] = (
        utils.MK0_GEV,
        utils.MK0_GEV,
        utils.MPI0_GEV,
    )

    def _integrand(self, s, t, m, gvuu, gvdd, gvss) -> float:
        a0, a1 = self.iso_spin_amplitudes(m, gvuu, gvdd, gvss)
        m1, m2, m3 = self.fsp_masses

        coeff = (a0 + a1) / np.sqrt(6.0) * 2 * self.g_ks_k_pi

        f = coeff * (
            self.kstar_propagator(s, m2, m3) + self.kstar_propagator(t, m1, m3)
        )

        return np.abs(f) ** 2 * self.lorentz_structure(s, t, m)


@dataclass
class FormFactorPi0KpKm(FormFactorPiKKBase):
    fsp_masses: Tuple[float, float, float] = (
        utils.MK_GEV,
        utils.MK_GEV,
        utils.MPI0_GEV,
    )

    def _integrand(self, s, t, m, gvuu, gvdd, gvss) -> float:
        a0, a1 = self.iso_spin_amplitudes(m, gvuu, gvdd, gvss)
        m1, m2, m3 = self.fsp_masses

        coeff = (a0 - a1) / np.sqrt(6.0) * 2 * self.g_ks_k_pi

        f = coeff * (
            self.kstar_propagator(s, m2, m3) + self.kstar_propagator(t, m1, m3)
        )

        return np.abs(f) ** 2 * self.lorentz_structure(s, t, m)


@dataclass
class FormFactorPiKK0(FormFactorPiKKBase):
    fsp_masses: Tuple[float, float, float] = (
        utils.MK0_GEV,
        utils.MK_GEV,
        utils.MPI_GEV,
    )

    def _integrand(self, s, t, m, gvuu, gvdd, gvss) -> float:
        a0, a1 = self.iso_spin_amplitudes(m, gvuu, gvdd, gvss)
        m1, m2, m3 = self.fsp_masses

        cs = (a0 + a1) / np.sqrt(6.0) * 2 * self.g_ks_k_pi
        ct = (a0 - a1) / np.sqrt(6.0) * 2 * self.g_ks_k_pi

        f = ct * self.kstar_propagator(s, m2, m3) + cs * self.kstar_propagator(
            t, m1, m3
        )

        return np.abs(f) ** 2 * self.lorentz_structure(s, t, m)
