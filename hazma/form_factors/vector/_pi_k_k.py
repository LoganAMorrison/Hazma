from dataclasses import InitVar, dataclass, field
from typing import Tuple, Union
import abc

import numpy as np

from . import _utils
from ._utils import RealArray
from ._base import VectorFormFactorPPP

MK = _utils.MK_GEV * 1e3
MK0 = _utils.MK0_GEV * 1e3
MPI = _utils.MPI_GEV * 1e3
MPI0 = _utils.MPI0_GEV * 1e3

KS_MASS_GEV = 0.8956  # KStar mass
KS_WIDTH_GEV = 0.047  # KStar width


ISO_SCALAR_MASSES = np.array([1019.461e-3, 1633.4e-3, 1957e-3])
ISO_SCALAR_WIDTHS = np.array([4.249e-3, 218e-3, 267e-3])
ISO_SCALAR_AMPS = np.array([0.0, 0.233, 0.0405])
ISO_SCALAR_PHASES = np.array([0, 1.1e-07, 5.19])  # * np.pi / 180.0


ISO_VECTOR_MASSES = np.array([775.26e-3, 1470e-3, 1720e-3])
ISO_VECTOR_WIDTHS = np.array([149.1e-3, 400e-3, 250e-3])
ISO_VECTOR_AMPS = np.array([-2.34, 0.594, -0.0179])
ISO_VECTOR_PHASES = np.array([0, 0.317, 2.57])  # * np.pi / 180.0


@dataclass
class _VectorFormFactorPiKKBase(VectorFormFactorPPP):
    fsp_masses: Tuple[float, float, float] = field(init=False)
    _fsp_masses: Tuple[float, float, float] = field(init=False)

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
        self._fsp_masses = tuple(m * 1e-3 for m in self.fsp_masses)

    def _iso_spin_amplitudes(self, m, gvuu, gvdd, gvss):
        """
        Compute the amplitude coefficients grouped in terms of iso-spin.
        """
        ci1 = gvuu - gvdd
        cs = -3 * gvss
        s = m**2

        a0 = np.sum(
            cs
            * self.__iso_scalar_amps
            * _utils.breit_wigner_fw(s, self.iso_scalar_masses, self.iso_scalar_widths)
        )
        a1 = np.sum(
            ci1
            * self.__iso_vector_amps
            * _utils.breit_wigner_fw(s, self.iso_vector_masses, self.iso_vector_widths)
        )

        return (a0, a1)

    def _kstar_propagator(self, s, m1, m2):
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
            _utils.breit_wigner_pwave(s, KS_MASS_GEV, KS_WIDTH_GEV, m1, m2)
            / KS_MASS_GEV**2
        )

    @abc.abstractmethod
    def _form_factor(self, q, s, t, gvuu, gvdd, gvss) -> float:
        raise NotImplementedError()

    def form_factor(self, q, s, t, gvuu, gvdd, gvss) -> float:
        qq = 1e-3 * q
        ss = 1e-6 * s
        tt = 1e-6 * t
        ff = self._form_factor(qq, ss, tt, gvuu, gvdd, gvss)
        return ff * 1e-9

    def width(
        self,
        mv: Union[float, RealArray],
        gvuu: float,
        gvdd: float,
        gvss: float,
        method: str = "rambo",
        npts: int = 1 << 14,
    ) -> Union[float, RealArray]:
        return self._width(
            mv=mv, gvuu=gvuu, gvdd=gvdd, gvss=gvss, method=method, npts=npts
        )


@dataclass
class VectorFormFactorPi0K0K0(_VectorFormFactorPiKKBase):
    fsp_masses: Tuple[float, float, float] = (MK0, MK0, MPI0)

    def _form_factor(self, q, s, t, gvuu, gvdd, gvss) -> float:
        a0, a1 = self._iso_spin_amplitudes(q, gvuu, gvdd, gvss)
        m1, m2, m3 = self._fsp_masses
        coeff = (a0 + a1) / np.sqrt(6.0) * 2 * self.g_ks_k_pi
        return coeff * (
            self._kstar_propagator(s, m2, m3) + self._kstar_propagator(t, m1, m3)
        )


@dataclass
class VectorFormFactorPi0KpKm(_VectorFormFactorPiKKBase):
    fsp_masses: Tuple[float, float, float] = (MK, MK, MPI0)

    def _form_factor(self, q, s, t, gvuu, gvdd, gvss) -> float:
        a0, a1 = self._iso_spin_amplitudes(q, gvuu, gvdd, gvss)
        m1, m2, m3 = self._fsp_masses
        coeff = (a0 - a1) / np.sqrt(6.0) * 2 * self.g_ks_k_pi
        return coeff * (
            self._kstar_propagator(s, m2, m3) + self._kstar_propagator(t, m1, m3)
        )


@dataclass
class VectorFormFactorPiKK0(_VectorFormFactorPiKKBase):
    fsp_masses: Tuple[float, float, float] = (MK0, MK, MPI)

    def _form_factor(self, q, s, t, gvuu, gvdd, gvss) -> float:
        a0, a1 = self._iso_spin_amplitudes(q, gvuu, gvdd, gvss)
        m1, m2, m3 = self._fsp_masses
        cs = (a0 + a1) / np.sqrt(6.0) * 2 * self.g_ks_k_pi
        ct = (a0 - a1) / np.sqrt(6.0) * 2 * self.g_ks_k_pi
        return ct * self._kstar_propagator(s, m2, m3) + cs * self._kstar_propagator(
            t, m1, m3
        )
