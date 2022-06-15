from dataclasses import dataclass, field
from typing import Union, overload, Tuple

import numpy as np

from hazma import parameters

from ._utils import ComplexArray, RealArray
from ._base import VectorFormFactorPA
from ._alpha import alpha_em

MPI0 = parameters.neutral_pion_mass


@dataclass
class VectorFormFactorPiGamma(VectorFormFactorPA):
    fsp_masses: Tuple[float] = field(init=False, default=(MPI0,))

    fpi: float = 0.09266
    amplitude0: float = 0.007594981126020603
    amplitude_rho: float = 1.0
    amplitude_omega: float = 0.8846540224221084
    amplitude_phi: float = -0.06460651106718258

    mass_rho: float = 0.77526
    mass_omega: float = 0.78265
    mass_phi: float = 1.01946

    width_rho: float = 0.1491
    width_omega: float = 0.00849
    width_phi: float = 0.004247

    def __form_factor(
        self, *, q: RealArray, gvuu: float, gvdd: float, gvss: float
    ) -> ComplexArray:
        """
        Compute the form factor for V-gamma-pi at given squared center of mass
        energ(ies).

        Parameters
        ----------
        s: NDArray[float]
            Array of squared center-of-mass energies or a single value.
        gvuu : float
            Coupling of vector mediator to the up quark.
        gvdd : float
            Coupling of vector mediator to the down quark.
        gvss : float
            Coupling of vector mediator to the strange quark.

        Returns
        -------
        ff: NDArray[complex]
            The form factors.
        """

        s = q**2

        def amp(c, m, w):
            return c / (s - m**2 + 1j * q * w)

        ci0 = 3.0 * (gvuu + gvdd)
        ci1 = gvuu - gvdd
        cs = -3.0 * gvss
        cd = 2 * gvuu + gvdd

        fpi = self.fpi

        a0 = self.amplitude0
        ar = self.amplitude_rho
        aw = self.amplitude_omega
        af = self.amplitude_phi

        amp_0 = a0 * 4.0 * np.sqrt(2) * s / (3.0 * fpi)
        amp_r = ar * amp(ci1, self.mass_rho, self.width_rho)
        amp_w = aw * amp(ci0, self.mass_omega, self.width_omega)
        amp_f = af * amp(cs, self.mass_phi, self.width_phi)

        form = amp_0 * (amp_r + amp_w + amp_f) - cd / (4.0 * np.pi**2 * fpi)
        return np.sqrt(4 * np.pi * alpha_em(s)) * form

    @overload
    def form_factor(self, *, q: float, gvuu, gvdd, gvss) -> complex:
        ...

    @overload
    def form_factor(self, *, q: RealArray, gvuu, gvdd, gvss) -> ComplexArray:
        ...

    def form_factor(
        self, *, q: Union[float, RealArray], gvuu, gvdd, gvss
    ) -> Union[complex, ComplexArray]:
        """
        Compute the pi-gamma-V form factor.

        Parameters
        ----------
        s: Union[float,npt.NDArray[np.float64]
            Square of the center-of-mass energy in MeV.

        Returns
        -------
        ff: Union[complex,npt.NDArray[np.complex128]]
            Form factor from pi-gamma-V.
        """
        single = np.isscalar(q)
        qq = np.atleast_1d(q).astype(np.float64) * 1e-3
        mask = qq > 1e-3 * sum(self.fsp_masses)
        ff = np.zeros_like(qq, dtype=np.complex128)

        ff[mask] = self.__form_factor(
            q=qq[mask],
            gvuu=gvuu,
            gvdd=gvdd,
            gvss=gvss,
        )

        if single:
            return ff[0]

        return ff * 1e-3

    def width(
        self, mv: Union[float, RealArray], gvuu, gvdd, gvss
    ) -> Union[complex, ComplexArray]:
        return self._width(mv=mv, gvuu=gvuu, gvdd=gvdd, gvss=gvss)
