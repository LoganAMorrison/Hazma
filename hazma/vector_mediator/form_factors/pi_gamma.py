from dataclasses import dataclass
from typing import Union, overload

import numpy as np

from hazma import parameters

from .cross_sections import width_to_cs
from .utils import MPI_GEV, ComplexArray, RealArray
from .widths import width_v_to_p_a


@dataclass
class FormFactorPiGamma:

    masses: RealArray = np.array([0.77526, 0.78284, 1.01952, 1.45, 1.70])
    widths: RealArray = np.array([0.1491, 0.00868, 0.00421, 0.40, 0.30])
    amps: RealArray = np.array([0.0426, 0.0434, 0.00303, 0.00523, 0.0])
    phases: RealArray = np.array([-12.7, 0.0, 158.0, 180.0, 0.0]) * np.pi / 180.0

    def __form_factor(
        self, s: RealArray, gvuu: float, gvdd: float, gvss: float
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

        ci0 = 3.0 * (gvuu + gvdd)
        ci1 = gvuu - gvdd
        cs = -3.0 * gvss

        c_rho_om_phi = np.array([ci1, ci0, cs, ci0, ci0])
        ss = s[:, np.newaxis]

        q = np.sqrt(ss)

        dens = self.masses**2 - ss - 1j * q * self.widths

        dens[:, 0:1] = (
            self.masses[0] ** 2
            - ss
            - 1j
            * q
            * (
                self.widths[0]
                * self.masses[0] ** 2
                / ss
                * np.clip(
                    (ss - 4 * MPI_GEV**2) / (self.masses[0] ** 2 - 4 * MPI_GEV**2),
                    0.0,
                    np.inf,
                )
                ** 1.5
            )
        )

        return np.sum(
            c_rho_om_phi
            * self.amps
            * self.masses**2
            * np.exp(1j * self.phases)
            / dens
            * np.sqrt(ss),
            axis=1,
        )

    @overload
    def form_factor(self, *, q: float, gvuu, gvdd, gvss) -> complex: ...

    @overload
    def form_factor(self, *, q: RealArray, gvuu, gvdd, gvss) -> ComplexArray: ...

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
        if hasattr(q, "__len__"):
            qq = 1e-3 * np.array(q)
        else:
            qq = 1e-3 * np.array([q])

        mask = qq > (parameters.neutral_pion_mass) * 1e-3
        ff = np.zeros_like(qq, dtype=np.complex128)
        ff[mask] = self.__form_factor(
            qq[mask] ** 2,
            gvuu,
            gvdd,
            gvss,
        )

        if len(ff) == 1 and not hasattr(q, "__len__"):
            return ff[0]
        return ff

    def width(self, *, mv, gvuu, gvdd, gvss):
        mp = parameters.neutral_pion_mass
        ff = self.form_factor(q=mv, gvuu=gvuu, gvdd=gvdd, gvss=gvss)
        return width_v_to_p_a(mv, mp, ff)

    def cross_section(self, *, cme, mx, mv, gvuu, gvdd, gvss, gamv):
        rescale = width_to_cs(cme=cme, mx=mx, mv=mv, wv=gamv)
        return rescale * self.width(mv=cme, gvuu=gvuu, gvdd=gvdd, gvss=gvss)
