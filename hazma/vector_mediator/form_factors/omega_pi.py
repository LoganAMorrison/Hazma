from dataclasses import dataclass
from typing import Union, overload

import numpy as np
import numpy.typing as npt

from hazma import parameters
from hazma.utils import kallen_lambda

from .cross_sections import cross_section_x_x_to_p_v
from .utils import MOMEGA_GEV, MPI0_GEV, ComplexArray, RealArray
from .widths import width_v_to_v_p


@dataclass(frozen=True)
class FormFactorOmegaPi0:
    """
    Class for computing the form factor for V-omega-pi0. See arXiv:1303.5198 for details
    on the default fit values.
    """

    g_rho_omega_pi: float = 15.9  # units of GeV^-1
    amps: npt.NDArray[np.float64] = np.array([1.0, 0.175, 0.014])
    phases: npt.NDArray[np.float64] = np.array([0.0, 124.0, -63.0]) * np.pi / 180.0
    rho_masses: npt.NDArray[np.float64] = np.array([0.77526, 1.510, 1.720])
    rho_widths: npt.NDArray[np.float64] = np.array([0.1491, 0.44, 0.25])
    frho: float = 5.06325

    def _rho_widths(self, s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        q = np.sqrt(s)
        widths = np.array(
            [
                np.full_like(s, self.rho_widths[i], dtype=np.float64)
                for i in range(len(self.rho_widths))
            ]
        )

        mu2p = MPI0_GEV**2 / s
        mu2w = MOMEGA_GEV**2 / s
        mu2r = self.rho_masses[0] ** 2 / s
        p = 0.5 * q * np.sqrt(kallen_lambda(1.0, mu2p, mu2w))
        widths[0] = (
            widths[0] * mu2r * ((1.0 - 4.0 * mu2p) / (mu2r - 4.0 * mu2p)) ** 1.5
        ) + self.g_rho_omega_pi**2 * p**3 / (12.0 * np.pi)
        return widths

    def __form_factor(
        self, *, s: npt.NDArray[np.float64], gvuu: float, gvdd: float
    ) -> npt.NDArray[np.complex128]:
        """
        Compute the V-omega-pi form-factor.

        Uses the parameterization from arXiv:1303.5198.

        Parameters
        ----------
        s: float or np.ndarray
            Square of the center of mass energy(ies).
        gvuu: float
            Coupling of vector to up-quarks.
        gvdd: float
            Coupling of vector to down-quarks.

        Returns
        -------
        ff: float or np.ndarray
            Form factor value(s).
        """
        ci1 = gvuu - gvdd

        widths = self._rho_widths(s)
        masses = self.rho_masses[:, np.newaxis]
        phases = self.phases[:, np.newaxis]
        ss = s[np.newaxis, :]

        dens = masses**2 - ss - 1j * np.sqrt(ss) * widths
        amps = self.amps[:, np.newaxis] * np.exp(1j * phases) * masses**2 / dens
        return self.g_rho_omega_pi * ci1 / self.frho * np.sum(amps, axis=0) * np.sqrt(s)

    @overload
    def form_factor(self, *, q: float, gvuu: float, gvdd: float) -> complex: ...

    @overload
    def form_factor(
        self, *, q: RealArray, gvuu: float, gvdd: float
    ) -> ComplexArray: ...

    def form_factor(
        self, *, q: Union[float, RealArray], gvuu: float, gvdd: float
    ) -> Union[complex, ComplexArray]:
        """
        Compute the V-omega-pi form factor.

        Parameters
        ----------
        q: Union[float,npt.NDArray[np.float64]
            Center-of-mass energy in MeV.

        Returns
        -------
        ff: Union[complex,npt.NDArray[np.complex128]]
            Form factor from V-omega-pi.
        """
        if hasattr(q, "__len__"):
            qq = 1e-3 * np.array(q)
        else:
            qq = 1e-3 * np.array([q])

        mask = qq > (parameters.neutral_pion_mass + parameters.omega_mass) * 1e-3
        ff = np.zeros_like(qq, dtype=np.complex128)
        ff[mask] = self.__form_factor(s=qq[mask] ** 2, gvuu=gvuu, gvdd=gvdd)

        if len(ff) == 1 and not hasattr(q, "__len__"):
            return ff[0]
        return ff

    def width(self, mv, gvuu, gvdd):
        ff = self.form_factor(q=mv, gvuu=gvuu, gvdd=gvdd)
        mvector = parameters.omega_mass
        mscalar = parameters.neutral_pion_mass
        return width_v_to_v_p(mv, ff, mvector, mscalar)

    def cross_section(self, cme, mx, mv, gvuu, gvdd, gamv):
        ff = self.form_factor(q=cme, gvuu=gvuu, gvdd=gvdd)
        mvector = parameters.omega_mass
        mscalar = parameters.neutral_pion_mass
        s = cme**2
        return cross_section_x_x_to_p_v(s, mx, mscalar, mvector, ff, mv, gamv)
