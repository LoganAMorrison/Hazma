from dataclasses import dataclass
from typing import Union, overload

import numpy as np
import numpy.typing as npt

from hazma import parameters

from .cross_sections import cross_section_x_x_to_p_v
from .utils import ComplexArray, RealArray, breit_wigner_fw
from .widths import width_v_to_v_p


@dataclass(frozen=True)
class FormFactorEtaOmega:
    """
    Class for storing the parameters needed to compute the form factor for
    V-eta-omega. See arXiv:1911.11147 for details on the default values.
    """

    # w', w''' parameters
    masses: npt.NDArray[np.float64] = np.array([1.43, 1.67])
    widths: npt.NDArray[np.float64] = np.array([0.215, 0.113])
    amps: npt.NDArray[np.float64] = np.array([0.0862, 0.0648])
    phase_factors: npt.NDArray[np.complex128] = np.exp(1j * np.array([0.0, np.pi]))

    def __form_factor(
        self,
        *,
        s: Union[float, npt.NDArray[np.float64]],
        gvuu: float,
        gvdd: float,
    ):
        """
        Compute the V-eta-omega form-factor.

        Uses the parameterization from arXiv:1911.11147.

        Parameters
        ----------
        s: float or np.ndarray
            Square of the center of mass energy(ies).
        gvuu: float
            Coupling of vector to up-quarks.
        gvdd: float
            Coupling of vector to down-quarks.
        fit: FormFactorEtaOmegaFit, optional
            Fit parameters.
        """
        # NOTE: This is a rescaled-version of the form factor defined in
        # arXiv:2201.01788. (multipled by energy to make it unitless)
        ci0 = 3 * (gvuu + gvdd)
        return (
            ci0
            * np.sum(
                self.amps
                * self.phase_factors
                * breit_wigner_fw(s, self.masses, self.widths, reshape=True),
                axis=1,
            )
            * np.sqrt(s)
        )

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
        Compute the V-eta-omega form factor.

        Parameters
        ----------
        q: Union[float,npt.NDArray[np.float64]
            Center-of-mass energy in MeV.

        Returns
        -------
        ff: Union[complex,npt.NDArray[np.complex128]]
            Form factor from V-eta-omega.
        """
        if hasattr(q, "__len__"):
            qq = 1e-3 * np.array(q)
        else:
            qq = 1e-3 * np.array([q])

        mask = qq > (parameters.eta_mass + parameters.omega_mass) * 1e-3

        ff = np.zeros_like(qq, dtype=np.complex128)
        ff[mask] = self.__form_factor(s=qq[mask] ** 2, gvuu=gvuu, gvdd=gvdd)

        if len(ff) == 1 and not hasattr(q, "__len__"):
            return ff[0]
        return ff

    def width(self, *, mv, gvuu, gvdd):
        ff = self.form_factor(q=mv, gvuu=gvuu, gvdd=gvdd)
        mvector = parameters.omega_mass
        mscalar = parameters.eta_mass
        return width_v_to_v_p(mv, ff, mvector, mscalar)

    def cross_section(self, *, cme, mx, mv, gvuu, gvdd, gamv):
        ff = self.form_factor(q=cme, gvuu=gvuu, gvdd=gvdd)
        mvector = parameters.omega_mass
        mscalar = parameters.eta_mass
        s = cme**2
        return cross_section_x_x_to_p_v(s, mx, mscalar, mvector, ff, mv, gamv)
