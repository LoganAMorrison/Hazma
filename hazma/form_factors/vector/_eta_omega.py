from dataclasses import dataclass, field
from typing import Union, overload, Tuple

import numpy as np
import numpy.typing as npt

from hazma import parameters

from ._utils import ComplexArray, RealArray, breit_wigner_fw
from ._base import VectorFormFactorPV

META = parameters.eta_mass
MOMEGA = parameters.omega_mass


@dataclass
class VectorFormFactorEtaOmega(VectorFormFactorPV):
    """
    Class for storing the parameters needed to compute the form factor for
    V-eta-omega. See arXiv:1911.11147 for details on the default values.
    """

    fsp_masses: Tuple[float, float] = field(init=False, default=(META, MOMEGA))

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
        return ci0 * np.sum(
            self.amps
            * self.phase_factors
            * breit_wigner_fw(s, self.masses, self.widths, reshape=True),
            axis=1,
        )

    @overload
    def form_factor(self, *, q: float, gvuu: float, gvdd: float) -> complex:
        ...

    @overload
    def form_factor(self, *, q: RealArray, gvuu: float, gvdd: float) -> ComplexArray:
        ...

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
        me = parameters.eta_mass
        mw = parameters.omega_mass

        single = np.isscalar(q)
        qq = np.atleast_1d(q).astype(np.float64) * 1e-3
        mask = qq > 1e-3 * (me + mw)
        ff = np.zeros_like(qq, dtype=np.complex128)

        ff[mask] = self.__form_factor(s=qq[mask] ** 2, gvuu=gvuu, gvdd=gvdd)

        if single:
            return ff[0]

        return ff * 1e-3

    def width(self, mv, gvuu, gvdd):
        return self._width(mv=mv, gvuu=gvuu, gvdd=gvdd)
