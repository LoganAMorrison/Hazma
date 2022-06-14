from dataclasses import dataclass
from typing import Union, overload

import numpy as np
import numpy.typing as npt

from hazma import parameters

from ._utils import ComplexArray, RealArray, breit_wigner_fw
from ._base import VectorFormFactorPV


@dataclass
class VectorFormFactorEtaPhi(VectorFormFactorPV):
    """
    Class for storing the parameters needed to compute the form factor for
    V-eta-omega. See arXiv:1911.11147 for details on the default values.
    """

    masses: npt.NDArray[np.float64] = np.array([1.67, 2.14])
    widths: npt.NDArray[np.float64] = np.array([0.122, 0.0435])
    amps: npt.NDArray[np.float64] = np.array([0.175, 0.00409])
    phase_factors: npt.NDArray[np.complex128] = np.exp(1j * np.array([0.0, 2.19]))

    def __form_factor(
        self,
        *,
        s: npt.NDArray[np.float64],
        gvss: float,
    ):
        """
        Compute the V-eta-phi form-factor.

        Uses the parameterization from arXiv:1911.11147.

        Parameters
        ----------
        s: float or np.ndarray
            Square of the center of mass energy(ies) in GeV.
        gvss: float
            Coupling of vector to strang-quarks.
        fit: FormFactorEtaOmegaFit, optional
            Fit parameters.
        """
        cs = -3.0 * gvss
        return cs * np.sum(
            self.amps
            * self.phase_factors
            * breit_wigner_fw(s, self.masses, self.widths, reshape=True),
            axis=1,
        )

    @overload
    def form_factor(self, *, q: float, gvss: float) -> complex:
        ...

    @overload
    def form_factor(self, *, q: RealArray, gvss: float) -> ComplexArray:
        ...

    def form_factor(
        self, *, q: Union[float, RealArray], gvss: float
    ) -> Union[complex, ComplexArray]:
        """
        Compute the V-eta-phi form factor.

        Parameters
        ----------
        q: Union[float,npt.NDArray[np.float64]
            Center-of-mass energy in MeV.

        Returns
        -------
        ff: Union[complex,npt.NDArray[np.complex128]]
            Form factor from V-eta-phi.
        """
        me = parameters.eta_mass
        mf = parameters.phi_mass

        single = np.isscalar(q)
        qq = np.atleast_1d(q).astype(np.float64) * 1e-3
        mask = qq > 1e-3 * (me + mf)
        ff = np.zeros_like(qq, dtype=np.complex128)

        ff[mask] = self.__form_factor(s=qq[mask] ** 2, gvss=gvss)

        if single:
            return ff[0]

        return ff * 1e-3

    def width(self, mv, gvss):
        fsp_masses = parameters.eta_mass, parameters.phi_mass
        return self._width(mv=mv, fsp_masses=fsp_masses, gvss=gvss)
