from dataclasses import dataclass
from typing import Union, overload

import numpy as np
import numpy.typing as npt

from hazma import parameters

from .cross_sections import cross_section_x_x_to_p_v
from .utils import ComplexArray, RealArray, breit_wigner_fw
from .widths import width_v_to_v_p


@dataclass(frozen=True)
class FormFactorEtaPhi:
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
        return (
            cs
            * np.sum(
                self.amps
                * self.phase_factors
                * breit_wigner_fw(s, self.masses, self.widths, reshape=True),
                axis=1,
            )
            * np.sqrt(s)
        )

    @overload
    def form_factor(self, *, q: float, gvss: float) -> complex: ...

    @overload
    def form_factor(self, *, q: RealArray, gvss: float) -> ComplexArray: ...

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
        if hasattr(q, "__len__"):
            qq = 1e-3 * np.array(q)
        else:
            qq = 1e-3 * np.array([q])

        mask = qq > (parameters.eta_mass + parameters.phi_mass) * 1e-3
        ff = np.zeros_like(qq, dtype=np.complex128)
        ff[mask] = self.__form_factor(s=qq[mask] ** 2, gvss=gvss)

        if len(ff) == 1 and not hasattr(q, "__len__"):
            return ff[0]
        return ff

    def width(self, *, mv, gvss):
        ff = self.form_factor(q=mv, gvss=gvss)
        mvector = parameters.phi_mass
        mscalar = parameters.eta_mass
        return width_v_to_v_p(mv, ff, mvector, mscalar)

    def cross_section(self, *, cme, mx, mv, gvss, gamv):
        ff = self.form_factor(q=cme, gvss=gvss)
        mvector = parameters.phi_mass
        mscalar = parameters.eta_mass
        s = cme**2
        return cross_section_x_x_to_p_v(s, mx, mscalar, mvector, ff, mv, gamv)
