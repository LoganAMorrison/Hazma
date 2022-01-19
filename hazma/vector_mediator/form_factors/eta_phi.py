from dataclasses import dataclass
from typing import Union

import numpy as np
import numpy.typing as npt

from hazma.vector_mediator.form_factors.utils import (
    breit_wigner_fw,
)


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

    def form_factor(
        self,
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
