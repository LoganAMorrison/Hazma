from dataclasses import dataclass
from typing import Union

import numpy as np
import numpy.typing as npt

from hazma.vector_mediator.form_factors.utils import (
    breit_wigner_fw,
)


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

    def form_factor(
        self,
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
