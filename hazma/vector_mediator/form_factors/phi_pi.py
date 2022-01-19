from dataclasses import dataclass
from typing import Union

import numpy as np
import numpy.typing as npt

from hazma.vector_mediator.form_factors.utils import MPI0_GEV


@dataclass(frozen=True)
class FormFactorPhiPi0:
    """
    Class for storing the parameters needed to compute the form factor for
    V-phi-pi. See arXiv:1911.11147 for details on the default values.
    """

    amps: npt.NDArray[np.float64] = np.array([0.045, 0.0315, 0.0])
    phases: npt.NDArray[np.float64] = np.array([np.pi, 0.0, np.pi])
    rho_masses: npt.NDArray[np.float64] = np.array([0.77526, 1.593, 1.909])
    rho_widths: npt.NDArray[np.float64] = np.array([0.1491, 0.203, 0.048])
    br4pi: npt.NDArray[np.float64] = np.array([0.0, 0.33, 0.0])

    def form_factor(
        self, s: npt.NDArray[np.float64], gvuu: float, gvdd: float
    ) -> npt.NDArray[np.complex128]:
        """
        Compute the V-phi-pi form-factor.

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

        ss = s[np.newaxis, :]
        ms = self.rho_masses[:, np.newaxis]
        amps = self.amps[:, np.newaxis] * np.exp(1j * self.phases[:, np.newaxis])
        brs = self.br4pi[:, np.newaxis]

        ws = self.rho_widths[:, np.newaxis] * (
            1.0
            - brs
            + brs
            * ms ** 2
            / ss
            * ((ss - 16.0 * MPI0_GEV ** 2) / (ms ** 2 - 16.0 * MPI0_GEV ** 2) ** 1.5)
        )
        # NOTE: This is a rescaled-version of the form factor defined
        # in arXiv:2201.01788. (multipled by s to make it unitless)
        return (
            ci1
            * np.sum(
                amps * ms ** 2 / (ms ** 2 - ss - 1j * np.sqrt(ss) * ws),
                axis=0,
            )
            * np.sqrt(ss)
        )
