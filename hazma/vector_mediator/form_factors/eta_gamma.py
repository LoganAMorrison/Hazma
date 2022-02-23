"""
Module for computing the form factor V-eta-gamma.
"""
from dataclasses import dataclass

import numpy as np

from hazma.vector_mediator.form_factors.utils import MPI_GEV
from hazma.vector_mediator.form_factors.utils import RealArray
from hazma.vector_mediator.form_factors.utils import ComplexArray


@dataclass(slots=True)
class FormFactorEtaGamma:
    masses: RealArray = np.array([0.77526, 0.78284, 1.01952, 1.465, 1.70])
    widths: RealArray = np.array([0.1491, 0.00868, 0.00421, 0.40, 0.30])
    amps: RealArray = np.array([0.0861, 0.00824, 0.0158, 0.0147, 0.0])
    phases: RealArray = np.array([0.0, 11.3, 170.0, 61.0, 0.0]) * np.pi / 180.0

    def form_factor(
        self, s: RealArray, gvuu: float, gvdd: float, gvss: float
    ) -> ComplexArray:
        """
        Compute the form factor for V-eta-gamma at given squared center of mass
        energ(ies).

        Parameters
        ----------
        s: Union[float, np.ndarray]
            Array of squared center-of-mass energies or a single value.
        gvuu : float
            Coupling of vector mediator to the up quark.
        gvdd : float
            Coupling of vector mediator to the down quark.
        gvss : float
            Coupling of vector mediator to the strange quark.

        Returns
        -------
        ff: Union[float, np.ndarray]
            The form factors.
        """
        ci0 = 3.0 * (gvuu + gvdd)
        ci1 = gvuu - gvdd
        cs = -3.0 * gvss

        c_rho_om_phi = np.array([ci1, ci0, cs, ci1, cs])

        ss = s[:, np.newaxis]
        q = np.sqrt(ss)
        dens = self.masses ** 2 - ss - 1j * q * self.widths
        dens[:, 0:1] = (
            self.masses[0] ** 2
            - ss
            - 1j
            * q
            * (
                self.widths[0]
                * self.masses[0] ** 2
                / ss
                * ((ss - 4 * MPI_GEV ** 2) / (self.masses[0] ** 2 - 4 * MPI_GEV ** 2))
                ** 1.5
            )
        )

        return np.sum(
            c_rho_om_phi
            * self.amps
            * self.masses ** 2
            * np.exp(1j * self.phases)
            / dens
            * np.sqrt(ss),
            axis=1,
        )
