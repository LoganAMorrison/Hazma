from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from hazma.utils import kallen_lambda
from hazma.vector_mediator.form_factors.utils import MPI0_GEV, MOMEGA_GEV


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

        mu2p = MPI0_GEV ** 2 / s
        mu2w = MOMEGA_GEV ** 2 / s
        mu2r = self.rho_masses[0] ** 2 / s
        p = 0.5 * q * np.sqrt(kallen_lambda(1.0, mu2p, mu2w))
        widths[0] = (
            widths[0] * mu2r * ((1.0 - 4.0 * mu2p) / (mu2r - 4.0 * mu2p)) ** 1.5
        ) + self.g_rho_omega_pi ** 2 * p ** 3 / (12.0 * np.pi)
        return widths

    def form_factor(
        self, s: npt.NDArray[np.float64], gvuu: float, gvdd: float
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

        dens = masses ** 2 - ss - 1j * np.sqrt(ss) * widths
        amps = self.amps[:, np.newaxis] * np.exp(1j * phases) * masses ** 2 / dens
        return self.g_rho_omega_pi * ci1 / self.frho * np.sum(amps, axis=0) * np.sqrt(s)
