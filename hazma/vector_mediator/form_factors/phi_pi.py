from dataclasses import dataclass
from typing import Union

import numpy as np
import numpy.typing as npt

from hazma.vector_mediator.form_factors.utils import MPI0_GEV
from hazma.utils import RealArray


ComplexArray = npt.NDArray[np.complex128]

# old fit values
# amp = [0.045, 0.0315, 0.0]
# phase = [180.0, 0.0, 180.0]

# Uncertainties
# amp1=0.0825858193110437
# amp2=0.004248886307513855
# amp3=0.0
# phase1=0.0
# phase2=16.826357320477726
# phase3=0.0


@dataclass(frozen=True)
class FormFactorPhiPi0:
    """
    Class for storing the parameters needed to compute the form factor for
    V-phi-pi. See arXiv:1911.11147 for details on the default values.
    """

    amps: RealArray = np.array([0.177522453644825, 0.023840592398187477, 0.0])
    phases: RealArray = np.array([0.0, 123.82008351626034, 0.0]) * np.pi / 180.0
    rho_masses: RealArray = np.array([0.77526, 1.593, 1.909])
    rho_widths: RealArray = np.array([0.1491, 0.203, 0.048])
    br4pi: RealArray = np.array([0.0, 0.33, 0.0])

    def form_factor(
        self, s: Union[RealArray, float], gvuu: float, gvdd: float
    ) -> Union[ComplexArray, complex]:
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

        ms = self.rho_masses
        amps = ci1 * self.amps * np.exp(1j * self.phases)
        brs = self.br4pi
        rho_ws = self.rho_widths

        if hasattr(s, "__len__"):
            s = np.expand_dims(s, 0)
            ms = np.expand_dims(ms, -1)
            amps = np.expand_dims(amps, -1)
            brs = np.expand_dims(brs, -1)
            rho_ws = np.expand_dims(rho_ws, -1)

        ws = rho_ws * (
            1.0
            - brs
            + brs
            * ms**2
            / s
            * ((s - 16.0 * MPI0_GEV**2) / (ms**2 - 16.0 * MPI0_GEV**2)) ** 1.5
        )
        # NOTE: This is a rescaled-version of the form factor defined
        # in arXiv:2201.01788. (multipled by s to make it unitless)
        return np.sqrt(s) * np.sum(
            amps * ms**2 / (ms**2 - s - 1j * np.sqrt(s) * ws),
            axis=0,
        )
