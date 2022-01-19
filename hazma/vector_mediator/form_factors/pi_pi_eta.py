"""
F_{eta,pi,pi} = (1/Z) * BW(s, 0) [
    a0*e^{i*p0}BW(q^2,0) +
    a1*e^{i*p1}BW(q^2,1) +
    a2*e^{i*p2}BW(q^2,2)
]

Z = a0*e^{i*p0} + a1*e^{i*p1} + a2*e^{i*p2}
"""


from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from hazma.utils import kallen_lambda
from hazma.vector_mediator.form_factors.utils import (
    MPI_GEV,
    META_GEV,
    FPI_GEV,
    RealArray,
)


@dataclass
class FormFactorPiPiEta:

    masses: RealArray = np.array([0.77549, 1.54, 1.76, 2.15])
    widths: RealArray = np.array([0.1494, 0.356, 0.113, 0.32])
    amps: RealArray = np.array([1.0, 0.326, 0.0115, 0.0])
    phases: RealArray = np.array([0, 3.14, 3.14, 0.0])

    def __bw0(self, s):
        m0 = self.masses[0]
        w0 = self.widths[0]
        w = (
            w0
            * m0 ** 2
            / s
            * ((s - 4.0 * MPI_GEV ** 2) / (m0 ** 2 - 4.0 * MPI_GEV ** 2)) ** 1.5
        )
        return m0 ** 2 / (m0 ** 2 - s - 1j * np.sqrt(s) * w)

    def __bw(self, s):
        w = self.widths * s / self.masses ** 2
        bw = self.masses ** 2 / (self.masses ** 2 - s - 1j * np.sqrt(s) * w)
        bw[0] = self.__bw0(s)
        return bw

    def form_factor(self, cme, s, gvuu, gvdd):
        """
        Compute the form factor for a vector decaying into two charged pions and
        an eta.

        Parameters
        ----------
        q2:
            Square of the center-of-mass energy in GeV.
        """
        pre = 1.0 / (4.0 * np.sqrt(3.0) * np.pi ** 2 * FPI_GEV ** 3)
        ci1 = gvuu - gvdd

        amps = self.amps * np.exp(1j * self.phases)
        amps /= np.sum(amps)

        return pre * ci1 * self.__bw0(s) * np.sum(amps * self.__bw(cme ** 2))

    def integrated_form_factor(self, cme: float, gvuu: float, gvdd: float) -> float:
        """
        Compute the form factor for a vector decaying into two charged pions and
        a neutral pion integrated over the three-body phase-space.

        Parameters
        ----------
        q2:
            Square of the center-of-mass energy in GeV.
        """
        mpi = MPI_GEV
        meta = META_GEV
        jac = 1 / (128.0 * np.pi ** 3 * cme ** 2)

        def integrand(s):
            f2 = np.abs(self.form_factor(cme, s, gvuu, gvdd)) ** 2
            k1 = kallen_lambda(s, cme ** 2, meta ** 2)
            k2 = kallen_lambda(s, mpi ** 2, mpi ** 2)
            return (k1 * k2) ** 1.5 * f2 / (72 * s ** 2)

        lb = (2 * mpi) ** 2
        ub = (cme - meta) ** 2
        return jac * quad(integrand, lb, ub)[0]
