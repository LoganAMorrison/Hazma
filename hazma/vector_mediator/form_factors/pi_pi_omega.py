from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from hazma.utils import kallen_lambda
from hazma.vector_mediator.form_factors.utils import (
    MPI_GEV,
    MOMEGA_GEV,
    RealArray,
)


@dataclass
class FormFactorPiPiOmega:

    masses: RealArray = np.array([0.783, 1.420, 1.6608543573197])
    widths: RealArray = np.array([0.00849, 0.315, 0.3982595005228462])
    amps: RealArray = np.array([0.0, 0.0, 2.728870588760009])
    phases: RealArray = np.array([0.0, np.pi, 0.0])

    def form_factor(self, cme, gvuu, gvdd):
        """
        Compute the form factor for a vector decaying into two charged pions and
        an eta-prime.
        """
        ci0 = 3 * (gvuu + gvdd)
        return ci0 * np.sum(
            self.amps
            * np.exp(1j * self.phases)
            * self.masses ** 2
            / ((self.masses ** 2 - cme ** 2) - 1j * cme * self.widths)
        )

    def integrated_form_factor(
        self, cme: float, gvuu: float, gvdd: float, imode
    ) -> float:
        """
        Compute the form factor for a vector decaying into two charged pions and
        an eta-prime integrated over the three-body phase-space.
        """
        mup = MPI_GEV / cme
        muo = MOMEGA_GEV / cme
        jac = cme ** 2 / (1536.0 * np.pi ** 3 * muo ** 2)
        f2 = np.abs(self.form_factor(cme, gvuu, gvdd)) ** 2

        def integrand(z):
            k1 = kallen_lambda(z, mup ** 2, mup ** 2)
            k2 = kallen_lambda(1, z, muo ** 2)
            p = (1 + 10 * muo ** 2 + muo ** 4 - 2 * (1 + muo ** 2) * z + z ** 2) / z
            return p * np.sqrt(k1 * k2) * f2

        lb = (2 * mup) ** 2
        ub = (1.0 - muo) ** 2
        res = jac * quad(integrand, lb, ub)[0]

        if imode == 0:
            return res / 2.0
        return res
