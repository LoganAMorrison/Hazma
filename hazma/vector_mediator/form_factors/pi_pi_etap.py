from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from hazma import parameters
from hazma.utils import kallen_lambda
from hazma.vector_mediator.form_factors.utils import (
    FPI_GEV,
    METAP_GEV,
    MPI_GEV,
    RealArray,
)

METAP = METAP_GEV * 1e3
MPI = MPI_GEV * 1e3


@dataclass
class FormFactorPiPiEtaP:

    masses: RealArray = np.array([0.77549, 1.54, 1.76, 2.11])
    widths: RealArray = np.array([0.1494, 0.356, 0.113, 0.176])
    amps: RealArray = np.array([1.0, 0.0, 0.0, 0.02])
    phases: RealArray = np.array([0, np.pi, np.pi, np.pi])

    def __bw0(self, s):
        m0 = self.masses[0]
        w0 = self.widths[0]
        w = (
            w0
            * m0**2
            / s
            * ((s - 4.0 * MPI_GEV**2) / (m0**2 - 4.0 * MPI_GEV**2)) ** 1.5
        )
        return m0**2 / (m0**2 - s - 1j * np.sqrt(s) * w)

    def __bw(self, s):
        w = self.widths * s / self.masses**2
        bw = self.masses**2 / (self.masses**2 - s - 1j * np.sqrt(s) * w)
        bw[0] = self.__bw0(s)
        return bw

    def form_factor(self, cme, s, gvuu, gvdd):
        """
        Compute the form factor for a vector decaying into two charged pions and
        an eta-prime.
        """
        pre = np.sqrt(2.0) / (4.0 * np.sqrt(3.0) * np.pi**2 * FPI_GEV**3)
        ci1 = gvuu - gvdd

        amps = self.amps * np.exp(1j * self.phases)
        amps /= np.sum(amps)

        return pre * ci1 * self.__bw0(s) * np.sum(amps * self.__bw(cme**2))

    def __integrated_form_factor(
        self, *, cme: float, gvuu: float, gvdd: float
    ) -> float:
        """
        Compute the form factor for a vector decaying into two charged pions and
        an eta-prime integrated over the three-body phase-space.
        """
        mpi = MPI_GEV
        metap = METAP_GEV
        if cme < 2 * mpi + metap:
            return 0.0

        jac = 1 / (128.0 * np.pi**3 * cme**2)

        def integrand(s):
            f2 = np.abs(self.form_factor(cme, s, gvuu, gvdd)) ** 2
            k1 = kallen_lambda(s, cme**2, metap**2)
            k2 = kallen_lambda(s, mpi**2, mpi**2)
            return (k1 * k2) ** 1.5 * f2 / (72 * s**2)

        lb = (2 * mpi) ** 2
        ub = (cme - metap) ** 2
        return jac * quad(integrand, lb, ub)[0]

    def integrated_form_factor(self, *, cme: float, gvuu: float, gvdd: float) -> float:
        """
        Compute the form factor for a vector decaying into two charged pions and
        an eta-prime integrated over the three-body phase-space.
        """
        cme_gev = cme * 1e-3
        integral = self.__integrated_form_factor(cme=cme_gev, gvuu=gvuu, gvdd=gvdd)
        return integral * 1e6

    def width(self, *, mv: float, gvuu: float, gvdd: float) -> float:
        if mv < 2 * MPI + METAP:
            return 0.0
        integral = self.integrated_form_factor(cme=mv, gvuu=gvuu, gvdd=gvdd)
        return integral / (2 * mv)

    def energy_distributions(self, cme, gvuu, gvdd, nbins):
        def edist_pi(e):
            s = cme**2 + MPI**2 - 2 * e * cme
            if s <= (MPI + METAP) ** 2 or s >= (cme - MPI) ** 2:
                return 0.0
            k1 = kallen_lambda(s, MPI**2, cme**2)
            k2 = kallen_lambda(s, MPI**2, METAP**2)
            return (k1 * k2) ** 1.5 / (s**2)

        def edist_eta(e):
            s = cme**2 + METAP**2 - 2 * e * cme
            if s <= 4 * MPI**2 or s >= (cme - METAP) ** 2:
                return 0.0
            k1 = kallen_lambda(s, MPI**2, MPI**2)
            k2 = kallen_lambda(s, METAP**2, cme**2)
            return (k1 * k2) ** 1.5 / (s**2)

        elow = MPI * (1 + 1e-10)
        ehigh = (cme**2 + MPI**2 - (MPI + METAP) ** 2) / (2 * cme) * (1 - 1e-10)
        norm = quad(edist_pi, elow, ehigh)[0]
        es_pi = np.linspace(elow, ehigh, nbins)
        dist_pi = [edist_pi(e) / norm for e in es_pi]

        elow = METAP * (1 + 1e-10)
        ehigh = (cme**2 + METAP**2 - 4 * MPI**2) / (2 * cme) * (1 - 1e-10)
        norm = quad(edist_eta, elow, ehigh)[0]
        es_eta = np.linspace(elow, ehigh, nbins)
        dist_eta = [edist_eta(e) / norm for e in es_eta]

        return [(dist_pi, es_pi), (dist_pi, es_pi), (dist_eta, es_eta)]
