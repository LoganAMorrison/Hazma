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

from .cross_sections import width_to_cs
from hazma.utils import kallen_lambda
from hazma.vector_mediator.form_factors.utils import (
    FPI_GEV,
    META_GEV,
    MPI_GEV,
    RealArray,
)

META = META_GEV * 1e3
MPI = MPI_GEV * 1e3


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
        an eta.

        Parameters
        ----------
        q2:
            Square of the center-of-mass energy in GeV.
        """
        pre = 1.0 / (4.0 * np.sqrt(3.0) * np.pi**2 * FPI_GEV**3)
        ci1 = gvuu - gvdd

        amps = self.amps * np.exp(1j * self.phases)
        amps /= np.sum(amps)

        return pre * ci1 * self.__bw0(s) * np.sum(amps * self.__bw(cme**2))

    def __integrated_form_factor(
        self, *, cme: float, gvuu: float, gvdd: float
    ) -> float:
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
        if cme < 2 * mpi + meta:
            return 0.0

        jac = 1 / (128.0 * np.pi**3 * cme**2)

        def integrand(s):
            f2 = np.abs(self.form_factor(cme, s, gvuu, gvdd)) ** 2
            k1 = kallen_lambda(s, cme**2, meta**2)
            k2 = kallen_lambda(s, mpi**2, mpi**2)
            return (k1 * k2) ** 1.5 * f2 / (72 * s**2)

        lb = (2 * mpi) ** 2
        ub = (cme - meta) ** 2
        return jac * quad(integrand, lb, ub)[0]

    def integrated_form_factor(self, *, cme: float, gvuu: float, gvdd: float) -> float:
        """
        Compute the form factor for a vector decaying into two charged pions and
        a neutral pion integrated over the three-body phase-space.

        Parameters
        ----------
        q2:
            Square of the center-of-mass energy in MeV.
        """
        cme_gev = cme * 1e-3
        integral = self.__integrated_form_factor(cme=cme_gev, gvuu=gvuu, gvdd=gvdd)
        return integral * 1e6

    def width(self, *, mv: float, gvuu: float, gvdd: float) -> float:
        if mv < 2 * MPI + META:
            return 0.0
        integral = self.integrated_form_factor(cme=mv, gvuu=gvuu, gvdd=gvdd)
        return integral / (2 * mv)

    def cross_section(
        self,
        *,
        cme,
        mx: float,
        mv: float,
        gvuu: float,
        gvdd: float,
        gamv: float,
    ):
        rescale = width_to_cs(cme=cme, mx=mx, mv=mv, wv=gamv)
        return rescale * self.width(mv=cme, gvuu=gvuu, gvdd=gvdd)

    def energy_distributions(self, cme, gvuu, gvdd, nbins):
        if cme < 2 * MPI + META:
            return [([], []), ([], []), ([], [])]

        def edist(e, m1, m2, m3):
            s = cme**2 + m1**2 - 2 * cme * e
            if s <= (m2 + m3) ** 2 or s >= (cme - m1) ** 2:
                return 0.0
            k1 = kallen_lambda(s, m1**2, cme**2)
            k2 = kallen_lambda(s, m2**2, m3**2)
            return (k1 * k2) ** 1.5 / (s**2)

        def ebounds(m1, m2, m3):
            return m1, (cme**2 + m1**2 - (m2 + m3) ** 2) / (2 * cme)

        def make_dist(m1, m2, m3):
            elow, ehigh = ebounds(m1, m2, m3)
            edges = np.linspace(elow, ehigh, nbins + 1)
            es = 0.5 * (edges[1:] + edges[:-1])
            norm = quad(lambda e: edist(e, m1, m2, m3), elow, ehigh)[0]
            dist = [edist(e, m1, m2, m3) / norm for e in es]
            return dist, es

        dist_pi, es_pi = make_dist(MPI, MPI, META)
        dist_eta, es_eta = make_dist(META, MPI, MPI)

        return [(dist_pi, es_pi), (dist_pi, es_pi), (dist_eta, es_eta)]
