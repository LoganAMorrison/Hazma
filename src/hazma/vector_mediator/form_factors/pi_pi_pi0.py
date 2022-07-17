from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from hazma import parameters
from hazma.phase_space import Rambo, PhaseSpaceDistribution1D
from hazma.utils import lnorm_sqr
from .cross_sections import width_to_cs
from hazma.vector_mediator.form_factors.utils import (
    MPI0_GEV,
    MPI_GEV,
    RealArray,
    breit_wigner_fw,
)


@dataclass
class FormFactorPiPiPi0:

    masses: RealArray = np.array([0.7824, 1.01924, 1.375, 1.631])
    widths: RealArray = np.array([0.00869, 0.00414, 0.250, 0.245])
    couplings: RealArray = np.array([18.20, -0.87, -0.77, -1.12])
    masses_rho_i0: RealArray = np.array([0.77609, 1.465, 1.7])
    widths_rho_i0: RealArray = np.array([0.14446, 0.31, 0.235])
    couplings_rho_i0: RealArray = np.array([0.0, -0.72, -0.59])

    masses_rho_i1: RealArray = np.array([0.77609, 1.7])
    widths_rho_i1: RealArray = np.array([0.14446, 0.26])
    mass_omega_i1: float = 0.78259
    width_omega_i1: float = 0.00849
    coupling_omega_pre: float = 3.768
    coupling_omega_pi_pi: float = 0.185
    sigma: float = -0.1

    def __gamma_rho(self, s, mass, width, mj, mk):
        # p-wave width
        return (
            width
            * mass**2
            / s
            * ((s - (mj + mk) ** 2) / (mass**2 - (mj + mk) ** 2)) ** 1.5
        )

    def __bw_rho(self, Qi2, mRho, gRho, mj, mk):
        # Breit-Wigner for rhos
        return mRho**2 / (
            Qi2
            - mRho**2
            + 1j * np.sqrt(Qi2) * self.__gamma_rho(Qi2, mRho, gRho, mj, mk)
        )

    def __hrho(self, s, t, u, mRho, gRho):
        return (
            self.__bw_rho(s, mRho, gRho, MPI_GEV, MPI_GEV)
            + self.__bw_rho(t, mRho, gRho, MPI0_GEV, MPI_GEV)
            + self.__bw_rho(u, mRho, gRho, MPI0_GEV, MPI_GEV)
        )

    def __iso_spin_zero(self, q2, s, t, u, ci0, cs):
        coups = np.full_like(self.couplings, ci0)
        coups[1] = cs
        c0 = np.sum(
            coups * self.couplings * breit_wigner_fw(q2, self.masses, self.widths),
        )

        f0 = c0 * self.__hrho(s, t, u, self.masses_rho_i0[0], self.widths_rho_i0[0])

        f0 += (
            cs
            * self.couplings_rho_i0[1]
            * breit_wigner_fw(q2, self.masses[1], self.widths[1])
            * self.__hrho(s, t, u, self.masses_rho_i0[1], self.widths_rho_i0[1])
        )
        f0 += (
            ci0
            * self.couplings_rho_i0[2]
            * breit_wigner_fw(q2, self.masses[3], self.widths[3])
            * self.__hrho(s, t, u, self.masses_rho_i0[2], self.widths_rho_i0[2])
        )
        return f0

    def __iso_spin_one(self, q2, s, ci1):
        if ci1 == 0:
            return 0
        f1 = (
            self.__bw_rho(
                s, self.masses_rho_i1[0], self.widths_rho_i1[0], MPI_GEV, MPI_GEV
            )
            / self.masses_rho_i1[0] ** 2
        )
        f1 += (
            self.sigma
            * self.__bw_rho(
                s, self.masses_rho_i1[1], self.widths_rho_i1[1], MPI_GEV, MPI_GEV
            )
            / self.masses_rho_i1[1] ** 2
        )
        gw = (
            self.coupling_omega_pre
            * self.masses_rho_i1[0] ** 2
            * self.coupling_omega_pi_pi
        )
        f1 *= (
            ci1
            * gw
            * breit_wigner_fw(q2, self.mass_omega_i1, self.width_omega_i1)
            / self.mass_omega_i1**2
        )
        return f1

    def __form_factor(self, q2, s, t, u, gvuu, gvdd, gvss):
        """
        Compute the form factor for a vector decaying into two charged pions and
        a neutral pion.

        Parameters
        ----------
        q2:
            Square of the center-of-mass energy in GeV.
        s:
            Mandelstam variable s = (P - p^{0})^2
        t:
            Mandelstam variable t = (P - p^{+})^2
        u:
            Mandelstam variable t = (P - p^{-})^2
        """
        ci1 = gvuu - gvdd
        ci0 = 3 * (gvuu + gvdd)
        cs = -3 * gvss

        return self.__iso_spin_zero(q2, s, t, u, ci0, cs) + self.__iso_spin_one(
            q2, s, ci1
        )

    def __msqrd(self, momenta, q2: float, gvuu: float, gvdd: float, gvss: float):
        p1 = momenta[:, 0]
        p2 = momenta[:, 1]
        p3 = momenta[:, 2]

        s = lnorm_sqr(p2 + p3)
        t = lnorm_sqr(p1 + p3)
        u = lnorm_sqr(p1 + p2)

        return (
            np.abs(self.__form_factor(q2, s, t, u, gvuu, gvdd, gvss)) ** 2
            * (
                -(MPI_GEV**4 * s)
                + MPI_GEV**2
                * (
                    -(MPI0_GEV**4)
                    - q2**2
                    + q2 * s
                    + MPI0_GEV**2 * (2 * q2 + s)
                    + 2 * s * t
                )
                - s * (MPI0_GEV**2 * (q2 - t) + t * (-q2 + s + t))
            )
            / 12.0
        )

    def __integrated_form_factor(
        self, *, q2: float, gvuu: float, gvdd: float, gvss: float, npts: int = 10000
    ) -> Tuple[float, float]:
        """
        Compute the form factor for a vector decaying into two charged pions and
        a neutral pion integrated over the three-body phase-space.

        Parameters
        ----------
        q2:
            Square of the center-of-mass energy in GeV.
        """
        if q2 < MPI0_GEV + 2 * MPI_GEV:
            return (0.0, 0.0)

        cme = np.sqrt(q2)
        phase_space = Rambo(cme, np.array([MPI0_GEV, MPI_GEV, MPI_GEV]))
        ps, ws = phase_space.generate(npts)

        ws = ws * self.__msqrd(ps, q2, gvuu, gvdd, gvss)

        avg: float = np.average(ws)  # type: ignore
        error: float = np.std(ws, ddof=1) / np.sqrt(npts)

        return avg, error

    def integrated_form_factor(
        self, *, q2: float, gvuu: float, gvdd: float, gvss: float, npts: int = 10000
    ) -> Tuple[float, float]:
        """
        Compute the form factor for a vector decaying into two charged pions and
        a neutral pion integrated over the three-body phase-space.

        Parameters
        ----------
        q2:
            Square of the center-of-mass energy in GeV.
        """
        q2_gev = q2 * 1e-6
        integral, error = self.__integrated_form_factor(
            q2=q2_gev, gvuu=gvuu, gvdd=gvdd, gvss=gvss, npts=npts
        )
        return integral * 1e6, error * 1e6

    def width(
        self, *, mv: float, gvuu: float, gvdd: float, gvss: float, npts: int = 10000
    ) -> float:
        mpi0 = parameters.neutral_pion_mass
        mpi = parameters.charged_pion_mass
        if mv < 2 * mpi + mpi0:
            return 0.0
        integral, _ = self.integrated_form_factor(
            q2=mv**2, gvuu=gvuu, gvdd=gvdd, gvss=gvss, npts=npts
        )
        return integral / (2 * mv)

    def cross_section(
        self,
        *,
        cme,
        mx: float,
        mv: float,
        gvuu: float,
        gvdd: float,
        gvss: float,
        gamv: float,
        npts: int = 10_000,
    ):
        rescale = width_to_cs(cme=cme, mx=mx, mv=mv, wv=gamv)
        return rescale * self.width(mv=cme, gvuu=gvuu, gvdd=gvdd, gvss=gvss, npts=npts)

    def energy_distributions(
        self,
        *,
        cme: float,
        gvuu: float,
        gvdd: float,
        gvss: float,
        npts: int = 10000,
        nbins: int = 25,
    ):
        q = cme * 1e-3

        def msqrd(momenta):
            return self.__msqrd(momenta, q**2, gvuu, gvdd, gvss)

        phase_space = Rambo(q, np.array([MPI0_GEV, MPI_GEV, MPI_GEV]), msqrd=msqrd)
        dist = phase_space.energy_distributions(npts, nbins)

        # e, and dp/de have units of GeV and GeV^-1. Convert to MeV:
        for i in range(len(dist)):
            dist[i] = PhaseSpaceDistribution1D(
                dist[i].bins * 1e-3, dist[i].probabilities * 1e3
            )
        return dist

    def invariant_mass_distributions(
        self,
        *,
        cme: float,
        gvuu: float,
        gvdd: float,
        gvss: float,
        npts: int = 10000,
        nbins: int = 25,
    ):
        q = cme * 1e-3

        def msqrd(momenta):
            return self.__msqrd(momenta, q**2, gvuu, gvdd, gvss)

        phase_space = Rambo(q, np.array([MPI0_GEV, MPI_GEV, MPI_GEV]), msqrd=msqrd)
        dist = phase_space.invariant_mass_distributions(n=npts, nbins=nbins)
        # e, and dp/de have units of GeV and GeV^-1. Convert to MeV:
        for key, val in dist.items():
            dist[key] = PhaseSpaceDistribution1D(
                val.bins * 1e-3, val.probabilities * 1e3
            )
        return dist
