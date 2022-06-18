from dataclasses import dataclass, field
from typing import Tuple, Union, overload

import numpy as np

from hazma.rambo import PhaseSpace
from hazma.utils import lnorm_sqr

from ._utils import MPI0_GEV, MPI_GEV, RealArray, breit_wigner_fw
from ._base import VectorFormFactorPPP

MPI0 = MPI0_GEV
MPI = MPI_GEV


@dataclass
class VectorFormFactorPiPiPi0(VectorFormFactorPPP):
    """Class for computing the vector form-factor for a 3-pion final state."""

    _fsp_masses: Tuple[float, float, float] = field(
        init=False, default=(MPI0_GEV, MPI_GEV, MPI_GEV)
    )
    fsp_masses: Tuple[float, float, float] = field(init=False, default=(MPI0, MPI, MPI))

    masses: RealArray = field(default=np.array([0.7824, 1.01924, 1.375, 1.631]))
    widths: RealArray = field(default=np.array([0.00869, 0.00414, 0.250, 0.245]))
    couplings: RealArray = field(default=np.array([18.20, -0.87, -0.77, -1.12]))
    masses_rho_i0: RealArray = field(default=np.array([0.77609, 1.465, 1.7]))
    widths_rho_i0: RealArray = field(default=np.array([0.14446, 0.31, 0.235]))
    couplings_rho_i0: RealArray = field(default=np.array([0.0, -0.72, -0.59]))

    masses_rho_i1: RealArray = field(default=np.array([0.77609, 1.7]))
    widths_rho_i1: RealArray = field(default=np.array([0.14446, 0.26]))
    mass_omega_i1: float = field(default=0.78259)
    width_omega_i1: float = field(default=0.00849)
    coupling_omega_pre: float = field(default=3.768)
    coupling_omega_pi_pi: float = field(default=0.185)
    sigma: float = field(default=-0.1)

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

    def form_factor(self, *, q, s, t, gvuu, gvdd, gvss):
        """
        Compute the form factor for a vector decaying into two charged pions and
        a neutral pion.

        Parameters
        ----------
        q:
            Center-of-mass energy in MeV.
        s:
            Mandelstam variable s = (P - p^{0})^2
        t:
            Mandelstam variable t = (P - p^{+})^2
        u:
            Mandelstam variable t = (P - p^{-})^2
        """
        q2 = q**2 * 1e-6
        ss = s * 1e-6
        tt = t * 1e-6
        uu = MPI0_GEV**2 + 2 * MPI_GEV**2 - ss - tt
        return self.__form_factor(
            q2=q2, s=ss, t=tt, u=uu, gvuu=gvuu, gvdd=gvdd, gvss=gvss
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

    def _integrated_form_factor(
        self, *, q: float, gvuu: float, gvdd: float, gvss: float, npts: int = 10000
    ) -> float:
        """
        Compute the form factor for a vector decaying into two charged pions and
        a neutral pion integrated over the three-body phase-space.

        Parameters
        ----------
        q: float
            Center-of-mass energy in GeV.
        """
        if q < MPI0_GEV + 2 * MPI_GEV:
            return 0.0

        phase_space = PhaseSpace(q, np.array(self._fsp_masses))
        ps, ws = phase_space.generate(npts)

        ws = ws * self.__msqrd(ps, q**2, gvuu, gvdd, gvss)

        avg: float = np.average(ws)  # type: ignore
        # error: float = np.std(ws, ddof=1) / np.sqrt(npts)

        return avg / q**2

    @overload
    def integrated_form_factor(
        self, *, q: float, gvuu: float, gvdd: float, gvss: float, npts: int
    ) -> float:
        ...

    @overload
    def integrated_form_factor(
        self, *, q: RealArray, gvuu: float, gvdd: float, gvss: float, npts: int
    ) -> RealArray:
        ...

    def integrated_form_factor(
        self,
        *,
        q: Union[float, RealArray],
        gvuu: float,
        gvdd: float,
        gvss: float,
        npts: int = 10000,
    ) -> Union[float, RealArray]:
        """
        Compute the form factor for a vector decaying into two charged pions and
        a neutral pion integrated over the three-body phase-space.

        Parameters
        ----------
        q:
            Center-of-mass energy in MeV.
        """
        scalar = np.isscalar(q)
        qq = np.atleast_1d(q) * 1e-3

        integral = np.array(
            [
                self._integrated_form_factor(
                    q=q_, gvuu=gvuu, gvdd=gvdd, gvss=gvss, npts=npts
                )
                for q_ in qq
            ]
        )

        if scalar:
            return integral[0]

        return integral

    @overload
    def width(
        self, mv: float, *, gvuu: float, gvdd: float, gvss: float, npts: int
    ) -> float:
        ...

    @overload
    def width(
        self, mv: RealArray, *, gvuu: float, gvdd: float, gvss: float, npts: int
    ) -> RealArray:
        ...

    def width(
        self,
        mv: Union[float, RealArray],
        *,
        gvuu: float,
        gvdd: float,
        gvss: float,
        npts: int = 10000,
    ) -> Union[float, RealArray]:
        return (
            0.5
            * mv
            * self.integrated_form_factor(
                q=mv, gvuu=gvuu, gvdd=gvdd, gvss=gvss, npts=npts
            )
        )

    @overload
    def cross_section(
        self,
        *,
        q: float,
        mx: float,
        mv: float,
        gvxx: float,
        wv: float,
        gvuu: float,
        gvdd: float,
        gvss: float,
        npts: int,
    ) -> float:
        ...

    @overload
    def cross_section(
        self,
        *,
        q: RealArray,
        mx: float,
        mv: float,
        gvxx: float,
        wv: float,
        gvuu: float,
        gvdd: float,
        gvss: float,
        npts: int,
    ) -> RealArray:
        ...

    def cross_section(
        self,
        *,
        q: Union[float, RealArray],
        mx: float,
        mv: float,
        gvxx: float,
        wv: float,
        gvuu: float,
        gvdd: float,
        gvss: float,
        npts: int = 10_000,
    ) -> Union[float, RealArray]:
        """Compute the cross-section of dark matter annihilation."""
        single = np.isscalar(q)
        qq = np.atleast_1d(q).astype(np.float64)

        s = qq**2
        pre = (
            gvxx**2
            * (s + 2 * mx**2)
            / (np.sqrt(s - 4 * mx**2) * ((s - mv**2) ** 2 + (mv * wv) ** 2))
        )
        pre = pre * 0.5 * qq
        cs = pre * self.integrated_form_factor(
            q=qq, gvuu=gvuu, gvdd=gvdd, gvss=gvss, npts=npts
        )

        if single:
            return cs[0]
        return cs
