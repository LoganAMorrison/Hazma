from dataclasses import InitVar, dataclass, field
from typing import Tuple, Union, List, Dict

import numpy as np

from hazma.phase_space import PhaseSpaceDistribution1D

from ._utils import MPI0_GEV, MPI_GEV, RealArray, breit_wigner_fw
from ._three_body import VectorFormFactorPPP

MPI0 = MPI0_GEV * 1e3
MPI = MPI_GEV * 1e3


@dataclass(frozen=True)
class VectorFormFactorPiPiPi0FitData:
    r"""Storage class for parameters used to compute the pi-pi-pi vector
    form-factor.

    Attributes
    ----------
    masses: RealArray
        VMD resonance masses.
    widths: RealArray
        VMD resonance widths.
    couplings: RealArray
        VMD resonance couplings.
    masses_rho_i0: RealArray
        VMD I=0 rho resonance masses.
    widths_rho_i0: RealArray
        VMD I=0 rho resonance widths.
    couplings_rho_i0: RealArray
        VMD I=0 rho resonance couplings.
    masses_rho_i1: RealArray
        VMD I=1 rho resonance masses.
    widths_rho_i1: RealArray
        VMD I=1 rho resonance widths.
    mass_omega_i1: float
        VMD I=1 omega mass.
    width_omega_i1: float
        VMD omega width.
    coupling_omega_pre: float
        VMD omega coupling.
    coupling_omega_pi_pi: float
        VMD omega-pi-pi coupling.
    sigma: float
    """

    masses: RealArray = field(repr=False)
    widths: RealArray = field(repr=False)
    couplings: RealArray = field(repr=False)
    masses_rho_i0: RealArray = field(repr=False)
    widths_rho_i0: RealArray = field(repr=False)
    couplings_rho_i0: RealArray = field(repr=False)

    masses_rho_i1: RealArray = field(repr=False)
    widths_rho_i1: RealArray = field(repr=False)
    mass_omega_i1: float = field(repr=False)
    width_omega_i1: float = field(repr=False)
    coupling_omega_pre: float = field(repr=False)
    coupling_omega_pi_pi: float = field(repr=False)
    sigma: float = field(repr=False)


@dataclass
class VectorFormFactorPiPiPi0(VectorFormFactorPPP):
    r"""Class for computing the pi-pi-pi0' vector form-factor.

    Attributes
    ----------
    fsp_masses: (float,float,float)
        Masses of the final state particles.
    fit_data: VectorFormFactorPiPiPi0FitData
        Fitted parameters for the pi-pi-pi0 vector form-factor.

    Methods
    -------
    form_factor
        Compute the un-integrated form-factor.
    integrated_form_factor
        Compute the form-factor integrated over phase-space.
    width
        Compute the decay width of a vector into pi-pi-pi0.
    cross_section
        Compute the dark matter annihilation cross section into pi-pi-pi0.
    """

    _fsp_masses: Tuple[float, float, float] = field(
        init=False, default=(MPI0_GEV, MPI_GEV, MPI_GEV)
    )
    fsp_masses: Tuple[float, float, float] = field(init=False, default=(MPI0, MPI, MPI))
    fit_data: VectorFormFactorPiPiPi0FitData = field(init=False)

    masses: InitVar[RealArray] = field(
        default=np.array([0.7824, 1.01924, 1.375, 1.631])
    )
    widths: InitVar[RealArray] = field(
        default=np.array([0.00869, 0.00414, 0.250, 0.245])
    )
    couplings: InitVar[RealArray] = field(
        default=np.array([18.20, -0.87, -0.77, -1.12])
    )
    masses_rho_i0: InitVar[RealArray] = field(default=np.array([0.77609, 1.465, 1.7]))
    widths_rho_i0: InitVar[RealArray] = field(default=np.array([0.14446, 0.31, 0.235]))
    couplings_rho_i0: InitVar[RealArray] = field(default=np.array([0.0, -0.72, -0.59]))

    masses_rho_i1: InitVar[RealArray] = field(default=np.array([0.77609, 1.7]))
    widths_rho_i1: InitVar[RealArray] = field(default=np.array([0.14446, 0.26]))
    mass_omega_i1: InitVar[float] = field(default=0.78259)
    width_omega_i1: InitVar[float] = field(default=0.00849)
    coupling_omega_pre: InitVar[float] = field(default=3.768)
    coupling_omega_pi_pi: InitVar[float] = field(default=0.185)
    sigma: InitVar[float] = field(default=-0.1)

    def __post_init__(
        self,
        masses: RealArray,
        widths: RealArray,
        couplings: RealArray,
        masses_rho_i0: RealArray,
        widths_rho_i0: RealArray,
        couplings_rho_i0: RealArray,
        masses_rho_i1: RealArray,
        widths_rho_i1: RealArray,
        mass_omega_i1: float,
        width_omega_i1: float,
        coupling_omega_pre: float,
        coupling_omega_pi_pi: float,
        sigma: float,
    ):
        self.fit_data = VectorFormFactorPiPiPi0FitData(
            masses=masses,
            widths=widths,
            couplings=couplings,
            masses_rho_i0=masses_rho_i0,
            widths_rho_i0=widths_rho_i0,
            couplings_rho_i0=couplings_rho_i0,
            masses_rho_i1=masses_rho_i1,
            widths_rho_i1=widths_rho_i1,
            mass_omega_i1=mass_omega_i1,
            width_omega_i1=width_omega_i1,
            coupling_omega_pre=coupling_omega_pre,
            coupling_omega_pi_pi=coupling_omega_pi_pi,
            sigma=sigma,
        )

    def __gamma_rho(self, s, mass, width, mj, mk):
        # p-wave width
        m2 = mass**2
        msum2 = (mj + mk) ** 2
        rat = (s - msum2) / (m2 - msum2)
        return width * m2 / s * rat**1.5

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
        coups = np.full_like(self.fit_data.couplings, ci0)
        coups[1] = cs
        c0 = np.sum(
            coups
            * self.fit_data.couplings
            * breit_wigner_fw(q2, self.fit_data.masses, self.fit_data.widths),
        )

        f0 = c0 * self.__hrho(
            s, t, u, self.fit_data.masses_rho_i0[0], self.fit_data.widths_rho_i0[0]
        )

        f0 += (
            cs
            * self.fit_data.couplings_rho_i0[1]
            * breit_wigner_fw(q2, self.fit_data.masses[1], self.fit_data.widths[1])
            * self.__hrho(
                s, t, u, self.fit_data.masses_rho_i0[1], self.fit_data.widths_rho_i0[1]
            )
        )
        f0 += (
            ci0
            * self.fit_data.couplings_rho_i0[2]
            * breit_wigner_fw(q2, self.fit_data.masses[3], self.fit_data.widths[3])
            * self.__hrho(
                s, t, u, self.fit_data.masses_rho_i0[2], self.fit_data.widths_rho_i0[2]
            )
        )
        return f0

    def __iso_spin_one(self, q2, s, ci1):
        if ci1 == 0:
            return 0
        f1 = (
            self.__bw_rho(
                s,
                self.fit_data.masses_rho_i1[0],
                self.fit_data.widths_rho_i1[0],
                MPI_GEV,
                MPI_GEV,
            )
            / self.fit_data.masses_rho_i1[0] ** 2
        )
        f1 += (
            self.fit_data.sigma
            * self.__bw_rho(
                s,
                self.fit_data.masses_rho_i1[1],
                self.fit_data.widths_rho_i1[1],
                MPI_GEV,
                MPI_GEV,
            )
            / self.fit_data.masses_rho_i1[1] ** 2
        )
        gw = (
            self.fit_data.coupling_omega_pre
            * self.fit_data.masses_rho_i1[0] ** 2
            * self.fit_data.coupling_omega_pi_pi
        )
        f1 *= (
            ci1
            * gw
            * breit_wigner_fw(
                q2, self.fit_data.mass_omega_i1, self.fit_data.width_omega_i1
            )
            / self.fit_data.mass_omega_i1**2
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

    def form_factor(self, q, s, t, *, gvuu: float, gvdd: float, gvss: float):
        """
        Compute the form factor for a vector decaying into two charged pions and
        a neutral pion.

        Parameters
        ----------
        q: float
            Center-of-mass energy in MeV.
        s: float or array-like
            Mandelstam variable s = (P - p^{0})^2
        t: float or array-like
            Mandelstam variable t = (P - p^{+})^2
        gvuu, gvdd, gvss: float
            Coupling of vector to up-, down-, and strange-quarks.

        Returns
        -------
        ff: complex or array-like
            Three pion form-factor.
        """
        q2 = q**2 * 1e-6
        ss = s * 1e-6
        tt = t * 1e-6
        uu = q2 + MPI0_GEV**2 + 2 * MPI_GEV**2 - ss - tt

        ff = self.__form_factor(
            q2=q2, s=ss, t=tt, u=uu, gvuu=gvuu, gvdd=gvdd, gvss=gvss
        )
        return ff * 1e-9

    def integrated_form_factor(
        self,
        *,
        q: Union[float, RealArray],
        gvuu: float,
        gvdd: float,
        gvss: float,
        method: str = "rambo",
        npts: int = 1 << 14,
    ) -> Union[float, RealArray]:
        """Compute the three pion form-factor integrated over phase-space.

        Parameters
        ----------
        q: float or array-like
            Center of mass energy.
        gvuu, gvdd, gvss: float
            Coupling of vector to up-, down-, and strange-quarks.
        method: str, optional
            Method used to integrate. Default is 'quad'. Options are 'quad' or
            'rambo'.
        npts: int, optional
            Number of phase-space points to use in integration. Ignored is
            method isn't 'rambo'. Default is 10_000.

        Returns
        -------
        ff: float or array-like
            Integrated form-factor.
        """
        return self._integrated_form_factor(
            q=q, method=method, npts=npts, gvuu=gvuu, gvdd=gvdd, gvss=gvss
        )

    def width(
        self,
        mv: Union[float, RealArray],
        *,
        gvuu: float,
        gvdd: float,
        gvss: float,
        method: str = "rambo",
        npts: int = 1 << 14,
    ) -> Union[float, RealArray]:
        r"""Compute the partial decay width of a massive vector into three pions.

        Parameters
        ----------
        mv: float or array-like
            Mass of the vector.
        gvuu, gvdd, gvss: float
            Coupling of vector to up-, donw-, and strange-quarks.
        nbins: float
            Number of bins used to generate distribution.
        method: str, optional
            Method used to integrate over phase-space.
            See `hazma.phase_space.energy_distributions_three_body`
            for availible methods.
        npts: int, optional
            Number of phase-space points to use in integration. Only used if
            `method='rambo'`. Default is 2^14.

        Returns
        -------
        width: float or array-like
            Decay width of vector into three pions.
        """

        return self._width(
            mv=mv, method=method, npts=npts, gvuu=gvuu, gvdd=gvdd, gvss=gvss
        )

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
        method: str = "rambo",
        npts: int = 1 << 14,
    ) -> Union[float, RealArray]:
        r"""Compute the cross section for dark matter annihilating into three pions.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy.
        mx: float
            Mass of the dark matter in MeV.
        mv: float
            Mass of the vector mediator in MeV.
        gvxx: float
            Coupling of vector to dark matter.
        wv: float
            Width of the vector in MeV.
        gvuu, gvdd, gvss: float
            Coupling of vector to up-, donw- and strange-quarks.
        nbins: float
            Number of bins used to generate distribution.
        method: str, optional
            Method used to integrate over phase-space. Default is "rambo".  See
            `hazma.phase_space.energy_distributions_three_body` for availible
            methods.
        npts: int, optional
            Number of phase-space points to use in integration. Only used if
            `method='rambo'`. Default is 2^14.

        Returns
        -------
        cs: float or array-like
            Annihilation cross section into three pions.
        """
        return self._cross_section(
            q=q,
            mx=mx,
            mv=mv,
            gvxx=gvxx,
            wv=wv,
            method=method,
            npts=npts,
            gvuu=gvuu,
            gvdd=gvdd,
            gvss=gvss,
        )

    def energy_distributions(
        self,
        q: float,
        nbins: int,
        *,
        gvuu: float,
        gvdd: float,
        gvss: float,
        method: str = "quad",
        npts: int = 1 << 14,
    ) -> List[PhaseSpaceDistribution1D]:
        r"""Compute the energy distributions of the final state pions.

        Parameters
        ----------
        q: float
            Center-of-mass energy.
        nbins: float
            Number of bins used to generate distribution.
        gvuu, gvdd, gvss: float
            Coupling of vector to up-, down-, and strange-quarks.
        method: str, optional
            Method used to integrate over phase-space. Default is "rambo".  See
            `hazma.phase_space.energy_distributions_three_body` for availible
            methods.
        npts: int, optional
            Number of phase-space points to use in integration. Only used if
            `method='rambo'`. Default is 2^14.

        Returns
        -------
        dists: List[PhaseSpaceDistribution1D]
            List of the energy distributions.
        """
        return self._energy_distributions(
            q=q, nbins=nbins, gvuu=gvuu, gvdd=gvdd, gvss=gvss, method=method, npts=npts
        )

    def invariant_mass_distributions(
        self,
        q: float,
        nbins: int,
        *,
        gvuu: float,
        gvdd: float,
        gvss: float,
        method: str = "quad",
        npts: int = 1 << 14,
    ) -> Dict[Tuple[int, int], PhaseSpaceDistribution1D]:
        r"""Compute the invariant-mass distributions of the all pairs of the
        final-state particles.

        Parameters
        ----------
        q: float
            Center-of-mass energy.
        nbins: float
            Number of bins used to generate distribution.
        gvuu, gvdd, gvss: float
            Coupling of vector to up-, down-, and strange-quarks.
        method: str, optional
            Method used to integrate over phase-space. Default is "rambo".  See
            `hazma.phase_space.energy_distributions_three_body` for availible
            methods.
        npts: int, optional
            Number of phase-space points to use in integration. Only used if
            `method='rambo'`. Default is 2^14.

        Returns
        -------
        dists: Dict[(int,int), PhaseSpaceDistribution1D]
            Dictionary of the invariant-mass distributions. Keys specify the
            pair of particles the distribution represents.
        """
        return self._invariant_mass_distributions(
            q=q, nbins=nbins, gvuu=gvuu, gvdd=gvdd, gvss=gvss, method=method, npts=npts
        )

    # def __msqrd(self, momenta, q2: float, gvuu: float, gvdd: float, gvss: float):
    #     p1 = momenta[:, 0]
    #     p2 = momenta[:, 1]
    #     p3 = momenta[:, 2]

    #     s = lnorm_sqr(p2 + p3)
    #     t = lnorm_sqr(p1 + p3)
    #     u = lnorm_sqr(p1 + p2)

    #     return (
    #         np.abs(self.__form_factor(q2, s, t, u, gvuu, gvdd, gvss)) ** 2
    #         * (
    #             -(MPI_GEV**4 * s)
    #             + MPI_GEV**2
    #             * (
    #                 -(MPI0_GEV**4)
    #                 - q2**2
    #                 + q2 * s
    #                 + MPI0_GEV**2 * (2 * q2 + s)
    #                 + 2 * s * t
    #             )
    #             - s * (MPI0_GEV**2 * (q2 - t) + t * (-q2 + s + t))
    #         )
    #         / 12.0
    #     )

    # def _msqrd(self, q: float, s_, t_, gvuu: float, gvdd: float, gvss: float):
    #     qq = q * 1e-3
    #     ss = s_ * 1e-6
    #     tt = t_ * 1e-6
    #     uu = q**2 + sum(self._fsp_masses) - ss - tt

    #     return (
    #         np.abs(self.__form_factor(qq**2, ss, tt, uu, gvuu, gvdd, gvss)) ** 2
    #         * (
    #             -(MPI_GEV**4 * ss)
    #             + MPI_GEV**2
    #             * (
    #                 -(MPI0_GEV**4)
    #                 - qq**4
    #                 + qq**2 * ss
    #                 + MPI0_GEV**2 * (2 * qq**2 + ss)
    #                 + 2 * ss * tt
    #             )
    #             - ss * (MPI0_GEV**2 * (qq**2 - tt) + tt * (-(qq**2) + ss + tt))
    #         )
    #         / 12.0
    #     )

    # def _integrated_form_factor(
    #     self, *, q: float, gvuu: float, gvdd: float, gvss: float, npts: int = 10000
    # ) -> float:
    #     """
    #     Compute the form factor for a vector decaying into two charged pions and
    #     a neutral pion integrated over the three-body phase-space.

    #     Parameters
    #     ----------
    #     q: float
    #         Center-of-mass energy in GeV.
    #     """
    #     if q < MPI0_GEV + 2 * MPI_GEV:
    #         return 0.0

    #     phase_space = PhaseSpace(q, np.array(self._fsp_masses))
    #     ps, ws = phase_space.generate(npts)

    #     ws = ws * self.__msqrd(ps, q**2, gvuu, gvdd, gvss)

    #     avg: float = np.average(ws)  # type: ignore
    #     # error: float = np.std(ws, ddof=1) / np.sqrt(npts)

    #     return avg / q**2
