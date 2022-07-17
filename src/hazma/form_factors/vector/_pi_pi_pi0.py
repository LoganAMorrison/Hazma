"""
Module implementing the pi-pi-pi0 form factor.
"""

from dataclasses import InitVar, dataclass, field
from typing import Tuple, Union, List, Dict, overload

import numpy as np

from hazma.phase_space import PhaseSpaceDistribution1D
from hazma.utils import RealOrRealArray, RealArray, ComplexArray, ComplexOrComplexArray

from ._utils import MPI0_GEV, MPI_GEV, breit_wigner_fw
from ._three_body import VectorFormFactorPPP, Couplings
from ._base import vector_couplings_to_isospin

MPI0 = MPI0_GEV * 1e3
MPI = MPI_GEV * 1e3


@dataclass(frozen=True)
class VectorFormFactorPiPiPi0FitData:  # pylint:disable=too-many-instance-attributes
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

    def __post_init__(  # pylint: disable=too-many-arguments
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

    @staticmethod
    def __gamma_rho(s, mass, width, mj, mk):
        # p-wave width
        m2 = mass**2
        msum2 = (mj + mk) ** 2
        rat = (s - msum2) / (m2 - msum2)
        return width * m2 / s * rat**1.5

    @staticmethod
    def __bw_rho(Qi2, mRho, gRho, mj, mk):
        # Breit-Wigner for rhos
        return mRho**2 / (
            Qi2
            - mRho**2
            + 1j
            * np.sqrt(Qi2)
            * VectorFormFactorPiPiPi0.__gamma_rho(Qi2, mRho, gRho, mj, mk)
        )

    @staticmethod
    def __hrho(s, t, u, mRho, gRho):
        return (
            VectorFormFactorPiPiPi0.__bw_rho(s, mRho, gRho, MPI_GEV, MPI_GEV)
            + VectorFormFactorPiPiPi0.__bw_rho(t, mRho, gRho, MPI0_GEV, MPI_GEV)
            + VectorFormFactorPiPiPi0.__bw_rho(u, mRho, gRho, MPI0_GEV, MPI_GEV)
        )

    def __iso_spin_zero(  # pylint: disable=too-many-arguments
        self, q2, s, t, u, ci0, cs
    ):
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

    def __form_factor(  # pylint: disable=too-many-arguments
        self, q2, s, t, u, couplings: Couplings
    ):
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
        ci0, ci1, cs = vector_couplings_to_isospin(*couplings)
        return self.__iso_spin_zero(q2, s, t, u, ci0, cs) + self.__iso_spin_one(
            q2, s, ci1
        )

    @overload
    def form_factor(  # pylint: disable=arguments-differ
        self, q: float, s: float, t: float, couplings: Couplings
    ) -> complex:
        ...

    @overload
    def form_factor(  # pylint: disable=arguments-differ
        self, q: float, s: RealArray, t: RealArray, couplings: Couplings
    ) -> ComplexArray:
        ...

    def form_factor(  # pylint: disable=arguments-differ
        self, q: float, s: RealOrRealArray, t: RealOrRealArray, couplings: Couplings
    ) -> ComplexOrComplexArray:
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

        ff = self.__form_factor(q2=q2, s=ss, t=tt, u=uu, couplings=couplings)
        return ff * 1e-9

    @overload
    def integrated_form_factor(  # pylint: disable=arguments-differ
        self,
        q: float,
        couplings: Couplings,
        *,
        method: str = "rambo",
        npts: int = 1 << 14,
        epsrel: float = 1e-3,
        epsabs: float = 0.0,
    ) -> float:
        ...

    @overload
    def integrated_form_factor(  # pylint: disable=arguments-differ
        self,
        q: RealArray,
        couplings: Couplings,
        *,
        method: str = "rambo",
        npts: int = 1 << 14,
        epsrel: float = 1e-3,
        epsabs: float = 0.0,
    ) -> RealArray:
        ...

    def integrated_form_factor(  # pylint: disable=arguments-differ
        self,
        q: Union[float, RealArray],
        couplings: Couplings,
        *,
        method: str = "rambo",
        npts: int = 1 << 14,
        epsrel: float = 1e-3,
        epsabs: float = 0.0,
    ) -> Union[float, RealArray]:
        """Compute the three pion form-factor integrated over phase-space.

        Parameters
        ----------
        q: float or array-like
            Center of mass energy.
        gvuu, gvdd, gvss: float
            Coupling of vector to up-, down-, and strange-quarks.

        Returns
        -------
        iff: float or array-like
            Integrated form-factor.

        Other Parameters
        ----------------
        method: str, optional
            Method used to integrate. Default is 'quad'. Options are 'quad' or
            'rambo'.
        npts: int, optional
            Number of phase-space points to use in integration. Ignored is
            method isn't 'rambo'. Default is 10_000.
        epsrel: float, optional
            Relative error tolerance. Default is 1e-3.
        epsabs: float, optional
            Absolute error tolerance. Default is 0.0.
        """
        return self._integrated_form_factor(
            q=q,
            method=method,
            npts=npts,
            couplings=couplings,
            epsrel=epsrel,
            epsabs=epsabs,
        )

    @overload
    def width(  # pylint: disable=arguments-differ
        self,
        mv: float,
        couplings: Couplings,
        *,
        method: str = "rambo",
        npts: int = 1 << 14,
        epsrel: float = 1e-3,
        epsabs: float = 0.0,
    ) -> float:
        ...

    @overload
    def width(  # pylint: disable=arguments-differ
        self,
        mv: RealArray,
        couplings: Couplings,
        *,
        method: str = "rambo",
        npts: int = 1 << 14,
        epsrel: float = 1e-3,
        epsabs: float = 0.0,
    ) -> RealArray:
        ...

    def width(  # pylint: disable=arguments-differ
        self,
        mv: RealOrRealArray,
        couplings: Couplings,
        *,
        method: str = "rambo",
        npts: int = 1 << 14,
        epsrel: float = 1e-3,
        epsabs: float = 0.0,
    ) -> RealOrRealArray:
        r"""Compute the partial decay width of a massive vector into three pions.

        Parameters
        ----------
        mv: float or array-like
            Mass of the vector.
        gvuu, gvdd, gvss: float
            Coupling of vector to up-, donw-, and strange-quarks.

        Returns
        -------
        width: float or array-like
            Decay width of vector into three pions.

        Other Parameters
        ----------------
        method: str, optional
            Method used to integrate. Default is 'quad'. Options are 'quad' or
            'rambo'.
        npts: int, optional
            Number of phase-space points to use in integration. Ignored is
            method isn't 'rambo'. Default is 10_000.
        epsrel: float, optional
            Relative error tolerance. Default is 1e-3.
        epsabs: float, optional
            Absolute error tolerance. Default is 0.0.
        """

        return self._width(
            mv=mv,
            method=method,
            npts=npts,
            couplings=couplings,
            epsrel=epsrel,
            epsabs=epsabs,
        )

    @overload
    def cross_section(  # pylint: disable=arguments-differ,too-many-arguments
        self,
        q: float,
        mx: float,
        mv: float,
        gvxx: float,
        wv: float,
        couplings: Couplings,
        *,
        method: str = ...,
        npts: int = ...,
        epsrel: float = ...,
        epsabs: float = ...,
    ) -> float:
        ...

    @overload
    def cross_section(  # pylint: disable=arguments-differ,too-many-arguments
        self,
        q: RealArray,
        mx: float,
        mv: float,
        gvxx: float,
        wv: float,
        couplings: Couplings,
        *,
        method: str = ...,
        npts: int = ...,
        epsrel: float = ...,
        epsabs: float = ...,
    ) -> RealArray:
        ...

    def cross_section(  # pylint: disable=arguments-differ,too-many-arguments
        self,
        q: RealOrRealArray,
        mx: float,
        mv: float,
        gvxx: float,
        wv: float,
        couplings: Couplings,
        *,
        method: str = "rambo",
        npts: int = 1 << 14,
        epsrel: float = 1e-3,
        epsabs: float = 0.0,
    ) -> RealOrRealArray:
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

        Returns
        -------
        cs: float or array-like
            Annihilation cross section into three pions.

        Other Parameters
        ----------------
        method: str, optional
            Method used to integrate. Default is 'quad'. Options are 'quad' or
            'rambo'.
        npts: int, optional
            Number of phase-space points to use in integration. Ignored is
            method isn't 'rambo'. Default is 10_000.
        epsrel: float, optional
            Relative error tolerance. Default is 1e-3.
        epsabs: float, optional
            Absolute error tolerance. Default is 0.0.
        """
        return self._cross_section(
            q=q,
            mx=mx,
            mv=mv,
            gvxx=gvxx,
            wv=wv,
            method=method,
            npts=npts,
            couplings=couplings,
            epsrel=epsrel,
            epsabs=epsabs,
        )

    def energy_distributions(  # pylint: disable=arguments-differ
        self,
        q: float,
        nbins: int,
        couplings: Couplings,
        *,
        method: str = "quad",
        npts: int = 1 << 14,
        epsrel: float = 1e-3,
        epsabs: float = 0.0,
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

        Returns
        -------
        dists: List[PhaseSpaceDistribution1D]
            List of the energy distributions.

        Other Parameters
        ----------------
        method: str, optional
            Method used to integrate. Default is 'quad'. Options are 'quad' or
            'rambo'.
        npts: int, optional
            Number of phase-space points to use in integration. Ignored is
            method isn't 'rambo'. Default is 10_000.
        epsrel: float, optional
            Relative error tolerance. Default is 1e-3.
        epsabs: float, optional
            Absolute error tolerance. Default is 0.0.
        """
        return self._energy_distributions(
            q=q,
            nbins=nbins,
            couplings=couplings,
            method=method,
            npts=npts,
            epsrel=epsrel,
            epsabs=epsabs,
        )

    def invariant_mass_distributions(  # pylint: disable=arguments-differ
        self,
        q: float,
        nbins: int,
        couplings: Couplings,
        *,
        method: str = "quad",
        npts: int = 1 << 14,
        epsrel: float = 1e-3,
        epsabs: float = 0.0,
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

        Returns
        -------
        dists: Dict[(int,int), PhaseSpaceDistribution1D]
            Dictionary of the invariant-mass distributions. Keys specify the

        Other Parameters
        ----------------
        method: str, optional
            Method used to integrate. Default is 'quad'. Options are 'quad' or
            'rambo'.
        npts: int, optional
            Number of phase-space points to use in integration. Ignored is
            method isn't 'rambo'. Default is 10_000.
        epsrel: float, optional
            Relative error tolerance. Default is 1e-3.
        epsabs: float, optional
            Absolute error tolerance. Default is 0.0.
            pair of particles the distribution represents.
        """
        return self._invariant_mass_distributions(
            q=q,
            nbins=nbins,
            couplings=couplings,
            method=method,
            npts=npts,
            epsrel=epsrel,
            epsabs=epsabs,
        )
