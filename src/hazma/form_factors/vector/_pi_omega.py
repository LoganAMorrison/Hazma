"""
Implementation of the pi0-omega form factor.
"""

from dataclasses import dataclass, field, InitVar
from typing import Union, overload, Tuple

import numpy as np
import numpy.typing as npt

from hazma import parameters
from hazma.utils import (
    ComplexOrComplexArray,
    RealOrRealArray,
    kallen_lambda,
    ComplexArray,
    RealArray,
)

from ._utils import MOMEGA_GEV, MPI0_GEV
from ._two_body import VectorFormFactorPV, Couplings

MPI0 = MPI0_GEV * 1e3
MOMEGA = MOMEGA_GEV * 1e3


@dataclass(frozen=True)
class VectorFormFactorPi0OmegaFitData:
    r"""Storage class for the pion-omega form factor."""

    g_rho_omega_pi: float = field(repr=False)
    amps: RealArray = field(repr=False)
    phases: RealArray = field(repr=False)
    rho_masses: RealArray = field(repr=False)
    rho_widths: RealArray = field(repr=False)
    frho: float = field(repr=False)


@dataclass
class VectorFormFactorPi0Omega(VectorFormFactorPV):
    """
    Class for computing the form factor for V-omega-pi0. See arXiv:1303.5198 for details
    on the default fit values.

    Attributes
    ----------
    fsp_masses: (float, float)
        Masses of the final-state particles.
    fit_data: VectorFormFactorPi0OmegaFitData
        Stored data used to compute form-factor.

    Methods
    -------
    form_factor
        Compute the un-integrated form-factor into a pion and omega.
    integrated_form_factor
        Compute the form-factor into a pion and omega integrated over
        phase-space.
    width
        Compute the decay width of a vector into a pion and omega.
    cross_section
        Compute the dark matter annihilation cross section into a pion and omega.
    """

    fsp_masses: Tuple[float, float] = field(init=False, default=(MPI0, MOMEGA))
    fit_data: VectorFormFactorPi0OmegaFitData = field(init=False)

    g_rho_omega_pi: InitVar[float] = field(default=15.9)  # units of GeV^-1
    amps: InitVar[RealArray] = field(default=np.array([1.0, 0.175, 0.014]))
    phases: InitVar[RealArray] = field(
        default=np.array([0.0, 124.0, -63.0]) * np.pi / 180.0
    )
    rho_masses: InitVar[RealArray] = field(default=np.array([0.77526, 1.510, 1.720]))
    rho_widths: InitVar[RealArray] = field(default=np.array([0.1491, 0.44, 0.25]))
    frho: InitVar[float] = field(default=5.06325)

    def __post_init__(  # pylint: disable=too-many-arguments
        self,
        g_rho_omega_pi: float,
        amps: RealArray,
        phases: RealArray,
        rho_masses: RealArray,
        rho_widths: RealArray,
        frho: float,
    ):
        self.fit_data = VectorFormFactorPi0OmegaFitData(
            g_rho_omega_pi=g_rho_omega_pi,
            amps=amps,
            phases=phases,
            rho_masses=rho_masses,
            rho_widths=rho_widths,
            frho=frho,
        )

    def _rho_widths(self, s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        q = np.sqrt(s)
        widths = np.array(
            [
                np.full_like(s, self.fit_data.rho_widths[i], dtype=np.float64)
                for i in range(len(self.fit_data.rho_widths))
            ]
        )

        mu2p = MPI0_GEV**2 / s
        mu2w = MOMEGA_GEV**2 / s
        mu2r = self.fit_data.rho_masses[0] ** 2 / s
        p = 0.5 * q * np.sqrt(kallen_lambda(1.0, mu2p, mu2w))
        widths[0] = (
            widths[0] * mu2r * ((1.0 - 4.0 * mu2p) / (mu2r - 4.0 * mu2p)) ** 1.5
        ) + self.fit_data.g_rho_omega_pi**2 * p**3 / (12.0 * np.pi)
        return widths

    def __form_factor(
        self, *, s: npt.NDArray[np.float64], gvuu: float, gvdd: float
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
        masses = self.fit_data.rho_masses[:, np.newaxis]
        phases = self.fit_data.phases[:, np.newaxis]
        ss = s[np.newaxis, :]

        dens = masses**2 - ss - 1j * np.sqrt(ss) * widths
        amps = (
            self.fit_data.amps[:, np.newaxis] * np.exp(1j * phases) * masses**2 / dens
        )
        return (
            self.fit_data.g_rho_omega_pi
            * ci1
            / self.fit_data.frho
            * np.sum(amps, axis=0)
        )

    @overload
    def form_factor(  # pylint: disable=arguments-differ
        self, q: float, couplings: Couplings
    ) -> complex:
        ...

    @overload
    def form_factor(  # pylint: disable=arguments-differ
        self, q: RealArray, couplings: Couplings
    ) -> ComplexArray:
        ...

    def form_factor(  # pylint: disable=arguments-differ
        self, q: Union[float, RealArray], couplings: Couplings
    ) -> ComplexOrComplexArray:
        """
        Compute the V-omega-pi form factor.

        Parameters
        ----------
        q: Union[float,npt.NDArray[np.float64]
            Center-of-mass energy in MeV.

        Returns
        -------
        ff: Union[complex,npt.NDArray[np.complex128]]
            Form factor from V-omega-pi.
        """
        mp = parameters.neutral_pion_mass
        mw = parameters.omega_mass

        single = np.isscalar(q)
        qq = np.atleast_1d(q).astype(np.float64) * 1e-3
        mask = qq > 1e-3 * (mp + mw)
        ff = np.zeros_like(qq, dtype=np.complex128)

        ff[mask] = self.__form_factor(
            s=qq[mask] ** 2, gvuu=couplings[0], gvdd=couplings[1]
        )

        if single:
            return ff[0]

        return ff * 1e-3

    @overload
    def integrated_form_factor(  # pylint: disable=arguments-differ
        self, q: float, couplings: Couplings
    ) -> float:
        ...

    @overload
    def integrated_form_factor(  # pylint: disable=arguments-differ
        self, q: RealArray, couplings: Couplings
    ) -> RealArray:
        ...

    def integrated_form_factor(  # pylint: disable=arguments-differ
        self, q: RealOrRealArray, couplings: Couplings
    ) -> RealOrRealArray:
        r"""Compute the pion-omega form-factor integrated over phase-space.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy.
        gvuu, gvdd: float
            Coupling of vector to up- and down-quarks.

        Returns
        -------
        iff: float or array-like
            Form-factor integrated over phase-space.
        """
        return self._integrated_form_factor(q=q, couplings=couplings)

    @overload
    def width(  # pylint: disable=arguments-differ
        self, mv: float, couplings: Couplings
    ) -> float:
        ...

    @overload
    def width(  # pylint: disable=arguments-differ
        self, mv: RealArray, couplings: Couplings
    ) -> RealArray:
        ...

    def width(  # pylint: disable=arguments-differ
        self, mv: RealOrRealArray, couplings: Couplings
    ) -> RealOrRealArray:
        r"""Compute the partial decay width of a massive vector into a pion and
        omega meson.

        Parameters
        ----------
        mv: float or array-like
            Mass of the vector.
        gvuu: float
            Coupling of vector to up-quarks.
        gvdd: float
            Coupling of vector to down-quarks.

        Returns
        -------
        width: float or array-like
            Decay width of vector into two kaons.
        """
        return self._width(mv=mv, couplings=couplings)

    @overload
    def cross_section(  # pylint: disable=arguments-differ,too-many-arguments
        self,
        q: float,
        mx: float,
        mv: float,
        gvxx: float,
        wv: float,
        couplings: Couplings,
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
    ) -> RealOrRealArray:
        r"""Compute the cross section for dark matter annihilating into two
        kaons.

        Parameters
        ----------
        q: float
            Center-of-mass energy.
        mx: float
            Mass of the dark matter in MeV.
        mv: float
            Mass of the vector mediator in MeV.
        gvxx: float
            Coupling of vector to dark matter.
        wv: float
            Width of the vector in MeV.
        gvuu: float
            Coupling of vector to up-quarks.
        gvdd: float
            Coupling of vector to down-quarks.
        gvss: float
            Coupling of vector to strange-quarks.

        Returns
        -------
        cs: float or array-like
            Annihilation cross section into two kaons.
        """
        return self._cross_section(
            q=q, mx=mx, mv=mv, gvxx=gvxx, wv=wv, couplings=couplings
        )
