"""
Module for computing the vector form factor for pi+pi.
"""

from dataclasses import dataclass, field, InitVar
from typing import Union, Tuple

import numpy as np
from scipy.special import gamma

from hazma.utils import RealOrRealArray

from ._two_body import VectorFormFactorPP, Couplings
from ._utils import (
    MPI0_GEV,
    MPI_GEV,
    ComplexArray,
    RealArray,
    breit_wigner_fw,
    breit_wigner_gs,
    dhhatds,
    gamma_generator,
    h,
    hhat,
)


@dataclass
class VectorFormFactorPiPiFitData:  # pylint: disable=too-many-instance-attributes
    r"""Class for storing parameters used to compute vector form-factor into two pions.

    Parameters
    ----------
    n_max: int
        Maximum number of resonances.
    """
    n_max: int = field(default=2000)
    omega_mag: float = field(init=False, default=0.00187, repr=False)
    omega_phase: float = field(init=False, default=0.106, repr=False)
    omega_mass: float = field(init=False, default=0.7824, repr=False)
    omega_width: float = field(init=False, default=0.00833, repr=False)
    omega_weight: complex = field(init=False, default=0j, repr=False)

    masses: RealArray = field(init=False, repr=False)
    widths: RealArray = field(init=False, repr=False)
    coup: ComplexArray = field(init=False, repr=False)
    hres: RealArray = field(init=False, repr=False)
    h0: RealArray = field(init=False, repr=False)
    dh: RealArray = field(init=False, repr=False)

    def __post_init__(self):
        self.masses = np.zeros(self.n_max, dtype=np.float64)
        self.widths = np.zeros(self.n_max, dtype=np.float64)
        self.coup = np.zeros(self.n_max, dtype=np.complex128)
        self.hres = np.zeros(self.n_max, dtype=np.float64)
        self.h0 = np.zeros(self.n_max, dtype=np.float64)
        self.dh = np.zeros(self.n_max, dtype=np.float64)

        # Set the rho-parameters
        rho_mag = np.array([1.0, 1.0, 0.59, 4.8e-2, 0.40, 0.43], dtype=np.float64)
        rho_phase = np.array([0.0, 0.0, -2.2, -2.0, -2.9, 1.19], dtype=np.float64)
        rho_masses = np.array(
            [0.77337, 1.490, 1.870, 2.12, 2.321, 2.567], dtype=np.float64
        )
        rho_widths = np.array(
            [0.1471, 0.429, 0.357, 0.3, 0.444, 0.491], dtype=np.float64
        )
        rho_wgt = rho_mag * np.exp(1j * rho_phase)

        beta = 2.148

        # Compute the two of couplings
        self.omega_weight = self.omega_mag * np.exp(1j * self.omega_phase)
        # set up the masses and widths of the rho resonances
        ixs = np.arange(self.n_max)
        gam_b = np.array(list(gamma_generator(beta, self.n_max)))
        gam_0 = gamma(beta - 0.5)

        self.coup = (
            gam_0
            / (0.5 + ixs)
            / np.sqrt(np.pi)
            * np.sin(np.pi * (beta - 1.0 - ixs))
            / np.pi
            * gam_b
            + 0j
        )
        self.coup[0] = 1.087633403691967
        self.coup[1::2] *= -1

        # set the masses and widths
        # calc for higher resonances
        self.masses = rho_masses[0] * np.sqrt(1.0 + 2.0 * ixs)
        self.masses[: len(rho_masses)] = rho_masses

        self.widths = rho_widths[0] / rho_masses[0] * self.masses
        self.widths[: len(rho_widths)] = rho_widths

        # parameters for the gs propagators
        self.hres = np.array(
            hhat(self.masses**2, self.masses, self.widths, MPI_GEV, MPI_GEV)
        )
        self.dh = np.array(dhhatds(self.masses, self.widths, MPI_GEV, MPI_GEV))
        self.h0 = np.array(
            h(
                0.0,
                self.masses,
                self.widths,
                MPI_GEV,
                MPI_GEV,
                self.dh,
                self.hres,
            )
        )

        # fix up the early weights
        nrhowgt = len(rho_wgt)
        cwgt = np.sum(self.coup[1:nrhowgt])
        rho_sum = np.sum(rho_wgt[1:])
        self.coup[1:nrhowgt] = rho_wgt[1:] * cwgt / rho_sum


@dataclass
class _VectorFormFactorPiPiBase(VectorFormFactorPP):
    r"""Base class for computing the vector form-factor into two pions."""

    _imode: int
    fsp_masses: Tuple[float, float] = field(init=False)
    __fsp_masses: Tuple[float, float] = field(init=False, repr=False)
    fit_data: VectorFormFactorPiPiFitData = field(init=False)
    n_max: InitVar[int] = 2000

    def __post_init__(self, n_max):
        self.fit_data = VectorFormFactorPiPiFitData(n_max)

        if self._imode == 0:
            self.__fsp_masses = (MPI0_GEV, MPI0_GEV)
        else:
            self.__fsp_masses = (MPI_GEV, MPI_GEV)

        self.fsp_masses = tuple(m * 1e3 for m in self.__fsp_masses)

    def __form_factor(self, *, s: RealArray, couplings: Couplings) -> ComplexArray:
        # Convert gvuu and gvdd to iso-spin couplings
        ci1 = couplings[0] - couplings[1]

        bw = breit_wigner_gs(
            s,
            self.fit_data.masses,
            self.fit_data.widths,
            MPI_GEV,
            MPI_GEV,
            self.fit_data.h0,
            self.fit_data.dh,
            self.fit_data.hres,
            reshape=True,
        )

        ff = ci1 * self.fit_data.coup * bw

        # include rho-omega if needed
        if self._imode != 0:
            ff[:, 0] *= (
                1.0
                / (1.0 + self.fit_data.omega_weight)
                * (
                    1.0
                    + self.fit_data.omega_weight
                    * breit_wigner_fw(
                        s, self.fit_data.omega_mass, self.fit_data.omega_width
                    )
                )
            )
        # sum
        ff = np.sum(ff, axis=1)
        # factor for cc mode
        if self._imode == 0:
            ff *= np.sqrt(2.0)
        return ff

    def form_factor(  # pylint: disable=arguments-differ
        self, *, q: Union[float, RealArray], couplings: Couplings
    ) -> Union[complex, ComplexArray]:
        """Compute the pion-pion form factor.

        Parameters
        ----------
        q: float or array-like
            Center of mass energy.
        gvuu: float
            Coupling of vector to up-quarks.
        gvdd: float
            Coupling of vector to down-quarks.

        Returns
        -------
        ff: complex or array-like
            Form-factor.
        """

        single = np.isscalar(q)
        qq = np.atleast_1d(q).astype(np.float64) * 1e-3
        mask = qq > sum(self.__fsp_masses)
        ff = np.zeros_like(qq, dtype=np.complex128)

        ff[mask] = self.__form_factor(s=qq[mask] ** 2, couplings=couplings)

        if single:
            return ff[0]
        return ff

    def integrated_form_factor(  # pylint: disable=arguments-differ
        self, q: RealOrRealArray, couplings: Couplings
    ) -> RealOrRealArray:
        r"""Compute the pion-pion form-factor integrated over phase-space.

        Parameters
        ----------
        q: float
            Center-of-mass energy in MeV.
        gvuu, gvdd: float
            Coupling of vector to up- and down-quarks.

        Returns
        -------
        iff: float
            Form-factor integrated over phase-space.
        """
        return self._integrated_form_factor(q=q, couplings=couplings)

    def width(  # pylint: disable=arguments-differ
        self, mv: RealOrRealArray, couplings: Couplings
    ) -> RealOrRealArray:
        r"""Compute the partial decay width of a massive vector into two pions.

        Parameters
        ----------
        mv: float
            Mass of the vector.
        gvuu: float
            Coupling of vector to up-quarks.
        gvdd: float
            Coupling of vector to down-quarks.

        Returns
        -------
        width: float
            Decay width of vector into two pions.
        """
        return self._width(mv=mv, couplings=couplings)

    def cross_section(  # pylint: disable=arguments-differ
        self, *, q, mx: float, mv: float, gvxx: float, wv: float, couplings: Couplings
    ):
        r"""Compute the cross section for dark matter annihilating into two
        pions.

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

        Returns
        -------
        cs: float or array-like
            Annihilation cross section into two pions.
        """
        return self._cross_section(
            q=q, mx=mx, mv=mv, gvxx=gvxx, wv=wv, couplings=couplings
        )


@dataclass
class VectorFormFactorPiPi(_VectorFormFactorPiPiBase):
    r"""Class for computing the vector form-factor into two charged pions.

    Attributes
    ----------
    fsp_masses: Tuple[float, float]
        Final state particle masses.
    fit_data: VectorFormFactorPiPiFitData
        Fit information used to compute the form-factor.

    Methods
    -------
    form_factor
        Compute the un-integrated form-factor into two charged pions.
    integrated_form_factor
        Compute the form-factor into two charged pions integrated over
        phase-space.
    width
        Compute the decay width of a vector into two charged pions.
    cross_section
        Compute the dark matter annihilation cross section into two charged pions.
    """

    _imode: int = field(init=False, default=1, repr=False)


@dataclass
class VectorFormFactorPi0Pi0(_VectorFormFactorPiPiBase):
    r"""Class for computing the vector form-factor into two neutral pions.

    Attributes
    ----------
    fsp_masses: Tuple[float, float]
        Final state particle masses.
    fit_data: VectorFormFactorPiPiFitData
        Fit information used to compute the form-factor.

    Methods
    -------
    form_factor
        Compute the un-integrated form-factor into two neutral pions.
    integrated_form_factor
        Compute the form-factor into two neutral pions integrated over
        phase-space.
    width
        Compute the decay width of a vector into two neutral pions.
    cross_section
        Compute the dark matter annihilation cross section into two neutral pions.
    """

    _imode: int = field(init=False, default=0, repr=False)
