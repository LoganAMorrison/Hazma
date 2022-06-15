"""
Module for computing the vector form factor for pi+pi.
"""
from dataclasses import dataclass, field
from typing import Union, Tuple

import numpy as np
from scipy.special import gamma

from ._base import VectorFormFactorPP
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
class _VectorFormFactorPiPiBase(VectorFormFactorPP):
    """
    Class for storing the parameters needed to compute the form factor for
    V-pi-pi.
    """

    _imode: int
    _fsp_masses: Tuple[float, float] = field(init=False)

    omega_mag: float = field(default=0.00187)
    omega_phase: float = field(default=0.106)
    omega_mass: float = field(default=0.7824)
    omega_width: float = field(default=0.00833)
    omega_weight: complex = field(default=0j)

    n_max: int = field(default=2000, kw_only=True)

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

        if self._imode == 0:
            self._fsp_masses = (MPI0_GEV, MPI0_GEV)
        else:
            self._fsp_masses = (MPI_GEV, MPI_GEV)

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
        gam_b = np.array([val for val in gamma_generator(beta, self.n_max)])
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
            h(0.0, self.masses, self.widths, MPI_GEV, MPI_GEV, self.dh, self.hres)
        )

        # fix up the early weights
        nrhowgt = len(rho_wgt)
        cwgt = np.sum(self.coup[1:nrhowgt])
        rho_sum = np.sum(rho_wgt[1:])
        self.coup[1:nrhowgt] = rho_wgt[1:] * cwgt / rho_sum

    def __form_factor(self, *, s: RealArray, gvuu: float, gvdd: float) -> ComplexArray:
        # Convert gvuu and gvdd to iso-spin couplings
        ci1 = gvuu - gvdd

        bw = breit_wigner_gs(
            s,
            self.masses,
            self.widths,
            MPI_GEV,
            MPI_GEV,
            self.h0,
            self.dh,
            self.hres,
            reshape=True,
        )

        ff = ci1 * self.coup * bw

        # include rho-omega if needed
        if self._imode != 0:
            ff[:, 0] *= (
                1.0
                / (1.0 + self.omega_weight)
                * (
                    1.0
                    + self.omega_weight
                    * breit_wigner_fw(s, self.omega_mass, self.omega_width)
                )
            )
        # sum
        ff = np.sum(ff, axis=1)
        # factor for cc mode
        if self._imode == 0:
            ff *= np.sqrt(2.0)
        return ff

    def form_factor(
        self, *, q: Union[float, RealArray], gvuu, gvdd
    ) -> Union[complex, ComplexArray]:
        """
        Compute the pi-pi-V form factor.

        Parameters
        ----------
        s: Union[float,RealArray
            Squared center-of-mass energy in MeV.
        imode: int, optional
            Iso-spin channel. Default is 1.

        Returns
        -------
        ff: Union[complex,ComplexArray]
            Form factor from pi-pi-V.
        """

        single = np.isscalar(q)
        qq = np.atleast_1d(q).astype(np.float64) * 1e-3
        mask = qq > sum(self._fsp_masses)
        ff = np.zeros_like(qq, dtype=np.complex128)

        ff[mask] = self.__form_factor(
            s=qq[mask] ** 2,
            gvuu=gvuu,
            gvdd=gvdd,
        )

        if single:
            return ff[0]
        return ff

    def width(
        self, mv: Union[float, RealArray], gvuu, gvdd
    ) -> Union[complex, ComplexArray]:
        fsp_masses = tuple(m * 1e3 for m in self._fsp_masses)
        return self._width(mv=mv, fsp_masses=fsp_masses, gvuu=gvuu, gvdd=gvdd)


@dataclass
class VectorFormFactorPiPi(_VectorFormFactorPiPiBase):
    """
    Class for storing the parameters needed to compute the form factor for
    V-pi-pi.
    """

    _imode: int = 1


@dataclass
class VectorFormFactorPi0Pi0(_VectorFormFactorPiPiBase):
    """
    Class for storing the parameters needed to compute the form factor for
    V-pi-pi.
    """

    _imode: int = 0
