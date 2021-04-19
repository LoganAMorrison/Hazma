"""
Module for computing the vector form factor for pi+pi.
"""
from typing import List, Dict

import numpy as np  # type:ignore
from scipy.special import gamma  # type:ignore

from hazma.parameters import (
    neutral_pion_mass as mpi0,
    charged_pion_mass as mpi,
    Qu,
    Qd,
    Qe,
    qe,
)

from hazma.vector_mediator.form_factors.utils import (
    hhat,
    dhhatds,
    h,
    breit_wigner_gs,
    breit_wigner_fw,
)

# Pion mass in GeV
_MPI = 0.13957018


class FormFactorPiPi:
    """
    Class for computing the pi-pi-vector form-factor.
    """

    def __init__(self, ci0: float, ci1: float) -> None:
        """
        Initialize the vector form-factor for.
        """

        self._ci0: float = ci0
        self._ci1: float = ci1

        # truncation parameter
        self._n_max: int = 2000
        # omega parameters, relevant for the rho-omega mixing in the 0th
        # order rho resonance
        self._omega: Dict[str, float] = {
            "mag": 0.00187,
            "phase": 0.106,
            "mass": 0.7824,
            "width": 0.00833,
            "wgt": 0.0,
        }

        self._beta: float = 2.148
        # rho parameters, for 0th to 5th order rho resonances
        self._rho: Dict[str, List[float]] = {
            "mag": [1.0, 1.0, 0.59, 4.8e-2, 0.40, 0.43],
            "phase": [0.0, 0.0, -2.2, -2.0, -2.9, 1.19],
            "mass": [0.77337, 1.490, 1.870, 2.12, 2.321, 2.567],
            "width": [0.1471, 0.429, 0.357, 0.3, 0.444, 0.491],
            "wgt": [],
        }

        self._mass: List[float] = []
        self._width: List[float] = []
        self._coup: List[float] = []
        self._hres: List[float] = []
        self._h0: List[float] = []
        self._dh: List[float] = []

        self._initialize()

    @property
    def ci0(self) -> float:
        return self._ci0

    @ci0.setter
    def ci0(self, val: float) -> None:
        self._ci0 = val

    @property
    def ci1(self) -> float:
        return self._ci1

    @ci1.setter
    def ci1(self, val: float) -> None:
        self._ci1 = val
        self._initialize()

    # initialize tower of couplings
    def _initialize(self) -> None:
        rho_sum = 0.0
        # print self._rho_wgt
        for ix, (mag, phase) in enumerate(zip(self._rho["mag"], self._rho["phase"])):
            self._rho["wgt"].append(mag * np.exp(1j * phase))
            if ix > 0:
                rho_sum += self._rho["wgt"][ix]
        self._omega_wgt = self._omega["mag"] * np.exp(1j * self._omega["phase"])
        # set up the masses and widths of the rho resonances
        gam_b = gamma(2.0 - self._beta)
        cwgt = 0.0
        for ix in range(self._n_max):
            # this is gam(2-beta+n)/gam(n+1)
            if ix > 0:
                gam_b *= ((1.0 - self._beta + float(ix))) / float(ix)
            c_n = (
                gamma(self._beta - 0.5)
                / (0.5 + float(ix))
                / np.sqrt(np.pi)
                * np.sin(np.pi * (self._beta - 1.0 - float(ix)))
                / np.pi
                * gam_b
            )
            if ix % 2 != 0:
                c_n *= -1.0
            if ix == 0:
                c_n = 1.087633403691967
            # set the masses and widths
            # calc for higher resonances
            if ix >= len(self._rho["mass"]):
                self._mass.append(self._rho["mass"][0] * np.sqrt(1.0 + 2.0 * float(ix)))
                self._width.append(
                    self._rho["width"][0] / self._rho["mass"][0] * self._mass[-1]
                )
            # input for lower ones
            else:
                self._mass.append(self._rho["mass"][ix])
                self._width.append(self._rho["width"][ix])
            if ix > 0 and ix < len(self._rho["wgt"]):
                cwgt += c_n
            # parameters for the gs propagators
            self._hres.append(
                hhat(
                    self._mass[-1] ** 2,
                    self._mass[-1],
                    self._width[-1],
                    self._mpi,
                    self._mpi,
                )
            )
            self._dh.append(dhhatds(self._mass[-1], self._width[-1], _MPI, _MPI))
            self._h0.append(
                h(
                    0.0,
                    self._mass[-1],
                    self._width[-1],
                    _MPI,
                    _MPI,
                    self._dh[-1],
                    self._hres[-1],
                )
            )
            self._coup.append(self._ci1 * c_n)
        # fix up the early weights
        for ix in range(1, len(self._rho["wgt"])):
            # print ix
            self._coup[ix] = self._ci1 * self._rho["wgt"][ix] * cwgt / rho_sum

    def __call__(self, q2, imode):
        ff = 0.0 + 0j
        # print coup_[0]
        for ix in range(self._n_max):
            term = self._coup[ix] * breit_wigner_gs(
                q2,
                self._mass[ix],
                self._width[ix],
                _MPI,
                _MPI,
                self._h0[ix],
                self._dh[ix],
                self._hres[ix],
            )
            # include rho-omega if needed
            if ix == 0 and imode != 0:
                term *= (
                    1.0
                    / (1.0 + self._omega_wgt)
                    * (
                        1.0
                        + self._omega_wgt
                        * breit_wigner_fw(
                            q2, self._omega["mass"], self._omega["weight"]
                        )
                    )
                )
            # sum
            ff += term
        # factor for cc mode
        if imode == 0:
            ff *= np.sqrt(2.0)
        return ff
