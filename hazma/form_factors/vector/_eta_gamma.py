"""
Module for computing the form factor V-eta-gamma.
"""
from dataclasses import dataclass, field, InitVar
from typing import Union, overload, Tuple

import numpy as np

from hazma import parameters
from hazma.utils import RealOrRealArray

from ._utils import MPI_GEV, ComplexArray, RealArray
from ._base import VectorFormFactorPA


@dataclass(frozen=True)
class VectorFormFactorEtaGammaFitData:
    r"""Storage class for the eta-photon form-factor."""

    masses: RealArray = np.array([0.77526, 0.78284, 1.01952, 1.465, 1.70])
    widths: RealArray = np.array([0.1491, 0.00868, 0.00421, 0.40, 0.30])
    amps: RealArray = np.array([0.0861, 0.00824, 0.0158, 0.0147, 0.0])
    phases: RealArray = np.array([0.0, 11.3, 170.0, 61.0, 0.0]) * np.pi / 180.0


@dataclass
class VectorFormFactorEtaGamma(VectorFormFactorPA):
    r""" "Class for computing the eta-photon form factor.

    Attributes
    ----------
    fsp_masses: Tuple[float]
        Final state particle masses (only eta in this case.)
    fit_data: VectorFormFactorEtaGammaFitData
        Fit information used to compute the form-factor.

    Methods
    -------
    form_factor
        Compute the un-integrated form-factor into an eta and photon.
    integrated_form_factor
        Compute the form-factor into an eta and photon integrated over
        phase-space.
    width
        Compute the decay width of a vector into an eta and photon .
    cross_section
        Compute the dark matter annihilation cross section into an eta and
        photon.

    """
    fsp_masses: Tuple[float] = field(init=False, default=(parameters.eta_mass,))
    fit_data: VectorFormFactorEtaGammaFitData = field(init=False)

    masses: InitVar[RealArray] = field(
        default=np.array([0.77526, 0.78284, 1.01952, 1.465, 1.70])
    )
    widths: InitVar[RealArray] = field(
        default=np.array([0.1491, 0.00868, 0.00421, 0.40, 0.30])
    )
    amps: InitVar[RealArray] = field(
        default=np.array([0.0861, 0.00824, 0.0158, 0.0147, 0.0])
    )
    phases: InitVar[RealArray] = field(
        default=np.array([0.0, 11.3, 170.0, 61.0, 0.0]) * np.pi / 180.0
    )

    def __post_init__(self, masses, widths, amps, phases):
        self.fit_data = VectorFormFactorEtaGammaFitData(
            masses=masses, widths=widths, amps=amps, phases=phases
        )

    def __form_factor(
        self, s: RealArray, gvuu: float, gvdd: float, gvss: float
    ) -> ComplexArray:
        """
        Compute the form factor for V-eta-gamma at given squared center of mass
        energ(ies).

        Parameters
        ----------
        s: Union[float, np.ndarray]
            Array of squared center-of-mass energies or a single value.
        gvuu : float
            Coupling of vector mediator to the up quark.
        gvdd : float
            Coupling of vector mediator to the down quark.
        gvss : float
            Coupling of vector mediator to the strange quark.

        Returns
        -------
        ff: Union[float, np.ndarray]
            The form factors.
        """
        ci0 = 3.0 * (gvuu + gvdd)
        ci1 = gvuu - gvdd
        cs = -3.0 * gvss

        c_rho_om_phi = np.array([ci1, ci0, cs, ci1, cs])

        ss = s[:, np.newaxis]
        q = np.sqrt(ss)
        dens = self.fit_data.masses**2 - ss - 1j * q * self.fit_data.widths
        dens[:, 0:1] = (
            self.fit_data.masses[0] ** 2
            - ss
            - 1j
            * q
            * (
                self.fit_data.widths[0]
                * self.fit_data.masses[0] ** 2
                / ss
                * (
                    (ss - 4 * MPI_GEV**2)
                    / (self.fit_data.masses[0] ** 2 - 4 * MPI_GEV**2)
                )
                ** 1.5
            )
        )

        return np.sum(
            c_rho_om_phi
            * self.fit_data.amps
            * self.fit_data.masses**2
            * np.exp(1j * self.fit_data.phases)
            / dens,
            axis=1,
        )

    @overload
    def form_factor(
        self, *, q: float, gvuu: float, gvdd: float, gvss: float
    ) -> complex:
        ...

    @overload
    def form_factor(
        self, *, q: RealArray, gvuu: float, gvdd: float, gvss: float
    ) -> ComplexArray:
        ...

    def form_factor(
        self, *, q: Union[float, RealArray], gvuu, gvdd, gvss
    ) -> Union[complex, ComplexArray]:
        """
        Compute the eta-gamma-V form factor.

        Parameters
        ----------
        s: Union[float,npt.NDArray[np.float64]
            Square of the center-of-mass energy in MeV.

        Returns
        -------
        ff: Union[complex,npt.NDArray[np.complex128]]
            Form factor from eta-gamma-V.
        """
        single = np.isscalar(q)
        qq = np.atleast_1d(q).astype(np.float64) * 1e-3
        mask = qq > 1e-3 * parameters.eta_mass
        ff = np.zeros_like(qq, dtype=np.complex128)

        ff[mask] = 1e-3 * self.__form_factor(
            qq[mask] ** 2,
            gvuu,
            gvdd,
            gvss,
        )

        if single:
            return ff[0]

        return ff

    def width(
        self, mv: Union[float, RealArray], gvuu: float, gvdd: float, gvss: float
    ) -> Union[complex, ComplexArray]:
        r"""Compute the partial decay width of a massive vector into an eta and
        photon.

        Parameters
        ----------
        mv: float or array-like
            Mass of the vector.
        gvuu: float
            Coupling of vector to up-quarks.
        gvdd: float
            Coupling of vector to down-quarks.
        gvss: float
            Coupling of vector to strange-quarks.

        Returns
        -------
        width: float or array-like
            Decay width of vector into an eta and photon.
        """
        return self._width(mv=mv, gvuu=gvuu, gvdd=gvdd, gvss=gvss)

    def cross_section(
        self,
        *,
        q: RealOrRealArray,
        mx: float,
        mv: float,
        gvxx: float,
        wv: float,
        gvuu: float,
        gvdd: float,
        gvss: float,
    ) -> RealOrRealArray:
        r"""Compute the cross section for dark matter annihilating into an eta
        and a photon.

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
        gvuu: float
            Coupling of vector to up-quarks.
        gvdd: float
            Coupling of vector to down-quarks.
        gvss: float
            Coupling of vector to strange-quarks.

        Returns
        -------
        cs: float or array-like
            Annihilation cross section into an eta and a photon.
        """
        return self._cross_section(
            q=q,
            mx=mx,
            mv=mv,
            gvxx=gvxx,
            wv=wv,
            gvuu=gvuu,
            gvdd=gvdd,
            gvss=gvss,
        )
