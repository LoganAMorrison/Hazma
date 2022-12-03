"""
Module for computing the form factor V-eta-gamma.
"""

# pylint: disable=invalid-name

from dataclasses import InitVar, dataclass, field
from typing import Tuple, Union, overload

import numpy as np

from hazma import parameters
from hazma.utils import RealOrRealArray

from ._base import vector_couplings_to_isospin
from ._two_body import Couplings, VectorFormFactorPA
from ._utils import MPI_GEV, ComplexArray, RealArray


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
    fsp_masses: Tuple[float, float] = (parameters.eta_mass, 0.0)
    fit_data: VectorFormFactorEtaGammaFitData = field(init=False)

    masses: InitVar[RealArray] = np.array([0.77526, 0.78284, 1.01952, 1.465, 1.70])
    widths: InitVar[RealArray] = np.array([0.1491, 0.00868, 0.00421, 0.40, 0.30])
    amps: InitVar[RealArray] = np.array([0.0861, 0.00824, 0.0158, 0.0147, 0.0])
    phases: InitVar[RealArray] = np.array([0.0, 11.3, 170.0, 61.0, 0.0]) * np.pi / 180.0

    def __post_init__(self, masses, widths, amps, phases):
        self.fit_data = VectorFormFactorEtaGammaFitData(
            masses=masses, widths=widths, amps=amps, phases=phases
        )

    def __form_factor(self, s: RealArray, couplings: Couplings) -> ComplexArray:
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
        ci0, ci1, cs = vector_couplings_to_isospin(*couplings)
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
    def form_factor(  # pylint: disable=arguments-differ
        self, *, q: float, couplings: Couplings
    ) -> complex:
        ...

    @overload
    def form_factor(  # pylint: disable=arguments-differ
        self, *, q: RealArray, couplings: Couplings
    ) -> ComplexArray:
        ...

    def form_factor(  # pylint: disable=arguments-differ
        self, *, q: Union[float, RealArray], couplings: Couplings
    ) -> Union[complex, ComplexArray]:
        r"""Compute the eta-photon form factor.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy in MeV.
        gvuu: float
            Coupling of vector to up-quarks.
        gvdd: float
            Coupling of vector to down-quarks.
        gvss: float
            Coupling of vector to strange-quarks.

        Returns
        -------
        ff: Union[complex,npt.NDArray[np.complex128]]
            Form factor from eta-gamma-V.
        """
        single = np.isscalar(q)
        qq = np.atleast_1d(q).astype(np.float64) * 1e-3
        mask = qq > 1e-3 * parameters.eta_mass
        ff = np.zeros_like(qq, dtype=np.complex128)

        ff[mask] = 1e-3 * self.__form_factor(qq[mask] ** 2, couplings)

        if single:
            return ff[0]

        return ff

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
        self, q: Union[float, RealArray], couplings: Couplings
    ) -> RealOrRealArray:
        r"""Compute the eta-photon form-factor integrated over phase-space.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy.
        gvuu: float
            Coupling of vector to up-quarks.
        gvdd: float
            Coupling of vector to down-quarks.
        gvss: float
            Coupling of vector to strange-quarks.

        Returns
        -------
        iff: float or array-like
            Integrated eta-photon form-factor.
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
        self, mv: Union[float, RealArray], couplings: Couplings
    ) -> RealOrRealArray:
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
            couplings=couplings,
        )
