"""
Implementation of the eta-omega form factor.
"""

from dataclasses import InitVar, dataclass, field
from typing import overload, Tuple

import numpy as np

from hazma import parameters
from hazma.utils import RealOrRealArray, ComplexOrComplexArray, ComplexArray, RealArray

from ._utils import breit_wigner_fw
from ._two_body import VectorFormFactorPV, Couplings
from ._base import vector_couplings_to_isospin

META = parameters.eta_mass
MOMEGA = parameters.omega_mass


@dataclass(frozen=True)
class VectorFormFactorEtaOmegaFitData:
    r"""Storage class for the eta-omega vector form-factor. See arXiv:1911.11147
    for details on the default values.
    """

    # w', w''' parameters
    masses: RealArray = field(repr=False)
    widths: RealArray = field(repr=False)
    amps: RealArray = field(repr=False)
    phases: RealArray = field(repr=False)


@dataclass
class VectorFormFactorEtaOmega(VectorFormFactorPV):
    r"""Class for computing the eta-omega vector form-factor.

    Attributes
    ----------
    fsp_masses: (float, float)
        Masses of the final-state particles.
    fit_data: VectorFormFactorEtaOmegaFitData
        Stored data used to compute form-factor.

    Methods
    -------
    form_factor
        Compute the un-integrated form-factor into an eta and omega.
    integrated_form_factor
        Compute the form-factor into an eta and omega integrated over
        phase-space.
    width
        Compute the decay width of a vector into an eta and omega.
    cross_section
        Compute the dark matter annihilation cross section into an eta and
        omega.
    """

    fsp_masses: Tuple[float, float] = field(init=False, default=(META, MOMEGA))
    fit_data: VectorFormFactorEtaOmegaFitData = field(init=False)

    # w', w''' parameters
    masses: InitVar[RealArray] = np.array([1.43, 1.67])
    widths: InitVar[RealArray] = np.array([0.215, 0.113])
    amps: InitVar[RealArray] = np.array([0.0862, 0.0648])
    phases: InitVar[RealArray] = np.exp(1j * np.array([0.0, np.pi]))

    def __post_init__(
        self,
        masses: RealArray,
        widths: RealArray,
        amps: RealArray,
        phases: RealArray,
    ):
        self.fit_data = VectorFormFactorEtaOmegaFitData(
            masses=masses,
            widths=widths,
            amps=amps,
            phases=phases,
        )

    def __form_factor(self, *, s: RealArray, couplings: Couplings):
        """
        Compute the V-eta-omega form-factor.

        Uses the parameterization from arXiv:1911.11147.

        Parameters
        ----------
        s: float or np.ndarray
            Square of the center of mass energy(ies).
        gvuu: float
            Coupling of vector to up-quarks.
        gvdd: float
            Coupling of vector to down-quarks.
        fit: FormFactorEtaOmegaFit, optional
            Fit parameters.
        """
        # NOTE: This is a rescaled-version of the form factor defined in
        # arXiv:2201.01788. (multipled by energy to make it unitless)
        ci0 = vector_couplings_to_isospin(*couplings)[0]
        return ci0 * np.sum(
            self.fit_data.amps
            * self.fit_data.phases
            * breit_wigner_fw(
                s, self.fit_data.masses, self.fit_data.widths, reshape=True
            ),
            axis=1,
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
        self, q: RealOrRealArray, couplings: Couplings
    ) -> ComplexOrComplexArray:
        """
        Compute the V-eta-omega form factor.

        Parameters
        ----------
        q: Union[float,npt.NDArray[np.float64]
            Center-of-mass energy in MeV.

        Returns
        -------
        ff: Union[complex,npt.NDArray[np.complex128]]
            Form factor from V-eta-omega.
        """
        me = parameters.eta_mass
        mw = parameters.omega_mass

        single = np.isscalar(q)
        qq = np.atleast_1d(q).astype(np.float64) * 1e-3
        mask = qq > 1e-3 * (me + mw)
        ff = np.zeros_like(qq, dtype=np.complex128)

        ff[mask] = self.__form_factor(s=qq[mask] ** 2, couplings=couplings)

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
        r"""Compute the eta-omega form-factor integrated over phase-space.

        Parameters
        ----------
        q: float
            Center-of-mass energy.
        gvuu, gvdd: float
            Coupling of vector to up and down-quarks.

        Returns
        -------
        iff: float
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
        r"""Compute the partial decay width of a massive vector into an eta and
        omega.

        Parameters
        ----------
        mv: float
            Mass of the vector.
        gvuu, gvdd: float
            Coupling of vector to up- and down-quarks.

        Returns
        -------
        width: float
            Decay width of vector into an eta and omega.
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
        and omega.

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
        gvuu, gvdd: float
            Coupling of vector to up and down-quarks.

        Returns
        -------
        cs: float or array-like
            Annihilation cross section into an eta and omega.
        """
        return self._cross_section(
            q=q,
            mx=mx,
            mv=mv,
            gvxx=gvxx,
            wv=wv,
            couplings=couplings,
        )
