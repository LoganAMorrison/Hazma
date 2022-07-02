"""
Implementation of the eta-phi form factor.
"""

from dataclasses import InitVar, dataclass, field
from typing import Union, overload, Tuple

import numpy as np
import numpy.typing as npt

from hazma import parameters
from hazma.utils import RealOrRealArray

from ._utils import ComplexArray, RealArray, breit_wigner_fw
from ._two_body import VectorFormFactorPV, Couplings

META = parameters.eta_mass
MPHI = parameters.phi_mass


@dataclass
class VectorFormFactorEtaPhiFitData:
    r"""Storage class for the parameters needed to compute the form factor into
    an eta and phi. See arXiv:1911.11147 for details on the default values.
    """

    masses: RealArray = field(repr=False)
    widths: RealArray = field(repr=False)
    amps: RealArray = field(repr=False)
    phases: ComplexArray = field(repr=False)


@dataclass
class VectorFormFactorEtaPhi(VectorFormFactorPV):
    r"""Class for computiing the vector form factor into and eta and phi.

    Attributes
    ----------
    fsp_masses: Tuple[float, float]
        Final state particle masses.
    fit_data: VectorFormFactorEtaPhiFitData
        Fit information used to compute the form-factor.

    Methods
    -------
    form_factor
        Compute the un-integrated form-factor into an eta and phi.
    integrated_form_factor
        Compute the form-factor into an eta and phi integrated over
        phase-space.
    width
        Compute the decay width of a vector into an eta and phi.
    cross_section
        Compute the dark matter annihilation cross section into an eta and phi.
    """

    fsp_masses: Tuple[float, float] = field(init=False, default=(META, MPHI))
    fit_data: VectorFormFactorEtaPhiFitData = field(init=False)

    masses: InitVar[RealArray] = np.array([1.67, 2.14])
    widths: InitVar[RealArray] = np.array([0.122, 0.0435])
    amps: InitVar[RealArray] = np.array([0.175, 0.00409])
    phases: InitVar[RealArray] = np.array([0.0, 2.19])

    def __post_init__(
        self,
        masses: RealArray,
        widths: RealArray,
        amps: RealArray,
        phases: RealArray,
    ):
        self.fit_data = VectorFormFactorEtaPhiFitData(
            masses=masses,
            widths=widths,
            amps=amps,
            phases=np.exp(1j * phases),
        )

    def __form_factor(
        self,
        *,
        s: npt.NDArray[np.float64],
        gvss: float,
    ):
        """
        Compute the V-eta-phi form-factor.

        Uses the parameterization from arXiv:1911.11147.

        Parameters
        ----------
        s: float or np.ndarray
            Square of the center of mass energy(ies) in GeV.
        gvss: float
            Coupling of vector to strang-quarks.
        fit: FormFactorEtaOmegaFit, optional
            Fit parameters.
        """
        cs = -3.0 * gvss
        return cs * np.sum(
            self.fit_data.amps
            * self.fit_data.phases
            * breit_wigner_fw(
                s, self.fit_data.masses, self.fit_data.widths, reshape=True
            ),
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
        """Compute the eta-phi vector form-factor.

        Parameters
        ----------
        q: float or array-like
            Center of mass energy.
        gvss: float
            Coupling of vector to strange-quarks.

        Returns
        -------
        ff: complex or array-like
            Form-factor.
        """
        single = np.isscalar(q)
        qq = np.atleast_1d(q).astype(np.float64) * 1e-3
        mask = qq > 1e-3 * sum(self.fsp_masses)
        ff = np.zeros_like(qq, dtype=np.complex128)

        ff[mask] = self.__form_factor(s=qq[mask] ** 2, gvss=couplings[2])

        if single:
            return ff[0]

        return ff * 1e-3

    def integrated_form_factor(  # pylint: disable=arguments-differ
        self, q: RealOrRealArray, couplings: Couplings
    ) -> RealOrRealArray:
        r"""Compute the eta-phi form-factor integrated over phase-space.

        Parameters
        ----------
        q: float
            Center-of-mass energy.
        gvss: float
            Coupling of vector to strange-quarks.

        Returns
        -------
        iff: float
            Form-factor integrated over phase-space.
        """
        return self._integrated_form_factor(q=q, couplings=couplings)

    def width(  # pylint: disable=arguments-differ
        self, mv: RealOrRealArray, couplings: Couplings
    ) -> RealOrRealArray:
        r"""Compute the partial decay width of a massive vector into an eta and
        phi.

        Parameters
        ----------
        mv: float
            Mass of the vector.
        gvss: float
            Coupling of vector to strange-quarks.

        Returns
        -------
        width: float
            Decay width of vector into an eta and phi.
        """
        return self._width(mv=mv, couplings=couplings)

    def cross_section(  # pylint: disable=arguments-differ
        self,
        *,
        q: RealOrRealArray,
        mx: float,
        mv: float,
        gvxx: float,
        wv: float,
        couplings: Couplings,
    ) -> RealOrRealArray:
        r"""Compute the cross section for dark matter annihilating into an eta
        and phi.

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
        gvss: float
            Coupling of vector to strange-quarks.

        Returns
        -------
        cs: float or array-like
            Annihilation cross section into an eta and phi.
        """
        return self._cross_section(
            q=q,
            mx=mx,
            mv=mv,
            gvxx=gvxx,
            wv=wv,
            couplings=couplings,
        )
