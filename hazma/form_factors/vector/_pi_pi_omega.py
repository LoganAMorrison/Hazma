"""
Module implementing the pi-pi-omega form factor.
"""

from dataclasses import InitVar, dataclass, field
from typing import Tuple, List, Dict, overload

import numpy as np

from hazma.phase_space import PhaseSpaceDistribution1D
from hazma.utils import RealArray, RealOrRealArray, ComplexArray, ComplexOrComplexArray

from ._utils import MOMEGA_GEV, MPI_GEV, MPI0_GEV
from ._three_body import VectorFormFactorPPV, Couplings


@dataclass(frozen=True)
class VectorFormFactorPiPiOmegaFitData:
    r"""Storage class for the fit-data needed to compute pion-pion-omega
    form-factor.

    Attributes
    ----------
    masses: RealArray
        VMD resonance masses.
    widths: RealArray
        VMD resonance widths.
    amps: RealArray
        VMD resonance amplitudes.
    phases: RealArray
        VMD resonance phases.
    """

    masses: RealArray = field(repr=False)
    widths: RealArray = field(repr=False)
    amps: RealArray = field(repr=False)
    phases: RealArray = field(repr=False)


@dataclass
class _VectorFormFactorPiPiOmegaBase(VectorFormFactorPPV):
    _imode: int

    _fsp_masses: Tuple[float, float, float] = field(init=False)
    fsp_masses: Tuple[float, float, float] = field(init=False)
    fit_data: VectorFormFactorPiPiOmegaFitData = field(init=False)

    masses: InitVar[RealArray] = np.array([0.783, 1.420, 1.6608543573197])
    widths: InitVar[RealArray] = np.array([0.00849, 0.315, 0.3982595005228462])
    amps: InitVar[RealArray] = np.array([0.0, 0.0, 2.728870588760009])
    phases: InitVar[RealArray] = np.array([0.0, np.pi, 0.0])

    def __post_init__(
        self,
        masses: RealArray,
        widths: RealArray,
        amps: RealArray,
        phases: RealArray,
    ):
        if self._imode == 0:
            self._fsp_masses = (MOMEGA_GEV, MPI0_GEV, MPI0_GEV)
        elif self._imode == 1:
            self._fsp_masses = (MOMEGA_GEV, MPI_GEV, MPI_GEV)
        self.fsp_masses = tuple(m * 1e3 for m in self._fsp_masses)

        self.fit_data = VectorFormFactorPiPiOmegaFitData(
            masses=masses,
            widths=widths,
            amps=amps,
            phases=phases,
        )

    def _form_factor(self, q, *, couplings: Couplings):
        """
        Compute the form factor for a vector decaying into two charged pions and
        an eta-prime.
        """
        ci0 = 3 * (couplings[0] + couplings[1])
        return ci0 * np.sum(
            self.fit_data.amps
            * np.exp(1j * self.fit_data.phases)
            * self.fit_data.masses**2
            / ((self.fit_data.masses**2 - q**2) - 1j * q * self.fit_data.widths)
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
        Compute the form factor for a vector decaying into two charged pions and
        an eta-prime.
        """
        return self._form_factor(q=q * 1e-3, couplings=couplings)

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
        """Compute the pion-pion-omega form factor integrated over phase-space.

        Parameters
        ----------
        q: float or array-like
            Center of mass energy.
        gvuu, gvdd: float
            Coupling of vector to up- and down-quarks.

        Returns
        -------
        ff: float or array-like
            Integrated form-factor.
        """
        pre = 0.5 if self._imode == 0 else 1.0
        return pre * self._integrated_form_factor(q=q, couplings=couplings)

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
        r"""Compute the partial decay width of a massive vector into two pions and
        an omega.

        Parameters
        ----------
        mv: float or array-like
            Mass of the vector.
        gvuu, gvdd: float
            Coupling of vector to up-, and down-quarks.

        Returns
        -------
        width: float or array-like
            Decay width of vector into two pions and an omega.
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
        pions and an omega.

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
        gvuu, gvdd: float
            Coupling of vector to up- and down-quarks.

        Returns
        -------
        cs: float or array-like
            Annihilation cross section into a pion and two kaons.
        """
        return self._cross_section(
            q=q, mx=mx, mv=mv, gvxx=gvxx, wv=wv, couplings=couplings
        )

    def energy_distributions(  # pylint: disable=arguments-differ
        self, q: float, nbins: int, *, couplings: Couplings
    ) -> List[PhaseSpaceDistribution1D]:
        r"""Compute the energy distributions of the final omega and pions.

        Parameters
        ----------
        q: float or array-like
            Center of mass energy.
        nbins: int
            Number of bins for the distributions.
        gvuu, gvdd: float
            Coupling of vector to up- and down-quarks.

        Returns
        -------
        dist_om, dist_pi1, dist_pi2: (array, array)
            Three tuples containing the probabilities and energies for each
            final state particle.
        """
        return self._energy_distributions(q=q, nbins=nbins, couplings=couplings)

    def invariant_mass_distributions(  # pylint: disable=arguments-differ
        self, q: float, nbins: int, *, couplings: Couplings
    ) -> Dict[Tuple[int, int], PhaseSpaceDistribution1D]:
        r"""Compute the invariant-mass distributions of the final state
        omega and pions.

        Parameters
        ----------
        q: float or array-like
            Center of mass energy.
        nbins: int
            Number of bins for the distributions.
        gvuu, gvdd: float
            Coupling of vector to up- and down-quarks.

        Returns
        -------
        dist_om, dist_pi1, dist_pi2: (array, array)
            Three tuples containing the probabilities and energies for each
            final state particle.
        """
        return self._invariant_mass_distributions(q=q, nbins=nbins, couplings=couplings)


@dataclass
class VectorFormFactorPi0Pi0Omega(_VectorFormFactorPiPiOmegaBase):
    r"""Class for computing the pi0-pi0-omega vector form factor.

    Attributes
    ----------
    fsp_masses: (float, float, float)
        Masses of the final-state particles.
    fit_data: VectorFormFactorPiPiOmegaFitData
        Stored data used to compute form-factor.

    Methods
    -------
    form_factor
        Compute the un-integrated form-factor into two neutral pions and an
        omega.
    integrated_form_factor
        Compute the form-factor into two neutral pions and an omega integrated
        over phase-space.
    width
        Compute the decay width of a vector into two neutral pions and an omega.
    cross_section
        Compute the dark matter annihilation cross section into two neutral
        pions and an omega.
    """

    _imode: int = 0


@dataclass
class VectorFormFactorPiPiOmega(_VectorFormFactorPiPiOmegaBase):
    r"""Class for computing the pi-pi-omega vector form factor.

    Attributes
    ----------
    fsp_masses: (float, float, float)
        Masses of the final-state particles.
    fit_data: VectorFormFactorPiPiOmegaFitData
        Stored data used to compute form-factor.

    Methods
    -------
    form_factor
        Compute the un-integrated form-factor into two charged pions and an
        omega.
    integrated_form_factor
        Compute the form-factor into two charged pions and an omega integrated
        over phase-space.
    width
        Compute the decay width of a vector into two charged pions and an omega.
    cross_section
        Compute the dark matter annihilation cross section into two charged
        pions and an omega.
    """

    _imode: int = 1
