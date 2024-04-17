"""
Module implementing the pi-pi-eta form factor.
"""

from dataclasses import dataclass, field, InitVar
from typing import Tuple, Union, List, Dict, overload

import numpy as np

from hazma.phase_space import PhaseSpaceDistribution1D
from hazma.utils import RealArray, ComplexArray, RealOrRealArray, ComplexOrComplexArray

from ._utils import FPI_GEV, META_GEV, MPI_GEV
from ._three_body import VectorFormFactorPPP2, Couplings

META = META_GEV * 1e3
MPI = MPI_GEV * 1e3


@dataclass
class VectorFormFactorPiPiEtaFitData:
    r"""Class for computing the pi-pi-eta vector form-factor.

    Attributes
    ----------
    masses: np.ndarray
        Fitted resonance masses of the VMD amplitudes.
    widths: np.ndarray
        Fitted resonance widths of the VMD amplitudes.
    amps: np.ndarray
        Fitted amplitudes of the VMD amplitudes.
    phases: np.ndarray
        Fitted phases of the VMD amplitudes.
    """

    masses: RealArray = field(repr=False)
    widths: RealArray = field(repr=False)
    amps: RealArray = field(repr=False)
    phases: RealArray = field(repr=False)


@dataclass
class VectorFormFactorPiPiEta(VectorFormFactorPPP2):
    r"""Class for computing the pi-pi-eta vector form-factor.

    Attributes
    ----------
    fsp_masses: (float,float,float)
        Masses of the final state particles.
    fit_data: VectorFormFactorPiPiEtaFitData
        Fitted parameters for the pion-pion-eta vector form-factor.

    Methods
    -------
    form_factor
        Compute the un-integrated form-factor.
    integrated_form_factor
        Compute the form-factor integrated over phase-space.
    width
        Compute the decay width of a vector into pi-pi-eta.
    cross_section
        Compute the dark matter annihilation cross section into pi-pi-eta.
    """

    fsp_masses: tuple[float, float, float] = field(init=False, default=(META, MPI, MPI))
    fit_data: VectorFormFactorPiPiEtaFitData = field(init=False)

    masses: InitVar[RealArray] = field(default=np.array([0.77549, 1.54, 1.76, 2.15]))
    widths: InitVar[RealArray] = field(default=np.array([0.1494, 0.356, 0.113, 0.32]))
    amps: InitVar[RealArray] = field(default=np.array([1.0, 0.326, 0.0115, 0.0]))
    phases: InitVar[RealArray] = field(default=np.array([0, 3.14, 3.14, 0.0]))

    def __post_init__(
        self,
        masses: RealArray,
        widths: RealArray,
        amps: RealArray,
        phases: RealArray,
    ):
        self.fit_data = VectorFormFactorPiPiEtaFitData(
            masses=masses,
            widths=widths,
            amps=amps,
            phases=phases,
        )

    def __bw0(self, s):
        m0 = self.fit_data.masses[0]
        w0 = self.fit_data.widths[0]
        w = (
            w0
            * m0**2
            / s
            * ((s - 4.0 * MPI_GEV**2) / (m0**2 - 4.0 * MPI_GEV**2)) ** 1.5
        )
        return m0**2 / (m0**2 - s - 1j * np.sqrt(s) * w)

    def __bw(self, s):
        w = self.fit_data.widths * s / self.fit_data.masses**2
        bw = self.fit_data.masses**2 / (
            self.fit_data.masses**2 - s - 1j * np.sqrt(s) * w
        )
        bw[0] = self.__bw0(s)
        return bw

    def _form_factor(self, cme, s, couplings: Couplings):
        """
        Compute the form factor for a vector decaying into two charged pions and
        an eta.

        Parameters
        ----------
        q2:
            Square of the center-of-mass energy in GeV.
        """
        pre = 1.0 / (4.0 * np.sqrt(3.0) * np.pi**2 * FPI_GEV**3)
        ci1 = couplings[0] - couplings[1]

        amps = self.fit_data.amps * np.exp(1j * self.fit_data.phases)
        amps /= np.sum(amps)

        return pre * ci1 * self.__bw0(s) * np.sum(amps * self.__bw(cme**2))

    @overload
    def form_factor(  # pylint: disable=arguments-differ
        self, q: float, s: float, couplings: Couplings
    ) -> complex: ...

    @overload
    def form_factor(  # pylint: disable=arguments-differ
        self, q: float, s: RealArray, couplings: Couplings
    ) -> ComplexArray: ...

    def form_factor(  # pylint: disable=arguments-differ
        self, q: float, s: RealOrRealArray, couplings: Couplings
    ) -> ComplexOrComplexArray:
        """
        Compute the form factor for a vector decaying into two charged pions and
        an eta.

        Parameters
        ----------
        q:
            Center-of-mass energy in MeV.
        s: float
            Squared invariant mass of the pions s = (p2+p3)^2.
        t: float
            Squared invariant mass of the eta and last pion t=(p1+p3)^2.
        gvuu: float
            Coupling of vector to up-quarks.
        gvdd: float
            Coupling of vector to down-quarks.
        """
        if q < sum(self.fsp_masses):
            return 0.0

        qq = q * 1e-3
        ss = s * 1e-6
        ff = self._form_factor(qq, ss, couplings=couplings) * 1e-9
        return ff

    @overload
    def integrated_form_factor(  # pylint: disable=arguments-differ
        self, q: float, couplings: Couplings
    ) -> float: ...

    @overload
    def integrated_form_factor(  # pylint: disable=arguments-differ
        self, q: RealArray, couplings: Couplings
    ) -> RealArray: ...

    def integrated_form_factor(  # pylint: disable=arguments-differ
        self, q: Union[float, RealArray], couplings: Couplings
    ) -> Union[float, RealArray]:
        """
        Compute the form factor for a vector decaying into two charged pions and
        an eta, integrated over phase-space.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy in MeV.
        gvuu: float
            Vector coupling to up-quarks.
        gvdd: float
            Vector coupling to down-quarks.
        """
        return self._integrated_form_factor(q=q, couplings=couplings)

    @overload
    def width(  # pylint: disable=arguments-differ
        self, mv: float, couplings: Couplings
    ) -> float: ...

    @overload
    def width(  # pylint: disable=arguments-differ
        self, mv: RealArray, couplings: Couplings
    ) -> RealArray: ...

    def width(  # pylint: disable=arguments-differ
        self, mv: RealOrRealArray, couplings: Couplings
    ) -> RealOrRealArray:
        r"""Compute the partial decay width of a massive vector into an eta and
        two pions.

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
            Decay width of vector into an eta and two pions.
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
    ) -> float: ...

    @overload
    def cross_section(  # pylint: disable=arguments-differ,too-many-arguments
        self,
        q: RealArray,
        mx: float,
        mv: float,
        gvxx: float,
        wv: float,
        couplings: Couplings,
    ) -> RealArray: ...

    def cross_section(  # pylint: disable=arguments-differ,too-many-arguments
        self,
        q: Union[float, RealArray],
        mx: float,
        mv: float,
        gvxx: float,
        wv: float,
        couplings: Couplings,
    ) -> Union[float, RealArray]:
        r"""Compute the cross section for dark matter annihilating into an eta
        and two pions.

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
            Annihilation cross section into an eta and two pions.
        """
        return self._cross_section(
            q=q, mx=mx, mv=mv, gvxx=gvxx, wv=wv, couplings=couplings
        )

    def energy_distributions(  # pylint: disable=arguments-differ
        self, q: float, nbins: int, *, couplings: Couplings
    ) -> List[PhaseSpaceDistribution1D]:
        r"""Compute the energy distributions of the final state pions and eta.

        Parameters
        ----------
        q: float
            Center-of-mass energy.
        nbins: float
            Number of bins used to generate distribution.
        gvuu, gvdd: float
            Coupling of vector to up- and down-quarks.

        Returns
        -------
        dists: List[PhaseSpaceDistribution1D]
            List of the energy distributions.
        """
        return self._energy_distributions(q=q, nbins=nbins, couplings=couplings)

    def invariant_mass_distributions(  # pylint: disable=arguments-differ
        self, q: float, nbins: int, *, couplings: Couplings
    ) -> Dict[Tuple[int, int], PhaseSpaceDistribution1D]:
        r"""Compute the invariant-mass distributions of the all pairs of the
        final-state particles.

        Parameters
        ----------
        q: float
            Center-of-mass energy.
        nbins: float
            Number of bins used to generate distribution.
        gvuu, gvdd: float
            Coupling of vector to up- and down-quarks.

        Returns
        -------
        dists: Dict[(int,int), PhaseSpaceDistribution1D]
            Dictionary of the invariant-mass distributions. Keys specify the
            pair of particles the distribution represents.
        """
        return self._invariant_mass_distributions(q=q, nbins=nbins, couplings=couplings)
