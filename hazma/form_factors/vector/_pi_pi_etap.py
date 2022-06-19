from dataclasses import InitVar, dataclass, field
from typing import Tuple, Union, overload

import numpy as np
from scipy.integrate import quad

from hazma.utils import RealOrRealArray, kallen_lambda

from ._utils import FPI_GEV, METAP_GEV, MPI_GEV, RealArray
from ._base import VectorFormFactorPPP

METAP = METAP_GEV * 1e3
MPI = MPI_GEV * 1e3


@dataclass
class VectorFormFactorPiPiEtaPrimeFitData:
    """Storage class for the fit parameters of the pi-pi-eta' vector
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

    masses: RealArray = np.array([0.77549, 1.54, 1.76, 2.11])
    widths: RealArray = np.array([0.1494, 0.356, 0.113, 0.176])
    amps: RealArray = np.array([1.0, 0.0, 0.0, 0.02])
    phases: RealArray = np.array([0, np.pi, np.pi, np.pi])


@dataclass
class VectorFormFactorPiPiEtaPrime(VectorFormFactorPPP):
    r"""Class for computing the pi-pi-eta' vector form-factor.

    Attributes
    ----------
    fsp_masses: (float,float,float)
        Masses of the final state particles.
    fit_data: VectorFormFactorPiPiEtaPrimeFitData
        Fitted parameters for the pion-pion-eta vector form-factor.

    Methods
    -------
    form_factor
        Compute the un-integrated form-factor.
    integrated_form_factor
        Compute the form-factor integrated over phase-space.
    width
        Compute the decay width of a vector into pi-pi-eta'.
    cross_section
        Compute the dark matter annihilation cross section into pi-pi-eta'.
    """
    fsp_masses: Tuple[float, float, float] = field(
        init=False, default=(METAP, MPI, MPI)
    )
    _fsp_masses: Tuple[float, float, float] = field(
        init=False, default=(METAP_GEV, MPI_GEV, MPI_GEV)
    )
    fit_data: VectorFormFactorPiPiEtaPrimeFitData = field(init=False)

    masses: InitVar[RealArray] = np.array([0.77549, 1.54, 1.76, 2.11])
    widths: InitVar[RealArray] = np.array([0.1494, 0.356, 0.113, 0.176])
    amps: InitVar[RealArray] = np.array([1.0, 0.0, 0.0, 0.02])
    phases: InitVar[RealArray] = np.array([0.0, np.pi, np.pi, np.pi])

    def __post_init__(
        self,
        masses: RealArray,
        widths: RealArray,
        amps: RealArray,
        phases: RealArray,
    ):
        self.fit_data = VectorFormFactorPiPiEtaPrimeFitData(
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

    def _form_factor(self, q, s, gvuu, gvdd):
        """
        Compute the form factor for a vector decaying into two charged pions and
        an eta-prime.
        """
        pre = np.sqrt(2.0) / (4.0 * np.sqrt(3.0) * np.pi**2 * FPI_GEV**3)
        ci1 = gvuu - gvdd

        amps = self.fit_data.amps * np.exp(1j * self.fit_data.phases)
        amps /= np.sum(amps)

        return pre * ci1 * self.__bw0(s) * np.sum(amps * self.__bw(q**2))

    def form_factor(self, q, s, _, gvuu, gvdd):
        r"""Compute the form factor for a vector decaying into two pions and an
        eta'.

        Parameters
        ----------
        q:
            Center-of-mass energy in MeV.
        s: float
            Squared invariant mass of the pions s = (p2+p3)^2.
        t: float
            Squared invariant mass of the eta' and last pion t=(p1+p3)^2.
        gvuu, gvdd: float
            Coupling of vector to up-quarks and down-quarks.
        """
        qq = q * 1e-3
        ss = s * 1e-6
        ff = self._form_factor(qq, ss, gvuu, gvdd)
        return ff

    def _integrated_form_factor(self, *, q: float, gvuu: float, gvdd: float) -> float:
        """
        Compute the form factor for a vector decaying into two charged pions and
        an eta-prime integrated over the three-body phase-space.
        """
        mpi = MPI_GEV
        metap = METAP_GEV
        if q < 2 * mpi + metap:
            return 0.0

        jac = 1 / (128.0 * np.pi**3 * q**4)

        def integrand(s):
            f2 = np.abs(self._form_factor(q, s, gvuu, gvdd)) ** 2
            k1 = kallen_lambda(s, q**2, metap**2)
            k2 = kallen_lambda(s, mpi**2, mpi**2)
            return (k1 * k2) ** 1.5 * f2 / (72 * s**2)

        lb = (2 * mpi) ** 2
        ub = (q - metap) ** 2
        return jac * quad(integrand, lb, ub)[0]

    @overload
    def integrated_form_factor(self, *, q: float, gvuu: float, gvdd: float) -> float:
        ...

    @overload
    def integrated_form_factor(
        self, *, q: RealArray, gvuu: float, gvdd: float
    ) -> RealArray:
        ...

    def integrated_form_factor(
        self, q: Union[float, RealArray], *, gvuu: float, gvdd: float
    ) -> Union[float, RealArray]:
        """
        Compute the form factor for a vector decaying into two charged pions and
        an eta' integrated over the three-body phase-space.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy in MeV.
        gvuu, gvdd: float
            Vector coupling to up-quarks and down-quarks.
        """
        scalar = np.isscalar(q)
        qq = np.atleast_1d(q) * 1e-3

        integral = np.array(
            [self._integrated_form_factor(q=q_, gvuu=gvuu, gvdd=gvdd) for q_ in qq]
        )

        if scalar:
            return integral[0]

        return integral

    @overload
    def width(self, mv: float, *, gvuu: float, gvdd: float) -> float:
        ...

    @overload
    def width(self, mv: RealArray, *, gvuu: float, gvdd: float) -> RealArray:
        ...

    def width(
        self, mv: Union[float, RealArray], *, gvuu: float, gvdd: float
    ) -> Union[float, RealArray]:
        r"""Compute the partial decay width of a massive vector into an eta' and
        two pions.

        Parameters
        ----------
        mv: float
            Mass of the vector.
        gvuu, gvdd: float
            Coupling of vector to up-quarks and down-quarks.

        Returns
        -------
        width: float
            Decay width of vector into an eta' and two pions.
        """
        return 0.5 * mv * self.integrated_form_factor(q=mv, gvuu=gvuu, gvdd=gvdd)

    @overload
    def cross_section(
        self,
        *,
        q: float,
        mx: float,
        mv: float,
        gvxx: float,
        wv: float,
        gvuu: float,
        gvdd: float
    ) -> float:
        ...

    @overload
    def cross_section(
        self,
        *,
        q: RealArray,
        mx: float,
        mv: float,
        gvxx: float,
        wv: float,
        gvuu: float,
        gvdd: float
    ) -> RealArray:
        ...

    def cross_section(
        self,
        *,
        q: RealOrRealArray,
        mx: float,
        mv: float,
        gvxx: float,
        wv: float,
        gvuu: float,
        gvdd: float
    ) -> RealOrRealArray:
        r"""Compute the cross section for dark matter annihilating into an eta'
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
        gvuu, gvdd: float
            Coupling of vector to up-quarks and down-quarks.

        Returns
        -------
        cs: float or array-like
            Annihilation cross section into an eta' and two pions.
        """
        single = np.isscalar(q)
        qq = np.atleast_1d(q).astype(np.float64)

        s = qq**2
        pre = (
            gvxx**2
            * (s + 2 * mx**2)
            / (np.sqrt(s - 4 * mx**2) * ((s - mv**2) ** 2 + (mv * wv) ** 2))
        )
        pre = pre * 0.5 * qq
        cs = pre * self.integrated_form_factor(q=qq, gvuu=gvuu, gvdd=gvdd)

        if single:
            return cs[0]
        return cs
