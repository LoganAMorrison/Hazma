from dataclasses import dataclass, field
from typing import Union, Tuple, overload

import numpy as np
from scipy.integrate import quad

from hazma.utils import kallen_lambda

from ._utils import MOMEGA_GEV, MPI_GEV, MPI0_GEV, RealArray
from ._base import VectorFormFactorPPP


@dataclass
class _VectorFormFactorPiPiOmegaBase(VectorFormFactorPPP):
    _imode: int

    _fsp_masses: Tuple[float, float, float] = field(init=False)
    fsp_masses: Tuple[float, float, float] = field(init=False)

    masses: RealArray = np.array([0.783, 1.420, 1.6608543573197])
    widths: RealArray = np.array([0.00849, 0.315, 0.3982595005228462])
    amps: RealArray = np.array([0.0, 0.0, 2.728870588760009])
    phases: RealArray = np.array([0.0, np.pi, 0.0])

    def __post_init__(self):
        if self._imode == 0:
            self._fsp_masses = (MPI0_GEV, MPI0_GEV, MOMEGA_GEV)
        elif self._imode == 1:
            self._fsp_masses = (MPI_GEV, MPI_GEV, MOMEGA_GEV)
        self.fsp_masses = tuple(m * 1e3 for m in self._fsp_masses)

    def _form_factor(self, *, q, gvuu, gvdd):
        """
        Compute the form factor for a vector decaying into two charged pions and
        an eta-prime.
        """
        ci0 = 3 * (gvuu + gvdd)
        return ci0 * np.sum(
            self.amps
            * np.exp(1j * self.phases)
            * self.masses**2
            / ((self.masses**2 - q**2) - 1j * q * self.widths)
        )

    def form_factor(self, *, q, gvuu, gvdd):
        """
        Compute the form factor for a vector decaying into two charged pions and
        an eta-prime.
        """
        return self._form_factor(q=q, gvuu=gvuu, gvdd=gvdd)

    def _integrated_form_factor(self, *, q: float, gvuu: float, gvdd: float) -> float:
        """
        Compute the form factor for a vector decaying into two charged pions and
        an eta-prime integrated over the three-body phase-space.
        """
        if sum(self._fsp_masses) > q:
            return 0.0

        mup = self._fsp_masses[0] / q
        muo = self._fsp_masses[2] / q
        jac = 1.0 / (1536.0 * np.pi**3 * muo**2)
        f2 = np.abs(self._form_factor(q=q, gvuu=gvuu, gvdd=gvdd)) ** 2

        def integrand(z):
            k1 = kallen_lambda(z, mup**2, mup**2)
            k2 = kallen_lambda(1, z, muo**2)
            p = (1 + 10 * muo**2 + muo**4 - 2 * (1 + muo**2) * z + z**2) / z
            return p * np.sqrt(k1 * k2) * f2

        lb = (2 * mup) ** 2
        ub = (1.0 - muo) ** 2
        res = jac * quad(integrand, lb, ub)[0]

        if self._imode == 0:
            return res / 2.0
        return res

    @overload
    def integrated_form_factor(self, *, q: float, gvuu: float, gvdd: float) -> float:
        ...

    @overload
    def integrated_form_factor(
        self, *, q: RealArray, gvuu: float, gvdd: float
    ) -> RealArray:
        ...

    def integrated_form_factor(
        self, *, q: Union[float, RealArray], gvuu: float, gvdd: float
    ) -> Union[float, RealArray]:
        """
        Compute the form factor for a vector decaying into two charged pions and
        an eta-prime integrated over the three-body phase-space.
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
        q: Union[float, RealArray],
        mx: float,
        mv: float,
        gvxx: float,
        wv: float,
        gvuu: float,
        gvdd: float
    ) -> Union[float, RealArray]:
        """Compute the cross-section of dark matter annihilation."""
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


@dataclass
class VectorFormFactorPi0Pi0Omega(_VectorFormFactorPiPiOmegaBase):
    """
    Class for storing the parameters needed to compute the form factor for
    V-pi0-pi0-omega.
    """

    _imode: int = 0


@dataclass
class VectorFormFactorPiPiOmega(_VectorFormFactorPiPiOmegaBase):
    """
    Class for storing the parameters needed to compute the form factor for
    V-pi-pi-omega.
    """

    _imode: int = 1
