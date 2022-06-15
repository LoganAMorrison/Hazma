"""
F_{eta,pi,pi} = (1/Z) * BW(s, 0) [
    a0*e^{i*p0}BW(q^2,0) +
    a1*e^{i*p1}BW(q^2,1) +
    a2*e^{i*p2}BW(q^2,2)
]

Z = a0*e^{i*p0} + a1*e^{i*p1} + a2*e^{i*p2}
"""


from dataclasses import dataclass, field
from typing import Tuple, Union, overload

import numpy as np
from scipy.integrate import quad

from hazma.utils import kallen_lambda

from ._utils import FPI_GEV, META_GEV, MPI_GEV, RealArray
from ._base import VectorFormFactorPPP

META = META_GEV * 1e3
MPI = MPI_GEV * 1e3


@dataclass
class VectorFormFactorPiPiEta(VectorFormFactorPPP):
    fsp_masses: Tuple[float, float, float] = field(init=False, default=(MPI, MPI, META))
    _fsp_masses: Tuple[float, float, float] = field(
        init=False, default=(MPI_GEV, MPI_GEV, META)
    )

    masses: RealArray = np.array([0.77549, 1.54, 1.76, 2.15])
    widths: RealArray = np.array([0.1494, 0.356, 0.113, 0.32])
    amps: RealArray = np.array([1.0, 0.326, 0.0115, 0.0])
    phases: RealArray = np.array([0, 3.14, 3.14, 0.0])

    def __bw0(self, s):
        m0 = self.masses[0]
        w0 = self.widths[0]
        w = (
            w0
            * m0**2
            / s
            * ((s - 4.0 * MPI_GEV**2) / (m0**2 - 4.0 * MPI_GEV**2)) ** 1.5
        )
        return m0**2 / (m0**2 - s - 1j * np.sqrt(s) * w)

    def __bw(self, s):
        w = self.widths * s / self.masses**2
        bw = self.masses**2 / (self.masses**2 - s - 1j * np.sqrt(s) * w)
        bw[0] = self.__bw0(s)
        return bw

    def _form_factor(self, cme, s, gvuu, gvdd):
        """
        Compute the form factor for a vector decaying into two charged pions and
        an eta.

        Parameters
        ----------
        q2:
            Square of the center-of-mass energy in GeV.
        """
        pre = 1.0 / (4.0 * np.sqrt(3.0) * np.pi**2 * FPI_GEV**3)
        ci1 = gvuu - gvdd

        amps = self.amps * np.exp(1j * self.phases)
        amps /= np.sum(amps)

        return pre * ci1 * self.__bw0(s) * np.sum(amps * self.__bw(cme**2))

    def form_factor(self, q, s, _, gvuu, gvdd):
        """
        Compute the form factor for a vector decaying into two charged pions and
        an eta.

        Parameters
        ----------
        q2:
            Square of the center-of-mass energy in GeV.
        """
        qq = q * 1e-3
        ss = s * 1e-6
        ff = self._form_factor(qq, ss, gvuu, gvdd)
        return ff

    def _integrated_form_factor(self, *, q: float, gvuu: float, gvdd: float) -> float:
        """
        Compute the form factor for a vector decaying into two charged pions and
        a neutral pion integrated over the three-body phase-space.

        Parameters
        ----------
        q2:
            Square of the center-of-mass energy in GeV.
        """
        mpi = MPI_GEV
        meta = META_GEV
        if q < 2 * mpi + meta:
            return 0.0

        jac = 1 / (128.0 * np.pi**3 * q**4)

        def integrand(s):
            f2 = np.abs(self._form_factor(q, s, gvuu, gvdd)) ** 2
            k1 = kallen_lambda(s, q**2, meta**2)
            k2 = kallen_lambda(s, mpi**2, mpi**2)
            return (k1 * k2) ** 1.5 * f2 / (72 * s**2)

        lb = (2 * mpi) ** 2
        ub = (q - meta) ** 2
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
        a neutral pion integrated over the three-body phase-space.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy in MeV.
        gvuu: float
            Vector coupling to up-quarks.
        gvdd: float
            Vector coupling to down-quarks.
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
