from dataclasses import InitVar, dataclass, field
from typing import Union, overload, Tuple

import numpy as np

from hazma import parameters
from hazma.utils import RealOrRealArray

from ._utils import MPI0_GEV, ComplexArray, RealArray
from ._base import VectorFormFactorPV

MPI0 = parameters.neutral_pion_mass
MPHI = parameters.phi_mass

# old fit values
# amp = [0.045, 0.0315, 0.0]
# phase = [180.0, 0.0, 180.0]

# Uncertainties
# amp1=0.0825858193110437
# amp2=0.004248886307513855
# amp3=0.0
# phase1=0.0
# phase2=16.826357320477726
# phase3=0.0


@dataclass
class VectorFormFactorPi0PhiFitData:
    r"""Storage class for the parameters needed to compute pion-phi vector
    form-factor. See arXiv:1911.11147 for details on the default values.
    """

    amps: RealArray = field(repr=False)
    phases: RealArray = field(repr=False)
    rho_masses: RealArray = field(repr=False)
    rho_widths: RealArray = field(repr=False)
    br4pi: RealArray = field(repr=False)


@dataclass
class VectorFormFactorPi0Phi(VectorFormFactorPV):
    r""" "Class for computing the pion-phi form factor.

    Attributes
    ----------
    fsp_masses: Tuple[float]
        Final state particle masses.
    fit_data: VectorFormFactorPi0PhiFitData
        Fit information used to compute the form-factor.

    Methods
    -------
    form_factor
        Compute the un-integrated form-factor into a pion and phi.
    integrated_form_factor
        Compute the form-factor into a pion and phi integrated over phase-space.
    width
        Compute the decay width of a vector into a pion and phi .
    cross_section
        Compute the dark matter annihilation cross section into a pion and phi.
    """

    fsp_masses: Tuple[float, float] = field(init=False, default=(MPI0, MPHI))
    fit_data: VectorFormFactorPi0PhiFitData = field(init=False)

    amps: InitVar[RealArray] = np.array([0.177522453644825, 0.023840592398187477, 0.0])
    phases: InitVar[RealArray] = (
        np.array([0.0, 123.82008351626034, 0.0]) * np.pi / 180.0
    )
    rho_masses: InitVar[RealArray] = np.array([0.77526, 1.593, 1.909])
    rho_widths: InitVar[RealArray] = np.array([0.1491, 0.203, 0.048])
    br4pi: InitVar[RealArray] = np.array([0.0, 0.33, 0.0])

    def __post_init__(
        self,
        amps: RealArray,
        phases: RealArray,
        rho_masses: RealArray,
        rho_widths: RealArray,
        br4pi: RealArray,
    ):
        self.fit_data = VectorFormFactorPi0PhiFitData(
            amps=amps,
            phases=phases,
            rho_masses=rho_masses,
            rho_widths=rho_widths,
            br4pi=br4pi,
        )

    def __form_factor(
        self, *, s: Union[RealArray, float], gvuu: float, gvdd: float
    ) -> Union[ComplexArray, complex]:
        """
        Compute the V-phi-pi form-factor.

        Uses the parameterization from arXiv:1303.5198.

        Parameters
        ----------
        s: float or np.ndarray
            Square of the center of mass energy(ies).
        gvuu: float
            Coupling of vector to up-quarks.
        gvdd: float
            Coupling of vector to down-quarks.

        Returns
        -------
        ff: float or np.ndarray
            Form factor value(s).
        """
        ci1 = gvuu - gvdd

        ms = self.fit_data.rho_masses
        amps = ci1 * self.fit_data.amps * np.exp(1j * self.fit_data.phases)
        brs = self.fit_data.br4pi
        rho_ws = self.fit_data.rho_widths

        if hasattr(s, "__len__"):
            s = np.expand_dims(s, 0)
            ms = np.expand_dims(ms, -1)
            amps = np.expand_dims(amps, -1)
            brs = np.expand_dims(brs, -1)
            rho_ws = np.expand_dims(rho_ws, -1)

        ws = rho_ws * (
            1.0
            - brs
            + brs
            * ms**2
            / s
            * ((s - 16.0 * MPI0_GEV**2) / (ms**2 - 16.0 * MPI0_GEV**2)) ** 1.5
        )
        res = np.sum(
            amps * ms**2 / (ms**2 - s - 1j * np.sqrt(s) * ws),
            axis=0,
        )
        return res.squeeze()

    @overload
    def form_factor(self, *, q: float, gvuu: float, gvdd: float) -> complex:
        ...

    @overload
    def form_factor(self, *, q: RealArray, gvuu: float, gvdd: float) -> ComplexArray:
        ...

    def form_factor(
        self, *, q: Union[float, RealArray], gvuu: float, gvdd: float
    ) -> Union[complex, ComplexArray]:
        r"""Compute the pion-phi form factor.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy in MeV.
        gvuu: float
            Coupling of vector to up-quarks.
        gvdd: float
            Coupling of vector to down-quarks.

        Returns
        -------
        ff: complex or array-like
            Form factor for pion-phi.
        """
        mp = parameters.neutral_pion_mass
        mv = parameters.phi_mass

        single = np.isscalar(q)
        qq = np.atleast_1d(q).astype(np.float64) * 1e-3
        mask = qq > 1e-3 * (mp + mv)
        ff = np.zeros_like(qq, dtype=np.complex128)

        ff[mask] = self.__form_factor(s=qq[mask] ** 2, gvuu=gvuu, gvdd=gvdd)

        if single:
            return ff[0]

        return ff * 1e-3

    def width(self, mv: RealOrRealArray, gvuu: float, gvdd: float) -> RealOrRealArray:
        r"""Compute the partial decay width of a massive vector into a pion and
        phi.

        Parameters
        ----------
        mv: float or array-like
            Mass of the vector.
        gvuu: float
            Coupling of vector to up-quarks.
        gvdd: float
            Coupling of vector to down-quarks.

        Returns
        -------
        width: float or array-like
            Decay width of vector into a pion and phi.
        """
        return self._width(mv=mv, gvuu=gvuu, gvdd=gvdd)

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
    ) -> RealOrRealArray:
        r"""Compute the cross section for dark matter annihilating into a pion
        and phi.

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

        Returns
        -------
        cs: float or array-like
            Annihilation cross section into a pion and phi.
        """
        return self._cross_section(
            q=q, mx=mx, mv=mv, gvxx=gvxx, wv=wv, gvuu=gvuu, gvdd=gvdd
        )
