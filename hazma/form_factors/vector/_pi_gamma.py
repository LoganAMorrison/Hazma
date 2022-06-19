from dataclasses import dataclass, field, InitVar
from typing import Union, overload, Tuple

import numpy as np

from hazma import parameters

from ._utils import ComplexArray, RealArray
from ._base import VectorFormFactorPA
from ._alpha import alpha_em

MPI0 = parameters.neutral_pion_mass


@dataclass(frozen=True)
class VectorFormFactorPiGammaFitData:
    """Data for the pion-photon vector form-factor."""

    fpi: float

    amplitude0: float
    amplitude_rho: float
    amplitude_omega: float
    amplitude_phi: float

    mass_rho: float
    mass_omega: float
    mass_phi: float

    width_rho: float
    width_omega: float
    width_phi: float


@dataclass
class VectorFormFactorPiGamma(VectorFormFactorPA):
    """Class for computing the pion-photon vector form-factor.

    Attributes
    ----------
    fsp_masses: Tuple[float]
        Final state particle masses (only pion in this case.)
    fit_data: VectorFormFactorPiGammaFitData
        Fit information used to compute the form-factor.

    Methods
    -------
    form_factor
        Compute the un-integrated form-factor into a pion and photon.
    integrated_form_factor
        Compute the form-factor into a pion and photon integrated over
        phase-space.
    width
        Compute the decay width of a vector into a pion and photon.
    cross_section
        Compute the dark matter annihilation cross section into a pion and
        photon.
    """

    fsp_masses: Tuple[float] = field(init=False, default=(MPI0,))
    fit_data: VectorFormFactorPiGammaFitData = field(init=False)

    fpi: InitVar[float] = 0.09266
    amplitude0: InitVar[float] = 0.007594981126020603
    amplitude_rho: InitVar[float] = 1.0
    amplitude_omega: InitVar[float] = 0.8846540224221084
    amplitude_phi: InitVar[float] = -0.06460651106718258

    mass_rho: InitVar[float] = 0.77526
    mass_omega: InitVar[float] = 0.78265
    mass_phi: InitVar[float] = 1.01946

    width_rho: InitVar[float] = 0.1491
    width_omega: InitVar[float] = 0.00849
    width_phi: InitVar[float] = 0.004247

    def __post_init__(
        self,
        fpi,
        amplitude0,
        amplitude_rho,
        amplitude_omega,
        amplitude_phi,
        mass_rho,
        mass_omega,
        mass_phi,
        width_rho,
        width_omega,
        width_phi,
    ):
        self.fit_data = VectorFormFactorPiGammaFitData(
            fpi=fpi,
            amplitude0=amplitude0,
            amplitude_rho=amplitude_rho,
            amplitude_omega=amplitude_omega,
            amplitude_phi=amplitude_phi,
            mass_rho=mass_rho,
            mass_omega=mass_omega,
            mass_phi=mass_phi,
            width_rho=width_rho,
            width_omega=width_omega,
            width_phi=width_phi,
        )

    def __form_factor(
        self, *, q: RealArray, gvuu: float, gvdd: float, gvss: float
    ) -> ComplexArray:
        """
        Compute the form factor for V-gamma-pi at given squared center of mass
        energ(ies).

        Parameters
        ----------
        s: NDArray[float]
            Array of squared center-of-mass energies or a single value.
        gvuu : float
            Coupling of vector mediator to the up quark.
        gvdd : float
            Coupling of vector mediator to the down quark.
        gvss : float
            Coupling of vector mediator to the strange quark.

        Returns
        -------
        ff: NDArray[complex]
            The form factors.
        """

        s = q**2

        def amp(c, m, w):
            return c / (s - m**2 + 1j * q * w)

        ci0 = 3.0 * (gvuu + gvdd)
        ci1 = gvuu - gvdd
        cs = -3.0 * gvss
        cd = 2 * gvuu + gvdd

        fpi = self.fit_data.fpi

        a0 = self.fit_data.amplitude0
        ar = self.fit_data.amplitude_rho
        aw = self.fit_data.amplitude_omega
        af = self.fit_data.amplitude_phi

        amp_0 = a0 * 4.0 * np.sqrt(2) * s / (3.0 * fpi)
        amp_r = ar * amp(ci1, self.fit_data.mass_rho, self.fit_data.width_rho)
        amp_w = aw * amp(ci0, self.fit_data.mass_omega, self.fit_data.width_omega)
        amp_f = af * amp(cs, self.fit_data.mass_phi, self.fit_data.width_phi)

        form = amp_0 * (amp_r + amp_w + amp_f) - cd / (4.0 * np.pi**2 * fpi)
        return np.sqrt(4 * np.pi * alpha_em(s)) * form

    @overload
    def form_factor(self, *, q: float, gvuu, gvdd, gvss) -> complex:
        ...

    @overload
    def form_factor(self, *, q: RealArray, gvuu, gvdd, gvss) -> ComplexArray:
        ...

    def form_factor(
        self, *, q: Union[float, RealArray], gvuu, gvdd, gvss
    ) -> Union[complex, ComplexArray]:
        """Compute the pion-photon vector form factor.

        Parameters
        ----------
        q: float or array-like
            Center of mass energy.
        gvuu: float
            Coupling of vector to up-quarks.
        gvdd: float
            Coupling of vector to down-quarks.
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

        ff[mask] = self.__form_factor(
            q=qq[mask],
            gvuu=gvuu,
            gvdd=gvdd,
            gvss=gvss,
        )

        if single:
            return ff[0]

        return ff * 1e-3

    def width(
        self, mv: Union[float, RealArray], gvuu, gvdd, gvss
    ) -> Union[complex, ComplexArray]:
        r"""Compute the partial decay width of a massive vector into a neutral
        pion and photon.

        Parameters
        ----------
        mv: float
            Mass of the vector.
        gvuu: float
            Coupling of vector to up-quarks.
        gvdd: float
            Coupling of vector to down-quarks.
        gvss: float
            Coupling of vector to strange-quarks.

        Returns
        -------
        width: float
            Decay width of vector into a pion and photon.
        """
        return self._width(mv=mv, gvuu=gvuu, gvdd=gvdd, gvss=gvss)

    def cross_section(
        self,
        *,
        q,
        mx: float,
        mv: float,
        gvxx: float,
        wv: float,
        gvuu: float,
        gvdd: float,
        gvss: float,
    ):
        r"""Compute the cross section for dark matter annihilating into a pion
        and a photon.

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
        gvss: float
            Coupling of vector to strange-quarks.

        Returns
        -------
        cs: float or array-like
            Annihilation cross section into two kaons.
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
