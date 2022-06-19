from dataclasses import dataclass, field, InitVar
import abc
from typing import Tuple, Sequence, Union, overload

import numpy as np

from hazma.utils import kallen_lambda
from hazma.utils import RealArray
from hazma.phase_space import integrate_three_body
from hazma.phase_space import energy_distributions_three_body
from hazma.phase_space import invariant_mass_distributions_three_body


def _normalize(xs, ps, axis=0):
    assert np.shape(xs) == np.shape(ps), "Invalid shapes."
    norm = np.trapz(ps, xs, axis=axis)
    if norm > 0.0:
        newps = ps / norm
    else:
        newps = np.zeros_like(ps)
    return newps


@dataclass
class VMDAmplitude:
    r"""Vector-Meson Dominance amplitude of the form:

    .. math::
        \sum_{k}a_{k} e^{i \phi_{k}} m^{2}_{i} / (m^{2}_{i} - s - i m^{2}_{k}\Gamma_{k})
    """
    _amplitudes: InitVar[RealArray]
    _phases: InitVar[RealArray]

    amplitudes: RealArray = field(init=False)
    masses: RealArray = field()
    widths: RealArray = field()

    def __post_init__(self, _amplitudes, _phases):
        self.amplitudes = _amplitudes * np.exp(1j * _phases)

    def __call__(self, s):
        m2 = self.masses**2
        return self.amplitudes * m2 / (m2 - s + 1j * self.masses * self.widths)


@dataclass
class VMDAmplitudeGS:
    r"""Vector-Meson Dominance Gounaris-Sakurai amplitude of the form:

    .. math::
        \sum_{k}a_{k} e^{i \phi_{k}} BW_{k}(s)

    where

    .. math::
        BW_{k}(s) = (m^{2}_{k} + H(0)) / (m^{2}_{i} - s + H(s) - i \sqrt{s}\Gamma_{k})
    """
    _amplitudes: InitVar[RealArray]
    _phases: InitVar[RealArray]

    amplitudes: RealArray = field(init=False)
    masses: RealArray = field()
    widths: RealArray = field()

    def __post_init__(self, _amplitudes, _phases):
        self.amplitudes = _amplitudes * np.exp(1j * _phases)

    def __call__(self, s):
        return np.zeros_like(s)


@dataclass
class VectorFormFactor(abc.ABC):
    """Base class for vector form factors."""

    fsp_masses: Sequence[float]

    @abc.abstractmethod
    def form_factor(self, *args, **kwargs):
        """Compute the squared matrix element."""
        raise NotImplementedError()

    @abc.abstractmethod
    def integrated_form_factor(self, *, q, **kwargs):
        r"""Compute the form-factor as a function of the center-of-mass energy.

        Notes
        -----
        Compute the integrated form-factor:

        .. math::
            \mathcal{J}(s) = -\frac{1}{3s}\int\Pi_{\mathrm{LIPS}}
            J_{\mathcal{H}}^{\mu}\bar{J}_{\mathcal{H}}^{\mu}

        where

        .. math::
            J_{\mathcal{H}}^{\mu} = \bra{0}J_{\mu}\ket{\mathcal{H}}

        """
        raise NotImplementedError()

    def _width(self, *, mv, **kwargs):
        """Compute the partial width of a vector decay."""
        single = np.isscalar(mv)
        q = np.atleast_1d(mv).astype(np.float64)
        w = 0.5 * q * self.integrated_form_factor(q=q, **kwargs)

        if single:
            return w[0]
        return w

    def width(self, *, mv, **kwargs):
        """Compute the partial width of a vector decay."""
        return self._width(mv=mv, **kwargs)

    def _cross_section(self, *, q, mx, mv, gvxx, wv, **kwargs):
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
        cs = pre * self.integrated_form_factor(q=qq, **kwargs)

        if single:
            return cs[0]
        return cs

    def cross_section(self, *, q, mx, mv, gvxx, wv, **kwargs):
        """Compute the cross-section of dark matter annihilation."""
        return self._cross_section(q=q, mx=mx, mv=mv, gvxx=gvxx, wv=wv, **kwargs)


@dataclass
class VectorFormFactorPP(VectorFormFactor):
    """Form Factor for a two psuedo-scalar meson final-state.

    This class requires the squared matrix element to have the accept the
    squared center-of-mass energy as its arguments.
    """

    fsp_masses: Tuple[float, float]

    @abc.abstractmethod
    def form_factor(self, *, q, **kwargs):
        """Compute the squared matrix element."""
        raise NotImplementedError()

    def integrated_form_factor(self, *, q, **kwargs):
        """Compute the integrated from factor for a two pseudo-scalar meson
        final-state.

        Parameters
        ----------
        s: float or array-like
            Squared center of mass energy.
        """
        m1, m2 = self.fsp_masses
        mu1 = m1 / q
        mu2 = m2 / q
        ff = self.form_factor(q=q, **kwargs)

        return (
            (1.0 - 2.0 * (mu1**2 + mu2**2))
            * np.abs(ff) ** 2
            * np.sqrt(np.clip(kallen_lambda(1.0, mu1**2, mu2**2), 0.0, None))
            / (24.0 * np.pi)
        )


@dataclass
class VectorFormFactorPA(VectorFormFactor):
    """Form Factor for a psuedo-scalar meson and photon final-state.

    This class requires the squared matrix element to have the accept the
    squared center-of-mass energy as its arguments.
    """

    fsp_masses: Tuple[float]

    @abc.abstractmethod
    def form_factor(self, *, q, **kwargs):
        """Compute the squared matrix element."""
        raise NotImplementedError()

    def integrated_form_factor(self, *, q, **kwargs):
        """Compute the integrated from factor for pseudo-scalar meson and
        photon final-state.

        Parameters
        ----------
        s: float or array-like
            Squared center of mass energy.
        """
        m1 = self.fsp_masses[0]
        ff = self.form_factor(q=q, **kwargs)
        s = q**2
        return (s - m1**2) ** 3 * np.abs(ff) ** 2 / (48.0 * np.pi * s**2)


@dataclass
class VectorFormFactorPV(VectorFormFactor):
    """Form Factor for a psuedo-scalar meson and vector meson final-state.

    This class requires the squared matrix element to have the accept the
    squared center-of-mass energy as its arguments.
    """

    fsp_masses: Tuple[float, float]

    @abc.abstractmethod
    def form_factor(self, *, q, **kwargs):
        """Compute the squared matrix element."""
        raise NotImplementedError()

    def integrated_form_factor(self, *, q, **kwargs):
        """Compute the integrated from factor for pseudo-scalar meson and
        vector meson final-state.

        Parameters
        ----------
        s: float or array-like
            Squared center of mass energy.
        """
        mp, mv = self.fsp_masses
        s = q**2
        ff = self.form_factor(q=q, **kwargs)
        pre = np.clip(kallen_lambda(s, mv**2, mp**2), 0.0, None) ** 1.5 / (
            48.0 * np.pi * s**2
        )
        return pre * np.abs(ff) ** 2


@dataclass
class VectorFormFactorPPP(VectorFormFactor):
    """Form Factor for a three psuedo-scalar meson final-state.

    This class requires the squared matrix element to ...
    """

    fsp_masses: Tuple[float, float, float]

    @abc.abstractmethod
    def form_factor(self, q, s, t, **kwargs):
        """Compute the squared matrix element."""
        raise NotImplementedError()

    def __phase_space_integrand(self, q, s, t, **kwargs):
        r"""Compute the integrand of the three-body phase-space."""
        m1, m2, m3 = self.fsp_masses

        ff = self.form_factor(q, s, t, **kwargs)
        ff2 = np.abs(ff) ** 2
        lor = (
            -(
                (m1 * m2 - q * m3)
                * (m1 * m2 + q * m3)
                * (-(q**2) + m1**2 + m2**2 - m3**2)
            )
            - (q - m1) * (q + m1) * (m2 - m3) * (m2 + m3) * t
            - s**2 * t
            + s
            * (
                -((q - m2) * (q + m2) * (m1 - m3) * (m1 + m3))
                + (q**2 + m1**2 + m2**2 + m3**2) * t
                - t**2
            )
        )

        return ff2 * lor / (12.0 * q**2)

    @overload
    def _integrated_form_factor(
        self, *, q: float, method: str = "rambo", npts: int = 10000, **kwargs
    ) -> float:
        ...

    @overload
    def _integrated_form_factor(
        self, *, q: RealArray, method: str = "rambo", npts: int = 10000, **kwargs
    ) -> RealArray:
        ...

    def _integrated_form_factor(
        self,
        *,
        q: Union[float, RealArray],
        method: str = "quad",
        npts: int = 10000,
        **kwargs,
    ) -> Union[float, RealArray]:
        """Compute the integrated from factor for a three pseudo-scalar meson
        final-state.

        Parameters
        ----------
        q: float or array-like
            Center of mass energy.
        method: str, optional
            Method used to integrate. Default is 'quad'. Options are 'quad' or
            'rambo'.
        npts: int, optional
            Number of phase-space points to use in integration. Ignored is
            method isn't 'rambo'. Default is 10_000.

        Returns
        -------
        ff: float or array-like
            Integrated form-factor.
        """

        def integrate_(q_):
            return integrate_three_body(
                lambda s, t: self.__phase_space_integrand(q_, s, t, **kwargs),
                q_,
                self.fsp_masses,
                method=method,
                npts=npts,
            )[0]

        scalar = np.isscalar(q)
        qs = np.atleast_1d(q)

        res = np.array([integrate_(q_) for q_ in qs])

        if scalar:
            return res[0]
        return res

    def _energy_distributions(
        self,
        *,
        q: float,
        nbins: int,
        method: str = "quad",
        npts: int = 10000,
        **kwargs,
    ):
        r"""Compute the energy distributions of the final state particles.

        Parameters
        ----------
        q: float
            Center-of-mass energy in MeV.
        method: str
            Method used to generate energy distributions. Can be 'quad' or
            'rambo'. Default is 'quad'.
        nbins: int
            Number of bins for the distributions.
        npts: int
            Number of phase-space points used to generate distributions. Only
            used if method is 'rambo'. Default is 10_000.

        Returns
        -------
        dist1, dist2, dist3: (array, array)
            Three tuples containing the probabilities and energies for each
            final state particle.
        """

        def integrand(s, t):
            return self.__phase_space_integrand(q, s, t, **kwargs)

        return energy_distributions_three_body(
            integrand, q, self.fsp_masses, method=method, nbins=nbins, npts=npts
        )

    def _invariant_mass_distributions(
        self, *, q: float, method: str, nbins: int, npts: int = 10000, **kwargs
    ):
        r"""Compute the invariant-mass distributions of the final state
        particles.

        Parameters
        ----------
        q: float
            Center-of-mass energy in MeV.
        method: str
            Method used to generate energy distributions. Can be 'quad' or
            'rambo'. Default is 'quad'.
        nbins: int
            Number of bins for the distributions.
        npts: int
            Number of phase-space points used to generate distributions. Only
            used if method is 'rambo'. Default is 10_000.

        Returns
        -------
        dist1, dist2, dist3: (array, array)
            Three tuples containing the probabilities and invariant massses for
            each final state particle.

        Other Parameters
        ----------------
        """

        def integrand(s, t):
            return self.__phase_space_integrand(q, s, t, **kwargs)

        return invariant_mass_distributions_three_body(
            integrand, q, self.fsp_masses, method=method, nbins=nbins, npts=npts
        )

    def _width(self, *, mv, method: str = "quad", npts: int = 10000, **kwargs):
        """Compute the partial width of a vector decay."""
        single = np.isscalar(mv)
        q = np.atleast_1d(mv).astype(np.float64)
        w = (
            0.5
            * q
            * self.integrated_form_factor(q=q, method=method, npts=npts, **kwargs)
        )

        if single:
            return w[0]
        return w

    def _cross_section(
        self, *, q, mx, mv, gvxx, wv, method="quad", npts: int = 10000, **kwargs
    ):
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
        cs = pre * self.integrated_form_factor(q=q, method=method, npts=npts, **kwargs)

        if single:
            return cs[0]
        return cs


@dataclass
class VectorFormFactorPPPP(VectorFormFactor):
    """Form Factor for a four psuedo-scalar meson final-state."""

    fsp_masses: Tuple[float, float, float, float]

    @abc.abstractmethod
    def form_factor(self, momenta, **kwargs):
        """Compute the squared matrix element."""
        raise NotImplementedError()

    @abc.abstractmethod
    def integrated_form_factor(self, q, **kwargs):
        """Compute the form-factor integrated over phase space.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy.
        """
        raise NotImplementedError()
