from dataclasses import dataclass, field, InitVar
import abc
from typing import Tuple

import numpy as np
from scipy import integrate

from hazma.utils import kallen_lambda, lnorm_sqr
from hazma.utils import RealArray
from hazma.rambo import PhaseSpace


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
    @abc.abstractmethod
    def _integrated_form_factor(self, *, q, fsp_masses, **kwargs):
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

    @abc.abstractmethod
    def form_factor(self, **kwargs):
        """Compute the squared matrix element."""
        raise NotImplementedError()

    def _width(self, *, mv, fsp_masses, **kwargs):
        """Compute the partial width of a vector decay."""
        single = np.isscalar(mv)
        q = np.atleast_1d(mv).astype(np.float64)
        w = 0.5 * q * self._integrated_form_factor(q=q, fsp_masses=fsp_masses, **kwargs)

        if single:
            return w[0]
        return w

    def _cross_section(self, *, q, mx, mv, gvxx, wv, fsp_masses, **kwargs):
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
        cs = pre * self._integrated_form_factor(q=qq, fsp_masses=fsp_masses, **kwargs)

        if single:
            return cs[0]
        return cs


@dataclass
class VectorFormFactorPP(VectorFormFactor):
    """Form Factor for a two psuedo-scalar meson final-state.

    This class requires the squared matrix element to have the accept the
    squared center-of-mass energy as its arguments.
    """

    @abc.abstractmethod
    def form_factor(self, *, q, **kwargs):
        """Compute the squared matrix element."""
        raise NotImplementedError()

    def _integrated_form_factor(self, *, q, fsp_masses: Tuple[float, float], **kwargs):
        """Compute the integrated from factor for a two pseudo-scalar meson
        final-state.

        Parameters
        ----------
        s: float or array-like
            Squared center of mass energy.
        """
        m1, m2 = fsp_masses
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

    @abc.abstractmethod
    def form_factor(self, *, q, **kwargs):
        """Compute the squared matrix element."""
        raise NotImplementedError()

    def _integrated_form_factor(self, *, q, fsp_masses: float, **kwargs):
        """Compute the integrated from factor for pseudo-scalar meson and
        photon final-state.

        Parameters
        ----------
        s: float or array-like
            Squared center of mass energy.
        """
        m1 = fsp_masses
        ff = self.form_factor(q=q, **kwargs)
        s = q**2
        return (s - m1**2) ** 3 * np.abs(ff) ** 2 / (48.0 * np.pi * s**2)


@dataclass
class VectorFormFactorPV(VectorFormFactor):
    """Form Factor for a psuedo-scalar meson and vector meson final-state.

    This class requires the squared matrix element to have the accept the
    squared center-of-mass energy as its arguments.
    """

    @abc.abstractmethod
    def form_factor(self, *, q, **kwargs):
        """Compute the squared matrix element."""
        raise NotImplementedError()

    def _integrated_form_factor(self, *, q, fsp_masses: Tuple[float, float], **kwargs):
        """Compute the integrated from factor for pseudo-scalar meson and
        vector meson final-state.

        Parameters
        ----------
        s: float or array-like
            Squared center of mass energy.
        """
        mp, mv = fsp_masses
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

    @abc.abstractmethod
    def form_factor(self, *, s, t, **kwargs):
        """Compute the squared matrix element."""
        raise NotImplementedError()

    def _make_phase_space_integrand(self, q, fsp_masses, **kwargs):

        m1, m2, m3 = fsp_masses
        mu1 = m1 / q
        mu2 = m2 / q
        mu3 = m3 / q
        s = q**2

        def integrand(s1, s2):
            x = (s - s1 + m1**2) / s
            y = (s - s2 + m2**2) / s
            ff = self.form_factor(s=s1, t=s2, **kwargs)
            ff2 = np.abs(ff) ** 2
            return ff2 * (
                -(y**2 * (1 + mu1**2))
                + x**2 * (-1 + y - mu2**2)
                + 2 * y * (1 + mu1**2 + mu2**2 - mu3**2)
                - (1 + (mu1 - mu2) ** 2 - mu3**2) * (1 + (mu1 + mu2) ** 2 - mu3**2)
                + x * (-2 + y) * (-1 + y - mu1**2 - mu2**2 + mu3**2)
            )

        return integrand

    def _make_phase_space(self, q, fsp_masses, **kwargs):
        msqrd = self._make_phase_space_integrand(q, fsp_masses, **kwargs)
        return PhaseSpace(q, masses=np.array(fsp_masses), msqrd=msqrd)  # type: ignore

    def _integrated_form_factor_dblquad(
        self, *, q, fsp_masses: Tuple[float, float, float], **kwargs
    ):
        """Compute the integrated from factor for a three pseudo-scalar meson
        final-state.

        Parameters
        ----------
        s: float or array-like
            Squared center of mass energy.
        """
        m1, m2, m3 = fsp_masses
        mu1 = m1 / q
        mu2 = m2 / q
        mu3 = m3 / q
        s = q**2

        pre = s**3 / (1536.0 * np.pi**3)

        integrand = self._make_phase_space_integrand(q, fsp_masses, **kwargs)

        def f1(x):
            return (2 - x) * (1 - x + mu1**2 + mu2**2 - mu3**2)

        def f2(x):
            return np.sqrt(
                kallen_lambda(1, mu1**2, 1 - x + mu1**2)
                * kallen_lambda(mu2**2, mu3**2, 1 - x + mu1**2)
            )

        def ymin(x):
            return (f1(x) - f2(x)) / (2 * (1.0 - x + mu1**2))

        def ymax(x):
            return (f1(x) + f2(x)) / (2 * (1.0 - x + mu1**2))

        xmin = 2.0 * mu1
        xmax = mu1**2 + (1.0 - (mu2 + mu3) ** 2)

        return pre * integrate.dblquad(integrand, xmin, xmax, ymin, ymax)[0]

    def _integrated_form_factor_rambo(
        self, *, q, fsp_masses: Tuple[float, float, float], npts: int, **kwargs
    ):
        s = q**2
        pre = s**3 / (1536.0 * np.pi**3)
        phase_space = self._make_phase_space(q, fsp_masses, **kwargs)
        return pre * phase_space.integrate(n=npts)[0]

    def _integrated_form_factor(
        self, *, q, fsp_masses: Tuple[float, float, float], method: str, **kwargs
    ):
        """Compute the integrated from factor for a three pseudo-scalar meson
        final-state.

        Parameters
        ----------
        s: float or array-like
            Squared center of mass energy.
        """
        if method == "quad":
            return self._integrated_form_factor_dblquad(
                q=q, fsp_masses=fsp_masses, **kwargs
            )
        return self._integrated_form_factor_rambo(q=q, fsp_masses=fsp_masses, **kwargs)

    def _energy_distributions(
        self,
        *,
        q,
        fsp_masses: Tuple[float, float, float],
        npts: int,
        nbins: int,
        **kwargs
    ):
        phase_space = self._make_phase_space(q, fsp_masses, **kwargs)
        return phase_space.energy_distributions(n=npts, nbins=nbins)

    def _invariant_mass_distributions(
        self,
        *,
        q,
        fsp_masses: Tuple[float, float, float],
        npts: int,
        nbins: int,
        **kwargs
    ):
        phase_space = self._make_phase_space(q, fsp_masses, **kwargs)
        return phase_space.invariant_mass_distributions(n=npts, nbins=nbins)
