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
    def form_factor(self, q, s, t, **kwargs):
        """Compute the squared matrix element."""
        raise NotImplementedError()

    def _make_phase_space_integrand(self, q, fsp_masses, **kwargs):
        m1, m2, m3 = fsp_masses

        def integrand(s, t):
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

        return integrand

    def _make_phase_space(self, q, fsp_masses, **kwargs):
        msqrd_ = self._make_phase_space_integrand(q, fsp_masses, **kwargs)

        def msqrd(momenta):
            s = lnorm_sqr(momenta[:, 1] + momenta[:, 2])
            t = lnorm_sqr(momenta[:, 0] + momenta[:, 2])
            return msqrd_(s, t)

        return PhaseSpace(q, masses=np.array(fsp_masses), msqrd=msqrd)  # type: ignore

    def _integrated_form_factor_dblquad(
        self, *, q: float, fsp_masses: Tuple[float, float, float], **kwargs
    ) -> float:
        """Compute the integrated from factor for a three pseudo-scalar meson
        final-state.

        Parameters
        ----------
        s: float or array-like
            Squared center of mass energy.
        """
        m1, m2, m3 = fsp_masses
        pre = 1.0 / (128.0 * np.pi**3 * q**2)
        integrand = self._make_phase_space_integrand(q, fsp_masses, **kwargs)

        def f1(s):
            return (
                -((q**2 - m1**2) * (m2**2 - m3**2))
                + (q**2 + m1**2 + m2**2 + m3**2) * s
                - s**2
            )

        def f2(s):
            return np.sqrt(
                kallen_lambda(s, q**2, m1**2) * kallen_lambda(s, m2**2, m3**2)
            )

        def tmin(s):
            return (f1(s) - f2(s)) / (2 * s)

        def tmax(s):
            return (f1(s) + f2(s)) / (2 * s)

        smin = (m2 + m3) ** 2
        smax = (q - m1) ** 2

        return pre * integrate.dblquad(integrand, smin, smax, tmin, tmax)[0]

    def _integrated_form_factor_rambo(
        self, *, q: float, fsp_masses: Tuple[float, float, float], npts: int, **kwargs
    ) -> float:

        msqrd = self._make_phase_space_integrand(q, fsp_masses, **kwargs)

        phase_space = PhaseSpace(q, masses=np.array(fsp_masses))
        ps, ws = phase_space.generate(npts)
        s = lnorm_sqr(ps[:, 1] + ps[:, 2])
        t = lnorm_sqr(ps[:, 0] + ps[:, 2])
        avg = np.nanmean(ws * msqrd(s, t))

        return avg

    def _integrated_form_factor(
        self,
        *,
        q,
        fsp_masses: Tuple[float, float, float],
        method: str = "rambo",
        **kwargs
    ):
        """Compute the integrated from factor for a three pseudo-scalar meson
        final-state.

        Parameters
        ----------
        s: float or array-like
            Squared center of mass energy.
        """
        if method == "quad":

            def f(qq):
                return self._integrated_form_factor_dblquad(
                    q=qq, fsp_masses=fsp_masses, **kwargs
                )

        else:

            def f(qq):
                return self._integrated_form_factor_rambo(
                    q=qq, fsp_masses=fsp_masses, **kwargs
                )

        msum = sum(fsp_masses)

        def integrator(qq):
            if qq < msum:
                return 0.0
            return f(qq)

        return np.array([integrator(qq) for qq in q])

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
