"""
Module for describing the high-level behavior of a vector form factor.
"""

# pylint: disable=invalid-name

import abc
from dataclasses import InitVar, dataclass, field
from typing import NamedTuple, Sequence

import numpy as np

from hazma.utils import RealArray


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


class VectorFormFactorCouplings(NamedTuple):
    r"""Named tuple to hold quark vector couplings.

    Attributes
    ----------
    gvuu: float
        Up-quark vector coupling.
    gvdd: float
        Down-quark vector coupling.
    gvss: float
        Strange-quark vector coupling.
    """

    gvuu: float
    gvdd: float
    gvss: float


def vector_couplings_to_isospin(gvuu: float, gvdd: float, gvss: float):
    """Convert the vector couplings to isospin couplings."""
    ci0 = 3.0 * (gvuu + gvdd)
    ci1 = gvuu - gvdd
    cs = -3.0 * gvss
    return ci0, ci1, cs


@dataclass
class VectorFormFactor(abc.ABC):
    """Base class for vector form factors."""

    fsp_masses: Sequence[float]

    @abc.abstractmethod
    def form_factor(self, *args, **kwargs):
        """Compute the squared matrix element."""
        raise NotImplementedError()

    @abc.abstractmethod
    def integrated_form_factor(self, q, couplings: Sequence[float], **kwargs):
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
    def width(self, mv, couplings: Sequence[float], **kwargs):
        r"""Compute the partial decay width of a massive vector."""
        raise NotImplementedError()

    @abc.abstractmethod
    def cross_section(self, *, q, mx, mv, gvxx, wv, **kwargs):
        r"""Compute the partial annihilation cross sectrion of dark matter."""
        raise NotImplementedError()

    def _width(self, *, mv, **kwargs):
        """Compute the decay width of a massive vector. Calls the underlying
        `integrated_from_factor` method.

        Parameters
        ----------
        mv: float or array-like
            Mass of the decaying vector.
        kwargs: dict
            Keyword arguments passed to underlying `integrated_from_factor`
            method.

        Returns
        -------
        width: float or array-like
            Partial width the of vector. Has same shape as `mv`.
        """
        single = np.isscalar(mv)
        q = np.atleast_1d(mv).astype(np.float64)

        mask = q > sum(self.fsp_masses)
        w = np.zeros_like(q)

        if np.any(mask):
            w[mask] = 0.5 * q[mask] * self.integrated_form_factor(q=q[mask], **kwargs)

        if single:
            return w[0]
        return w

    def _cross_section(self, *, q, mx, mv, gvxx, wv, **kwargs):
        """Compute the dark matter annihilation cross section. Calls the underlying
        `integrated_from_factor` method.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy.
        mx: float
            Mass of the dark matter.
        mv: float or array-like
            Mass of the vector mediator.
        gvxx: float
            Coupling of dark matter to vector mediator.
        wv: float
            Width of the vector mediator.
        kwargs: dict
            Keyword arguments passed to underlying `integrated_from_factor`
            method.

        Returns
        -------
        sigma: float or array-like
            Dark matter annihilation cross-section. Has same shape as `q`.
        """
        single = np.isscalar(q)
        qq = np.atleast_1d(q).astype(np.float64)

        mask = qq > sum(self.fsp_masses)
        cs = np.zeros_like(qq)

        if np.any(mask):
            s = qq[mask] ** 2
            pre = (
                gvxx**2
                * (s + 2 * mx**2)
                / (np.sqrt(s - 4 * mx**2) * ((s - mv**2) ** 2 + (mv * wv) ** 2))
            )
            pre = pre * 0.5 * qq[mask]
            cs[mask] = pre * self.integrated_form_factor(q=qq[mask], **kwargs)

        if single:
            return cs[0]
        return cs
