"""
Base classes for two-body final state form factors.
"""

from dataclasses import dataclass
import abc
from typing import Tuple, Union, Sequence

import numpy as np

from hazma.phase_space._utils import two_body_phase_space_prefactor

from ._base import VectorFormFactor
from ._base import VectorFormFactorCouplings

Couplings = Union[VectorFormFactorCouplings, Sequence[float]]


@dataclass
class VectorFormFactorTwoBody(VectorFormFactor):
    r"""Vector form factor for a two meson final-state."""

    fsp_masses: Tuple[float, float]

    @abc.abstractmethod
    def form_factor(  # pylint: disable=arguments-differ
        self, *, q, couplings: Couplings
    ):
        raise NotImplementedError()

    @abc.abstractmethod
    def squared_lorentz_structure(self, q):
        r"""Compute the value of the Lorentz structure of the amplitude."""
        raise NotImplementedError()

    def _integrated_form_factor(self, q, **kwargs):
        m1, m2 = self.fsp_masses
        ff = self.form_factor(q=q, **kwargs)
        lor = self.squared_lorentz_structure(q)
        pre = two_body_phase_space_prefactor(q, m1, m2)
        return pre * lor * np.abs(ff) ** 2


@dataclass
class VectorFormFactorPP(VectorFormFactorTwoBody):
    r"""Abstract base class for form factors involving two pseudo-scalar mesons.

    The two pseudo-scalar currents are given by:

    .. math::

        J_{\mu} = -(p_{1}^{\mu} - p_{2}^{\mu}) F_{PP}(q^2)

    where :math:`F_{PP}(q^2)` is the form factor, :math:`q=p_{1}+p_{2}` and
    :math:`p_{1}` and :math:`p_{2}` the momenta of the meson and photon,
    respectively.
    """

    fsp_masses: Tuple[float, float]

    @abc.abstractmethod
    def form_factor(self, *, q, **kwargs):
        """Compute the squared matrix element."""
        raise NotImplementedError()

    def squared_lorentz_structure(self, q):
        m1, m2 = self.fsp_masses
        s = q**2
        return (s - 2 * m1**2 - 2 * m2**2) / (3 * s)


@dataclass
class VectorFormFactorPA(VectorFormFactorTwoBody):
    r"""Abstract base class for form factors involving a pseudo-scalar meson and
    a photon.

    The pseudo-scalar + photon currents are given by:

    .. math::

        J_{\mu} = \epsilon_{\mu\nu\alpha\beta}
        q^{\nu}\epsilon^{\alpha}(k)k^{\beta}
        F_{P\gamma}(q^2)

    where :math:`F_{P\gamma}(q^2)` is the form factor,
    :math:`\epsilon^{\alpha}(k)` is the polarization vector of the photon, and
    :math:`q=p+k`, with :math:`p` and :math:`k` the momenta of the meson and
    photon, respectively.
    """

    fsp_masses: Tuple[float, float]

    @abc.abstractmethod
    def form_factor(self, *, q, **kwargs):
        """Compute the squared matrix element."""
        raise NotImplementedError()

    def squared_lorentz_structure(self, q):
        m = self.fsp_masses[0]
        s = q**2
        return (s - m**2) ** 2 / (6 * s)


@dataclass
class VectorFormFactorPV(VectorFormFactorTwoBody):
    r"""Abstract base class for form factors involving a pseudo-scalar meson and
    a vector meson.

    The pseudo-scalar + vector meson currents are given by:

    .. math::

        J_{\mu} = \epsilon_{\mu\nu\alpha\beta}
        q^{\nu}\epsilon^{\alpha}(k)p^{\beta}
        F_{PV}(q^2)

    where :math:`F_{PV}(q^2)` is the form factor and :math:`q=p+k`, with
    :math:`p` and :math:`k` the momenta of the meson and vector, respectively.
    """

    fsp_masses: Tuple[float, float]

    @abc.abstractmethod
    def form_factor(self, *, q, **kwargs):
        raise NotImplementedError()

    def squared_lorentz_structure(self, q):
        mp, mv = self.fsp_masses
        s = q**2
        return (s - (mp - mv) ** 2) * (s - (mp + mv) ** 2) / (6 * s)
