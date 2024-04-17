"""
Base classes for 4-body final state form factors.
"""

# pylint: disable=arguments-differ


import abc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

from ._base import VectorFormFactor, VectorFormFactorCouplings

Couplings: TypeAlias = VectorFormFactorCouplings | Sequence[float]


@dataclass
class VectorFormFactorPPPP(VectorFormFactor):
    """Abstract base class for 4 pseudo-scalar meson final state vector form factors."""

    fsp_masses: Sequence[float]

    @abc.abstractmethod
    def form_factor(self, momenta, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def integrated_form_factor(self, q, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def energy_distributions(self, q: float, nbins: int, **kwargs):
        r"""Compute the energy distributions of the final states."""
        raise NotImplementedError()

    @abc.abstractmethod
    def invariant_mass_distributions(self, q: float, nbins: int, **kwargs):
        """Compute the invariant-mass distributions of all pairs of final state particles."""
        raise NotImplementedError()
