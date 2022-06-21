from dataclasses import dataclass
import abc
from typing import Tuple

from ._base import VectorFormFactor


@dataclass
class VectorFormFactorPPPP(VectorFormFactor):
    r"""Abstract base class for 4 pseudo-scalar meson final state vector form factors."""

    fsp_masses: Tuple[float, float, float, float]

    @abc.abstractmethod
    def form_factor(self, momenta, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def integrated_form_factor(self, q, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def energy_distributions(self, q: float, nbins: int, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def invariant_mass_distributions(self, q: float, nbins: int, **kwargs):
        raise NotImplementedError()
