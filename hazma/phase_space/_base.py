"""Interface definitions for phase-space generators."""

# pylint: disable=invalid-name,too-few-public-methods

import abc
from typing import Iterable


class AbstractPhaseSpaceIntegrator(abc.ABC):
    """Abstract base class for Lorentz invariant phase-space integrators."""

    @abc.abstractmethod
    def integrate(self, n: int):
        """Integrate of phase-space."""
        raise NotImplementedError()


class AbstractPhaseSpaceGenerator(abc.ABC):
    """Abstract base class for Lorentz invariant phase-space generators."""

    @abc.abstractmethod
    def generate(self, n: int) -> Iterable:
        """Generate phase-space points."""
        raise NotImplementedError()


class AbstractPhaseSpaceDistribution(abc.ABC):
    @abc.abstractmethod
    def limits(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError()
