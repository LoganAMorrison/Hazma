import abc
from typing import Iterable


class AbstractPhaseSpaceIntegrator(abc.ABC):
    """Abstract base class for Lorentz invariant phase-space integrators."""

    @abc.abstractmethod
    def integrate(self, *args, **kwargs):
        raise NotImplementedError()


class AbstractPhaseSpaceGenerator(abc.ABC):
    """Abstract base class for Lorentz invariant phase-space generators."""

    @abc.abstractmethod
    def generate(self, *args, **kwargs) -> Iterable:
        """Generate phase-space points."""
        raise NotImplementedError()


class AbstractPhaseSpaceDistribution(abc.ABC):
    @abc.abstractmethod
    def limits(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError()
