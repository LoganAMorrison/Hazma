"""Spin-one mediator with vector and axial couplings to Dirac dark matter.

Energies, masses and widths use MeV, and rates and cross sections use
MeV⁻², per dark-matter pair.
"""

from ._couplings import LeptonCouplings
from ._model import AxialVectorMediator

__all__ = ["AxialVectorMediator", "LeptonCouplings"]
