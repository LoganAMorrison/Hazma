"""Results for explicitly normalized axial source spectral functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real

import numpy as np

# Keep runtime-resolvable annotations, as required by AGENTS.md.
from hazma.utils import RealArray  # noqa: TC001

from ._sources import SourceCoverage


def _immutable_value(array: np.ndarray) -> float | RealArray:
    if array.ndim == 0:
        return float(array)
    # Immutable bytes backing also prevents callers from re-enabling writes.
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


@dataclass(frozen=True, slots=True)
class AxialSpectralResult:
    r"""Integrated spectral densities with declared normalization and coverage.

    Parameters
    ----------
    q_mev : float or array-like
        Center-of-mass energy in MeV, inside the declared input coverage.
    transverse : float or array-like
        ``A_T = integral dPhi |H_CM spatial|**2 / 3`` in ``units``.
    source_longitudinal : float or array-like
        ``A_L = integral dPhi |H_CM time|**2`` in ``units``. This is the
        q-parallel source response, not an on-shell helicity-zero fraction.
    threshold_mev : float
        Channel threshold in MeV. Both densities must vanish at and below it.
    coverage : SourceCoverage
        Channel input domain, approximation status and provenance.
    units : str
        Units of both densities, declared by the current/measure provider.
        For dimensionless LO three-pion currents this is ``"MeV^2"``.

    Notes
    -----
    Densities include the complete phase-space measure and channel symmetry
    factor, with no optical-theorem factor 1/2. They carry no dark-matter,
    mediator-propagator, flux or decay-width prefactor. Inputs broadcast to
    a common shape; array fields are copied and made immutable.
    """

    q_mev: float | RealArray
    transverse: float | RealArray
    source_longitudinal: float | RealArray
    threshold_mev: float = field(kw_only=True)
    coverage: SourceCoverage = field(kw_only=True)
    units: str = field(kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.coverage, SourceCoverage):
            raise TypeError("coverage must be a SourceCoverage.")
        if not isinstance(self.units, str) or not self.units.strip():
            raise ValueError("Spectral density units must be declared.")
        if isinstance(self.threshold_mev, (bool, np.bool_)) or not isinstance(
            self.threshold_mev, Real
        ):
            raise ValueError("threshold_mev must be a finite nonnegative scalar.")
        threshold = float(self.threshold_mev)
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold_mev must be a finite nonnegative scalar.")
        arrays = []
        for name in ("q_mev", "transverse", "source_longitudinal"):
            value = np.asarray(getattr(self, name))
            if np.iscomplexobj(value):
                raise ValueError(f"{name} must be real.")
            value = np.asarray(value, dtype=np.float64)
            if not np.all(np.isfinite(value)) or np.any(value < 0):
                raise ValueError(f"{name} must be finite and nonnegative.")
            arrays.append(value)
        energy, transverse, longitudinal = np.broadcast_arrays(*arrays)
        if not np.all(self.coverage.contains(energy)):
            raise ValueError("q_mev lies outside the declared input coverage.")
        closed = energy <= threshold
        if np.any(transverse[closed] != 0) or np.any(longitudinal[closed] != 0):
            raise ValueError("Spectral densities must vanish at and below threshold.")
        object.__setattr__(self, "threshold_mev", threshold)
        for name, value in zip(
            ("q_mev", "transverse", "source_longitudinal"),
            (energy, transverse, longitudinal),
            strict=True,
        ):
            object.__setattr__(self, name, _immutable_value(value))
