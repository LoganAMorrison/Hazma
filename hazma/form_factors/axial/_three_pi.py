r"""Leading-order, exact-isospin neutral axial three-pion currents.

For ``G_A = g3 diag(1,-1,0)``, write the SU(2) field as
``U = (sigma + i tau.phi)/F``, ``sigma = sqrt(F**2-phi**2)``. The
O(p²) action gives ``J_A^3 = 2 g3 (sigma dphi3 - phi3 dsigma)``.
The overall external-state phase is chosen to make this current real.
Its cubic contact terms are ``2 g3 (q-2p0)/F`` for pi+ pi- pi0 and
``2 g3 q/F`` for three pi0. The off-shell four-pion vertices multiply
these by the single-pion pole contributions
``-2 g3 q (s+- - m²)/(F (q²-m²))`` and
``-2 g3 q q²/(F (q²-m²))``, respectively. Both terms are essential.

Equivalently, exponential pion coordinates give a charged contact term
``4 g3 (q-3p0)/(3F)`` and an off-shell vertex
``(3s+- - q² - 2m²)/(3F²)``; their sum is identical. This field-coordinate
check also fixes the relative contact/pole sign. These are tree-level
currents, without resonances, loops, electromagnetic or isospin corrections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from hazma._core import axial as _axial
from hazma.parameters import neutral_pion_mass
from hazma.utils import RealOrRealArray

from ._base import AxialSpectralResult
from ._sources import AxialCurrent, SourceCoverage
from ._tensors import _components

ArrayLike = RealOrRealArray | list[float] | tuple[float, ...]
_MIN_ORDER = 8
_MAX_ORDER = 512
_MOMENTA_NDIM = 2
_LO_MAX_MEV = 600.0


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[0] * b[0] - np.sum(a[1:] * b[1:], axis=0)


def _real_array(value: ArrayLike, name: str) -> np.ndarray:
    if not np.isrealobj(value):
        raise ValueError(f"{name} must be real and finite.")
    result = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be real and finite.")
    return result


def _scalar(value: float, name: str, *, positive: bool) -> float:
    result = _real_array(value, name)
    if result.ndim != 0 or (result <= 0 if positive else result < 0):
        raise ValueError(
            f"{name} must be a {'positive' if positive else 'nonnegative'} scalar."
        )
    return float(result)


def _g3(couplings: AxialCurrent) -> float:
    if not isinstance(couplings, AxialCurrent):
        raise TypeError("couplings must be an AxialCurrent with source metadata.")
    if couplings.gas != 0.0 or couplings.gau != -couplings.gad:
        raise ValueError("LO three-pion currents support only G_A = g3 diag(1,-1,0).")
    return np.float64(couplings.gau)


@dataclass(frozen=True, kw_only=True)
class _AxialThreePion:
    pion_mass: float = neutral_pion_mass
    decay_constant: float = 92.2
    q_max: float = 600.0
    _neutral: ClassVar[bool] = False
    _channel: ClassVar[str] = "pi+ pi- pi0"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "pion_mass", _scalar(self.pion_mass, "pion_mass", positive=False)
        )
        object.__setattr__(
            self,
            "decay_constant",
            _scalar(self.decay_constant, "decay_constant", positive=True),
        )
        object.__setattr__(self, "q_max", _scalar(self.q_max, "q_max", positive=True))
        if self.q_max > _LO_MAX_MEV:
            raise ValueError(
                "q_max cannot exceed the validated LO support ceiling of 600 MeV."
            )
        if self.q_max <= self.threshold:
            raise ValueError("q_max must be above the three-pion threshold.")

    @property
    def fsp_masses(self) -> tuple[float, float, float]:
        """Common pion masses in MeV, in the current's particle order."""
        return (self.pion_mass,) * 3

    @property
    def threshold(self) -> float:
        """Three-pion threshold in MeV."""
        return 3 * self.pion_mass

    @property
    def coverage(self) -> SourceCoverage:
        """Declared LO support, not an accuracy claim for chiral perturbation theory."""
        return SourceCoverage(
            channel=self._channel,
            q_min_mev=0.0,
            q_max_mev=self.q_max,
            input_status="lo",
            missing_directions=("octet", "singlet"),
            provenance="SU(2) O(p²) action; exact isospin; contact plus pion pole; restricted LO domain, not a precision bound",
        )

    @property
    def _parameters(self) -> tuple[float, float, bool]:
        """Pion mass, decay constant and channel, as the kernels take them."""
        return self.pion_mass, self.decay_constant, self._neutral

    @np.errstate(over="ignore", invalid="ignore", divide="ignore")
    def _kinematics(
        self, momenta: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ps = _real_array(momenta, "momenta")
        if ps.ndim < _MOMENTA_NDIM or ps.shape[:2] != (4, 3):
            raise ValueError(
                "momenta must have shape (4,3,...), ordered by final state."
            )
        if np.any(ps[0] < 0):
            raise ValueError("Pion energies must be nonnegative.")
        mass2 = _real_array(_dot(ps, ps), "On-shell invariants")
        # Cancellation in boosted on-shell vectors scales with E², not m².
        tolerance = 2e-10 * np.maximum(1.0, ps[0] ** 2)
        if np.any(np.abs(mass2 - np.square(self.pion_mass)) > tolerance):
            raise ValueError(
                "All pions must be on shell with the declared common pion_mass."
            )
        q = np.sum(ps, axis=1)
        q2 = _real_array(_dot(q, q), "Total invariant mass squared")
        if np.any(q2 <= 0):
            raise ValueError("The total momentum must be timelike and nonzero.")
        invariant_mass = np.sqrt(q2)
        if np.any(invariant_mass < self.threshold * (1 - 2e-10)):
            raise ValueError("Total momentum is below the three-pion threshold.")
        if np.any(invariant_mass > self.q_max * (1 + 2e-12)):
            raise ValueError("Total momentum is outside the declared LO support.")
        return ps, q, q2

    def current(self, momenta: ArrayLike, couplings: AxialCurrent) -> np.ndarray:
        r"""Return the full contravariant current H^mu, dimensionless.

        Parameters
        ----------
        momenta : array-like, shape (4, 3, ...)
            On-shell four-momenta in MeV, using metric (+---). Particle order
            is (pi+, pi-, pi0), or three pi0 for the neutral channel.
        couplings : AxialCurrent
            Dimensionless pure isovector source, including QCD metadata.

        Returns
        -------
        current : ndarray, shape (4, ...)
            Real current including both contact and single-pion-pole terms.
            A single event of shape (4,3) returns shape (4,).
        """
        g3 = _g3(couplings)
        ps, _, _ = self._kinematics(momenta)
        shape = ps.shape[2:]
        value = _axial.three_pi_current(
            _components(ps, shape, 2), g3, *self._parameters
        )
        return _real_array(value.reshape((4,) + shape), "Current")

    def differential_weight(
        self, momenta: ArrayLike, couplings: AxialCurrent
    ) -> float | np.ndarray:
        r"""Return invariant spin-averaged |epsilon.H|², dimensionless.

        ``momenta`` (MeV) and ``couplings`` have the same contract as
        :meth:`current`. The return shape is the event shape. This is the
        physical rest-frame unpolarized weight, with no phase-space measure
        or identical-particle factor included. It excludes the source-
        longitudinal projection. Three pi0 has exactly zero physical weight.
        """
        g3 = _g3(couplings)
        ps, _, _ = self._kinematics(momenta)
        shape = ps.shape[2:]
        result = _axial.three_pi_differential_weight(
            _components(ps, shape, 2), g3, *self._parameters
        )
        result = _real_array(result.reshape(shape), "Differential weight")
        return float(result) if result.ndim == 0 else result

    def spectral_functions(
        self, q: ArrayLike, couplings: AxialCurrent, *, quadrature_order: int = 64
    ) -> AxialSpectralResult:
        r"""Integrate A_T and source A_L in MeV² over invariant phase space.

        Parameters
        ----------
        q : array-like
            Nonnegative invariant energy in MeV, at most ``q_max``.
        couplings : AxialCurrent
            Dimensionless pure isovector source with QCD metadata.
        quadrature_order : int
            Gauss-Legendre order in the smooth invariant-mass coordinate;
            8 through 512. Default 64; compare higher orders if needed.

        Returns
        -------
        result : AxialSpectralResult
            Spectral functions with the shape of ``q``. They vanish at and
            below threshold. The neutral-channel factor 1/3! is applied once.

        Notes
        -----
        With s=(p1+p2)², all angular integrals are analytic. Integrate over
        4m² <= s <= (q-m)² using
        dPhi3/ds = sqrt(lambda(q²,s,m²))*sqrt(1-4m²/s)/(128*pi³*q²),
        which is dimensionless times ds (MeV²), yielding a phase volume in MeV².
        The substitution s=4m²+[(q-m)²-4m²]*(1-cos(theta))/2 removes
        the endpoint square roots. For m=0 the total volume is q²/(256*pi³).
        This is the full dPhi measure, without a Gram-matrix factor of 1/2.
        """
        g3 = _g3(couplings)
        energies = _real_array(q, "q")
        if np.any(energies < 0) or np.any(energies > self.q_max):
            raise ValueError(
                "q must be nonnegative and within the declared LO support."
            )
        if (
            isinstance(quadrature_order, (bool, np.bool_))
            or not isinstance(quadrature_order, (int, np.integer))
            or not _MIN_ORDER <= quadrature_order <= _MAX_ORDER
        ):
            raise ValueError("quadrature_order must be an integer from 8 through 512.")
        transverse, longitudinal = _axial.three_pi_spectral_functions(
            energies.reshape(-1),
            g3,
            *self._parameters,
            int(quadrature_order),
        )
        return AxialSpectralResult(
            q_mev=energies,
            transverse=transverse.reshape(energies.shape),
            source_longitudinal=longitudinal.reshape(energies.shape),
            threshold_mev=self.threshold,
            coverage=self.coverage,
            units="MeV^2",
        )

    def width(
        self, q: ArrayLike, couplings: AxialCurrent, *, quadrature_order: int = 64
    ) -> float | np.ndarray:
        """Return physical spin-1 width A_T/(2q) in MeV.

        Parameters match :meth:`spectral_functions`. All physical
        polarizations, including helicity zero, are in A_T. The source A_L
        is excluded. At zero energy and below threshold the width is zero.
        """
        result = self.spectral_functions(
            q, couplings, quadrature_order=quadrature_order
        )
        energies = np.asarray(result.q_mev)
        value = np.divide(
            result.transverse,
            2 * energies,
            out=np.zeros_like(energies),
            where=energies > 0,
        )
        return float(value) if value.ndim == 0 else value


class AxialFormFactorPiPiPi0(_AxialThreePion):
    """LO pi+ pi- pi0 current in exact isospin.

    Keyword-only ``pion_mass``, ``decay_constant`` and ``q_max`` are in MeV.
    Defaults are the neutral pion mass, F_pi=92.2 MeV, and restricted LO
    support through 600 MeV. One common pion mass is used for every leg;
    zero mass is supported for chiral-limit checks. ``q_max`` may lower but
    cannot exceed the 600 MeV support ceiling. No precision claim is
    attached to the support endpoint. Only a pure isovector source is accepted.
    """


class AxialFormFactorPi0Pi0Pi0(_AxialThreePion):
    """LO three-pi0 current; identical parameters to AxialFormFactorPiPiPi0.

    The full current is longitudinal to q. Its source A_L is nonzero for
    massive pions; its physical spin-1 width and transverse weight vanish.
    """

    _neutral: ClassVar[bool] = True
    _channel: ClassVar[str] = "pi0 pi0 pi0"
