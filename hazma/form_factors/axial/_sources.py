"""Immutable light-flavor axial sources and declared matching coverage."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Literal

import numpy as np

from hazma.utils import RealOrRealArray

ArrayLike = RealOrRealArray | list[float] | tuple[float, ...]

InputStatus = Literal["lo", "measured", "fitted", "parameterized", "missing"]
_INPUT_STATUSES = ("lo", "measured", "fitted", "parameterized", "missing")


def _finite_real(value: Real, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar.")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real scalar.") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        domain = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must be a {domain} real scalar.")
    return result


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")


def _status(value: InputStatus) -> None:
    if value not in _INPUT_STATUSES:
        raise ValueError(f"input_status must be one of {_INPUT_STATUSES}.")


@dataclass(frozen=True, slots=True)
class AxialCurrent:
    r"""Dimensionless light axial Wilson coefficients in a declared QCD scheme.

    Parameters
    ----------
    gau, gad, gas : float
        Finite real coefficients in flavor order (u, d, s), defined by
        :math:`\mathcal L=Z_\mu\bar q\gamma^\mu\gamma_5G_Aq` with
        :math:`G_A=\mathrm{diag}(g_{Au},g_{Ad},g_{As})`. No factor of 1/2
        or flavor-trace subtraction is implicit.
    qcd_scheme : str
        Name of the QCD-current renormalization convention.
    scale_mev : float
        Positive finite hadronic Wilson/current scale in MeV.
    provenance : str
        Named input or user-supplied provenance, required explicitly.

    Notes
    -----
    Generators satisfy Tr(T_a T_b)=2 delta_ab, with T3=diag(1,-1,0),
    T8=diag(1,1,-2)/sqrt(3), and T0=sqrt(2/3) I. Scheme and scale are
    metadata; this object does not evolve coefficients between scales.
    """

    gau: float
    gad: float
    gas: float
    qcd_scheme: str = field(kw_only=True)
    scale_mev: float = field(kw_only=True)
    provenance: str = field(kw_only=True)

    def __post_init__(self) -> None:
        for name in ("gau", "gad", "gas"):
            object.__setattr__(self, name, _finite_real(getattr(self, name), name))
        object.__setattr__(
            self, "scale_mev", _finite_real(self.scale_mev, "scale_mev", positive=True)
        )
        _text(self.qcd_scheme, "qcd_scheme")
        _text(self.provenance, "provenance")

    @property
    def trace(self) -> float:
        """Return Tr(G_A), dimensionless; reject floating-point overflow."""
        try:
            return math.fsum((self.gau, self.gad, self.gas))
        except OverflowError as exc:
            raise ValueError("Source trace exceeds the finite float domain.") from exc

    @property
    def generator_coefficients(self) -> tuple[float, float, float]:
        """Return dimensionless (c3, c8, c0) in G_A=c3 T3+c8 T8+c0 T0."""
        try:
            coefficients = (
                math.fsum((self.gau / 2.0, -self.gad / 2.0)),
                math.fsum(
                    (
                        self.gau / (2.0 * math.sqrt(3.0)),
                        self.gad / (2.0 * math.sqrt(3.0)),
                        -self.gas / math.sqrt(3.0),
                    )
                ),
                math.fsum(
                    value / math.sqrt(6.0) for value in (self.gau, self.gad, self.gas)
                ),
            )
        except OverflowError as exc:
            raise ValueError(
                "Generator coefficients exceed the finite float domain."
            ) from exc
        return coefficients

    @classmethod
    def from_generator_coefficients(  # noqa: PLR0913 - coefficients plus explicit metadata
        cls,
        c3: float,
        c8: float,
        c0: float,
        *,
        qcd_scheme: str,
        scale_mev: float,
        provenance: str,
    ) -> AxialCurrent:
        """Construct from dimensionless generator coefficients and MeV scale.

        Parameters
        ----------
        c3, c8, c0 : float
            Finite real coefficients multiplying T3, T8, T0, respectively.
        qcd_scheme, provenance : str
            Explicit nonempty QCD convention and input provenance.
        scale_mev : float
            Positive finite QCD-current scale in MeV.

        Returns
        -------
        AxialCurrent
            Flavor coefficients reconstructed without discarding the singlet.
        """
        c3 = _finite_real(c3, "c3")
        c8 = _finite_real(c8, "c8") / math.sqrt(3.0)
        c0 = _finite_real(c0, "c0") * math.sqrt(2.0 / 3.0)
        try:
            gau = math.fsum((c3, c8, c0))
            gad = math.fsum((-c3, c8, c0))
            gas = math.fsum((-2.0 * c8, c0))
        except OverflowError as exc:
            raise ValueError(
                "Flavor coefficients exceed the finite float domain."
            ) from exc
        return cls(
            gau,
            gad,
            gas,
            qcd_scheme=qcd_scheme,
            scale_mev=scale_mev,
            provenance=provenance,
        )

    def rescale_singlet(
        self, factor: float, *, qcd_scheme: str, provenance: str
    ) -> AxialCurrent:
        r"""Change the finite singlet-current convention, without scale evolution.

        Parameters
        ----------
        factor : float
            Finite nonzero dimensionless factor defining J0'=factor J0.
            The new Wilson coefficient is c0'=c0/factor. To preserve the
            physical amplitude, a corresponding singlet matrix element or
            residue must be multiplied by factor; this method does not alter
            any separately held hadronic input.
        qcd_scheme, provenance : str
            Explicit names for the new convention and its provenance.

        Returns
        -------
        AxialCurrent
            New convention's source, at the original scale in MeV.
        """
        factor = _finite_real(factor, "factor")
        if factor == 0.0:
            raise ValueError("factor must be nonzero.")
        c3, c8, c0 = self.generator_coefficients
        return self.from_generator_coefficients(
            c3,
            c8,
            c0 / factor,
            qcd_scheme=qcd_scheme,
            scale_mev=self.scale_mev,
            provenance=provenance,
        )

    @property
    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 of coefficients and source metadata."""
        payload = {
            "gau": self.gau,
            "gad": self.gad,
            "gas": self.gas,
            "qcd_scheme": self.qcd_scheme,
            "scale_mev": self.scale_mev,
            "provenance": self.provenance,
            "normalization": "L=Z_mu qbar gamma^mu gamma5 diag(gau,gad,gas) q",
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class MatchingCard:
    """Declare source matching choices without inferring a UV completion.

    Parameters
    ----------
    current : AxialCurrent
        Hadronic Wilson/current convention and scale in MeV.
    chiral_scale_mev : float
        Positive finite chiral subtraction scale in MeV, separate from the
        QCD-current scale held in current.
    normalization : str
        Declared matching normalization; does not change current's definition.
    anomaly_matching : str or None
        Named heavy-fermion/anomaly matching input; None declares no input.
    uv_completion : str or None
        Named UV completion; None declares none specified.
    input_status : {'lo', 'measured', 'fitted', 'parameterized', 'missing'}
        Declared input category, not an automatically certified precision.
    provenance : str
        Explicit input provenance.
    """

    current: AxialCurrent
    chiral_scale_mev: float
    normalization: str
    anomaly_matching: str | None
    uv_completion: str | None
    input_status: InputStatus
    provenance: str

    def __post_init__(self) -> None:
        if not isinstance(self.current, AxialCurrent):
            raise ValueError("current must be an AxialCurrent.")
        object.__setattr__(
            self,
            "chiral_scale_mev",
            _finite_real(self.chiral_scale_mev, "chiral_scale_mev", positive=True),
        )
        _text(self.normalization, "normalization")
        _text(self.provenance, "provenance")
        for name in ("anomaly_matching", "uv_completion"):
            if getattr(self, name) is not None:
                _text(getattr(self, name), name)
        _status(self.input_status)


@dataclass(frozen=True, slots=True)
class SourceCoverage:
    """Declared inclusive invariant-mass domain and missing source inputs.

    Parameters
    ----------
    channel : str
        Explicit channel identifier.
    q_min_mev, q_max_mev : float
        Finite invariant-mass domain endpoints in MeV, satisfying
        0 <= q_min_mev <= q_max_mev. A validity endpoint is not implicitly
        the channel's kinematic threshold.
    input_status : {'lo', 'measured', 'fitted', 'parameterized', 'missing'}
        Declared input category. No uncertainty or precision is inferred.
    missing_directions : tuple of str
        Names of unsupported input directions; immutable after construction.
    provenance : str
        Explicit input provenance.
    """

    channel: str
    q_min_mev: float
    q_max_mev: float
    input_status: InputStatus
    missing_directions: tuple[str, ...] = ()
    provenance: str = field(kw_only=True)

    def __post_init__(self) -> None:
        _text(self.channel, "channel")
        _text(self.provenance, "provenance")
        _status(self.input_status)
        for name in ("q_min_mev", "q_max_mev"):
            object.__setattr__(self, name, _finite_real(getattr(self, name), name))
        if self.q_min_mev < 0.0 or self.q_max_mev < self.q_min_mev:
            raise ValueError("Coverage requires 0 <= q_min_mev <= q_max_mev.")
        if isinstance(self.missing_directions, str):
            raise ValueError("missing_directions must be a sequence of names.")
        try:
            directions = tuple(self.missing_directions)
        except TypeError as exc:
            raise ValueError("missing_directions must be a sequence of names.") from exc
        for direction in directions:
            _text(direction, "missing_directions entry")
        object.__setattr__(self, "missing_directions", directions)

    def contains(self, q_mev: ArrayLike) -> bool | np.ndarray:
        """Test finite nonnegative q in MeV against the inclusive domain.

        Parameters
        ----------
        q_mev : float or array_like
            Finite nonnegative invariant masses in MeV.

        Returns
        -------
        bool or numpy.ndarray
            Membership for each mass, preserving array shape.
        """
        raw = np.asarray(q_mev)
        if raw.dtype.kind not in "iuf":
            raise ValueError("q_mev must contain finite nonnegative real masses.")
        q = np.asarray(raw, dtype=np.float64)
        if np.any(~np.isfinite(q)) or np.any(q < 0.0):
            raise ValueError("q_mev must contain finite nonnegative real masses.")
        result = (q >= self.q_min_mev) & (q <= self.q_max_mev)
        return bool(result) if result.ndim == 0 else result
