"""Converged, independently integrated values for the thermal averages.

``cross_sections.scalar.thermal_cross_section`` and its vector twin are
the two corpus cases whose stored arrays hold the *unconverged* estimate
QUADPACK returns when its default ``epsabs`` of 1.49e-8 is applied to an
integral of order 1e-27: the absolute criterion is satisfied by the very
first Gauss-Kronrod pass, so the initial three-interval partition comes
back unrefined and no subdivision ever happens. Roster entry ``B6`` in
``projects/parity-pinned-defect-repair`` zeroes that ``epsabs`` in both
kernels, which leaves the relative criterion binding.

This module supplies the value the repaired kernels are then held to.
It rebuilds each model's ``sigma_xx_to_all`` from the per-channel
kernels ``hazma._core`` exports -- the same six summands in the same
order the Rust integrand uses internally (``sigma_xx_to_all`` in
``rust/src/kernels/vector_xs.rs`` and in ``scalar_xs.rs``) -- applies
the Boltzmann weight, and integrates with **scipy's** QUADPACK at a
convergent tolerance.

Independence is in the integrator, which is exactly the component the
repair changes: scipy ships its own QUADPACK, the crate a Rust
transcription of it. The integrand is the controlled variable and is
deliberately shared, because it is not what moved -- the five
closed-form vector kernels reproduce the pre-port Cython bit for bit at
every one of the 5,811 corpus positions that sample them.

That reference integrates the interval the kernels ran over when
``B6`` landed, ``[2, max(floor, 50/x)]``, and it shares their defect
there: at large ``x`` the whole integrand sits within a few ``1/x`` of
threshold, so on a fixed interval QUADPACK's first nodes land in the tail
and its error estimate misses the peak, at any tolerance. Roster entry
``C7`` builds both kernels' interval from ``1/x`` and their channel
thresholds (``rust/src/kernels/thermal_window.rs``), and `interval_term` is
what that adds on top of `reference_values`: the difference between a
**decay-length-split** integral, which no interval choice can mislead, and
the ``B6`` value.

The two large-``x`` rules below are the kernels' own and are **not**
what either repair touches: the scalar hard-returns ``0.0`` above
``x = 300`` while the vector clips ``x`` to 300 and saturates. That
divergence between the two models predates the port and is untouched
here.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.integrate as si
from scipy.integrate import IntegrationWarning
from scipy.special import k1, kn

from hazma import parameters
from hazma._core import scalar_mediator as _core_scalar
from hazma._core import vector_mediator as _core_vector

if TYPE_CHECKING:
    from cases import Block

#: Electron and muon masses in MeV, spelled as the two cross-section
#: kernels carry them (``rust/src/kernels/vector_xs.rs:123,125`` and
#: ``rust/src/kernels/scalar_xs.rs:104,106``), which hard-code them
#: rather than reading ``constants::pdg``. Hard-coded here for the same
#: reason the kernels do: an oracle built on a different electron mass
#: would be integrating a different function.
ME = 0.510998928
MMU = 105.6583715

#: Floor on the upper integration limit, per model, before ``C7``. The
#: kernels integrated to ``max(floor, 50/x)`` -- 100 for the scalar, 150
#: for the vector -- so the floor bound everywhere above ``x = 0.5`` and
#: ``x = 1/3`` respectively.
UPPER_FLOOR = {"scalar": 100.0, "vector": 150.0}

#: Where each kernel stops integrating in ``x = m_x / T``. Above it the
#: Bessel prefactor ``x / (2 K_2(x))^2`` overflows a double.
X_CLIP = 300.0

#: Lower limit of the integral, in ``z = e_cm / m_x``: the pair
#: production threshold ``e_cm = 2 m_x``, where every cross section
#: divides by zero and the ``(z^2 - 4)`` weight is zero.
Z_THRESHOLD = 2.0

#: Relative tolerance for the reference quadrature, with ``epsabs`` off
#: so it is the criterion that binds. Four decades tighter than the
#: 1.49e-8 the repaired kernels run at, which is what makes this a
#: reference for them rather than a second opinion at the same accuracy.
REFERENCE_EPSREL = 1e-12

#: Subdivision limit. The integrand is a near-exponential spike against
#: an interval running out to 150, so the converged partition is deep;
#: 200 is twice the kernels' own ``THERMAL_LIMIT`` of 100 subdivisions
#: beyond their break points, which keeps the reference from being the
#: shallower of the two quadratures.
REFERENCE_LIMIT = 200

#: Where the converged integral is split, in decay lengths ``1/x`` past
#: each of `_features`. The last is twice the ``100`` the kernels run past
#: their last feature (``rust/src/kernels/thermal_window.rs``), so the
#: reference does not share their upper limit. Every piece sees its share
#: of the ``exp(-x z)`` fall-off, so no single Gauss-Kronrod pass can
#: sample only the tail.
DECAY_LENGTHS = (0.0, 1.0, 4.0, 16.0, 50.0, 100.0, 200.0)

#: Ratio between successive splits bracketing the resonance, in units of
#: its width. Half the kernels' ``RESONANCE_LADDER_RATIO`` of 4
#: (``rust/src/kernels/thermal_window.rs``), so the reference's pieces
#: around the peak are not the kernels' and no piece is many widths long
#: with the peak at its end, where no Gauss-Kronrod node samples it.
RESONANCE_LADDER_RATIO = 2.0

#: Index of the mediator width in each model's argument tuple.
_WIDTH_INDEX = {"scalar": 7, "vector": 8}


def _sigma_all(
    model: str, args: list[float]
) -> tuple[Callable[[float], float], float, float]:
    """``sigma_xx_to_all`` for one model, plus its ``m_x`` and mediator mass.

    Parameters
    ----------
    model : {'scalar', 'vector'}
        Which mediator family.
    args : list of float
        The block's stored argument tuple, i.e. everything the entry
        point takes after the swept ``x``.

    Returns
    -------
    sigma_all : callable
        ``e_cm`` in MeV to the summed annihilation cross section in
        MeV^-2.
    mx, m_med : float
        Dark matter and mediator masses in MeV.
    """
    mx, m_med = args[0], args[1]
    if model == "scalar":
        # (gsxx, gsff, gsGG, gsFF, lam, width_s, vs), and the .pyx's own
        # summation order: e, mu, gg, pi0pi0, pipi, ss.
        rest = tuple(args[2:9])

        def sigma_all(e_cm: float) -> float:
            return (
                _core_scalar.sigma_xx_to_s_to_ff(e_cm, mx, m_med, *rest, ME)
                + _core_scalar.sigma_xx_to_s_to_ff(e_cm, mx, m_med, *rest, MMU)
                + _core_scalar.sigma_xx_to_s_to_gg(e_cm, mx, m_med, *rest)
                + _core_scalar.sigma_xx_to_s_to_pi0pi0(e_cm, mx, m_med, *rest)
                + _core_scalar.sigma_xx_to_s_to_pipi(e_cm, mx, m_med, *rest)
                + _core_scalar.sigma_xx_to_ss(e_cm, mx, m_med, *rest)
            )

        return sigma_all, mx, m_med

    # (gvxx, gvuu, gvdd, gvss, gvee, gvmumu, width_v), and the .pyx's own
    # summation order: e, mu, pipi, pi0g, pi0v, vv. The lepton entry
    # point takes the one coupling it uses; the rest take the full tuple
    # and drop what they do not.
    rest = tuple(args[2:9])
    gvxx, _gvuu, _gvdd, _gvss, gvee, gvmumu, width_v = rest

    def sigma_all(e_cm: float) -> float:
        return (
            _core_vector.sigma_xx_to_v_to_ff(e_cm, mx, m_med, gvxx, gvee, width_v, ME)
            + _core_vector.sigma_xx_to_v_to_ff(
                e_cm, mx, m_med, gvxx, gvmumu, width_v, MMU
            )
            + _core_vector.sigma_xx_to_v_to_pipi(e_cm, mx, m_med, *rest)
            + _core_vector.sigma_xx_to_v_to_pi0g(e_cm, mx, m_med, *rest)
            + _core_vector.sigma_xx_to_v_to_pi0v(e_cm, mx, m_med, *rest)
            + _core_vector.sigma_xx_to_vv(e_cm, mx, m_med, *rest)
        )

    return sigma_all, mx, m_med


def _clip(model: str, x: float) -> float | None:
    """The kernels' own large-``x`` rule: ``None`` where the scalar returns 0."""
    if model == "scalar":
        return None if x > X_CLIP else x
    return min(x, X_CLIP)


def _features(model: str, mx: float, m_med: float) -> tuple[float, ...]:
    """Every ``z = e_cm / m_x`` at which a channel opens or the mediator peaks.

    Masses are `hazma.parameters`' rather than the kernels' hard-coded
    ones. A split only has to sit near the feature, and the upper limit
    runs 200 decay lengths past the last one, so a feature moved by a
    fraction of an MeV moves neither the partition's purpose nor the
    value.
    """
    thresholds = [
        2.0 * parameters.electron_mass,
        2.0 * parameters.muon_mass,
        2.0 * parameters.charged_pion_mass,
    ]
    if model == "scalar":
        thresholds.append(2.0 * parameters.neutral_pion_mass)
    else:
        thresholds += [
            parameters.neutral_pion_mass,
            parameters.neutral_pion_mass + m_med,
        ]
    return tuple(m / mx for m in (*thresholds, m_med, 2.0 * m_med))


def _integral(
    model: str,
    args: list[float],
    xnew: float,
    bounds: tuple[float, float],
    points: list[float] | None = None,
) -> float:
    """``<sigma v>`` in MeV^-2 from one ``quad`` call over ``bounds``.

    ``points`` are interior break points; QUADPACK drops any on or outside
    the interval, and filtering them here keeps scipy from raising on the
    duplicates the kernels pass.
    """
    sigma_all, mx, _m_med = _sigma_all(model, args)
    prefactor = xnew / (2.0 * kn(2, xnew)) ** 2
    lo, hi = bounds
    interior = [p for p in points or () if lo < p < hi]

    def integrand(z: float) -> float:
        return sigma_all(mx * z) * z * z * (z * z - 4.0) * k1(xnew * z)

    with warnings.catch_warnings():
        # A few of the corpus positions report roundoff in the
        # extrapolation table at this tolerance. The warning is about
        # what QUADPACK can still certify, not about the value it
        # returns: asking for epsrel 1e-9, 1e-10, 1e-11 and 1e-12 in
        # turn moves the answer by at most 7.8e-10 relative, two decades
        # under the budget `deltas.py` holds the repair to. Suppressed so
        # the parity run stays quiet; widen the tolerance instead if that
        # stability ever stops holding.
        warnings.simplefilter("ignore", IntegrationWarning)
        value, _abserr = si.quad(
            integrand,
            lo,
            hi,
            points=interior or None,
            epsabs=0.0,
            epsrel=REFERENCE_EPSREL,
            limit=REFERENCE_LIMIT,
        )
    return prefactor * value


def thermal_cross_section(model: str, args: list[float], x: float) -> float:
    """``<sigma v>(x)`` in MeV^-2 over the pre-``C7`` interval.

    Integrates ``[2, max(floor, 50/x)]`` with the kernels' break points, to
    `REFERENCE_EPSREL`. This is the ``B6`` reference, and at large ``x``
    it carries the interval's defect: it is 1.9e-4 high at the vector
    ``closed_resonance`` block above ``x = 200``.

    Parameters
    ----------
    model : {'scalar', 'vector'}
        Which mediator family.
    args : list of float
        The block's stored argument tuple.
    x : float
        ``m_x / T``, dimensionless.

    Returns
    -------
    float
        The thermally averaged cross section in MeV^-2.
    """
    xnew = _clip(model, x)
    if xnew is None:
        return 0.0
    ratio = args[1] / args[0]
    upper = max(UPPER_FLOOR[model], 50.0 / xnew)
    return _integral(model, args, xnew, (Z_THRESHOLD, upper), [ratio, 2.0 * ratio])


def converged_thermal_cross_section(model: str, args: list[float], x: float) -> float:
    """``<sigma v>(x)`` in MeV^-2, split so no interval choice can bias it.

    Integrates from threshold to 200 decay lengths past the last of
    `_features`, in pieces at every feature, at `DECAY_LENGTHS` past each,
    and at ``z_r +/- g 2^k`` around the resonance ``z_r = m_med / m_x``
    of width ``g`` in units of ``m_x``, each piece to `REFERENCE_EPSREL`.

    Parameters
    ----------
    model : {'scalar', 'vector'}
        Which mediator family.
    args : list of float
        The block's stored argument tuple.
    x : float
        ``m_x / T``, dimensionless.

    Returns
    -------
    float
        The thermally averaged cross section in MeV^-2.
    """
    xnew = _clip(model, x)
    if xnew is None:
        return 0.0
    openings = {
        Z_THRESHOLD,
        *(max(z, Z_THRESHOLD) for z in _features(model, *args[:2])),
    }
    upper = max(openings) + DECAY_LENGTHS[-1] / xnew
    splits = {z + k / xnew for z in openings for k in DECAY_LENGTHS}
    z_res = args[1] / args[0]
    offset = args[_WIDTH_INDEX[model]] / args[0]
    while 0.0 < offset < upper - Z_THRESHOLD:
        splits.update((z_res - offset, z_res + offset))
        offset *= RESONANCE_LADDER_RATIO
    edges = sorted(z for z in splits if Z_THRESHOLD <= z <= upper)
    return sum(_integral(model, args, xnew, pair) for pair in pairwise(edges))


ReferenceFn = Callable[[Callable[..., Any], "Block"], dict[str, np.ndarray]]

#: Which mediator family an entry point belongs to, read off the compiled
#: module it came from. The two thermal cases sweep the same block labels
#: over argument tuples of the same length, so the entry point is what
#: separates them -- and it is what `deltas.Reference` is handed.
_MODEL_BY_MODULE = {
    "hazma._core.scalar_mediator": "scalar",
    "hazma._core.vector_mediator": "vector",
}


def _model(fn: Callable[..., Any]) -> str:
    """Which mediator family an entry point belongs to."""
    module = getattr(fn, "__module__", "")
    try:
        return _MODEL_BY_MODULE[module]
    except KeyError:  # pragma: no cover - a new case would have to opt in
        msg = f"no thermal reference for an entry point from {module!r}"
        raise KeyError(msg) from None


def reference_values(fn: Callable[..., Any], block: Block) -> dict[str, np.ndarray]:
    """`deltas.Reference` callable for either mediator's thermal case.

    The ``B6`` reference: `thermal_cross_section` on the block's grid.

    Parameters
    ----------
    fn : callable
        The corpus entry point under comparison. Read only to tell the
        scalar case from the vector one; the reference never calls it,
        which is the point of being a reference.
    block : Block
        The corpus block, supplying the grid and the argument tuple.

    Returns
    -------
    dict
        ``{"values": <sigma v> in MeV^-2 on the block's grid}``.
    """
    model = _model(fn)
    args = block.params["args"]
    return {
        "values": np.array(
            [thermal_cross_section(model, args, float(x)) for x in block.grid],
            dtype=np.float64,
        )
    }


def interval_term(fn: Callable[..., Any], block: Block) -> dict[str, np.ndarray]:
    """`deltas.Additive` term for ``C7``, composed after `reference_values`.

    What rebuilding the kernels' interval from ``1/x`` adds to the
    ``B6`` value: `converged_thermal_cross_section` minus
    `thermal_cross_section`, on the block's grid.

    Parameters
    ----------
    fn : callable
        The corpus entry point under comparison, read only for its model.
    block : Block
        The corpus block, supplying the grid and the argument tuple.

    Returns
    -------
    dict
        ``{"values": the term in MeV^-2 on the block's grid}``.
    """
    model = _model(fn)
    args = block.params["args"]
    return {
        "values": np.array(
            [
                converged_thermal_cross_section(model, args, float(x))
                - thermal_cross_section(model, args, float(x))
                for x in block.grid
            ],
            dtype=np.float64,
        )
    }
