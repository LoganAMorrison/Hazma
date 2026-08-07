"""Tests for the pure-Python kinematic helpers in :mod:`hazma.utils`.

These cover the two functions that took over from the deleted Cython module
``hazma.field_theory_helper_functions.common_functions`` in the
Cython-to-Rust migration (project ``cython-to-rust``, Task 0.3):
``minkowski_dot`` (relocated here) and ``cross_section_prefactor`` (already
present; its two callers at the time, ``hazma.deprecated.rambo`` and
``hazma.gamma_ray``, were repointed here before Task 0.2 deleted both),
plus ``two_body_momentum``, the numerically stable momentum both
``cross_section_prefactor`` and the two-body phase-space integrators are
built on.

The pinned tolerances below are chosen from the expected floating-point
error of each closed form, not from what makes the assertion pass; each
one states its reasoning.
"""

from fractions import Fraction

import numpy as np
import pytest

from hazma.parameters import (
    charged_kaon_mass,
    charged_pion_mass,
    electron_mass,
    muon_mass,
    neutral_pion_mass,
)
from hazma.utils import (
    cross_section_prefactor,
    kallen_lambda,
    ldot,
    minkowski_dot,
    two_body_momentum,
)

#: Mass pairs spanning four decades of mass ratio, from e-e (1:1) to
#: e-K (1:966). The unequal pairs are where an ill-ordered subtraction
#: shows up; the equal ones are where the Kaellen cancellation is worst.
MASS_PAIRS = [
    (electron_mass, electron_mass),
    (electron_mass, muon_mass),
    (electron_mass, charged_kaon_mass),
    (muon_mass, muon_mass),
    (muon_mass, neutral_pion_mass),
    (neutral_pion_mass, charged_pion_mass),
    (charged_pion_mass, charged_kaon_mass),
    (charged_kaon_mass, charged_kaon_mass),
]

#: Masses for the tests that must sit exactly on threshold. Only an equal
#: pair can: ``m + m = 2m`` is exact in binary, while ``m1 + m2`` for
#: unequal masses rounds to one side of the true threshold or the other.
THRESHOLD_MASSES = [electron_mass, muon_mass, charged_pion_mass, charged_kaon_mass]


def exact_two_body_momentum(cme: float, m1: float, m2: float) -> float:
    """Reference `p` with the Kaellen polynomial evaluated in exact rationals.

    Every float is an exact rational, so `Fraction` carries the polynomial
    with no rounding at all; the only error left is the single rounding of
    the result to a float, the square root, and the final divide. That puts
    the reference within ~2 ulp (~5e-16 relative) of the true value no
    matter how close `cme` sits to threshold, which is what makes it an
    independent oracle rather than a restatement of either implementation.
    """
    s = Fraction(cme) ** 2
    a = Fraction(m1) ** 2
    b = Fraction(m2) ** 2
    lam = s * s + a * a + b * b - 2 * s * a - 2 * s * b - 2 * a * b
    return np.sqrt(float(lam)) / (2 * cme)


# ===================================================================
# ---- two_body_momentum --------------------------------------------
# ===================================================================


@pytest.mark.parametrize("cme", [10.0, 100.0, 1000.0])
def test_two_body_momentum_massless_limit(cme: float) -> None:
    """Give exactly cme/2 for two massless particles.

    Every factor is then cme, so the product is cme**4 and the square root
    is exact in binary floating point. Pin equality, not a tolerance.
    """
    assert two_body_momentum(cme, 0.0, 0.0) == cme / 2.0


@pytest.mark.parametrize(("m1", "m2"), MASS_PAIRS)
def test_two_body_momentum_matches_exact_reference(m1: float, m2: float) -> None:
    """Match the exact-rational reference far above threshold.

    `rel=1e-15` is a few ulp: with cme at 4x threshold no factor cancels,
    so both sides are short products with only rounding between them.
    """
    cme = 4.0 * (m1 + m2)
    assert two_body_momentum(cme, m1, m2) == pytest.approx(
        exact_two_body_momentum(cme, m1, m2), rel=1e-15
    )


@pytest.mark.parametrize(("m1", "m2"), MASS_PAIRS)
@pytest.mark.parametrize("eps", [1e-1, 1e-3, 1e-5, 1e-7, 1e-10, 1e-12])
def test_two_body_momentum_stable_near_threshold(
    m1: float, m2: float, eps: float
) -> None:
    """Hold full relative accuracy arbitrarily close to threshold.

    This is the property the factored, heavier-mass-first form buys, and
    the reason it replaced ``sqrt(kallen_lambda(...))`` — see
    ``docs/followups/done/cross-section-prefactor-threshold-cancellation.md``.
    `rel=1e-14` sits ~20x above the 4.4e-16 measured worst case over this
    grid, which leaves room for a different libm sqrt without leaving room
    for a regression: the old form is off by 4e-5 at eps=1e-12 and 1e-7 at
    eps=1e-7, both of which blow straight through this bound.
    """
    cme = (m1 + m2) * (1.0 + eps)
    assert two_body_momentum(cme, m1, m2) == pytest.approx(
        exact_two_body_momentum(cme, m1, m2), rel=1e-14
    )


@pytest.mark.parametrize(("m1", "m2"), MASS_PAIRS)
def test_two_body_momentum_beats_kallen_near_threshold(m1: float, m2: float) -> None:
    """Beat the expanded Kaellen form by orders of magnitude near threshold.

    A direct A/B against the form this replaced, so the improvement is
    pinned rather than asserted in a docstring. At 1e-10 above threshold
    the Kaellen route keeps ~5 of 16 digits; the factored route keeps all
    of them. `100x` is a deliberately weak bound — the measured ratio is
    ~1e6 — chosen so the test states "strictly better, by a lot" without
    depending on the exact roundoff of either route.
    """
    cme = (m1 + m2) * (1.0 + 1e-10)
    reference = exact_two_body_momentum(cme, m1, m2)
    kallen = np.sqrt(kallen_lambda(cme**2, m1**2, m2**2)) / (2 * cme)

    err_new = abs(two_body_momentum(cme, m1, m2) - reference)
    err_old = abs(kallen - reference)
    assert err_new * 100.0 < err_old


@pytest.mark.parametrize(("m1", "m2"), MASS_PAIRS)
def test_two_body_momentum_symmetric_in_masses(m1: float, m2: float) -> None:
    """Return the same value under a swap of the two masses.

    The implementation sorts the masses before subtracting, so this holds
    bit-for-bit. It is worth pinning because the sort is exactly the part
    a future simplification would be tempted to drop.
    """
    cme = 1.5 * (m1 + m2)
    assert two_body_momentum(cme, m1, m2) == two_body_momentum(cme, m2, m1)


@pytest.mark.parametrize("mass", THRESHOLD_MASSES)
def test_two_body_momentum_vanishes_at_threshold(mass: float) -> None:
    """Return exactly zero at threshold, not a roundoff residue.

    At ``cme = m1 + m2`` the physical momentum is zero. Equal masses are
    used because ``m + m = 2m`` is exact in binary, so `cme` lands on the
    true threshold instead of an ulp to one side of it (the unequal case
    is `test_two_body_momentum_threshold_resolved_to_the_last_bit`). The
    heavier-first subtraction is exact there, so the leading factor is an
    exact zero and no tolerance is needed. The expanded Kaellen form
    instead left a positive residue here, which is what made
    `cross_section_prefactor` return a large finite number where it should
    diverge.
    """
    assert two_body_momentum(2.0 * mass, mass, mass) == 0.0


@pytest.mark.parametrize(("m1", "m2"), MASS_PAIRS)
def test_two_body_momentum_threshold_resolved_to_the_last_bit(
    m1: float, m2: float
) -> None:
    """Put the domain boundary exactly where exact arithmetic puts it.

    For unequal masses ``m1 + m2`` rounds, so the float handed in as `cme`
    sits an ulp above or below the true threshold — and which side it
    falls on decides whether a two-body state exists at all. The factored
    form resolves that to the last bit: it returns a real momentum when
    the rounded sum is at or above the exact threshold and NaN when it is
    below. The Kaellen form could not, since its own roundoff residue at
    this scale is larger than the distance being resolved.
    """
    cme = m1 + m2  # fl(m1 + m2), which need not be the exact threshold
    at_or_above = Fraction(cme) >= Fraction(m1) + Fraction(m2)

    with np.errstate(invalid="ignore"):
        p = two_body_momentum(cme, m1, m2)

    if at_or_above:
        assert np.isfinite(p)
        assert 0.0 <= p < min(m1, m2)  # an ulp of headroom, not a real momentum
    else:
        assert np.isnan(p)


@pytest.mark.parametrize(("m1", "m2"), MASS_PAIRS)
@pytest.mark.parametrize(
    "fraction",
    [0.999, 0.9, 0.5, 0.1, 1e-3, 1e-9],
    ids=["thr999", "thr9", "thr5", "thr1", "thr1e-3", "thr1e-9"],
)
def test_two_body_momentum_nan_below_threshold(
    m1: float, m2: float, fraction: float
) -> None:
    """Return NaN at every `cme` below threshold, where no state exists.

    The fractions sweep the whole unphysical domain, not just the part
    where the Kaellen polynomial happens to be negative. `lambda` has two
    roots, ``|m1 - m2|`` and ``m1 + m2``, and is negative only *between*
    them; below the lower root it turns positive again. A single square
    root over the full product therefore returned a finite, meaningless
    momentum down there — `two_body_momentum(1.0, 10.0, 1.0)` gave 48.99
    despite a threshold of 11. The two-root grouping fixes that, and the
    small fractions here are what pin it: `1e-9` is far below ``|m1 - m2|``
    for every unequal pair in `MASS_PAIRS`.
    """
    with np.errstate(invalid="ignore"):
        assert np.isnan(two_body_momentum(fraction * (m1 + m2), m1, m2))


def test_two_body_momentum_nan_below_lower_kallen_root() -> None:
    """Pin the exact reviewer reproducer for the lower-root region.

    Kept as a literal alongside the swept test above so the regression
    that motivated the two-root grouping is named and reproducible rather
    than only implied by a parametrization. Masses 10 and 1 put the
    Kaellen roots at 9 and 11; `cme = 1` sits below both, where the old
    single-root form returned 48.98979485566356.
    """
    with np.errstate(invalid="ignore"):
        assert np.isnan(two_body_momentum(1.0, 10.0, 1.0))
        assert np.isnan(cross_section_prefactor(10.0, 1.0, 1.0))


def test_two_body_momentum_broadcasts_over_cme() -> None:
    """Accept arrays and broadcast, per the arrays-in/arrays-out contract."""
    cme = np.array([300.0, 400.0, 500.0])
    got = two_body_momentum(cme, muon_mass, charged_pion_mass)
    expected = np.array(
        [two_body_momentum(float(q), muon_mass, charged_pion_mass) for q in cme]
    )
    assert got.shape == cme.shape
    assert np.array_equal(got, expected)


def test_two_body_momentum_broadcasts_over_masses() -> None:
    """Broadcast over the masses too, with `cme` held scalar."""
    masses = np.array([electron_mass, muon_mass, charged_pion_mass])
    got = two_body_momentum(1000.0, masses, masses)
    expected = np.array([two_body_momentum(1000.0, float(m), float(m)) for m in masses])
    assert np.array_equal(got, expected)


# ===================================================================
# ---- minkowski_dot ------------------------------------------------
# ===================================================================


def test_minkowski_dot_sign_convention() -> None:
    """West-coast (+,-,-,-) metric, pinned on an exactly-representable case."""
    fv1 = np.array([7.0, 1.0, 2.0, 3.0])
    fv2 = np.array([5.0, 4.0, 8.0, 16.0])
    # 7*5 - 1*4 - 2*8 - 3*16 = 35 - 4 - 16 - 48 = -33
    assert minkowski_dot(fv1, fv2) == -33.0  # noqa: PLR2004 -- the pinned value


def test_minkowski_dot_is_not_euclidean() -> None:
    """Guard against a (+,+,+,+) regression.

    The sign test above can miss this on its own if someone flips both the
    metric and the operand order, so pin a self-product too.
    """
    fv = np.array([2.0, 1.0, 0.0, 0.0])
    assert minkowski_dot(fv, fv) == 3.0  # noqa: PLR2004 -- 4 - 1, not 4 + 1


@pytest.mark.parametrize("mass", [electron_mass, muon_mass, charged_pion_mass])
def test_minkowski_dot_on_shell_invariant(mass: float) -> None:
    """Check p.p == m^2 for an on-shell four-momentum.

    Tolerance: E^2 - |p|^2 is a cancellation of two numbers of size E^2, so
    the relative error grows like (E/m)^2 * eps. With |p| = 3 * m the ratio
    E^2/m^2 is 10, which puts the expected error near 1e-15; 1e-12 leaves
    three decades of headroom without being able to absorb a real bug (the
    smallest one, a dropped spatial component, moves the result by ~90%).
    """
    p3 = mass * np.array([1.0, 2.0, 2.0])  # |p| = 3 * mass
    energy = np.sqrt(mass**2 + p3 @ p3)
    fv = np.array([energy, *p3])
    assert minkowski_dot(fv, fv) == pytest.approx(mass**2, rel=1e-12)


def test_minkowski_dot_matches_ldot() -> None:
    """Agree bit-for-bit with `ldot`, the array-oriented twin.

    Both evaluate p0 - p1 - p2 - p3 in that order, so on a single
    four-vector there is no room for a difference.
    """
    rng = np.random.default_rng(20260804)
    for _ in range(100):
        fv1 = rng.normal(scale=100.0, size=4)
        fv2 = rng.normal(scale=100.0, size=4)
        assert minkowski_dot(fv1, fv2) == ldot(fv1, fv2)


def test_minkowski_dot_accepts_lists() -> None:
    """Accept plain sequences; the implementation is index-based."""
    assert minkowski_dot([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]) == 1.0


# ===================================================================
# ---- cross_section_prefactor --------------------------------------
# ===================================================================


@pytest.mark.parametrize("cme", [10.0, 100.0, 1000.0])
def test_cross_section_prefactor_massless_limit(cme: float) -> None:
    """Reduce to 1/(2 cme^2) for m1 = m2 = 0.

    There p = cme/2 exactly, so the general 1/(4 p cme) collapses to a
    closed form. `rel=1e-15` is one ulp-ish: both sides are short exact
    expressions with no cancellation.
    """
    assert cross_section_prefactor(0.0, 0.0, cme) == pytest.approx(
        1.0 / (2.0 * cme**2), rel=1e-15
    )


@pytest.mark.parametrize("mass", [electron_mass, muon_mass, charged_pion_mass])
def test_cross_section_prefactor_equal_masses(mass: float) -> None:
    """Match the equal-mass closed form 1/(4 p cme), p = sqrt(cme^2/4 - m^2).

    That route to p goes through cme^2/4 - m^2, which shares no
    floating-point path with the four-factor product `two_body_momentum`
    forms, so this is an independent check rather than a restatement.
    `rel=1e-13` covers the mild cancellation in cme^2/4 - m^2 at cme = 4m
    (where the two terms differ by 4x).
    """
    cme = 4.0 * mass  # comfortably above the 2m threshold
    p = np.sqrt(cme**2 / 4.0 - mass**2)
    assert cross_section_prefactor(mass, mass, cme) == pytest.approx(
        1.0 / (4.0 * p * cme), rel=1e-13
    )


def test_cross_section_prefactor_scales_as_inverse_cme_squared() -> None:
    """Fall as 1/cme^2 far above threshold, where the masses are negligible.

    A factor-10 step in cme must drop the prefactor by 100. `rel=1e-8` is
    loose on purpose: the residual mass dependence, not roundoff, sets the
    error here.
    """
    lo = cross_section_prefactor(electron_mass, electron_mass, 1.0e4)
    hi = cross_section_prefactor(electron_mass, electron_mass, 1.0e5)
    assert lo / hi == pytest.approx(100.0, rel=1e-8)


def test_cross_section_prefactor_grows_toward_threshold() -> None:
    """Grow monotonically as cme -> m1 + m2, since p -> 0.

    The steps run to 1e-12 above threshold. They used to stop at 1e-4,
    because the `kallen_lambda` route lost too many digits below that to
    stay monotone; `two_body_momentum` holds full relative accuracy all
    the way down, so the floor is now set by float64 itself rather than by
    the algebra.
    """
    threshold = 2.0 * muon_mass
    values = [
        cross_section_prefactor(muon_mass, muon_mass, threshold * (1.0 + eps))
        for eps in [1.0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    ]
    # strict=False: pairing a list with its own tail is intentionally ragged.
    assert all(a < b for a, b in zip(values, values[1:], strict=False))


@pytest.mark.parametrize("mass", THRESHOLD_MASSES)
def test_cross_section_prefactor_diverges_at_threshold(mass: float) -> None:
    """Diverge at cme = m1 + m2, where the relative velocity vanishes.

    The flux factor is 1/(4 p cme) and p is exactly zero at threshold, so
    the physical answer is +inf. This replaces a test that pinned the old
    limitation: `kallen_lambda` left a roundoff residue instead of zero, so
    the function returned a large finite number here. Resolved in
    ``docs/followups/done/cross-section-prefactor-threshold-cancellation.md``.
    """
    with np.errstate(divide="ignore"):
        value = cross_section_prefactor(mass, mass, 2.0 * mass)
    assert np.isposinf(value)


@pytest.mark.parametrize(("m1", "m2"), MASS_PAIRS)
def test_cross_section_prefactor_matches_exact_reference_near_threshold(
    m1: float, m2: float
) -> None:
    """Track 1/(4 p cme) from the exact-rational p to 1e-10 above threshold.

    The end-to-end statement of the fix: the shift the follow-up predicted
    lands in the returned prefactor, not just in the momentum helper.
    `rel=1e-14` is the same few-ulp budget as
    `test_two_body_momentum_stable_near_threshold`, since the extra divide
    contributes one rounding.
    """
    cme = (m1 + m2) * (1.0 + 1e-10)
    expected = 1.0 / (4.0 * exact_two_body_momentum(cme, m1, m2) * cme)
    assert cross_section_prefactor(m1, m2, cme) == pytest.approx(expected, rel=1e-14)
