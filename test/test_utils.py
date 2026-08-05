"""Tests for the pure-Python kinematic helpers in :mod:`hazma.utils`.

These cover the two functions that took over from the deleted Cython module
``hazma.field_theory_helper_functions.common_functions`` in the
Cython-to-Rust migration (project ``cython-to-rust``, Task 0.3):
``minkowski_dot`` (relocated here) and ``cross_section_prefactor`` (already
present; ``hazma.deprecated.rambo`` and ``hazma.gamma_ray`` now call it).

The pinned tolerances below are chosen from the expected floating-point
error of each closed form, not from what makes the assertion pass; each
one states its reasoning.
"""

import numpy as np
import pytest

from hazma.parameters import charged_pion_mass, electron_mass, muon_mass
from hazma.utils import cross_section_prefactor, ldot, minkowski_dot

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

    That route to p shares no floating-point path with the `kallen_lambda`
    one the implementation uses, so this is an independent check rather
    than a restatement. `rel=1e-13` covers the mild cancellation in
    cme^2/4 - m^2 at cme = 4m (where the two terms differ by 4x).
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

    The steps stop at 1e-4 above threshold; closer than that the
    `kallen_lambda` evaluation loses too many digits to stay monotone (see
    `test_cross_section_prefactor_threshold_cancellation`).
    """
    threshold = 2.0 * muon_mass
    values = [
        cross_section_prefactor(muon_mass, muon_mass, threshold * (1.0 + eps))
        for eps in [1.0, 1e-1, 1e-2, 1e-3, 1e-4]
    ]
    # strict=False: pairing a list with its own tail is intentionally ragged.
    assert all(a < b for a, b in zip(values, values[1:], strict=False))


def test_cross_section_prefactor_threshold_cancellation() -> None:
    """Pin a known limitation so it cannot change unnoticed.

    `cross_section_prefactor` builds p from `kallen_lambda`, a sum of terms
    of size cme^4 that cancels to zero at threshold. At cme = m1 + m2 the
    residue is roundoff rather than zero, so the function returns a large
    finite number instead of diverging. Recorded in
    ``docs/followups/todo/cross-section-prefactor-threshold-cancellation.md``;
    if that follow-up lands, this test is what tells you to update it.
    """
    value = cross_section_prefactor(muon_mass, muon_mass, 2.0 * muon_mass)
    assert np.isfinite(value)
    # Far below what the true 1/(4 p cme) divergence would give.
    assert value < 1e6  # noqa: PLR2004 -- an order-of-magnitude sanity bound
