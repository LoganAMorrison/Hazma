"""``hazma._core.interp`` against ``numpy.interp``.

Twelve ``cdef`` functions in the compiled layer call ``np.interp`` on a
shipped table: the five rest-frame photon spectra
(``hazma/spectra/_photon/{_eta,_eta_prime,_kaon,_omega,_phi}.pyx``) and
the four mediator spectrum modules. cython-to-rust Task 3.4 reimplements
it in ``rust/src/interp.rs``, and this module is the gate on that.

Why the comparison has two modes
--------------------------------
Whether ``np.interp`` fuses ``slope * (x - xp[j]) + fp[j]`` is a property
of *the NumPy binary that happens to be installed*, not of this port. On
macOS/arm64 -- the platform the parity corpus was captured on, and whose
numbers this port targets -- the C compiler contracts it and the
comparison is bit-exact. On a target built without hardware FMA (baseline
x86-64, which is what the Linux wheels are built for) NumPy computes the
unfused values instead, and "does the Rust match the local NumPy
bit-for-bit" stops being a question about the port.

Until 2026-08-12 this module tried to *detect* that condition instead of
declaring it: ``numpy_contracts()`` compared ``np.interp`` against an
unfused transcription on interior points and skipped
:class:`TestAgainstNumpy` and :class:`TestFusedArithmetic` wherever the
two agreed. Task 4.1 showed the same mechanism to be unsound in
``test/test_core_positron_muon.py`` (PR #63, runs 31562223329 and
31564747071), and the 2026-08-12 rewrite of ``test/test_core_boost.py``
retired it there: a probe over one contraction mechanism cannot see the
others, so it answers "contracts" on platforms where nothing does and the
bit-equality assertions then fail, or answers "does not contract" and
silently voids the whole comparison. **This module was resolving the
second way.** Built for linux/amd64 the probe returned ``False``, and all
nine of the module's cross-implementation claims skipped -- seven
parametrised table comparisons, the random-grid comparison, and the fused
arithmetic check -- leaving ``hazma._core.interp`` checked against nothing
but its own clamping contract, quirks and error paths on every CI entry
but macOS.

So the *mode* is declared from the platform, and the divergence off it was
**measured rather than assumed** -- by building this worktree for
linux/amd64 (Debian bookworm, glibc 2.36, CPython 3.12.13, NumPy 2.5.1)
and comparing ``hazma._core.interp`` against that build's own
``np.interp`` directly, over exactly the sweeps :class:`TestAgainstNumpy`
runs:

==================== =================== ============= ================
comparison           points differing    max relative  max ``|Δ|``/peak
==================== =================== ============= ================
``eta``              1,571 / 20,304      5.9e-05       1.8e-17
``eta_prime``        543 / 21,504        8.2e-06       7.2e-17
``charged_kaon``     302 / 21,504        1.5e-05       1.4e-16
``long_kaon``        575 / 21,504        1.6e-06       1.0e-16
``short_kaon``       90 / 21,504         1.2e-05       1.5e-17
``omega``            479 / 21,504        1.6e-05       2.8e-19
``phi``              343 / 21,504        8.0e-06       3.3e-18
50 random grids      307,598 / 1,004,682 4.0e-02       2.2e-16
==================== =================== ============= ================

Over 1.15 million abscissae: no NaN, no infinity, and -- the statement
rounding cannot excuse -- **no disagreement anywhere about which
abscissae return exactly zero**.

:data:`OFF_PLATFORM_BUDGET` is scaled to the **peak of the compared
array** rather than applied pointwise, and the last two columns are why:
they differ by fourteen orders of magnitude. This module is the sharpest
case for peak scaling in the tree -- Task 3.4 rejected a tolerance over
exactly this comparison, because the worst *relative* gap lands at a
catastrophic cancellation point: the eta table's tail, where the
interpolant is ``2.4e-26`` against a table whose scale is ``0.2``, an
absolute gap of ``1.4e-30``. A pointwise ``rtol`` admitting the 4.0e-02
above would be no tolerance at all; against the peak -- what a downstream
integral or limit sees -- the whole population fits in 2.2e-16, which is
one ulp. Both arms are asserted (``atol = BUDGET * peak`` with
``rtol = BUDGET``), and 1e-12 is 4.6e3x the worst measured peak-relative
disagreement.

That budget is deliberately the *weaker* of this module's two
off-platform gates, and it can afford to be, because peak scaling is
blind in one direction: a defect confined to a value far below the peak
is small against the peak too. What covers that case is
:class:`TestFusedArithmetic`, whose reference is not a fused multiply-add
in isolation but a full transcription of ``rust/src/interp.rs`` --
bisection, clamps, exact-node short circuit and NumPy's two-step NaN
rescue included -- asserted **bit-for-bit on every platform**. A wrong
cell index, a wrong clamp or a dropped term fails there exactly,
wherever it runs. The budget's job is the narrower one of catching gross
divergence from the independent NumPy oracle.

The parity corpus (``test/parity/``) is scoped by platform the same way
(``.github/workflows/ci.yml`` passes ``--ignore=test/parity`` off the
capturing platform), and :data:`CAPTURE_MACHINE` is read out of its
manifest so the two scopes cannot drift apart.

The quirks stay exact everywhere
--------------------------------
:class:`TestClamping`, :class:`TestQuirks`, :class:`TestErrors` and
:class:`TestDispatch` are structural, not numerical -- NaN propagation,
the one-point grid's asymmetry, the duplicate-node tie-break, both
infinite-cell rescues, NumPy's two refusals. None of them reaches the
fused multiply-add, so none of them gets a budget: they are exact
assertions on every platform and always were.

Lifetime
--------
Nothing here parses Cython, so this module outlives the ``.pyx``. After
Phase 06 it remains the standing check that the Rust interpolation still
tracks NumPy -- a property the parity corpus cannot see, because the
corpus pins spectra rather than the routines underneath them.
:class:`TestFusedArithmetic` earns that independently: it compares the
port against a *fused Python reference* rather than against NumPy, so
"the port fuses here" is a claim it can assert on every platform, which
is what the old NumPy-versus-unfused form could only do on one.
"""

from __future__ import annotations

import json
import math
import platform
import sys
import zlib
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from hazma._core import interp as core_interp
from hazma.spectra._photon import _eta, _eta_prime, _kaon, _omega, _phi

interp = core_interp.interp

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The smallest grid the multi-point path handles; below it NumPy takes
#: its one-point branch, whose behavior differs (see :class:`TestQuirks`).
MULTI_POINT = 2


def photon_tables() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The live tables ``np.interp`` is called on, keyed by kernel."""
    return {
        "eta": (_eta.eta_data_energies, _eta.eta_data_dnde),
        "eta_prime": (
            _eta_prime.eta_prime_data_energies,
            _eta_prime.eta_prime_data_dnde,
        ),
        "charged_kaon": (
            _kaon.charged_kaon_data_energies,
            _kaon.charged_kaon_data_dnde,
        ),
        "long_kaon": (_kaon.long_kaon_data_energies, _kaon.long_kaon_data_dnde),
        "short_kaon": (_kaon.short_kaon_data_energies, _kaon.short_kaon_data_dnde),
        "omega": (_omega.omega_data_energies, _omega.omega_data_dnde),
        "phi": (_phi.phi_data_energies, _phi.phi_data_dnde),
    }


# --------------------------------------------------------------------------
# Which platform this is, and what the comparison costs off it
# --------------------------------------------------------------------------

#: The platform the parity corpus was captured on, read from its own
#: manifest so the two can never drift apart. ``test/parity`` is scoped to
#: this platform for exactly the reason in the module docstring, and CI
#: enforces it with ``--ignore=test/parity`` everywhere else
#: (``.github/workflows/ci.yml``); this module is the same kind of oracle
#: and carries the same scope.
CAPTURE_MACHINE = json.loads(
    (REPO_ROOT / "test" / "parity" / "data" / "manifest.json").read_text()
)["environment"]["machine"]

ON_THE_CAPTURING_PLATFORM = (
    sys.platform == "darwin" and platform.machine() == CAPTURE_MACHINE
)

#: The off-platform budget, as a fraction of the peak of the array being
#: compared. Set from the linux/amd64 measurement in the module docstring:
#: 4.6e3x the worst peak-relative disagreement observed there (2.2e-16, on
#: the random grids), which is itself one ulp. The pointwise relative
#: reading cannot set this figure -- it reaches 4.0e-02 at the
#: cancellation points -- which is why the peak scaling is load-bearing
#: rather than cosmetic. Applied as ``assert_allclose``'s ``atol`` scaled
#: by the peak, with ``rtol`` at the same figure so a large value is held
#: to the same standard.
OFF_PLATFORM_BUDGET = 1e-12

#: How far off a value has to be for the budget to reject it, as a
#: fraction of the peak. 1e4x :data:`OFF_PLATFORM_BUDGET`, and ~5e7x the
#: largest disagreement ever measured between two builds (2.2e-16 of the
#: peak), while still far too small to see in a plot.
BUDGET_PROBE_ERROR = 1e-8


def _as_array(values: object) -> np.ndarray:
    """A 1-D float64 view of a scalar or array, for the comparisons below."""
    return np.atleast_1d(np.asarray(values, dtype=np.float64))


def assert_within_the_off_platform_budget(
    got: object, want: object, context: str
) -> None:
    """Assert two results agree to :data:`OFF_PLATFORM_BUDGET` of the peak.

    Split out from :func:`assert_matches_numpy` so the budget can be
    exercised on *every* platform, including the one where the caller
    would otherwise take the bit-equality branch and leave this untested
    -- :meth:`TestOffPlatformBudget.test_the_budget_rejects_a_real_error`.

    ``atol`` is scaled by the peak rather than left at zero because the
    relative error is unbounded where the interpolant cancels: see "Why
    the comparison has two modes".
    """
    got, want = _as_array(got), _as_array(want)
    finite = np.isfinite(want)
    peak = float(np.abs(want[finite]).max()) if finite.any() else 0.0
    np.testing.assert_allclose(
        got,
        want,
        rtol=OFF_PLATFORM_BUDGET,
        atol=OFF_PLATFORM_BUDGET * peak,
        err_msg=(
            f"{context}: the port left np.interp's budget of "
            f"{OFF_PLATFORM_BUDGET:.0e} x the peak ({peak:.6e}). Rounding "
            f"between two builds was measured at 2.2e-16 x peak, so this is "
            f"a defect, not a platform difference."
        ),
    )


def assert_matches_numpy(got: object, want: object, context: str) -> None:
    """The oracle, in whichever of its two modes this platform gets.

    Bit-for-bit where the corpus was captured -- the port was written
    against *this* build's arithmetic and reproduces it exactly, which is
    a far stronger statement than any tolerance. A budget elsewhere,
    because off it the comparison measures NumPy's C compiler rather than
    the port.
    """
    if ON_THE_CAPTURING_PLATFORM:
        got_array, want_array = _as_array(got), _as_array(want)
        mismatched = int(np.count_nonzero(got_array != want_array))
        # `want` is exactly zero past the spectrum endpoint, so the
        # relative gap is reported against a floor rather than dividing
        # by it -- otherwise the diagnostic reads `nan` precisely when it
        # is needed.
        scale = np.where(want_array == 0.0, 1.0, np.abs(want_array))
        assert mismatched == 0, (
            f"{context}: {mismatched} of {want_array.size} points differ from "
            f"np.interp on the platform the corpus was captured on, where the "
            f"port reproduces it exactly; worst relative "
            f"{np.max(np.abs(got_array - want_array) / scale):.3e}"
        )
        return
    assert_within_the_off_platform_budget(got, want, context)


# --------------------------------------------------------------------------
# A reference implementation, parameterised by its multiply-add
# --------------------------------------------------------------------------


def fma(a: float, b: float, c: float) -> float:
    """``a * b + c`` with a single rounding, without ``math.fma``.

    ``math.fma`` arrived in Python 3.13 and this suite supports 3.10, so
    the fused product is computed exactly as a rational and rounded once
    by ``float()``, which rounds to nearest. Passed to
    :func:`interp_reference` to reproduce ``f64::mul_add``, which is
    correctly rounded on every target Rust supports whether or not the
    hardware has an FMA.

    Finite arguments only -- ``Fraction`` has no infinities. Every caller
    below evaluates on a finite grid; the non-finite paths are structural
    and are pinned exactly by :class:`TestQuirks` instead.
    """
    return float(Fraction(a) * Fraction(b) + Fraction(c))


def unfused(a: float, b: float, c: float) -> float:
    """``a * b + c`` with the product rounded before the sum."""
    return a * b + c


def nan_rescue(
    offset: float,
    left: float,
    right: float,
    slope: float,
    *,
    mul_add: Callable[[float, float, float], float],
) -> float:
    """NumPy's two-step recovery when the interpolation comes out NaN.

    ``rust/src/interp.rs:120-127``: re-anchor on the cell's right node,
    and if that is NaN too fall back to the node value when the cell is
    flat. Reachable only when a cell edge is infinite -- pinned by
    :meth:`TestQuirks.test_an_infinite_ordinate_falls_back_to_the_cells_other_end`
    and :meth:`TestQuirks.test_an_infinitely_wide_cell_is_rescued_only_when_flat`.

    Parameters
    ----------
    offset : float
        ``x - xp[j + 1]``, the abscissa relative to the cell's right node.
    left, right : float
        The cell's two ordinates, ``fp[j]`` and ``fp[j + 1]``.
    slope : float
        The cell's slope, as the caller computed it.
    mul_add : callable
        ``(a, b, c) -> a * b + c``, fused or not.
    """
    from_right = mul_add(slope, offset, right)
    if math.isnan(from_right) and left == right:
        return left
    return from_right


def interp_reference(
    x: float,
    xp: np.ndarray,
    fp: np.ndarray,
    *,
    mul_add: Callable[[float, float, float], float],
) -> float:
    """``rust/src/interp.rs`` with its one multiply-add injected.

    ``mul_add=fma`` is ``rust/src/interp.rs:83-130`` transcribed site for
    site -- the bisection, the clamps, the exact-node short circuit and
    NumPy's two-step NaN rescue included; ``mul_add=unfused`` is the same
    algorithm with the interpolation step written the obvious way.
    :class:`TestFusedArithmetic` asserts the port is the first and not the
    second, which pins the fusion without asking what any compiler did.

    Parameters
    ----------
    x : float
        The abscissa, in the grid's units.
    xp, fp : ndarray
        The ascending grid and its ordinates, same length.
    mul_add : callable
        ``(a, b, c) -> a * b + c``, fused or not.

    Returns
    -------
    float
        ``numpy.interp(x, xp, fp)`` in that arithmetic.
    """
    n = xp.size
    if n == 1:
        return float(fp[0])
    if math.isnan(x):
        return x
    if x < float(xp[0]) or x > float(xp[-1]):
        # Outside the grid NumPy clamps; it never extrapolates.
        return float(fp[0] if x < float(xp[0]) else fp[n - 1])

    # NumPy's bisection: ties resolve to the *last* matching index.
    imin, imax = 0, n
    while imin < imax:
        imid = imin + ((imax - imin) >> 1)
        if x >= float(xp[imid]):
            imin = imid + 1
        else:
            imax = imid
    j = imin - 1

    if j == n - 1 or float(xp[j]) == x:
        return float(fp[j])

    slope = (float(fp[j + 1]) - float(fp[j])) / (float(xp[j + 1]) - float(xp[j]))
    value = mul_add(slope, x - float(xp[j]), float(fp[j]))
    if math.isnan(value):
        value = nan_rescue(
            x - float(xp[j + 1]),
            float(fp[j]),
            float(fp[j + 1]),
            slope,
            mul_add=mul_add,
        )
    return value


# --------------------------------------------------------------------------
# Fixtures and constants
# --------------------------------------------------------------------------


def table_seed(name: str) -> int:
    """A stable seed for ``name``'s sweep.

    ``hash`` is randomised per process unless ``PYTHONHASHSEED`` is set,
    so seeding from it drew a different sweep on every run and put the
    budget measurement out of reach of reproduction. ``crc32`` is fixed.
    """
    return zlib.crc32(name.encode())


def spread(lo: float, hi: float, count: int, rng: np.random.Generator) -> np.ndarray:
    """``count`` draws in ``[lo, hi)``, identically on every platform.

    ``rng.uniform(lo, hi, count)`` is **not** platform-independent: NumPy
    computes ``lo + (hi - lo) * u`` in C, where the compiler is free to
    contract it, and on macOS/arm64 it does -- 6,532 of the 20,000 draws
    over the eta grid land on a different double than the same seed gives
    on Linux. Written as separate ufunc calls the contraction has nowhere
    to happen, and the underlying doubles ``rng.random`` produces are
    identical everywhere (verified by hashing both on macOS/arm64 and
    linux/amd64). That is what makes the counts this module records --
    ``1,571 of 20,304`` and the rest -- reproducible rather than
    per-platform anecdotes.
    """
    return lo + (hi - lo) * rng.random(count)


def sweep_abscissae(xp: np.ndarray, seed: int) -> np.ndarray:
    """Abscissae covering every branch of the interpolation.

    Random interior points find generic cells; the nodes themselves hit
    the exact-node short circuit; the nodes nudged by one part in 1e13
    land just inside the cells on either side of a node, which is where a
    wrong cell index shows up; and the four out-of-range points exercise
    the clamps.
    """
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [
            spread(float(xp[0]), float(xp[-1]), 20_000, rng),
            xp,
            xp * (1.0 + 1e-13),
            xp * (1.0 - 1e-13),
            [xp[0] - 1.0, xp[-1] + 1.0, xp[0], xp[-1]],
        ]
    )


#: The smallest miss the unfused arithmetic is recorded as producing on
#: the eta table; a smaller one means :func:`interp_reference` stopped
#: discriminating between the two arithmetics. One ulp, which is the
#: honest scale: away from the cancellation tail the unfused form misses
#: by a median 1.8e-16 relative and at most 7.3e-16 over the 467 points
#: above 1e-3 of the peak. The 5.9e-5 the tail itself reaches is a
#: property of *where the sweep happens to land*, so a floor pinned to it
#: would break the moment it landed elsewhere; the count below is the
#: robust half of this pair.
MIN_RECORDED_UNFUSED_MISS = 1e-16

#: The smallest number of points the two arithmetics are recorded as
#: disagreeing on over the eta sweep. Measured at 1,571 of 20,304 -- and
#: 755 of those sit above 1e-9 of the peak, so the population is not the
#: cancellation tail alone. A hundred leaves room for a NumPy or table
#: change without letting the test pass on a handful of coincidences.
MIN_UNFUSED_DISAGREEMENTS = 100


class TestAgainstNumpy:
    """Agreement with ``np.interp`` on every live table.

    Bit-for-bit where the corpus was captured, within
    :data:`OFF_PLATFORM_BUDGET` of the peak elsewhere -- see the module
    docstring. Every platform runs one of the two; none skips.
    """

    @pytest.mark.parametrize("name", list(photon_tables()))
    def test_matches_numpy_on_the_live_tables(self, name: str) -> None:
        xp, fp = photon_tables()[name]
        x = sweep_abscissae(xp, seed=table_seed(name))
        got = interp(x, xp, fp)
        want = np.interp(x, xp, fp)
        # Where the spectrum vanishes is structural: the tables carry
        # exact zeros past their endpoints and no rounding difference
        # explains a disagreement about which abscissae land there. So
        # this is checked first, exactly, on every platform.
        assert np.array_equal(got == 0.0, want == 0.0), (
            f"{name}: the port and np.interp disagree about which abscissae "
            f"return exactly zero"
        )
        assert_matches_numpy(got, want, name)

    def test_matches_numpy_on_a_random_grid(self) -> None:
        """A grid NumPy has never seen, with cells of wildly unequal width.

        The live tables are smooth and near-geometric; this one is not,
        so a cell-index or slope error that the tables happen to forgive
        has somewhere to show. The nodes span twelve decades and the
        ordinates sixteen, both drawn through :func:`spread` so the grids
        are the same on every platform.
        """
        rng = np.random.default_rng(20260810)
        for trial in range(50):
            n = int(rng.integers(2, 60))
            xp = np.sort(spread(-1e3, 1e3, n, rng) * 10.0 ** spread(-6, 6, n, rng))
            xp = np.unique(xp)
            if xp.size < MULTI_POINT:
                continue
            size = xp.size
            fp = spread(-4.0, 4.0, size, rng) * 10.0 ** spread(-8, 8, size, rng)
            x = sweep_abscissae(xp, seed=int(rng.integers(2**31)))
            assert_matches_numpy(
                interp(x, xp, fp), np.interp(x, xp, fp), f"random grid {trial}"
            )


class TestFusedArithmetic:
    """The interpolation step is fused, and it has to be.

    NumPy computes ``slope * (x - xp[j]) + fp[j]`` in C, where the default
    ``-ffp-contract=on`` lets the compiler emit a fused multiply-add --
    and on this project's reference platform (macOS/arm64) it does. Rust
    never contracts on its own, so ``rust/src/interp.rs`` spells the
    fusion out with ``mul_add``.

    :func:`interp_reference` is that kernel written twice over -- once
    with :func:`fma` at exactly the site ``rust/src/interp.rs`` spells
    ``mul_add``, and once the obvious way. Both are pure Python, so both
    are the same numbers on every platform, and so is the port:
    ``f64::mul_add`` is correctly rounded whether or not the target has an
    FMA instruction. That makes "the port fuses here and only here" a
    claim this class can assert **everywhere**.

    It could not before. Until 2026-08-12 the discriminator was NumPy,
    which fuses only where its compiler chose to, so the class was gated
    on a probe and skipped wherever the probe said no -- which off
    macOS/arm64 was always.
    """

    @pytest.mark.parametrize("name", list(photon_tables()))
    def test_the_port_is_the_fused_reference(self, name: str) -> None:
        """Bit-for-bit against ``mul_add=fma``, on every platform."""
        xp, fp = photon_tables()[name]
        x = sweep_abscissae(xp, seed=table_seed(name))
        got = interp(x, xp, fp)
        want = np.array([interp_reference(float(v), xp, fp, mul_add=fma) for v in x])
        assert np.array_equal(got, want), (
            f"{name}: the port is not the fused reference at "
            f"{int(np.count_nonzero(got != want))} of {x.size} points"
        )

    def test_the_unfused_form_would_be_a_different_number(self) -> None:
        """Otherwise the test above would pass against either arithmetic.

        The recorded figures are the point. On the eta sweep the two
        arithmetics part company at **1,571 of 20,304** points -- by a
        median 1.8e-16 relative, one ulp, rising to 5.9e-5 where the
        interpolant cancels -- and the unfused branch is not a hypothesis:
        built for linux/amd64, ``np.interp`` *is* ``mul_add=unfused``
        bit-for-bit on all seven live tables, exactly as it is
        ``mul_add=fma`` bit-for-bit on macOS/arm64 (module docstring).
        """
        xp, fp = photon_tables()["eta"]
        x = sweep_abscissae(xp, seed=table_seed("eta"))

        differ = 0
        worst = 0.0
        for value in x:
            want = interp_reference(float(value), xp, fp, mul_add=fma)
            unfused_value = interp_reference(float(value), xp, fp, mul_add=unfused)
            if want not in (unfused_value, 0.0):
                differ += 1
                worst = max(worst, abs(unfused_value - want) / abs(want))
        assert differ > MIN_UNFUSED_DISAGREEMENTS, (
            f"the two arithmetics differ at only {differ} of {x.size} points on "
            f"the eta table, so this test barely distinguishes them"
        )
        assert (
            worst > MIN_RECORDED_UNFUSED_MISS
        ), f"unfused worst miss {worst:.3e} is smaller than recorded"


class TestOffPlatformBudget:
    """:data:`OFF_PLATFORM_BUDGET` is not vacuous.

    Asserted where the budget is *not* used: on the capturing platform
    every comparison above takes its exact branch, so nothing else would
    exercise the tolerance and it could rot to ``inf`` unnoticed. That is
    the failure mode Task 4.1 recorded -- "the capturing platform cannot
    see a bug in its own skip logic" -- one level down.
    """

    def test_the_budget_rejects_a_real_error(self) -> None:
        """A perturbation of :data:`BUDGET_PROBE_ERROR` of the peak must fail.

        Called through the budget predicate rather than through
        :func:`assert_matches_numpy`, so it exercises the tolerance on
        every platform -- going through the dispatcher would take the
        exact branch here and prove nothing about the budget, which is
        the whole point.
        """
        xp, fp = photon_tables()["eta"]
        x = sweep_abscissae(xp, seed=table_seed("eta"))
        want = np.interp(x, xp, fp)
        nudged = want.copy()
        nudged[nudged.argmax()] += BUDGET_PROBE_ERROR * want.max()

        assert_within_the_off_platform_budget(want, want, "unperturbed")
        with pytest.raises(AssertionError):
            assert_within_the_off_platform_budget(nudged, want, "perturbed")

    def test_this_platform_gets_the_mode_it_is_supposed_to(self) -> None:
        """One ulp: rejected where the corpus was captured, tolerated off it.

        The guard on the *strict* branch, and on the dispatch into it,
        which the test above does not cover. The failure mode is silent in
        one direction: an :data:`ON_THE_CAPTURING_PLATFORM` that had
        rotted to ``False`` would route every comparison in this module
        through the budget and every one of them would still pass. So the
        expected mode is re-derived here from ``sys``/``platform`` rather
        than read back out of the module -- reading it back would agree
        with the dispatcher by construction and assert nothing.
        """
        expected_strict = (
            sys.platform == "darwin" and platform.machine() == CAPTURE_MACHINE
        )
        value = 1.0
        nudged = float(np.nextafter(value, np.inf))
        assert_matches_numpy(value, value, "identical")
        if expected_strict:
            with pytest.raises(AssertionError):
                assert_matches_numpy(nudged, value, "one ulp")
        else:
            assert_matches_numpy(nudged, value, "one ulp")


#: A three-node toy grid with cells of unequal width, shared by the
#: contract tests below. The names let the assertions say *which* node
#: they expect rather than repeating its value.
TOY_XP = np.array([1.0, 2.0, 4.0])
TOY_FP = np.array([10.0, 20.0, -5.0])
TOY_FIRST, TOY_MIDDLE, TOY_LAST = (float(value) for value in TOY_FP)
#: Midpoints of the two cells: (10 + 20)/2 and (20 - 5)/2.
TOY_FIRST_CELL_MID = 15.0
TOY_SECOND_CELL_MID = 7.5


class TestClamping:
    """Outside the grid ``np.interp`` clamps; it never extrapolates."""

    XP = TOY_XP
    FP = TOY_FP

    @pytest.mark.parametrize("x", [0.0, -1e300, -np.inf, 0.999_999])
    def test_below_the_grid_returns_the_first_value(self, x: float) -> None:
        assert interp(x, self.XP, self.FP) == TOY_FIRST

    @pytest.mark.parametrize("x", [5.0, 1e300, np.inf, 4.000_001])
    def test_above_the_grid_returns_the_last_value(self, x: float) -> None:
        assert interp(x, self.XP, self.FP) == TOY_LAST

    def test_nodes_return_their_own_values(self) -> None:
        assert np.array_equal(interp(self.XP, self.XP, self.FP), self.FP)

    def test_the_midpoint_of_a_cell_is_the_mean_of_its_ends(self) -> None:
        assert interp(1.5, self.XP, self.FP) == TOY_FIRST_CELL_MID
        assert interp(3.0, self.XP, self.FP) == TOY_SECOND_CELL_MID


#: Values the quirk tests assert on by name.
ONE_POINT_VALUE = 7.0
LAST_DUPLICATE = 9.0
INFINITE_NODE_VALUE = 1.0
FLAT_CELL_VALUE = 3.0


class TestQuirks:
    """Behaviors that are NumPy's rather than linear interpolation's.

    Each is reproduced deliberately and checked against NumPy in the same
    assertion, so the pin cannot drift away from the thing it pins. None
    of these reaches the fused multiply-add, so all of them are exact on
    every platform -- they carry no budget and never did.
    """

    def test_nan_propagates_on_a_multi_point_grid(self) -> None:
        xp, fp = np.array([1.0, 2.0]), np.array([ONE_POINT_VALUE, 8.0])
        assert np.isnan(interp(np.nan, xp, fp))
        assert np.isnan(np.interp(np.nan, xp, fp))

    def test_a_one_point_grid_answers_everything_with_its_one_value(self) -> None:
        """NumPy's one-point branch runs before its NaN check.

        So a NaN abscissa returns ``fp[0]`` there while it returns NaN on
        any longer grid -- an asymmetry with no principle behind it,
        carried because the corpus is pinned to what NumPy does.
        """
        xp, fp = np.array([2.0]), np.array([ONE_POINT_VALUE])
        for x in (np.nan, -1.0, 2.0, 5.0):
            assert interp(x, xp, fp) == ONE_POINT_VALUE
            assert np.interp(x, xp, fp) == ONE_POINT_VALUE

    def test_duplicate_nodes_resolve_to_the_last_copy(self) -> None:
        xp = np.array([0.0, 1.0, 1.0, 2.0])
        fp = np.array([0.0, 5.0, LAST_DUPLICATE, LAST_DUPLICATE])
        assert interp(1.0, xp, fp) == np.interp(1.0, xp, fp) == LAST_DUPLICATE

    def test_an_infinite_ordinate_falls_back_to_the_cells_other_end(self) -> None:
        xp, fp = np.array([0.0, 1.0]), np.array([np.inf, 0.0])
        assert interp(0.5, xp, fp) == np.interp(0.5, xp, fp) == np.inf

    def test_an_infinite_node_returns_its_own_value(self) -> None:
        """The exact-node short circuit, in the only place it is visible.

        At an ordinary node the interpolation gives ``slope * 0 + fp[j]``
        = ``fp[j]`` anyway, so the guard NumPy carries to "avoid potential
        non-finite interpolation" is unobservable — until the cell is
        infinitely wide. Here ``slope`` is 0 and ``x - xp[j]`` is
        ``-inf - -inf`` = NaN, so the product is NaN and both NaN rescues
        fail; only the short circuit returns a number.
        """
        xp = np.array([-np.inf, 0.0, 1.0])
        fp = np.array([INFINITE_NODE_VALUE, 2.0, 3.0])
        assert interp(-np.inf, xp, fp) == INFINITE_NODE_VALUE
        assert np.interp(-np.inf, xp, fp) == INFINITE_NODE_VALUE

    def test_an_infinitely_wide_cell_is_rescued_only_when_flat(self) -> None:
        xp = np.array([-np.inf, np.inf])
        flat = np.array([FLAT_CELL_VALUE, FLAT_CELL_VALUE])
        sloped = np.array([FLAT_CELL_VALUE, FLAT_CELL_VALUE + 1.0])
        assert interp(0.0, xp, flat) == FLAT_CELL_VALUE
        assert np.interp(0.0, xp, flat) == FLAT_CELL_VALUE
        assert np.isnan(interp(0.0, xp, sloped))
        assert np.isnan(np.interp(0.0, xp, sloped))


class TestErrors:
    """The two grids NumPy refuses, refused with NumPy's own wording."""

    def test_an_empty_grid_raises(self) -> None:
        empty = np.array([], dtype=np.float64)
        with pytest.raises(ValueError, match="array of sample points is empty"):
            interp(1.0, empty, empty)
        with pytest.raises(ValueError, match="array of sample points is empty"):
            np.interp(1.0, empty, empty)

    def test_mismatched_lengths_raise(self) -> None:
        xp = np.array([1.0, 2.0])
        fp = np.array([1.0])
        with pytest.raises(ValueError, match="not of the same length"):
            interp(1.0, xp, fp)
        with pytest.raises(ValueError, match="not of the same length"):
            np.interp(1.0, xp, fp)


class TestDispatch:
    """The abscissa follows the contract every ported entry point uses.

    The full branch-by-branch pinning lives in
    ``test/test_core_dispatch.py``; this only checks that ``interp`` is
    wired into it rather than re-deriving it.
    """

    XP = TOY_XP
    FP = TOY_FP

    def test_scalar_in_float_out(self) -> None:
        got = interp(1.5, self.XP, self.FP)
        assert isinstance(got, float)
        assert got == TOY_FIRST_CELL_MID

    def test_array_in_fresh_array_out(self) -> None:
        x = np.array([1.5, 3.0])
        got = interp(x, self.XP, self.FP)
        assert isinstance(got, np.ndarray)
        assert got.dtype == np.float64
        assert np.array_equal(got, [TOY_FIRST_CELL_MID, TOY_SECOND_CELL_MID])
        assert not np.shares_memory(got, x)

    def test_array_path_equals_the_scalar_path(self) -> None:
        x = np.linspace(0.5, 4.5, 101)
        assert np.array_equal(
            interp(x, self.XP, self.FP),
            [interp(float(v), self.XP, self.FP) for v in x],
        )

    def test_empty_array_round_trips(self) -> None:
        got = interp(np.array([]), self.XP, self.FP)
        assert isinstance(got, np.ndarray)
        assert got.size == 0

    def test_two_dimensional_abscissae_raise(self) -> None:
        with pytest.raises(ValueError, match="0 or 1-dimensional"):
            interp(np.zeros((2, 2)), self.XP, self.FP)

    def test_non_float64_abscissae_raise(self) -> None:
        with pytest.raises(ValueError, match="float64 array"):
            interp(np.array([1, 2], dtype=np.int64), self.XP, self.FP)
