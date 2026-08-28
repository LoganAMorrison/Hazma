"""Per-entry-point tolerance budgets for the golden parity corpus.

The corpus (`cases`, `generate`) pins what the pre-port Cython returned.
This module declares how far a *replacement* implementation is allowed to
move each of those numbers, and `test_parity` enforces it.

A budget is a contract, not a prediction. It says "a port that lands here
is accepted without further argument"; it does not say the port will land
here. Widening one is therefore a real decision:
``projects/cython-to-rust/rules.md`` rule 2 requires a one-line
justification in this file plus a note in the widening task's task note,
and rule 3 requires any shift beyond 1e-12 relative to be declared in the
PR body and the project's "Numerical impact so far" log even when the
budget absorbs it.

Budget classes
--------------
Every entry point falls into one of five classes, set by *what the
implementation does* rather than by what it computes:

``EXACT`` (rtol 0)
    Closed-form arithmetic: `+ - * /`, `sqrt`, `pow`, `log`, `exp`,
    `atan`, `atanh` and branch selection, nothing else. The phase file
    asks for bit-equality here against the capturing commit. Note this is
    strict enough to catch a libm change — Rust's `f64::exp` need not
    agree with the platform C library in the last ulp — which is
    deliberate: such a shift should be measured and declared, not
    absorbed silently.

    This sentence used to end "everything in this class reaches only
    `libc.math`", and Task 5.1 found that two members do not.
    `sigma_xx_to_v_to_pipi` and `sigma_xx_to_v_to_pi0v` raise a double to
    the power `1.5`, and Cython 3's default `cpow` semantics compile the
    whole enclosing expression in `double _Complex` — so the Cython
    reaches `cpow` and compiler-rt's `__divdc3`, neither of which agrees
    with its real-arithmetic spelling (up to 9.0e-15 and 4.0e-16
    relative respectively, measured over 3.7M arguments). The class still
    holds, and at `rtol = 0`: the port reproduces both routines rather
    than approximating them (`rust/src/kernels/vector_xs.rs`), and all
    five closed-form vector kernels came back bit-equal at every pinned
    value. What the correction changes is the *reason* — a future member
    of this class has to be checked against the generated C, not against
    the `.pyx`.
``SPECFUN`` (rtol 1e-13)
    Closed form apart from a special function. ADR-0002 replaces scipy's
    cephes bindings with cephes-lineage Rust, which is
    algorithm-for-algorithm parity rather than merely value parity; 1e-13
    relative is the acceptance figure the numerics reference already sets
    for that swap
    (`projects/cython-to-rust/references/numerics-replacements.md`,
    "Special functions", Task 3.2 check 2). Task 4.3 swapped the class's
    only member and found the budget was *not* enough: the kernel forms
    `(5/beta) * (spence(xm) - spence(xp))` and this corpus samples
    `beta = 1.4e-6`, so a two-ulp difference in `spence` arrives as a
    3.2e-11 relative shift -- 320x this budget. The answer was to make
    `spence` bit-equal to scipy rather than to widen the budget
    (`rust/src/special.rs`), after which the port reproduced the Cython
    exactly at 144,000 sampled points and the budget went unused. It is
    kept rather than tightened for the same reason ``TABULATED`` is: it is
    the right contract where scipy's cephes is compiled without FP
    contraction, which is where it will be needed.
``TABULATED`` (rtol 1e-12)
    Driven by `boost_integrate_linear_interp` over a shipped CSV table:
    `np.trapezoid` across interior cells, closed-form partial cells at
    both edges, an analytic `1/E` tail below the table. A Rust rewrite
    keeps the algorithm and changes the summation order, so the drift is
    accumulated rounding, not method error: the tables are 100 data rows
    (eta) or 500 (the other six), and 500 · 2^-52 is ~1.1e-13, so 1e-12
    leaves roughly a decade of headroom over the worst case. (The row
    counts read 101/501 here until Task 4.2; those were the CSVs' line
    counts, which include a `#` header `numpy.loadtxt` skips. The Rust
    rewrite landed in that task and in the event reproduced the capturing
    platform bit-for-bit, by also reproducing NumPy's summation order --
    so the headroom went unused there, and stays declared for the
    platforms where it will not.)
``QUAD`` (rtol 1e-8, or `PORTED_QUAD_RTOL` = 1e-12 once measured)
    One adaptive `scipy.integrate.quad` call. The numerics reference
    expects a faithful netlib-QUADPACK port to reproduce these to ~1e-12
    or better; the phase file sets the opening budget at 1e-8 and tightens
    it once Phase 03 has measured the port. Phase 03 Task 3.3 measured it
    over 11,274 random (integrand, tolerance, limit, points) combinations:
    the 4,461 that converged agreed with scipy to **8.2e-11** relative at
    worst, and the rest can separate without bound because Wynn's
    epsilon-algorithm is chaotic on a non-converging sequence.

    The tightening is applied **per case as each is ported and measured**
    rather than class-wide, because 8.2e-11 is the envelope over arbitrary
    integrands and each live shape lands far inside it: Task 4.4 measured
    `spectra.photon.charged_pion` at **2.6e-15** relative over its 1,500
    pinned values (317 of them not bit-equal), i.e. a dozen ulp, which is
    accumulated last-bit arithmetic rather than method error. Task 4.6
    then measured the last two spectra members,
    `spectra.positron.charged_pion` at **5.5e-15** over 1,460 values and
    `spectra.neutrino.charged_pion` at **9.7e-16** over 4,185. So all
    three ported members take `PORTED_QUAD_RTOL`. Task 5.1 then measured
    the fourth, `cross_sections.vector.thermal_cross_section`, at
    **2.1e-14** relative over its 285 pinned values (64 bit-equal) and
    tightened it the same way; the drift there is the Bessel prefactor
    and weight rather than the integrator, since `bessel_kn(2, ·)` agrees
    with scipy to 8.9e-16 and the prefactor squares it. Task 5.2 then
    measured the fifth and last, `cross_sections.scalar.
    thermal_cross_section`, at **3.1e-15** (104 of 285 bit-equal) and
    tightened it too. That leaves the class with **no case at the
    opening figure**: `QUAD_RTOL` is now the documented starting point
    for the next unported member rather than a live budget, and Phase 06
    is where the next one arrives. Nothing external moves under a ported
    case any
    more -- the reference values are stored, scipy no longer participates,
    and the remaining variation is the platform libm, which the corpus is
    scoped to anyway.
``NESTED`` (rtol 1e-6, or `PORTED_NESTED_RTOL` = 1e-9 once measured)
    An adaptive quadrature whose integrand is itself an adaptive
    quadrature. Subdivision is a discontinuous function of the integrand:
    a last-ulp change inside can move an outer bisection decision, and the
    outer call is then only as accurate as its own `epsrel`, which is
    1e-5 at every live site. 1e-6 sits one decade inside that, so the
    opening budget tracks the integrator's own accuracy claim rather than
    the resolution of the arithmetic.

    Task 4.5 ported the class's first two members -- the project's
    declared numerical stress test -- and measured that the fear was
    priced too high. Over the 1,395 values the corpus pins for each,
    `spectra.photon.charged_rho` moved by at most **1.5e-13** relative and
    `spectra.photon.neutral_rho` by **3.2e-15**, with three quarters of
    the points bit-equal; a denser off-corpus sweep (3,200 points,
    parent energies the corpus does not sample) reached **2.5e-11**, at a
    photon energy whose boost window straddles the pi0 box's upper edge,
    where a jump discontinuity sits inside the interval and one bisection
    decision can flip. Even there the difference is five decades below the
    `abserr` the integrator itself reports. So the two rho cases took
    `PORTED_NESTED_RTOL` first, and Phase 06 measured the seven
    mediator-spectrum cases and moved every one of them onto it too --
    the three photon cases in Task 6.2 (worst 5.33e-12) and the four
    positron cases in Task 6.3 (worst 2.33e-12). No case holds the
    opening figure any more; the worst drift anywhere in the class is
    still two and a half decades inside it.

Abscissae are their own class
-----------------------------
The swept grid and the scalar probe say *where* an entry point was
sampled, not what it returned, so they get `ABSCISSA_RTOL` rather than
the case's value budget. Task 1.2 compared them bit-exactly in both
modes, on the stated premise that "grids are arithmetic on constants".
That premise is false across platforms: `cases` builds every grid with
`numpy.geomspace`, which evaluates `10 ** linspace(log10(lo), log10(hi))`
— two transcendental calls into the platform libm. Task 1.3's first CI
run measured the consequence on Linux/glibc against the macOS/arm64
corpus: **all 623 blocks** differed, every one of them by at most
`2.219e-16` — one ulp — and not a single value comparison was reached
because the grid assertion fires first.

1e-13 is derived, not chosen to make that run pass. `geomspace` carries
≤1 ulp from `log10`, ≤1 ulp from the `linspace` arithmetic, and ≤1 ulp
from the final power. The exponent `x` spans about ±3.5 for these grids,
so the absolute error in `x` is ~2·eps·3.5 ≈ 1.6e-15, amplified into the
result by `d(10**x)/10**x = ln(10)·dx` ≈ 3.6e-15, plus the power's own
ulp. Worst case ~4e-15; 1e-13 leaves ~25x headroom over that and is
still five decades tighter than the loosest value budget.

What it must not absorb is a *moved measurement point* — Task 1.2's
actual concern, which stands. Changing a grid endpoint, its point count,
or its spacing law moves abscissae by 1e-3 relative at the very least and
usually by O(1), ten orders of magnitude outside this budget. On the
capturing tree the comparison stays bit-exact, same as the values.

`atol` is 0.0 everywhere
------------------------
An absolute floor is scale-dependent — spectra run to ~1e-3 MeV^-1 and
cross sections to ~1e-20 MeV^-2 — so one floor either does nothing or
hides a real shift in the smaller quantity. It is also unnecessary: the
below-threshold and above-endpoint regions return exactly ``0.0``, and
``|0 - 0| <= rtol * 0`` holds. A port that returns 1e-300 where the
Cython returned zero therefore fails, which is the intended answer.

Citations
---------
Every ``file:line`` below was read against the tree whose kernel digest is
``f5e6e269be47`` — the digest the corpus manifest records. Phases 04-06
delete the files they point into; the line numbers are evidence for the
classification, not live references.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cases as corpus  # (imported after the sys.path entry above)
import generate as corpus_generate

#: Bit-equality against the capturing commit, in the sense
#: `numpy.testing.assert_allclose` gives it: NaN matches NaN, infinities
#: match in position and sign, and every other value must be equal as an
#: IEEE double. Signed zeros and NaN payloads are not distinguished --
#: byte-level identity of the *stored* corpus is what
#: `generate.py --check` covers.
EXACT_RTOL = 0.0
SPECFUN_RTOL = 1e-13
TABULATED_RTOL = 1e-12
#: The opening `QUAD` budget, held by no case since cython-to-rust
#: Task 5.2 measured and tightened the last of the five. It stays as the
#: figure a newly ported quadrature-backed case starts at, before its own
#: drift is measured.
QUAD_RTOL = 1e-8
#: The `QUAD` budget after a case has been ported and its drift measured.
#: See the "Budget classes" section: 380x headroom over Task 4.4's
#: measured 2.6e-15, and still four decades inside the opening figure.
PORTED_QUAD_RTOL = 1e-12
#: The opening `NESTED` budget, held by no case since cython-to-rust
#: Task 6.3 measured and tightened the last four -- the mediator positron
#: pair. Like `QUAD_RTOL` above it stays as the figure a newly ported
#: case in this class starts at.
NESTED_RTOL = 1e-6
#: The `NESTED` budget after a case has been ported and its drift
#: measured. See the "Budget classes" section: 6,600x headroom over
#: Task 4.5's worst pinned drift (1.5e-13) and 40x over the worst the same
#: task found anywhere off-corpus (2.5e-11), while staying three decades
#: inside the opening figure.
PORTED_NESTED_RTOL = 1e-9

#: What the ``EXACT`` class means once the **libm** changes. The class is
#: bit-equality, and bit-equality against a different platform's `atan`,
#: `log` and `exp` is unreachable: glibc and macOS libm are each
#: correctly rounded to within 1-2 ulp of the true result but not to each
#: other. The corpus's own sampling then sets the size. Its grids anchor
#: at ``1 +- 1e-9`` times each threshold, and a closed-form cross section
#: carries a ``sqrt(1 - 4 mx**2 / s)``-shaped factor whose relative error
#: at a fractional distance ``d`` from threshold is ``eps / (2 d)``:
#: ``2.2e-16 / 2e-9 = 1.1e-7``. So the sampling the phase file asks for
#: implies a ~1e-7 floor, whatever the arithmetic does. 1e-6 sits one
#: decade above that.
#:
#: Measured against it (`docs/followups/`... the parity-corpus follow-up,
#: 2026-08-18): over the whole corpus on Linux/x86_64 and Linux/aarch64,
#: with the `stability` mask applied, the worst any ``EXACT``-class block
#: moves is 5.6e-8 -- `sigma_xx_to_ss[closed_resonance]` at exactly the
#: ``2 mx`` anchor, i.e. the mechanism above, landing where the
#: derivation says it should.
#:
#: This applies **only** when `Provenance.same_platform` is false. A port
#: on the capturing platform is still held to `EXACT_RTOL`: Tasks 4.1-4.5
#: each reproduced their kernel bit-for-bit there, so relaxing it would
#: give up a gate that is demonstrably achievable.
PLATFORM_EXACT_RTOL = 1e-6

#: The same idea for the ``SPECFUN`` class, whose one member is
#: `spectra.photon.muon`. Task 4.3 made `spence` bit-equal **to scipy**
#: rather than widening the 1e-13 budget, and that is what holds the class
#: at 1e-13 -- on the capturing libm. Off it the `log` and `sqrt` around
#: the `spence` calls move in their last bit instead, and the class
#: docstring's own mechanism amplifies them: the kernel forms
#: ``(5/beta) * (spence(xm) - spence(xp))`` and the corpus samples
#: ``beta = 1.4e-6`` at the `rest_plus_eps` anchor, so a two-ulp
#: difference arrives as **3.2e-11** relative (measured in Task 4.3).
#:
#: 1e-9 is ~30x that documented worst case, and three decades tighter
#: than both `PLATFORM_EXACT_RTOL` and the ``QUAD`` opening budget.
#:
#: Measured, and the reason this constant exists: PR #71's first CI run
#: with the corpus enabled on every matrix entry failed exactly one
#: assertion -- `spectra.photon.muon[rest_plus_eps].scalar_values` at
#: **1.85e-13**, 1.85x the declared budget, on ubuntu-latest/py3.12 and on
#: no other entry. Three local platforms (macOS/arm64, Linux/aarch64,
#: Linux/x86_64 under Rosetta's SSE2 libm) had all passed, which is the
#: same "which points visibly break depends on the libm code path" that
#: `stability` documents, one class further out.
PLATFORM_SPECFUN_RTOL = 1e-9

#: How far the *abscissae* may move off the capturing tree. Not a value
#: budget: see "Abscissae are their own class" above for the derivation
#: and for why bit-equality here was unreachable on any second platform.
ABSCISSA_RTOL = 1e-13

#: Absolute floor allowed at the positions `stability.PORTABILITY_ZEROS`
#: declares, as a fraction of the median non-zero magnitude in the same
#: array. **Only** at those positions: every other stored ``0.0`` keeps
#: the exact-zero contract the section below argues for.
#:
#: "`atol` is 0.0 everywhere" below argues that a below-threshold region
#: returns exactly zero, so ``|0 - 0| <= rtol * 0`` holds and no floor is
#: needed. That is true nearly everywhere, and false at exactly one kind
#: of point: a quadrature whose integrand sits at *its* threshold.
#: `spectra.positron.charged_pion` integrates the positron-muon spectrum
#: over ``cos(theta)``, and at ``E = m_e`` whether QUADPACK's weighted sum
#: lands on ``0.0`` or on 2.6e-13 is a property of the libm underneath.
#: Four positions do the latter on Linux/aarch64; with `atol` at zero
#: that is an *infinite* relative error and the case's 1e-8 budget never
#: gets a chance to speak.
#:
#: The **scope** of that exemption is the whole design question, and the
#: first version of this fix got it wrong: it floored every exact zero in
#: every non-``EXACT`` array — 66,840 positions across 605 arrays — so
#: `spectra.photon.long_kaon[rest_plus_eps]` would have accepted 1.69e-07
#: where the Cython returns exactly zero. `stability.PORTABILITY_ZEROS`
#: now names the four, and the other 66,836 are back to exact
#: (PR #71 review round 1).
#:
#: The **size** is a fraction of the array's own median non-zero
#: magnitude rather than of its maximum, because a block can span nine
#: decades — `spectra.photon.long_kaon[rest_plus_eps]` peaks at 6.6e5 near
#: its endpoint, and a fraction of *that* is a floor set by the spike. On
#: the four declared positions the floor lands between 8.8e-13 and 4.4e-12
#: against residues of 2.7e-14 to 2.6e-13: 6x to 80x headroom, derived
#: from each block's own scale rather than chosen.
ZERO_FLOOR_FRACTION = 1e-9


@dataclass(frozen=True)
class Budget:
    """How far one entry point's values may move, and why that far.

    Parameters
    ----------
    rtol, atol : float
        Passed straight to `numpy.testing.assert_allclose`.
    why : str
        One-line justification. Required: a tolerance without a stated
        reason is one nobody can argue against later.
    """

    rtol: float
    atol: float
    why: str


@dataclass(frozen=True)
class Provenance:
    """Whether the live tree is the one the corpus was captured from.

    Parameters
    ----------
    exact : bool
        True when the kernels, the toolchain and the numerics libraries
        all match what the manifest records.
    same_platform : bool
        True when this host's `_libm_identity` -- OS family and CPU
        architecture -- matches the manifest's. Tracked apart from
        `exact` because it is the one
        axis that moves a closed-form kernel without anybody having
        changed an implementation: a different libm rounds `atan`, `log`
        and `exp` differently in the last bit, which is a *fact about the
        host* rather than a drift to declare under
        ``projects/cython-to-rust/rules.md`` rule 3. `effective_budget`
        is the only reader.
    detail : str
        Empty when `exact`; otherwise a human-readable list of what
        differs. Surfaced as the skip reason on
        `test_parity.test_running_on_the_capturing_tree`, so the mode is
        never something a reader has to infer.
    """

    exact: bool
    same_platform: bool
    detail: str


# ===========================================================================
# ---- The budget table -----------------------------------------------------
# ===========================================================================

#: Case name -> budget. Keys are `cases.build_cases()` names, which is
#: also what the manifest is keyed by; `test_parity` asserts the two sets
#: agree, so a corpus case can never quietly run without a declared budget.
BUDGETS: dict[str, Budget] = {
    # -- spectra: photon ----------------------------------------------------
    "spectra.photon.muon": Budget(
        rtol=SPECFUN_RTOL,
        atol=0.0,
        why="closed form apart from two `spence` calls "
        "(hazma/spectra/_photon/_muon.pyx:113); the cephes-lineage "
        "replacement is held to <=1e-13 relative by the numerics "
        "reference's own Task 3.2 acceptance check.",
    ),
    "spectra.photon.charged_pion": Budget(
        rtol=PORTED_QUAD_RTOL,
        atol=0.0,
        why="one QAGP over cos(theta) "
        "(hazma/spectra/_photon/_pion.pyx:123, epsabs=1e-10, epsrel=1e-5); "
        "the integrand is the spence-bearing muon spectrum, whose 1e-13 "
        "class sits well inside the quadrature budget. Tightened from "
        "QUAD_RTOL by Task 4.4 on its own measurement -- 2.6e-15 worst "
        "relative over the 1,500 pinned values, against 1e-12 here.",
    ),
    "spectra.photon.neutral_pion": Budget(
        rtol=EXACT_RTOL,
        atol=0.0,
        why="closed-form boosted box spectrum, arithmetic and branch "
        "selection only (hazma/spectra/_photon/_pion.pyx:147-171); the "
        "`cdef float` beta and return value are part of the arithmetic.",
    ),
    "spectra.photon.charged_rho": Budget(
        rtol=PORTED_NESTED_RTOL,
        atol=0.0,
        why="quad at hazma/spectra/_photon/_rho.pyx:123 whose integrand "
        "calls the charged- and neutral-pion point kernels "
        "(hazma/spectra/_photon/_rho.pyx:10-11 and :95-96), the first of "
        "which quads again — the nested case the phase file singles out. "
        "Tightened from NESTED_RTOL by Task 4.5 on a measured worst 1.5e-13 "
        "over these 1,395 pinned values.",
    ),
    "spectra.photon.neutral_rho": Budget(
        rtol=PORTED_NESTED_RTOL,
        atol=0.0,
        why="quad at hazma/spectra/_photon/_rho.pyx:52 over an integrand "
        "that calls the quad-backed charged-pion point kernel "
        "(hazma/spectra/_photon/_rho.pyx:26) — the second nested-rho "
        "entry point. Tightened from NESTED_RTOL by Task 4.5 on a measured "
        "worst 3.2e-15 over these 1,395 pinned values.",
    ),
    "spectra.photon.charged_kaon": Budget(
        rtol=TABULATED_RTOL,
        atol=0.0,
        why="boost integral over the shipped CSV table "
        "(hazma/spectra/_photon/_kaon.pyx:57-58); drift is trapezoid "
        "summation order, not method. Ported to Rust in Task 4.2, which "
        "reproduces this platform bit-for-bit -- the class is kept rather "
        "than tightened to EXACT because, unlike spectra.positron.muon, "
        "the boost integral's summation order is a NumPy implementation "
        "detail rather than an arithmetic identity.",
    ),
    "spectra.photon.long_kaon": Budget(
        rtol=TABULATED_RTOL,
        atol=0.0,
        why="boost integral over long_kaon_photon.csv, same mechanism as "
        "the charged kaon (hazma/spectra/_photon/_kaon.pyx:57-58).",
    ),
    "spectra.photon.short_kaon": Budget(
        rtol=TABULATED_RTOL,
        atol=0.0,
        why="boost integral over short_kaon_photon.csv, same mechanism as "
        "the charged kaon (hazma/spectra/_photon/_kaon.pyx:57-58).",
    ),
    "spectra.photon.eta": Budget(
        rtol=TABULATED_RTOL,
        atol=0.0,
        why="boost integral over eta_photon.csv "
        "(hazma/spectra/_photon/_eta.pyx:98), no quadrature on the live "
        "path. Ported to Rust in Task 4.2 alongside the kaons; see that "
        "entry for why the class is kept rather than tightened.",
    ),
    "spectra.photon.eta_prime": Budget(
        rtol=TABULATED_RTOL,
        atol=0.0,
        why="boost integral over eta_prime_photon.csv, same mechanism as "
        "the eta (hazma/spectra/_photon/_eta_prime.pyx).",
    ),
    "spectra.photon.omega": Budget(
        rtol=TABULATED_RTOL,
        atol=0.0,
        why="boost integral over omega_photon.csv, same mechanism as the "
        "eta (hazma/spectra/_photon/_omega.pyx).",
    ),
    "spectra.photon.phi": Budget(
        rtol=TABULATED_RTOL,
        atol=0.0,
        why="boost integral over phi_photon.csv, same mechanism as the eta "
        "(hazma/spectra/_photon/_phi.pyx).",
    ),
    # -- spectra: positron --------------------------------------------------
    "spectra.positron.muon": Budget(
        rtol=EXACT_RTOL,
        atol=0.0,
        why="ported to Rust (Task 4.1) against this platform's arithmetic, "
        "which it reproduces bit-for-bit; exact because that was achieved, "
        "NOT because the closed form is well conditioned — it is not, and "
        "off this platform the same kernel needs ~1e-8 "
        "(test/test_core_positron_muon.py, 'Why the comparison has two "
        "modes').",
    ),
    "spectra.positron.charged_pion": Budget(
        rtol=PORTED_QUAD_RTOL,
        atol=0.0,
        why="one quad over the positron's rest-frame energy "
        "(hazma/spectra/_positron/_pion.pyx:58, epsabs=1e-10, "
        "epsrel=1e-4) over the closed-form positron-muon spectrum. "
        "Tightened from QUAD_RTOL by Task 4.6 on its own measurement -- "
        "5.5e-15 worst relative over the 1,460 pinned values (1,304 of "
        "them bit-equal), against 1e-12 here.",
    ),
    # -- spectra: neutrino --------------------------------------------------
    "spectra.neutrino.muon": Budget(
        rtol=EXACT_RTOL,
        atol=0.0,
        why="closed-form per-flavor spectra, libc math only "
        "(hazma/spectra/_neutrino/_muon.pyx:8). Ported to Rust in Task "
        "4.6, which reproduces this platform bit-for-bit at all 3,795 "
        "pinned values -- exact because that was achieved, not because "
        "the closed form is well conditioned.",
    ),
    "spectra.neutrino.charged_pion": Budget(
        rtol=PORTED_QUAD_RTOL,
        atol=0.0,
        why="two energy-space quads at scipy's default tolerances "
        "(hazma/spectra/_neutrino/_pion.pyx:124,127) over the closed-form "
        "neutrino-muon spectrum. Tightened from QUAD_RTOL by Task 4.6 on "
        "its own measurement -- 9.7e-16 worst relative over the 4,185 "
        "pinned values (3,793 of them bit-equal), against 1e-12 here.",
    ),
    # -- scalar-mediator cross sections -------------------------------------
    "cross_sections.scalar.sigma_xx_to_s_to_ff": Budget(
        rtol=EXACT_RTOL,
        atol=0.0,
        why="closed-form s-channel cross section; arithmetic and the "
        "below-threshold branch only.",
    ),
    "cross_sections.scalar.sigma_xx_to_s_to_gg": Budget(
        rtol=EXACT_RTOL,
        atol=0.0,
        why="closed-form xx -> s* -> g g, arithmetic only.",
    ),
    "cross_sections.scalar.sigma_xx_to_s_to_pi0pi0": Budget(
        rtol=EXACT_RTOL,
        atol=0.0,
        why="closed-form xx -> s* -> pi0 pi0, arithmetic only.",
    ),
    "cross_sections.scalar.sigma_xx_to_s_to_pipi": Budget(
        rtol=EXACT_RTOL,
        atol=0.0,
        why="closed-form xx -> s* -> pi pi, arithmetic only.",
    ),
    "cross_sections.scalar.sigma_xx_to_ss": Budget(
        rtol=EXACT_RTOL,
        atol=0.0,
        why="closed-form t/u-channel result, arithmetic only.",
    ),
    "cross_sections.scalar.sigma_ss_to_xx": Budget(
        rtol=EXACT_RTOL,
        atol=0.0,
        why="closed-form crossing of sigma_xx_to_ss, arithmetic only.",
    ),
    "cross_sections.scalar.sigma_xl_to_xl": Budget(
        rtol=EXACT_RTOL,
        atol=0.0,
        why="closed-form elastic scattering; the negatives and infinities "
        "the corpus stores are branch behavior, not integration noise.",
    ),
    "cross_sections.scalar.sigma_xpi_to_xpi": Budget(
        rtol=EXACT_RTOL,
        atol=0.0,
        why="closed-form elastic scattering, arithmetic only.",
    ),
    "cross_sections.scalar.sigma_xpi0_to_xpi0": Budget(
        rtol=EXACT_RTOL,
        atol=0.0,
        why="closed-form elastic scattering, arithmetic only.",
    ),
    "cross_sections.scalar.sigma_xg_to_xg": Budget(
        rtol=EXACT_RTOL,
        atol=0.0,
        why="closed-form elastic scattering, arithmetic only.",
    ),
    "cross_sections.scalar.sigma_xs_to_xs": Budget(
        rtol=EXACT_RTOL,
        atol=0.0,
        why="closed-form elastic scattering, arithmetic only.",
    ),
    "cross_sections.scalar.thermal_cross_section": Budget(
        rtol=PORTED_QUAD_RTOL,
        atol=0.0,
        why="QAGP over z with mediator breakpoints "
        "(hazma/scalar_mediator/_c_scalar_mediator_cross_sections.pyx:1411) "
        "and Bessel K1/K2 prefactors (:1361, :1404); the x > 300 cutoff "
        "is a branch, not a tolerance question. Tightened from QUAD_RTOL "
        "by Task 5.2 on its own measurement -- 3.12e-15 worst relative "
        "over the 285 pinned values, 104 of them bit-equal, so 1e-12 "
        "leaves 320x headroom.",
    ),
    # -- vector-mediator cross sections -------------------------------------
    "cross_sections.vector.sigma_xx_to_v_to_ff": Budget(
        rtol=EXACT_RTOL,
        atol=0.0,
        why="closed-form s-channel cross section, arithmetic only.",
    ),
    "cross_sections.vector.sigma_xx_to_v_to_pipi": Budget(
        rtol=EXACT_RTOL,
        atol=0.0,
        why="closed form; the TypeError the corpus pins at e_cm = 2 mx is "
        "a branch, replayed rather than tolerated.",
    ),
    "cross_sections.vector.sigma_xx_to_v_to_pi0g": Budget(
        rtol=EXACT_RTOL,
        atol=0.0,
        why="closed-form xx -> v* -> pi0 gamma, arithmetic only.",
    ),
    "cross_sections.vector.sigma_xx_to_v_to_pi0v": Budget(
        rtol=EXACT_RTOL,
        atol=0.0,
        why="closed form; like sigma_xx_to_v_to_pipi it raises at "
        "e_cm = 2 mx, which the runner replays.",
    ),
    "cross_sections.vector.sigma_xx_to_vv": Budget(
        rtol=EXACT_RTOL,
        atol=0.0,
        why="closed-form t/u-channel result, arithmetic only.",
    ),
    "cross_sections.vector.thermal_cross_section": Budget(
        rtol=PORTED_QUAD_RTOL,
        atol=0.0,
        why="QAGP over z with mediator breakpoints "
        "(hazma/vector_mediator/_c_vector_mediator_cross_sections.pyx:656) "
        "and Bessel K1/K2 prefactors (:606, :650); the x > 300 saturation "
        "is a branch, not a tolerance question. Tightened from QUAD_RTOL "
        "by Task 5.1 on its own measurement -- 2.06e-14 worst relative "
        "over the 285 pinned values, 64 of them bit-equal, so 1e-12 "
        "leaves 49x headroom.",
    ),
    # -- mediator spectra ---------------------------------------------------
    "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum": Budget(
        rtol=PORTED_NESTED_RTOL,
        atol=0.0,
        why="cos(theta) QAGP "
        "(hazma/scalar_mediator/scalar_mediator_decay_spectrum.pyx:184) "
        "over an integrand that calls the quad-backed charged-pion photon "
        "kernel (:2-3). Tightened from NESTED_RTOL by Task 6.2 on its own "
        "measurement -- 5.33e-12 worst relative over the 8,610 pinned "
        "values, 6,379 of them bit-equal, worst at "
        "ms_550.boosted_strong.default. 188x headroom. The residual is the "
        "quadrature port's rather than the transliteration's: at "
        "eng_s == ms the boost integrand is a constant and every channel "
        "agrees to within one ulp.",
    ),
    "mediator_spectra.scalar.positron.dnde_decay_s": Budget(
        rtol=PORTED_NESTED_RTOL,
        atol=0.0,
        why="cos(theta) QAGP "
        "(hazma/scalar_mediator/scalar_mediator_positron_spec.pyx:209) "
        "over the quad-backed positron charged-pion kernel (:2). "
        "Tightened from NESTED_RTOL by Task 6.3 on its own measurement -- "
        "2.33e-12 worst relative over the 16,740 pinned values, 13,403 of "
        "them bit-equal, worst at ms_550.boosted_strong.pi_pi. 429x "
        "headroom. As for the photon pair, the residual is the quadrature "
        "port's rather than the transliteration's.",
    ),
    "mediator_spectra.scalar.positron.dnde_decay_s_pt": Budget(
        rtol=PORTED_NESTED_RTOL,
        atol=0.0,
        why="scalar-argument twin of dnde_decay_s, same nested quadrature "
        "(hazma/scalar_mediator/scalar_mediator_positron_spec.pyx:209), "
        "and bit-for-bit the same values -- Task 6.3 measured the two "
        "entry points identical over the whole corpus, because the port "
        "serves both from one kernel. Tightened with its twin, on the "
        "same 2.33e-12.",
    ),
    "mediator_spectra.vector.photon.dnde_decay_v": Budget(
        rtol=PORTED_NESTED_RTOL,
        atol=0.0,
        why="cos(theta) QAGP "
        "(hazma/vector_mediator/vector_mediator_decay_spectrum.pyx:219) "
        "over an integrand that calls the quad-backed charged-pion photon "
        "kernel (:10). Tightened from NESTED_RTOL by Task 6.2 on its own "
        "measurement -- 1.19e-12 worst relative over the 29,295 pinned "
        "values, 22,918 of them bit-equal, worst at "
        "mv_900.boosted_strong.mu_mu. 838x headroom.",
    ),
    "mediator_spectra.vector.photon.dnde_decay_v_pt": Budget(
        rtol=PORTED_NESTED_RTOL,
        atol=0.0,
        why="scalar-argument twin of the vector photon dnde_decay_v, same "
        "nested quadrature "
        "(hazma/vector_mediator/vector_mediator_decay_spectrum.pyx:219), "
        "and bit-for-bit the same values -- Task 6.2 measured the two "
        "entry points identical over its whole grid, because the port "
        "serves both from one kernel. Tightened with its twin, on the "
        "same 1.19e-12.",
    ),
    "mediator_spectra.vector.positron.dnde_decay_v": Budget(
        rtol=PORTED_NESTED_RTOL,
        atol=0.0,
        why="cos(theta) QAGP "
        "(hazma/vector_mediator/vector_mediator_positron_spec.pyx:210) "
        "over the quad-backed positron charged-pion kernel (:10). "
        "Tightened from NESTED_RTOL by Task 6.3 on its own measurement -- "
        "1.50e-12 worst relative over the 16,740 pinned values, 13,684 of "
        "them bit-equal, worst at mv_900.boosted_strong.total. 665x "
        "headroom. One Rust kernel serves this case and the scalar one "
        "above, the two .pyx having been the same text.",
    ),
    "mediator_spectra.vector.positron.dnde_decay_v_pt": Budget(
        rtol=PORTED_NESTED_RTOL,
        atol=0.0,
        why="scalar-argument twin of the vector positron dnde_decay_v, "
        "same nested quadrature "
        "(hazma/vector_mediator/vector_mediator_positron_spec.pyx:210), "
        "and bit-for-bit the same values, for the reason its scalar "
        "counterpart is. Tightened with its twin, on the same 1.50e-12.",
    ),
}


def budget_for(case_name: str) -> Budget:
    """The declared budget for one corpus case.

    Raises
    ------
    KeyError
        If the case has no declared budget. A corpus case without one is
        a gap in the gate, not something to fall back from — the port
        would then be free to move that entry point by any amount.
    """
    try:
        return BUDGETS[case_name]
    except KeyError:
        raise KeyError(
            f"no tolerance budget declared for corpus case {case_name!r}. "
            "Add one to test/parity/tolerances.py with a one-line "
            "justification (projects/cython-to-rust/rules.md, rule 2)."
        ) from None


# ===========================================================================
# ---- Which budget actually applies ----------------------------------------
# ===========================================================================

#: Manifest environment keys that can move a captured value. `hazma`'s own
#: version is deliberately absent: Phase 07 bumps it without touching a
#: number, and a version bump must not silently relax the gate.
_NUMERICS_ENVIRONMENT_KEYS = (
    "python",
    "numpy",
    "scipy",
    "cython",
    "platform",
    "machine",
)


def _libm_identity(environment: dict[str, Any]) -> tuple[str, str]:
    """Which libm an environment implies: OS family and CPU architecture.

    Coarser than the `platform` key on purpose. `platform.platform()`
    records the point release -- ``macOS-26.5.2-arm64-arm-64bit`` -- and
    the capturing machine has since moved to 26.6.1. Comparing the whole
    string would call that a platform change and drop the `EXACT` class
    from bit-equality to `PLATFORM_EXACT_RTOL` on the very machine the
    corpus was captured on, silently weakening the gate Tasks 4.1-4.5
    rely on, on an OS update nobody connected to Hazma.

    The first component and `machine` are what actually select an
    implementation of `atan`. If a point release *does* move one, the
    corpus fails loudly at ``rtol = 0`` and somebody looks -- which is
    the outcome the ``EXACT`` class docstring asks for ("such a shift
    should be measured and declared, not absorbed silently"), not the
    one a version-string comparison would give.

    Parameters
    ----------
    environment : dict
        A `generate.environment()` mapping, live or from the manifest.
    """
    return (
        str(environment.get("platform", "")).partition("-")[0],
        str(environment.get("machine", "")),
    )


def provenance(manifest: dict[str, Any]) -> Provenance:
    """Decide whether this tree *is* the tree the corpus was captured from.

    The comparison covers three things that can each move a value on their
    own: the kernel sources (`generate.kernel_digest`), the toolchain and
    numerics libraries the manifest records, and whether any kernel has
    already been ported to Rust.

    Parameters
    ----------
    manifest : dict
        The parsed ``data/manifest.json``.

    Returns
    -------
    Provenance
        `exact` is True only when all three agree.
    """
    differences: list[str] = []

    stored_env = manifest["environment"]
    live_env = corpus_generate.environment()
    differences.extend(
        f"{key} {stored_env.get(key)!r} -> {live_env.get(key)!r}"
        for key in _NUMERICS_ENVIRONMENT_KEYS
        if stored_env.get(key) != live_env.get(key)
    )
    same_platform = _libm_identity(stored_env) == _libm_identity(live_env)

    stored_digest = manifest["kernel_digest"]["sha256"]
    live_digest = corpus_generate.kernel_digest()["sha256"]
    if stored_digest != live_digest:
        differences.append(f"kernel digest {stored_digest[:12]} -> {live_digest[:12]}")

    # A *served* kernel, not merely the extension's existence. From
    # cython-to-rust Phase 02 the scaffold ships in every build while every
    # value still comes from Cython, so keying on importability would drop
    # the whole of Phases 02-03 out of bit-equality mode for no reason —
    # exactly when the port most needs a gate that catches one ulp.
    served = corpus.rust_core_kernels()
    if served:
        differences.append(f"hazma._core serves {len(served)} kernel(s)")

    return Provenance(
        exact=not differences,
        same_platform=same_platform,
        detail="; ".join(differences),
    )


def effective_budget(case_name: str, tree: Provenance) -> Budget:
    """The budget the runner enforces, given what tree it is running on.

    On the capturing tree the declared budgets are not the right gate.
    The corpus was captured from these exact kernels in this exact
    environment, so any difference at all is a regression somebody
    introduced, and the phase file asks for bit-equality accordingly
    ("Running against unmodified Cython passes bit-exact"). Once the tree
    diverges — a ported kernel, a new scipy, a different platform — the
    declared budget takes over, because that is precisely the situation it
    was written for.

    Parameters
    ----------
    case_name : str
        A `cases.build_cases()` key.
    tree : Provenance
        From `provenance`.
    """
    # Looked up before the branch on purpose: a case with no declared
    # budget must fail even on the capturing tree, where the override
    # would otherwise never consult the table.
    declared = budget_for(case_name)
    if tree.exact:
        return Budget(
            rtol=EXACT_RTOL,
            atol=0.0,
            why="running on the capturing tree, where the corpus pins this "
            "implementation against itself",
        )
    if not tree.same_platform and declared.rtol in _PLATFORM_FLOORS:
        # The relaxations with a host, not an implementation, behind
        # them. `EXACT` means "reaches only libc.math and must agree
        # bit-for-bit"; off the capturing libm the second half of that is
        # not something any implementation can deliver, so the class
        # falls back to the figure the corpus's own threshold sampling
        # implies. `SPECFUN` is held at 1e-13 by `spence` being bit-equal
        # to scipy, which is likewise a statement about one libm.
        #
        # Deliberately a two-row table rather than
        # `max(declared, some_floor)`: `TABULATED` and the two `PORTED_*`
        # budgets are also tighter than `PLATFORM_EXACT_RTOL`, and
        # nothing measured says they need relaxing. Widening them on the
        # theory that they might is the same over-broad exemption the
        # zero floor got wrong (PR #71 review round 1). A class that does
        # need it should arrive as a loud failure somebody measures.
        floor, name = _PLATFORM_FLOORS[declared.rtol]
        return Budget(
            rtol=floor,
            atol=0.0,
            why=f"{declared.why} -- held to {name} rather than the declared "
            "figure because this is not the capturing platform, so "
            "libc.math itself differs in the last ulp",
        )
    return declared


#: Declared budget -> (budget off the capturing libm, its constant name).
#: Only the two classes measured to need it; see `effective_budget`.
_PLATFORM_FLOORS: dict[float, tuple[float, str]] = {
    EXACT_RTOL: (PLATFORM_EXACT_RTOL, "PLATFORM_EXACT_RTOL"),
    SPECFUN_RTOL: (PLATFORM_SPECFUN_RTOL, "PLATFORM_SPECFUN_RTOL"),
}


def zero_floor(expected: np.ndarray) -> float:
    """How large a `stability.PORTABILITY_ZEROS` position may come back.

    The caller decides *where* this applies — only at the declared
    positions — so this function answers only "how big is this array?".

    Parameters
    ----------
    expected : numpy.ndarray
        The stored values of a single block array, with any `stability`
        mask already applied. Non-finite entries are ignored when taking
        the scale: three blocks pin a ``nan`` where the entry point
        raised, and nine positions across the scalar elastic cross
        sections pin ``+-inf`` at ``e_cm = 2 mx`` (all nine masked by
        `stability`, so they do not reach here). Neither says anything
        about how big the function is.

    Returns
    -------
    float
        ``ZERO_FLOOR_FRACTION`` times the median non-zero magnitude, or
        ``0.0`` for an array with no finite non-zero value -- where there
        is no scale to be a fraction of, exact zero remains the contract.
    """
    scale = np.abs(expected[np.isfinite(expected) & (expected != 0.0)])
    if scale.size == 0:
        return 0.0
    return float(ZERO_FLOOR_FRACTION * np.median(scale))


def abscissa_budget(tree: Provenance) -> Budget:
    """How far a *sampling point* may move, given what tree we are on.

    Separate from `effective_budget` because an abscissa is not a value:
    it records where the entry point was probed, and every case is probed
    on grids built the same way, so one budget covers all of them.

    Bit-exact on the capturing tree, for the same reason the values are.
    Off it, `ABSCISSA_RTOL` — the platform libm reaches `numpy.geomspace`
    and moves the last ulp, which is not a moved measurement point.

    Parameters
    ----------
    tree : Provenance
        From `provenance`.
    """
    if tree.exact:
        return Budget(
            rtol=EXACT_RTOL,
            atol=0.0,
            why="running on the capturing tree, where the grid must "
            "reproduce bit-for-bit",
        )
    return Budget(
        rtol=ABSCISSA_RTOL,
        atol=0.0,
        why="geomspace goes through the platform libm (log10 then a "
        "power); ~4e-15 is the worst this mechanism can produce, and a "
        "genuinely redesigned grid moves points by >=1e-3 relative",
    )
