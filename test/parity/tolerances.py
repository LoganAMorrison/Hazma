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
    `atan`, `atanh` and branch selection, nothing else — everything in
    this class reaches only `libc.math`. The phase file asks for
    bit-equality here against the capturing commit. Note this is strict
    enough to catch a libm change — Rust's `f64::exp` need not agree with
    the platform C library in the last ulp — which is deliberate: such a
    shift should be measured and declared, not absorbed silently.
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
``QUAD`` (rtol 1e-8)
    One adaptive `scipy.integrate.quad` call. The numerics reference
    expects a faithful netlib-QUADPACK port to reproduce these to ~1e-12
    or better; the phase file sets the opening budget at 1e-8 and tightens
    it once Phase 03 has measured the port.
``NESTED`` (rtol 1e-6)
    An adaptive quadrature whose integrand is itself an adaptive
    quadrature. Subdivision is a discontinuous function of the integrand:
    a last-ulp change inside can move an outer bisection decision, and the
    outer call is then only as accurate as its own `epsrel`, which is
    1e-5 at every live site. 1e-6 sits one decade inside that, so the
    budget tracks the integrator's own accuracy claim rather than the
    resolution of the arithmetic.

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
QUAD_RTOL = 1e-8
NESTED_RTOL = 1e-6

#: How far the *abscissae* may move off the capturing tree. Not a value
#: budget: see "Abscissae are their own class" above for the derivation
#: and for why bit-equality here was unreachable on any second platform.
ABSCISSA_RTOL = 1e-13


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
    detail : str
        Empty when `exact`; otherwise a human-readable list of what
        differs. Surfaced as the skip reason on
        `test_parity.test_running_on_the_capturing_tree`, so the mode is
        never something a reader has to infer.
    """

    exact: bool
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
        rtol=QUAD_RTOL,
        atol=0.0,
        why="one QAGP over cos(theta) "
        "(hazma/spectra/_photon/_pion.pyx:123, epsabs=1e-10, epsrel=1e-5); "
        "the integrand is the spence-bearing muon spectrum, whose 1e-13 "
        "class sits well inside the quadrature budget.",
    ),
    "spectra.photon.neutral_pion": Budget(
        rtol=EXACT_RTOL,
        atol=0.0,
        why="closed-form boosted box spectrum, arithmetic and branch "
        "selection only (hazma/spectra/_photon/_pion.pyx:168-196).",
    ),
    "spectra.photon.charged_rho": Budget(
        rtol=NESTED_RTOL,
        atol=0.0,
        why="quad at hazma/spectra/_photon/_rho.pyx:123 whose integrand "
        "calls the charged- and neutral-pion point kernels "
        "(hazma/spectra/_photon/_rho.pyx:10-11 and :95-96), the first of "
        "which quads again — the nested case the phase file singles out.",
    ),
    "spectra.photon.neutral_rho": Budget(
        rtol=NESTED_RTOL,
        atol=0.0,
        why="quad at hazma/spectra/_photon/_rho.pyx:52 over an integrand "
        "that calls the quad-backed charged-pion point kernel "
        "(hazma/spectra/_photon/_rho.pyx:26) — the second nested-rho "
        "entry point.",
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
        rtol=QUAD_RTOL,
        atol=0.0,
        why="one quad over cos(theta) "
        "(hazma/spectra/_positron/_pion.pyx:58, epsabs=1e-10, "
        "epsrel=1e-4) over the closed-form positron-muon spectrum.",
    ),
    # -- spectra: neutrino --------------------------------------------------
    "spectra.neutrino.muon": Budget(
        rtol=EXACT_RTOL,
        atol=0.0,
        why="closed-form per-flavor spectra, libc math only "
        "(hazma/spectra/_neutrino/_muon.pyx:8).",
    ),
    "spectra.neutrino.charged_pion": Budget(
        rtol=QUAD_RTOL,
        atol=0.0,
        why="two energy-space quads at scipy's default tolerances "
        "(hazma/spectra/_neutrino/_pion.pyx:124,127) over the closed-form "
        "neutrino-muon spectrum.",
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
        rtol=QUAD_RTOL,
        atol=0.0,
        why="QAGP over z with mediator breakpoints "
        "(hazma/scalar_mediator/_c_scalar_mediator_cross_sections.pyx:1411) "
        "and Bessel K1/K2 prefactors (:1361, :1404); the quadrature, not "
        "the 1e-13 Bessel class, sets the budget.",
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
        rtol=QUAD_RTOL,
        atol=0.0,
        why="QAGP over z with mediator breakpoints "
        "(hazma/vector_mediator/_c_vector_mediator_cross_sections.pyx:656) "
        "and Bessel K1/K2 prefactors (:606, :650); the x > 300 saturation "
        "is a branch, not a tolerance question.",
    ),
    # -- mediator spectra ---------------------------------------------------
    "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum": Budget(
        rtol=NESTED_RTOL,
        atol=0.0,
        why="cos(theta) QAGP "
        "(hazma/scalar_mediator/scalar_mediator_decay_spectrum.pyx:184) "
        "over an integrand that calls the quad-backed charged-pion photon "
        "kernel (:2-3).",
    ),
    "mediator_spectra.scalar.positron.dnde_decay_s": Budget(
        rtol=NESTED_RTOL,
        atol=0.0,
        why="cos(theta) QAGP "
        "(hazma/scalar_mediator/scalar_mediator_positron_spec.pyx:209) "
        "over the quad-backed positron charged-pion kernel (:2).",
    ),
    "mediator_spectra.scalar.positron.dnde_decay_s_pt": Budget(
        rtol=NESTED_RTOL,
        atol=0.0,
        why="scalar-argument twin of dnde_decay_s, same nested quadrature "
        "(hazma/scalar_mediator/scalar_mediator_positron_spec.pyx:209).",
    ),
    "mediator_spectra.vector.photon.dnde_decay_v": Budget(
        rtol=NESTED_RTOL,
        atol=0.0,
        why="cos(theta) QAGP "
        "(hazma/vector_mediator/vector_mediator_decay_spectrum.pyx:219) "
        "over an integrand that calls the quad-backed charged-pion photon "
        "kernel (:10).",
    ),
    "mediator_spectra.vector.photon.dnde_decay_v_pt": Budget(
        rtol=NESTED_RTOL,
        atol=0.0,
        why="scalar-argument twin of the vector photon dnde_decay_v, same "
        "nested quadrature "
        "(hazma/vector_mediator/vector_mediator_decay_spectrum.pyx:219).",
    ),
    "mediator_spectra.vector.positron.dnde_decay_v": Budget(
        rtol=NESTED_RTOL,
        atol=0.0,
        why="cos(theta) QAGP "
        "(hazma/vector_mediator/vector_mediator_positron_spec.pyx:210) "
        "over the quad-backed positron charged-pion kernel (:10).",
    ),
    "mediator_spectra.vector.positron.dnde_decay_v_pt": Budget(
        rtol=NESTED_RTOL,
        atol=0.0,
        why="scalar-argument twin of the vector positron dnde_decay_v, "
        "same nested quadrature "
        "(hazma/vector_mediator/vector_mediator_positron_spec.pyx:210).",
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

    return Provenance(exact=not differences, detail="; ".join(differences))


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
    return declared


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
