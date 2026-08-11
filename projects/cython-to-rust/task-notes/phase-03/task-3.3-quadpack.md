# Task 3.3: QUADPACK port (qk15, qk21, qelg, qags, qagp)

**Date:** 2026-08-10
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-03-numerics-foundation.md`
(Task 3.3), `../../references/numerics-replacements.md`
(§`scipy.integrate.quad` (QUADPACK) call sites), `../../rules.md` rule 5
(Licensing 1), rules 6–9 (Rust conventions 1–4)
**Related ADRs:** ADR-0002 (Accepted 2026-08-04 — fixes the provenance:
public-domain netlib QUADPACK, nothing GSL-derived)
**Depends On:** none

## Objective

Replace the twelve live `scipy.integrate.quad` call sites' integrator
with an in-tree Rust translation of the netlib QUADPACK Fortran —
`qk15`, `qk21`, `qelg`, `qpsrt`, `qagse`, `qagpe` — plus the
scipy-shaped driver that reproduces scipy's own break-point
preprocessing, so a ported kernel calls one function and inherits the
argument handling the `.pyx` inherited from scipy.

## Exit Criteria

Copied from `../../phases/phase-03-numerics-foundation.md` §Task 3.3:

- Finite-interval `qags` and `qagp` in `rust/src/quad.rs`, translated
  from netlib QUADPACK Fortran (provenance header per rules.md rule 5),
  closure-based API carrying `epsabs`/`epsrel`/`limit`/breakpoints.
- **Breakpoint preprocessing contract, pinned empirically against
  scipy** (do not design it from the QUADPACK docs alone): determine by
  experiment what `scipy.integrate.quad(points=...)` does with unsorted
  lists, duplicates, points coinciding with the endpoints, and points
  outside `[a, b]` — then replicate that behavior (including any raised
  errors) exactly. Both degenerate cases occur live. Tests cover three
  parameter regimes per thermal call: breakpoints interior (resonance
  active), breakpoints at/near threshold, and breakpoints outside the
  interval (inactive).
- Unit tests: QUADPACK's own reference problems, plus every live
  integrand *shape* from the call-site table in
  `../../references/numerics-replacements.md`, compared against
  `scipy.integrate.quad` at matching settings — agreement within 10× the
  requested tolerance, and within 1e-12 rel on smooth cases.
- Error/abnormal-termination behavior mapped (roundoff, max
  subdivisions, invalid breakpoints) — returns a Result, never panics
  across FFI.

Plus the four **criteria added during execution**, which are patched
into the phase file in this same PR (see `## Plan Impact`).

## Inputs Reviewed

- `../../PLAN.md` (Scope, Numerical impact, Phases), `../README.md`,
  `phase-03/README.md`, `../../phases/phase-03-numerics-foundation.md`.
- `../../references/numerics-replacements.md` §quad call sites — the
  seven-row table (twelve call sites) and the breakpoint-degeneracy
  paragraph.
- `../../rules.md` (all sections), `../../adrs/ADR-0002-license-clean-numerics.md`.
- `docs/agents/lessons.md`, `docs/agents/environment.md`,
  `docs/agents/doc-consistency.md`.
- The live Cython: `hazma/spectra/_photon/_pion.pyx`,
  `_photon/_rho.pyx`, `_positron/_pion.pyx`, `_neutrino/_pion.pyx`, both
  `_c_*_mediator_cross_sections.pyx`, and the four mediator spectrum
  modules.
- **netlib QUADPACK Fortran**, retrieved 2026-08-10 from
  <https://www.netlib.org/quadpack/>: `dqk15.f`, `dqk21.f`, `dqelg.f`,
  `dqpsrt.f`, `dqagse.f`, `dqagpe.f`.
- `scipy/integrate/_quadpack_py.py` (scipy 1.18.0), which is where the
  break-point contract actually lives.
- Task 3.2's `rust/src/special.rs` / `special_probe.rs` and
  `test/test_core_special.py` as the pattern for a PyO3-free kernel
  module with a Python-visible test surface.

## Findings

- **The break-point contract is scipy's, not QUADPACK's.** The exit
  criterion's instruction to pin it empirically was right for a reason
  it did not state: the whole contract is three lines of Python in
  `scipy/integrate/_quadpack_py.py`'s `_quad` — `np.unique(points)`, then
  `[a < p]`, then `[p < b]` — run *after* `quad` has ordered the limits
  and before `qagpe` sees anything. QUADPACK's own breakpoint handling
  (sort, and `ier = 6` if the extremes do not match `a` and `b`) is
  unreachable from scipy. Reading the QUADPACK documentation would have
  produced a port that *errors* on the five `points=[-1, 1]` call sites,
  because QUADPACK rejects a breakpoint equal to an endpoint and scipy
  silently drops it.
- **Both live degeneracies are discards, and that changes which routine
  runs.** `points=[-1, 1]` on `[-1, 1]` leaves **zero** interior
  breakpoints; the heavy-mediator thermal entries `m/mx`, `2 m/mx`
  likewise drop when they exceed `max(50/x, 100|150)`. But scipy
  dispatches on `points is None` *before* filtering, so five of the twelve
  live call sites run **`qagpe` with an empty list**, not `qagse`. A port
  that treated "no breakpoint survived" as "no breakpoints" would pick
  the wrong routine.
- **…and the two routines are almost, but not quite,
  indistinguishable.** `qagpe` decides an interval is the smallest by
  subdivision `level`, `qagse` by comparing length against `small`, so
  `qagpe` extrapolates one bisection earlier. Over 3,776 random
  (integrand, tolerance, limit) combinations they returned identical
  values, `neval` and `last` on every run that converged and differed on
  45 — all of them runs that exhausted `limit`. The test that pins the
  dispatch had to be *built around* that: the obvious singular integrands
  cannot tell the routines apart, and a test using one would have passed
  against either. `|x − 1/3|^(−9/10)·cos(50x)` at `limit = 10` puts an
  11% gap between them.
- **Only `qk21` is on the live path.** `scipy.integrate.quad` on a finite
  interval runs `qagse` or `qagpe`, and both evaluate with the 21-point
  Gauss–Kronrod rule and nothing else. `qk15` — named in this task's
  first exit criterion — is reachable from no hazma call site. It is
  ported anyway (the criterion says so) and earns its keep as an
  independent second rule: the cross-rule agreement test would not exist
  without it.
- **The port and scipy diverge essentially only where QUADPACK says it
  failed.** 11,274 random (integrand, tolerance, limit, points)
  combinations: the 4,461 that converged reproduced scipy's `neval` and
  `last` on all but **5** (0.11%), with the value within 3.6e-2 of the
  requested tolerance and 8.2e-11 relative at worst. The 6,813 that
  exhausted `limit` can separate without bound — 4.5e-5 in that sweep,
  11% on a hand-picked case. Termination flags agreed on all 11,274. The
  mechanism is not a translation defect: the subdivision is identical
  there too (pinned in `TestDivergenceRegime`), and Wynn's ε-algorithm is
  chaotic on a sequence that is not converging, so a few ulp in the table
  is enough. Every live shape returns `ier = 0`, asserted rather than
  assumed.
- **A sweep's parameter space is part of its result, and this one nearly
  shipped a cleaner claim than the truth.** An earlier 6,000-combination
  design drew `points` from `{None, [], [0.5], [0.25, 0.75], [-1, 1]}` —
  at most two break points — and reported **zero** subdivision mismatches
  among 2,812 converged runs. That would have gone into three durable
  docs as "reproduces scipy's subdivision exactly". Adding 9- and
  39-point grids to the draw produced 5 mismatches in 4,461, and the
  first many-break-point case looked at by hand
  (`1/((x − 0.16619)² + 1e-14)` over a 19-point grid) diverges in
  `neval`/`last` while both implementations converge and agree on the
  value to 9.2e-13. The published figures are the wider sweep's.
- **Two adaptive-loop heuristics survive every test built from the spec.**
  `qagpe`'s `ndin` flag and `qagse`'s roundoff counters change only which
  subinterval is bisected next, so mutating either passed every test the
  module held at the time — 51 of what were then 53, the two additions
  being the tests written to catch these — and all 24 `cargo` units
  besides. Inputs that expose them exist but had to be
  *searched for* with the mutation in place: `sin(293.25/x)` over a
  39-point grid moves by a factor of 48 without `ndin`, and a near-delta
  spike at 0.16309 with `points=[0.5]` moves by 2,800 when the 0.99
  roundoff threshold is relaxed. Both live in the limit-exhausted regime,
  so both are pinned at a coarse `rtol = 1e-6` — enough for defects that
  move the answer by factors, and not a claim about digits the two
  implementations may legitimately disagree on.
- **A mutation harness can poison its own baseline, and nothing in it
  will say so.** The first campaign was launched, appeared to fail on an
  unset shell variable, and was re-launched — but the first run *was*
  alive, so two campaigns interleaved writes to `rust/src/quad.rs` and
  the second read a "pristine" `original` that already carried the
  first's mutation. Every result it produced was measured against a
  wrong table. The tell was subtle and easy to rationalise: mutating a
  `qk15` weight reported `qk21` tests failing. What settled it was a
  check that owed nothing to the crate — re-parsing the Fortran `data`
  statements and comparing f64 bit patterns against the Rust literals.
  The rewritten harness asserts a green baseline before it starts, holds
  a lock file, and re-asserts green at the end. **Not** written into
  `docs/agents/lessons.md`: that file's contract puts the append step in
  `review-respond` and requires a real PR citation, which this task does
  not have yet. If review agrees the class is worth carrying, the entry
  belongs there with this PR's number —
  `[harness-contaminated-its-own-baseline]`, generalising
  `[measurement-taken-before-the-task-ended]` from a measurement that
  expired to a measurement whose *baseline* was never what it claimed.
- **`cargo test`'s default parallelism corrupts a scraped failure list.**
  The harness writes each test's name and its result as two separate
  writes, so with threads in play a single output line can carry one
  test's name and another's `FAILED`. A scraper keyed on "the line starts
  with the word test and contains FAILED" then attributes the failure to
  whichever name came first. That produced a stable set of phantom
  failures across every mutation in the poisoned run and made the real
  signal harder to see. Passing `--test-threads=1` through to the harness
  fixes it.

## Decisions and Implementation Notes

- **Review round 2 (PR #60, 2026-08-10) — the `limit`-length workspaces
  are grown on demand, not allocated at `limit`.** Round 1 capped `limit`
  at `i32::MAX` to match scipy and called the hazard closed; it was not.
  `qagse` allocates five `limit`-length arrays (40 bytes a subinterval)
  and `qagpe` six (44), so a `limit` just *under* the new cap still
  reserved ~80 GiB and ~88 GiB before the first panel was evaluated —
  and Rust aborts the process on allocation failure rather than
  unwinding, so it could never have become a Python exception. The
  round-1 test even asserted the opposite ("never allocates the full
  workspace"), which was false as written. The arrays now start at
  `WORKSPACE_SEED = 64` subintervals (or the initial partition, whichever
  is larger) and double, capped at `limit + 2`; every index in both
  routines is at most `last`, which grows by one per iteration, so one
  slot of headroom is sufficient. Three `cargo` tests cover it, and the
  one that matters asks for `usize::MAX / 4096` — a size no allocator can
  satisfy, so it discriminates on every platform rather than only where
  the allocator declines to overcommit. Reverting to eager sizing dies
  with `memory allocation of 36028797018963976 bytes failed` / SIGABRT.
  **This is why the round-1 measurement was not evidence**: peak RSS went
  18.2 → 18.6 MB at `limit = INT_MAX`, because macOS maps zero pages
  lazily. The reservation was real and the test could not see it.
- **Review round 1 (PR #60, 2026-08-10) — `limit` is validated at the
  Python boundary, not by PyO3.** The probe took `limit: usize`, so
  `limit = -1` died in PyO3's conversion with `OverflowError` where scipy
  raises `ValueError` — the docstring's "raises `ValueError` for exactly
  the inputs scipy raises `ValueError` for" was simply false. It now
  takes `i64` and reproduces *both* of scipy's rejections: `< 1` folds
  onto `0` and takes the existing `QuadError::LimitTooSmall` path (one
  message, not two spellings), and `> i32::MAX` raises `OverflowError`
  exactly as scipy's own C-`int` conversion does. That upper guard is
  load-bearing beyond the contract: `limit` sizes the `qagse`/`qagpe`
  workspace at 16 bytes an entry, so `limit = 10**12` was a 16 TB
  allocation request that this machine happened to satisfy lazily —
  whether it survives is a property of the platform's overcommit policy,
  not of the code, and the exit criteria say "never panics across FFI".
  Six new tests, each validity-checked against a mutation.
- **Literal translation, 1-based indexing preserved.** Every array in
  `quad.rs` carries a dead element 0 so the Fortran's `alist(maxerr)`
  reads as `alist[maxerr]`, and every `go to` is a labelled `break`
  carrying the original statement number in a comment. Idiomatic Rust
  would read better and be much harder to check against the source; the
  point of the module is that a reviewer can put the two side by side.
  The cost is three module-level clippy `allow`s
  (`needless_range_loop`, `explicit_counter_loop`, `int_plus_one`), each
  with the reason written down — clippy's rewrites are all correct and
  each one costs a line of the correspondence.
- **`quad` is the entry point, `qagse`/`qagpe` are public but secondary.**
  The kernels being ported call `scipy.integrate.quad`, so the port
  exposes the same shape: limits ordered and the result negated if they
  were reversed, `points` filtered exactly as scipy filters, then
  dispatch. Doing it once here is what stops twelve Phase 04–06 call
  sites from each re-deriving it.
- **`ier` rides along in `Ok`; only invalid input is `Err`.** scipy
  raises `ValueError` for `ier = 6` and merely *warns* for 1–5, handing
  back a usable value — and hazma's call sites all read `quad(...)[0]`
  and never see the warning. Modelling 1–5 as errors would have changed
  behavior at exactly the inputs where the Cython silently carried on.
- **The tables are extracted, not typed.** The `data` statements were
  parsed out of `dqk15.f`/`dqk21.f` by script into Rust literals, and a
  second script compares the two as f64 bit patterns. Beyond that,
  `cargo test` pins each rule by **degree of exactness** (22 for `qk15`,
  31 for `qk21`) plus a complement test that the next even degree is
  *not* exact — a wrong digit breaks exactness, where a spot check
  against one integral could be passed by a rule that is merely close.
  `docs/agents/lessons.md`, `[hand-written-population-in-a-derived-check]`.
- **The Python probe takes a callable, deliberately.** A probe exposing a
  menu of Rust integrands would compare a Rust integrand against a Python
  one and blame the difference on the quadrature. With a callback the
  integrand is byte-identical on both sides and every remaining
  difference is the algorithm — which is also what the Cython does today,
  since `quad` re-enters Python once per node there too. An exception
  from the integrand is captured on first occurrence, short-circuits the
  rest of the run to `NaN`, and is re-raised unchanged.
- **`hazma._core.quad` joins the existing test-only exemption.**
  `cases._CORE_TEST_ONLY_MODULES` already carried
  `hazma._core.special` for exactly this reason (Task 3.2), and
  `test_test_only_core_submodules_have_no_importer` keeps both honest by
  failing the moment anything under `hazma/` imports one. Reusing the
  mechanism rather than widening it is the point —
  `docs/agents/lessons.md`, `[gate-disabled-stays-green]`.
- **`resabs` is the rule applied to `|f|`, not `∫|f|`.** A first test
  asserted the latter and failed at 3.7e-3, because `|x|` is not a
  polynomial. Pinning it against the rule run on `|x|` states the
  invariant exactly and still distinguishes `resabs` from `result` for a
  sign-changing integrand.
- **`qagi` and the infinite-interval machinery are out of scope**, as
  `PLAN.md` says: every live integral is over a finite interval. The
  module docstring says so too, so a later reader does not read the
  omission as an oversight.

## Files Changed

- `rust/src/quad.rs` — **new**, 1,972 lines: `qk15`, `qk21`, `qelg`,
  `qpsrt`, `qagse`, `qagpe`, the `quad` driver and `filter_points`, a
  `# Sources and licensing` provenance header, the call-site table, and
  24 unit tests.
- `rust/src/quad_probe.rs` — **new**, registration-only
  `hazma._core.quad` exposing `quad`, `qk15` and `qk21` to the scipy
  comparison.
- `rust/src/lib.rs` — `pub mod quad;` + `mod quad_probe;`, the submodule
  registration, and the reconciled paragraphs on why the two probe
  modules are the exception to "registration only means per-domain".
- `test/test_core_quad.py` — **new**, 58 tests in 8 classes.
- `test/parity/cases.py`, `test/parity/README.md`,
  `test/parity/test_parity.py` — `hazma._core.quad` added to
  `_CORE_TEST_ONLY_MODULES`, and the three places that named `special`
  alone reconciled.
- `hazma/_core.pyi` — the unstubbed-submodule comment now covers both
  probes (the only change under `hazma/`, and non-executable).
- Canonical patches: `../../phases/phase-03-numerics-foundation.md`
  (four Task 3.3 criteria added during execution),
  `../../references/numerics-replacements.md` (the measured breakpoint
  contract).

Nothing else under `hazma/`, and nothing under `docs/`.

## Verification

Every count below comes from the command printed beside it, run against
the final tree.

### Gates

| Gate | Command | Result |
| --- | --- | --- |
| Rust units | `cargo test --manifest-path rust/Cargo.toml --no-default-features` | `43 passed` (27 new; `grep -c '#\[test\]' rust/src/quad.rs` → 27) |
| Rust format | `cargo fmt --manifest-path rust/Cargo.toml --check` | clean |
| Rust lint | `cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings` | clean |
| New module | `pytest test/test_core_quad.py -q` | `58 passed in 5.10s` |
| Full suite | `pytest -q` (via preflight) | `1212 passed, 13 skipped` — see below |
| Full gate | `scripts/agents/preflight.sh --paths … --md …` | **RESULT: PASS** |

**These are the round-2 figures and they supersede the pre-review ones**
(`40` / `53` / `1207`, and `24` Rust units). Review round 1 added six
Python tests and round 2 three Rust ones, so every count in this section
had to be re-derived rather than adjusted — see the note under
`### Validity` on why the arithmetic was not trusted.

Test population, derived rather than counted by hand:

```sh
pytest test/test_core_quad.py --collect-only -q \
  | awk -F'::' '/::/{print $2}' | sort | uniq -c
```

8 classes / 58 tests — `TestBreakPointPreprocessing` 8,
`TestReferenceProblems` 9, `TestLiveIntegrandShapes` 14,
`TestKronrodRules` 5, `TestTerminationFlags` 6,
`TestDivergenceRegime` 2, `TestAdaptiveHeuristics` 2,
`TestErrorBehavior` 12.

**The parity corpus ran in bit-equality mode.** `1212 passed, 13 skipped`
is +58 on Task 3.2's `1154 passed, 13 skipped`, all of them this task's
new tests, and **the skip count is unchanged** — which is what proves the
mode, since forcing budget mode drops one test to a skip rather than
failing. Confirmed independently before the run:
`tolerances.provenance(manifest)` → `Provenance(exact=True, detail='')`.

### What the tests cover

- **Break-point preprocessing** (8): sorting, deduplication,
  endpoint-coincident points, out-of-interval points, all-points-dropped,
  `NaN`, an empty list still selecting `qagpe`, reversed limits, and the
  `limit <= npts` raise. Each uses a genuinely singular integrand, so a
  break point that survives filtering moves `neval` and `last` visibly —
  on a smooth integrand the whole contract is unobservable.
- **QUADPACK reference problems** (9): the endpoint log singularity at
  six exponents, the interior algebraic singularity with a break point, a
  smooth single-panel case, and an oscillatory cancellation. Analytic
  values, so scipy is not the only oracle in the module.
- **Live integrand shapes** (14): the cos-θ boost-Jacobian sites at four
  β; the three boosted-energy-window sites at their own tolerances
  (including the neutrino row's scipy defaults); the nested ρ integral
  with both levels on the port; and **both** mediators' thermal ⟨σv⟩ at
  the three break-point regimes the exit criteria name, running the
  *actual* Cython integrand via `sigma_xx_to_all`. Each asserts
  `ier = 0`.
- **The bare rules** (5): `qk21` is bit-identical to `quad` on a single
  panel, both rules integrate a low-degree polynomial exactly, `resabs`
  is the rule applied to `|f|`, and the two rules agree.
- **Termination flags** (6): `ier` 0–5 each driven by a purpose-built
  input and compared against scipy's own code, decoded from the message
  `full_output` appends.
- **Divergence regime** (2) and **adaptive heuristics** (2): documented
  above.
- **Error behavior** (12): invalid tolerances; every `limit < 1`
  (`0`, `-1`, `-100`) raising `ValueError` as scipy does and every
  `limit > i32::MAX` raising `OverflowError` as scipy does, plus the
  largest accepted `limit`; the subdivision limit; an exception raised by
  the integrand (propagated unchanged, integrand called once); a
  non-float return; a `NaN` integrand; and a zero-width interval.

### Validity

**The Gauss–Kronrod tables were checked against netlib as f64 bit
patterns**, by a script that parses the Fortran `data` statements and the
Rust `const` arrays independently of the crate: `XGK15`/`WGK15`/`WG15`
(8/8/4 values) and `XGK21`/`WGK21`/`WG21` (11/11/5) → `MISMATCHES: 0`.
That check is what caught the poisoned baseline described in Findings;
it owes nothing to the code under test.

**Seventeen mutations against `rust/src/quad.rs`**, each applied alone
from a green baseline and reverted after, with the baseline re-asserted
green at the end:

| # | Mutation | `cargo test` | `pytest` |
| --- | --- | --- | --- |
| M1 | `XGK21[0]` digit | caught (4) | caught (16) |
| M2 | `WGK15` centre weight digit | caught (2) | caught (2) |
| M3 | `qk21` Gauss sum uses one node of the pair | caught (3) | caught (21) |
| M4 | `qk21` Kronrod weight index off by one | caught (9) | caught (37) |
| M5 | `qk21` error estimate drops the 3/2 power | — | caught (6) |
| M6 | filter keeps endpoint-coincident points | caught (1) | caught (11) |
| M7 | filter does not deduplicate | caught (1) | caught (1) |
| M8 | filter does not sort | caught (1) | caught (1) |
| M9 | empty `points` falls back to `qagse` | — | caught (3) |
| M10 | `qelg` drops the table truncation | — | caught (1) |
| M11 | `qelg` error estimate drops one of three terms | — | caught (4) |
| M12 | `qagpe` ignores the `ndin` flag | — | caught (1) |
| M13 | `qagpe` drops the sign of a reversed interval | caught (1) | — |
| M14 | `quad` drops the flip negation | caught (1) | caught (1) |
| M15 | `qagse` `neval` formula | caught (1) | caught (8) |
| M16 | `qagse` roundoff threshold `0.99` → `0.9` | — | caught (1) |
| M17 | `ier` code 1 mapped to the wrong variant | caught (1) | caught (1) |

M13 is caught by `cargo` only, and correctly so: `quad` orders the limits
before it calls `qagpe`, so the Fortran's own `sign` branch is
unreachable from Python and only the Rust unit test can reach it. M12 and
M16 survived the first pass entirely — the two tests in
`TestAdaptiveHeuristics` were written afterwards, against inputs found by
searching with each mutation in place, and both mutations were re-run
against the final tree to confirm they are now caught.

### Deferred

- No Rust unit test pins the `abserr` refinement (M5) or `qelg`'s
  bookkeeping (M10, M11): the Python scipy comparison catches all three,
  and an independent Rust-side expectation would have to reimplement the
  formula it is checking. `cargo test` alone is therefore not a complete
  gate on this module — noted here rather than papered over.

## Open Questions

- None blocking. One observation for Phase 04–06: the port and scipy can
  disagree by ~10% once `limit` is exhausted (see Findings). No live call
  site reaches that regime today and `test/test_core_quad.py` asserts
  `ier = 0` for every live shape, so a future kernel that *does* reach it
  would be a new behavior rather than a regression — but it would be a
  silent one, since QUADPACK returns a number either way.

## Plan Impact

**Impact Level:** Phase file patched (plus the reference).

`../../phases/phase-03-numerics-foundation.md`'s Task 3.3 block gained
four "criteria added during execution" bullets, for the same reason Task
3.2's did: the criteria as written were satisfiable in a way that would
have been wrong, and the shape of the answer belongs in the canonical
document rather than only here. The breakpoint contract is scipy's
rather than QUADPACK's and both live degeneracies are discards (so five
of the twelve call sites run `qagpe` with an empty list); only `qk21` is
on the live
path; the agreement criterion is met with four orders of headroom and its
boundary is `limit`; and the corpus's served-kernel predicate stays sound
through the existing exemption.

`../../references/numerics-replacements.md`'s quad section gained the
measured contract, because its own sentence — "What scipy does … must be
pinned empirically" — is the instruction, and leaving the answer only in
a task note would make the next reader re-derive it.

No ADR: nothing revises ADR-0002 (the provenance is exactly what it
prescribes, public-domain netlib QUADPACK), no interface or task ordering
moves, and every decision here is carried by the code, the phase file and
this note.

## Stale-state sweep

Run against the final branch. `## Verification` holds the gates' full
results; this block is the evidence each command ran.

**Full change inventory** — `git status --short` (with `git add -N`, so
untracked files appear and were each read end-to-end):

```text
 M hazma/_core.pyi
 M projects/cython-to-rust/phases/phase-03-numerics-foundation.md
 M projects/cython-to-rust/references/numerics-replacements.md
 M projects/cython-to-rust/task-notes/README.md
 M projects/cython-to-rust/task-notes/phase-03/README.md
 A projects/cython-to-rust/task-notes/phase-03/task-3.3-quadpack.md
 M rust/src/lib.rs
 A rust/src/quad.rs
 A rust/src/quad_probe.rs
 M test/parity/README.md
 M test/parity/cases.py
 M test/parity/test_parity.py
 A test/test_core_quad.py
```

`git diff origin/master --stat` → `13 files changed, 4015 insertions(+),
35 deletions(-)`.

**Numerical-impact statement** — `git diff origin/master --stat --
hazma`:

```text
 hazma/_core.pyi | 18 +++++++++++-------
 1 file changed, 11 insertions(+), 7 deletions(-)
```

One file, and it is the non-executable stub: the hunk rewrites a comment
block describing which `hazma._core` submodules are deliberately
unstubbed. No executable line under `hazma/` is reachable from this diff,
so no grid evaluation applies. The positive evidence is the parity corpus
running in **bit-equality mode** inside the bare suite — `rtol = 0`
across all 41 consumed entry points and 179,695 pinned values, at
`1212 passed, 13 skipped` with the skip count unchanged (see
`## Verification`).

**Doc citations** — `scripts/agents/check_doc_citations.py <the six
touched docs>`:

```text
docs scanned: 6
in-repo citations checked: 18
  resolved by exact: 10
  resolved by suffix: 8
external citations skipped: 0
out-of-range or ambiguous: NONE
```

Paths were passed explicitly rather than `--changed-vs origin/master`,
which reports `no docs to check` on an uncommitted tree — a
success-shaped line for a zero-file scan (`docs/agents/lessons.md`,
`[changed-vs-sees-only-commits]`).

**Call-site count, re-derived rather than carried over.** Every prose
copy of "eleven live call sites" and "six of the eleven" was wrong, in
seven files, and had been copied from the reference's *seven-row table*
rather than counted. Re-derived by classifying each match as live or
commented:

```sh
grep -rn "quad(" --include='*.pyx' hazma \
  | awk -F: '{l=$0; sub(/^[^:]*:[^:]*:/,"",l); gsub(/^[ \t]*/,"",l);
              print (substr(l,1,1)=="#" ? "COMMENTED" : "LIVE"), $1":"$2}'
```

→ **12 live** sites and 11 commented ones, of which **5** live sites pass
`points=[-1, 1]` (the sixth `points=[-1, 1]` match,
`hazma/spectra/_positron/_muon.pyx:134`, is commented out).

**The first pass of this sweep was itself incomplete, and review caught
it** (PR #60). It grepped the *paired* phrases `eleven` and
`six of the eleven` and fixed twelve occurrences across
`rust/src/quad.rs`, `test/test_core_quad.py`, this note,
`phase-03/README.md`, `../README.md`,
`../../references/numerics-replacements.md` and the phase file — then
recorded "all twelve occurrences were swept", which was a completeness
claim the grep could not support. A thirteenth copy survived in this
note's own `## Plan Impact` section, reading "so six call sites run
`qagpe` with an empty list": the bare number word with no `eleven`
anywhere near it, so the pattern never saw it.

Re-sweeping with a pattern keyed on the *claim* rather than on the
phrasing — every number word or numeral within 40 characters of
`call site` / `live call` / `points=[-1` / `qagpe with an empty`, in
either order — then turned up a **fourteenth**, in
`test/test_core_quad.py:307` ("the branch all six `points=[-1, 1]` call
sites take"). Both stragglers are the same shape as the twelve that were
found: a count copied into prose that no longer sits beside the number it
was copied from. **Sweep the bare numeral and the number word against the
claim, not the phrase that happened to carry them** —
`docs/agents/lessons.md`, `[sibling-copies-of-a-fixed-claim]` and
`[derived-count-not-rederived]`. The final pattern, and its output, are
in the review-response record below.

**Stale-sibling sweep on the submodule prose** — `rg -n 'per-domain
submodule|five per-domain|sixth submodule' rust/ test/ hazma/ docs/`:

```text
rust/src/kernels.rs:7://! [`crate::dispatch`] and the per-domain submodules.
rust/src/special_probe.rs:3://! Registration only, like the per-domain submodules. These three
rust/src/quad_probe.rs:3://! Registration only, like the per-domain submodules. This is a **test
rust/src/lib.rs:3://! One `cdylib`, five per-domain submodules, built against CPython's
hazma/_core.pyi:6:# per-domain submodules — photon, positron, neutrino, scalar_mediator,
test/parity/test_parity.py:265:    per-domain submodules are empty until Phase 04; and `special`
```

All six still true — `quad` is not a per-domain submodule, and `lib.rs`,
`_core.pyi` and `test_parity.py` each name it separately in the same
file. The last three were updated by this task; Task 3.2's "sixth
submodule" wording in `_core.pyi` was rewritten rather than incremented
to a seventh, because the count was the fragile part.

**Predicate references** — `rg -n '_CORE_TEST_ONLY_MODULES' test/ docs/
hazma/` returns 9 live hits across `cases.py`, `test_parity.py` and
`test/parity/README.md`; each was read and reconciled with the new
exemption. Occurrences under `projects/` are dated Phase 01–03 records
and were left alone.

**Forbidden tokens** — `grep -nE 'TODO|FIXME|breakpoint\(|import
pdb|^\s*print\(' rust/src/quad.rs rust/src/quad_probe.rs
test/test_core_quad.py` → no matches; preflight's own diff scan agrees
(`PASS forbidden tokens — none added`).

**Measurement re-derivation.** The port-vs-scipy agreement figures were
measured twice. The first sweep (6,000 combinations, `points` drawn from
five options of at most two break points) reported *zero* subdivision
mismatches among 2,812 converged runs, worst relative value 9.4e-11 and
worst `|Δ|/tolerance` 2.0e-3 — and those numbers were already written
into this note, `phase-03/README.md`, `../README.md`, the phase file and
the test module's docstring before the design was questioned. Adding 9-
and 39-point break-point grids to the draw produced 5 mismatches in
4,461. Every copy was swept to the wider sweep's figures (11,274 runs;
4,461 converged; 5 mismatches; 3.6e-2 of tolerance; 8.2e-11 relative) —
`docs/agents/lessons.md`,
`[measurement-taken-before-the-task-ended]`, second shape.

**Table provenance.** The Gauss–Kronrod literals were re-verified against
the netlib Fortran as f64 bit patterns after every mutation run, not only
at the end: `MISMATCHES: 0` over all 47 values.

## Handoff to Next Task

- **Read first:** `../../phases/phase-03-numerics-foundation.md` (Task
  3.4 and 3.5, plus Task 3.3's added criteria), then `phase-03/README.md`.
- **Safe to assume:** `hazma_core::quad` exists and is PyO3-free.
  `quad(&mut f, a, b, &QuadOpts { epsabs, epsrel, limit, points })` is
  the function a Phase 04–06 kernel calls — **not** `qagse`/`qagpe`,
  which skip scipy's argument preprocessing. It returns
  `Result<QuadOutcome, QuadError>`: `Err` only where scipy raises
  `ValueError`, and `QuadOutcome::ier` carrying what scipy would have
  warned about. `hazma._core.quad` is a test surface and must stay
  importer-free, or the parity corpus leaves bit-equality mode.
- **Risky / unknown:** the ~10% divergence once `limit` is exhausted
  (Findings, Open Questions). And when porting a call site, copy its
  `epsabs`/`epsrel`/`points` from the `.pyx` verbatim — the twelve sites
  use five different tolerance combinations, and two of them reach
  scipy's defaults by passing no keyword at all.
