# Task 3.2: Special functions

**Date:** 2026-08-09
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-03-numerics-foundation.md`
(Task 3.2), `../../references/numerics-replacements.md`
(§SciPy C-level dependencies), `../../rules.md` rule 5 (Licensing 1),
rules 6–9 (Rust conventions 1–4)
**Related ADRs:** ADR-0002 (Accepted 2026-08-04 — fixes the provenance:
cephes lineage only, nothing GSL-derived)
**Depends On:** none

## Objective

Give the Rust crate the three `scipy.special.cython_special` functions
the Cython layer cimports — `spence`, `k1`, `kn` — as a thin,
PyO3-free `rust/src/special.rs` over the cephes-lineage `spec_math`
crate, with their argument conventions and their agreement with scipy
pinned by test rather than assumed.

## Exit Criteria

Copied from `../../phases/phase-03-numerics-foundation.md` §Task 3.2:

- `spence`, `bessel_k1`, `bessel_kn` exposed via a thin
  `rust/src/special.rs` over `spec_math` (or in-tree cephes translation
  on any gap — ADR-0002 fallback).
- Convention pinned: Rust `spence`-wrapper matches
  `scipy.special.spence` (Li₂(1−z) convention) on a grid covering
  (0,1), [1,∞), z→0⁺, z=1, z=2 — rtol ≤ 1e-13.
- `k1`/`kn` swept vs scipy over the thermal domain incl. large-argument
  underflow region — rtol ≤ 1e-13.

Three criteria were **added to the phase file during execution** (see
§Plan Impact): the `kn` deviation and its justification, the bound on
where the underflow criterion can hold for `kn`, and keeping the parity
corpus's served-kernel predicate sound.

## Inputs Reviewed

- `../../PLAN.md` (Scope, Numerical impact, Dependencies).
- `../../phases/phase-03-numerics-foundation.md` (Goal, Prerequisites,
  Task 3.2).
- `../../rules.md` (all sections; rule 5 governs provenance headers,
  rule 8 the PyO3-free kernel layering).
- `../README.md` and `README.md` (project and phase working memory).
- `../../references/numerics-replacements.md` §"SciPy C-level
  dependencies being replaced" — the call-site table and the two
  mandatory implementation-time checks.
- `hazma/spectra/_photon/_muon.pyx` (the `spence` call site, lines 13
  and 113), `hazma/scalar_mediator/_c_scalar_mediator_cross_sections.pyx`
  (`k1` at 1361, `kn` at 1404) and
  `hazma/vector_mediator/_c_vector_mediator_cross_sections.pyx`
  (`k1` at 606, `kn` at 650).
- `rust/src/{lib,dispatch,constants,photon}.rs` — crate layering and the
  registration pattern.
- `test/parity/{cases.py,test_parity.py,README.md}` — the served-kernel
  predicate.
- `docs/agents/lessons.md`, `docs/agents/environment.md`.
- `spec_math` 0.1.6 source under
  `~/.cargo/registry/src/index.crates.io-*/spec_math-0.1.6/`
  (`src/lib.rs`, `src/cephes64/{spence,k1,kn}.rs`).

## Findings

- **`scipy.special.kn` is not cephes `kn`, and the difference is four
  orders past this task's gate.** scipy dispatches integer orders to
  `kv`; only `k0`/`k1` are still cephes there. `spec_math`'s faithful
  cephes `kn` — the obvious implementation, and the one the plan
  assumed — misses `scipy.special.kn(2, ·)` by up to **5.1e-9**
  relative over `x ∈ [1e-8, 300]`, peaking at `x = 9.531` on the low
  side of that routine's own `x = 9.55` branch switch. The plan's
  fallback ("vendor a direct Rust translation of the specific cephes
  routine") would have reproduced the same miss, because the gap is in
  scipy rather than in `spec_math`. This matters concretely: the
  mediator prefactor is `x/(2·kn(2, x))²`, so the miss enters
  `thermal_cross_section` **squared**, right at the parity corpus's
  1e-8 budget for that function. A Phase 05 swap that used cephes `kn`
  could have landed inside the budget and moved published numbers.
- **The fix is a recurrence on the two routines scipy *does* still take
  from cephes.** `K_{m+1}(x) = K_{m-1}(x) + (2m/x)·K_m(x)`
  (DLMF 10.29.1), seeded on `k0`/`k1`, upward — the stable direction for
  `K`. Measured against `scipy.special.kn`: **≤ 3.4e-15** for every
  order n = 0..5 over `x ∈ [1e-6, 300]`, and **9.786e-16** at the n = 2
  hazma uses.
- **`spence` and `k1` need no such care** — both sides are the same
  cephes routine, and they agree to a few ulp (2.425e-15 and 1.215e-15).
  `spec_math`'s `Polylog::li2` does expose scipy's `Li₂(1−z)`
  convention, because its body is `cephes64::spence`. **The name is the
  trap, not the function**: a kernel author reading `li2` has every
  reason to think it means `Li₂(z)`, and `dnde_photon_muon` subtracts
  two of these, so the wrong convention returns a smooth, plausible,
  wrong spectrum rather than an error.
- **The `cython_special` C symbols and the `scipy.special` ufuncs are
  bit-identical** for all three functions (checked through
  `__pyx_capi__` capsules over the same grids). That is what makes a
  `scipy.special`-based test a parity gate rather than a plausibility
  check, and it is asserted rather than assumed —
  `TestOracleIdentity`. Resolving the capsules has one gotcha worth
  carrying: `spence` and `kn` are *fused-type* exports, mangled
  `__pyx_fuse_<i><name>` with `i` depending on declaration order, so the
  test matches on the capsule's own signature string rather than on a
  hardcoded index. (`k1` is unfused and exported under its bare name.)
- **The underflow tails of `k1` and `kn` behave differently, and only
  `kn`'s diverges.** `k1`: both sides decay into the subnormals with no
  explicit cutoff and reach zero together at `x = 742.09`; agreement
  through the whole tail is 3.078e-16. `kn`: scipy inherits `kv`'s
  conservative exponent limit and flushes to zero from `x = 697.88`,
  where `K₂` is still `3.9e-305` — **a normal double**, so scipy is
  discarding about three decades of representable values, not rounding
  an already-lost one. The recurrence keeps going to `x = 742.09`. On
  that ~44-wide window the two disagree wholesale. Unreachable from
  hazma (`thermal_cross_section` short-circuits above `x = 300`, where
  `K₂ ≈ 3.7e-132`), and pinned rather than only documented.
- **A Python-visible test surface on `hazma._core` reads as a started
  port.** `cases.rust_core_kernels()` counts every public callable on
  the extension except the literal name `roundtrip`, so registering
  `hazma._core.special` immediately flipped the parity corpus out of
  bit-equality mode — `Provenance(exact=False, detail='hazma._core
  serves 3 kernel(s)')` — for the whole of Phases 03–06, with nothing
  turning red. Same class Task 2.1 fixed once already
  (`docs/agents/lessons.md`, `[gate-disabled-stays-green]`); the
  predicate's own docstring anticipated it ("something non-kernel became
  public on the extension and needs adding to
  `cases._CORE_SCAFFOLD_NAMES`"). **The general shape: a gate whose
  exemption list is keyed on names will be widened by the next thing
  that is not a kernel, and widening it is indistinguishable from
  disabling it unless the exemption is conditional on a checkable
  property of the tree.**

## Decisions and Implementation Notes

- **`bessel_kn` is a recurrence, not `spec_math::bessel_kn`** — the
  measurement above. Rejected alternatives: (a) cephes `kn` as-is
  (5.1e-9 miss); (b) vendoring cephes `kn.c` per ADR-0002's fallback
  (same miss, more code); (c) reproducing `kv`'s underflow cutoff so the
  tail matches scipy exactly (would mean reverse-engineering an AMOS
  exponent limit in order to return `0` where the true value is
  `3.9e-305` — fake precision, and outside hazma's domain either way).
  The recurrence is original work over cephes seeds, so ADR-0002's
  provenance rule is satisfied without a new dependency.
- **General `n` is carried even though only `n = 2` is live.** The
  recurrence is general anyway, and the order sweep at n = 0..5 is what
  catches a dropped `m` factor — at n = 2 the factor `2m/x` is `2/x`,
  so the error is invisible. Both the Rust Wronskian test (ν = 1 and
  ν = 2) and the Python order sweep exist for that reason; the ν = 2
  case was added after a mutation showed `cargo test` alone missed it.
- **`rust/src/special.rs` is PyO3-free (rule 8); `rust/src/special_probe.rs`
  is the Python half.** The probe registers `hazma._core.special` with
  three `#[pyfunction]`s routed through `dispatch::map_unary`, so all
  three take a scalar or a 1-D `float64` array under the same contract
  every ported kernel will use. Array support is not decoration: the
  sweeps are 8k–25k points each, and a Python-level loop at that size is
  the kind of test that gets quietly trimmed to a dozen points later.
- **The probe exists only because the oracle lives in Python.** Phase
  04–06 kernels will call `crate::special` directly in Rust. Nothing
  under `hazma/` imports the submodule, and that is now a test
  (`test_test_only_core_submodules_have_no_importer`) rather than an
  intention.
- **`_CORE_TEST_ONLY_MODULES` exempts a submodule, not a name.**
  Adding `spence`/`bessel_k1`/`bessel_kn` to `_CORE_SCAFFOLD_NAMES`
  would have worked and would have been wrong: it would exempt those
  names anywhere on the extension, including a future real kernel. The
  submodule-level exemption is narrower, and it is paired with the
  importer check so it cannot quietly become a hole.
- **`hazma/_core.pyi` gains a comment, not a stub.** A `.pyi` cannot
  nest submodules, and stubbing `special` would advertise a surface the
  package does not mean to offer. The comment says why.
- **`clippy::excessive_precision` did not fire this time** — unlike
  Task 3.1's constants, nothing here transcribes a literal; the two
  written-out constants (`π²/6`, `π²/12`) are in test code at full
  double precision and are accepted.

## Files Changed

- `rust/src/special.rs` — **new.** `spence`, `bessel_k1`, `bessel_kn`
  over `spec_math`, with a `# Sources and licensing` provenance header,
  the call-site table, the `Li₂(1−z)` convention note, the `kn`
  deviation rationale, and 7 unit tests (including an independent
  `Iₙ` power series so the `K` Wronskian is not a restatement of the
  recurrence).
- `rust/src/special_probe.rs` — **new.** Registration-only module
  exposing the three as `hazma._core.special` for the scipy sweep.
- `rust/src/lib.rs` — `pub mod special;` + `mod special_probe;`, the
  `add_submodule(module, "special", …)` call, and the paragraph on why
  `special` is `pub` and why the probe is the exception to
  "registration-only means per-domain".
- `rust/Cargo.toml` — `spec_math = "0.1.6"` with the licensing/provenance
  comment; `rust/Cargo.lock` updated.
- `test/test_core_special.py` — **new**, 53 tests in 7 classes.
- `test/parity/cases.py` — new `_CORE_TEST_ONLY_MODULES`; the
  served-kernel walk skips those submodules.
- `test/parity/test_parity.py` — new
  `test_test_only_core_submodules_have_no_importer`;
  `test_scaffolded_core_serves_no_kernels`'s docstring reconciled.
- `test/parity/README.md` — the served-kernel paragraph names the second
  exemption and the check that keeps it honest.
- `hazma/_core.pyi` — comment recording the unstubbed `special`
  submodule and why.
- `projects/cython-to-rust/phases/phase-03-numerics-foundation.md` —
  three exit criteria added (§Plan Impact).
- `projects/cython-to-rust/references/numerics-replacements.md` — the
  measured Task 3.2 block, correcting the doc's claim that scipy's `kn`
  is a cephes wrapper.
- Project bookkeeping: this note, `README.md` (phase working memory),
  `../README.md` (project working memory).

## Verification

Environment: CPython 3.12.12, macOS/arm64, `numpy==2.5.1`,
`scipy==1.18.0`, `cython==3.2.9` — the corpus's capturing environment,
built per `../README.md`'s pinned recipe, so the parity suite runs in
bit-equality mode (`Provenance(exact=True, detail='')`, checked before
and after). `python -c "import hazma._core; print(hazma._core.__file__)"`
resolves inside the worktree; the tree was rebuilt with
`uv pip install -e . --no-build-isolation` after the last `.rs` edit and
after each mutation below.

- `pytest test/test_core_special.py -q` → **`53 passed in 0.26s`**
  (53 collected). Coverage by class:
  - `TestOracleIdentity` (3) — the `cython_special` C symbols equal the
    `scipy.special` ufuncs, bit for bit, on each function's grid.
  - `TestSpenceConvention` (3) — closed forms at z = 0, 1, 2; the
    `Li₂(1−z)`-vs-`Li₂(z)` discriminator at z = 0.25/0.75 against an
    independent series; the reflection identity.
  - `TestSpenceAgreement` (5) — the exit-criterion grid, segment by
    segment, plus bit-equality at the named points.
  - `TestSpenceEdges` (5) — negative, ±∞, NaN.
  - `TestBesselK1` (8) — thermal-domain sweep, underflow tail, the
    subnormal-before-zero property, edges, NaN.
  - `TestBesselKn` (13) — live-domain sweep, orders 0–5, negative-order
    folding, the cephes discriminator, edges, NaN.
  - `TestBesselKnUnderflowTail` (4) — the declared divergence and both
    flush points.
  - `TestDispatch` (12) — array path equals scalar path bit for bit,
    scalar→float, empty array, 2-D `ValueError` wording.
- `cargo test --manifest-path rust/Cargo.toml --no-default-features` →
  **`15 passed`** (7 new in `special::tests`): `spence` closed forms,
  `spence` reflection, `spence` edges, the `K₁` Wronskian, `k1` edges,
  the `K` Wronskian at ν = 1 and ν = 2, `kn` at its seed orders, `kn`
  edges.
- `cargo fmt --check` and
  `cargo clippy --all-targets -- -D warnings` — clean.
- `black --check` / `isort --check-only` / `ruff check` (configured
  form) over the four touched Python files — clean. `ruff check` on
  `test/parity/{cases,test_parity}.py` was clean on the trunk too
  (checked via `git stash`), so this is a zero-delta.
- `scripts/agents/preflight.sh --paths "test/test_core_special.py
  test/parity/cases.py test/parity/test_parity.py hazma/_core.pyi"
  --md "<the six touched docs>"` on the final tree:

  ```text
  PASS   black --check           <the four Python files>
  PASS   isort --check-only      <the four Python files>
  PASS   ruff check              <the four Python files>
  PASS   cargo fmt --check       rust/
  PASS   cargo clippy            rust/
  PASS   cargo test              rust/
  PASS   pytest                  1142 passed, 13 skipped, 5 warnings in 645.29s
  PASS   import hazma            version 2.1.0
  PASS   markdownlint            <the six touched docs>
  SKIP   version bump            not a closing PR (pass --closing)
  PASS   forbidden tokens        none added
  RESULT: PASS
  ```

  Two earlier runs came back `FAIL markdownlint` — first an over-long
  line in `test/parity/README.md`, then hard tabs this note had picked
  up from a pasted `git diff --numstat`. Both are fixed above.

  The only file edited after that run is **this note**, to paste the row
  table in; of the eleven gates only `markdownlint` reads it, and it was
  re-run against the final note directly (clean). Nothing under
  `hazma/`, `rust/` or `test/` moved, so the other ten rows still
  describe the tree being handed off.

**Measured agreement with scipy** (the exit criteria, in numbers):

| Sweep | Points | Max rel. error | At |
| --- | --- | --- | --- |
| `spence`, exit-criterion grid | 11,003 | **2.425e-15** | x = 0.524 |
| `k1`, thermal domain `[1e-8, 690]` | 20,402 | **1.215e-15** | x = 1.7129 |
| `k1`, underflow tail `[690, 745]` | 1,895 | **3.078e-16** | x = 695.28 |
| `kn(2, ·)`, live domain `[1e-8, 300]` | 22,002 | **9.786e-16** | x = 1.92889 |
| `kn(n, ·)`, n = 0..5, `[1e-6, 300]` | 5,001 each | **4.007e-15** (worst, n = 0) | x = 1.97292 |
| `kn(2, ·)`, `[600, 697.88)` | 2,000 | **5.734e-14** | deep subnormals |

Rejected implementation, measured on the same two `kn` grids after
rebuilding the extension with the substitution in place: `spec_math`'s
cephes `bessel_kn` → **5.055e-9** at x = 9.531 on the live-domain grid,
and 6.213e-9 (n = 0) / 2.650e-9 (n = 2) across the order sweep — every
one of them just below cephes `kn`'s own `x = 9.55` branch switch.

**Test-validity campaign.** Eight mutations of `rust/src/special.rs`,
each rebuilt into the tree and run against both suites:

| # | Mutation | Caught by |
| --- | --- | --- |
| 1 | `spence` returns `Li₂(z)` | 4 cargo + 9 pytest (`TestSpenceConvention`, both agreement classes, edges) |
| 2 | `bessel_kn` falls back to cephes `kn` | 2 cargo + 9 pytest, incl. `test_beats_cephes_kn_at_its_worst_argument` |
| 3 | recurrence seeds swapped | 1 cargo + 7 pytest |
| 4 | recurrence drops the order factor `m` | 1 cargo (ν = 2 Wronskian) + 3 pytest (orders 3–5 only) |
| 5 | `bessel_k1` returns `K₀` | 3 cargo + 2 pytest |
| 6 | one extra recurrence step (`K₃` for `K₂`) | 1 cargo + 7 pytest |
| 7 | negative order no longer folded | 1 cargo + 1 pytest |
| 8 | `bessel_kn(0, ·)` returns `K₁` | 1 cargo + 1 pytest |

Mutation 4 is the reason the ν = 2 Wronskian exists: on the first pass
`cargo test` passed it, because at n = 2 the dropped factor is 1.

Two further mutations against the corpus guard:

| # | Mutation | Caught by |
| --- | --- | --- |
| A | a `hazma/` module imports `hazma._core.special` | `test_test_only_core_submodules_have_no_importer` (names the file and line) |
| B | `_CORE_TEST_ONLY_MODULES` emptied | `test_scaffolded_core_serves_no_kernels` + 2 siblings |

**Deferred:** nothing. `bessel_k0`/`bessel_k1e`/`bessel_k0e` are in
`spec_math` and are not exposed — no Cython call site cimports them, and
rule 1 gives no parity to preserve for a function nothing calls.

## Open Questions

- Should the crate's `kn` reproduce scipy's early flush-to-zero so the
  two agree everywhere? Answered *no* for this project (it would return
  `0` where the true value is a normal double, and hazma stops at
  `x = 300`), but it is the sort of thing a later consumer outside hazma
  would want to know. Recorded in `rust/src/special.rs` and pinned in
  `test/test_core_special.py`; no follow-up filed, because there is no
  deferred *work* — only a documented, tested boundary.
- Nothing else. `../README.md`'s standing question about
  `spec_math::li2`'s convention vs `scipy.special.spence` is **resolved**
  by this task: they are the same convention, because `li2` delegates to
  `cephes64::spence`.

## Plan Impact

**Impact Level:** Phase file patched (no ADR).

`../../phases/phase-03-numerics-foundation.md` §Task 3.2 gained three
"Criteria added during execution" bullets, because the criteria as
written could not all be met as written and one implicit assumption in
them was false:

1. The first criterion's parenthetical anticipated a gap in `spec_math`
   and prescribed an in-tree cephes translation. The gap was in
   **scipy**, and a cephes translation would have reproduced the miss —
   so the realized answer (a recurrence on cephes `k0`/`k1`) is now
   named, with its measurement.
2. The third criterion's "incl. large-argument underflow region — rtol
   ≤ 1e-13" is **unachievable for `kn` above `x ≈ 698`** and always
   will be, since scipy returns `0` there. The criterion is now bounded
   at scipy's flush point, with the divergence declared and the boundary
   pinned.
3. Keeping the parity corpus in bit-equality mode was a real deliverable
   of this task, not an incidental fix, and the plan said nothing about
   it — same widening Task 2.1 recorded for its own exit criteria.

`../../references/numerics-replacements.md` was patched too: its
sentence "scipy's `spence`, `k1`, `kn` are themselves cephes wrappers"
is false for `kn`, and that sentence is what made cephes `kn` look like
the safe choice. No ADR: nothing here revises ADR-0002 (the recurrence
is original work over cephes seeds, so the provenance rule holds), no
interface or ordering changes, and the decision is a per-function
implementation choice fully carried by the code, the phase file and this
note.

## Stale-state sweep

Run against the final branch. `## Verification` holds the gates' full
results; this block is the evidence each command ran.

**Full change inventory** — `git status --short` (staged, so untracked
files appear and were each read end-to-end):

```text
M  hazma/_core.pyi
M  projects/cython-to-rust/phases/phase-03-numerics-foundation.md
M  projects/cython-to-rust/references/numerics-replacements.md
M  projects/cython-to-rust/task-notes/README.md
M  projects/cython-to-rust/task-notes/phase-03/README.md
A  projects/cython-to-rust/task-notes/phase-03/task-3.2-specfun.md
M  rust/Cargo.lock
M  rust/Cargo.toml
M  rust/src/lib.rs
A  rust/src/special.rs
A  rust/src/special_probe.rs
M  test/parity/README.md
M  test/parity/cases.py
M  test/parity/test_parity.py
A  test/test_core_special.py
```

`git diff origin/master --stat` → `15 files changed, 1716 insertions(+),
38 deletions(-)`.

**Numerical-impact statement** — `git diff origin/master --stat --
hazma`:

```text
 hazma/_core.pyi | 8 ++++++++
 1 file changed, 8 insertions(+)
```

Eight added lines, zero removed, in a comment block of a non-executable
stub. No executable line under `hazma/` is reachable from this diff, so
no grid evaluation applies; the parity corpus in bit-equality mode
inside the bare suite is the positive evidence (see `## Verification`).

**Doc citations** —
`scripts/agents/check_doc_citations.py <the six touched docs>`:

```text
docs scanned: 6
in-repo citations checked: 18
  resolved by exact: 10
  resolved by suffix: 8
external citations skipped: 0
out-of-range or ambiguous: NONE
```

Paths were passed explicitly rather than `--changed-vs origin/master`,
which reported `no docs to check` on the uncommitted tree — a
success-shaped line for a zero-file scan
(`docs/agents/lessons.md`, `[changed-vs-sees-only-commits]`).

**Stale-sibling sweep** — `rg -n 'five submodule|five per-domain|per-domain
submodule' rust/ test/ hazma/ docs/`:

```text
rust/src/kernels.rs:7://! [`crate::dispatch`] and the per-domain submodules.
rust/src/special_probe.rs:3://! Registration only, like the per-domain submodules. These three
rust/src/lib.rs:3://! One `cdylib`, five per-domain submodules, built against CPython's
hazma/_core.pyi:6:# per-domain submodules — photon, positron, neutrino, scalar_mediator,
test/parity/test_parity.py:265:    per-domain submodules are empty until Phase 04; and `special`
```

All five still true: `special` is not a per-domain submodule, and both
`lib.rs` and `_core.pyi` name it separately in the same file.
`test_parity.py:265` was updated by this task. The remaining repo-wide
hits are under `projects/`, in dated Phase 02 records.

**Predicate references** — `rg -n
'rust_core_kernels|_CORE_SCAFFOLD_NAMES|_CORE_TEST_ONLY_MODULES' test/
docs/` returns 20 live hits across `cases.py`, `test_parity.py`,
`tolerances.py` and `test/parity/README.md`; each was read and
reconciled with the new exemption. Occurrences under `projects/` are
dated Phase 01/02 records and were left alone.

**Forbidden tokens** — `rg -n 'TODO|FIXME|breakpoint\(|import pdb|^\s*print\('
rust/src/special.rs rust/src/special_probe.rs test/test_core_special.py`
→ no matches; preflight's own diff scan agrees (`PASS forbidden tokens
— none added`).

**Measurement re-derivation.** The cephes-`kn` miss was first measured
on an exploratory grid (5.176e-9 at x = 9.4925) and re-measured on the
grid the committed tests actually use (5.055e-9 at x = 9.531) by
rebuilding the extension with the substitution in place. Every one of
the eleven copies of that figure across seven files was swept to the
re-derived value — `docs/agents/lessons.md`,
`[measurement-taken-before-the-task-ended]` and
`[sibling-copies-of-a-fixed-claim]`.

## Handoff to Next Task

**Read first:** `../README.md` (project working memory) → `README.md`
(this phase) → `../../phases/phase-03-numerics-foundation.md`, whose
Task 3.2 block now carries three criteria that were not in the original
plan.

**Now safe to assume:**

- `hazma_core::special::{spence, bessel_k1, bessel_kn}` exist, are
  PyO3-free, and track `scipy.special` to ≤ 4.0e-15 over every domain
  hazma reaches. **Phase 05's `thermal_cross_section` port calls these
  directly in Rust** — do not go through `hazma._core.special`, which is
  a test surface and must stay importer-free or the parity corpus's
  bit-equality mode dies.
- `spence` is scipy's convention, `Li₂(1−z)`. The muon photon kernel
  (Phase 04) wants `special::spence(xm) - special::spence(xp)` with the
  same arguments the `.pyx` passes — no reflection.
- Registering anything public on `hazma._core` that is *not* a ported
  kernel now has a documented mechanism and a guard test; use
  `_CORE_TEST_ONLY_MODULES`, and expect
  `test_test_only_core_submodules_have_no_importer` to hold you to it.

**Still risky / unknown:**

- **Do not "simplify" `bessel_kn` to `spec_math`'s.** It is a 5.1e-9
  miss that the parity corpus's 1e-8 `thermal_cross_section` budget
  would absorb. The discriminator test names the exact substitution.
- `derived::positron_pion::ENG_MU_PI_RF` vs
  `derived::photon_pion::ENG_MU_PIRF` (Task 3.1) is still the sharpest
  trap in this phase; unrelated to this task but unchanged.
- Task 3.3 (QUADPACK) remains the phase's fiddliest item, and it now
  has a precedent to reuse: **do not design against the documentation of
  what scipy is supposed to do — measure what it does.** This task's
  whole `kn` finding, and 3.3's own breakpoint-preprocessing criterion,
  are the same lesson.
