# Task 4.1: `_positron/_muon` — the walking-skeleton kernel swap

**Date:** 2026-08-11
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-04-spectra-kernels.md` (Goal, Part 1,
Task 4.1); `../../rules.md` rules 1–3 (parity discipline), 4 (constants),
6–9 (Rust conventions)
**Related ADRs:** ADR-0001 (framework), ADR-0002 (license-clean numerics)
**Depends On:** Phase 03 complete (constants, boost, dispatch)

## Objective

Move `dnde_positron_muon` from Cython to `hazma._core`, gated by the parity
corpus, and in doing so establish the per-kernel recipe Tasks 4.2–4.6 and
Phases 05–06 copy.

## Exit Criteria

From the phase file's Task 4.1 block:

- `dnde_positron_muon` on Rust; corpus exact-or-≤1e-13; wrapper swapped.
  (Twin is a Phase 06 capi survivor.)
- Establishes the per-kernel PR template — port → corpus diff → swap →
  delete-or-defer → drift note — later tasks copy.

Both met. The corpus landed at the *exact* end of that range: the declared
budget for `spectra.positron.muon` is `EXACT_RTOL = 0.0`, so the swap was
gated at bit-equality, not at 1e-13.

## Inputs Reviewed

- `../../PLAN.md`; `../README.md`; `../../task-notes/README.md` (Findings,
  Numerical impact so far, Handoff); `../../rules.md`.
- `../../phases/phase-04-spectra-kernels.md` — Goal (the capi-survivor
  exception) and Task 4.1.
- `hazma/spectra/_positron/_muon.pyx`, `_muon.pxd`, `__init__.py`.
- `rust/src/{dispatch,boost,constants,kernels,positron}.rs`.
- `test/parity/{cases,generate,tolerances,test_parity}.py`;
  `test/test_core_{dispatch,boost}.py`.
- [`docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md`](../../../../docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md)
  — read before starting, per the project handoff.
- `docs/agents/lessons.md`, `docs/agents/environment.md`.

## Findings

- **The shipped kernel divides by its normalization where it should
  multiply, and every positron spectrum in hazma is 0.0374% low.**
  `R_FACTOR = 1.0001870858234163` is, as its comment says, the *reciprocal*
  of the un-normalized Michel integral: `scipy.integrate.quad` puts that
  integral at `0.999812949171142` against `1/R_FACTOR = 0.999812949171142`
  and the closed form `1 − 8r² + 8r⁶ − r⁸ − 12r⁴ln(r²) = 0.9998129491711419`
  — all three agree to the last digit. Normalizing therefore means
  multiplying by `R_FACTOR`; `_muon.pyx` divides, in both the rest-frame and
  the in-flight expression, so `∫ dN/dE dE = 1/R_FACTOR² = 0.9996259` at
  every parent energy instead of 1. **The sibling settles that this is an
  inversion rather than a convention:**
  `hazma/spectra/_neutrino/_muon.pyx` declares the identical constant
  (`:23`) and *multiplies* by it (`:58`, `:114`). It propagates —
  `dnde_positron_charged_pion` integrates to `0.999623` at `E_π = 500 MeV`
  — and so reaches both mediator positron spectra and every positron-based
  limit. Reproduced per rule 1, pinned in both languages, filed as
  [`docs/followups/todo/positron-muon-spectrum-normalization-inverted.md`](../../../../docs/followups/todo/positron-muon-spectrum-normalization-inverted.md),
  blocked behind Phase 06 Task 6.4 like its Task 3.4 sibling. **This is the
  second live 2.1.0 numerical defect the port has surfaced by writing an
  analytic test the original never had**, and both were found the same way:
  by asserting a property of the physics rather than a property of the code.
- **The FMA map is readable straight out of the shipped object code, and
  guessing it would have been wrong three ways.**
  `objdump -d hazma/spectra/_positron/_muon.cpython-312-darwin.so` shows
  exactly nine FMA instructions, all inside `dnde_positron_muon_point`
  because clang inlines both `dndx` helpers into it: two in the rest-frame
  branch (inlined twice → four instructions) and five in the boosted one.
  The three expressions that look fusable and are **not** fused are
  `x² − 4r²` and `x² − r₂₂` (an `fmul` then an `fadd` against a negated
  folded constant), `1 − β²` inside `γ²`, and `(4x)/3 + (−3 − 3r²)` — the
  division breaks the contraction. Written from the disassembly the port was
  bit-equal on the first build; there was no bisection round.
- **Reading the folded constants out of the `movk` sequences is worth doing
  and easy to get wrong.** Seven compile-time constants (`2r`, `1 + r²`,
  `4r²`, `8r²`, `−3 − 3r²`, `3 + 3r²`, `2/m_μ`) are pinned against the
  immediates the disassembly builds. The halfwords are little-endian and
  assembled high-last, and transposing two of them is exactly the kind of
  error the test then catches — it did, on the first `cargo test`.
- **A `NaN` energy does not propagate through this kernel**, in either
  language. Neither threshold comparison fires on a `NaN`, so it reaches the
  boosted branch, where `fmax`/`fmin` (`fmaxnm`/`fminnm`, and Rust's
  `f64::max`/`min`) return their *non*-`NaN` operand: both kinematic limits
  collapse onto the rest-frame support, the window survives, and
  `dnde_positron_muon(nan, 500.0)` returns `0.0034089975030640665`. The
  rest-frame branch has no `fmax`/`fmin` and does propagate. The corpus does
  not sample `NaN`, so nothing else would have caught a port that differed
  here.
- **Clippy objects to two guards that are faithful and must stay.**
  `beta < 0.0 || beta > 1.0` is *not* `!(0.0..=1.0).contains(&beta)` —
  `contains` is false for a `NaN`, so the "simplification" would return 0.0
  where the Cython falls through to the arithmetic. And
  `emu - MASS_MU < f64::EPSILON` trips `float_equality_without_abs` while
  being a genuine one-sided threshold (`emu >= MASS_MU` is already
  established). Both carry an `#[allow]` with the reason. The third finding
  — `xm > xp || xp < xm` — *was* redundant, and collapsing it to one
  comparison changes nothing including the `NaN` behavior.
- **"Does this compiler contract" is the wrong question to scope a
  bit-equality oracle on; "did *this* build produce the values" is the
  right one.** Found by CI, twice, after two green macOS runs (PR #63,
  runs 31562223329 and 31564747071). The first version of
  `test/test_core_positron_muon.py` skipped its against-the-Cython class
  by comparing the compiled kernel against an unfused Python
  transcription: agree ⇒ this build does not contract ⇒ the comparison is
  about the platform, so skip. Two things went wrong, and the second is
  the one that matters.
  1. The transcription reproduced the Cython's *operations* but not its
     *associations* — `pre * numerator / denominator` where the Cython
     divides inside `dndx_positron_muon` and multiplies by `pre` in its
     caller. Last bit different, so the probe reported "contracts"
     everywhere. Fixed, and it was not enough.
  2. With the association fixed, Linux **still** disagrees with the
     unfused reference — on a build with no `-march` flag and so no
     hardware FMA for the probe's own mechanism to explain. The cause was
     not localized (a different set of contracted expressions under gcc,
     or a libm rounding, are both consistent with the evidence) and
     **it does not need to be**: any of them breaks the comparison, and a
     probe over one mechanism cannot see the others.
  The fix is to declare the mode from the platform — read out of
  `test/parity/data/manifest.json` so this module's scope and the
  corpus's cannot drift — rather than detect it. **The clever probe was a
  worse version of a mechanism the repo already had.**
- **Then the divergence was measured, and it is small enough to gate on,
  so the off-platform mode is a budget rather than a skip.** The 16
  failures of run 31564747071 print both byte arrays in full; decoding
  them recovers 21,953 differing values, and the disagreement is rounding
  amplified by the kernel's own conditioning — median 7.2e-15, no sign
  flip, no `NaN`, no disagreement about support or zeros.

  | `emu` / MeV | max relative | max \|Δ\| / peak |
  | --- | --- | --- |
  | 105.6583745 (`m_μ`) | 4.2e-16 | 3.7e-16 |
  | 105.658374501 | 6.0e-11 | 1.9e-11 |
  | 110 | 2.7e-14 | 7.8e-16 |
  | 150 | 3.7e-13 | 5.7e-16 |
  | 500 | 6.4e-12 | 3.6e-15 |
  | 1500 | 2.2e-11 | 3.0e-14 |
  | 100000 | 1.5e-07 | 1.3e-10 |

  **The budget is scaled to the peak of the spectrum, not applied
  pointwise**, and that choice is the whole content of the fix. Pointwise
  the worst case is 1.5e-7, but it sits at a value 4.3e-4 of the peak, at
  a 100 GeV muon — three orders above this library's domain. A pointwise
  `rtol` loose enough to admit it (≥1e-6) would be loose enough to hide a
  real defect; against the peak the worst disagreement anywhere is
  1.3e-10. `OFF_PLATFORM_BUDGET = 1e-8` therefore clears every decoded
  Linux block by **at least 84×** (replayed through the shipped
  assertion), while a wrong branch or dropped term lands at O(1). Nothing
  in the module skips on any platform now: 47 tests, two modes.
- **Both regimes that amplify are the conditioning of the formula, not
  the port**: `β → 0` just off rest and `γ ≫ 1` both form `xm`/`xp` as
  `γ²(x ∓ β·root)` and then difference nearly-equal terms. Two Cython
  builds would show the same spread. This is the same population the
  corpus's own `docs/followups/todo/parity-corpus-pins-ill-conditioned-
  points.md` describes, one kernel further in.
- **The corpus budget for `spectra.positron.muon` stayed at `rtol=0`,
  and finding out why is the reason to check before loosening.** Task 4.1
  moved provenance off the capturing tree (`hazma._core` serves a kernel,
  so the kernel digest changed), which means `effective_budget` now hands
  out the *declared* budget on macOS too — and the declared budget is
  `EXACT_RTOL`, which the port meets and macOS CI proves green. Loosening
  it to match the measurement would have weakened a gate that currently
  passes, for a platform the corpus does not run on. Only the `why`
  changed: it claimed exactness followed from the kernel being closed-form
  with no `quad`, and the measurement above says the opposite — the
  closed form is ill-conditioned and the exactness is a property of the
  platform.
- **The capturing platform cannot see a bug in its own skip logic**, which
  is why both CI rounds were needed. On macOS the probe answers True
  whether or not it is correct, so every local run was green and the
  module's own tests could not distinguish a working guard from a broken
  one. This is the mirror of Task 3.4's
  `[platform-scoped-oracle-asserted-globally]`: there a platform-scoped
  claim was asserted everywhere; here the scoping mechanism itself was
  wrong, and only the other platform could say so. **Any Phase 04–06 task
  copying this module gets the declared two-mode comparison, not a
  probe** — and the budget carries its own
  `test_the_off_platform_budget_rejects_a_real_error`, because on the
  capturing platform nothing else exercises the tolerance branch and it
  would rot unnoticed.
- **A fused Python reference reproduces the shipped macOS Cython
  bit-for-bit, which confirms the FMA map from outside the disassembly.**
  Built with a correctly-rounded `fma` (`Fraction`-based, since `math.fma`
  needs 3.13 and the suite supports 3.10) at exactly the seven sites the
  Rust uses: **0 mismatches in 21,000 points** across seven parent
  energies, against 11,713 for the unfused form on the same draw. That is
  a genuine second confirmation of the `objdump` reading — and note it
  says nothing about Linux, which is the whole reason the scope is a
  platform rather than an arithmetic property.
- **The corpus case had to be repointed, or the gate would have measured the
  implementation the swap replaced.** `test/parity/cases.py` names the
  `.pyx` module for every entry point; leaving `spectra.positron.muon`
  pointed at `hazma.spectra._positron._muon` would have kept the corpus
  calling Cython while the wrapper called Rust — green, and vacuous. The
  runner keys everything by *case name*, never by `module:function`, so
  repointing disturbs no stored data. `PORTED_ENTRY_POINTS` keeps the
  origin so `assert_full_coverage` still balances, and it grew a second
  assertion: it is now an error for a ported entry point's `.pyx` to still
  export its `def`, which is rule 1's no-drift-window in code.

## Decisions and Implementation Notes

- **The capi survivors lose their Python `def`, not their file.** The phase
  Goal says the four extensions stay "built but Python-unreferenced"; the
  cheapest way to *make* that true rather than assert it is to delete the
  `def` and keep the `cdef`s, which is what the `__pyx_capi__` capsules are
  built from. The extension still builds, `_positron/_pion.pyx` and both
  mediator positron modules still `cimport` it, and no Python caller can
  reach the replaced implementation. This is the closest Phase 04 can get to
  rule 1 under its own declared exception, and Tasks 4.3/4.4 (the other two
  survivors) should do the same.
- **The per-kernel test module is not a copy of
  `test/test_core_dispatch.py`, and that reverses Task 2.3's instruction.**
  That module's docstring — written when the plan was one dispatch
  implementation per kernel — says to copy all 118 tests per swap. Task 3.5
  then replaced that with three shared helpers, so those tests now cover code
  every kernel routes through *unchanged*; transcribing them sixteen more
  times would re-test one function sixteen times and leave sixteen copies to
  keep in sync. `test/test_core_positron_muon.py` keeps what is genuinely
  per-kernel — which helper the wrapper reached for, which quantity wording
  it passed, one assertion per contract branch — and spends the rest on the
  two things only this kernel has: the `cdef` twin as an oracle, and
  physics. 47 tests rather than ~160, and the ratio is the point.
- **The comparison against the twin has two declared modes**, the Task 3.4
  lesson `[platform-scoped-oracle-asserted-globally]`: bit-for-bit on the
  platform the parity corpus was captured on (read from its manifest, so the
  two scopes cannot drift), and within `OFF_PLATFORM_BUDGET = 1e-8` of the
  spectrum's *peak* everywhere else. Nothing skips. The budget is measured
  rather than guessed and clears every observed Linux block by ≥84×; the
  peak scaling, not the figure, is what keeps it from being vacuous.
- **`hazma._core`'s per-domain submodules stay unstubbed**, and `_core.pyi`
  now says why rather than promising a stub file: `_core` is one extension,
  so a submodule stub needs a `hazma/_core/` stub *package* that would shadow
  `_core.pyi` itself. The typed surface users see is the `@overload`-annotated
  wrapper, which is what `docs/versioning.md` defines the public API against.
  Phase 07 owns the packaging and should own this with it.
- **`test_scaffolded_core_serves_no_kernels` became
  `test_the_served_roster_is_exactly_the_ported_entry_points`.** Asserting
  the roster is empty was right for Phases 02–03 and is a contradiction from
  the first swap. Its successor checks the roster against
  `cases.PORTED_ENTRY_POINTS` — one served callable per ported row, matched
  on the leaf function name — which keeps the original's real job (catching a
  non-kernel that became public on the extension) while growing with the
  port. `test_a_served_kernel_is_found_and_blocks_regeneration` was rewritten
  to measure a *delta* on the live roster for the same reason.

## Files Changed

- `rust/src/kernels/positron_muon.rs` — **new.** The ported kernel:
  `dndx_rest_frame`, `dndx`, `dnde_positron_muon`, seven folded constants,
  and 9 `cargo test` units.
- `rust/src/kernels.rs` — declares `pub mod positron_muon` (one submodule per
  ported `.pyx`).
- `rust/src/positron.rs` — registers `dnde_positron_muon` through
  `dispatch::map_unary` with the quantity wording `"Positron energies"`.
- `hazma/spectra/_positron/__init__.py` — calls
  `hazma._core.positron.dnde_positron_muon`; the docstring now states units.
- `hazma/spectra/_positron/_muon.pyx` — Python `def` removed; the `cdef`s
  and a comment explaining the capi-survivor status remain.
- `hazma/spectra/_positron/_muon.pyi` — **deleted** (stub for a `def` that no
  longer exists).
- `hazma/_core.pyi` — records that `positron` now carries a kernel, and why
  the per-domain submodules stay unstubbed.
- `test/parity/cases.py` — `spectra.positron.muon` repointed to the wrapper;
  `PORTED_ENTRY_POINTS` added; `assert_full_coverage` handles ported cases
  and rejects a surviving Cython `def`.
- `test/parity/test_parity.py` — the two served-kernel predicate tests
  rewritten for a non-empty roster.
- `test/test_core_positron_muon.py` — **new**, 47 tests.
- `docs/followups/todo/positron-muon-spectrum-normalization-inverted.md` —
  **new**, plus its index row in `docs/followups/README.md`.
- `projects/cython-to-rust/phases/phase-04-spectra-kernels.md` — the swap
  recipe this task established, written into the phase Goal.
- `projects/cython-to-rust/task-notes/phase-04/README.md`, `../README.md`,
  and this note — bookkeeping.

## Verification

Environment built from clean in the task worktree with the corpus's pinned
numerics (`numpy==2.5.1`, `scipy==1.18.0`, `cython==3.2.9`, CPython 3.12.12),
`uv pip install -e . --no-build-isolation`, and confirmed in-tree:
`hazma.__file__` and `hazma._core.__file__` both resolve inside the worktree.
Rebuilt after every `.pyx` and `.rs` edit before any Python-side number was
quoted.

- `cargo test --manifest-path rust/Cargo.toml --no-default-features` →
  `80 passed; 0 failed` (was 69 before this task; +11 units for the new
  module, three of them added in response to the mutation campaign
  below). Covers: the seven folded constants against the disassembled
  immediates; the rest-frame support at and either side of both edges; the
  inverted normalization; the `β → 0` limit agreeing with the rest-frame
  branch; unphysical `β` and empty windows; both public thresholds; the
  rest-frame Jacobian; in-flight number conservation; and `R_FACTOR` against
  the quartic closed form.
- `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` clean.
- **Mutation campaign against `rust/src/kernels/positron_muon.rs`:**
  18 mutations, run sequentially from a green baseline with the baseline
  re-asserted green afterwards, each gated by `cargo test` **and** a
  rebuild plus `pytest test/test_core_positron_muon.py` (the Python side
  is the stronger oracle — it is the one that sees an unfused
  multiply-add). **13 caught on the first pass, 16 after three tests were
  added, and the two that remain are provably equivalent mutants.**
  - `x.mul_add(2.0, C)` → `x * 2.0 + C` survives **because it is the same
    function**: multiplying by a power of two is exact, so there is no
    intermediate rounding for the fusion to remove. One of the nine
    `fmadd` instructions in the shipped object code is therefore not
    observable from Python at all. Worth knowing before a later task
    treats "the disassembly says `fmadd`" as "this site is load-bearing".
  - `(beta + beta)` → `2.0 * beta` was included as a deliberate control
    and survives for the same reason.
  - The three that were caught only after new tests all moved a **branch
    boundary** without moving any value — the same shape as Task 3.4's
    three survivors. `x <= 2r` → `x < 2r` hides behind a *signed* zero
    (`==` cannot see it; `to_bits()` can); `xm > xp` → `xm >= xp` needs an
    `x` for which `x·x − r₂₂` is exactly `0.0`, which no swept grid
    contains and the test now searches for; and `beta < f64::EPSILON` →
    `2.0 * f64::EPSILON` is invisible end-to-end because **that branch is
    unreachable from `dnde_positron_muon`** — the outer
    `E − m_μ < f64::EPSILON` guard already routes everything that could
    produce such a `beta`, the smallest surviving one being
    `sqrt(2·eps/m_μ) ≈ 6.5e-9`. The guard is kept because the Cython has
    it, and is now pinned by a direct `dndx` test.
- `pytest test/test_core_positron_muon.py -q` → `47 passed in 0.52s` on the
  capturing platform, and `47 passed in 0.34s` again with
  `platform.machine` forced to `x86_64` so the budget branch is the one
  under test. Nothing skips in either mode.
  Covers: the eleven dispatch-contract branches with this kernel's wording;
  wrapper and `hazma.spectra` re-export identity; the removed `def` and the
  two surviving capsules (plus the capsule's C-signature name);
  bit-equality against the twin on swept, random and edge grids at seven
  muon energies; `NaN`; and eight physics statements.
- `pytest test/parity -q -rs` → `629 passed, 1 skipped in 310.99s`, the skip
  being `test_running_on_the_capturing_tree` with reason
  `declared budgets in force: kernel digest f5e6e269be47 -> fdbae2c19d87;
  hazma._core serves 1 kernel(s)` — both differences expected and named.
- `pytest test/test_theory_aggregation.py -q` → `69 passed` (the
  platform-independent model-layer gate, run either side of the swap per the
  project handoff).
- **Bare `pytest -q` → `1422 passed, 14 skipped in 551.70s`** on the
  capturing platform, against `1378 passed, 13 skipped` at Task 3.5. The
  arithmetic closes exactly: `+45` passes for the new module, `−1` pass /
  `+1` skip for `test_running_on_the_capturing_tree`, which now skips
  because the corpus is in budget mode. **That skip is the designed signal
  that the corpus has left bit-equality mode**, and from this task on it is
  permanent. Off the capturing platform 17 more skip and the parity suite
  is ignored entirely. **That skip is the
  designed signal that the corpus has left bit-equality mode**, and from
  this task on it is permanent.
- Ad-hoc bit-equality sweep against the Cython `cdef` through
  `__pyx_capi__` before any wrapper was touched: **126,182 points across 14
  muon energies** (rest, `+1e-16`, `+1e-9`, mildly and strongly boosted,
  `1e9`, below threshold, zero) on geometric, linear, random and
  edge-enumerated grids — **0 not bit-equal, worst relative deviation
  0.000e+00**.

**Preflight, and the one gate that needed scoping.** With
`--paths` covering all five touched Python files, `preflight.sh` reports
`FAIL ruff check` — **24 findings, every one of them in
`hazma/spectra/_positron/__init__.py` and every one of them already on the
trunk.** Proved rather than asserted: `ruff check --output-format concise`
run against the branch's file and against
`git show origin/master:hazma/spectra/_positron/__init__.py` produces two
finding lists that `diff` reports as identical (24 each) — 13 `UP007`,
2 `UP006`, and one each of `UP035`/`UP045`/`PLR0913`/`PLR0917`/`F841`/
`D205`/`D400`/`D412`. They are typing- and docstring-modernization
findings across a 500-line public wrapper, not anything this diff
introduced; `UP007` in particular would rewrite `Union[...]` annotations
in a module `pyproject.toml` marks `runtime-typing = true`, so
"just autofix them" is a public-API change, not a cleanup. This is
[`preflight-isort-ruff-red-on-trunk.md`](../../../../docs/followups/todo/preflight-isort-ruff-red-on-trunk.md)
exactly, and the documented handling is to scope `--paths` to the files
whose verdict the diff can move. Re-run over the other four:
**RESULT: PASS**, with `black`, `isort`, `ruff`, all three cargo gates,
`pytest` (`1422 passed, 14 skipped in 584.61s`), `import hazma`,
`markdownlint` over the six changed docs, and the forbidden-token scan
all green.

Intentionally deferred: nothing in this task's scope. The normalization
defect is reproduced rather than repaired, under rule 1 and with a
follow-up; the `hazma._core` submodule stubs are left to Phase 07.

## Numerical impact

**No public value changes** — measured, not argued.

The only entry point this diff can reach is `dnde_positron_muon`, and its
"before" is still in the tree: the pre-port Cython `cdef`
`dnde_positron_muon_point`, reachable through `_muon.pyx`'s `__pyx_capi__`
now that its `def` is gone.

- **`dnde_positron_muon`:** `np.logspace(-2, 3, 200)` MeV at muon energies
  150 / 500 / 1500 MeV — 3 arrays, 600 values — **bit-for-bit identical**
  to the capsule (max relative deviation `0.000e+00`). The wider 126,182-point
  sweep above says the same thing over 14 parent energies and every
  kinematic edge.
- **The parity corpus says it more strictly:** `spectra.positron.muon`'s
  declared budget is `EXACT_RTOL = 0.0`, so the swap passed at `rtol = 0`
  against its pre-port pins — the gate did not weaken for the entry point
  being swapped.
- **Everything else on the public compiled surface is untouched code.**
  `git diff origin/master -- hazma` is four files: `hazma/_core.pyi`
  (comment only), `hazma/spectra/_positron/__init__.py` (one call site plus
  a units line), `_muon.pyx` (the removed `def`), and the deleted
  `_muon.pyi`. No other kernel, constant or build input is reachable from
  it. 213 arrays evaluated without error as a smoke check: the 34
  two-argument `dnde_*` entry points at three parent energies, plus both
  models' `spectra()` / `positron_spectra()` /
  `annihilation_cross_sections()` / `thermal_cross_section()` at three
  mediator masses.

**What *did* change is the gate's mode, permanently, and that is worth
stating precisely** because the project handoff flagged it as a risk. From
this swap `tolerances.provenance` reports `exact=False`, so
`effective_budget` returns the *declared* budget for every case rather than
`rtol = 0`. The declared budget for the `EXACT` class is itself `0.0`, so
**19 of the 41 cases lose nothing**; the other 22 loosen — `SPECFUN` (1
case) to 1e-13, `TABULATED` (7) to 1e-12, `QUAD` (5) to 1e-8 and `NESTED`
(9) to 1e-6 — plus the abscissa comparison to 1e-13. All 41 still pass
(counts derived from `test/parity/tolerances.py`'s `BUDGETS`, not
transcribed). Two reasons
are recorded in the skip message and only one is the swap: the kernel digest
also moved, because removing a `def` changes the `.pyx` bytes the digest
covers — so this mode flip was unavoidable in any task that touches a
surviving `.pyx` at all, swap or no swap.

## Open Questions

- **Should the mode switch become per-case?** `provenance` is one global
  verdict, so a still-Cython `TABULATED` kernel is now compared at 1e-12
  rather than bit-exactly even though nothing about it moved. Splitting the
  verdict — environment and digest differences global, the served-kernel
  difference scoped to `PORTED_ENTRY_POINTS` — would keep the unported
  kernels bit-exact for the rest of Phases 04–06. Not done here: it is a
  corpus-design change, it belongs with
  [`parity-corpus-pins-ill-conditioned-points.md`](../../../../docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md)'s
  option 2 rather than beside a kernel swap, and the digest half of the
  verdict would still fire. Recorded there is the right home; recorded here
  is the measurement that says how much is at stake (22 of 41 cases).
- **The ill-conditioned-points repair did not land before this swap**, which
  the project handoff asked for. It is not blocking for *this* kernel — none
  of the six affected blocks is `spectra.positron.muon` (five are scalar
  cross sections, Phase 05; the sixth is `spectra.photon.eta`, Task 4.2) —
  but corpus **regeneration** is closed from now on, so options 1 and 2 in
  that follow-up have to work from a pre-Phase-04 checkout. **Task 4.2 is
  the first task that actually meets one of the six**, and it should not
  start until that follow-up is resolved or explicitly waived.

## Plan Impact

**Impact Level:** Phase file patched.

No ADR. The two decisions with reach beyond this task — the swap recipe
(repoint the corpus case, record the origin, remove the survivor's `def`)
and the per-kernel test-module shape — are Phase 04–06 *procedure*, not
architecture, so they are written into
`../../phases/phase-04-spectra-kernels.md`'s Goal where Tasks 4.2–4.6 will
read them, rather than into an ADR. Nothing in `PLAN.md`, `rules.md` or any
existing ADR became wrong: rule 1's delete-twin-same-PR is satisfied as far
as the phase's own declared capi exception allows, and the phase Goal
already said the survivors would be "Python-unreferenced" — this task made
that literally true rather than approximately so.

## Stale-state sweep

Run against this branch after every prose edit was frozen, then re-run to
confirm a fixed point. Hit lists are **folded to one row per identifier**
with a disposition; the counts are the commands' real output, taken on
the final tree — the first pass was taken before this block and the
preflight paragraph existed, and every count that moved did so because
those additions cite the identifiers they describe. Re-running each
command after this edit reproduces the numbers below: the deterministic
ones (`check_doc_citations.py`, `cargo test`, `--collect-only`,
`objdump | grep -c`) byte-identical, and the multi-directory `rg` sweeps
identical after `sort`, since `rg` walks directories in parallel.

### Identifier sweep

`rg -n '<identifier>' projects/ docs/ README.md hazma/ test/ rust/`

| Identifier | Hits | Disposition |
| --- | --- | --- |
| `dnde_positron_muon` | 88 | KEPT — the public name; every hit is the wrapper, the Rust registration, a test, or a project record. |
| `positron_muon` | 142 | KEPT — includes `constants::derived::positron_muon` (Task 3.1) and the new kernel module. |
| `PORTED_ENTRY_POINTS` | 23 | NEW — `test/parity/cases.py` (definition + docs), `test/parity/test_parity.py`, the phase file, and the notes. |
| `test_scaffolded_core_serves_no_kernels` | 4 | 2 EDITED (this note — the rename in §Decisions and this row), 2 KEPT — both in `phase-03/task-3.2-specfun.md`, which is a dated record of what it was called then. |
| `_muon.pyi` | 5 | KEPT — all five are this task's own records of the deletion (this note ×3 including this row, `phase-04/README.md`, `../README.md`). No build input, `setup.py` entry or import references it. |

### Line-number citation sweep

`scripts/agents/check_doc_citations.py <the six markdown files this task
touched>` — note `--changed-vs origin/master` reports nothing pre-commit,
because it diffs `HEAD`, not the working tree
([`citation-checker-skips-deleted-inrepo-files.md`](../../../../docs/followups/todo/citation-checker-skips-deleted-inrepo-files.md)
is the same blind spot); explicit paths are what works before the commit.

```text
docs scanned: 6
in-repo citations checked: 13
  resolved by exact: 12
  resolved by suffix: 1
external citations skipped: 0
out-of-range or ambiguous: NONE
```

Two citations the checker does not parse, verified by hand:
`hazma/spectra/_neutrino/_muon.pyx:23,58,114` (comma form) — `:23` is
`DEF R_FACTOR = 1.0001870858234163`, `:58` is
`common = R_FACTOR * x**2 * ...`, `:114` is
`pre = R_FACTOR * e_to_x / (2.0 * beta)`; and
`hazma/spectra/_positron/_muon.pyx:28`, the rest-frame `return ... /
R_FACTOR`, re-read after the `def` was removed (the deletion is at the
end of the file, so the line did not move).

### Forward-looking phrase sweep

`rg -n '(Task [0-9.]+ will|will be added|still pending|today: ?stub|are
empty until|Not started)'` over every file this task touched. **10 hits
in 3 files**, per `rg -c`:

| File | Hits | Disposition |
| --- | --- | --- |
| `../README.md` | 3 | KEPT — Phases-table cells for 05, 06, 07. |
| `phase-04/README.md` | 5 | KEPT — Tasks-table cells for 4.2–4.6. |
| `phase-04/task-4.1-positron-muon.md` | 2 | SELF — the two lines the command itself is printed on, immediately above. Recorded rather than filtered, per `doc-consistency.md` rule 3; the dispositions below deliberately avoid quoting the phrases they disposition, so this count is stable under its own edits. |

Nothing else matches, and that is the result: the one row the sweep
exists to catch has already been fixed. `phases/phase-04-spectra-kernels.md`
had `status:` set to the not-started value in its frontmatter and is now
`In Progress`, so the phase file no longer appears above. Corrected
without the sweep prompting it: `hazma/_core.pyi` described the
per-domain submodules as unfilled until Phases 03-06, which `positron` no
longer is.

### Count sweep

| Claim location | Command | Actual | Status |
| --- | --- | --- | --- |
| `cargo test` → 80 | `cargo test --manifest-path rust/Cargo.toml --no-default-features` | `80 passed` | OK |
| 11 new Rust units | count of `kernels::positron_muon::tests` lines | 11 | OK |
| 47 Python tests, no skips in either mode | `pytest -q`, then again under a forced `platform.machine` | `47 passed` both times | OK |
| Budget clears the observed Linux data | replay the 15 decoded failure blocks of run 31564747071 through the shipped assertion | all PASS, tightest headroom 84× | OK |
| Bare suite 1422 / 14 | `pytest -q` | `1422 passed, 14 skipped in 551.70s` | OK |
| Parity 629 / 1 | `pytest test/parity -q -rs` | `629 passed, 1 skipped in 310.99s` | OK |
| Aggregation gate 69 | `pytest test/test_theory_aggregation.py -q` | `69 passed` | OK |
| Nine FMA instructions | `objdump -d ... \| grep -cE 'fmadd\|fmsub'` | 9 | OK |
| Two surviving capsules | `len(_muon.__pyx_capi__)` | 2 | OK |
| One served kernel | `cases.rust_core_kernels()` | `['hazma._core.positron.dnde_positron_muon']` | OK |
| 19 EXACT / 22 loosened of 41 | counted over `tolerances.BUDGETS` | EXACT 19, SPECFUN 1, TABULATED 7, QUAD 5, NESTED 9 = 41 | OK |
| Bit-equality sweep 126,182 pts | ad-hoc `__pyx_capi__` comparison | `0 not bit-equal; worst rel 0.000e+00` | OK |
| Mutation campaign 18 / 13 / 16 | sequential campaign + three re-runs | `caught=13 survived=5`, then 3 of the 5 caught after new tests | OK |
| Deficit 0.0374% | `1 - 1/R_FACTOR**2` | `0.00037406666970341007` | OK |
| Fused reference is the Cython | 21,000-point sweep, `Fraction`-rounded `fma` at the 7 sites | `fused != Cython: 0`, `unfused != Cython: 11713` | OK |
| Wrapper ruff findings unchanged | `diff` of `ruff check --output-format concise` on the branch file vs `origin/master`'s | empty diff, 24 findings both sides | OK |

### Numerical-impact statement

**No public value changes** (verified: `dnde_positron_muon` over
`np.logspace(-2, 3, 200)` MeV at muon energies 150 / 500 / 1500 MeV —
3 arrays, 600 values — **bit-for-bit identical** to the pre-port Cython
`cdef` through `__pyx_capi__`, max relative deviation `0.000e+00`; plus a
126,182-point sweep over 14 parent energies and every kinematic edge, 0
mismatches; plus `pytest test/parity -q`, which holds this entry point to
`rtol = 0` against its pre-port pins). `git diff origin/master -- hazma`
touches four files and no other kernel. The gate's *mode* changed —
details in `## Numerical impact` above.

### Exit Criteria → test mapping

| Exit criterion | Satisfied by |
| --- | --- |
| `dnde_positron_muon` on Rust | `rust/src/kernels/positron_muon.rs`; `cases.rust_core_kernels()` returns it |
| corpus exact-or-≤1e-13 | `pytest test/parity -q` → 629 passed, at `EXACT_RTOL = 0.0` for this case |
| wrapper swapped | `hazma/spectra/_positron/__init__.py`; `TestWrapperAndPublicApi` (3 tests) |
| twin deferred (capi survivor) | `def` removed from `_muon.pyx`; `test_the_cython_module_no_longer_exports_a_python_entry_point` and `test_the_cdef_capsules_the_mediator_modules_cimport_are_intact` |
| establishes the per-kernel PR template | the eight-step recipe in `../../phases/phase-04-spectra-kernels.md`; `test/test_core_positron_muon.py`'s module docstring |

### Task-note self-consistency

`**Status:** Complete` matches every Exit Criterion having a mapping row.
Every file named in §Files Changed appears in `git status --short` /
`git diff --stat origin/master --`, and every identifier cited in
§Findings and §Decisions (`R_FACTOR`, `PORTED_ENTRY_POINTS`,
`assert_full_coverage`, `map_unary`, `dndx`, `dndx_rest_frame`,
`__pyx_capi__`, `CYTHON_CONTRACTS`) resolves to a line in the diff or a
created file. The phase README's Tasks row, this note's header, and the
project README's Phases row all read Task 4.1 Complete / Phase 04 In
Progress, and the phase file's frontmatter now agrees.

## Handoff to Next Task

**Read first:** this note's Open Questions, then
`../../phases/phase-04-spectra-kernels.md`'s Goal (the swap recipe is now
there), then `test/test_core_positron_muon.py`'s module docstring — it is
the template, and it says why it is not a copy of the dispatch tests.

**Now safe to assume:**

- The swap mechanism works end to end and is proven: kernel in
  `rust/src/kernels/<pyx name>.rs`, registration in the per-domain module
  through `dispatch::map_unary`, wrapper repointed, corpus case repointed
  with a `PORTED_ENTRY_POINTS` row, survivor's `def` removed.
- The disassembly-first order pays. Map the FMAs out of the shipped `.so`
  *before* writing the Rust; this kernel was bit-equal on the first build
  because of it, and three of its multiply-adds are not fused.
- `hazma._core.positron.dnde_positron_muon` exists and is bit-equal to the
  `cdef` the mediator modules still cimport, so Task 4.6's
  `_positron/_pion` port has a verified Rust dependency to call natively.
- The corpus is in budget mode and cannot be regenerated. Every later drift
  line is measured against declared budgets, except the `EXACT`-class cases
  which are still `rtol = 0`.

**Still risky / unknown:**

- **Task 4.2 meets `spectra.photon.eta`**, one of the six ill-conditioned
  blocks, and it is also the first `TABULATED` swap — now budgeted at 1e-12
  rather than bit-exactly, against a Task 3.4 measurement that says the
  unfused arithmetic misses by up to 3.6e-12. Read
  `parity-corpus-pins-ill-conditioned-points.md` and Task 3.4's note
  together before starting.
- The normalization defect is reproduced and will be reproduced again by
  Task 4.6's `_positron/_pion` (which boosts this spectrum) and by Phase 06's
  mediator positron modules. Do not "fix" it in passing; the corpus pins the
  low values.
