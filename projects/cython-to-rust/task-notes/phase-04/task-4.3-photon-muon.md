# Task 4.3: `_photon/_muon` (spence)

**Date:** 2026-08-16
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-04-spectra-kernels.md` (Goal — the
eight-step swap recipe and the capi-survivor exception; Task 4.3),
`../../rules.md` rules 1–3 (parity discipline), 5 (licensing), 6–9 (Rust
conventions)
**Related ADRs:** ADR-0001 (framework), ADR-0002 (license-clean numerics —
the cephes transcription below lands squarely inside its permitted
provenance)
**Depends On:** Task 4.1 (the template swap), Task 3.2 (`special::spence`)

## Objective

Port `hazma/spectra/_photon/_muon.pyx` — the radiative muon-decay photon
spectrum of arXiv:hep-ph/9909265, the project's only `spence`-bearing
kernel — to `hazma._core.photon.dnde_photon_muon`, keeping the Rust `fn`
in the PyO3-free layer so Task 4.4 and Phase 06 can call it natively.

## Exit Criteria

From the phase file's Task 4.3 block:

- `dnde_photon` (radiative muon decay, hep-ph/9909265 spectrum incl. the
  `spence(xm) - spence(xp)` term) corpus-green at ≤1e-12 rel. **Met, and
  bettered:** the corpus runs it at `SPECFUN` (1e-13) and the port is
  bit-equal to the twin at every point the corpus samples.
- The kernel stays cimport-compatible-in-spirit: its Rust `fn` is callable
  natively by Phase 06 — kept in the PyO3-free kernel layer. **Met**;
  `dnde_photon_muon` and `dnde_photon_muon_rest_frame` are both `pub` in
  `rust/src/kernels/photon_muon.rs`, which imports no PyO3.

Plus the phase Goal's recipe, steps 1–8: FMA map read from the shipped
`.so` before writing Rust; kernel in `kernels/<pyx name>.rs`; registered
through `dispatch::map_unary` with the twin's wording; wrapper repointed;
corpus case repointed and a `PORTED_ENTRY_POINTS` row added; the twin's
`def` deleted (capi survivor — the file stays); `test/test_core_photon_muon.py`
added; drift recorded here and in the working-memory README.

## Inputs Reviewed

- `../../PLAN.md` (Numerical impact), `../../phases/phase-04-spectra-kernels.md`,
  `../../rules.md`, `../README.md` (phase working memory), `../../../README.md`.
- `../phase-03/task-3.2-specfun.md` — the measured scipy agreement table,
  which is where this task's central problem was already visible.
- `../../learnings/phase-03-numerics-foundation.md` §"`spec_math::Polylog::li2`
  *is* `scipy.special.spence`".
- `task-4.1-positron-muon.md` (the template) and `task-4.2-photon-table-family.md`.
- `docs/followups/done/parity-corpus-pins-ill-conditioned-points.md` — the
  follow-up that predicted the failure mode this task hit.
- `docs/agents/lessons.md`, `docs/agents/environment.md`,
  `docs/agents/doc-consistency.md`.
- Sources: `hazma/spectra/_photon/_muon.{pyx,pxd,pyi}`,
  `rust/src/{special,boost,photon,kernels}.rs`,
  `rust/src/kernels/positron_muon.rs`, `test/parity/{cases,tolerances,test_parity}.py`,
  `test/test_core_{positron_muon,dispatch,special}.py`, `setup.py`.
- `~/.cargo/registry/.../spec_math-0.1.6/src/cephes64/{spence,polevl}.rs`.

## Findings

### The port was bit-equal on the first build; `spence` was not

Written from the disassembly, the kernel reproduced the shipped Cython
**exactly** everywhere except through `special::spence`. That is not an
inference — it is the measurement. Of 70,000 sampled points the first
build differed at 11,306, and re-deriving each difference as
`(5/β)·Δspence·α/(3π·E_μ)` from the *same* `xm`, `xp` the kernel forms
reproduced the observed difference to a ratio of **1.000** at every one of
the 24 corpus points that failed. Nothing else in 22 FMA sites, two `pow`
calls, eight logarithms and four folded constants was off by an ulp.

### `1/β` turns two ulps of `spence` into 320x the budget

The closed form carries `(5/β)·(spence(x₋) − spence(x₊))`, and
`test/parity/cases.py` samples every parent at `mass·(1 + 1e-12)` —
`β = 1.4142764231806604e-06`. The amplification is `5/β ≈ 3.5e6`, so
`spec_math`'s ≤2.0e-15 relative disagreement with `scipy.special.spence`
arrived as a **3.15e-11** relative shift in the spectrum, against a
`SPECFUN` budget of 1e-13. The absolute size was never more than
1.15e-14, on a block whose peak is 17.2 — which is exactly the shape
`parity-corpus-pins-ill-conditioned-points.md` describes: a relative
budget is the wrong instrument where the kernel cancels.

Widening the budget 300x was the obvious move and the wrong one. The
alternative turned out to be cheap.

### `spec_math` differs from scipy only by **contraction**, and the fix is exact

`spec_math::cephes64::spence` is a faithful line-for-line translation of
cephes `spence.c`; the coefficients are identical. What differs is that
scipy ships that C compiled by clang with `-ffp-contract=on`, so
`polevl`'s `ans = ans*x + c` is a chain of `fma`, and Rust does not
contract. Reproduced in Python with correctly-rounded `Fraction` FMAs on
an 8,000-point sweep of `(0, 1)`:

| Spelling | Mismatches vs scipy | Max rel. |
| --- | --- | --- |
| `polevl` unfused (what `spec_math` does) | 2289 / 8000 | 2.042e-15 |
| `polevl` fused | 279 / 8000 | 3.959e-16 |
| `polevl` fused + the reflection term fused | **0 / 8000** | **0** |

The residual 279 were all in the `x < 0.5` arm, i.e. the
`π²/6 − ln(x)·ln(1−x) − y` reflection, whose product clang folds into an
`fnmsub`. Extending the sweep to all four branches (13,000 points, `x` from
1e-12 to 1e6) with the `1/x` arm's `−0.5·z·z − y` fused as well:
**0 mismatches**.

So `rust/src/special.rs` now transcribes cephes `spence` in-tree with that
contraction map instead of calling `spec_math`. It is the same algorithm
and the same coefficients — only the roundings move, and *toward* scipy.
Fusing is also the more accurate Horner, so nothing was traded.

With it, the kernel is bit-equal to the Cython at **144,000** sampled
points across nine parent energies, and the whole parity suite is green
with the `SPECFUN` budget unused.

### Two `pow` calls a port would never guess

`y**3` and `y**4` in the rest-frame branch are **libm `pow` calls**, not
repeated multiplication: `objdump -d` shows two `bl` to `_pow` with `3.0`
and `4.0` in `d1` (`otool -Iv` names the stub). `y**2` is a plain `fmul`,
because clang folds `pow(x, 2.0)` to `x*x` (exact) and refuses the cubic.
Writing `y*y*y` would have been a different double.

### The FMA map is bigger than the positron muon's and part of it is vectorized

22 FMA instructions, all in `dnde_photon_muon_point`: 17 `fmadd`, one
scalar `fmla.d`, and **four `fmla.2d`** — clang runs the two
log-coefficient Horner chains in `x₋` and `x₊` as one 2-wide chain, with
the lane constants `{2, −2}`, `{−9, 9}`, `{18, −18}`, `{−18, 18}`,
`{−9, 9}` in rodata. A vector FMA is the same operation per lane, so each
lane transcribes to an ordinary `mul_add` chain — but a reader counting
`grep -c fmadd` sees 17 and misses five sites. The rest frame is inlined
**once** (the `.pyx` calls it from one branch), unlike the positron
sibling's twice.

### A fifth live 2.1.0 defect: the rest-frame branch stops 0.25 MeV early

Found by writing the identity the original never asserted — that the
in-flight closed form is the boost integral of the rest-frame
distribution. It is, to **machine precision** (relative difference 1e-15
or exactly 0 wherever the boost window is not truncated) — but only when
the rest-frame form is integrated to `y = 1 − r`. Against the shipped cut
at `y = 1 − √r` the two disagree by 3.2e-6.

`hazma/spectra/_photon/_muon.pyx:41` guards the rest frame with
`y >= 1.0 - MASS_E / MASS_MU`
while `r = (MASS_E/MASS_MU)**2` is defined two lines above and the
in-flight branch (`hazma/spectra/_photon/_muon.pyx:88`) uses
`(1.0 - r)`. The kinematic endpoint is `(m_μ² − m_e²)/(2m_μ)`, which
`hazma/spectra/_photon/_pion.pyx:16` also hard-codes
as `ENG_GAM_MAX_MURF = 52.82795006985128`. So three places in hazma say
`1 − r` and one says `1 − √r`.

Measured consequence: `dnde_photon_muon(E, m_μ)` is exactly `0.0` over
`52.5736877769 < E < 52.8279515698 MeV` (0.2543 MeV, 0.481% of the
endpoint) where the spectrum runs from `5.34e-7 MeV⁻¹` down to zero, so
the published spectrum is **discontinuous in the parent energy at
`E_μ = m_μ`**: at the cut, a muon one part in `10¹²` above rest returns
`5.335612532537976e-07` where a muon exactly at rest returns `0.0`.
`5.45e-8` photons per decay are lost — `1.09e-6` of the yield above
1 MeV, `4.18e-6` above 10 MeV, `7.22e-6` of the radiated energy above
1 MeV.

Reproduced per rule 1, filed as
[`photon-muon-rest-frame-endpoint-uses-the-wrong-power-of-r.md`](../../../../docs/followups/todo/photon-muon-rest-frame-endpoint-uses-the-wrong-power-of-r.md),
blocked behind Phase 06 Task 6.4 like its three siblings. **Five blocked
defects now share one eventual corpus regeneration.**

### The spectrum goes slightly negative at the boosted endpoint, in both

Inside the last 0.1% of the support the closed form's terms cancel to a
residual that dips negative — `−2.9e-12` at `E_μ = 110 MeV` on a
4001-point grid. Measured against the Cython twin, the dip is
`2.783e-4` of the value at 0.99 of the endpoint, and it is the **same
fraction at every parent energy from 110 MeV to 1e5 MeV**, because it
depends only on the scaled variable. Both implementations do it; a naive
"the spectrum is non-negative" test fails on the shipped Cython too.

### The corpus cannot see a single wrong FMA site; the hand-written oracle can

Measured, not supposed — see the mutation table under Verification.
Unfusing one of the 22 multiply-adds leaves all 630 corpus assertions
green and all 109 `cargo` tests green, and is caught only by
`test/test_core_photon_muon.py`'s bit-equality sweep, at four of ten
thousand random arguments. The corpus samples where the phase file told
it to; a per-kernel oracle that reaches arbitrary arguments is what covers
the rest. This is the concrete answer to "why not let the corpus be the
only numerical gate for a swap".

### `E_γ = 0` is `NaN` in flight and `0.0` at rest

`x = 0` passes both support tests (`0 < 0` is false, and `0 >= (1−r)/(1−β)`
is false), so the closed form runs with `x₋ = x₊ = 0` and takes `ln 0`.
The rest-frame branch's `y <= 0.0` guard is the one that returns zero.
The corpus samples no exact zero, so only a hand-written test sees this.
Opposite in kind to Task 4.1's finding for the positron muon, where a
`NaN` energy comes back *finite* — there `fmax`/`fmin` clip; nothing
clips here, so a `NaN` propagates on both branches.

## Decisions and Implementation Notes

- **Transcribe cephes `spence` rather than widen the `SPECFUN` budget.**
  The budget would have had to go from 1e-13 to 1e-10 to admit a 2-ulp
  `spence` difference at `β = 1.4e-6`, which is the "widening a tolerance
  until it passes makes the gate vacuous exactly where the numerics are
  most fragile" that `parity-corpus-pins-ill-conditioned-points.md` warns
  against. Transcribing cost ~60 lines and made the difference **zero**.
  `special.rs` already had the precedent: Task 3.2 dropped `spec_math`'s
  `bessel_kn` because scipy's `kn` is not cephes `kn`. The module's
  contract is *match scipy*, not *use spec_math*.
- **The `SPECFUN` budget is kept, not tightened to `EXACT`.** Same
  reasoning Task 4.2 applied to `TABULATED`: bit-equality here rests on
  scipy's C being compiled with FP contraction, which is a property of
  that build. `EXACT` would be the wrong contract rather than a tighter
  one. Recorded in the class docstring in `test/parity/tolerances.py`.
- **The `.pyx`'s `def` was named `dnde_photon`, not `dnde_photon_muon`** —
  the first ported entry point whose Cython name differs from its public
  one. That broke
  `test_parity.py::test_the_served_roster_is_exactly_the_ported_entry_points`,
  which derived the expected roster from `PORTED_ENTRY_POINTS`'s *values*
  (the `.pyx` origin, which `assert_full_coverage` needs). Fixed by
  deriving the roster from the live cases' `function` instead. Tasks
  4.4–4.6 inherit the fix; nothing else was reachable from it.
- **`test/test_core_dispatch.py`'s Cython oracle moved to `_photon/_pion`.**
  Three `TestDeclaredDivergencesFromCython` tests called
  `_muon.dnde_photon` as the Cython half of a declared widening.
  `_pion.dnde_photon_charged_pion` has the identical
  `hasattr(__len__)`/`assert` shape and the identical `"Photon energies"`
  wording, so the assertions are unchanged. Task 4.4 swaps that one too
  and will have to move it again or retire the class — recorded in the
  class docstring and in the handoff, not left to be rediscovered.
- **The rest-frame endpoint constant is named `Y_MAX` and the boosted one
  `ONE_MINUS_R`, and a test asserts they are different.** Both sit within
  5e-3 of 1, so swapping them yields a spectrum that still looks like a
  spectrum; the separation is pinned in a `const` block (clippy refuses a
  runtime `assert!` on compile-time constants).
- **Review round 1 (PR #67): four citation/count corrections, no code
  change.** All four were valid and all four were mine. The mechanism
  behind three of them is one thing: **`preflight.sh` has no gate row for
  `check_doc_citations.py`**, so an all-green gate said nothing about the
  citations in the six docs this PR touched, and I read it as coverage.
  Fixed: the reference doc's `:148-153` line range into
  `hazma/spectra/_photon/_muon.pyx` (deleting the `def` cut the file to
  148 lines) now cites a surviving example instead — and note that
  *quoting* the broken range in citation form reproduces the failure, so
  it is written split here; six bare
  `_muon.pyx` / `_pion.pyx` citations are full repository-relative paths
  — that basename is ambiguous across all three spectra packages;
  `183 passed` → `184` (recorded before the `spence` bit-equality test was
  added to `test_core_special.py`); the sweep block's path count is
  re-derived from `git diff --name-status` (21: 16 `M`, 4 `A`, 1 `D`)
  rather than from a mid-session `git status`. Swept the same classes
  across the whole diff, which also caught a "fourth live 2.1.0 defect"
  heading that should have read "fifth". Lessons ledger updated.
- **The public wrapper now states its units** — `MeV⁻¹`, both arguments in
  `MeV` — matching the seven wrappers Task 4.2 annotated. Non-blocking
  review point; in scope because the convention in that file is that the
  task swapping a wrapper annotates it. `_pion` and `_rho` are Tasks
  4.4–4.5's by the same rule.
- **No new `constants::derived` submodule.** Unlike `_positron/_muon.pyx`,
  this `.pyx` declares no module-level constants — `r` is a function
  local — so the four folded values are `const` in the kernel module and
  pinned against the disassembled immediates there.

## Files Changed

- `rust/src/kernels/photon_muon.rs` — **new.** The kernel, PyO3-free:
  `dnde_photon_muon_rest_frame`, `dnde_photon_muon`, four folded
  constants, 11 unit tests.
- `rust/src/kernels.rs` — register the submodule; extend the
  one-module-per-`.pyx` note.
- `rust/src/photon.rs` — register `dnde_photon_muon` through
  `dispatch::map_unary` with `"Photon energies"`.
- `rust/src/special.rs` — `spence` transcribed from cephes with the
  contraction map (`SPENCE_A`, `SPENCE_B`, `polevl`, `PI_SQ_OVER_SIX`);
  `spec_math::Polylog` dropped from the imports; module docs record that
  two of the three functions now bypass `spec_math` and why; 2 new tests.
- `hazma/spectra/_photon/__init__.py` — `dnde_photon_muon` calls
  `hazma._core.photon`; `_muon` dropped from the `.pyx` import list.
- `hazma/spectra/_photon/_muon.pyx` — `def dnde_photon` deleted, replaced
  by a comment naming the capi-survivor exception. Both `cdef`s stay.
- `hazma/spectra/_photon/_muon.pyi` — **deleted** (it stubbed only the
  `def`).
- `hazma/_core.pyi` — the roster comment names Task 4.3.
- `setup.py` — the Task 4.2 comment updated: `_muon` keeps its extension
  for its capsules.
- `test/test_core_photon_muon.py` — **new**, 53 tests.
- `test/test_core_special.py` — bit-equality-with-scipy test for `spence`,
  scoped to the capturing platform the same way `test/parity` is.
- `test/test_core_dispatch.py` — the spectra oracle repointed to `_pion`.
- `test/parity/cases.py` — the `photon.muon` case repointed to the
  wrapper; `PORTED_ENTRY_POINTS` row added.
- `test/parity/test_parity.py` — served-roster derived from the cases.
- `test/parity/tolerances.py` — the `SPECFUN` class docstring records what
  Task 4.3 measured and why the budget stayed.
- `docs/followups/todo/photon-muon-rest-frame-endpoint-uses-the-wrong-power-of-r.md`
  — **new**; `docs/followups/README.md` — its index row.
- `projects/cython-to-rust/references/numerics-replacements.md` — the
  `spec_math`-replaces-`spence` claim it made is no longer true, so it
  now records what Task 4.3 measured and why the vendoring fallback it
  already named was taken; the Task 2.1 dispatch snapshot is marked as
  one.
- `projects/cython-to-rust/task-notes/{README.md,phase-04/README.md}` —
  working memory (status, findings, decisions, numerical impact,
  handoff).

## Verification

Commands and their real output, run in
`.claude/worktrees/cython-to-rust/task-4.3-photon-muon-spence` against a
tree rebuilt with `uv pip install -e . --no-build-isolation` after every
Rust edit (`hazma._core.__file__` confirmed inside the worktree).

- `cargo test --manifest-path rust/Cargo.toml --no-default-features` →
  `109 passed` (96 before this task: +11 in `kernels::photon_muon`, +2 in
  `special`).
- `cargo fmt --manifest-path rust/Cargo.toml --check` → clean;
  `cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings`
  → clean.
- `pytest -q test/test_core_photon_muon.py` → `53 passed`.
- `pytest -q test/test_core_special.py` → `66 passed` (65 before: +1, the
  bit-equality test; the other two new `special` tests are `cargo`).
- `pytest -q test/test_core_dispatch.py test/test_core_special.py` →
  `184 passed` (118 + 66).
- `pytest -q test/parity` → `628 passed, 1 skipped` before the roster
  meta-test was repointed, `629 passed, 1 skipped` after; the
  `spectra.photon.muon` case is green at `SPECFUN` in all five blocks.
- **Bare `pytest -q` (the gate) → `1682 passed, 15 skipped, 5 warnings in
  559.08s`.** Collection `1643 → 1697` against `origin/master`
  (`pytest --collect-only -q | tail -n 1`): +53 in
  `test/test_core_photon_muon.py`, +1 in `test/test_core_special.py`.

What the new tests cover, by kind rather than by count:

- **Rust (11).** The four folded constants against the disassembled
  immediates; that `Y_MAX` and `ONE_MINUS_R` are different constants;
  the rest-frame support at both edges on the bit pattern; that `y**3`
  and `y**4` go through libm `pow` and not repeated multiplication; the
  rest-frame/boosted continuity across the `E − m < ε` guard; both
  thresholds; the forward-cone endpoint either side; finiteness plus
  bounded endpoint cancellation; the **boost-integral identity** (both
  `w₊` branches, four parent energies, seven energies each); the shipped
  endpoint defect as the step it leaves; `NaN` propagation on both
  branches.
- **Rust `special` (2).** That `polevl` is fused (asserted as a
  difference from the unfused evaluation, argument searched for); that
  `PI_SQ_OVER_SIX` folds to the literal the closed-form tests use.
- **Python (53).** Twelve dispatch-contract branches with this kernel's
  wording; six wrapper/public-API and capsule assertions (including that
  `_pion` still imports); the Cython twin as a two-mode oracle over swept
  grids, random arguments and kinematic edges at seven parent energies,
  the support-identity check, the budget's own validity, `NaN`, `E = 0`,
  and below-threshold; nine physics statements including the endpoint
  defect and the monotone infrared fall.

### Test validity (stash-proof)

Every claim below was produced by mutating the production code, rebuilding,
and running — not by reading.

| # | Mutation | `cargo test` | `test/parity` | `test_core_{photon_muon,special}` |
| --- | --- | --- | --- | --- |
| 1 | `special::spence` back to `spec_math`'s `x.li2()` | — | 1 (`spectra.photon.muon[rest_plus_eps]`) | **14** |
| 2 | `polevl` unfused inside the transcription | — | 1 (same block) | **14** |
| 3 | `y.powf(3.0)` → `y * y * y` | — | 0 | **2** |
| 4 | `Y_MAX` ↔ `ONE_MINUS_R` in the rest-frame guard | **3** | 1 (`spectra.photon.muon[rest]`) | **6** |
| 5 | `(xp*xm).mul_add(inner, 102.0)` written unfused | 0 | **0** | **10** |

Two of these are the reason the hand-written oracle exists.

**Mutation 5 is invisible to the corpus.** Unfusing one of the boosted
polynomial's 22 FMA sites leaves every one of the 630 corpus assertions
green *and* every `cargo` test green; only
`test_core_photon_muon.py`'s bit-equality sweep sees it, and only at
`emu = 1500` and `1e5` on the *random* grid — the swept grid misses it
too. The corpus samples where the phase file told it to; a fusion that
matters at four arguments in ten thousand is not something a fixed grid
finds. **Mutation 3** is the same shape one notch louder: the `pow` sites
live in the rest-frame branch, which only `emu == m_μ` exactly reaches, so
two tests carry the whole weight.

**Mutations 1 and 2 are the reverse**: the corpus catches them at exactly
one block — the `β = 1.4e-6` probe — which is the block this task
existed to make pass.

### Numerical impact (rule 3, phase-file recipe step 8)

**Public values: none.** `hazma.spectra.dnde_photon_muon` is bit-equal to
the implementation it replaces at every point sampled — 144,000 arguments
across nine parent energies (`m_μ`, `m_μ(1+1e-12)`, `m_μ+1e-9`, 110, 150,
500, 1500, 1e5, 1e9 MeV), 0 mismatching doubles, both a geometric sweep and
a uniform random draw at each. The parity corpus agrees: all five
`spectra.photon.muon` blocks pass at `rtol = 1e-13` with a measured
difference of exactly zero.

Nothing else moved. No other public entry point's code path was touched:
`special::spence` has exactly one consumer inside `hazma` (this kernel —
`rg spence hazma/ rust/src` outside `special*.rs` and the `.pyx` returns
only `_photon/_muon.pyx:113`), and the `test_core_special` sweeps confirm
the transcription tracks scipy at least as closely as `spec_math` did
everywhere else.

The two behavior changes that are *not* value changes, both inherited from
the dispatch contract Task 3.5 declared and already covered by
`test/test_core_dispatch.py::TestDeclaredDivergencesFromCython`: a 0-d
array now returns a float instead of raising `AssertionError`, and a
rank error is a `ValueError` with the same message instead of an
`AssertionError`.

## Open Questions

- **Does the rest-frame endpoint defect reach `_pion` and `_rho`?** Their
  quadratures integrate `dnde_photon_muon_point` over the boost cone, so
  the rest-frame branch is reachable only where a muon lands within one
  `DBL_EPSILON` MeV of rest. Whether any live grid does is not settled
  here; recorded on the follow-up.
- **Was `1 − √r` deliberate in hep-ph/9909265?** The boost-integral
  identity says the in-flight formula was derived with `1 − r`, which
  makes a deliberate `1 − √r` internally inconsistent rather than merely
  approximate. Worth one read of the paper before the repair, not before
  filing.
- Carried forward from Task 4.2: should the corpus's mode switch become
  per-case? This task is the first evidence that the *budget* granularity
  matters too — a per-block `atol` would have absorbed the `spence` drift
  without a 300x rtol. Belongs with
  [`parity-corpus-pins-ill-conditioned-points.md`](../../../../docs/followups/done/parity-corpus-pins-ill-conditioned-points.md).

## Plan Impact

**Impact Level:** None canonical; one **reference** file patched.

Checked against the canonical text rather than assumed:

- `phases/phase-04-spectra-kernels.md` — the Goal's capi-survivor list and
  its "delete the `def`, keep the `cdef`s" rule describe exactly what
  happened; the eight-step recipe was followed step for step; Task 4.3's
  two exit-criterion bullets are both met as written (the corpus is green
  *tighter* than the ≤1e-12 the bullet asks for, which does not make the
  bullet wrong). Task 4.4's "Depends on: Task 4.3 (cimports muon point
  function)" still holds — the `cdef` is intact and the Rust `fn` is `pub`.
  No sentence needed patching.
- `rules.md` — rule 1 is satisfied as far as the exception allows (the
  `def` is gone), rule 2 was *not* invoked (no budget widened; the
  `SPECFUN` docstring gained a record, not a number), rule 3 has nothing
  to declare, rule 5's provenance list already names "in-tree cephes
  translations", rule 9's edge guards all survive explicitly.
- `PLAN.md` — the Numerical impact section's two declared exceptions
  (quadrature, `assert`→raise) are untouched; `version_bump: major` is
  unaffected by a bit-equal swap.
- ADR-0002 — §Decision permits "cephes-lineage code (`spec_math` or
  in-tree cephes translations)". The transcription is the second of those
  two, named in the ADR, so no amendment and no new ADR: this is an
  implementation choice inside an accepted boundary, not a change to it.
  `special.rs`'s header carries the upstream citation rule 5 requires.

The one file that *was* factually wrong afterwards is
`references/numerics-replacements.md`, which named `spec_math` as the
replacement for all three special functions and its own vendoring
fallback as a contingency. Patched in this task rather than deferred,
per the canonical-contract rule — a reference that tells the next agent
to reach for `spec_math::li2` is worse than no reference.

## Stale-state sweep

<!-- see docs/agents/doc-consistency.md, "The sweep block" -->

| Check | Command | Result |
| --- | --- | --- |
| Changed-path inventory | `git diff origin/master --name-status \| awk '{print $1}' \| uniq -c` | **21 paths: 16 `M`, 4 `A`, 1 `D`.** (An earlier draft said 20 — it counted a pre-commit `git status --short` in which the deletion was already staged and a stray `-k` from the mutation harness was still present. Re-derived from the diff, which is the artifact under review.) |
| Full diff read | `git diff origin/master --` | reviewed in full; the 4 additions walked as fresh creations |
| Citations resolve | `scripts/agents/check_doc_citations.py --changed-vs origin/master` | `docs scanned: 6`, `in-repo citations checked: 39`, **0 problems** |
| `.pyx` `def` really gone | `rg -n "^def " hazma/spectra/_photon/_muon.pyx` | **no occurrences** |
| No live caller of the deleted `def` | `rg -n "_muon\.dnde_photon\b" -g '!*.so' .` | 3 hits, **all prose**: `references/numerics-replacements.md:298` (Task 2.1 snapshot — marked stale in this task), this note, and `task-2.3-plumbing-test.md`. No code. |
| `cimport`ers unbroken | `rg -n "_photon\._muon cimport" hazma/` | 3 — `hazma/spectra/_photon/_pion.pyx:9`, `hazma/scalar_mediator/scalar_mediator_decay_spectrum.pyx:1`, `hazma/vector_mediator/vector_mediator_decay_spectrum.pyx:9`; all `cdef`s, all intact (and `test_the_cdef_capsules_the_cimporters_need_are_intact` asserts the capsule set) |
| Deleted stub not referenced | `rg -n "_photon/_muon\.pyi" .` | 2 hits, both this task's own bookkeeping; no build file or import |
| Cited line numbers | `rg -n "MASS_E / MASS_MU\|1.0 - r\|spence" hazma/spectra/_photon/_muon.pyx`, `sed -n '16p' hazma/spectra/_photon/_pion.pyx` | in `hazma/spectra/_photon/_muon.pyx`, `:41` is the rest-frame guard, `:88` the in-flight edge and `:113` the `spence` call; `hazma/spectra/_photon/_pion.pyx:16` is `ENG_GAM_MAX_MURF` — all four as cited (an earlier draft said `:37`; corrected) |
| Public name unchanged in docs | `rg -n "dnde_photon_muon" docs/source/` | 10 hits in `spectra.rst`, all the **public** name, which this swap does not change — no edit needed |
| `spec_math::Polylog` no longer used | `rg -n "Polylog\|li2" rust/src/` | 2 hits, both prose in `special.rs`'s "why this is transcribed and not" docs; **no `use`, no call** |
| Blocked-defect count | `rg -n "blocked defects" projects/ docs/` | live working memory reads five in both READMEs and the two "four" sentences are now followed by "a fifth joined in Task 4.3"; `task-4.2-...md:598` keeps its historical "four" as that task's record |
| Follow-up indexed | `rg -n "photon-muon-rest-frame" docs/followups/README.md` | 1 row under Open |
| Preflight | `scripts/agents/preflight.sh --paths … --md …` | **RESULT: PASS**, all eleven rows (no WARN) |
| Numerical-impact statement | see `## Verification` → Numerical impact | **No public value changes** (verified: 144,000-point bit-equality sweep vs the Cython `cdef` through `__pyx_capi__`, 0 mismatching doubles; `pytest -q test/parity` green at `rtol = 1e-13` with a measured difference of exactly zero) |

## Handoff to Next Task

**Read first:** `../../phases/phase-04-spectra-kernels.md` (Goal — the
recipe and the capi-survivor exception), then `../README.md`, then this
note. Task 4.4 (`_photon/_pion`) is next and is the phase's **first `qagp`
consumer**.

**Now safe to assume:**

- `crate::kernels::photon_muon::{dnde_photon_muon, dnde_photon_muon_rest_frame}`
  are `pub`, PyO3-free, and bit-equal to the `cdef`
  `hazma/spectra/_photon/_pion.pyx` still cimports (144,000 points, 0
  mismatches). Task 4.4 should call the Rust `fn` directly as its
  integrand rather than going through Python.
- `crate::special::spence` is **bit-identical to `scipy.special.spence`**
  on the capturing platform (13,000 points, all four branches). Anything
  Phase 05 builds on `bessel_k1`/`bessel_kn` is untouched by that change.
- `hazma._core.photon` serves eight kernels. `rust/src/photon.rs` shows
  both registration shapes: a guard resolved before any element is mapped
  (the tabulated seven) and a kernel that guards per element (this one).
- The corpus's served-roster meta-test no longer assumes a ported `.pyx`
  `def` shares its name with the public entry point.

**Risky / unknown:**

- **`test/test_core_dispatch.py`'s spectra oracle is now `_photon/_pion`,
  which Task 4.4 deletes the `def` of.** Move it again (the last spectra
  `def` to fall is in Task 4.6) or retire
  `TestDeclaredDivergencesFromCython`'s three spectra tests, whose Cython
  side stops existing at the end of this phase.
- **Five blocked defects now share one corpus regeneration** — the boost
  integral (3.4), the positron normalization (4.1), the η′ line weight and
  the φ line energies (both 4.2), and this task's rest-frame endpoint. Do
  not "fix" any in passing.
- **`1/β` amplification is a property of this family, not of `spence`.**
  `_pion` boosts this kernel and `_rho` boosts that, so Task 4.5's nested
  drift analysis should expect the near-rest blocks to be the loud ones
  again. The lesson that generalizes: when a corpus block fails, attribute
  the difference to a single term *before* touching a budget — here that
  turned a proposed 300x widening into a 60-line fix.
- The off-platform budget in `test/test_core_photon_muon.py` is
  **derived, not measured** — Task 4.1's was measured (PR #63) and this
  one reuses its figure from a derivation. PR #67's CI **held it green on
  Linux/glibc across py3.10–3.14**, so it is not too tight; that says
  nothing about the margin, because the assertion reports nothing on
  success. Task 4.1 got its number from an accidental failure. If a later
  task wants the real Linux spread here it has to provoke one.
