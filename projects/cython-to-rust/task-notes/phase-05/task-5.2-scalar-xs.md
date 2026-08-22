# Task 5.2: Scalar cross sections

**Date:** 2026-08-21
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-05-mediator-cross-sections.md`
(Goal, Task 5.2); `../../rules.md` rules 1–4, 6–9
**Related ADRs:** ADR-0001 (framework), ADR-0002 (license-clean numerics
— `spec_math`'s cephes lineage is what `bessel_k1`/`bessel_kn` rest on)
**Depends On:** Task 5.1 (layout, dispatch pattern, complex-arithmetic
shims); Phase 03 (specfun, `qagp`, dispatch)

## Objective

Port the twelve consumed public `def`s of
`hazma/scalar_mediator/_c_scalar_mediator_cross_sections.pyx` to
`hazma._core.scalar_mediator`, drop its unconsumed `sigma_xx_to_all`,
repoint the wrapper, and delete the Cython twin — the largest single
transliteration in the project (1,606 lines, two 90-line expressions).

## Exit Criteria

Copied from the phase file at task start:

- [x] All 11 consumed scalar kernels ported, incl. the two 90-line
      expressions (`__sigma_xpi_to_xpi`, `__sigma_xpi0_to_xpi0`) —
      factor the 8× repeated subexpression into a named local (identical
      arithmetic order; confirm no value shift).
- [x] `sigma_xx_to_all` dropped under the same importer-check rule as
      Task 5.1.
- [x] `thermal_cross_section` on Rust `qagp` (breakpoints
      `[2, ms/mx, 2ms/mx]`).
- [x] `np.log(4)` at line 283 becomes the constant `LN_4` (value change:
      none — see Findings).
- [x] Corpus green; wrapper swapped; twin deleted.

## Inputs Reviewed

- `../../PLAN.md` (Scope, Numerical impact), `../README.md`, `../phase-05/README.md`,
  the phase file, `../../rules.md`.
- [`task-5.1-vector-xs.md`](task-5.1-vector-xs.md)'s Findings and
  `rust/src/kernels/vector_xs.rs`'s module docs — the template, and the
  source of the two complex-arithmetic shims this task reused.
- `../../learnings/phase-04-spectra-kernels.md` §2 (the eight-step swap
  recipe), §3 (contraction quirks — `_photon/_rho.pyx` contracting
  nothing turned out to be the same phenomenon this file shows in one
  kernel) and §4 (test-module shapes).
- `test/parity/{cases,tolerances,generate,test_parity}.py`;
  [`docs/followups/todo/scalar-elastic-cross-sections-cancel-in-atan-difference.md`](../../../../docs/followups/todo/scalar-elastic-cross-sections-cancel-in-atan-difference.md).
- `docs/agents/{lessons,environment,doc-consistency}.md`.

## Findings

### The handoff was wrong: this module *does* go through complex arithmetic

Phase 05's README told the next agent "the scalar module has **no**
`** 1.5`, so it probably does not need [`map_unary_try`] — check
`grep -c SoftComplexToDouble` on the generated C before assuming either
way." The check is what settles it, and it says **six** occurrences, of
which one is a live call site:

```console
$ grep -n '__Pyx_SoftComplexToDouble(' \
    hazma/scalar_mediator/_c_scalar_mediator_cross_sections.c
2417:  static double __Pyx_SoftComplexToDouble(...);       # prototype
5915:  __pyx_t_3 = __Pyx_SoftComplexToDouble(__Pyx_c_quot_double(...
18873: static double __Pyx_SoftComplexToDouble(...) {      # definition
```

Line 5915 is `__sigma_xx_to_s_to_ff`, whose
`(-4 * mf ** 2 + e_cm ** 2) ** 1.5` at `.pyx:31` is easy to miss among
ten other factors. The shipped object confirms it: `nm -u` lists `_cpow`,
and `objdump` shows a `bl ___divdc3` in that kernel and in no other. So
`soft_complex_pow_1_5` and `complex_quotient_real_denominator` were
required here too — without them `sigma_xx_to_s_to_ff` misses
bit-equality at **355 of 935** points on the electron block alone.

That is why the shims moved out of `vector_xs.rs` into a shared
[`crate::kernels::soft_complex`] rather than being copied.

### The fusion rule is a syntactic property of the *C* tree

The transliterator (below) had to decide where clang emits an FMA, and
138 FMA sites across eleven kernels made per-site disassembly reading
impractical. The rule that reproduces every one of them is clang's own,
in `CGExprScalar.cpp`'s `EmitFMulAdd`: **`A + B` contracts when `A` is a
multiply, else when `B` is** — decided on the C tree Cython emits, where
`x ** 2` is `pow(x, 2.0)`, a *call*, and `-x**2` is an `FNeg`.

That single rule explains all three of the puzzles Task 5.1 recorded
separately:

- `-4 * mx**2 + e_cm**2` fuses — its left operand is the multiply
  `-4.0 * pow(mx, 2.0)`.
- `ms**2 - e_cm**2` does not — both operands are `pow` calls.
- `-mpi0**2 + e_cm**2` does not — the left operand is an `FNeg` of a
  call. (Task 5.1's docs attributed this to `mv²` having a second use.
  That is true but not the cause; the operand shape is.)

The rule was not assumed. It was implemented, run over all eleven
kernels, and checked point-for-point against the live Cython **before**
any Rust was written: 10 of 11 came back bit-equal on the first pass and
the eleventh is the finding below.

### One `np.log(4)` boxes half of `sigma_xl_to_xl` into Python

The one kernel the rule got wrong at first, at 713/935 bit-equal and a
worst relative error of **3.1e-5** — large because the same kernel's
`atan` difference cancels.

`.pyx:283` writes `ms**2 * np.log(4)` — a NumPy call inside a `cdef
double` function. Cython therefore evaluates every operation between
that call and the root through `PyNumber_Multiply` / `PyNumber_Add` on
boxed `PyFloat`s, and clang never sees an expression to contract there.
The pure-C operands of those Python operations are still doubles and
still fuse *inside* themselves, which is why the `atan` half of the
kernel is full of FMAs and the `log` tail has almost none. Teaching the
transliterator "a subtree containing a Python-level call does not fuse"
took it to 935/935.

Phase 04 found the same effect in `_photon/_rho.pyx` from a different
cause (untyped `cdef` locals). One stray `np.` does it too.

`log(16)` beside it is the ordinary C call, folded by clang. Both became
named constants; `LN_16` is deliberately **not** spelled `2 * LN_4`,
though on this platform they are the same double (`mutation M6`).

### The `xs_to_xs` threshold branch is a finite limit the corpus samples

`__sigma_xs_to_xs` carries its own closed form at exactly `e_cm = 2 mx`,
where the general expression is 0/0 — the only place in the file where
the `.pyx` supplies the limit rather than the singularity. `2 mx` is a
grid anchor, so the branch is pinned; it is reached only by the
`closed_resonance` model point, the one with `ms < mx`, because the other
two are still below the `mx + ms` threshold there.

Measured, the general expression converges to that branch value
**linearly** — relative gap 1.96e-5, 1.96e-6, 1.96e-7 at offsets 1e-5,
1e-6, 1e-7 — and then stops: the gap *grows* back to 4.0e-7 at an offset
of 1e-10 and 1.9e-6 at 1e-11, which is the `atan` cancellation the
follow-up describes eating the answer. `sigma_xg_to_xg` has the same 0/0
and returns `0.0` there instead, under a comment claiming "complete
destructive interference". The two treatments of the same singularity
disagree, and settling that is the follow-up's job, not this task's.

### Smaller facts

- Six of the eleven kernels ignore couplings their signature declares
  (`CYTHON_UNUSED` in the generated C). `sigma_xx_to_ss`, `sigma_ss_to_xx`
  and `sigma_xs_to_xs` read only `gsxx` of the six couplings.
- The scalar module's `alpha_em` is `1/137.04`, the same third value the
  vector module's own table carries and neither shared header has.
- The wrapper passes `hazma.parameters`' lepton masses into the kernels,
  which are **not** the `.pyx`'s own `me`/`mmu` — 0.51099895 against
  0.510998928. That was true before the port too; the internal
  `sigma_xx_to_all` still uses the module's own. Recorded because the
  new test module had to be written around it.

## Decisions and Implementation Notes

- **The file was transliterated by a program, not by hand.** `rules.md`
  requires the mediator dumps be "scripted or expression-by-expression
  copy, never retyped". A small AST transpiler parses each `.pyx`
  expression and prints Rust, applying the fusion rule above; a
  companion `rustlike.F` class gives a Python double the *Rust* method
  spellings (`mul_add`, `powf`, `sqrt`, `ln`, `atan`), so the emitted
  text is simultaneously valid Rust and valid Python and could be
  checked against the live Cython before any Rust existed. The three
  scripts (`transpile.py`, `rustlike.py`, `ref.py`) were a one-shot
  harness and are **not committed** — they cannot be re-run once the
  `.pyx` is gone, and the same reasoning applied to Task 5.1's
  `scratch/cx2.c`. What is committed is the artifact (`scalar_xs.rs`),
  the rule the harness encodes (that module's docs, three numbered
  rules), and the measurement (below). To redo it, restore the `.pyx`
  with `git show <sha>:hazma/scalar_mediator/_c_scalar_mediator_cross_sections.pyx`.
- **CSE, not hand-hoisting, produced the four `let` bindings** in the two
  90-line kernels. The transpiler counts repeated subtrees across the
  growing forest of bindings and binds the largest first, so
  `coupling_combination` — the factor the `.pyx` repeats eight times — is
  hoisted out of the two prefactors that themselves repeat twice. Pure
  common-subexpression elimination on a parsed expression cannot change
  a value, and the corpus confirms it: bit-equal before and after.
- **The complex shims moved to `crate::kernels::soft_complex`** rather
  than being duplicated. A code move, verified by `cargo test` holding at
  186 across it and the vector corpus cases staying green.
- **The corpus drives the wrapper's short aliases, not the kernels' own
  names.** Unlike the vector wrapper, this one already uses every
  canonical name for a mixin method taking `self`, so it imports the
  kernels as `sig_ff`, `sig_gg` and so on. `Case.function` stays the
  kernel's name (the manifest and the served-roster check read it) and a
  new `Case.attribute` field carries the alias. `hazma._core.
  scalar_mediator` cannot be named directly: a PyO3 submodule has no
  `__file__`, so `assert_module_is_repo_tree` rejects it.
- **`UNCONSUMED_EXPORTS` is now empty**, and the check that walks it
  stays. Both modules that exported a `sigma_xx_to_all` are gone; Phase
  06 has four `.pyx` left that may need it again.
- **The atan-cancellation defect is reproduced, not fixed.** The
  follow-up says doing it *during* Phase 05 "costs almost nothing
  extra". It could not be done inside this PR: `rules.md` rule 1 gates
  the swap on the corpus and rule 2 forbids regenerating it, so a
  stabilised kernel cannot pass the gate that lets it ship; `PLAN.md`'s
  Scope excludes "any physics change" and its Numerical impact section
  declares two exceptions, neither of them this. The follow-up is
  updated with what actually happened, repointed at the Rust, and given
  the two tests that now bracket the question.
- **Two `test_core_dispatch.py` cases retired** with the module they
  compared against. Both were evidence that the port's 0-d and
  sequence widenings were the cross sections' own behavior; the Cython
  side is gone and the port's half is pinned in the new test module.

## Files Changed

### Task 5.2

- `rust/src/kernels/scalar_xs.rs` — **new.** The eleven kernels, the
  private `sigma_xx_to_all`, `thermal_cross_section`, and 15 unit tests.
- `rust/src/kernels/soft_complex.rs` — **new.** `NonRealResult`,
  `ilogb_finite`, `scalbn`, `soft_complex_pow_1_5`,
  `complex_quotient_real_denominator` and their 6 tests, moved out of
  `vector_xs.rs` unchanged.
- `rust/src/kernels/vector_xs.rs` — imports the shims from their new
  home; module docs point there.
- `rust/src/scalar_mediator.rs` — the twelve PyO3 entry points, the
  `TypeError` mapping, the keyword surface.
- `rust/src/{kernels,vector_mediator}.rs` — module registration and the
  `NonRealResult` import path.
- `hazma/scalar_mediator/_c_scalar_mediator_cross_sections.pyx` —
  **deleted** (1,606 lines).
- `hazma/scalar_mediator/_scalar_mediator_cross_sections.py` — imports
  repointed at `hazma._core.scalar_mediator`.
- `setup.py` — extension dropped from the scalar-mediator list.
- `hazma/_core.pyi` — roster comment.
- `test/test_core_scalar_xs.py` — **new**, 79 tests.
- `test/parity/cases.py` — twelve cases repointed via the new
  `Case.attribute`; twelve `PORTED_ENTRY_POINTS` rows;
  `UNCONSUMED_EXPORTS` emptied with the three prose sites that described
  it.
- `test/parity/tolerances.py` — thermal budget tightened; the `QUAD`
  class narrative and `QUAD_RTOL`'s own comment updated now that no case
  holds the opening figure.
- `test/test_core_quad.py` — the scalar thermal integrand rebuilt from
  the Rust kernels; `VECTOR_ME`/`VECTOR_MMU` renamed `MEDIATOR_*`.
- `test/test_core_dispatch.py` — two cases retired with their oracle.
- `docs/followups/todo/scalar-elastic-cross-sections-cancel-in-atan-difference.md`
  — trigger, status, entry points and remedy repointed at the Rust.
- `projects/cython-to-rust/task-notes/{README.md,phase-05/README.md}`,
  this note.

## Verification

Every command below was run in this worktree against an editable install
(`uv pip install -e .`; `hazma.__file__` and `hazma._core.__file__` both
inside the worktree).

- `pytest -q` → **`2091 passed, 15 skipped, 7 warnings in 49.20s`**
  (from `2013 passed, 15 skipped` at Task 5.1's close: +79 new
  `test/test_core_scalar_xs.py`, −2 retired `test_core_dispatch.py`
  cases, +1 replacing one of them).
- `pytest test/parity -q` → **`658 passed, 1 skipped in 36.92s`** — the
  same count as Task 5.1, since the twelve scalar cases were already in
  the corpus and only moved.
- `pytest test/test_theory_aggregation.py -q` → **`69 passed`** — the
  model-layer gate, run either side of the swap and unchanged.
- `cargo test --manifest-path rust/Cargo.toml --no-default-features` →
  **`201 passed`** (from 186 at Task 5.1's close: +15).
- `cargo fmt --manifest-path rust/Cargo.toml --check` — clean.
- `cargo clippy --manifest-path rust/Cargo.toml --all-targets --
  -D warnings` — clean.

What the tests cover, by kind rather than by count:

- **Parity** — the twelve entry points against the stored pre-port
  arrays at three model points each (open, narrow and closed resonance;
  six blocks each for `sigma_xx_to_s_to_ff` and `sigma_xl_to_xl`, which
  are captured once per lepton). Eleven of the twelve are `EXACT` class,
  `rtol = 0`.
- **Against the live Cython, before deletion** — every kernel over every
  corpus grid, measured directly rather than through the corpus, by
  calling both implementations on the same 12,440 abscissae: see
  **Numerical impact**.
- **Against an independent Python implementation** — the four
  re-derivable channels (`f f̄`, `γγ`, both two-pion) rebuilt from the
  physics with no shared code (`ReferenceCrossSections`), at six
  energies spanning four decades, plus a check that the 1e-13 budget
  would notice a coefficient wrong in the sixth digit.
- **Analytic limits (Rust)** — every channel's threshold to the exact
  `0.0` (including the three that open at a pion or lepton mass rather
  than at `2 m_x`), the `f f̄` high-energy limit `σ s → g² g² m_f²/16π
  v_h²`, the `1/Γ²` resonance peak, the identical-particle factor 2
  between the two annihilation pion channels and the charge-sum factor 2
  between the two elastic ones, the sum-of-channels identity, the
  `x > 300` cutoff, the `⟨σv⟩` fall with `x` over four decades, and the
  continuity of the `max(50/x, 100)` branch switch.
- **Conditioning (Rust)** — `sigma_xs_to_xs`'s threshold branch as the
  general expression's limit, over the three decades where it converges.
- **The argument surface (Python)** — scalar/array/0-d dispatch, the
  array path against the scalar path bit-for-bit for all eleven, rank
  and type errors, every argument by keyword, each of the fifteen
  declared-but-unused couplings proven inert and each read one proven
  live, the twelve wrapper aliases, the served roster, the model layer's
  twelve mixin methods against the kernels, and that the `.pyx` and its
  `setup.py` entry are gone.

**Deferred:** the relic-density end-to-end check and the headline
benchmark are Task 5.3's exit criteria, not this task's. No performance
claim is made here, so `rules.md` rule 12 is not engaged.

### Mutation campaign

Six mutants, each built and run against `cargo test`,
`pytest test/parity -k scalar` and `pytest test/test_core_scalar_xs.py`:

| # | mutation | killed by |
| --- | --- | --- |
| M1 | `soft_complex_pow_1_5(t)` → `t.powf(1.5)` | parity (6 blocks) |
| M2 | `complex_quotient_real_denominator(n, d)` → `n / d` | parity (6) |
| M3 | the boxed `4mx² − s + ms²·ln4` fused into an FMA | parity (3) |
| M4 | `x > 300` returns `f64::MIN_POSITIVE` instead of `0.0` | cargo + parity (3) |
| M5 | `sigma_xx_to_all` drops the `π⁺π⁻` channel | cargo + parity (3) |
| M6 | `LN_16` respelled `2.0 * LN_4` | **survives — equivalent** |

M6 is not a hole: `2 * log(4)` and `log(16)` are the same double on this
platform (`2.772588722239781` both), so the mutant computes the same
program. `log_constants_match_libm` still pins `LN_16` against
`16.0_f64.ln()`, which is the statement worth making. M1 and M2 are the
two that would have shipped silently had the handoff's "no `** 1.5`
here" been believed; M3 is the one the fusion rule got wrong before the
Python-boxing case was added, and it is a real bug this campaign
re-runs.

## Numerical impact

**All eleven closed-form kernels are bit-equal to the Cython at every
one of the 12,155 values the parity corpus compares them on**, on the
capturing platform, at `rtol = 0`. Measured directly against the live
Cython before its deletion, sweeping each case's whole
grid — 935 points per single-block case, 1,870 for the two captured per
lepton (the harness is the uncommitted one described under Decisions):

| entry point | compared | bit-equal | worst relative |
| --- | --- | --- | --- |
| `sigma_xx_to_s_to_ff` (e, μ) | 1,870 | 1,870 | 0 |
| `sigma_xx_to_s_to_gg` | 935 | 935 | 0 |
| `sigma_xx_to_s_to_pi0pi0` | 935 | 935 | 0 |
| `sigma_xx_to_s_to_pipi` | 935 | 935 | 0 |
| `sigma_xx_to_ss` | 935 | 935 | 0 |
| `sigma_ss_to_xx` | 935 | 935 | 0 |
| `sigma_xl_to_xl` (e, μ) | 1,870 | 1,870 | 0 |
| `sigma_xpi_to_xpi` | 935 | 935 | 0 |
| `sigma_xpi0_to_xpi0` | 935 | 935 | 0 |
| `sigma_xg_to_xg` | 935 | 935 | 0 |
| `sigma_xs_to_xs` | 935 | 935 | 0 |
| **`thermal_cross_section`** | **285** | **104** | **3.12e-15** |
| total | 12,440 | 12,259 | — |

`thermal_cross_section` is the only entry point that moves. Worst point:
`open_resonance`, `x = 0.116895`, `5.560975522996041e-09` →
`5.560975522996024e-09`. That is below `rules.md` rule 3's 1e-12
declaration threshold, so it is recorded here and in the working-memory
log without a CHANGELOG line of its own. As with the vector twin the
drift is the Bessel functions rather than the integrator.

**One budget tightened, none widened**: the thermal case from
`QUAD_RTOL` (1e-8) to `PORTED_QUAD_RTOL` (1e-12), 320× headroom over the
measurement. That was the last case holding the opening figure, so
`QUAD_RTOL` now has no holder — its comment says so and says what it is
still for.

`np.log(4)` → `LN_4` moves nothing: `math.log(4.0)` is
`1.3862943611198906` and that is the double NumPy returned.

No other public value moves: eleven kernels bit-equal, and
`pytest test/test_theory_aggregation.py -q` (69 model-layer identities,
the only numerical gate not scoped to the capturing platform) unchanged
either side of the swap.

## Open Questions

- [`scalar-elastic-cross-sections-cancel-in-atan-difference.md`](../../../../docs/followups/todo/scalar-elastic-cross-sections-cancel-in-atan-difference.md)
  is now a standalone declared numerical change rather than a
  ride-along, and it is updated to say so. Four kernels
  (`sigma_xl_to_xl`, `sigma_xpi_to_xpi`, `sigma_xpi0_to_xpi0`,
  `sigma_xg_to_xg`) carry the defect into Rust unchanged.
- The two `e_cm = 2 mx` treatments in this file disagree —
  `sigma_xs_to_xs` supplies the finite limit, `sigma_xg_to_xg` returns
  `0.0`. Folded into the same follow-up, with the two tests that bracket
  it named there.
- Task 5.1's three follow-ups are unchanged and none blocked this task.
  The thermal quadrature still does not converge; Task 5.3's relic sweep
  is where its size shows.

## Plan Impact

**Impact Level:** None.

Re-read against what shipped: the phase file's Goal, Task 5.2's five
exit criteria, Task 5.3's, the phase Exit Criteria, and
ADR-0001/0002's gate sentences. Every one is still factually correct.
The Goal's accounting held under the re-run importer check — 13 defs, 12
consumed — and the "no hazma cimports" claim held: the file had no
`.pxd` and exported no capsules, so it went whole.

The one canonical text that *was* wrong is not a plan file but the
phase's own working memory, and it is this task's to fix: the Handoff
section of [`README.md`](README.md) told the next agent the scalar module
had no `** 1.5`. It has one. Corrected in place rather than deferred, per
the canonical-contract rule, and the correction is why
`crate::kernels::soft_complex` exists.

`../../PLAN.md`'s **Numerical impact** section anticipated this shape —
"closed-form kernels: ≤1e-13" — and the measurement came in under it
(closed forms bit-equal, quadrature at 3.1e-15). No revision needed.
`version_bump: major` is unaffected: it is driven by API removals, and
this task removes only an export nothing imported.

## Stale-state sweep

```console
$ grep -rn "_c_scalar_mediator_cross_sections" --include="*.py" --include="*.rs" \
    --include="*.pyi" --include="*.pyx" --include="*.md" . \
    | grep -vE "^\./(projects|docs/followups|\.venv|rust/target)/"
setup.py:73                          hazma/scalar_mediator/_scalar_mediator_cross_sections.py:3
test/test_core_scalar_xs.py:39,390,395   test/test_core_dispatch.py:763,770
test/test_core_quad.py:82,569,630    test/parity/reference.py:32,87
test/parity/tolerances.py:560        test/parity/cases.py:687,697,1550,1557-1601
rust/src/quad.rs:41                  rust/src/special.rs:39
rust/src/scalar_mediator.rs:10       rust/src/kernels/scalar_xs.rs:2,99
```

Every surviving mention is **provenance** — a line-numbered citation
into a file this same commit deletes, which is the pattern
`PORTED_ENTRY_POINTS` exists to keep honest and which Tasks 4.2, 4.6 and
5.1 established. No live import remains. Five were *claims about the
present* rather than citations, and all five were repointed in this
change: `rust/src/quad.rs:41` and `rust/src/special.rs:39` now carry
"(ported, Task 5.2 — now `crate::kernels::scalar_xs`)" the way Task 5.1
annotated their vector rows; `test/parity/reference.py:32` says the path
is provenance and names `git show` as the way to read it; and the two
follow-ups that said "until Task 5.2 ports it"
([the `2 m_x` raise](../../../../docs/followups/todo/vector-cross-sections-raise-at-the-two-mx-threshold.md),
[the unconverged quadrature](../../../../docs/followups/todo/thermal-cross-section-quadrature-never-converges.md))
now point at the Rust and say what porting settled.

```console
$ grep -rn "\bsigma_xx_to_all\b" hazma/ --include="*.py" --include="*.pyx" --include="*.pyi"
$ echo $?
1
```

The exit-criterion importer check, re-run at execution time rather than
inherited: **zero hits anywhere under `hazma/`** now that the last
defining `.pyx` is gone. Before the deletion it was two, both inside
that file (its own `__vec_` loop and its own `def`) — quoted in full in
the PR body. Nothing could reach either export, so dropping both removes
nothing a caller had. `UNCONSUMED_EXPORTS` is correspondingly empty.

```console
$ grep -nE "TODO|FIXME|breakpoint\(\)|\bpdb\b|print\(" \
    rust/src/kernels/scalar_xs.rs rust/src/kernels/soft_complex.rs \
    rust/src/scalar_mediator.rs test/test_core_scalar_xs.py
$ echo $?
1
```

Nothing introduced. Unlike Task 5.1's sweep there is not even a quoted
`TODO`: the scalar `.pyx` carried none above the expressions this task
transcribed.

```console
$ pytest -q | tail -1
2091 passed, 15 skipped, 7 warnings in 49.20s

$ pytest test/parity -q | tail -1
658 passed, 1 skipped in 34.07s

$ pytest test/test_theory_aggregation.py -q | tail -1
69 passed, 2 warnings in 2.37s

$ cargo test --manifest-path rust/Cargo.toml --no-default-features | grep 'test result' | head -1
test result: ok. 201 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out

$ cargo fmt --manifest-path rust/Cargo.toml --check   # no output
$ cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings
    Finished `dev` profile [unoptimized + debuginfo] target(s)

$ python -c "import hazma, hazma._core; print(hazma.__file__, hazma._core.__file__)"
/…/task-5.2-scalar-cross-sections/hazma/__init__.py
/…/task-5.2-scalar-cross-sections/hazma/_core.abi3.so

$ python -c "import sys; sys.path.insert(0,'.'); from test.parity.cases import \
    rust_core_kernels, PORTED_ENTRY_POINTS; print(len(rust_core_kernels()), len(PORTED_ENTRY_POINTS))"
34 34

$ git ls-files 'hazma/**/*.pyx' | wc -l ; git ls-files 'hazma/**/*.pxd' | wc -l
9
8
```

`scripts/agents/preflight.sh` — **`RESULT: PASS`**, all eleven gates,
with `--paths` over the six changed/added Python files whose configured
ruff is clean and `--md` over the seven changed Markdown files.
Including the seventh changed Python file — the wrapper — flips gate 3
to FAIL on **findings that are the trunk's, not this change's**, proven
per file rather than asserted by re-running the configured rule set
against `git show origin/master:<path>`:

| file | trunk | branch |
| --- | --- | --- |
| `setup.py` | 0 | 0 |
| `test/parity/cases.py` | 0 | 0 |
| `test/parity/tolerances.py` | 0 | 0 |
| `test/parity/reference.py` | 0 | 0 |
| `test/test_core_quad.py` | 0 | 0 |
| `test/test_core_dispatch.py` | 0 | 0 |
| `hazma/scalar_mediator/_scalar_mediator_cross_sections.py` | **65** | **64** |
| `test/test_core_scalar_xs.py` (new) | — | 0 |

A delta of **−1**, and the new file is clean. The 64 are the untouched
`ScalarMediatorCrossSections` mixin's missing annotations and docstring
shapes; cleaning them would be a several-hundred-line unrelated diff in
a physics file, which is what
[`preflight-isort-ruff-red-on-trunk.md`](../../../../docs/followups/todo/preflight-isort-ruff-red-on-trunk.md)
exists to decide once rather than per task.
`ruff check --isolated --select E9,F63,F7,F82 --exclude hazma/experimental
--exclude notebooks .` — the form CI runs — **passes**.

**Numerical-impact statement:** eleven closed-form kernels bit-equal to
the Cython at all 12,155 compared values on the capturing platform;
`thermal_cross_section` moves by at most **3.12e-15** relative over 285
pinned values (104 of them bit-equal), inside the 1e-12 this change
tightens its budget to and below `rules.md` rule 3's 1e-12 declaration
threshold. No other public value moves —
`pytest test/test_theory_aggregation.py -q` → `69 passed`, unchanged
either side of the swap. One budget tightened, none widened.

**Bookkeeping consistency:** this note's `**Status:** Complete`, the
Task 5.2 row in [`README.md`](README.md), the Phase 05 row in
[`../README.md`](../README.md), and
`../../phases/phase-05-mediator-cross-sections.md`'s `status: In
Progress` all agree: two of three tasks done, phase open.
`../../PLAN.md`'s phase table is untouched, which is correct — it moves
when the phase closes. No new follow-up was filed; one existing entry
was updated in place because this task closed the window it named.

## Handoff to Next Task

**Read first:** this note's Findings, then
`rust/src/kernels/scalar_xs.rs`'s module docs, whose three numbered
rules are the transliteration contract Phase 06 inherits.

**Currently safe to assume:**

- **The fusion rule is now stated as a rule, not as a list of cases.**
  `A + B` fuses when `A` is a multiply in the *C* tree, else when `B`
  is; `x ** n` is a `pow` call there and never a multiply; a leading
  unary minus blocks it; and any subtree containing a Python-level call
  is boxed and fuses nowhere on its path to the root. Phase 06's four
  `.pyx` are spectrum modules with the same Mathematica provenance.
- `crate::kernels::soft_complex` is shared and is where the next
  `SoftComplexToDouble` finding goes. **Run
  `grep -c SoftComplexToDouble` on the generated C of every remaining
  `.pyx` before assuming it has none** — this task is the second time
  that grep changed the answer.
- The layout is settled and unchanged: kernels in
  `crate::kernels::<name>_xs`, PyO3 in `crate::<model>_mediator`, module
  constants beside the kernels.
- `Case.attribute` exists for a wrapper that cannot re-export a kernel
  under its own name. Phase 06's mediator spectrum wrappers may need it.
- `UNCONSUMED_EXPORTS` is empty and its check is a no-op today.

**Currently risky / unknown:**

- **Task 5.3 is the phase's remaining task and owns both deferred
  items**: the relic-density end-to-end check through
  `hazma/relic_density/_thermal_functions.py`, and the headline
  benchmark. Both thermal cross sections are now Rust, so the relic path
  no longer re-enters Python per quadrature node — but the benchmark
  must be taken from a **release** build (`docs/followups/todo/
  editable-installs-build-the-rust-extension-in-debug.md`), or it will
  say the port is ~20× slower.
- The scalar `thermal_cross_section` returns exactly `0.0` above
  `x = 300` where the vector saturates. Task 5.3's relic sweep crosses
  that boundary; the corpus pins both and they must not be unified.
- The unconverged thermal quadrature (Task 5.1's finding) is now on
  *both* mediator paths. Its relic-density consequence is unmeasured and
  is Task 5.3's to measure.
