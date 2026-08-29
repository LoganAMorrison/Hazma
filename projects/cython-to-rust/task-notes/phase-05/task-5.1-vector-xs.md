# Task 5.1: Vector cross sections (template)

**Date:** 2026-08-20
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-05-mediator-cross-sections.md`
(Goal, Task 5.1); `../../rules.md` rules 1–4, 6–9, 12
**Related ADRs:** ADR-0001 (framework), ADR-0002 (license-clean numerics
— `spec_math`'s cephes lineage is what `bessel_k1`/`bessel_kn` rest on)
**Depends On:** Phase 03 (specfun, `qagp`, dispatch)

## Objective

Port the six consumed public `def`s of
`hazma/vector_mediator/_c_vector_mediator_cross_sections.pyx` to
`hazma._core.vector_mediator`, drop its unconsumed `sigma_xx_to_all`,
repoint the wrapper, and delete the Cython twin — opening Phase 05 and
establishing the layout Task 5.2 copies.

## Exit Criteria

Copied from the phase file at task start:

- [x] The 5 consumed vector kernels transliterated **mechanically**;
      tiers 2–3 replaced by the generic dispatch helper.
- [x] `thermal_cross_section` on Rust `qagp` (breakpoints
      `[2, mv/mx, 2mv/mx]`, incl. the out-of-interval regime) +
      `bessel_k1`/`bessel_kn`.
- [x] `sigma_xx_to_all` dropped after re-running the importer check.
- [x] Corpus green across the parameter grid incl. near-resonance.
- [x] Wrapper `_vector_mediator_cross_sections.py` swapped; Cython twin
      deleted.

## Inputs Reviewed

- `../../PLAN.md` (Scope, Numerical impact), `../README.md`,
  `../phase-05/README.md`, the phase file, `../../rules.md`.
- `../../learnings/phase-04-spectra-kernels.md` — §2's eight-step swap
  recipe, §3's contraction quirks, §4's test-module shapes. All four
  carried directly; §2's step 1 (read the disassembly before writing
  Rust) is what made this task's five kernels bit-equal on the first
  build.
- `test/parity/{cases,tolerances,generate,test_parity}.py` — the corpus
  contract, including `evaluate_block`'s *pinned raises*, which turned
  out to govern this task (see Findings).
- `docs/agents/{lessons,environment,doc-consistency}.md`.

## Findings

### The `**` operator compiled to complex arithmetic

The load-bearing discovery, and it is invisible in the `.pyx`.
`__sigma_xx_to_v_to_pipi` and `__sigma_xx_to_v_to_pi0v` raise a
kinematic factor to the power `1.5`; Cython 3's default `cpow` semantics
say a `double ** double` may be complex, so it compiles the **whole
enclosing expression** in `double _Complex` and converts back with
`__Pyx_SoftComplexToDouble`. `grep -c SoftComplexToDouble` over the
generated C finds exactly two call sites, and the shipped object agrees:
both call `cpow` and compiler-rt's `___divdc3`, the other three kernels
call neither.

That is not a detail a real-arithmetic transliteration can absorb.
Measured in C on the capturing platform over 3.7M logarithmically spaced
arguments (`scratch/cx2.c`, reproduced in the module docs):

| comparison | fraction differing | worst relative |
| --- | --- | --- |
| `cpow(t+0i, 1.5+0i)` vs `pow(t, 1.5)` | 90% | 9.0e-15 |
| `cpow(t+0i, 1.5+0i)` vs `t·√t` | 90% | 9.1e-15 |
| `cpow(t+0i, 1.5+0i)` vs `exp(1.5·log t)` | **0** | **0** |
| `(a+0i)/(c+0i)` vs `a/c` | 32% | 4.0e-16 |

So `cpow` at zero imaginary part **is** `cexp(w·clog(z))` — bit-for-bit
— and `exp(1.5·ln t)` reproduces it exactly. `__divdc3` needed a
faithful port of its scaled `(a·c′)/(c′·c′)`, which is C99 Annex G
§G.5.1's recommended practice (Smith 1962), written from that
specification rather than from any implementation's source (ADR-0002).
Both live in `rust/src/kernels/vector_xs.rs`.

The corpus's `EXACT` class docstring claimed the class "reaches only
`libc.math`". That was false for these two members, and the docstring is
corrected in `test/parity/tolerances.py` — the class still holds at
`rtol = 0`, but for a different reason than it recorded.

### The Cython raises `TypeError` at `e_cm = 2 m_x`, and the corpus pins it

At the annihilation threshold `√(e_cm² − 4 m_x²)` is exactly zero, so
the whole denominator is. In the four real kernels that is an `inf` (or
a `nan` for `sigma_xx_to_vv`, which multiplies by the same vanishing
root); in the two complex ones `__divdc3`'s zero-denominator recovery
returns `(±inf, nan)`, `__Pyx_SoftComplexToDouble` sees a non-zero
imaginary part and raises. `test/parity/generate.py` documents this in
`evaluate_block` and the manifest records three `raises` entries, which
`test_parity.py` **replays rather than skips** — so reproducing the
raise was a hard requirement, not a nicety.

That drove the one new piece of dispatch machinery, `map_unary_try`
(`rust/src/dispatch.rs`): the fourth live entry-point shape, a kernel
that can fail at some arguments. One bad element takes the whole array
down in both languages — the Cython's `__vec_*` loop jumps to its error
label on the first failing index rather than filling that slot.

Filed as [a follow-up](../../../../docs/followups/todo/vector-cross-sections-raise-at-the-two-mx-threshold.md):
one kinematic point, six entry points, three different answers, one of
which is an exception.

### `thermal_cross_section` never converges

The `.pyx` passes `quad` neither `epsabs` nor `epsrel`, so it runs at
scipy's default `epsabs = 1.49e-8` against an integrand whose integral
is of order `1e-27`. The absolute criterion is met by the first
Gauss–Kronrod pass and QUADPACK returns on its initial three-interval
partition. Measured against the same integrand and integrator at
`epsabs = 0, epsrel = 1e-11`, the shipped answer is **0.5%–5% off** for
every `x ≳ 5`, i.e. across the whole freeze-out region. Filed as
[a follow-up](../../../../docs/followups/todo/thermal-cross-section-quadrature-never-converges.md);
Task 5.3's relic sweep is where the downstream size shows.

### `pip install -e .` builds the extension in debug

`setuptools_rust` infers `debug = self.inplace or self.debug`, and an
editable install passes `--inplace`. The first benchmark taken this way
said the port was **20× slower** than the Cython. In release it is
1.1×–3.2× faster. Filed as
[a follow-up](../../../../docs/followups/done/editable-installs-build-the-rust-extension-in-debug.md)
rather than fixed here: `[profile.release]`'s `lto = true` +
`codegen-units = 1` make a one-file rebuild cost 64 s against ~10 s, so
the choice is a project-wide dev-loop decision, not a Task 5.1 one.

### Smaller facts

- `clang` folds `pow(x, 2.0)` to `x·x` and `pow(M_PI, 4.0)` / `(…, 5.0)`
  to immediates, but leaves `pow(x, 3.0)` and `pow(x, 4.0)` as libm
  calls — `_pow` is in the shipped object's lazy-bind table, reached 8×
  from `vv`, 4× from `pi0g` and 1× from `pi0v`. `x.powf(3.0)` is
  therefore **not** interchangeable with `x·x·x` here.
- `pow(π,4)` happens to equal the left-associated `π·π·π·π` on this
  platform but not `(π·π)·(π·π)`; `pow(π,5)` equals neither product.
  Both guards are in `tests::pi_powers_match_libm`.
- The unfused/fused split is per-expression, as Phase 04 found. In this
  file `−4 m² + e_cm²` is fused everywhere it appears, but the otherwise
  identical-looking `−m_π0² + e_cm²` in `pi0g` is **not**: a coefficient
  of `−1` gives clang a negation to fold into the subtraction and no
  multiply-add to form. `m_v² − e_cm²` is unfused in all five kernels
  because `m_v²` has a second use.
- The `.pyx`'s six module constants are its own `cdef double`s, not
  either shared `.pxd`, so they live in the kernel module rather than in
  `constants::derived` — which `test/test_core_constants.py` scores
  against surviving `.pyx` files, and this one is deleted. Four match
  `constants::legacy`; `fpi` is in no shared table and `alpha_em` is
  `1/137.04`, a third value beside `pdg`'s and `legacy`'s.

## Decisions and Implementation Notes

- **Reproduce `cpow` and `__divdc3` rather than widen the budget.**
  `rules.md` rule 2 permits a widening with a written justification, and
  the alternative here was a ~1e-14 budget on two `EXACT` cases. Both
  routines turned out to be exactly reproducible in ten lines, so the
  stronger gate was also the cheaper one. Nothing was widened.
- **The `TypeError` keeps its type and loses its wording.** Cython's
  message advises `use 'cython.cpow(True)'`, which is wrong advice in a
  tree with no Cython and worse after Phase 07. `crate::dispatch`'s rule
  is that an explicit raise keeps its *type*; the corpus records only
  the type (`type(err).__name__`). Recorded as a user-visible string
  change in the PR body.
- **The four kernels that ignore `gvss`/`gvee`/`gvmumu` drop them from
  their Rust signatures**, and the PyO3 layer still accepts all ten by
  keyword — the public signature is unchanged. `TestTheUnusedCouplings`
  in `test/test_core_vector_xs.py` is the only layer that can check
  that, since the kernels no longer take the arguments.
- **`sigma_xx_to_all` is dropped from the public surface and kept
  private.** The plan says "dropped, not ported"; the thermal integrand
  is the only caller it ever had, so it survives as a private helper.
- **The corpus cases point at the wrapper, under the kernels' own
  names.** Unlike the spectra wrappers this module defines no function of
  its own (its surface is a mixin class taking `self`), so it re-exports
  the six from `hazma._core.vector_mediator` under their canonical names
  and keeps its short `sig_*` aliases as assignments beside them.
  `test_the_served_roster_is_exactly_the_ported_entry_points` compares
  the served leaf names against `Case.function`, which is what forced the
  canonical spelling; `TestTheWrapperReExports` pins each alias.
- **PR #75 review round 1** — one blocking finding, a stale pinned-value
  count. `rust/src/kernels/vector_xs.rs:61` said 4,667 where this note
  said 5,667. Re-derived from the stored corpus rather than reconciled
  to either: **5,814 stored, less 3 replayed raises, is 5,811**. Both
  circulating figures were short because they counted only the swept
  grids and not the 144 scalar-probe values `test_parity.py:243` sends
  through the same budget — `scalar_values` is not in its `ABSCISSAE`
  set. All six occurrences swept and corrected; the class is cited on
  `[hand-written-population-in-a-derived-check]` in
  [`lessons.md`](../../../../docs/agents/lessons.md). Nothing executable
  changed.

## Files Changed

### Task 5.1

- `rust/src/kernels/vector_xs.rs` — **new.** The six kernels, the two
  complex-arithmetic shims, `ilogb`/`scalbn`, and 15 unit tests.
- `rust/src/vector_mediator.rs` — the six PyO3 entry points, the
  `TypeError` mapping, the keyword surface.
- `rust/src/kernels.rs` — register `vector_xs`; module docs.
- `rust/src/dispatch.rs` — **new** `map_unary_try`, the fallible
  entry-point shape.
- `hazma/vector_mediator/_c_vector_mediator_cross_sections.pyx` —
  **deleted** (674 lines).
- `hazma/vector_mediator/_vector_mediator_cross_sections.py` — imports
  repointed at `hazma._core.vector_mediator`.
- `setup.py` — extension dropped from the vector-mediator list.
- `hazma/_core.pyi` — roster comment.
- `test/test_core_vector_xs.py` — **new**, 78 tests.
- `test/parity/cases.py` — six cases repointed; six
  `PORTED_ENTRY_POINTS` rows; the vector `UNCONSUMED_EXPORTS` row
  removed with its `.pyx`, and the three prose sites that said "the
  two `sigma_xx_to_all` exports" updated with it.
- `test/parity/README.md` — the same "two exports" sibling.
- `test/parity/tolerances.py` — thermal budget tightened; `EXACT`-class
  docstring corrected.
- `test/test_core_quad.py` — the vector thermal integrand rebuilt from
  the Rust kernels, since its `sigma_xx_to_all` oracle is gone.
- `docs/followups/todo/{vector-cross-sections-raise-at-the-two-mx-threshold,
  thermal-cross-section-quadrature-never-converges,
  editable-installs-build-the-rust-extension-in-debug}.md` + index rows.
- `docs/agents/lessons.md` — PR #75 cited on
  `[hand-written-population-in-a-derived-check]` (review round 1).
- `projects/cython-to-rust/phases/phase-05-mediator-cross-sections.md` —
  `status: In Progress`.
- `projects/cython-to-rust/task-notes/{README.md,phase-05/README.md}`,
  this note.

## Verification

Every command below was run in this worktree against an editable install
(`uv pip install -e .`, `hazma.__file__` inside the worktree).

- `pytest -q` → **`2013 passed, 15 skipped, 7 warnings in 131.71s`**
  (from `1935 passed, 15 skipped` at Phase 04's close: +78, exactly the
  new `test/test_core_vector_xs.py`).
- `pytest test/parity -q` → **`658 passed, 1 skipped in 83.95s`**. The
  six vector cases are green at their budgets, including the three
  pinned `TypeError` replays.
- `pytest test/test_theory_aggregation.py -q` → **`69 passed`** — the
  model-layer gate, run either side of the swap and unchanged.
- `cargo test --manifest-path rust/Cargo.toml --no-default-features` →
  **`186 passed`** (from 169 at Phase 04's close: +17).
- `cargo fmt --manifest-path rust/Cargo.toml --check` — clean.
- `cargo clippy --manifest-path rust/Cargo.toml --all-targets --
  -D warnings` — clean.
- `scripts/agents/preflight.sh` — **`RESULT: PASS`**, all eleven gates,
  with `--paths` over the five changed Python files whose configured
  ruff is clean and `--md` over the nine changed Markdown files.
  Including the sixth changed Python file — the wrapper — flips gate 3
  to FAIL on **28 findings that are the trunk's, not this change's.**
  Proven per file rather than asserted, by re-running the configured
  rule set against `git show origin/master:<path>`:

  | file | trunk | branch |
  | --- | --- | --- |
  | `setup.py` | 0 | 0 |
  | `test/parity/cases.py` | 0 | 0 |
  | `test/parity/tolerances.py` | 0 | 0 |
  | `test/test_core_quad.py` | 0 | 0 |
  | `hazma/vector_mediator/_vector_mediator_cross_sections.py` | **28** | **28** |
  | `test/test_core_vector_xs.py` (new) | — | 0 |

  Zero delta, and the new file is clean. The 28 are in the untouched
  `VectorMediatorCrossSections`
  class's missing annotations and docstring shapes, and cleaning them
  would be a 400-line unrelated diff in a physics file — which is what
  [`preflight-isort-ruff-red-on-trunk.md`](../../../../docs/followups/todo/preflight-isort-ruff-red-on-trunk.md)
  exists to decide once rather than per task. One regression *was*
  introduced and fixed rather than argued away: moving the `sig_*`
  aliases above the module's remaining imports added three `E402`s, so
  they moved below every import instead.
  `ruff check --isolated --select E9,F63,F7,F82 --exclude
  hazma/experimental --exclude notebooks .` — the form CI runs —
  **passes**.

What the tests cover, by kind rather than by count:

- **Parity** — the six entry points against the stored pre-port arrays
  at three mediator model points each (open, narrow and closed
  resonance; six blocks for `sigma_xx_to_v_to_ff`, which is captured
  once per lepton). For the five closed forms that is **5,814 stored
  values** — 5,670 on the swept grids and 144 on the scalar probes,
  which `test_parity.py` holds to the same budget because
  `scalar_values` is not in its `ABSCISSAE` set — of which 3 stand in
  for a replayed raise rather than for a number, leaving **5,811
  compared numerically**. `thermal_cross_section` adds 285 more, on a
  grid with no scalar branch.
- **Against an independent Python implementation** — all five closed
  forms re-derived from the physics with no shared code
  (`ReferenceCrossSections`), at six energies spanning four decades.
- **Conditioning** — `sigma_xx_to_vv`'s high-energy cancellation, whose
  law (`difference ∝ ε/c(s)`) is asserted across four decades rather
  than absorbed by a tolerance.
- **Analytic limits (Rust)** — the `f f̄` channel's `σ·s → g⁴/12π`
  high-energy limit, the propagator on resonance, the sum-of-open-
  channels identity, each channel's threshold to the exact `0.0`.
- **The complex shims (Rust)** — that `soft_complex_pow_1_5` is not
  `powf` and not `t√t`; that `complex_quotient_real_denominator` is not
  `a/c` and is within 2 ulp of it; `ilogb`/`scalbn` through the
  subnormals.
- **The thermal average (Rust)** — saturation above `x = 300` (exact
  equality at 300, 301, 1e3, 1e6); the integral against a
  Simpson reference under the branch-point-removing substitution
  `z = √(4+w²)`, at two standards — 2% for the shipped tolerances and
  1e-6 for the same integrator at a convergent one.
- **The argument surface (Python)** — scalar/array/0-d dispatch, the
  array path against the scalar path bit-for-bit, rank and type errors,
  every argument by keyword, the six wrapper aliases, and that the
  `.pyx` and its `setup.py` entry are gone.

**Deferred:** the relic-density end-to-end check and the headline
benchmark are Task 5.3's exit criteria, not this task's. The benchmark
below is recorded anyway because it turned up the debug-build finding.

### Mutation campaign

Six mutants, each built and run against `cargo test`, `pytest
test/parity -k vector` and `pytest test/test_core_vector_xs.py`:

| # | mutation | killed by |
| --- | --- | --- |
| M1 | `exp(1.5·ln t)` → `t.powf(1.5)` | cargo + parity |
| M2 | `complex_quotient_real_denominator` → `a / c` | parity |
| M3 | thermal weight `bessel_k1(xnew·z)` → `bessel_k1(x·z)` | cargo + parity |
| M4 | `vv` log argument unfused | parity |
| M5 | `pi0g` threshold `m_π0` → `2 m_π0` | cargo + parity |
| M6 | `pi0g` reads `gvss` (PyO3 layer) | `test_core_vector_xs` (7 failures) |

No survivors. M3 is not hypothetical: it was a **real bug in this
port**, found by the corpus's `x = 1000` grid point, and the mutation
re-runs it. M2 and M4 are killed only by the corpus, which is the
expected division of labour — `test_core_vector_xs.py` compares against
a reference that shares the *formula*, so an arithmetic-order change
sits inside its 1e-13 budget by construction.

## Numerical impact

**All five closed-form kernels are bit-equal to the Cython at every one
of the 5,811 values the parity corpus compares them on**, on the
capturing platform, at `rtol = 0`. Verified with
`pytest test/parity -q -k vector` → `351 passed`, and measured directly
against the Cython before its deletion (`scratch/ref.py`, sweeping each
case's whole grid — 945 points for the four single-block cases, 1,890
for `sigma_xx_to_v_to_ff`, which is captured per lepton): `0` of them
differ, for every one of the five.

The count is derived rather than quoted, because two wrong figures were
in circulation before PR #75's review round 1 — 4,667 in the Rust module
docs and 5,667 in this note. From the stored corpus:
`sigma_xx_to_v_to_ff` holds 1,890 array-path values and 48 scalar-probe
values across six blocks; the other four hold 945 and 24 each across
three; total **5,814 stored**, of which `pipi` contributes 2 and `pi0v`
1 position that stand in for a replayed `TypeError` rather than for a
number. **5,814 − 3 = 5,811 compared numerically.** The scalar probes
belong in that count: `test_parity.py:243` sends every array not in
`ABSCISSAE` through the case's value budget, and `scalar_values` is not
in it.

`thermal_cross_section` moves, and is the only one that does:

| case | worst relative | bit-equal | budget |
| --- | --- | --- | --- |
| `cross_sections.vector.thermal_cross_section` | **2.06e-14** | 64 / 285 | `PORTED_QUAD_RTOL` = 1e-12 |

Worst point: `open_resonance`, `x = 0.298`, `9.316997739611058e-08` →
`9.316997739610866e-08`. The drift is the Bessel functions rather than
the integrator — `bessel_kn(2, ·)` agrees with scipy to 8.9e-16 and the
prefactor squares it — and it is below `rules.md` rule 3's 1e-12
declaration threshold, so it is recorded here and in the working-memory
log without a CHANGELOG line of its own.

**One budget tightened, none widened**: the thermal case from
`QUAD_RTOL` (1e-8) to `PORTED_QUAD_RTOL` (1e-12), 49× headroom over the
measurement.

No other public value moves: the five kernels are bit-equal, and
`pytest test/test_theory_aggregation.py -q` (69 model-layer identities,
the only numerical gate not scoped to the capturing platform) is
unchanged either side of the swap.

### Benchmark

Not required by this task's exit criteria (Task 5.3 owns the headline
benchmark) but recorded because it is what surfaced the debug-build
finding. macOS/arm64, Python 3.13.7, best of three runs, pre-swap Cython
rebuilt standalone from `git show HEAD:…_c_vector_mediator_cross_sections.pyx`
on the same machine. **Release profile** — see the Findings above and
`docs/followups/done/editable-installs-build-the-rust-extension-in-debug.md`
for why an editable install does not give you this.

| entry point | Cython | Rust | speedup |
| --- | --- | --- | --- |
| `sigma_xx_to_v_to_ff`, 1k array | 3.0 us | 2.4 us | 1.25x |
| `sigma_xx_to_v_to_pipi`, 1k array | 16.0 us | 9.3 us | 1.72x |
| `sigma_xx_to_v_to_pi0g`, 1k array | 12.6 us | 11.5 us | 1.10x |
| `sigma_xx_to_v_to_pi0v`, 1k array | 21.1 us | 17.3 us | 1.22x |
| `sigma_xx_to_vv`, 1k array | 7.0 us | 6.6 us | 1.07x |
| `sigma_xx_to_v_to_pi0g`, 1k scalar calls | 94.6 us | 97.2 us | 0.97x |
| `thermal_cross_section`, `x = 20` | 15.2 us | 4.8 us | **3.16x** |
| `thermal_cross_section`, `x = 0.5` | 126.4 us | 65.2 us | 1.94x |
| `thermal_cross_section`, `x = 1/3` | 124.4 us | 66.3 us | 1.88x |

The thermal path is where the win is, and it is the win the phase file
predicted: it no longer re-enters Python per quadrature node. The
scalar-call row is a wash because it is per-call FFI overhead on both
sides, not arithmetic.

## Open Questions

- Three follow-ups filed, all listed under Findings. None blocks
  Task 5.2.
- `test/test_core_quad.py`'s scalar half still uses the live Cython
  `sigma_xx_to_all`; Task 5.2 deletes that module and has to do for the
  scalar branch what this task did for the vector one (rebuild the sum
  from the ported kernels). The shape is already there to copy.

## Plan Impact

**Impact Level:** None.

Re-read against what shipped: the phase file's Goal, Task 5.1's five
exit criteria, Task 5.2's and 5.3's, the phase Exit Criteria, and
ADR-0001/0002's gate sentences. Every one is still factually correct.
The `sigma_xx_to_all` accounting ("7 defs, **6 consumed**") held under
the re-run importer check, and the "no hazma cimports" claim held — the
file had no `.pxd` and exported no capsules, so it went whole.

The one canonical text that *was* wrong is not in this project's files:
`test/parity/tolerances.py`'s `EXACT` class docstring asserted that
every member "reaches only `libc.math`". Corrected in place rather than
deferred, per the canonical-contract rule.

`../../PLAN.md`'s **Numerical impact** section anticipated exactly this
shape — "quadrature moves from scipy's QUADPACK binding to an in-tree
QUADPACK port … closed-form kernels: ≤1e-13" — and the measurement came
in under it (closed forms bit-equal, quadrature at 2.1e-14). No revision
needed. `version_bump: major` is unaffected: it is driven by API
removals, and this task removes only an export nothing imported.

## Stale-state sweep

```console
$ grep -rn "_c_vector_mediator_cross_sections" --include="*.py" --include="*.rs" \
    --include="*.pyi" --include="*.pyx" --include="*.md" . \
    | grep -vE "^\./(projects|docs/followups|\.venv|rust/target)/"
hazma/_core.pyi:12                     hazma/vector_mediator/_vector_mediator_cross_sections.py:10
setup.py:83                            rust/src/kernels.rs:23
rust/src/kernels/vector_xs.rs:2,65,127 rust/src/vector_mediator.rs:10
rust/src/quad.rs:42                    rust/src/special.rs:39  (scalar twin)
test/parity/cases.py:676,687,1044,1468,1475,1479,1483,1487,1491,1495
test/parity/tolerances.py:586          test/test_core_quad.py:80,569,616
test/test_core_vector_xs.py:4,614,619
```

Every surviving mention is **provenance** — a line-numbered citation
into a file this same commit deletes, which is the pattern
`PORTED_ENTRY_POINTS` exists to keep honest and which Tasks 4.2 and 4.6
established. No live import remains; the two that were *claims about the
present* rather than citations were repointed in this change:
`rust/src/quad.rs:42`'s call-site table row now reads "(ported, Task 5.1
— now `crate::kernels::vector_xs`)", matching how Task 4.5 annotated the
rho row, and `hazma/_core.pyi`'s roster comment now says
`vector_mediator` is filled.

```console
$ grep -rn "\bsigma_xx_to_all\b" hazma/ --include="*.py" --include="*.pyx" --include="*.pyi"
hazma/scalar_mediator/_c_scalar_mediator_cross_sections.pyx:670
hazma/scalar_mediator/_c_scalar_mediator_cross_sections.pyx:1051
```

The exit-criterion importer check, re-run at execution time rather than
inherited from the 2026-08-03 inventory: **two hits, both inside the
scalar `.pyx` that defines it** — its own `__vec_` loop and its own
`def`. Zero importers of either export, so the vector one was unconsumed
and dropping it removes nothing a caller could reach. (Untracked build
artifacts — the generated `.c` and the built `.so` — also match and are
gitignored; `test_the_cython_twin_is_gone` asserts on the source files
and the `setup.py` entry for exactly the reason Phase 04's §3 gives.)

```console
$ grep -nE "TODO|FIXME|breakpoint\(\)|\bpdb\b|print\(" \
    rust/src/kernels/vector_xs.rs rust/src/vector_mediator.rs \
    rust/src/dispatch.rs test/test_core_vector_xs.py
rust/src/kernels/vector_xs.rs:431:/// `# TODO: UPDATE THIS!` above this expression; the port transcribes it
```

Nothing introduced. The single hit is a *quotation* of the deleted
`.pyx`'s own comment above `sigma_xx_to_v_to_pi0v`, kept because the
port transcribes that expression unchanged and a reader should know the
original author flagged it.

```console
$ pytest -q | tail -1
2013 passed, 15 skipped, 7 warnings in 131.71s (0:02:11)

$ pytest test/parity -q | tail -1
658 passed, 1 skipped in 83.95s (0:01:23)

$ cargo test --manifest-path rust/Cargo.toml --no-default-features | grep 'test result' | head -1
test result: ok. 186 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out

$ cargo fmt --manifest-path rust/Cargo.toml --check   # no output
$ cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings
    Finished `dev` profile [unoptimized + debuginfo] target(s)

$ python -c "import hazma, hazma._core; print(hazma.__file__, hazma._core.__file__)"
/…/cython-to-rust-next-575dc7/hazma/__init__.py /…/cython-to-rust-next-575dc7/hazma/_core.abi3.so
```

**Numerical-impact statement:** five closed-form kernels bit-equal to
the Cython at all 5,811 compared values on the capturing platform;
`thermal_cross_section` moves by at most **2.06e-14** relative over 285
pinned values (64 of them bit-equal), inside the 1e-12 this change
tightens its budget to and below `rules.md` rule 3's 1e-12 declaration
threshold. No other public value moves —
`pytest test/test_theory_aggregation.py -q` → `69 passed`, unchanged
either side of the swap. One budget tightened, none widened.

**Bookkeeping consistency:** this note's `**Status:** Complete`, the
Task 5.1 row in `../phase-05/README.md`, the Phase 05 row in
`../README.md`, and
`../../phases/phase-05-mediator-cross-sections.md`'s `status: In
Progress` all agree: one of three tasks done, phase open. **Historical —
this paragraph records Task 5.1's closeout state.** Task 5.2 landed on
2026-08-21, so the live count is two of three; the phase is still open,
and [`README.md`](README.md) carries the current tally.
`../../PLAN.md`'s phase table is untouched, which is correct — it moves
when the phase closes. The three new follow-ups each have a row in
`docs/followups/README.md`'s Open table.

## Handoff to Next Task

> **Historical (as of 2026-08-21).** This section was written for
> Task 5.2, which has since landed. Individual claims it got wrong or
> left open are corrected inline below; for the phase's live handoff
> read [`README.md`](README.md) and
> [`task-5.2-scalar-xs.md`](task-5.2-scalar-xs.md) instead.

**Read first:** this note's Findings, then
`rust/src/kernels/vector_xs.rs`'s module docs. Task 5.2 is the same
shape at four times the size.

**Currently safe to assume:**

- The layout is settled: kernels in `crate::kernels::<name>_xs`, PyO3 in
  `crate::<model>_mediator`, module constants beside the kernels rather
  than in `constants::derived`.
- `dispatch::map_unary_try` exists and is the shape for a kernel that
  raises. ~~The scalar module has **no** `** 1.5`, so it probably does
  not need it~~ — **wrong, corrected by Task 5.2 on 2026-08-21**: the
  scalar module has one, in `__sigma_xx_to_s_to_ff`
  (`_c_scalar_mediator_cross_sections.pyx:31`), and needed both shims.
  The second half of the sentence is the part that held: check
  `grep -c SoftComplexToDouble` on the generated C before assuming
  either way, and do it for every `.pyx`, not only ones with a visible
  fractional exponent.
- `crate::quad`'s `qagp` and `crate::special`'s `bessel_k1`/`bessel_kn`
  are exercised by a live thermal path now, not only by their probes.
- `pow(x, 2.0)` folds; `pow(x, 3.0)` and `pow(x, 4.0)` do not.

**Currently risky / unknown:**

- The scalar file's two 90-line expressions (`__sigma_xpi_to_xpi`,
  `__sigma_xpi0_to_xpi0`) are the phase's real transliteration risk, and
  `docs/followups/todo/scalar-elastic-cross-sections-cancel-in-atan-difference.md`
  says four of its kernels cancel away every significant bit — read
  `test/parity/stability.py` before trusting a comparison there.
- ~~`test/test_core_quad.py`'s scalar `sigma_xx_to_all` oracle dies with
  that module; the vector branch shows the replacement.~~ — **settled by
  Task 5.2 on 2026-08-21**: it did, and 5.2 rebuilt it from the ported
  kernels exactly as the vector branch does.
- Whether the scalar `thermal_cross_section` tightens to
  `PORTED_QUAD_RTOL` too is a measurement, not an inheritance —
  Phase 04's §1 is emphatic that per-kernel drift is not predictable
  from shape. **Measured by Task 5.2 on 2026-08-21**: it does — the
  scalar side moves by 3.12e-15 relative and its budget is tightened to
  `PORTED_QUAD_RTOL`, same as the vector's. Note that the scalar model
  **short-circuits to `0.0`** above `x = 300` where the vector
  saturates; the corpus pins both.
