# Phase 05 Learnings: Mediator cross sections

Synthesized at phase close (2026-08-21) from the three task notes
([5.1](../task-notes/phase-05/task-5.1-vector-xs.md),
[5.2](../task-notes/phase-05/task-5.2-scalar-xs.md),
[5.3](../task-notes/phase-05/task-5.3-thermal-sweep.md)) and
[`../task-notes/phase-05/README.md`](../task-notes/phase-05/README.md).
Read this instead of the notes; the notes are history.

## 1. Implementation Reality Check

The phase delivered what it promised. **18 consumed defs** — 16 cross
sections plus both thermal ⟨σv⟩ — moved from Cython to `hazma._core`
across two porting tasks, each gated on the parity corpus; the two
unconsumed `sigma_xx_to_all` exports were dropped rather than ported;
and both `_c_*_cross_sections.pyx` were deleted in the same PR as their
swap, so no dual-implementation window ever opened. `grep -rn
sigma_xx_to_all hazma/` returns nothing and `hazma/*/_c_*.pyx` matches
nothing.

Numerically the phase came out better than its budgets. **Sixteen of
eighteen entry points are bit-equal** to the Cython they replace, at
`rtol = 0`, over 17,966 compared corpus values on the capturing
platform. The two that move are the thermal ⟨σv⟩ — 2.06e-14 (vector) and
3.12e-15 (scalar) — and both moved *down* in budget, from `QUAD_RTOL`
(1e-8) to `PORTED_QUAD_RTOL` (1e-12). No budget in this phase was
widened, and `QUAD_RTOL` ended the phase with **no holder at all**.

Task 5.3 then took the phase past the corpus grid to the live consumer.
`relic_density(semi_analytic=True)` — the only relic path with no
adaptive solver in it — reproduces the pre-port tree to **4.11e-16**,
four of six model points bit-equal. The headline benchmark landed too:
relic density through a real mediator is **1.46×–1.93×** faster and the
⟨σv⟩ kernel itself **1.8×–3.2×**, because the integrand no longer
re-enters Python per quadrature node.

## 2. Critical Context for Future Work

- **`**` in a `.pyx` is not real arithmetic, and reading the source will
  not tell you.** Cython 3's default `cpow` semantics compile a
  `double ** double` — and the whole expression around it — in
  `double _Complex`, reaching `cpow` and compiler-rt's `__divdc3`
  instead of `pow` and `/`. Neither agrees with the real spelling (up to
  9.0e-15). The detector is `grep -c SoftComplexToDouble` on the
  **generated C**, and it must be run: Task 5.1 asserted from the source
  that the scalar module had none, and Task 5.2's grep found one live
  site (`__sigma_xx_to_s_to_ff`). The two shims live in
  `crate::kernels::soft_complex` and are shared. **Phase 06's four
  `.pyx` have still not been grepped.**
- **Where clang fuses is one syntactic rule, not a case list.**
  `EmitFMulAdd` contracts `A + B` when `A` is a multiply in the *C* tree
  Cython emits, else when `B` is; `x ** n` is a `pow` **call** there and
  never a multiply, and a leading unary minus is an `FNeg` that blocks
  it. Hence `-4*mx**2 + e_cm**2` fuses, `ms**2 - e_cm**2` does not.
  Task 5.2 reproduced all 138 FMA sites in eleven kernels from that rule
  alone, without a disassembly per site.
- **One Python-level call boxes everything above it.** A single
  `np.log(4)` in the middle of a kernel makes Cython evaluate the whole
  enclosing expression as Python objects, which changes the rounding of
  everything on its path to the root. Grep for `np.` inside a `.pyx`
  before transliterating.
- **`pow(x, 2.0)` folds to `x·x`; `pow(x, 3.0)` and `pow(x, 4.0)` do
  not.** Writing `x·x·x` is a different double.
- **The cargo profile is a speed choice, never a correctness one.**
  `pip install -e .` builds `hazma._core` in **debug**
  (`setuptools_rust` infers `debug = self.inplace or self.debug`), which
  makes it ~20× slower and flips the sign of every benchmark — in debug
  the port looks like a 20× regression, in release it is a 1.1×–3.2×
  win. But Task 5.3 verified the two profiles are **bit-identical**
  across 90 values, so a *parity* result from a debug tree is valid and
  only its timing is not. Phase 07 Task 7.1 makes the profile decision;
  [the follow-up](../../../docs/followups/todo/editable-installs-build-the-rust-extension-in-debug.md)
  carries it.
- **Tell solver noise from drift by tightening the solver, not the
  kernel.** `relic_density(semi_analytic=False)` moved by up to 3.82e-5
  across the port, which looks alarming against a 2e-14 kernel drift and
  is not amplification: `solve_ivp` at `rtol=1e-5` lets a last-bit input
  change flip a step-acceptance decision. Tightening `rtol` to 1e-8 and
  1e-10 collapses the difference to 2.75e-7 and 3.84e-9. Run that sweep
  before treating any adaptive-solver delta as a regression.

## 3. Quirk Log & Edge Cases

- **Three corpus records are `raises`, not values** — `TypeError` at
  `e_cm = 2 m_x` in the two complex vector channels — and
  `test_parity.py` replays them rather than skipping. That forced
  `dispatch::map_unary_try`, the fourth live entry-point shape: a kernel
  that can fail per element and takes the whole array down with it, as
  the Cython's `__vec_*` loop did.
- **The one user-visible string change in the phase** is that
  `TypeError`'s wording: it keeps its type and loses Cython's advice to
  `use 'cython.cpow(True)'`, a directive that will not exist after
  Phase 07. The corpus records only the type.
- **The two mediators disagree above `x = 300`**: the scalar
  `thermal_cross_section` returns exactly `0.0` where the vector
  saturates. Both are pinned; do not unify them.
- **Four scalar elastic-scattering kernels cancel in an `atan`
  difference** near `e_cm = 2 m_x` at small width, producing the wrong
  sign and a fabricated pole. Ported **with** the defect: rule 1 gates a
  swap on a corpus rule 2 forbids regenerating, so a stabilised kernel
  could not pass the gate that would let it ship.
- **Six annihilation channels disagree at threshold** — two raise where
  four return `inf`/`nan`. Also reproduced, also filed.
- **The thermal integrals never converge, and this is the phase's
  biggest number by four orders of magnitude.** Both `.pyx` passed
  `quad` neither `epsabs` nor `epsrel`, so scipy's default *absolute*
  tolerance (1.49e-8) is satisfied by the first Kronrod pass on an
  integrand whose integral is ~1e-27; the shipped answer is 0.5%–5% off
  the true integral for every `x ≳ 5`, i.e. across all of freeze-out.
  Task 5.3 measured the consequence: relic abundance goes as 1/⟨σv⟩, so
  every relic density Hazma has published inherits that error roughly
  linearly. Reproduced under rule 1, not fixed —
  [the follow-up](../../../docs/followups/todo/thermal-cross-section-quadrature-never-converges.md)
  now carries the downstream size and is unblocked.

## 4. Test Infrastructure State

- `test/test_core_vector_xs.py` and `test/test_core_scalar_xs.py` are
  the per-kernel unit gates; `test/parity` is the swap gate.
  `cargo test --no-default-features` ended the phase at **201 passing**
  Rust unit tests.
- **`Case.attribute`** was added in Task 5.2 for a wrapper that cannot
  re-export a kernel under its own name — the scalar wrapper defines a
  function per kernel, the vector one is a mixin class with no functions
  of its own. Phase 06's mediator spectrum wrappers may need the same.
  `hazma._core.<sub>` can never be named directly in a case: a PyO3
  submodule has no `__file__` and `assert_module_is_repo_tree` rejects
  it.
- **`test/test_relic_density.py` now has real coverage.** Before Task
  5.3 its only model was a `ToyModel` supplying a constant
  `thermal_cross_section`, which `_thermal_functions` short-circuits to
  — so every relic assertion in the suite bypassed the compiled layer.
  `TestMediatorRelicDensity` pins twelve pre-port values (six model
  points × two solvers, captured at `14f1c66`) with two separately
  justified tolerances: `1e-12` semi-analytic, a real gate; `1e-4`
  Boltzmann, a smoke gate bounding step-selection spread. Both were
  shown to fail against a perturbed kernel.
- **The corpus cannot be extended from here on.** Rule 2 forbids
  regenerating it from a tree where Rust serves the kernels, and
  `generate.py` enforces it — true of both mediators since Task 5.2.
  Consumer-level regressions belong in `test/`, not `test/parity/data/`.
- **Establishing a pre-port baseline now costs a build.** Both `_c_*`
  `.pyx` are deleted, so "pre-port" is a git commit: `14f1c66` is the
  merge before the first Phase 05 swap. Task 5.3's recipe — detached
  worktree, its own venv with the same pins, one sweep script run
  against both interpreters — is the cheapest reliable way and Phase 06
  will need it again.
- **`UNCONSUMED_EXPORTS` is empty** and its coverage check is a no-op
  today.

## 5. Follow-on seeds

All four are filed; none blocks Phase 06.

- [The unconverged thermal quadrature](../../../docs/followups/todo/thermal-cross-section-quadrature-never-converges.md)
  — now unblocked and quantified at the consumer. The most consequential
  open defect in the project.
- [The `2 m_x` raise](../../../docs/followups/todo/vector-cross-sections-raise-at-the-two-mx-threshold.md)
  — six channels, two behaviors, at the annihilation threshold.
- [The `atan` cancellation](../../../docs/followups/todo/scalar-elastic-cross-sections-cancel-in-atan-difference.md)
  — its "fix it during Phase 05" window has closed; the note records
  what the standalone change now costs.
- [The debug editable build](../../../docs/followups/todo/editable-installs-build-the-rust-extension-in-debug.md)
  — Phase 07 Task 7.1 revisits it, and Task 5.3 removed its one
  numerical risk.
