# Task 5.3: Thermal ⟨σv⟩ validation sweep

**Date:** 2026-08-21
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-05-mediator-cross-sections.md`
(Task 5.3, phase Exit Criteria); `../../rules.md` rule 12 (Process 3)
**Related ADRs:** none
**Depends On:** Task 5.1, Task 5.2

## Objective

Close Phase 05 by checking the ported thermal ⟨σv⟩ through its live
consumer — `hazma/relic_density/` — rather than only through the corpus
grid, and record the phase's headline benchmark from a release build.

## Exit Criteria

- Relic densities computed through
  `hazma/relic_density/_thermal_functions.py`'s `thermal_cross_section`
  path match pre-port values within a declared budget, for one scenario
  per mediator (delivered: three per mediator).
- The check is a committed test, not a one-off script — Phase 01 pinned
  the kernel, not the consumer.
- Benchmark recorded per rule 12, from a **release** build, against the
  pre-swap Cython on the same machine.
- Phase 05 closed: phase file `status: Complete`, phase README rows,
  learnings file, `PLAN.md` phase row, numerical-impact entry.

## Inputs Reviewed

- `../../phases/phase-05-mediator-cross-sections.md` — Task 5.3 block, phase
  Exit Criteria.
- `../README.md`, `README.md` (phase 05) — status tables, Task 5.2 handoff.
- `../numerical-impact.md` — Task 5.1/5.2 drift lines for both
  `thermal_cross_section`.
- `../../rules.md` — rule 12 (benchmark provenance), rule 2 (no corpus
  regeneration).
- `task-5.1-vector-xs.md` §Benchmark — the release-build gotcha and the
  per-kernel table this task extends end-to-end.
- `hazma/relic_density/{_rd,_thermal_functions,_diffeq,_approx}.py`;
  `test/test_relic_density.py`.
- `test/parity/cases.py` — `_scalar_model_points` / `_vector_model_points`,
  reused verbatim as the sweep's scenarios.
- `docs/followups/todo/editable-installs-build-the-rust-extension-in-debug.md`.

## Findings

- **The suite had no end-to-end coverage of the thermal path at all.**
  `test/test_relic_density.py`'s `ToyModel` defines its own
  `thermal_cross_section` returning a constant, and
  `_thermal_functions.thermal_cross_section` short-circuits to
  `model.thermal_cross_section(x)` when the model has one — so every
  pre-existing relic-density assertion bypassed the compiled layer
  entirely. Phase 01 pinned the kernel on its own grid; nothing pinned
  the consumer. That gap is what this task's test closes.
- **The two relic solvers propagate the port's drift by nine orders of
  magnitude apart, and only one of them is measuring physics.** The
  semi-analytic path is a closed-form composition and carries the
  kernel's ≤2.06e-14 drift through essentially undamped — **≤4.2e-16**
  over six scenarios, four of them bit-equal. The Boltzmann path moves
  by up to **3.82e-5**, which is *not* amplification: `solve_ivp`'s
  adaptive stepping makes a last-bit input change flip a step-acceptance
  decision, and the answer then differs at the solver's own tolerance.
  The decisive evidence is the tolerance sweep below — tightening
  `rtol` 1e-5 → 1e-8 → 1e-10 collapses the pre↔post difference by four
  orders while the pre-port answer itself moves by only 1.3e-5. A true
  drift would not shrink when the *solver* is tightened.
- **Debug and release `hazma._core` are numerically bit-identical.**
  The whole six-scenario sweep (12 relic densities + 78
  `thermal_cross_section` values) reproduces at `rtol = 0` across the two
  cargo profiles, so `[profile.release]`'s LTO and `codegen-units = 1`
  buy speed without moving a value. That removes the one real risk in
  the open follow-up's proposed `debug=False` one-liner, and is recorded
  there.
- **The predicted win is real and it is the end-to-end number, not just
  the kernel.** Relic density through a real mediator is **1.46×–1.93×**
  faster; the kernel itself is 1.8×–3.2×. The gap is the pure-Python
  Boltzmann ODE, which the port does not touch — the thermal integral
  stopped being the dominant cost.

## Decisions and Implementation Notes

- **Pinned pre-port values, not a cross-tree comparison, as the
  committed gate.** The `_c_*_cross_sections.pyx` were deleted in Tasks
  5.1/5.2, so "pre-port" is a git commit (`14f1c66`, the merge before
  the first Phase 05 swap), built into a throwaway worktree and swept
  with the same script. Twelve numbers came back and are now constants
  in the test; the worktree is not a build dependency of the suite.
- **Two tolerances, two different justifications, both in the test.**
  `SEMI_ANALYTIC_RTOL = 1e-12` is ~2000× the measured 4.2e-16 — sharp
  enough that a real kernel regression cannot hide behind it (verified:
  a 1e-9 relative perturbation fails all six subtests).
  `BOLTZMANN_RTOL = 1e-4` bounds the measured 3.82e-5 step-selection
  spread with ~3× headroom; it is a coarse end-to-end smoke gate by
  construction, and the note says so rather than implying a physics
  budget.
- **Reused `test/parity/cases.py`'s six model points verbatim** rather
  than inventing scenarios, so a failure here and a corpus failure
  implicate the same kernels at the same couplings. They are not
  physical abundances and the test says so.
- **Did not touch `setup.py`.** The release build was obtained by
  temporarily setting `debug=False` on the `RustExtension`, measuring,
  and reverting. Choosing the profile is the open follow-up's call and
  Phase 07 Task 7.1 revisits it; making it here would be scope drift.
- **Did not extend the parity corpus.** Rule 2 forbids regenerating it
  from a tree where Rust serves the kernels, which is now true of both
  mediators — `generate.py` would refuse. The relic scenarios live in
  `test/test_relic_density.py` instead, which is where a consumer-level
  regression belongs anyway.

## Files Changed

- `test/test_relic_density.py` — new `TestMediatorRelicDensity`: six
  mediator scenarios × two solvers, pinned to pre-port values with
  stated tolerances.
- `docs/followups/todo/editable-installs-build-the-rust-extension-in-debug.md`
  — records that the two cargo profiles are numerically bit-identical.
- `docs/followups/todo/thermal-cross-section-quadrature-never-converges.md`
  — unblocked, and now carries the measured downstream size.
- Project bookkeeping, all under `projects/cython-to-rust/`: this note;
  `task-notes/phase-05/README.md` (row, status, findings, handoff);
  `task-notes/README.md` (Phases row, findings sweep, handoff);
  `task-notes/history-findings.md` (Phase 05's two bullets, verbatim);
  `task-notes/numerical-impact.md` (Task 5.3 entry);
  `learnings/phase-05-mediator-cross-sections.md` (new);
  `phases/phase-05-mediator-cross-sections.md` (`status: Complete`);
  `PLAN.md` (Phase 05 row).

## Measurements

Machine: macOS/arm64 (Darwin 25.6.0), CPython 3.12.12, NumPy 2.5.1,
SciPy 1.18.0. Pre-port tree: `14f1c66` built in a detached worktree with
the same interpreter and pins. Post-port tree: this branch.

### Relic density, pre-port vs ported (relative difference)

Six model points, taken verbatim from `test/parity/cases.py`.

| scenario | semi-analytic | Boltzmann (`rtol=1e-5`) |
| --- | --- | --- |
| `scalar.open_resonance` | 0 (bit-equal) | 7.94e-10 |
| `scalar.narrow_resonance` | 4.03e-16 | 1.94e-07 |
| `scalar.closed_resonance` | 0 (bit-equal) | 2.81e-06 |
| `vector.open_resonance` | 1.73e-16 | 2.93e-05 |
| `vector.narrow_resonance` | 0 (bit-equal) | 3.82e-05 |
| `vector.closed_resonance` | 4.11e-16 | 2.32e-06 |

### The Boltzmann column is step selection, not drift

Same comparison, tightening only the ODE solver:

| scenario | `rtol=1e-5` | `rtol=1e-8` | `rtol=1e-10` |
| --- | --- | --- | --- |
| `scalar.closed_resonance` | 2.81e-06 | 9.40e-10 | 2.70e-13 |
| `vector.open_resonance` | 2.93e-05 | 2.42e-07 | 1.00e-09 |
| `vector.narrow_resonance` | 3.82e-05 | 2.75e-07 | 3.84e-09 |

For reference, the *pre-port* answer itself moves by 1.3e-5
(`vector.narrow_resonance`: 0.3270447 at `rtol=1e-5` vs 0.3270524 at
`rtol=1e-10`) — the same order as the pre↔post difference at the default
tolerance, which is the point.

### `thermal_cross_section` on the sweep grid

78 values (6 scenarios × 13 points in `x = mx/T`, spanning 0.1–500 and
both sides of the scalar kernel's `x = 300` cutoff): worst relative
difference **1.2799e-15**, 37 of 78 bit-equal. Consistent with the
corpus-grid figures Tasks 5.1/5.2 recorded (2.06e-14 / 3.12e-15) — this
grid simply does not land on their worst point.

### Benchmark (rule 12) — release build, best of three

| row | Cython | Rust | speedup |
| --- | --- | --- | --- |
| scalar `thermal_cross_section`, `x = 0.5` | 68.0 us | 37.4 us | 1.82x |
| scalar `thermal_cross_section`, `x = 20` | 16.7 us | 6.0 us | 2.77x |
| scalar `relic_density(semi_analytic=True)` | 1.4 ms | 929.0 us | 1.51x |
| scalar `relic_density(semi_analytic=False)` | 55.6 ms | 38.2 ms | 1.46x |
| vector `thermal_cross_section`, `x = 0.5` | 77.7 us | 40.3 us | 1.93x |
| vector `thermal_cross_section`, `x = 20` | 15.5 us | 4.9 us | 3.18x |
| vector `relic_density(semi_analytic=True)` | 2.1 ms | 1.1 ms | 1.93x |
| vector `relic_density(semi_analytic=False)` | 61.9 ms | 40.7 ms | 1.52x |

Both mediators use `KineticMixing`/`HiggsPortal` at the
`open_resonance` point. The **release** caveat from Task 5.1 applies:
the same table taken from a plain `pip install -e .` tree reports the
port as ~20× slower.

## Verification

- `.venv/bin/python -m pytest test/test_relic_density.py -q` —
  `3 passed, 1 warning, 12 subtests passed in 2.31s`. Covers: the
  pre-existing four-point `ToyModel` sanity check; six mediator
  scenarios (scalar and vector × open/narrow/closed resonance) on the
  semi-analytic solver; the same six on the Boltzmann solver.
- **Test validity (perturbation-proof).** The port cannot be `git
  stash`ed — it is two merged commits — so validity was shown by
  perturbing the kernel instead: multiplying both models'
  `thermal_cross_section` by `1 + 1e-9` fails all 6 semi-analytic
  subtests; by `1 + 1e-3` fails all 6 of both tests. Neither test passes
  against a wrong kernel.
- Debug/release identity: the full sweep re-run against the release
  `_core.abi3.so` reproduces the debug run at `rtol = 0` across all 90
  values.
- Full suite + preflight: see `## Stale-state sweep`.
- **Deferred:** no relic-density scenario was added to
  `test/parity/data/` — rule 2 forbids regenerating the corpus from a
  Rust-serving tree, and `generate.py` enforces it.

## Numerical impact

`relic_density` is a public function and the diff reaches it, so it was
measured directly (tables above). Semi-analytic: **no public value
changes beyond 4.2e-16**, four of six bit-equal — below rule 3's 1e-12
threshold, no CHANGELOG line of its own. Boltzmann: up to **3.82e-5**,
which is `solve_ivp` step selection at the caller's own `rtol=1e-5`, not
a drift in the physics; the underlying answer is unchanged to ~1e-9 once
the solver is tightened. Logged in `../numerical-impact.md`. No change
to `version_bump:` — it was already `major` on API-removal grounds.

## Open Questions

- The unconverged thermal quadrature
  (`docs/followups/todo/thermal-cross-section-quadrature-never-converges.md`)
  is now measured at the consumer: it is 0.5%–5% wrong on ⟨σv⟩ across
  freeze-out, and relic abundance goes as 1/⟨σv⟩, so the shipped relic
  densities inherit that error more or less linearly. This task
  reproduces it under rule 1 rather than fixing it (`PLAN.md` Scope
  excludes physics changes); the follow-up now carries the consumer-level
  consequence. Phase 07's CHANGELOG names it.

## Plan Impact

**Impact Level:** None.

Re-read against what shipped: Task 5.3's two exit-criteria bullets, the
phase Exit Criteria's four, and rules 1–3 and 12. All still factually
correct. The parenthetical "extend the corpus with one relic-density
scenario per mediator if Phase 01 didn't" is satisfied in spirit and
could not be satisfied literally — rule 2 outranks it and the phase file
already says so elsewhere — so the check landed in
`test/test_relic_density.py`. That is a placement choice inside the
criterion, not a change to it.

## Stale-state sweep

Run against `claude/cython-to-rust/task-5.3-thermal-sweep`.

| Check | Command | Result |
| --- | --- | --- |
| Phase-status agreement | `grep -n '^status:' phases/phase-05-*.md`; `grep -n '^\*\*Status:\*\*' task-notes/phase-05/{README,task-5.3-*}.md`; `grep -n '^. 05 ' PLAN.md task-notes/README.md` | All six read Complete / **Complete (2026-08-21)**; no `In Progress` left for Phase 05 |
| Stale "Phase 05 is open" text | `grep -rn 'Phase 05 is open\|Phases 00–04 are closed' projects docs` (excluding `history-*` and the closed 5.1/5.2 notes) | none stale |
| Stale "Task 5.3 will/should" text | `grep -rn 'Task 5.3 should\|Task 5.3 owns\|Task 5.3 remains' projects docs` | one hit, in `task-5.1-vector-xs.md` §Benchmark, historically accurate and left alone |
| Phase Exit Criteria hold | `ls hazma/*/_c_*.pyx`; `grep -rn sigma_xx_to_all hazma/` | both empty — twins deleted, exports dropped |
| Archive is verbatim | `git show cbe5555:…/task-notes/README.md \| sed -n '72,91p'` diffed against the moved block in `history-findings.md` | identical (`diff` → no output) |
| Followups index unaffected | `grep -n 'thermal-cross-section-quadrature\|editable-installs' docs/followups/README.md` | two rows, neither restates the edited blocker/risk text — no index change needed |
| Forbidden tokens | `git diff origin/master -- test/ \| grep -E '^\+.*(TODO\|FIXME\|breakpoint\|pdb\|print\()'` | none |
| Lint on the touched module | `ruff check test/test_relic_density.py` | `All checks passed!` — was 10 findings at `origin/master`, so the file left cleaner than it arrived |
| Numerical-impact statement | measured, tables in §Measurements; logged in `../numerical-impact.md` | semi-analytic `relic_density` ≤4.11e-16 (4 of 6 bit-equal); Boltzmann ≤3.82e-5, identified as `solve_ivp` step selection by the tolerance sweep, not drift; no `version_bump:` change |
| Preflight | `env PATH=".venv/bin:$PATH" scripts/agents/preflight.sh --paths … --md …` | `RESULT: PASS` (see §Verification) |

## Handoff to Next Task

**Read first:** `../../learnings/phase-05-mediator-cross-sections.md` —
it replaces this note and the two before it for every later reader.

**Currently safe to assume:**

- Phase 05 is closed. All 18 consumed defs are on Rust, both `_c_*`
  `.pyx` are gone, both `sigma_xx_to_all` exports are dropped
  (`grep -rn sigma_xx_to_all hazma/` → no hits), and the relic-density
  consumer is pinned.
- The port's cost model is settled: kernels are 1.1×–3.2× faster in
  release and the remaining relic-density time is pure-Python ODE, not
  the compiled layer.
- Debug and release `hazma._core` produce identical doubles, so a
  benchmark taken in debug is wrong about *speed only*. Phase 07 Task
  7.1 can set the profile without a parity re-run.

**Currently risky / unknown:**

- Phase 06's four mediator-spectrum `.pyx` have still not been run
  through `grep -c SoftComplexToDouble` on their generated C. Task 5.2's
  warning stands: run it, do not eyeball the source.
- Nothing in the suite pins `relic_density` at a *physical* abundance
  through a mediator model; the six pins are stress points chosen for
  the cross sections. If Phase 07 wants a physics-facing assertion, that
  is a new scenario, not a retune of these tolerances.
