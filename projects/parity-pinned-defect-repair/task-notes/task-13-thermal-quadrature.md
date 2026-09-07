# Task 13 — Repair B5: the thermal averages never converged

**Status:** Complete
**Task:** Task 13 (added after the original twelve)
**Plan reference:** `../PLAN.md`, "Task 13"
**Defect:** `B5` in `../references/defect-blast-radius.md`

## Exit Criteria

- [x] `epsabs = 0` at all four sites that share the defect, not only the
      two in Rust.
- [x] Both kernels reach the criterion they now bind to — a subdivision
      limit above `quad`'s default of 50.
- [x] `pytest test/parity` green with the six thermal arrays declared,
      and red when the repair is reverted.
- [x] The twelve pinned values in `test/test_relic_density.py`
      re-derived from the corrected kernel, not absorbed by a widened
      tolerance.
- [x] An independent oracle agrees (`../rules.md` rule 3) and a physics
      invariant holds (rule 4).
- [x] `git diff --stat -- test/parity/data` empty (rule 1).

## Inputs Reviewed

- `../PLAN.md`, `../rules.md`, `../references/corpus-repinning.md`,
  `../references/defect-blast-radius.md`.
- `docs/followups/done/thermal-cross-section-quadrature-never-converges.md`
  (the follow-up this task closes).
- `test/parity/deltas.py` and `test/parity/test_parity.py` (the B4
  declaration and the runner that consumes it).
- `rust/src/kernels/{vector,scalar}_xs.rs`, `rust/src/quad.rs`.
- `test/test_core_quad.py` (`TestLiveIntegrandShapes`), which already
  rebuilt the thermal integrand in Python.

## Findings

**The defect is in the integrator, not in a closed form.** Every other
roster entry is a wrong expression; this one is a right expression the
integrator never resolved. Both kernels inherited
`scipy.integrate.quad`'s default `epsabs = 1.49e-8` against an integral
of order 1e-27, and QUADPACK stops as soon as *either* criterion is met,
so the absolute one was satisfied by the first Gauss–Kronrod pass and
the initial three-interval partition came back unrefined.

**It is worse than the follow-up estimated.** That file measured 0.5% to
5% on one model point. Against
`test/parity/thermal_reference.py` over all 570 corpus positions the
shipped values are wrong by **up to 1.00 relative** — at both
`closed_resonance` blocks the shipped number retains none of the true
one. The integrand decays like `e^{-x z}` over an interval running to
150, so at large `x` essentially all its mass sits within a few
hundredths of the lower limit and a 15-point rule spread across the
whole range misses it.

**Four call sites, two of them never ported.**
`hazma/relic_density/_thermal_functions.py` (the generic fallback) and
`hazma/vector_mediator/_gev/thermal_cross_section.py` carry the same
defect in pure Python. A repair confined to the Rust kernels would have
left them, which is why the follow-up called it a four-site class.

**Zeroing `epsabs` is only half the fix.** At `quad`'s default
`limit = 50`, 33 of the 540 thermal positions the corpus integrates
exhausted the subdivision table and returned QUADPACK's `ier = 1`.
Raising the limit to 100 clears 17 of those and improves the worst error
against the reference from 2.5e-8 to 3.6e-9, after which accuracy
plateaus: the remaining 16 are at the roundoff floor of the
extrapolation table, and 200 or 500 only let them subdivide deeper for
no gain.

**The declaration needed a relation the layer did not have.**
`Additive` presupposes a term the physics names and `Exact` a transform
of the stored array; neither exists when the stored value is the true
one *unresolved*. `Reference` supersedes the stored array against an
independently computed value. Relations now answer one `term_for`
question, so the runner no longer knows which kind it holds — the shape
the spec's unimplemented `Exact` and `Bounded` will plug into.

## Decisions and Implementation Notes

- **`THERMAL_EPSABS = 0.0`, `THERMAL_EPSREL` unchanged** at both
  kernels. QUADPACK's "tolerance unachievable" check then rests on
  `epsrel >= max(50 ε, 5e-29)` = 1.11e-14, which 1.49e-8 clears by six
  decades (`rust/src/quad.rs:735`, `:1113`).
- **`THERMAL_LIMIT = 100`**, a new const in both kernels, chosen from
  the measurement above rather than for headroom.
- **The reference is scipy's QUADPACK, not mpmath.** Rule 3 wants an
  independent oracle, and the component under repair is the *integrator*
  — so the integrand is the controlled variable and is deliberately
  shared (it is bit-exact against the pre-port Cython at all 5,811
  corpus positions that sample it), while the quadrature is
  independent. An mpmath transcription of `sigma_xx_to_all` would have
  re-derived the half that is not in question and left the half that is
  compared against itself.
- **`test/parity/thermal_reference.py` is a new module, not an addition
  to `reference.py`.** That file is regeneration-only — `pyproject.toml`
  records that `pytest` imports neither it nor mpmath — and this oracle
  has to run inside the gate. It costs 0.76 s for all 570 positions.
- **The declaration's `rtol` comes from the kernel's own tolerance.**
  Locally the repaired kernels sit 3.6e-9 from the reference, but they
  are only *held* to `epsrel = 1.49e-8`, and a platform whose libm
  steers QUADPACK to a different accepted partition may land anywhere
  inside that. `rtol = 1e-7` is 6.7x the bound that must hold
  everywhere, not 28x the figure this machine gives.
- **`SEMI_ANALYTIC_RTOL` in `test/test_relic_density.py` goes 1e-12 →
  1e-6** for the same reason: the old budget was set from the port's
  2.06e-14 drift on a kernel that never subdivided. This is not a
  tolerance widened to absorb a failure (`../rules.md` rule 2) — the
  pins are re-derived, and the budget now describes a mechanism that
  changed. It stays ~10,000x tighter than the smallest shift B5 itself
  produced (0.071%).
- **Review round 1 (PR #91) added coverage for the two pure-Python
  sites.** The repair fixed four call sites but only the two Rust ones
  were reachable from the parity corpus and the mediator relic-density
  pins, so reverting `epsabs = 0` at either Python site left the suite
  green. `test/test_relic_density.py::TestThermalQuadratureConverges`
  now pins both against the same integrand at `epsrel = 1e-12`, and each
  fails when *only its own* site is reverted (17 subtests for the
  fallback, 6 for the GeV site). The first oracle drafted for them used a
  synthetic analytic cross section and was **not** revert-sensitive —
  both Python sites integrate to `50/x`, which tracks the integrand's
  decay length, so they lack the pathology the Rust kernels' fixed
  `max(50/x, 150)` interval creates; only a realistic resonant integrand
  exposes it (0.765 worst relative error at the default `epsabs`).
  Ledgered as `[fix-covered-only-where-tests-already-ran]`.
- **Review round 1 also found the scalar kernel's function-level doc
  still describing the pre-repair behavior** — the `THERMAL_EPSABS`
  const doc was updated but `thermal_cross_section`'s own was not. Swept
  the class across both kernels; the vector twin's doc makes no
  tolerance claim, so scalar was the only occurrence.
- **The `IntegrationWarning` the reference raises at 16 positions is
  suppressed with its justification**: requesting `epsrel` of 1e-9,
  1e-10, 1e-11 and 1e-12 in turn moves the answer by at most 7.8e-10,
  two decades under the budget. The warning is about what QUADPACK can
  certify, not about the value.

## Numerical impact

**Public values move; that is the deliverable.** Measured on this
worktree against `test/parity/thermal_reference.py`:

| Quantity | Grid | Shipped error vs reference | Repaired |
| --- | --- | --- | --- |
| ⟨σv⟩ scalar + vector | 570 corpus positions | up to **1.00**; medians 7.2e-6 – 8.1e-2 | within 3.6e-9 |
| `relic_density` semi-analytic | 6 pinned model points | −91.85% … +2.04% | re-pinned |
| `relic_density` Boltzmann | 6 pinned model points | −99.88% … +2.40% | re-pinned |

539 of 570 stored positions move. Of the 31 that do not, 30 are the ten
points per scalar block above `x = 300` (that kernel returns `0.0`
before integrating) and the last is vector `narrow_resonance` at
`x = 0.1367`, where the relative criterion already bound.

Per-model relic-density shifts (semi-analytic, Boltzmann):

| Model point | Semi-analytic | Boltzmann |
| --- | --- | --- |
| `scalar.open_resonance` | −0.0715% | −0.0740% |
| `scalar.narrow_resonance` | +2.0388% | +2.3955% |
| `scalar.closed_resonance` | **−91.8497%** | **−92.7054%** |
| `vector.open_resonance` | +0.5925% | +0.5815% |
| `vector.narrow_resonance` | +0.2012% | +0.2043% |
| `vector.closed_resonance` | **−99.8630%** | **−99.8813%** |

**Cost** (the follow-up asked for it, since `relic_density` calls this
inside an ODE right-hand side): 12 `relic_density` solves go
**1.31 s → 5.4 s**; one `thermal_cross_section` call goes
**4.6–48 µs → 21–136 µs**, the ratio largest at `x ≈ 20` where the
defect was worst.

**Independent oracle** (rule 3): scipy's QUADPACK at `epsrel = 1e-12`
over the same integrand, in `test/parity/thermal_reference.py`.

**Physics invariant** (rule 4): the Rust
`the_thermal_integral_matches_a_composite_rule` sums the same integrand
with Simpson's rule on a uniform grid under the substitution
`z = √(4 + w²)`, with panel boundaries on every channel threshold —
an algorithm that shares nothing with QUADPACK. The entry point now
lands **4.199e-8** from it, which is Simpson's own residual; before the
repair it sat 0.79% away.

## Verification

```sh
cargo fmt --manifest-path rust/Cargo.toml --check                       # clean
cargo clippy --manifest-path rust/Cargo.toml --all-features \
    --all-targets -- -D warnings                                        # clean
cargo test --manifest-path rust/Cargo.toml --no-default-features \
    --features test-probes                                              # 261 passed
pytest test/parity -q                                                   # 668 passed, 1 skipped
pytest -q                                                               # 2246 passed, 15 skipped, 12 subtests passed
```

What the new and changed tests cover:

- **The corpus gate** — six declared arrays across both thermal cases.
  Only the 539 positions the reference actually moves are compared
  against it; the remaining 179,156 of the corpus's 179,695 pinned
  values still compare against the stored array under their own budget,
  and that includes the 31 unmoved positions *inside* the declared
  arrays, which `MOVED` leaves at the case budget rather than sweeping
  in (`../rules.md` rule 5).
- **The live call configuration** —
  `test_core_quad.py::test_thermal_cross_section_site` now exercises
  `epsabs = 0, limit = 100` alongside the break-point-filtering probe,
  asserting scipy agreement, `ier == 0`, and that zeroing `epsabs`
  drives the partition past the filtered one.
- **The independent quadrature** — the Rust composite-rule test,
  re-pinned from 2e-2 to 1e-6.
- **End-to-end** — the twelve `relic_density` pins, re-derived.

**Mutation proof** (rule 6 / spec proof obligation 2). Reverting
`THERMAL_EPSABS` to `DEFAULT_EPSREL` in both kernels and rebuilding:

```text
6 failed, 662 passed, 1 skipped
FAILED ... cross_sections.{scalar,vector}.thermal_cross_section[{open,narrow,closed}_resonance]
```

Exactly the six declared blocks, and nothing else. The declaration
cannot outlive the repair.

**Preflight** over the touched paths plus the touched markdown:

```text
RESULT: FAIL — blocked commit.
FAIL   isort --check-only
FAIL   ruff check
(9 other rows PASS; version bump SKIP, not a closing PR)
```

**Both red rows are trunk debt in the two `hazma/` modules, not this
change**, which is what
`docs/followups/todo/preflight-isort-ruff-red-on-trunk.md` predicts for
any PR that touches package code. Proved side by side rather than
asserted, against the same files at `origin/master`:

```sh
$ git show origin/master:hazma/relic_density/_thermal_functions.py > base/a.py
$ git show origin/master:hazma/vector_mediator/_gev/thermal_cross_section.py > base/b.py
$ isort --check-only base/a.py base/b.py
ERROR: base/a.py Imports are incorrectly sorted and/or formatted.
ERROR: base/b.py Imports are incorrectly sorted and/or formatted.

$ ruff check base/a.py base/b.py | grep -oE 'Found [0-9]+ error'  # origin/master
Found 32 error
$ ruff check hazma/relic_density/_thermal_functions.py \
      hazma/vector_mediator/_gev/thermal_cross_section.py \
  | grep -oE 'Found [0-9]+ error'
Found 32 error
```

Identical count before and after: this change adds no lint finding. The
same gate scoped to the five files the task authored or edited outside
`hazma/` is green on both rows:

```sh
$ scripts/agents/preflight.sh --paths "test/parity/thermal_reference.py \
      test/parity/deltas.py test/parity/test_parity.py \
      test/test_relic_density.py test/test_core_quad.py"
PASS   black --check
PASS   isort --check-only
PASS   ruff check
```

`isort` would fix the two `hazma/` files with a three-line import
reorder each, and running it here was tried and reverted: it is churn
unrelated to the repair, it does not change `RESULT` (ruff's 32
pre-existing findings still fail the run), and `AGENTS.md`'s
stay-in-scope rule puts it in its own change. The tracked follow-up owns
the tree-wide fix.

**Rule 1:** `git diff --stat -- test/parity/data` is empty.

**Rule 7 (no overlapping declarations):** B5's two cases are
`cross_sections.*`. No other roster entry reaches a `cross_sections`
case — the containment algebra in
`../references/defect-blast-radius.md` has B5 disjoint from A1–A4 and
from B1–B4.

## Plan Impact

**Impact Level:** Phase file patched (`PLAN.md` + the canonical
reference).

- `PLAN.md`: eight defects → nine; frontmatter `deliverable`, Goal,
  Scope, Numerical impact, "The defects", and Task 12's bump check.
  Task 13 added to Task Details.
- `references/defect-blast-radius.md`: B5 section, roster row, and the
  coverage arithmetic re-derived — 25 → 27 slots, union 20 → 22,
  untouched 21 → 19, still summing to 41.
- `task-notes/README.md`: Tasks table, dependency diagram, Exit
  Criteria, Findings, Decisions, the numerical-impact log, Files
  Changed, and the handoff.
- No ADR. `ADR-0001` established that corpus repairs are declared
  deltas; `Reference` is a second relation inside that decision, not a
  change to it.

`version_bump: minor` re-checked and unchanged: no public name,
signature, return shape or documented unit moves. Recorded in
`CHANGELOG.md` under Unreleased, since Task 12 ships the bump.

## Stale-state sweep

Run against this branch after the last edit.

```sh
$ grep -rn "eight defects\|eight parity-pinned\|eight deliberate" \
      projects/parity-pinned-defect-repair/ | grep -v task-13-thermal
(no occurrences)

$ grep -n "EXPECTED_DECLARED_ARRAYS = " test/parity/test_parity.py
128:EXPECTED_DECLARED_ARRAYS = 36

$ python -c "import sys; sys.path.insert(0,'test/parity'); import deltas; \
      print(len(deltas.DECLARED_DELTAS), \
            sum(1 for d in deltas.DECLARED_DELTAS.values() if d.repair=='B5'), \
            sorted(deltas.REPAIRS))"
36 6 ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4', 'B5']

$ grep -rn "DEFAULT_EPSABS\|DEFAULT_LIMIT" rust/src/kernels/vector_xs.rs \
      rust/src/kernels/scalar_xs.rs
rust/src/kernels/vector_xs.rs:173:/// `crate::quad::DEFAULT_LIMIT` because [`THERMAL_EPSABS`] is zero.
rust/src/kernels/scalar_xs.rs:164:/// `crate::quad::DEFAULT_LIMIT` because [`THERMAL_EPSABS`] is zero.
# prose only; both imports dropped, and clippy -D warnings is clean.

$ grep -rn "2\.5e-8\|2\.499e-08" test/parity/ test/test_relic_density.py
(no occurrences — superseded by 3.6e-9 once THERMAL_LIMIT rose to 100)

$ git diff --stat -- test/parity/data
(empty)
```

**Follow-up link sweep.** Moving the file to `done/` dangled 8 files'
worth of inbound paths. `docs/workflow.md:291` and
`docs/followups/todo/moved-followups-leave-dangling-inbound-paths.md`
both say to repoint **every** reference — that follow-up exists because
a previous PR read surviving references as "frozen evidence" and skipped
them, which it records as an error. All repointed:

```text
CHANGELOG.md
rust/src/kernels/scalar_xs.rs
docs/followups/todo/relic-density-odes-in-rust.md   (sibling link, ../done/)
projects/cython-to-rust/task-notes/numerical-impact.md
projects/cython-to-rust/task-notes/phase-05/README.md
projects/cython-to-rust/task-notes/phase-05/task-5.1-vector-xs.md
projects/cython-to-rust/task-notes/phase-05/task-5.2-scalar-xs.md
projects/cython-to-rust/task-notes/phase-05/task-5.3-thermal-sweep.md
projects/cython-to-rust/learnings/phase-05-mediator-cross-sections.md
```

That follow-up's own detector, re-run afterwards:

```sh
$ for p in $(grep -rhoE 'docs/followups/(todo|done)/[a-z0-9-]+\.md' \
      projects/ docs/ hazma/ test/ rust/ README.md CHANGELOG.md | sort -u); do
    [ -f "$p" ] || echo "DANGLING: $p"; done
DANGLING: docs/followups/todo/cross-section-prefactor-threshold-cancellation.md
DANGLING: docs/followups/todo/legacy-parameters-width-exponent-bug.md
DANGLING: docs/followups/todo/oracle-restore-revisions-for-the-mediator-decay-pyx.md
```

Those three are the pre-existing ones that follow-up already documents.
This task added none.

One brace-elided mention survives at
`projects/cython-to-rust/task-notes/phase-05/task-5.1-vector-xs.md:222`,
inside `docs/followups/todo/{a,b,c}.md`. It is not a resolvable path, it
does not match the detector's pattern, and it was already stale before
this task (one of its other two slugs moved in cython-to-rust Task 7.1),
so it is left for that follow-up rather than half-corrected here.

**Numerical-impact statement:** ⟨σv⟩ moves at 539 of 570 pinned corpus
positions, by up to 1.00 relative; `relic_density` moves by −99.88% to
+2.40% at the six pinned model points. Intended, declared as `B5`, and
recorded in `CHANGELOG.md` and the project's numerical-impact log.

## Open Questions

- **Both pure-Python sites return `0.0` for every `x >= 25`**, found
  while writing the round-1 regression tests: they integrate to `50/x`
  with no floor, so the interval closes at `x = 25` and inverts above it.
  Freeze-out is `x ~ 20`–`30`, and this is what makes
  `VectorMediatorGeV.relic_density` return `nan` — verified to predate
  `B5`, which shares only the two lines. A separate, larger defect than
  the one this task repaired; filed as
  `docs/followups/todo/thermal-fallback-upper-limit-collapses-at-x-25.md`
  rather than folded in. `TestThermalQuadratureConverges` caps its grid
  below 25 because above it there is no integral to check.
- **The two models disagree above `x = 300` and this task did not touch
  it.** The scalar hard-returns `0.0`; the vector clips `x` to 300 and
  saturates. The follow-up flagged the divergence and explicitly scoped
  it out, and it is unrelated to the quadrature. No follow-up filed:
  `docs/followups/` has no entry for it, but it belongs to whoever
  reconciles the two mediator kernels rather than to this repair.
- **`test/test_core_quad.py` rebuilds the thermal integrand and so does
  `test/parity/thermal_reference.py`.** The two live under different
  `sys.path` roots and take different parameters (fixed couplings vs the
  corpus blocks' own), so they are not literally the same call, but the
  six-channel sum is written twice. Folding them together means moving
  one across the `test/` ↔ `test/parity/` boundary, which is out of
  scope here.

## Handoff to Next Task

- **B5 is done and needs nothing from Tasks 3–10.** It touched no
  spectrum kernel, no boost integral and no photon table.
- **Tasks 11 and 12 inherit work from it.** Task 11's prose sweep should
  include B5's follow-up (already in `done/`). Task 12 aggregates the
  numerical impact: B5's figures are in `task-notes/README.md` under
  "Numerical impact so far" and are the largest in the project by four
  orders of magnitude, so the `CHANGELOG` entry should lead with them.
- **The delta layer now has two relations.** Pick `Additive` when the
  physics names the term, `Reference` when only a second implementation
  can say what the value should be. Both answer `term_for`; the runner
  needs no change for either. `EXPECTED_DECLARED_ARRAYS` is 36 — 30 for
  B4, 6 for B5 — and must move with any new declaration.
- **Build before believing anything.** These are Rust edits;
  `cargo test` does not re-link the extension. `uv pip install -e .
  --config-setting build-args="--features test-probes"`.
