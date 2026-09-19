# Task 8 review response — PR #97

**Date:** 2026-09-18
**Reviewed head:** `278b1ee9602e26cacb1caea12c4320fce5033960`
**Status:** Complete — published fixes passed CI at 3d124214
**Scope:** Respond to the supplied REQUEST CHANGES review; no kernel change.

## Assessment

| Item | Category | Action |
| --- | --- | --- |
| Blocking 1: shared A3 tolerance is too tight for nested consumers | fix | Keep the pion's existing 1e-12 budget and register A3/nested at the existing 1e-9 nested budget; replay the Linux values through the actual gate |
| Non-blocking 1: Complete conflicts with red CI | fix | Mark task and working-memory row In Progress until the revised code passes CI |
| Non-blocking 2: section headings lack PR attribution | fix | Add PR #97 to Files Changed and Numerical impact; label original verification and sweep by their commit |
| Non-blocking 3: deleted convergence reasoning | fix | Remeasure the clipped-interval grid and retain the reason not to pin the off-domain flag map |
| Non-blocking 4: docstrings narrate the change | fix | Describe the current support and pinned oracle value in present tense |
| Non-blocking 5: inline mass literals | fix | Name pion/electron masses and the derived endpoint at module scope |

All six comments are accepted. No deferred or rejected items. The proposed
NumPy SIMD explanation is plausible but is not established by the evidence.

## Numerical assessment

The published Linux log is the measurement, not the reviewer's claim:

```sh
gh api --allow-escape-sequences repos/LoganAMorrison/Hazma/actions/jobs/105845601264/logs
```

CI run 35423602586, Linux Python 3.11, reports actual
0.0005252181267229549 versus captured 0.0005252181267179045 MeV^-1.
Computing `abs(actual - captured) / abs(captured)` gives
**9.615666106602723e-12**. The sibling point entry fails identically.
The original relation used 1e-12 on positions whose existing case budget
is 1e-9. The relation's portability model was incorrectly shared; the
case tolerances and integration stopping criteria are not changed.

The 66 standalone A3 declarations now split into six pion arrays at
1e-12 and 60 nested arrays at 1e-9. The 20 A3+B4 arrays retain their 1e-3
composite budget and strict scalar rest check. There remain 165 declared
arrays, with unchanged position sets and independent captures.

`DELTA_MODELS["A3/nested"]` names the consumer variant; its `Delta.repair`
is still A3. Shape tests require the prefix to equal the roster repair,
the variant to be an identifier, evidence to exist, and declarations to
reference registered model objects. No new physics repair or relation
protocol is introduced. Corpus-repinning guidance documents the key form.

## Convergence remeasurement

Temporarily print `epi`, `egam` and `outcome.ier` after the production-bound
quad in the existing Rust nonconvergence test, then run:

```sh
cargo test --manifest-path rust/Cargo.toml --no-default-features --features test-probes the_non_converging_regime_is_reachable -- --nocapture
```

The print was removed before rebuilding the Python extension and running
the final gates. On macOS/arm64, the sampled grid gives:

| Pion energy, MeV | Nonconverging photon energies, MeV |
| --- | --- |
| 1000 | none |
| 30000 | none |
| 40000 | none |
| 60000 | 0.01 |
| 80000 | 0.01, 0.1 |
| 100000 | 0.001, 0.01, 0.1 |

The grid is the test's eight photon energies from 0.001 to 10000 MeV.
Of its 48 pairs, two are above support and return zero without a quad;
46 quadratures run, six report Divergent. This is a sampled transition,
not a universal 60 GeV boundary. The comment retains the rationale:
adaptive subdivision and extrapolation amplify platform arithmetic
variation, so the exact off-domain flag map is not a portable invariant.

## Files changed — PR #97 review follow-up

- `test/parity/deltas.py`: separate registered nested-consumer variant.
- `test/parity/test_parity.py`: validate named model variants.
- `test/parity/test_pion_repair.py`: named constants, budget guard, Linux
  replay and unrepaired-array rejection.
- `test/test_core_photon_pion.py`: present-tense docstrings.
- `rust/src/kernels/photon_pion.rs`: measured convergence comment only.
- Corpus-repinning reference, task note, working memory, this response,
  and the existing historical-label/portability lesson entries.

## Numerical impact — PR #97 review follow-up

No public values change in this review follow-up. The task note's capture
recipe was run before editing and after the rebuilt extension was verified
inside this worktree. Comparing with `array_equal(..., equal_nan=True)`
gives **309 arrays, 86,179 values; zero changed arrays**. This includes
all six A3 corpus cases, seven pion energy grids and two generic two-pion
grids. Source changes in the Rust kernel are comments only.

## Verification — local review before 3d124214

The Linux replay substitutes the three logged values at absolute indices
144, 146 and 148 into the captured vector array, and calls the production
`_assert_declared_delta`. It also substitutes the unrepaired corpus and
requires failure under the corrected nested budget. Restoring only
`test/parity/deltas.py` to the reviewed head while keeping the new tests:

```text
pytest -n 0 -q test/parity/test_pion_repair.py::test_linux_vector_residuals_and_reversion
2 failed in 0.56s
```

After restoring the consumer variants:

```text
pytest -q test/parity test/test_core_photon_pion.py
729 passed, 1 skipped in 4.24s
```

Full preflight after the rebuild (Python paths plus all six modified
Markdown files; Rust checks run automatically):

```sh
PATH="$PWD/.venv/bin:$PATH" scripts/agents/preflight.sh \
  --paths "test/parity/deltas.py test/parity/test_parity.py test/parity/test_pion_repair.py test/test_core_photon_pion.py" \
  --md "docs/agents/lessons.md docs/agents/lessons-examples.md projects/parity-pinned-defect-repair/references/corpus-repinning.md projects/parity-pinned-defect-repair/task-notes/README.md projects/parity-pinned-defect-repair/task-notes/task-8-charged-pion-cone.md projects/parity-pinned-defect-repair/task-notes/task-8-review-response.md"
```

```text
preflight — /Users/logan.morrison/dev/Hazma/.codex/worktrees/parity-pinned-defect-repair/task-8-charged-pion-cone (base origin/master)
-------------------------------------------------------------------
PASS   black --check           test/parity/deltas.py test/parity/test_parity.py test/parity/test_pion_repair.py test/test_core_photon_pion.py
PASS   isort --check-only      0 new, 2 fixed (0 pre-existing at 19c054e3b9a5)
PASS   ruff check              0 new, 2 fixed (0 pre-existing at 19c054e3b9a5)
PASS   cargo fmt --check       rust/
PASS   cargo clippy            rust/
PASS   cargo test              rust/
PASS   pytest                  2305 passed, 16 skipped, 1 warning, 37 subtests passed in 27.25s
PASS   import hazma            version 2.2.0
PASS   markdownlint            docs/agents/lessons.md docs/agents/lessons-examples.md projects/parity-pinned-defect-repair/references/corpus-repinning.md projects/parity-pinned-defect-repair/task-notes/README.md projects/parity-pinned-defect-repair/task-notes/task-8-charged-pion-cone.md projects/parity-pinned-defect-repair/task-notes/task-8-review-response.md
SKIP   version bump            not a closing PR (pass --closing)
PASS   forbidden tokens        none added
-------------------------------------------------------------------
RESULT: PASS
```

An earlier run caught one overlong Markdown line; it was wrapped. Later
edits only complete this evidence and clarify the registry's docstring;
formatting, documentation, and citation checks were repeated after them.

This is standalone review-respond work: no commit, push, or fresh remote CI
is claimed. The pushed head's CI remains red until the fixes are published.

## Stale-state sweep — local review before 3d124214

Commands were run after edits, and repeated after this block was pasted.
Filename lists below are folded from sorted output; all references to
this note in the quoted patterns also match the note itself.

### Pre-fix claims

```sh
rg -n --hidden 'tighter than the nested|A3 uses|1.55e-13|Task 8 restricts|restores this value|convergence boundary' projects/ docs/ hazma/ test/ .claude/ .codex/ README.md CHANGELOG.md
```

```text
projects/parity-pinned-defect-repair/task-notes/README.md
projects/parity-pinned-defect-repair/task-notes/task-8-charged-pion-cone.md
test/parity/deltas.py
test/test_core_photon_pion.py
test/vector_mediator/herwig4dm/4pi/run.neutral.1.29.dat
test/vector_mediator/herwig4dm/4pi/run.neutral.2.89.dat
test/vector_mediator/herwig4dm/4pi/run.neutral.4.19.dat
```

EDITED: deltas, pion-test docstrings, Task 8 note and working memory.
The local residual is retained with explicit macOS scope. KEPT: the
three numerical data files matched by the unescaped decimal regex;
those entries do not describe a relation budget or convergence claim.

### Post-fix claims and identifiers

```sh
rg -n --hidden 'tighter than the nested|A3 uses.*1e-12|Task 8 restricts|restores this value' projects/ docs/ hazma/ test/ .claude/ .codex/ README.md CHANGELOG.md rust/
```

Folded output: Task 8 note and this review note. KEPT: the Task 8 line
now explicitly splits 1e-12 pion from 1e-9 consumers; this note quotes
the prior claims as sweep evidence. No live shared-1e-12 claim remains.

```sh
rg -n --hidden 'A3/nested|_A3_NESTED|test_linux_vector_residuals_and_reversion|successful_result_length|PHOTON_ENDPOINT_PIRF' projects/ docs/ hazma/ test/ .claude/ .codex/ README.md CHANGELOG.md rust/
```

```text
projects/parity-pinned-defect-repair/references/corpus-repinning.md
projects/parity-pinned-defect-repair/task-notes/task-8-charged-pion-cone.md
projects/parity-pinned-defect-repair/task-notes/task-8-review-response.md
rust/src/kernels/photon_pion.rs
test/parity/deltas.py
test/parity/test_pion_repair.py
test/test_core_photon_pion.py
```

All KEPT: current implementation, tests, scoped measurement or contract.

```sh
rg -n --hidden 'one object per|one entry per modelled|keyed.*roster|by roster label|DELTA_MODELS.*label' projects/ docs/ hazma/ test/ .claude/ .codex/ README.md CHANGELOG.md
```

EDITED: working memory's live one-object-per-repair claim and registry
module guidance. KEPT: the dated Task 4 implementation record. The
registry comment now allows named variants. This note's quoted sweep
matches itself and is KEPT as evidence.

### Status, forward prose and line citations

```sh
rg -n '\*\*Status:\*\*|\| 8 \| Repair A3' projects/parity-pinned-defect-repair/task-notes/task-8-charged-pion-cone.md projects/parity-pinned-defect-repair/task-notes/README.md projects/parity-pinned-defect-repair/task-notes/task-8-review-response.md
```

Task 8 note, working-memory row and review response all say In Progress
with updated CI pending. KEPT: the project's own In Progress status and
the command quoted here. The old Complete claims in the task note's
initial sweep are labeled historical at 278b1ee9.

```sh
rg -n 'Task [0-9]+ will|will be added|still pending|today: ?stub|currently|In Progress' projects/parity-pinned-defect-repair/ hazma/
```

```text
hazma/theory/_theory_constrain.py
projects/parity-pinned-defect-repair/PLAN.md
projects/parity-pinned-defect-repair/task-notes/README.md
projects/parity-pinned-defect-repair/task-notes/_template.md
projects/parity-pinned-defect-repair/task-notes/task-10a-neutrino-pion-line.md
projects/parity-pinned-defect-repair/task-notes/task-13-thermal-quadrature.md
projects/parity-pinned-defect-repair/task-notes/task-3-closed-form-deltas.md
projects/parity-pinned-defect-repair/task-notes/task-4-boost-window.md
projects/parity-pinned-defect-repair/task-notes/task-6-phi-lines.md
projects/parity-pinned-defect-repair/task-notes/task-7-photon-muon-endpoint.md
projects/parity-pinned-defect-repair/task-notes/task-8-charged-pion-cone.md
projects/parity-pinned-defect-repair/task-notes/task-8-review-response.md
```

KEPT: active project/task status, future work and explicitly historical
records; no fresh Complete claim is used to represent the red CI head.

```sh
rg -n --hidden 'deltas\.py:[0-9]+|photon_pion\.rs:[0-9]+|test_core_photon_pion\.py:[0-9]+|test_pion_repair\.py:[0-9]+' projects/ docs/ hazma/ test/ .claude/ .codex/ README.md CHANGELOG.md rust/
```

Folded output: cython-to-rust Task 4.5 note and parity-pinned-defect-repair
Task 3 note. KEPT: dated sweep records. No affected live numeric citation.
The explicit six-document citation check returns:

```text
docs scanned: 6
in-repo citations checked: 1
  resolved by suffix: 1
external citations skipped: 1
  hazma/_utils/boost.pyx (1)
out-of-range or ambiguous: NONE
```

The deleted boost path is in the existing historical PR #61 example;
it is not a claim that the twin remains importable.

### Counts, invariants and exit mapping

| Claim | Re-derivation | Result |
| --- | --- | --- |
| Review items | Assessment table | 6 fix, 0 deferred, 0 rejected |
| Changed review files | `git diff --name-only` plus untracked response note | 11 |
| A3 budgets | `Counter(d.relation.rtol for d in deltas.DECLARED_DELTAS.values() if d.repair == "A3")` | 60 at 1e-9; 6 at 1e-12 |
| Declared arrays | `len(deltas.DECLARED_DELTAS)` | 165, unchanged |
| Physics preserved | Before/after capture recipe from Task 8 | 309 arrays, 86179 values, zero changed arrays |
| Immutable artifacts | `git diff HEAD --stat -- test/parity/data test/parity/oracles/data test/parity/tolerances.py` | Empty |
| Replay is discriminating | Run replay tests with old deltas, then restore | 2 failed under old model; corrected tests pass |
| Live task status | `scripts/agents/resolve_task.py --project parity-pinned-defect-repair` | Task 8 remains the next actionable task |

The Assessment table maps every supplied comment to its fix. The Linux
replay and full preflight verify the budget correction; source comments
and measured grid address convergence; headings and status align the
records. No production expression changed. Fresh remote CI is explicitly
outstanding rather than represented by the local replay.

## Published verification

The fixes were committed and pushed as
`3d1242144a85a9c8467880b7143e3358d48003c9`. The remote branch and local
HEAD matched.
[CI run 35424652888](https://github.com/LoganAMorrison/Hazma/actions/runs/35424652888)
completed successfully on that exact code revision:

```sh
gh run view 35424652888 --json headSha,conclusion,url,jobs --jq '{headSha,conclusion,url,jobs: [.jobs[] | {name,conclusion}]}'
```

```text
headSha: 3d1242144a85a9c8467880b7143e3358d48003c9
conclusion: success
Rust (fmt, clippy, test): success
Test (ubuntu-latest, py3.13): success
Test (ubuntu-latest, py3.10): success
Test (ubuntu-latest, py3.12): success
Test (ubuntu-latest, py3.11): success
Test (ubuntu-latest, py3.14): success
Lint: success
Test (macos-latest, py3.14): success
```

The output above is folded from the command's JSON. All eight checks
passed, including both previously failing Linux jobs. Task 8 is Complete;
request ACCEPT. The local-review sections above remain dated evidence of
the earlier uncommitted state, not claims about the published head.

### Completion status sweep

```sh
rg -n --hidden 'review fixes prepared|prepared locally|pushed head.*CI|pushed head.*red|updated CI pending|published CI pending|current status and review|remains In Progress|no commit, push|standalone review-respond' projects/parity-pinned-defect-repair/ docs/agents/ .claude/ .codex/
```

Before: EDITED the Task 8 working-memory row and handoff, task-note status
and Open Questions, review-response status and remaining-verification
section, and the lesson example's current-state wording. KEPT the Task 7
project-status history. After: remaining hits are in explicitly historical
verification/sweep records, the still-active project's status, and this
quoted command. No active Task 8 status says the fixes await publication
or the code's CI remains red. The project itself remains In Progress.
