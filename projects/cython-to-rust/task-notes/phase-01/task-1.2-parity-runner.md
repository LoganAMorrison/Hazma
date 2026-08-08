# Task 1.2: Pytest runner and tolerance budgets

**Date:** 2026-08-07
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-01-parity-corpus.md` (Task 1.2);
`../../rules.md` rules 1–3 (parity discipline);
`../../references/numerics-replacements.md` (quad call-site table,
special-function parity figures)
**Related ADRs:** ADR-0002 (license-clean numerics — sets what the
replacement implementations will be, and therefore what the budgets have
to absorb)
**Depends On:** Task 1.1

## Objective

Turn the Task 1.1 corpus into a running pytest gate: re-evaluate every
pinned entry point against the live implementation, compare within a
declared per-function tolerance budget, and replay the exceptions the
corpus recorded rather than comparing the `nan` that stands in for them.

## Exit Criteria

From `../../phases/phase-01-parity-corpus.md`, Task 1.2:

- `test/parity/test_parity.py` parametrizes over the manifest and
  compares live imports against stored arrays.
- Per-function budgets live in `test/parity/tolerances.py` with a
  one-line justification each: exact (bit-equal) for pure closed-form
  kernels against the capturing commit, documented budgets for
  quad-backed kernels (start 1e-8 rel, tighten after Phase 03
  measurement; nested-ρ gets its own line).
- The manifest's per-block `raises` records are replayed, not skipped:
  where the corpus says an entry point raised, the runner asserts the
  live implementation raises the same exception type at the same
  argument.
- Running against unmodified Cython passes bit-exact.

## Inputs Reviewed

- `../../PLAN.md` (Numerical impact; Phase table ordering).
- `../../phases/phase-01-parity-corpus.md` — Task 1.2 block and the
  phase Prerequisites.
- `../../rules.md` — rules 1–3 (corpus gates every swap; corpus is
  generated only from pre-port Cython; every numerical shift declared).
- `../../references/numerics-replacements.md` — the `quad` call-site
  table with each site's `epsabs`/`epsrel`, the `spec_math` parity
  figure (≤1e-13 relative), and the `boost_integrate_linear_interp`
  description.
- `../README.md` and `task-1.1-corpus-generator.md` — Findings and
  Handoff.
- `test/parity/README.md`, `test/parity/cases.py`,
  `test/parity/generate.py`.
- `docs/agents/lessons.md`, `docs/agents/environment.md`.
- The surviving `.pyx` sources, to classify each entry point by
  mechanism rather than by name (see Findings).

## Findings

- **Every entry point falls into one of five mechanism classes**, and
  the class — not the physics — is what sets its budget. Derived by
  reading the live sources rather than the inventory:

  | Class | rtol | Cases |
  | --- | --- | --- |
  | closed form | 0 (exact) | 19 — neutral pion, positron/neutrino muon, and all 16 non-thermal cross sections |
  | closed form + `spence` | 1e-13 | 1 — `spectra.photon.muon` |
  | tabulated boost integral | 1e-12 | 7 — 3 kaons, eta, eta', omega, phi |
  | one `quad` | 1e-8 | 5 — charged pion (photon/positron/neutrino), both `thermal_cross_section` |
  | nested `quad` | 1e-6 | 9 — both rho, all 7 mediator spectra |

  Counts re-derived from the table itself, not by hand — an earlier
  hand-count of this row said 16 and was wrong:

  ```python
  collections.Counter(b.rtol for b in tolerances.BUDGETS.values())
  # {0.0: 19, 1e-13: 1, 1e-12: 7, 1e-08: 5, 1e-06: 9}  -> 41
  ```

- **The nested class is larger than "the ρ integrals".** All seven
  mediator-spectrum entry points cimport a quad-backed pion kernel into
  a cos θ quad integrand
  (`hazma/scalar_mediator/scalar_mediator_decay_spectrum.pyx:2-3,184`
  and the three siblings), so they share ρ's failure mode. The phase
  file asks for ρ to get its own line, which it has; the mechanism is
  what put the other seven beside it.
- **`hazma/spectra/_photon/_muon.pyx` and
  `hazma/spectra/_positron/_muon.pyx` import `quad` but never call it**
  on the live path (`hazma/spectra/_positron/_muon.pyx:134` is commented
  out; `hazma/spectra/_photon/_muon.pyx:113` reaches `spence` and no
  further). Classifying from the import list rather than the call sites
  would have put two exact kernels in the 1e-8 bucket.
- **The declared budgets are not the gate on the capturing tree.** The
  corpus was captured from these exact kernels in this exact
  environment, so on that tree any difference at all is a regression and
  a 1e-8 budget would hide it. `tolerances.effective_budget` therefore
  demands bit-equality whenever the kernel digest, the toolchain and the
  numerics libraries all match the manifest, and falls back to the
  declared budgets otherwise. This is what makes "Running against
  unmodified Cython passes bit-exact" a standing gate rather than a
  one-off observation.
- **Cross-platform bit-reproducibility is still open, and now degrades
  safely.** The corpus was captured on macOS/arm64 with numpy 2.5.1 /
  scipy 1.18.0 / Cython 3.2.9. A Linux CI runner differs in `platform`
  at minimum, so it lands in budget mode rather than failing an
  exactness claim it was never in a position to meet. The `-rs` skip
  reason on `test_running_on_the_capturing_tree` names exactly what
  differed. Task 1.3 wires CI and is where the Linux numbers first get
  measured.
- **Reusing `generate.evaluate_block` is what makes the comparison
  meaningful.** It owns the `IntegrationWarning` suppression and the
  point-by-point recovery a batched call needs when one grid point
  raises. A re-implementation in the runner would have made a harness
  difference indistinguishable from an implementation difference — and
  it is also how the raise replay comes for free: the function returns
  the same `{"index", "argument", "type"}` records the manifest stores,
  so `raised == manifest_block.get("raises", {})` catches a swallowed
  raise *and* a new one.

## Decisions and Implementation Notes

- **One test per corpus block** (623), not per case (41) or per array
  (1,580). A block is one grid at one fixed argument set, which is the
  granularity at which a failure is diagnosable; the id is
  `case[block-label]`, so a failure names the model point and parent
  energy directly.
- **The stored grid is asserted against the re-derived one before any
  value is compared.** `cases.py` is re-evaluated live, so a change to
  grid construction would otherwise move every abscissa and leave the
  value comparison comparing two different samplings. The assertion is
  exact and unconditional — grids are arithmetic on constants.
- **Abscissae are compared exactly in both modes.** `grid` and
  `scalar_grid` record *where* an entry point was sampled, not what it
  returned; no tolerance on a value compensates for having moved the
  point it was measured at, so the budget never reaches them.
- **`atol = 0.0` for every case.** One absolute floor cannot serve
  spectra at ~1e-3 MeV⁻¹ and cross sections at ~1e-20 MeV⁻²; it is also
  unnecessary, since the sub-threshold and above-endpoint regions return
  exactly `0.0` and `|0 − 0| ≤ rtol·0` holds. A port returning 1e-300
  where the Cython returned zero fails, which is the intended answer.
- **No measurement/reporting hook.** An earlier draft added a
  `--parity-report` pytest option; dropped for two reasons.
  `pytest_addoption` is only honored in an *initial* conftest, which
  `test/parity/conftest.py` is not under `pytest test`, and
  `numpy.testing.assert_allclose` already prints the max relative
  difference on breach. Phase 03's tightening loop is therefore "set the
  budget to the value you want and read what the failure reports", with
  no extra machinery to keep working.
- **Three small changes to Task 1.1's modules rather than duplicates
  here:** `cases.rust_core_available()` (the predicate
  `assert_no_rust_core` already needed, now also read by
  `tolerances.provenance`), `generate.load_manifest()` (one spelling of
  where the manifest lives, shared with `check()`), and the
  all-points-raised guard in `generate._sweep_pointwise` (see
  Verification).
- **`hazma`'s own version is excluded from the provenance comparison.**
  Phase 07 bumps it without touching a number; letting a version bump
  silently drop the gate out of exact mode would be the wrong trade.

## Files Changed

- `test/parity/test_parity.py` — new. 623 block tests plus three
  guards (capturing-tree report, budget-table coverage, budget
  justification).
- `test/parity/tolerances.py` — new. `Budget`, `Provenance`, the 41-entry
  budget table, `budget_for`, `provenance`, `effective_budget`.
- `test/parity/cases.py` — `rust_core_available()` extracted from
  `assert_no_rust_core`.
- `test/parity/generate.py` — `load_manifest()` extracted from `check()`;
  `_sweep_pointwise` now raises a `RuntimeError` naming an
  all-points-raised block instead of dying inside `np.concatenate`.
- `test/parity/README.md` — the runner and the two comparison modes
  documented alongside the generator.
- `../../phases/phase-01-parity-corpus.md` — Prerequisites context bullet
  and Task 1.3's `pytest -q test` figure, both falsified by this task.
- `task-1.1-corpus-generator.md` — one forward-looking claim about what
  Task 1.2 would find out, annotated with what it actually did.
- `../../learnings/phase-00-dead-code-purge.md` — its present-tense
  "zero compiled-layer pinned tests run anywhere" annotated as
  half-closed, with the new suite figures.
- `README.md`, `../README.md`, this file — status bookkeeping.

Nothing under `hazma/` was touched.

## Verification

Environment: a scratchpad venv on Python 3.12.12, numpy 2.5.1, scipy
1.18.0, macOS-26.5.2-arm64 — the same tuple the manifest records, and
the kernel digest matches (`f5e6e269be47`), so the runs below are in
**exact** mode.

### The gate itself

```text
$ pytest -q test/parity -rs
626 passed in 278.83s (0:04:38)
```

626 = 623 corpus blocks + `test_running_on_the_capturing_tree` (passed,
i.e. exact mode was active and not silently skipped) +
`test_every_corpus_case_has_a_budget` + `test_every_budget_states_a_reason`.
This is the phase file's "Running against unmodified Cython passes
bit-exact" criterion: every one of the 41 entry points reproduced its
pinned values with `rtol=0, atol=0`.

Collection sanity (pytest exits 5 on zero collected — it did not):

```text
$ pytest test/parity --collect-only -q | tail -n 1
626 tests collected in 4.47s
```

Corpus integrity is unaffected by this task's edits to `generate.py`:

```text
$ python test/parity/generate.py --check
corpus OK: 41 cases / 1580 arrays match the manifest
(generated at 010747c6125d, kernel digest f5e6e269be47)
```

### Test validity (stash-proof)

A parity gate that passes is uninformative unless its assertions fire on
a broken implementation, and the production code here is Cython that
cannot be stashed. Each scenario instead patches `Case.resolve` (or the
specification) to break exactly one thing, then runs the real test
function. Baselines are run first, on the very blocks the scenarios
break, so a failure proves the scenario and not the block.

```text
$ python negative_tests.py
raising block: cross_sections.vector.sigma_xx_to_v_to_pipi[1]
PASS  baseline raising block: clean
PASS  baseline closed-form block: clean
PASS  baseline quad-backed block: clean
PASS  raise swallowed -> returns nan: raised AssertionError -- ...exceptions changed.
PASS  new raise where corpus has none: raised RuntimeError -- every one of the 250 grid
      points raised (TypeError); the entry point is broken, not merely singular at an edge
PASS  new raise at one grid point: raised AssertionError -- ...exceptions changed.
PASS  closed form shifted by 1e-15 rel: raised AssertionError -- Not equal to tolerance
      rtol=0, atol=0
PASS  quad kernel shifted by 1e-6 rel against a 1e-8 budget: raised AssertionError --
      Not equal to tolerance rtol=1e-08, atol=0
PASS  cases.py grid drifted: raised AssertionError -- Arrays are not equal
PASS  case with no declared budget: raised AssertionError -- ...disagree.
PASS  budget with no corpus case: raised AssertionError -- ...disagree.
PASS  budget with a blank reason: raised AssertionError -- budgets with no justification:
      ['spectra.positron.muon']

all negative tests fired as expected
```

What each one covers, against the Exit Criteria:

- **Raise replay** (the criterion the phase file added in Task 1.1) —
  scenarios 1, 2, 2b: an implementation that swallows the pinned
  `TypeError` and returns `nan` fails; one that raises everywhere fails;
  one that starts raising at a single new point fails. Scenario 1 is
  precisely the failure a `nan`-only comparison would miss.
- **Budgets bite in both modes** — scenarios 3 and 4: a 1e-15 relative
  shift fails on the capturing tree, and a 1e-6 shift fails against a
  1e-8 budget with the exact-mode override disabled. Without scenario 4
  the declared budgets would be untested until Phase 04.
- **Grid drift** — scenario 5: perturbing `cases.ANCHOR_OFFSETS` fails
  before any value is compared.
- **Table guards** — scenarios 6 and 7: a corpus case with no budget, a
  budget with no case, and a blank justification each fail.

The script lives in the session scratchpad rather than the repo: it
mutates module state globally (patching `Case.resolve`, swapping
`tolerances.BUDGETS`), which is fine for a one-shot proof and would be a
liability inside a suite that also runs the real gate.

### Two negative results worth recording

- **`spectra.photon.neutral_pion[rest]` is unusable as a perturbation
  target.** It has exactly one non-zero value and that value is `inf`
  (the rest-frame two-body delta), so `inf * (1 + 1e-15)` is still
  `inf`. Scenario 3 silently passed against a "broken" implementation
  until the case was switched to `spectra.positron.muon`. Recorded in
  the phase README's Findings — the same trap awaits anyone spot-checking
  a port against that block.
- **`generate._sweep_pointwise` had no answer for a block where every
  point raises.** The fill shape is read off a successful result; with
  none, `shape` fell back to `()` and `np.concatenate` died with
  `ValueError: zero-dimensional arrays cannot be concatenated`. The gate
  still failed, but on a message that names numpy rather than the
  kernel. Now a `RuntimeError` saying so. Unreachable during generation
  (no corpus block raises everywhere) and reachable in Phase 04 the
  first time a port breaks a kernel outright, which is why it was worth
  fixing here rather than filing.

### Numerical impact

**No public value changes (verified: `pytest -q test/parity` → 626
passed in exact mode; `git diff origin/master --name-only -- 'hazma/*'`
→ empty).** Nothing under `hazma/` is touched; the whole diff is
`test/parity/` plus project bookkeeping. The parity run is itself the
strongest available statement of no-change: all 41 compiled entry points
reproduce their pinned values bit-for-bit.

### Suites

Both re-run on the final tree, after every code edit:

```text
$ pytest -q test -rs
870 passed, 20 skipped in 527.17s (0:08:47)

$ pytest -q
57 passed, 10 skipped in 0.42s
```

### Preflight gate

```text
$ scripts/agents/preflight.sh \
    --paths "test/parity/cases.py test/parity/generate.py \
             test/parity/test_parity.py test/parity/tolerances.py" \
    --tests "test" --md "<the seven markdown files above>"
PASS   black --check
PASS   isort --check-only
PASS   ruff check
PASS   pytest                  870 passed, 20 skipped in 535.60s (0:08:55)
PASS   import hazma            version 2.1.0
PASS   markdownlint
SKIP   version bump            not a closing PR (pass --closing)
PASS   forbidden tokens        none added
RESULT: PASS
```

No WARN rows: every gate ran. `--paths` names the four `.py` files
rather than the `test/parity` directory, per the project README's note
that a directory drags in unrelated unformatted files.

`pytest -q test` is 244 + 626: the pre-existing suite is untouched and
the parity gate is additive. The 20 skips are the pre-existing ones (9 +
8 "Needs to be updated" mediator classes, 3 form-factor cases) — Task 1.4
owns the first two groups. Bare `pytest` is unchanged at 57/10 because
`setup.cfg`'s `testpaths = hazma` still keeps it inside the package; CI
runs that form, which is exactly the gap Task 1.3 closes.
_(Closed 2026-08-07: Task 1.3 moved the config to `pyproject.toml` with
`testpaths = ["hazma", "test"]`. There is one suite now — bare
`pytest -q` → `935 passed, 30 skipped` — and the 57/10 figure above is
history.)_

## Open Questions

- **Cross-platform bit-reproducibility is measured by nobody yet.** The
  corpus was captured on macOS/arm64; no Linux runner was available
  here. This is no longer a risk to the gate — a differing `platform`
  puts the run in budget mode by construction — but the actual Linux
  numbers are unknown, and Task 1.3 is where they first appear. The
  plausible outcome is last-ulp differences in the transcendental-libm
  kernels (`exp`, `log`, `spence`), which every declared budget absorbs
  with room to spare. If any case *fails* its declared budget on Linux,
  that is a genuine finding about the tolerance, not about the harness.
- **The parity suite costs ~4.6 minutes.** Task 1.3 puts it on the CI
  matrix. Whether that warrants a marker or a separate job is a policy
  call that belongs to the task doing the wiring; Task 1.2 deliberately
  did not invent one, and registered no new markers.

## Plan Impact

**Impact Level:** Update phase file.

`../../phases/phase-01-parity-corpus.md` changed in two places, both
because this task falsified them:

1. The Prerequisites "Context" bullet said pinned-value tests over
   compiled code execute **nowhere**, and that Task 1.1 "added
   `test/parity/` but no pytest module, so this still holds". Half of
   that is now false: `pytest test` reaches the parity suite. The other
   half is not — `setup.cfg`'s `testpaths = hazma` still keeps CI out of
   it, which is exactly Task 1.3's job. Reworded to say which half moved.
2. Task 1.3's exit criteria quote `pytest -q test` → 244 passed / 20
   skipped "as of 2026-08-07". This task adds 626 tests to that
   collection. Re-derived rather than left to rot, per the phase file's
   own instruction to re-derive rather than quote.

No ADR. Nothing about the port's architecture, interfaces, units or task
ordering moved. The tolerance table is a new contract, but the phase file
already specified that it would exist and what shape it would take.

## Stale-state sweep

### Identifier sweep

New/changed names: `test_parity`, `tolerances`, `Budget`, `Provenance`,
`BUDGETS`, `budget_for`, `provenance`, `effective_budget`,
`rust_core_available`, `load_manifest`, `ABSCISSAE`.

```text
$ rg -n 'effective_budget|budget_for|rust_core_available|load_manifest|test_parity|tolerances' \
    projects/ docs/ README.md hazma/ test/ | grep -v '^test/parity/'
projects/cython-to-rust/phases/phase-01-parity-corpus.md:22   ...numerics-replacements.md` (call-site tolerances)   KEPT (unrelated word)
projects/cython-to-rust/phases/phase-01-parity-corpus.md:30   `test/parity/test_parity.py` runs under `pytest test`   EDITED (this task)
projects/cython-to-rust/phases/phase-01-parity-corpus.md:69   exit criterion naming test_parity.py                    KEPT (the spec; satisfied)
projects/cython-to-rust/phases/phase-01-parity-corpus.md:71   "test/parity/tolerances.py (or `.toml`)"                KEPT (spec permitted either; `.py` chosen)
projects/cython-to-rust/rules.md:12                           "test/parity/tolerances.*"                          KEPT (glob covers the `.py` that now exists)
hazma/spectra/_fsr.py:347                                     "Absolute and relative tolerances"                  KEPT (unrelated word)
docs/workflow.md:178                                          "not tolerances"                                    KEPT (unrelated word)
projects/cython-to-rust/references/numerics-replacements.md:3,58                                                  KEPT (unrelated word)
projects/cython-to-rust/task-notes/phase-01/task-1.2-parity-runner.md:*                                            KEPT (this note)
```

No file outside `test/parity/` names a symbol this task added, so there
was nothing to repoint.

### Line-number citation sweep

Every markdown file this task touched, on the final tree:

```text
$ python3 scripts/agents/check_doc_citations.py \
    projects/cython-to-rust/task-notes/phase-01/task-1.2-parity-runner.md \
    projects/cython-to-rust/task-notes/phase-01/README.md \
    projects/cython-to-rust/task-notes/README.md \
    projects/cython-to-rust/phases/phase-01-parity-corpus.md \
    projects/cython-to-rust/task-notes/phase-01/task-1.1-corpus-generator.md \
    projects/cython-to-rust/learnings/phase-00-dead-code-purge.md \
    test/parity/README.md
docs scanned: 7
in-repo citations checked: 11
  resolved by exact: 11
external citations skipped: 0
out-of-range or ambiguous: NONE
```

An earlier run of the same command reported one **suffix** resolution —
a basename-only citation of the positron muon kernel in this note —
EDITED to the full repo-relative path
(`hazma/spectra/_positron/_muon.pyx:134`) per the `elided-doc-paths`
lesson, which is why the final run is 11/11 exact. The checker does not
scan `.py`, so `tolerances.py`'s citations were swept by hand for the
same class; two basename-only `_rho.pyx` citations were EDITED to full
paths there.

### Forward-looking phrase sweep

```text
$ rg -n '(Task [0-9.]+ will|will be added|still pending|today: ?stub)' \
    projects/cython-to-rust/ test/parity/
phase-00/task-0.1-relocate-constants.md:414    the sweep command, quoted            KEPT
phase-01/task-1.1-corpus-generator.md:452      the sweep's own null result, quoted  KEPT
phase-01/task-1.1-corpus-generator.md:562      "Task 1.2 will find out when it
                                                sets tolerances"                    EDITED
phase-01/task-1.2-parity-runner.md:411         this block, quoting the hit above    KEPT
```

Task 1.1's claim was falsified by this task — 1.2 set the tolerances but
did not find out, because no Linux runner was available. Annotated in
place with what actually happened and where the measurement moved
(Task 1.3), rather than rewritten.

A present-tense sibling of the same claim turned up outside the pattern,
in `../../learnings/phase-00-dead-code-purge.md:148` ("**Zero
compiled-layer pinned tests run anywhere**"). Found by grepping the
claim's distinctive phrase rather than the forward-looking pattern —
the `sibling-copies-of-a-fixed-claim` lesson — and annotated as
half-closed with the new suite figures:

```text
$ rg -n "Zero compiled-layer pinned tests|zero pinned-value|no pinned-value" \
    projects/ docs/ .claude/ .codex/
projects/cython-to-rust/learnings/phase-00-dead-code-purge.md:148             EDITED
projects/cython-to-rust/task-notes/README.md:67 ("Zero compiled-layer
  pinned tests run anywhere today")                                          EDITED
projects/cython-to-rust/phases/phase-01-parity-corpus.md:24 ("zero
  pinned-value tests ... execute anywhere")                                  EDITED
```

### Count sweep

| Claim location | Command | Actual | Status |
| --- | --- | --- | --- |
| This note, class table (19/1/7/5/9) | `Counter(b.rtol for b in tolerances.BUDGETS.values())` | `{0.0: 19, 1e-13: 1, 1e-12: 7, 1e-08: 5, 1e-06: 9}` = 41 | **Corrected** — an earlier hand-count said 16 exact |
| This note + phase README, "623 blocks / 626 tests" | `pytest test/parity --collect-only -q \| tail -1` | `626 tests collected` | OK |
| This note, corpus size "41 cases / 1580 arrays" | `python test/parity/generate.py --check` | `41 cases / 1580 arrays` | OK |
| Phase file Task 1.3, `pytest -q test` | `pytest -q test` | `870 passed, 20 skipped` | **Updated** from `244 / 20` — this task added 626 |
| Phase file Task 1.3, off-capture figure `869/21` | not runnable here (no Linux runner) | — | **Derived, not measured** — stated as an expectation with its reason, not as a reading |
| Project README handoff, bare-`pytest` figure | `pytest -q` | `57 passed, 10 skipped` | OK — unchanged, `testpaths = hazma` |
| TABULATED budget rationale, "501 rows" | `wc -l hazma/spectra/_photon/data/*.csv` | 501 (six tables), 101 (eta) | OK |
| Phase README, "~4.6 min" parity runtime | `pytest -q test/parity` | `278.83s (0:04:38)` | OK |

### Numerical-impact statement

**No public value changes.** Nothing under `hazma/` is in the diff
(`git diff origin/master --name-only -- 'hazma/*'` → empty; `git status
--short` lists only `test/parity/` and `projects/`). The grid is the
corpus itself — all 41 consumed compiled entry points over 623 blocks /
179,695 pinned values — and every value reproduced at `rtol=0, atol=0`
(`pytest -q test/parity` → `626 passed`).

### Exit Criteria → test mapping

| Exit criterion | What satisfies it |
| --- | --- |
| Runner parametrizes over the manifest, compares live imports to stored arrays | `test/parity/test_parity.py::test_entry_point_matches_corpus`, 623 params built from `MANIFEST["cases"]` |
| Per-function budgets with a one-line justification each | `test/parity/tolerances.py::BUDGETS` (41 entries); enforced by `test_every_corpus_case_has_a_budget` and `test_every_budget_states_a_reason` |
| Exact for closed-form kernels against the capturing commit | `tolerances.effective_budget` + `tolerances.provenance`; asserted live by `test_running_on_the_capturing_tree` (passed, not skipped) |
| Quad-backed budgets start 1e-8; nested-ρ its own line | `QUAD_RTOL = 1e-8` on 5 cases; `NESTED_RTOL = 1e-6` with explicit `spectra.photon.charged_rho` and `spectra.photon.neutral_rho` entries |
| `raises` records replayed, not skipped | `assert raised == manifest_block.get("raises", {})`; negative scenarios 1, 2 and 2b prove it fires |
| Running against unmodified Cython passes bit-exact | `pytest -q test/parity` → `626 passed` in exact mode |

### Task-note self-consistency

`**Status:** Complete` matches the phase README row and every Exit
Criterion having a mapping row. Every symbol named in §Files Changed and
§Decisions (`Budget`, `Provenance`, `BUDGETS`, `budget_for`,
`provenance`, `effective_budget`, `rust_core_available`,
`load_manifest`, `_sweep_pointwise`) appears in `git diff --stat
origin/master --` or in a file this task created.

## Handoff to Next Task

**Read first:** `test/parity/README.md` ("What the gate compares" is
new), then `test/parity/tolerances.py`'s module docstring, then this
note's Findings.

**Currently safe to assume:**

- The corpus reproduces bit-exactly on the capturing environment, and
  the gate proves it on every run rather than by report. Task 1.3 is
  wiring, not repair.
- The tolerance table cannot silently drift out of sync with the corpus:
  `test_every_corpus_case_has_a_budget` fails in both directions.
- The two comparison modes are not a hidden switch. `pytest
  test/parity -rs` names the active one; a skipped
  `test_running_on_the_capturing_tree` means budget mode and its reason
  says what differed.
- `generate.evaluate_block` is now shared between the generator and the
  gate. Changing it changes both — which is the point, but it means a
  change there needs a corpus `--check` *and* a parity run, not one of
  the two.

**Currently risky / unknown:**

- Expect the parity suite to land in **budget mode** on CI, with one
  extra skip. That is correct behavior, not a regression; a Task 1.3
  agent who sees `625 passed, 1 skipped` where this note says `626
  passed` is looking at a Linux runner, and `-rs` will say so.
- The ~4.6-minute cost, per Open Questions.
- Do not regenerate the corpus to make a failing case pass. `rules.md`
  rule 2 and `tolerances.py`'s docstring both say so; the remedy is a
  declared budget widening plus an entry in "Numerical impact so far".
- Do not hoist `cases.py`'s deferred model imports — `generate.py
  --check` still has to work on an unbuilt tree, and `tolerances.py`
  imports `cases` at module scope now, so the runner shares that
  constraint.
