# Task 1.4: Retire or regenerate the legacy `.npy` suites

**Date:** 2026-08-08
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-01-parity-corpus.md` (Task 1.4 and
the phase Exit Criteria); `../../rules.md` rules 1–3 (parity discipline)
**Related ADRs:** none
**Depends On:** Task 1.2 (the corpus runner is the comparison target for
the redundant-vs-complementary call)

## Objective

Close the last hole Phase 01 left in the one pytest gate: the two
`@pytest.mark.skip(reason="Needs to be updated")` mediator classes and
their 90 `.npy` reference arrays, the 0-byte `test/positron/test_positron.py`,
and the two `test/rh_neutrino/` modules that match no `python_files`
pattern. Every one of them looks like coverage in a directory listing and
gates nothing.

## Exit Criteria

Copied from `../../phases/phase-01-parity-corpus.md` ("Task 1.4"), plus
the two modules Task 1.3 folded in here:

- The skipped `TestScalarMediator` and `TestVectorMediator` classes
  (`skip("Needs to be updated")`), which read 90 `.npy` reference arrays
  from the eight `data/sm_*` and `data/vm_*` directories they name, are
  either regenerated-and-unskipped or deleted with their intent
  explicitly mapped to corpus coverage in this note.
- `test/positron/test_positron.py` (0 bytes) deleted or filled.
- No `@pytest.mark.skip` remains whose reason is "needs update".
- (From the phase README's Open Questions, folded in here by Task 1.3:)
  `test/rh_neutrino/integration.py` and `test/rh_neutrino/widths.py`
  match no `python_files` pattern, so the merged collection does not
  reach them. Both get a call.

## Inputs Reviewed

- `../../PLAN.md` (Goal, Scope, Numerical impact); `../README.md`
  (Findings, Numerical impact so far, Handoff).
- `../../phases/phase-01-parity-corpus.md` — Task 1.4 and the phase
  Exit Criteria.
- `../../rules.md` — rule 1 (the corpus gates every swap) and rule 2
  (corpus data only from pre-port Cython).
- `README.md` (phase working memory) — Tasks 1.1–1.3 findings, and the
  Open Question naming the two uncollected `rh_neutrino` modules.
- `test/scalar_mediator/test_scalar_mediator.py`,
  `test/vector_mediator/test_vector_mediator.py`, and their
  `generate_test_data.py` producers.
- `test/parity/cases.py` (`build_cases()`) — the 41 pinned entry points,
  the comparison target for the redundant-vs-complementary call.
- `hazma/theory/__init__.py` — the aggregation contract the deleted
  classes were the only (nominal) cover for.
- `docs/agents/lessons.md`, `docs/agents/environment.md`,
  `docs/agents/doc-consistency.md`.

## Findings

### The regenerate-vs-delete call, and the evidence that settled it

The phase file offered two outcomes. The deciding question was not
"is this data old" but "would regenerating it produce a gate worth
running". It would not.

**1. Unskipped against the current tree, 11 of the 17 tests fail** —
and the failures are structured, not noisy:

| Quantity | Direction | Interpretation |
| --- | --- | --- |
| scalar `g g`, `pi0 pi0`, `pi pi`, `s s`, `total` cross sections | stored = 2 × current | superseded identical-particle symmetry factor |
| scalar `partial_widths["e e"]` | stored = ¼ × current | same, applied twice |
| scalar `mu mu` / `e e` branching fractions | stored = ½ × current | knock-on: the total moved, the numerators did not |
| scalar `s s` spectrum | 41% | a real physics revision, not a factor |

Six pass: the two `list_final_states`, the two `spectrum_funcs` smoke
calls, `compute_vs`, and the vector `positron_lines`. Those six are the
only assertions in the pair that do not read a stored number.

**2. The suites are structurally broken independently of their data.**

- `test_scalar_mediator.py`'s `load_sm2_data` reads `sm1_dir` for all
  twelve arrays (lines 81–97). `sm_2` was never loaded; `self.sm2` was a
  duplicate of `self.sm1`. Nine tests, one parameter point.
- The vector generator's `mvs = 2 * [125.0, 550.0]` makes `vm_5` and
  `vm_6` byte-equal parameter points to `vm_3` and `vm_4`. Eight data
  directories, six distinct points, four distinct vector ones.
- Every loader docstring misdescribes its own point. All six vector ones
  say "kinetic mixing-like model with mx = 250., gvxx = 1., eps = 0.1";
  the stored params say `mx = 125.0`, and four of the six are
  `VectorMediator` with explicit quark couplings and no `eps` at all.
  The two scalar docstrings give `ms = 550` and `ms = 200` for points
  whose stored `ms` are `125` and `550`.
- The generators still in the tree disagree with the data they produced:
  both set `mx = 250.0` for the vector points, the stored params say
  `125.0`. Whatever regenerated the data last was not this file.
- `gamma_ray_lines.npy` is written by both generators and read by no
  test.

**3. Regenerating would mint a second golden corpus with worse
provenance.** `test/parity/` already pins every compiled kernel these
models call, at bit-equality, with a manifest, per-array hashes, a
kernel digest, an import-inside-the-repo guard and a sub-second
`--check`. A regenerated `.npy` set would carry none of that, at
`rtol=1e-4`, and would have to be kept in sync with the corpus through
Phases 04–06.

### What the corpus does *not* cover, and what replaced it

Mapping the deleted classes' intent onto corpus coverage, as the exit
criterion requires:

| Deleted assertion | Covered by | Status |
| --- | --- | --- |
| `annihilation_cross_sections` values | `cross_sections.{scalar,vector}.*` (11 + 5 cases) | redundant — corpus is bit-exact on far denser grids |
| `spectra` / `positron_spectra` values | `spectra.*` (14 cases), `mediator_spectra.*` (7 cases) | redundant at the kernel level |
| `partial_widths` values | — | **not covered**: `_scalar_mediator_widths.py` / `_vector_mediator_widths.py` are pure Python |
| `annihilation_branching_fractions` | — | **not covered**: `Theory.annihilation_branching_fractions` is pure Python |
| the `"total"` entries | — | **not covered**: assembled in `hazma/theory/__init__.py` |
| `positron_lines` / `gamma_ray_lines` | — | **not covered**: pure Python, and the line *energies* are closed forms |
| `compute_vs` | — | **not covered** |
| `list_annihilation_final_states` | — | **not covered** |

Everything in the "not covered" rows is pure Python above the Cython
boundary — `hazma/theory/__init__.py` plus four model modules — and the
corpus reaches none of it by construction (`cases.py` enumerates
top-level `def`s in surviving `.pyx`). That is a real gate for
Phases 04–06: a swap that repoints a kernel correctly but loses a
branching-fraction weight passes the corpus and moves every published
spectrum.

`test/test_theory_aggregation.py` covers exactly those rows, as
identities rather than stored numbers (see Decisions).

### Two defects found while doing the mapping

- **Both mediator positron kernels return `nan` at exactly
  `0.510998928`** — the legacy `MASS_E` in
  `hazma/_utils/legacy_parameters.pxd:18`, against `0.5109989461` in
  `hazma/_utils/constants.pxd:5` and `hazma/parameters.py:50`. One
  point, not a window: a 2,000,001-point sweep of
  `[0.5109988, 0.5109990]` finds that single value with `0.0` on both
  sides, and the vector kernel behaves identically. Found because the
  deleted `sm_1/e_ps.npy` grid *starts* at `0.510998928`
  (`np.geomspace(me, ...)` from when `electron_mass` was that value), so
  the stale data walked straight into it. The corpus does not pin it —
  zero `nan` across 19,610 pinned positron values. The divergence itself
  is already recorded in `../../references/cython-inventory.md` §Bugs
  item 3; this consequence was not.
- **`Theory.spectra` and `Theory.positron_spectra` reject scalar
  energies** they document as accepted, for two different reasons (a
  `len()` of a float in a channel wrapper, and the compiled
  `np.ndarray` signature). `total_spectrum` and
  `total_positron_spectrum` accept a scalar, which is why this has gone
  unnoticed.

Both are filed; neither is fixed here. Fixing either mid-Phase-01 would
be an undeclared numerical change to a surface the corpus was captured
against, which `rules.md` rules 1 and 3 forbid.

### Test-collection trap

`test/rh_neutrino/test_integration.py` — the obvious rename — collides
with `test/spectra/test_integration.py`. `test/` has no `__init__.py`,
so pytest derives both module names as `test_integration` and aborts the
**entire** run at collection with an import-file-mismatch error, not
just the one module. This is the other half of the `test_utils.py`
finding Task 1.3 recorded. Renamed to
`test_rh_neutrino_integration.py`.

## Decisions and Implementation Notes

- **Deleted, not regenerated**, on the evidence above. Both
  `generate_test_data.py` producers went with their data, following
  Task 0.3's precedent for `test/decay/`; leaving a generator for
  deleted arrays is how the next agent regenerates them by accident.
- **The replacement asserts identities, not values.**
  `test/test_theory_aggregation.py` pins the relations the aggregation
  must satisfy for any model at any energy — `total` is the channel sum,
  a branching fraction is a cross-section ratio, a spectrum is
  `bf × kernel`, a line's `bf` is its channel's — plus three two-body
  kinematic closed forms. Three consequences: it needs no data files, so
  it cannot rot the way the arrays did; it holds bit-for-bit on every
  platform, making it the one numerical gate in the repo *not* scoped to
  the capturing platform; and it does not duplicate the corpus, which
  owns the kernel values.
- **Exact equality (`assert_array_equal`, `==`) wherever it is exact**,
  and a stated tolerance only where it is not. Two places need one:
  `sum(bfs.values()) == 1` and `partial_widths["total"] ==
  sum(channels)`, both `rtol=1e-15`, because the test sums in a
  different order than the implementation accumulated (measured 1.4e-16
  for the `ms = 550 MeV` point). Everything else is bit-exact, verified.
- **Four model points, not the deleted eight**: two per model class
  straddling the mediator threshold, with `gvdd` flipped on the vector
  pair. Two of the deleted eight were duplicates and one was
  unreachable.
- `test/rh_neutrino/widths.py` is **deleted, not renamed** — a
  matplotlib plotting script under `if __name__ == "__main__"` with no
  assertions. Renaming it into the collection would import matplotlib
  (not a test dependency) to run nothing.
- `test/rh_neutrino/integration.py` is **renamed, not rewritten** — it
  passes as-is (2 tests, 15s) and asserts a real invariant: every decay
  channel whose threshold exceeds `mx` has zero width, across 37 masses
  and three lepton flavors. Renaming pulls it into the lint gate's scope,
  so it also took the mechanical fixes preflight then demanded
  (annotations, import order, two copy-pasted docstrings). Task 1.3's
  `test/spectra/integration.py` rename set that precedent.
- **The scalar-input case is deliberately absent** from the new suite,
  with a pointer to its follow-up in `positron_energies`'s docstring.
  Adding a test that documents a defect as intended behavior is worse
  than the gap.
- **Review round 1 (2026-08-08), both findings accepted.** (1) The phase
  file said "Three unrelated skips survive" while enumerating five. The
  underlying error was wider than the count: the note also attributed the
  13 skipped *tests* to 2 + 3 markers plus a `skipif`, conflating marker
  sites with skipped tests and crediting a `skipif` that does not fire
  here. Re-derived with `pytest -rs` — 5 marker sites, 13 skipped tests
  (5 + 5 + 3, the two `hazma/` markers sitting on parametrized classes),
  and `test_resolve_phase.py:47`'s `skipif` contributing 0 — and fixed at
  all four sites that stated it, not just the cited one. (2) The
  stale-state sweep used placeholders (`<the 7 changed docs>`, `<2 test
  files>`, `<9 docs>`, `<macos job>`), which defeats its purpose; every
  row now carries the literal command, with the two long argument lists
  named once as `$DOCS` / `$TESTS` above the table, and the whole block
  was re-run rather than re-typed. Re-running it moved the citation count
  27 → 29, because the sweep's own edits added citations — recorded at
  the post-edit value and re-confirmed stable.
- No ADR. Nothing about the port's architecture, interfaces or ordering
  changed.

## Files Changed

Deleted (96 files):

- `test/scalar_mediator/test_scalar_mediator.py` (234 lines, 9 skipped
  tests), `test/vector_mediator/test_vector_mediator.py` (449 lines, 8
  skipped tests).
- `test/scalar_mediator/generate_test_data.py`,
  `test/vector_mediator/generate_test_data.py` — the producers.
- `test/scalar_mediator/data/` (24 `.npy`) and
  `test/vector_mediator/data/` (66 `.npy`) — the 90 arrays the phase
  file names.
- `test/positron/test_positron.py` (0 bytes; the directory goes with it).
- `test/rh_neutrino/widths.py` (matplotlib script, no assertions).

Renamed:

- `test/rh_neutrino/integration.py` →
  `test/rh_neutrino/test_rh_neutrino_integration.py` (`git mv`, 91%
  similarity). The only content change is what the lint gates asked for
  on a now-in-scope file: the isort import reorder, two `-> None`
  annotations, and two docstrings that said "Test
  RHNeutrino.decay_widths" on both methods — including the one that
  tests `spectra` — replaced by what each actually asserts. No assertion
  changed; the file went from 7 configured-ruff findings to 0.

Added:

- `test/test_theory_aggregation.py` — 21 test functions, 69 collected
  (16 parametrized over 4 model points, 5 unparametrized), covering the
  pure-Python aggregation layer.
- `docs/followups/done/positron-spectrum-nan-at-legacy-electron-mass.md`
  (created under `todo/`; moved when cython-to-rust Task 6.3 resolved it),
  `docs/followups/todo/model-spectra-reject-scalar-energies.md`, plus
  their two rows in `docs/followups/README.md`.

Project bookkeeping:

- `../../phases/phase-01-parity-corpus.md` — frontmatter
  `status: Complete`; the Prerequisites bullet moved to past tense;
  Task 1.4's exit criteria carry their realized outcomes and the two
  folded-in `rh_neutrino` modules; the phase Exit Criteria carry the
  realized suite counts.
- `../../PLAN.md` — the Phases-table row for 01.
- `../../learnings/phase-01-parity-corpus.md` — **new**, the phase
  distillation.
- `README.md` (phase working memory) — status header, Task 1.4 row,
  Exit Criteria, Findings, Decisions, Verification, Handoff.
- This file.

Nothing under `hazma/` was touched.

## Verification

- **Full gate, the command CI runs:** `pytest -q` (bare; `testpaths =
  ["hazma", "test"]`) → **`1006 passed, 13 skipped in 582.63s`**, parity
  suite included and in exact mode.
  - Roots reconcile: `pytest --collect-only -q` → 1019 total, 67 from
    `hazma` and 952 from `test`.
  - Against Task 1.3's `935 passed, 30 skipped`: **+69** aggregation
    tests, **+2** from the `rh_neutrino` rename, **−17** skips
    (9 scalar and 8 vector) as the two legacy classes left.
    935 + 71 = 1006; 30 − 17 = 13. ✓
  - The 13 skipped **tests** are unrelated to this task's criterion, and
    come from 5 marker **sites**: `pytest -rs` attributes them 5 to
    `hazma/form_factors/vector/_eta_gamma_test.py:23`, 5 to
    `_pi_gamma_test.py:23` (both "Known to be broken", both on
    parametrized classes) and 3 to
    `test/vector_mediator/test_form_factors.py` (:137, :195, :230).
    `test/agents/test_resolve_phase.py:47` is a `skipif` that does not
    fire here — the script it guards exists — so it contributes 0.
    `rg -n "mark.skip" test hazma --type py | grep -ic "needs to be
    updated"` → `0`. ✓
- **The new suite alone:** `pytest -q test/test_theory_aggregation.py` →
  `69 passed, 2 warnings in 0.56s`. The 21 test functions by category — 4
  cross-section and branching-fraction identities (keys, total-is-the-sum,
  ratio, normalization), 1 all-channels-closed path, 1 partial-widths sum,
  7 spectrum identities (photon and positron weighting, both channel-sum
  totals, both `total_*` wrappers, closed-channel zeros), 4 line
  assertions (`bf` provenance plus three closed-form energies), 3
  final-state-list assertions, 1 vev pin.
- **The parity corpus is untouched and still self-consistent:**
  `python test/parity/generate.py --check` → `corpus OK: 41 cases / 1580
  arrays match the manifest (generated at 010747c6125d, kernel digest
  f5e6e269be47)`.
- **Test validity (negative tests).** No production change exists to
  stash, so each assertion class was proved to fire by mutating the
  implementation and re-running (mutation reverted after each; final
  `git diff origin/master -- hazma` is empty):

  | Mutation | Result |
  | --- | --- |
  | `sigmas["total"] = sum(...) * (1 + 1e-12)` | 8 failed |
  | `annihilation_branching_fractions` zero-total guard removed | 4 failed |
  | `specs[fs] = dnde_func(...)` (photon weight dropped) | 4 failed |
  | `specs[fs] = 1.0 * dnde_pos_func(...)` (positron weight dropped) | 4 failed |
  | a final state dropped from `ScalarMediator.list_annihilation_final_states` | 7 failed |
  | scalar `g g` line `bf` → `1.0` | 2 failed |
  | scalar `g g` line energy → `e_cm * 0.5000000000000001` | 1 failed |
  | scalar `e e` line `bf` → `1.0` | 2 failed |
  | scalar `e e` line energy → `cme * 0.5000000000000001` | 2 failed |
  | vector `pi0 g` line `bf` → `1.0` | 2 failed |
  | `compute_vs` → `1e-30` | 1 failed |

  Baseline between every mutation: `69 passed`. **The positron-weighting
  test exists because of this pass**: the first draft had only the
  total-is-the-sum check, which survived the "positron weight dropped"
  mutation with 65 passed / 0 failed.
- **Two mutations that did *not* fire, recorded rather than hidden:**
  rewriting the `pi0 g` line energy as `e_cm/2 - m²/(2 e_cm)` or as
  `(e_cm - m)(e_cm + m)/(2 e_cm)` is bit-identical at this parameter
  point, so the exact assertion is not sensitive to how the
  implementation groups its terms. The test's docstring says exactly
  that rather than claiming a ulp-level guarantee it cannot demonstrate.
- **The 11-of-17 failure figure** was measured by stripping the two
  `@pytest.mark.skip` decorators and running
  `pytest test/scalar_mediator/test_scalar_mediator.py
  test/vector_mediator/test_vector_mediator.py -q` → `11 failed, 6
  passed`; the files were restored with `git checkout` before deletion.
- **Environment:** built in this worktree after clearing 40 inherited
  build artifacts; 20 `.so` against `setup.py`'s 20 declared extensions;
  `python -c "import hazma; print(hazma.__file__)"` resolves inside the
  worktree.

### CI, and a gate that was not running

Watching this task's PR (#53) surfaced a defect in the CI wiring Task 1.3
landed. The first run passed all seven checks while the macOS job
reported `380 passed, 13 skipped` — the same as Linux, where a run
including `test/parity` collects ~1019. The job's env explained it:
`PARITY: --ignore=test/parity` on macOS. The expression

```yaml
PARITY: ${{ runner.os == 'macOS' && '' || '--ignore=test/parity' }}
```

cannot select its macOS branch, because Actions' `&&`/`||` return values
and `''` is falsy: `true && ''` is `''`, and `'' || '--ignore=...'` is
`'--ignore=...'`. **The corpus therefore ran on no CI entry at all from
PR #52 until this PR**, and nothing went red, because removing a gate
never does.

Fixed here by inverting the condition so the non-empty value sits on the
true branch. The re-run gives the intended split and, for the first time,
an actual observation of the phase's central claim:

| Entry | `PARITY` | Result |
| --- | --- | --- |
| macOS py3.14 | *(empty)* | `1005 passed, 14 skipped` — corpus running |
| Linux ×5 | `--ignore=test/parity` | `380 passed, 13 skipped` |

macOS's 1005/14 against 1006/13 locally is the documented budget-mode
signature: one test moves from passed to skipped, which is
`test_running_on_the_capturing_tree` standing down because the runner's
toolchain differs from the manifest, leaving the declared per-function
budgets to do the gating. The job log does not print skip reasons, so
that attribution is an inference from the arithmetic and the mechanism
rather than a quoted reason — worth re-confirming with `-rs` if it ever
matters.

The Linux findings PR #52 recorded (~70–75 blocks failing, six flipping
sign) are unaffected: they were measured before that step gained the
`PARITY` env, when the corpus still ran on every entry.

### Numerical impact

**No public value changes** (verified: `git diff origin/master -- hazma`
is empty — 0 lines). The diff touches only `test/`, `docs/followups/`
and `projects/`; no library module, signature, constant or build input
is reachable from it, so no grid evaluation applies. The two defects
recorded under Findings are pre-existing behaviors this task *measured*,
not drifts it introduced.

## Open Questions

- Whether the `nan` at the legacy `MASS_E` moves for other
  `(e_med, m_med, pw)` combinations is unswept — the mechanism is the
  shared constant, so it should not, but the fixing PR confirms rather
  than assumes. Tracked in the follow-up, not here.
- `test/vector_mediator/test_form_factors.py` carries three skip markers
  with substantive reasons ("Need to check why this form-factor seems to
  be so wrong"), and `hazma/form_factors/vector/{_eta_gamma,_pi_gamma}_test.py`
  two more marked "Known to be broken" — five marker sites, 13 skipped
  tests between them. All five are outside Task 1.4's
  criterion, which named only the "needs update" reason, and outside this
  project's scope — the form factors are pure Python and are not ported.
  Flagged so a later reader does not mistake the silence for coverage.

## Plan Impact

**Impact Level:** Update phase file.

- `../../phases/phase-01-parity-corpus.md`: frontmatter
  `status: In Progress` → `Complete`. The Prerequisites context bullet
  said "What is still open in this phase is Task 1.4" and described
  `test/positron/test_positron.py` in the present tense; both are now
  false and were re-derived. Task 1.4's exit criteria gained their
  realized outcomes (deleted rather than regenerated, with the reason),
  the two `rh_neutrino` modules Task 1.3 folded in here, and the
  basename-collision constraint on the rename — which the plan did not
  anticipate. The phase Exit Criteria's first bullet carries the realized
  counts.
- `../../PLAN.md`'s Phases table: row 01 marked Complete with a pointer
  to the learnings.

No ADR: nothing about the port's architecture, interfaces, invariants or
ordering changed. Deleting a rotted test corpus and adding an
identity-based one is a testing decision the phase file already
authorized both halves of.

## Stale-state sweep

Every row is a command as actually run — no placeholders — against
`a74f42c`, this PR's head. Two rows take the full set of docs this PR
changes; that set is named once here so the commands stay readable and
still copy-paste from the repo root (bash):

```bash
DOCS="projects/cython-to-rust/task-notes/phase-01/task-1.4-legacy-npy.md \
projects/cython-to-rust/task-notes/phase-01/README.md \
projects/cython-to-rust/task-notes/README.md \
projects/cython-to-rust/learnings/phase-01-parity-corpus.md \
projects/cython-to-rust/phases/phase-01-parity-corpus.md \
projects/cython-to-rust/PLAN.md \
docs/followups/README.md \
docs/followups/todo/model-spectra-reject-scalar-energies.md \
docs/followups/done/positron-spectrum-nan-at-legacy-electron-mass.md \
docs/agents/lessons.md"
TESTS="test/test_theory_aggregation.py test/rh_neutrino/test_rh_neutrino_integration.py"
```

| Check | Command | Result |
| --- | --- | --- |
| Exit criterion: no "needs update" skip | `rg -n "mark.skip" test hazma --type py \| grep -ic "needs to be updated"` | `0` |
| Skip inventory re-derived | `pytest -q --no-header -rs \| grep '^SKIPPED'` | 5 marker sites → 13 skipped tests: `_eta_gamma_test.py:23` [5], `_pi_gamma_test.py:23` [5], `test_form_factors.py` :137 [1] :195 [1] :230 [1] |
| Deleted paths still referenced in code | `rg -l "scalar_mediator/data\|vector_mediator/data\|generate_test_data" --type py .` | no occurrences |
| Deleted paths still referenced in docs | `rg -n "test/scalar_mediator/data\|test_scalar_mediator\|rh_neutrino/widths" .` | only dated records: the phase file's Task 1.4 spec, this note, the phase README, and the `MASS_E` follow-up's provenance sentence |
| Task 1.4 still described as open | `rg -n "Task 1.4" projects/cython-to-rust --glob '!*task-1.4*'` | every hit reads Complete / closed / realized; no "next" or "not started" survives |
| Doc citations resolve | `python scripts/agents/check_doc_citations.py $DOCS` | `docs scanned: 10` / `in-repo citations checked: 29` / `out-of-range or ambiguous: NONE` |
| Forbidden tokens in new code | `rg -n "TODO\|FIXME\|breakpoint()\|import pdb\|print(" $TESTS` | no occurrences |
| Corpus integrity unchanged | `python test/parity/generate.py --check` | `corpus OK: 41 cases / 1580 arrays match the manifest (generated at 010747c6125d, kernel digest f5e6e269be47)` |
| Suite counts as claimed | `pytest -q` | `1006 passed, 13 skipped, 5 warnings in 577.11s (0:09:37)` |
| Collection reconciles | `pytest --collect-only -q`, then the same for `hazma` and `test` | `1019` = `67` + `952` |
| **Numerical-impact statement** | `git diff origin/master -- hazma \| wc -l` | `0` — **no public value changes.** Nothing under `hazma/` is touched, so no grid evaluation applies. |
| Preflight gate | `scripts/agents/preflight.sh --paths "$TESTS" --md "$DOCS"` | black / isort / ruff / pytest / import / markdownlint / forbidden-tokens all PASS |
| CI actually runs the corpus | `gh run view 31272431024 --log --job 93140536644 \| grep -E "PARITY:\|passed, "` | `PARITY:` (empty) and `1005 passed, 14 skipped, 6 warnings in 886.77s` — before the `ci.yml` fix in this PR the same grep on job `93131021984` gave `PARITY: --ignore=test/parity` and `380 passed, 13 skipped` |

Bookkeeping consistency, checked by reading rather than by command: the
Task 1.4 row in `README.md`, this note's `**Status:**` header, the phase
file's `status: Complete` frontmatter, `../../PLAN.md`'s Phases-table
row, `../README.md`'s Phases-table row, and
`../../learnings/phase-01-parity-corpus.md`'s opening line all say the
same thing on the same date. This is not a project-closing PR, so
`preflight.sh --closing` does not apply — Phases 02–07 remain and
`PLAN.md` keeps `status: In Progress`.

## Handoff to Next Task

**Phase 01 is Complete (2026-08-08). The next task is Phase 02,
Task 2.1** (the Rust
scaffold). Read, in order:
[`../../learnings/phase-01-parity-corpus.md`](../../learnings/phase-01-parity-corpus.md)
— **not** this note or the other three, which are history — then
`../../phases/phase-02-rust-scaffold.md`, then `../../rules.md`.

**Currently safe to assume:**

- **One command is the suite.** Bare `pytest -q` → `1006 passed, 13
  skipped` on the capturing environment; `preflight.sh` with no
  `--tests` runs exactly that, and so does CI. Build editable first
  (`uv pip install -e .`) — the parity suite refuses a `hazma` resolving
  outside the repository.
- **No skipped test in the repo is waiting on this project.** The five
  survivors are form-factor issues in pure-Python code the port does not
  touch.
- **`test/` holds no golden `.npy` corpus for the mediator models any
  more.** `test/parity/data/` is the only pinned-value store, and it has
  a manifest and a `--check`. If a future task wants model-level
  reference values, that is a new decision, not a restoration.
- **`test/test_theory_aggregation.py` is the Phase 04–06 wiring gate.**
  It fires on a lost branching-fraction weight, a dropped channel, a
  detached line `bf` and a broken `total`, none of which the corpus sees.
  Run it before and after every kernel swap; it takes 0.6s.
- **Test-file basenames must be unique across `hazma/` and `test/`
  together.** A collision aborts the whole collection, and adding an
  `__init__.py` under `test/` makes it worse, not better.
- Everything Task 1.3's handoff listed still holds — the corpus is
  bit-exact on the capturing environment, `cases.py` is the single source
  of the call convention, budgets are declared acts, and the parity
  suite's cost is settled policy.

**Currently risky / unknown:**

- **Read
  [`../../../../docs/followups/done/parity-corpus-pins-ill-conditioned-points.md`](../../../../docs/followups/done/parity-corpus-pins-ill-conditioned-points.md)
  before Phase 04.** Six corpus blocks gate nothing for the port, not
  just for CI.
- Two new follow-ups ripen inside this project:
  [the `MASS_E` `nan`](../../../../docs/followups/done/positron-spectrum-nan-at-legacy-electron-mass.md)
  before Phases 05/06, and
  [the scalar-energy contract](../../../../docs/followups/todo/model-spectra-reject-scalar-energies.md)
  during 04–06, where the compiled half resolves itself if the port
  normalizes at the public boundary.
- The new suite's four model points are a sample, not a sweep. An
  aggregation bug that only manifests at, say, a resonant mediator mass
  would not be caught. Widening it is cheap (`_models()` is one list) if
  a Phase 04–06 swap ever suggests the need.
