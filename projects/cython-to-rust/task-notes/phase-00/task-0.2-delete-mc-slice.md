# Task 0.2: Delete the phase-space / gamma-ray slice

**Date:** 2026-08-06
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-00-dead-code-purge.md` (Task 0.2);
`../../PLAN.md` §Scope, §Numerical impact; `../../rules.md` Process rule 1
(verify-before-delete)
**Related ADRs:** ADR-0003 (project-scoped, Accepted 2026-08-04, with an
Addendum the same day) — this task executes its deletion
**Depends On:** Task 0.1 (constants header relocated off `_decay/`),
Task 0.5 (docs repointed off `hazma.gamma_ray` ahead of the delete)

## Objective

Delete the Monte-Carlo slice of the Cython layer — the C++ `_phase_space`
and `_gamma_ray` extensions, the `hazma.gamma_ray` module ADR-0003
removes, `hazma/deprecated/rambo.py`, and the never-built four-body RH
neutrino FSR kernel — together with the three dependents the delete would
otherwise strand. This is the last deletion in Phase 00; Task 0.4 does
the build/packaging reconciliation behind it.

## Exit Criteria

From the phase file's Task 0.2 block (the last three bullets were added
by this task — see §Plan Impact):

- Deleted: `hazma/_phase_space/`, `hazma/_gamma_ray/`,
  `hazma/deprecated/rambo.py`, `hazma/gamma_ray.py` (per ADR-0003),
  `hazma/rh_neutrino/_rh_neutrino_fsr_four_body.{pyx,pyi}`.
- The three stranded `gamma_ray` dependents go with it, each evidenced
  callerless at delete time: `hazma/rh_neutrino/_rh_neutrino_spectra.py`,
  the `electron` helper in `hazma/spectra/_photon/__init__.py`, and
  `test/test_gamma_ray.py`. Porting the five `gamma_ray_decay` call sites
  to `dnde_photon` is **not** in scope.
- `setup.py` drops the `_gamma_ray` / `_phase_space` extension groups and
  `test/conftest.py` drops its `test_gamma_ray.py` ignore, in this task.
- Importer check re-run at delete time and quoted in the PR body.
- PR body states both `major` calls explicitly (`deprecated/rambo.py` per
  `docs/versioning.md`; `gamma_ray` per ADR-0003) and notes they are
  absorbed by the project-level `version_bump: major`.

## Inputs Reviewed

- `../../PLAN.md` (all sections; §Scope and §Numerical impact bound what
  this task may change), `../README.md`, `./README.md`,
  `../../phases/phase-00-dead-code-purge.md`, `../../rules.md`.
- `../../references/cython-inventory.md` — dead-code map rows for
  `_phase_space/`, `_gamma_ray/`, `deprecated/rambo.py`,
  `rh_neutrino/_rh_neutrino_fsr_four_body.pyx`, and the audit's
  "Dead code (do not resurrect)" list.
- `../../adrs/ADR-0003-remove-gamma-ray-module.md` including its
  2026-08-04 Addendum.
- `./task-0.5-gamma-ray-decision.md` — the four `gamma_ray` loose ends it
  handed forward.
- `docs/agents/lessons.md`, `docs/agents/environment.md`,
  `docs/agents/doc-consistency.md`, `docs/agents/preflight.md`.

## Findings

- **The delete is closed under its own importers, with one exception.**
  Every live importer of a deletion target is itself a deletion target —
  `deprecated/rambo.py` → `_phase_space`, `gamma_ray.py` → `_gamma_ray`,
  `_gamma_ray/gamma_ray_fsr.pyx` → `_phase_space`,
  `_rh_neutrino_fsr_four_body.pyx` → `_gamma_ray` — except
  `hazma/rh_neutrino/_rh_neutrino_spectra.py`, the sole live importer of
  `gamma_ray_decay`. Full output in §Verification.
- **`_rh_neutrino_spectra.py` is a legacy twin with a live successor.**
  `hazma/rh_neutrino/_spectra.py` is the maintained implementation — it
  is what `_model.py` reaches through `_configure.py`, and it already
  calls `hazma.spectra.dnde_photon`, i.e. exactly ADR-0003's named
  replacement for `gamma_ray_decay`. The legacy file is reachable only
  from the commented-out `RHNeutrino` class body
  (`hazma/rh_neutrino/__init__.py:90`), and it has been broken on import
  since `hazma.rambo` was deleted, because line 24 pulls the
  broken-on-import `hazma.gamma_ray`. Same shape as the `hazma/__*.py`
  shims Task 0.3 deleted.
- **Porting its call sites was the wrong repair.** `gamma_ray_decay`
  (`particles, cme, photon_energies, msqrd`) and `dnde_photon`
  (`photon_energies, cme, final_states, *, msqrd, …`) differ in argument
  order, in FSR handling (`dnde_photon` defaults `include_fsr=True`
  where the five call sites all sit inside `spectrum_type == "decay"`
  branches), and in the three-body `msqrd` signature convention
  (`'st'` vs `'momenta'`). Rewriting them would be an unoracled physics
  change inside unreachable code — forbidden by `PLAN.md` §Scope ("no
  physics change") and unpinnable, since the module cannot run today to
  produce a baseline. That is ADR-0003's own reasoning applied one level
  down.
- **`hazma/deprecated/` becomes empty, not merely smaller.** `rambo.py`
  was its only module and the package has no `__init__.py`, so after the
  delete `import hazma.deprecated` raises `ModuleNotFoundError` rather
  than yielding an empty package. `docs/versioning.md` §6 and
  `AGENTS.md` both stated "it stays importable" as a fact about the tree;
  both now carry the rule with its scope corrected (the rule binds the
  next module parked there).
- **`git rm` + `rm -rf` is still required, and `git stash` undoes the
  staging.** The Task 0.3 finding held (an untracked `__pycache__` keeps
  the directory alive as a namespace package). New this task: a
  `git stash -u` / `git stash pop` round-trip — used to baseline the
  linters against the trunk — restores the deletions as *unstaged*, so
  `git ls-files` still lists the removed paths and
  `scripts/agents/check_doc_citations.py` then tracebacks with
  `FileNotFoundError` on an indexed-but-absent file instead of reporting
  it. `git add -A` after every stash pop.
- **Removing `electron` orphaned `numpy` in a live wrapper.**
  `hazma/spectra/_photon/__init__.py`'s `np` import existed only for that
  helper; `List` and `warn` were already unused on the trunk. All three
  went. The module has no `__all__` and no star-importer, and
  `hazma/spectra/__init__.py:127` names its imports explicitly, so
  nothing observable moved.
- **`_gamma_ray/gamma_ray_generator.pyx` never called `_photon.electron`
  anyway** — it handled electrons inline
  (`if part == "electron" or part == "neutrino"`). The helper's own
  docstring already recorded that nothing in hazma calls it.
- **The two suites still differ, for a reason unrelated to
  `collect_ignore`.** With `test/test_gamma_ray.py` gone,
  `test/conftest.py` ignores no test module at all. A bare `pytest` still
  collects a different set from `pytest test`, because `setup.cfg`'s
  `[tool:pytest] testpaths` is `hazma`. `docs/agents/environment.md` said
  the difference was the `collect_ignore`; corrected.

## Decisions and Implementation Notes

- **Three deletions beyond the phase file's list, each declared rather
  than absorbed.** `_rh_neutrino_spectra.py`, the `electron` helper, and
  `test/test_gamma_ray.py` are dependents this task's delete would have
  left importing a module that no longer exists. Task 0.3's precedent is
  the rule applied: repoint or remove the references *this* task makes
  dangle; leave what was already dead on the trunk. The phase file's
  Task 0.2 exit criteria were patched to name all three, so the widening
  is on the record and not inferred from the diff.
- **`setup.py`'s `cpp=True` branch was left for Task 0.4.** Dropping the
  two extension groups was forced here (the build fails on an
  `Extension` whose source is gone), but the now-unreachable `cpp`
  parameter and `language="c++"` branch of `make_extension` are
  `setup.py` reconciliation, which is Task 0.4's stated job. The phase
  file's Task 0.4 criterion was rewritten to say so — it previously
  claimed the `_gamma_ray` / `_phase_space` groups themselves would still
  be there for 0.4 to remove, which the same file's own Task 0.3 lesson
  had already made impossible.
- **`docs/source/rambo.rst` deleted, not rewritten**, on Task 0.5's
  orphan-page precedent — but only after checking that it is not the only
  home of live prose. It is not. `docs/source/phase_space.rst` is its
  published successor: reached from `docs/source/index.rst:11`, carrying
  the same section structure and examples, and autodoccing the live
  `hazma.phase_space::Rambo` / `ThreeBody` / `PhaseSpaceDistribution1D`.
  `rambo.rst` reaches no toctree, its `autoclass hazma.rambo::PhaseSpace`
  names a module path that has not existed for years, and the only
  content unique to it is `autofunction` for the six `hazma.rambo.*`
  functions that lived in the `hazma/deprecated/rambo.py` this task
  deletes. Nothing published is lost.
- **Historical citations pinned to `c6991a6`** rather than stripped, per
  `docs/agents/lessons.md` `[touched-doc-inherits-its-citations]`. Three
  follow-up records cited line numbers inside files this task deleted;
  each now reads "`<path>` line N **as of `c6991a6`**" with the
  `git show` command to retrieve it.
- **CHANGELOG entry landed now, under `[Unreleased]`.** The removals are
  user-facing and the replacement wording was settled by ADR-0003's
  Addendum; deferring it to Phase 07 would mean reconstructing it from
  memory, which `../README.md` explicitly forbids.

## Files Changed

### Deleted (library)

- `hazma/_gamma_ray/` — 7 files (`gamma_ray_generator.pyx`,
  `gamma_ray_fsr.pyx`, `gamma_ray_fsr.pxd`, three `.pyi`, `__init__.py`).
- `hazma/_phase_space/` — 9 files (`generator.pyx`, `histogram.pyx`,
  `modifiers.pyx`, `generator.pxd`, four `.pyi`, `__init__.py`).
- `hazma/gamma_ray.py` — ADR-0003.
- `hazma/deprecated/rambo.py` — the package's last module; the directory
  goes with it.
- `hazma/rh_neutrino/_rh_neutrino_fsr_four_body.pyx` and its stale
  `.pyi`.
- `hazma/rh_neutrino/_rh_neutrino_spectra.py` — declared scope addition.

### Edited (library)

- `hazma/spectra/_photon/__init__.py` — `electron` helper removed
  (declared scope addition), with the `numpy`, `List` and `warn` imports
  it orphaned.
- `setup.py` — `_gamma_ray` and `_phase_space` extension groups dropped.

### Tests

- `test/test_gamma_ray.py` deleted; `test/conftest.py`'s
  `old_tests_ignore` list removed with it (`collect_ignore` now holds
  only the repo's `setup.py`).
- `test/test_utils.py` — module docstring no longer says
  `hazma.deprecated.rambo` / `hazma.gamma_ray` "now call"
  `cross_section_prefactor`.

### Durable docs

- `CHANGELOG.md` — new `### Removed` block under `[Unreleased]`, naming
  both `major` removals with their replacements and stating the
  159-array bit-identity result.
- `AGENTS.md` — layout tree and the Layering list no longer name the
  deleted packages; the `hazma/deprecated/` convention keeps its rule
  with the scope corrected.
- `docs/versioning.md` — §6 same correction; the private-package example
  moved from `hazma/_phase_space/` to `hazma/spectra/_photon/`.
- `docs/PR_GUIDELINES.md` — `phase` scope row drops `hazma/_phase_space/`.
- `docs/agents/preflight.md` — import-smoke trigger list drops the two
  deleted packages.
- `docs/agents/environment.md` — the C++/`_gamma_ray` build-error entry
  and the `collect_ignore` entry both rewritten (see §Findings).
- `docs/agents/review-lenses.md` — zero-collection guard drops its
  `test_gamma_ray.py` sentence.
- `docs/source/rambo.rst` deleted (orphan Sphinx page).
- `docs/followups/done/cross-section-prefactor-threshold-cancellation.md`,
  `docs/followups/done/msqrd-driven-fsr-generator.md`,
  `docs/followups/todo/kallen-under-sqrt-remaining-call-sites.md` —
  citations into deleted files pinned to `c6991a6`; the first also had a
  wrong claim that Phase 00 deletes `hazma/_utils/kinematics.pxd`
  (Phase 06 Task 6.4 does).
- `docs/followups/todo/utils-public-surface-redundant-helpers.md` —
  half-ripe → ripe, its blocker being this deletion.

### Project files

- `projects/cython-to-rust/phases/phase-00-dead-code-purge.md` — Task 0.2
  and Task 0.4 exit criteria (see §Plan Impact).
- This note; `./README.md`; `../README.md`.

## Verification

Environment: fresh `uv venv` on CPython 3.12.12, `uv pip install -e .`
after `find hazma -name '*.c' -o -name '*.cpp' -o -name '*.so' | xargs
rm -f`. `python -c "import hazma; print(hazma.__file__)"` resolves inside
the worktree, not an installed copy.

**Verify-before-delete** (`../../rules.md` Process rule 1), re-run
against the tree at delete time over `hazma/` and `test/`. Every
`<path>:<line>` below is **as of `c6991a6`**, the pre-delete tree — two
of the files survive at different lengths, so read the block as a dated
snapshot, not as current coordinates:

```text
### hazma\._phase_space|hazma\._gamma_ray
hazma/deprecated/rambo.py:22:from hazma._phase_space import generator, histogram
hazma/deprecated/rambo.py:23:from hazma._phase_space.modifiers import apply_matrix_elem
hazma/rh_neutrino/_rh_neutrino_fsr_four_body.pyx:6:from hazma._gamma_ray.gamma_ray_fsr cimport c_gamma_ray_fsr
hazma/_gamma_ray/gamma_ray_fsr.pyx:4:from hazma._phase_space.generator cimport c_generate_space
hazma/gamma_ray.py:12:from hazma._gamma_ray.gamma_ray_generator import (

### hazma\.gamma_ray\b|from hazma import gamma_ray
hazma/rh_neutrino/_rh_neutrino_spectra.py:24:from hazma.gamma_ray import gamma_ray_decay
hazma/spectra/_fsr.py:4:``hazma.gamma_ray.gamma_ray_fsr`` (see ``docs/adrs/ADR-0001``): a
hazma/gamma_ray.py:112:        from hazma.gamma_ray import gamma_ray_decay
test/test_gamma_ray.py:1:from hazma import gamma_ray
test/conftest.py line 10:# broken-on-import hazma.gamma_ray module.
test/test_utils.py:7:present; ``hazma.deprecated.rambo`` and ``hazma.gamma_ray`` now call it),

### _rh_neutrino_fsr_four_body
hazma/rh_neutrino/__init__.py:78-80:  (all three commented out)

### _rh_neutrino_spectra
hazma/rh_neutrino/__init__.py:90:#     from ._rh_neutrino_spectra import (

### _photon import electron|_photon\.electron|spectra\.electron
  (no matches)
```

Every hit is either a file this task deletes, a comment, prose, or —
`hazma/spectra/_fsr.py:4` — a statement that *becomes* true here ("the
removed `hazma.gamma_ray.gamma_ray_fsr`"). One further hit is elided
above as a false positive: `hazma/limits/_existing_telescope.py` line 1
matched the alternation only as a prefix of `gamma_ray_parameters`, a
different, live module.

The `test/conftest.py` line above is written unpinned-style on purpose:
it is the one path in the block that still exists but has since changed
length, and `scripts/agents/check_doc_citations.py` bounds-checks a
`path:line` citation against the *current* file. The rest are skipped
only because their files are now absent — see §Open Questions.

**Build.** Clean rebuild after the deletions:

```text
built .so : 20        (25 before; the phase Exit Criteria name 20)
.pyx      : 20        (8 spectra/_photon + 2 _positron + 3 _neutrino
                       + 6 mediator + _utils/boost.pyx)
.pxd      : 17
git grep -l 'std::' -- hazma/   → no output
```

**Import smoke.** `hazma.theory`, `hazma.limits`, `hazma.cmb`,
`hazma.pbh`, `ScalarMediator`, `VectorMediator`, `RHNeutrino`,
`hazma.spectra`, `hazma.spectra._photon._muon`, `hazma.phase_space`,
`hazma.single_channel` — all import. Negative checks, all
`ModuleNotFoundError`: `hazma._phase_space`, `hazma._gamma_ray`,
`hazma.gamma_ray`, `hazma.deprecated`, `hazma.deprecated.rambo`,
`hazma.rh_neutrino._rh_neutrino_spectra`. `hasattr(hazma.spectra._photon,
"electron")` is `False`.

**Tests.**

```text
.venv/bin/python -m pytest -q test  → 244 passed, 20 skipped in 254.78s
.venv/bin/python -m pytest -q       →  57 passed, 10 skipped in 0.42s
```

No coverage was lost: `pytest test --collect-only -q` reports
**264 tests collected** both on this branch and on the stashed trunk.
`test/test_gamma_ray.py` contributed zero either way — it was in
`conftest.py`'s `collect_ignore`, and its single test class also carried
`@pytest.mark.skip(reason="Deprecated")`.

The suite covers the deletion in the only way a deletion can be covered:
the 264 collected tests exercise the surviving public API (spectra,
FSR, mediators, limits, utils) against a tree the deleted modules are
absent from, and the negative-import checks above assert the removals
themselves. No new test was written — there is no new behavior to pin,
and the numerical statement below is the regression evidence.

**Lint.** `black --check` clean on all four changed `.py`.
`ruff check --isolated --select E9,F63,F7,F82` (the CI form) — all checks
passed. Configured `ruff check` over the four changed files went
**22 → 17** findings, i.e. this change only removed findings; `isort`
was red on `hazma/spectra/_photon/__init__.py` — **identically on the
trunk** (verified by `git stash`), i.e. the known
[`preflight-isort-ruff-red-on-trunk`](../../../../docs/followups/todo/preflight-isort-ruff-red-on-trunk.md)
debt rather than anything this task introduced. Since that is the one
file under `hazma/` this task edits, its import block was sorted here and
the row is now **PASS**; the reorder is eight lines of submodule imports
in a private wrapper, `black --check` stays clean, and the 159-array
snapshot was re-run afterwards and is still bit-identical. The remaining
red row is configured `ruff`. Its redness was checked as a **set
difference**, not a count: every finding on the branch is also on the
trunk, six trunk findings were removed by this diff, and **none is new**.
What is left is 11 docstring-style findings in
`hazma/spectra/_photon/__init__.py` (`D205` ×6, `D412` ×4, `D400`) plus 5
typing findings in `setup.py` (`UP006` ×2, `UP035`, `ANN001`, `ANN201`) —
all pre-existing prose and annotation debt, out of scope for a deletion
task, and part of the same trunk redness the follow-up tracks. CI's
`--isolated` ruff form passes.
`markdownlint --dot` clean over all 16 changed `.md`.
`scripts/agents/check_doc_citations.py` over the same 16: 32 in-repo
citations, all resolved exact, `out-of-range or ambiguous: NONE`.

**Numerical-impact statement. No public value changes.** Every
compiled-backed public entry point was captured before the deletion and
again after the deletion plus a full clean rebuild, by one script run
twice: the 12 `dnde_photon_*`, 2 `dnde_positron_*` and 2
`dnde_neutrino_*` over `np.logspace(-2, 3, 200)` MeV at parent energies
{200, 1000, 5000} MeV, plus `ScalarMediator` / `VectorMediator`
`spectra()`, `positron_spectra()`, `annihilation_cross_sections()` and
`thermal_cross_section()` at mediator masses {200, 400, 900} MeV —
**159 arrays**:

```text
arrays compared: 159
arrays NOT bit-identical: 0
max relative deviation: 0.000e+00
```

Expected: everything deleted was either unbuilt, unimported, or
broken on import, and nothing surviving cimports or imports it. (Task
0.3's comparable snapshot was 171 arrays; the difference is this script's
model keyword choices and final-state sets, not coverage of a different
surface.) No line is owed to `../README.md`'s "Numerical impact so far"
beyond the no-change record.

## Open Questions

- **`scripts/agents/check_doc_citations.py` tracebacks instead of
  reporting** when a citation resolves to a path that git still tracks
  but that is absent from the working tree — the exact state a
  `git stash pop` leaves after a deletion. Recorded as a second failure
  mode on the existing
  [`docs/followups/todo/citation-checker-skips-deleted-inrepo-files.md`](../../../../docs/followups/todo/citation-checker-skips-deleted-inrepo-files.md)
  rather than as a new entry: same script, same resolution logic, same
  fix. (That follow-up also predicted this task would grow its skip
  list; it did not — every citation into a deleted file was pinned to
  `c6991a6` instead, and the sweep below reports zero unresolved.)
- **Already-dead-on-trunk references are left alone**, on Task 0.3's
  boundary rule: the ~20 `hazma.rambo` / `hazma.decay` imports in
  `notebooks/`, and the root `searchindex.js` — a committed Sphinx
  search index from a pre-2.0 build that still indexes `hazma.decay`,
  `hazma.rambo` and `hazma.gamma_ray`. None was made stale by this task.

## Plan Impact

**Impact Level:** Phase file patched (two criteria). No ADR — ADR-0003
already covers the decision this task executes, and its Addendum already
fixed the replacement wording; nothing here revises it.

1. **Task 0.2's exit criteria** now name the three declared scope
   additions and the `setup.py` / `conftest.py` edits this task was
   forced to make, and state explicitly that porting the
   `gamma_ray_decay` call sites is out of scope.
2. **Task 0.4's exit criteria** were factually wrong and are rewritten.
   They claimed the `_gamma_ray` / `_phase_space` extension groups would
   still be waiting for Task 0.4, in the same sentence that recorded why
   a deletion task cannot defer its own groups. What is actually left for
   0.4 is the survivor-count reconciliation, the packaging/sdist check,
   and the now-unreachable `cpp=True` branch of `make_extension`.

`PLAN.md` needs no edit: its §Anticipated ADRs line ("Task 0.2 still owes
the deletion itself") is the only forward-looking claim about this task,
and Step 8's working-memory update closes it.

## Stale-state sweep

| Claim | Command | Output | Verdict |
| --- | --- | --- | --- |
| Deletion targets are gone from the index and the disk | `git ls-files hazma/_gamma_ray/ hazma/_phase_space/ hazma/deprecated/ hazma/gamma_ray.py` | (empty) | OK |
| 20 extensions build | `find hazma -name '*.so' \| wc -l` | `20` | OK |
| 20 `.pyx` / 17 `.pxd` survive | `find hazma -name '*.pyx' \| wc -l`; `-name '*.pxd'` | `20`, `17` | OK |
| No C++ remains | `git grep -l 'std::' -- hazma/` | (no output) | OK |
| No live importer of a deleted path | `bash importer_check.sh` (quoted above) | every hit is a deleted file, a comment, or prose | OK |
| Test count unchanged vs trunk | `pytest test --collect-only -q` on branch, then on stashed trunk | `264` / `264` | OK |
| Suites green | `pytest -q test`; `pytest -q` | `244 passed, 20 skipped`; `57 passed, 10 skipped` | OK |
| Changed docs' citations resolve | `check_doc_citations.py $(git diff --cached --name-only --diff-filter=ACMR origin/master \| grep '\.md$')` | `docs scanned: 16`, `32 checked, resolved by exact: 32`, `out-of-range or ambiguous: NONE`; 9 skipped, every one a deliberate historical reference into a file this task or Task 0.3 deleted, each pinned to `c6991a6` in prose | OK |
| Changed docs lint | `markdownlint --dot <the same 16 .md>` | no output, scope confirmed at 16 files | OK |
| Diff size, re-derived | `git diff --cached origin/master --shortstat`; same `-- hazma test` | `43 files, +816 / −4,413`; `25 files, +6 / −4,023` | OK |
| CI-form ruff green | `ruff check --isolated --select E9,F63,F7,F82 --exclude hazma/experimental --exclude notebooks .` | `All checks passed!` | OK |
| Configured ruff adds no finding | `ruff check --output-format=concise` on the 4 changed `.py`, branch vs stashed trunk, compared with `comm` on the line-stripped sets | new on branch: **none**; removed by branch: 6 | OK (remaining red is trunk debt) |
| isort redness was pre-existing, then fixed | `isort --check-only` on the same 4, branch vs stashed trunk; then `isort` on the one touched file | red on the same single file both ways → sorted → gate PASS | OK |
| Sorting imports moved no value | `capture_surface.py` re-run after the sort, vs the same `before.npz` | `159 arrays, 0 not bit-identical` | OK |
| No stray debug statements introduced | `git diff --cached -- '*.py' \| rg '^\+.*(breakpoint\(\)\|import pdb\|^\+\s*print\()'` | (no output) | OK |
| **Numerical-impact statement** | `capture_surface.py` before/after + `compare.py` | `159 arrays, 0 not bit-identical, max rel dev 0.000e+00` | **No public value changes** |

## Handoff to Next Task

**Read first:** `../README.md`, then `./README.md`, then the phase file's
Task 0.4 block — whose exit criteria this task rewrote, so read them
fresh rather than from memory of the original plan.

**Task 0.4 is the last task in Phase 00**, and it is now smaller than the
phase file used to imply. No extension group is left to delete; what
remains is:

1. reconcile `setup.py`'s extension list against the survivor count
   (20 `.so` from a clean `pip install -e .`, verified here);
2. remove `make_extension`'s now-unreachable `cpp=True` parameter and
   `language="c++"` branch — `setup.py:18,28-35` — since no caller passes
   it;
3. confirm `pyproject.toml`'s `[tool.setuptools.package-data]` and
   `MANIFEST.in` dangle nothing (neither named the deleted packages; the
   `packages.find` include is the glob `["hazma", "hazma.*"]`, so the
   removed directories drop out automatically) and **run the sdist**,
   which this task did not — `build` is not installed in the venv;
4. write `../../learnings/phase-00-dead-code-purge.md` and flip the phase
   file's frontmatter to `status: Complete`.

**Currently safe to assume:** 20 extensions, 20 `.pyx`, 17 `.pxd`, zero
C++, zero `.pyx` outside the live surface. `test/conftest.py` skips no
test module. `hazma/deprecated/` does not exist. `hazma.gamma_ray`,
`hazma._gamma_ray`, `hazma._phase_space` and
`hazma.rh_neutrino._rh_neutrino_spectra` all raise
`ModuleNotFoundError`. The `[Unreleased]` CHANGELOG already carries the
`### Removed` block for both `major` removals — Phase 07 aggregates it,
it does not rewrite it. Clean stale `.c`/`.cpp`/`.so` before every
rebuild, `rm -rf` after `git rm -r`, and `git add -A` after any
`git stash pop`.

**Currently risky / unknown:** the sdist has not been built since the
deletions (Task 0.4 owes it). Whether any downstream user textually
imported `hazma.gamma_ray` or `hazma.deprecated.rambo` is unknowable
in-repo — that is what the project's `version_bump: major` is for.
