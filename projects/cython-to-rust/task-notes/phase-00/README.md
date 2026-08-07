# Working Memory: Phase 00 — Dead-code purge

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Phase:** 00
**Status:** In Progress (Tasks 0.1, 0.2, 0.3, 0.5 Complete; only 0.4
remains)
**Plan References:** `../../phases/phase-00-dead-code-purge.md`
**Related ADRs:** ADR-0003 (accepted 2026-08-04, with an Addendum the
same day; Task 0.5 executed its non-deletion steps 2026-08-05 and Task
0.2 executed the deletion 2026-08-06 — **ADR-0003 is fully discharged**)
**Depends On:** none

## Objective

Track live per-task status and phase-scoped findings for the dead-code
purge.

## Tasks

| # | Task | Depends on | Status | Task Note |
| --- | ------ | ------------ | -------- | ----------- |
| 0.1 | Relocate legacy constants header | — | Complete | [task-0.1-relocate-constants.md](task-0.1-relocate-constants.md) |
| 0.2 | Delete phase-space / gamma-ray slice | 0.1 (ADR-0003 accepted 2026-08-04) | Complete | [task-0.2-delete-mc-slice.md](task-0.2-delete-mc-slice.md) |
| 0.3 | Delete superseded kernels + helpers | 0.1 | Complete | [task-0.3-delete-superseded.md](task-0.3-delete-superseded.md) |
| 0.4 | Prune build and packaging config | 0.2, 0.3 | Not started | [task-0.4-prune-build.md](task-0.4-prune-build.md) |
| 0.5 | Execute ADR-0003 (`gamma_ray`) — ratified 2026-08-04 | — | Complete | [task-0.5-gamma-ray-decision.md](task-0.5-gamma-ray-decision.md) |

## Exit Criteria

- All rows Complete; phase file frontmatter `status: Complete`.
- Phase learnings at `../../learnings/phase-00-dead-code-purge.md`.

## Inputs Reviewed

- `../../phases/phase-00-dead-code-purge.md`; `../README.md`;
  `../../references/cython-inventory.md` (dead-code map).

## Findings

- **Build hygiene (load-bearing for every task in this project).** A
  worktree can carry stale generated `.c`/`.cpp` from another
  environment; their mtimes suppress re-cythonization and the build
  fails deep in generated code (seen in Task 0.1 as `no member named
  'subarray' in '_PyArray_Descr'` from a `.cpp` generated against an
  older NumPy). Always run
  `find hazma -name '*.c' -o -name '*.cpp' -o -name '*.so' | xargs rm -f`
  before `uv pip install -e .`. The tree builds clean on Cython 3.2.9 /
  NumPy 2.5.1 once you do.
- **`_gamma_ray/gamma_ray_generator.pyx` compiles but has never been
  importable** — `from hazma import rambo` (line 11) and
  `hazma/rambo.py` does not exist. True on unmodified `master`. It is
  nevertheless a live `Extension` in `setup.py`, so it must keep
  *compiling* until Task 0.2 removes it; exclude it from import smoke.
- **Constants-header include sites: seven, not four** (Task 0.1). The
  four live mediator ones, plus `_gamma_ray/gamma_ray_generator.pyx`
  (built → build-breaking if skipped), plus two unbuilt `_decay/*.pyx`.
  The phase file's Task 0.1 criterion was patched accordingly.
- **The mediator cross-section `.pyx` include nothing** — closes the
  project-level open question. `_c_scalar_mediator_cross_sections.pyx`
  and `_c_vector_mediator_cross_sections.pyx` carry no `include`
  directive, so Task 6.4 has no constants-header entanglement there.
- **`git rm -r` on a package leaves it importable** (Task 0.3). An
  untracked `__pycache__` keeps the directory on disk, and an empty
  directory on `sys.path` is a *namespace package* — `import
  hazma.field_theory_helper_functions` still succeeded right after the
  `git rm`. `rm -rf` the directory as well, then re-run the negative
  import check. Any verify-after-delete that only inspects the git index
  will miss this.
- **Deleting a `.pyx` forces the matching `setup.py` edit in the same
  task** (Task 0.3). `pip install -e .` fails immediately on an
  `Extension` whose source is gone, so a deletion task cannot defer its
  own extension groups to Task 0.4. Same shape as Task 0.1's fifth
  include site. `test/conftest.py` has the same property for
  `test/decay/`: its `iterdir()` is unconditional and raises at
  *collection*, taking the whole suite with it.
- **The two `common_functions` twins are algebraically identical but not
  numerically identical** (Task 0.3). `cross_section_prefactor` differs
  by ≤2.1e-7 relative within 1e-7 of threshold (`kallen_lambda`
  cancellation vs the deleted factored form) and ≤5e-15 elsewhere;
  `minkowski_dot` by ≤2.7e-14 (the C compiler contracts `a*b - c*d`
  into an FMA). Both declared under "Numerical impact so far" in
  `../README.md`.
- **`hazma/experimental/axial_vector_mediator/__init__.py` is broken on
  the trunk** — `from hazma.theory import Theory`, but `hazma.theory`
  exports `TheoryAnn` / `TheoryDec`. Pre-existing; exclude the package
  from import smoke and load `avm_msqrd.py` by path if you need it.
- ~~**Task 0.2 inherits four `gamma_ray` loose ends**~~ — **all four
  closed by Task 0.2 (2026-08-06).** For the record, and because the
  first two were resolved by deletion rather than repointing:
  1. `hazma/rh_neutrino/_rh_neutrino_spectra.py` line 24 as of
     `c6991a6` — `from hazma.gamma_ray import gamma_ray_decay`, the only
     live in-library textual importer. **Deleted**, not repointed: it is
     the legacy twin of the live `hazma/rh_neutrino/_spectra.py` (which
     already calls `hazma.spectra.dnde_photon`), it is reachable only
     from the commented-out class body at
     `hazma/rh_neutrino/__init__.py:90`, and porting its five call sites
     to `dnde_photon` would have been an unoracled physics change in dead
     code — the signatures, the FSR default, and the three-body `msqrd`
     convention all differ.
  2. `hazma/spectra/_photon/electron` — **deleted**, along with the
     `numpy`, `List` and `warn` imports it orphaned in that live wrapper.
     Confirmed callerless: not re-exported by `hazma/spectra/__init__.py`
     (which names its imports explicitly), no star-importer, and the
     compiled generator that nominally motivated it handled electrons
     inline instead.
  3. `docs/source/rambo.rst` — **deleted**, on Task 0.5's orphan-page
     precedent.
  4. `test/conftest.py` and `docs/agents/{environment,preflight,
     review-lenses}.md` — all four **swept**. `conftest.py` now ignores
     no test module at all; `environment.md`'s entry was rewritten to
     name the real remaining difference between the two suites
     (`setup.cfg`'s `testpaths`), not the retired `collect_ignore`.
- **A `git stash` round-trip un-stages a deletion** (Task 0.2). Stashing
  to baseline the linters against the trunk and popping restores the
  removals as *unstaged*, so `git ls-files` still lists the deleted
  paths — which then makes `scripts/agents/check_doc_citations.py`
  traceback with `FileNotFoundError` on a tracked-but-absent file
  instead of reporting it. `git add -A` after every pop. Recorded on
  [`docs/followups/todo/citation-checker-skips-deleted-inrepo-files.md`](../../../../docs/followups/todo/citation-checker-skips-deleted-inrepo-files.md).
- **Deleting a package's last module deletes the package** (Task 0.2).
  `hazma/deprecated/` had no `__init__.py`, so removing `rambo.py` left
  no importable `hazma.deprecated` at all. Two durable docs
  (`AGENTS.md`, `docs/versioning.md` §6) asserted "it stays importable"
  as a fact about the tree rather than as a policy about whatever lives
  there; both were rescoped rather than dropped.
- **`docs/source/` has orphan pages that no toctree reaches** — checked
  in Task 0.5: `index.rst` lists nine documents, `limits.rst` nests
  `gamma_ray_limits` + `cmb`, `models.rst` nests the two mediators.
  Everything else in `docs/source/*.rst` is unreferenced. Sphinx still
  builds an orphan into the output, so "unlinked" is not "not shipped" —
  but it does mean deleting one breaks no navigation.

## Decisions and Implementation Notes

- Task 0.1 repointed all seven `.pyx`/`.pxd` include sites but left
  `_decay/_decay_charged_pion.pyx.bak` — a `.bak` is a frozen artifact
  Cython can never compile (`cythonize()` is called only on explicit
  `Extension` objects, never a glob). Task 0.3 deletes the directory.
- Task 0.1 added a PROVENANCE paragraph to the relocated header's
  docstring (values byte-identical) so the deliberate divergence from
  `constants.pxd` is visible in-file and not "cleaned up" silently.
- Task 0.3 put `minkowski_dot` in `hazma/utils.py` (beside
  `cross_section_prefactor` and its array twin `ldot`) but deliberately
  **did not** add it to `docs/source/utils.rst` — it is a relocation of
  an existing public name, and `PLAN.md` Scope forbids API additions.
- Task 0.3 kept `boost_jac` / `boost_eng` in `_utils/boost.pyx`. They
  have zero cimporters, but unlike the three functions it deleted they
  *are* declared in `boost.pxd`, i.e. published C-level API. Whether
  they die with the rest of the header is a Task 6.4 call.
- Task 0.3 did **not** repair `cross_section_prefactor`'s threshold
  cancellation. Switching `hazma.utils` to the factored form would move
  `hazma.phase_space.PhaseSpace.cross_section`, a live public API — a
  declared numerical change that does not belong inside a dead-code
  deletion. Filed as
  [`docs/followups/done/cross-section-prefactor-threshold-cancellation.md`](../../../../docs/followups/done/cross-section-prefactor-threshold-cancellation.md)
  with the measured magnitudes.
- Task 0.3's notebook boundary: repoint the references *this task* made
  dangle; leave the ~20 `hazma.decay` / `hazma.positron_spectra`
  imports that were already dead on the trunk (those names only existed
  via the commented-out alias block in `hazma/__init__.py`).
- Task 0.5 deleted `docs/source/gamma_ray.rst` rather than rewriting it
  into a pointer page: it is in no toctree, and both functions it
  documented are already covered by the published `docs/source/
  spectra.rst`. Precedent for the phase: an orphan page whose entire
  subject is being removed goes away; it is not converted into a stub
  Sphinx cannot redirect from.
- Task 0.5 left ADR-0003's Decision body untouched and patched the
  *gate* text instead (phase file Task 0.5 criterion, `PLAN.md`
  §Anticipated ADRs, the follow-up's §Entry points). An ADR is a dated
  record with an Addendum mechanism; a forward-looking exit criterion is
  not, so that is where a superseded claim has to be fixed.
- Task 0.5 stopped at docs. The dangling `rh_neutrino` importer and the
  dead `_photon.electron` are library code — Task 0.2's shape, recorded
  in Findings rather than fixed opportunistically.
- Task 0.2 deleted three things the phase file's Task 0.2 list did not
  name (`_rh_neutrino_spectra.py`, the `electron` helper,
  `test/test_gamma_ray.py`) and patched that list to name them, rather
  than absorbing the widening into the diff. Precedent for the project:
  a dependent that this task's delete would strand goes in this task's
  scope, and the canonical exit criteria are amended in the same PR.
- Task 0.2 patched **Task 0.4's** exit criteria too. They claimed the
  `_gamma_ray` / `_phase_space` extension groups would still be waiting
  for 0.4, in the same sentence that recorded why a deletion task cannot
  defer its own groups. What is left for 0.4 is the survivor-count
  reconciliation, the sdist, and `make_extension`'s now-unreachable
  `cpp=True` branch.
- Task 0.2 landed the `### Removed` CHANGELOG block under `[Unreleased]`
  instead of deferring it to Phase 07: the replacement wording was
  already settled by ADR-0003's Addendum, and `../README.md` forbids
  reconstructing the closing entry from memory.

## Files Changed

### Task 0.1

- `hazma/_decay/parameters.pxd` → `hazma/_utils/legacy_parameters.pxd`
  (git rename; PROVENANCE note added, constant definitions untouched).
- Include repointed in seven `.pyx`: both `scalar_mediator/*_spec*`,
  both `vector_mediator/*_spec*`, `_gamma_ray/gamma_ray_generator`,
  `_decay/decay_electron`, `_decay/_decay_muon_bak`.
- `projects/cython-to-rust/phases/phase-00-dead-code-purge.md` — Task
  0.1 exit criteria corrected (four → five built sites).
- `docs/followups/done/legacy-parameters-width-exponent-bug.md` (+ index
  row in `docs/followups/README.md`). Created under `todo/` by this task;
  since resolved, so the live path is the `done/` one given here.

### Task 0.3

- Deleted `hazma/_decay/` (incl. `interpolation_data/`),
  `hazma/_positron/`, `hazma/_neutrino/`,
  `hazma/field_theory_helper_functions/`, the three `hazma/__*.py`
  shims, `hazma/spectra/_positron/_kaon.pyx`, `test/decay/`, and the
  three dead `cdef`s in `hazma/_utils/boost.pyx` (462 → 242 lines).
- `minkowski_dot` added to `hazma/utils.py`; `cross_section_prefactor`
  and `minkowski_dot` callers repointed there
  (`deprecated/rambo.py`, `gamma_ray.py`, `experimental/.../avm_msqrd.py`).
- Config: `setup.py` (3 extension groups), `test/conftest.py`,
  `pyproject.toml` + `MANIFEST.in` (`_decay` package data).
- `test/test_utils.py` added (16 pinned tests);
  `docs/followups/done/cross-section-prefactor-threshold-cancellation.md`
  filed; nine durable docs swept. Full list in the task note.

### Task 0.2

- Deleted `hazma/_gamma_ray/` (7 files), `hazma/_phase_space/` (9),
  `hazma/gamma_ray.py`, `hazma/deprecated/rambo.py` (the package's last
  module), `hazma/rh_neutrino/_rh_neutrino_fsr_four_body.{pyx,pyi}`,
  `hazma/rh_neutrino/_rh_neutrino_spectra.py`, `test/test_gamma_ray.py`,
  and `docs/source/rambo.rst`.
- `hazma/spectra/_photon/__init__.py` — dead `electron` helper and three
  orphaned imports removed. `setup.py` — two extension groups dropped.
  `test/conftest.py` — `collect_ignore` reduced to `setup.py`.
  `test/test_utils.py` — stale docstring sentence corrected.
- `CHANGELOG.md` gained a `### Removed` block under `[Unreleased]`;
  ten further durable docs swept — `AGENTS.md`, `docs/versioning.md`,
  `docs/PR_GUIDELINES.md`, three `docs/agents/*.md`, and four
  `docs/followups/` records, with every citation into a deleted file
  pinned to `c6991a6`.
- 43 files in total (+816 / −4,413); under `hazma/` and `test/` alone,
  25 files and **−4,023 lines against +6**. Full list in the task note.

### Task 0.5

- Deleted `docs/source/gamma_ray.rst` (orphan page for the two removed
  functions).
- `hazma/spectra/_photon/__init__.py` — `electron` docstring repointed
  off `hazma.gamma_ray` (docstring text only; no code).
- `docs/PR_GUIDELINES.md` — `gamma_ray.py` dropped from the `limits`
  scope row.
- `docs/followups/done/msqrd-driven-fsr-generator.md` — §Entry points no
  longer calls the removal replacement-free.
- `docs/followups/todo/utils-public-surface-redundant-helpers.md` —
  repointed off the deleted `docs/source/gamma_ray.rst:85`, which it
  cited as `minkowski_dot`'s sole public-docs reference. The file landed
  on the trunk mid-task (PR #46).
- `docs/followups/todo/preflight-isort-ruff-red-on-trunk.md` (new) +
  index row — the trunk isort/ruff debt that makes `preflight.sh` return
  `FAIL` for any file under `hazma/`.
- `docs/adrs/ADR-0001-fsr-generator-takes-both-matrix-elements.md` and
  `docs/adrs/README.md` — status Proposed → Accepted (PR #41 merged
  2026-08-05).
- `projects/cython-to-rust/phases/phase-00-dead-code-purge.md` (Task 0.5
  exit criteria) and `projects/cython-to-rust/PLAN.md` (§Anticipated
  ADRs) — replacement wording corrected per ADR-0003's Addendum.
- `projects/cython-to-rust/task-notes/phase-00/task-0.5-gamma-ray-decision.md`
  added. Full list in the task note.

## Verification

- Per task: importer re-check (`rg` the module path) quoted in PR body;
  `pip install -e .` + import smoke + full preflight.
- Task 0.1: `pytest -q test` → `52 passed, 20 skipped`; bare
  `pytest -q` → `57 passed, 10 skipped`; 32 `.so` built; public
  spectra bit-for-bit unchanged over 64 arrays. Details in the task
  note.
- Task 0.2: `pytest -q test` → `244 passed, 20 skipped`; bare `pytest -q`
  → `57 passed, 10 skipped`; **20** `.so` built (25 → 20, the count the
  phase Exit Criteria name), 20 `.pyx` / 17 `.pxd` remain,
  `git grep -l 'std::' -- hazma/` empty. No coverage lost:
  `pytest test --collect-only -q` reports 264 both on the branch and on
  the stashed trunk. The 159-array public compiled surface is
  bit-for-bit unchanged across the deletion and a clean rebuild. Details
  in the task note.
- Task 0.3: `pytest -q test` → `68 passed, 20 skipped` (52 + the 16 new
  `test/test_utils.py` cases); bare `pytest -q` → `57 passed, 10
  skipped`; **25** `.so` built (32 → 25; Task 0.2's five take it to the
  20 the phase Exit Criteria name); 26 `.pyx` / 19 `.pxd` remain; the
  171-array public compiled surface bit-for-bit unchanged. Details in
  the task note.
- Task 0.5: docs-only, so no build and no test-count change — the diff's
  single file under `hazma/` is a docstring hunk in
  `hazma/spectra/_photon/__init__.py`. Checks run instead: no toctree
  reaches the deleted page; no reference dangles after the delete;
  `_photon.electron` confirmed callerless; PR #41's merge date pinned
  from `git log`. Commands and output in the task note.

## Open Questions

- ~~ADR-0003 sign-off — required before Task 0.2 deletes anything~~ —
  **fully discharged.** Accepted 2026-08-04; non-deletion steps executed
  in Task 0.5 on 2026-08-05; the deletion itself landed in Task 0.2 on
  2026-08-06, with the `### Removed` CHANGELOG block. `gamma_ray_fsr` is
  **not** replacement-free: it is superseded by
  `hazma.spectra.dnde_photon_fsr`, which closed
  [`docs/followups/done/msqrd-driven-fsr-generator.md`](../../../../docs/followups/done/msqrd-driven-fsr-generator.md)
  in PR #41 — see ADR-0003's Addendum (2026-08-04).
- ~~`WIDTH_K` / `WIDTH_PI` in the legacy tables are written with `**`
  where a decimal exponent was meant (no consumer today)~~ — **closed
  2026-08-05.** Task 0.3 left `legacy_parameters.pxd` as the only copy of
  the bad literals; both names have since been deleted from it outright,
  `constants.pxd` being the canonical PDG-cited source. No published
  value moves. See
  [`docs/followups/done/legacy-parameters-width-exponent-bug.md`](../../../../docs/followups/done/legacy-parameters-width-exponent-bug.md).
- **`preflight.sh` cannot return zero for a file under `hazma/`**
  (found in Task 0.5): gates 2 (`isort --check-only`) and 3 (configured
  `ruff check`) fail on unmodified trunk code, while CI enforces only
  `black` plus `ruff --isolated --select E9,F63,F7,F82`. Every later
  task in this phase will inherit the same two red rows and must prove
  they are pre-existing — `git stash` the change and re-run both
  commands. Filed as
  [`docs/followups/todo/preflight-isort-ruff-red-on-trunk.md`](../../../../docs/followups/todo/preflight-isort-ruff-red-on-trunk.md).
  Task 0.2 confirmed it and added the recipe's missing step: `git add -A`
  after the `git stash pop`, or the deletions come back unstaged. Its own
  delta was `isort` red on the same one file both ways, and configured
  `ruff` **down** 22 → 17 findings. It also narrowed the debt by one
  file: `hazma/spectra/_photon/__init__.py` was the only file under
  `hazma/` it edited, so its import block was sorted in-task and that
  gate now passes. The `ruff` row stays red — those findings are `UP006`
  / `ANN001` in `setup.py` and the test files, out of any single task's
  scope.
- ~~`cross_section_prefactor`'s threshold cancellation — deferred;
  sequencing matters, because if Phase 01 captures the corpus first the
  Rust port inherits the cancelling values~~ — **closed: repaired
  out-of-band before Phase 01 started**, via `two_body_momentum`'s
  factored form. See
  [`docs/followups/done/cross-section-prefactor-threshold-cancellation.md`](../../../../docs/followups/done/cross-section-prefactor-threshold-cancellation.md)
  and "Out-of-band" under "Numerical impact so far" in `../README.md`.
  The corpus will pin the post-fix values.

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).
Tasks 0.1, 0.2, 0.3, and 0.5 each patched canonical files; those patches
are recorded in their own task notes' Plan Impact sections (0.5 also
patched `../../PLAN.md`; 0.2 rewrote **Task 0.4's** exit criteria as well
as its own).

## Handoff to Next Task

**For the next agent working in Phase 00:** read `../../PLAN.md`, then
`../README.md`, then this file, then the phase file. **Tasks 0.1, 0.2,
0.3 and 0.5 are done — Task 0.4 is all that remains, and it closes the
phase.** No sign-off is outstanding anywhere in this phase; all three
project ADRs are Accepted and ADR-0003 is fully discharged.

**Every deletion in this phase has landed.** Task 0.4 is build and
packaging reconciliation only, and it is smaller than the original phase
text implied — Task 0.2 rewrote its exit criteria, so read them from the
phase file rather than from memory. What is left:

1. reconcile `setup.py`'s extension list against the survivor count
   (20 `.so` from a clean `pip install -e .`);
2. delete `make_extension`'s now-unreachable `cpp=True` parameter and
   `language="c++"` branch (`setup.py:18,28-35`) — no caller passes it;
3. confirm `pyproject.toml` package-data and `MANIFEST.in` dangle
   nothing, and **run the sdist** — Task 0.2 did not, `build` was not in
   the venv;
4. write `../../learnings/phase-00-dead-code-purge.md` and flip the phase
   file frontmatter to `status: Complete`.

**Currently safe to assume:** the tree carries **20 extensions, 20
`.pyx`, 17 `.pxd`**, zero C++ (`git grep -l 'std::' -- hazma/` is empty),
and no `.pyx` outside the live surface — re-derive with
`find hazma -name '*.so' | wc -l` rather than quoting this.
`test/conftest.py` skips no test module at all; `collect_ignore` holds
only the repo's `setup.py`. `hazma/deprecated/` no longer exists.
`hazma.gamma_ray`, `hazma._gamma_ray`, `hazma._phase_space` and
`hazma.rh_neutrino._rh_neutrino_spectra` all raise `ModuleNotFoundError`.
`hazma.utils` is the single home for `cross_section_prefactor` and
`minkowski_dot`. `CHANGELOG.md`'s `[Unreleased]` already carries the
`### Removed` block for both `major` removals with their replacement
wording — Phase 07 aggregates it, it does not rewrite it. Clean stale
`.c`/`.cpp`/`.so` before every rebuild, `rm -rf` a package directory
after `git rm -r`, and `git add -A` after any `git stash pop`
(see Findings).

**Currently risky / unknown:** the sdist has not been built since the
deletions — Task 0.4 owes it. And whether external user code imported
the removed public names (`hazma.gamma_ray`, `hazma.deprecated.rambo`,
the double-underscore legacy shims) is unknowable in-repo: the
verify-before-delete checks found no in-repo importer that was not itself
being deleted, but they cannot see downstream users. That is exactly what
the project's `version_bump: major` is for.
