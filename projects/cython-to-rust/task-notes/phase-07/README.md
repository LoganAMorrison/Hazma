# Working Memory: Phase 07 — Packaging cutover and close

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Phase:** 07
**Status:** Complete (2026-08-29) — all four tasks done; the project
closed at hazma 3.0.0
**Plan References:** `../../phases/phase-07-cutover.md`
**Related ADRs:** ADR-0001
**Depends On:** Phase 06 complete

## Objective

Track live per-task status and phase-scoped findings for the maturin
cutover and project close.

## Tasks

| # | Task | Depends on | Status | Task Note |
| --- | ------ | ------------ | -------- | ----------- |
| 7.1 | Backend switch to maturin | — | **Complete (2026-08-27)** | [task-7.1-maturin-backend.md](task-7.1-maturin-backend.md) |
| 7.2 | Release pipeline (abi3 wheels) | 7.1 | **Complete (2026-08-29)** | [task-7.2-release-pipeline.md](task-7.2-release-pipeline.md) |
| 7.3 | Documentation sweep | 7.1 | **Complete (2026-08-29)** | [task-7.3-docs-sweep.md](task-7.3-docs-sweep.md) |
| 7.4 | Close the project | 7.1–7.3 | **Complete (2026-08-29)** | [task-7.4-close.md](task-7.4-close.md) |

## Exit Criteria

- All rows Complete; phase file frontmatter `status: Complete`;
  release candidate **builds and tests** from CI, with the `publish`
  job's release gate observed holding on a non-release event.
- Phase learnings at `../../learnings/phase-07-cutover.md` and the
  project retrospective at `../../learnings/project-retrospective.md`.

**All met on 2026-08-29, against the revised release clause.** The four
task rows are Complete, the phase file's frontmatter reads
`status: Complete`, and both learnings files exist. `release.yml` has
been dispatched three times and run once through its `pull_request`
trigger, producing both `cp310-abi3` wheels and the sdist and passing the
workflow's own assertions, with `publish` reporting `skipping` in all
four — the release gate holding, which is what this clause now asks for.

**The clause was revised, not merely qualified, and the first row above
records the revision.** It previously read "release candidate publishes
from CI", which no closing PR can satisfy: `publish` is gated on
`github.event_name == 'release'`, a release needs the `3.0.0` tag, and
that tag exists only after the closing PR merges. See
`../../phases/phase-07-cutover.md` §"Revision of the release clause" for
the reasoning and the residual risk — **trusted publishing under
`maturin-action` has never executed**, and the 3.0.0 release is its first
run.

## Inputs Reviewed

- `../../phases/phase-07-cutover.md`; `../README.md`; the drift table
  in `../numerical-impact.md` (moved out of `../README.md`'s "Numerical
  impact so far" section on 2026-08-21) — input to the CHANGELOG.

## Findings

### Task 7.4

- **The `[Unreleased]` section *was* the release section.** Its
  `Added` / `Removed` / `Changed` blocks were written by Tasks 0.2, 0.3
  and 0.5 and explicitly left as "the settled wording for the Phase 07
  aggregate" (`../numerical-impact.md`, Task 0.2). Closing was therefore
  a promotion of that heading plus the Phase 02–06 material, not a fresh
  section written beside it. One sentence inside it had gone stale in the
  meantime — "Wheels now ship 20 extension modules instead of 25" was
  true when Phase 00 wrote it and is not true of the release — which is
  the cost of drafting a release note twenty-six days early.
- **The declared `major` survives the re-check, and nothing numerical
  drives it.** The largest drift in the whole port is
  `scalar_mediator_decay_spectrum` at 5.3327e-12 relative, a
  `patch`-level number under `docs/versioning.md`. `major` rests entirely
  on Phase 00's two API removals, exactly as `PLAN.md` §"Numerical
  impact" predicted on day one.
- **Exactly the fourteen entry points that moved are the fourteen whose
  budget was tightened**, and the other 27 are bit-equal. Derived from
  `test/parity/tolerances.py` rather than from the prose record: 11
  budgets carry "Tightened from" and 3 `_pt` twins carry "Tightened with
  its twin", against 41 cases total. `QUAD_RTOL` and `NESTED_RTOL` have
  **zero** holders. Two entries whose rationale contains the word
  "tightened" are false positives — `spectra.photon.{charged_kaon,eta}`
  say the class is kept *rather than* tightened — so a bare
  `grep -c tighten` overcounts by two.
- **Twelve reproduced 2.1.0 defects, not eleven.** The running log's
  running count drifts (Task 5.3 calls the unconverged thermal quadrature
  "the eleventh" where Task 5.1 had already filed it as the ninth), so
  the roster is better derived from `docs/followups/` than from the
  ordinals: one from Task 3.4, seven from Phase 04, three from Phase 05,
  one from Task 6.3.
- **The aarch64 / Windows question needed a stub after all**, and that is
  a change of circumstance rather than a reversal of Task 7.2. That task
  declined to file because `PLAN.md` §Scope recorded the option — true
  while `PLAN.md` was live. Closing it makes the record archival, and
  `docs/followups/todo/` is the live backlog by construction. The
  decision is unchanged; only its location moved.

### Task 7.3

- **The phase file's enumeration found 14 of the 34 stale sites.** Its
  grep was scoped to `AGENTS.md` and `docs/agents/environment.md` (plus
  three package-data sites Task 7.2 added by hand). Run over
  `README.md`, `docs/`, `.claude/skills/` and `.codex/skills/` as well,
  the same pattern returns 20 more — every one a live instruction, not a
  historical aside. The largest population is the rebuild trigger
  ``.pyx` / `.pxd` / `rust/` / `setup.py``, copied verbatim into seven
  skill files; a claim written once and pasted is a claim that has to be
  swept by its *text*, not by the file that first stated it.
- **`hazma/_utils/` outlived its contents.** Task 6.4 deleted the four
  Cython headers and left the package's empty `__init__.py`, so the
  directory has been an importable no-op since. Nothing imports
  `hazma._utils` — every `._utils` hit in the tree is a sibling package
  (`rh_neutrino/`, `spectra/_positron/`, `spectra/_neutrino/`,
  `phase_space/`, `form_factors/vector/`). `docs/versioning.md` names it
  as the example of a non-public package, which is what made the stale
  directory look load-bearing.
- **Two of the four "editor leftovers" are not backups.**
  `A_eff/gecco.dat.bak` cites a different source than `A_eff/gecco.dat`
  ("Taken from slides from Alexander Moiseev" against "From Alexander
  Moiseev Febuary 7th, 2022") and tabulates a different grid, starting at
  0.1 MeV rather than 0.2. A `.bak` suffix is not evidence that a file is
  a copy of its neighbor.
- **The Sphinx build was never broken by the port, and it does not
  gate.** It exits 0 with 107 warnings on this tree (`sphinx-build
  9.1.0`, CPython 3.12, into an *empty* build directory — an incremental
  re-run reports 23, so the count is a recipe rather than a constant),
  none of them from a page this task touched, and there is no
  `.readthedocs.yaml` in the repository at all — the published docs are
  built by RTD's default
  detection, not by a committed config. Nothing in CI or `preflight.sh`
  builds them.

### Task 7.2

- **A wheel filename's last three fields are compressed tag *sets*, not
  tags.** Each is dot-separated and the wheel carries their cross
  product, one `Tag:` line per member. The manylinux wheel is
  `cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64` — one platform
  field standing for two tags — and the macOS wheel is one tag per field,
  so an assertion comparing field against `Tag:` line as a single string
  passes on macOS and rejects a correct Linux wheel. Both halves of a
  two-platform matrix have to run before a format assertion is believed;
  a locally built wheel is the macOS shape only.
- **`twine check` cannot fail an unbuildable sdist.** It reads metadata.
  An archive missing `rust/Cargo.toml` passes it and then fails every
  `pip install --no-binary`, and nothing else in the pipeline notices,
  because both wheel jobs build from the checkout rather than from the
  sdist.
- **The manylinux container ships its own Rust toolchain.** Phase 02's
  `CIBW_BEFORE_ALL_LINUX` rustup install, its `CIBW_ENVIRONMENT_LINUX`
  `PATH` edit, and the host `dtolnay/rust-toolchain` step that covered
  macOS all came out with cibuildwheel; `maturin-action` needs none of
  them. The phase-02 learnings entry that handed this recipe forward is
  settled in place.
- **`ci.yml` never had Cython caching to drop.** The Task 7.2 criterion's
  first clause was written in the August 2026 analysis against an
  anticipated shape. `rg -in 'cython' .github/workflows/` returns six
  hits on `origin/master`, every one prose inside a comment; no step,
  flag or cache key mentions Cython. The sole caching in the file is
  `actions/setup-python`'s `cache: pip`, which is unrelated and stays.

### Task 7.1

- **maturin fixed the wheel tag as a side effect.** setuptools-rust
  emitted `cp314-cp314` despite `py_limited_api=True` and a correct
  `_core.abi3.so` inside; maturin reads the crate's `abi3-py310` feature
  and emits `cp310-abi3`, verified by installing a 3.14-built wheel under
  CPython 3.10. Task 7.2's abi3 criterion is narrowed accordingly.
- **Five per-module twin tests read `setup.py`**, not just the one the
  Phase 06 handoff named. Their build-declaration claim is now carried
  tree-wide by `test_no_cython_remains.py`, since no `Extension` can
  exist without `setuptools` in `[build-system] requires`.
- **A bit-equality assertion against a compiled kernel can be scoped to
  the cargo profile, not just the platform.**
  `test_core_mediator_tables.py`'s grid comparison was exact against
  `numpy.logspace` under debug and one ulp off under release, at 5 of
  1000 sampled abscissae. Since `pip install -e .` built debug under
  setuptools-rust and builds release under maturin, the assertion had
  only ever been measured against the profile users do *not* get. Proved
  by running origin/master's tree against each `.so` in turn: 70 passed
  (debug) versus 7 failed, 63 passed (release), with `rust/src`
  byte-identical.
- **maturin's editable install is a release build.** The debug default
  belonged to `setuptools_rust.build_rust`'s
  `debug = self.inplace or self.debug`; only the `maturin develop` CLI,
  which this repo never invokes, defaults to debug.
- **`[tool.setuptools.package-data]` had drifted both ways:** it shipped
  `*.pyd` where it meant `.pxd`, and omitted the six
  `form_factors/vector/testdata/*.json` that the in-package tests it
  *did* ship need to run.

## Decisions and Implementation Notes

### Task 7.3

- **`requirements.txt` and the `Dockerfile` are deleted, not updated.**
  The former pinned `Cython>=0.29.12` and `flake8` and was read by
  nothing but the latter; the latter builds on
  `jupyter/scipy-notebook`, installs from that file, and enables
  `jupyter_contrib_nbextensions`, which Notebook 7 dropped. Neither has
  worked for some time and `pyproject.toml` answers both questions.
- **`setup.cfg` survives with `[aliases]` removed.** Its `[flake8]` and
  `[mypy]` sections are stale in a different way — this repo lints with
  ruff and type-checks with pyright — but that is a lint-tooling
  question, not setuptools residue, and
  `test/test_no_cython_remains.py` reads the file. Left for whoever
  settles the linter set.
- **`hazma/_utils/` deleted.** `docs/versioning.md` excludes
  leading-underscore packages from the public surface explicitly, so an
  empty one that nothing imports is removable, and removing it is what
  makes `AGENTS.md`'s layout tree true without a footnote. The two docs
  that used it as an example now name a package that still has contents.
- **The four tracked non-source files under `hazma/` are kept, and the
  question is now tracked rather than open.** Deleting a superseded
  detector response with its own cited provenance is a physics call, and
  the four cost users nothing: `[tool.maturin] exclude` keeps them out
  of both artifacts and `test/test_no_cython_remains.py` asserts it.
  Stub: [`../../../../docs/followups/todo/tracked-non-source-files-under-hazma.md`](../../../../docs/followups/todo/tracked-non-source-files-under-hazma.md).
- **Historical prose was left alone.** `docs/agents/lessons-examples.md`,
  `docs/followups/`, `docs/adrs/ADR-0002`, and every `projects/` note
  cite `.pyx` files as evidence of what happened. They are correct as
  written; a sweep that rewrites them destroys the record the ledger
  exists to keep.

### Task 7.2

- **`PyO3/maturin-action@v1` over cibuildwheel's maturin support.**
  cibuildwheel's value here was the CPython matrix and the manylinux
  container; `abi3-py310` removes the first and maturin-action supplies
  the second, so keeping it would mean narrowing `CIBW_BUILD` to one
  version to stop it rebuilding the same wheel five times.
- **No aarch64 or Windows wheels** — the decision the criterion asks for,
  default kept. The support surface is unchanged and neither target has a
  user asking for it. They are a matrix row each under maturin, which is
  why this is a deliberate no rather than an oversight; `PLAN.md`'s Scope
  already records it as a cheap follow-up, so no stub was filed.
- **The two import checks use `--no-deps`.** The claim is that one
  `cp310-abi3` wheel loads on both ends of the range it advertises, and
  `hazma/__init__.py` imports only the standard library. Installing
  numpy/scipy/matplotlib/scikit-image would let a third-party wheel gap
  on the newest CPython read as an abi3 failure; the full-dependency
  install smoke stays in `ci.yml`, on every matrix entry.
- **A path-filtered `pull_request` trigger** on `release.yml` and
  `pyproject.toml` retires the `[unrun-workflow-cannot-close-a-criterion]`
  workaround for packaging edits, at the cost of two rare paths rather
  than every PR. `rust/**` is excluded: `ci.yml` already compiles the
  crate on both operating systems for every PR. The `publish` job's
  existing `if: github.event_name == 'release'` gate is what keeps the
  new trigger from uploading.
- **One cargo cache per OS across the whole Python matrix.** abi3 links
  against the limited API, so the cargo artifacts do not vary with the
  interpreter; `workspaces: rust` points `Swatinem/rust-cache@v2` at the
  crate's own `[workspace]` root, and pip's in-tree build means both
  installs in that job reuse `rust/target`.

### Task 7.1

- `[project] version` is the source of truth; `hazma.VERSION` reads it
  back from `importlib.metadata` with no sentinel fallback. Rationale: a
  backend cannot import the package it has not built.
- `preflight.sh --closing` isolates the `[project]` table before matching
  `version`, so a `[tool.*]` `version` cannot answer for it.
- `[tool.maturin] exclude` carries only what it has to. maturin honors
  `.gitignore`, so build output (`*.so`, `*.c`, `__pycache__`) stays out
  on its own — probed, not assumed. The one class an ignore rule cannot
  reach is a **tracked** file that does not belong in a release, which is
  what the four editor leftovers under `hazma/` are. They are excluded,
  not deleted — that is a source question, handed to Task 7.3. Residual
  sharp edge: a file that is untracked *and* unignored still reaches both
  artifacts, so a release should be built from a clean tree.
- The logspace grid comparison moved to the one-ulp budget the module
  already derived, on every platform; the interp comparison keeps its
  platform split, measured unaffected by the profile.
- Two follow-ups closed (`sdist-ships-generated-c-and-docs`,
  `editable-installs-build-the-rust-extension-in-debug`), both of which
  named this task as their window. The latter's "the two profiles are
  numerically identical" risk note is corrected in place rather than
  inherited.

## Files Changed

### Task 7.4

- Release: `CHANGELOG.md` (`[Unreleased]` → `[3.0.0]`, plus the drift
  table, the behavior-change list, `Fixed` and a new `Known issues`
  section), `pyproject.toml` (`[project] version` 2.1.0 → 3.0.0)
- Project close: `../../PLAN.md` (`status: Complete`, Phase 07 row,
  the Scope bullet on aarch64/Windows), `projects/README.md` (row moved
  to Completed), `../../phases/phase-07-cutover.md` (frontmatter)
- Learnings: `../../learnings/phase-07-cutover.md` and
  `../../learnings/project-retrospective.md` (both new)
- Working memory: `../README.md` (Status, Phases row, one open question
  closed, Handoff rewritten; its two Phase 07 cross-phase entries swept
  to `../history-{findings,decisions}.md`), this file, the task note, and
  `../numerical-impact.md`
- Follow-ups: `docs/followups/todo/{consolidate-the-two-constants-tables,free-threaded-abi3t-wheels,relic-density-odes-in-rust,wheels-for-aarch64-and-windows}.md`
  (new) and their four rows in `docs/followups/README.md`

### Task 7.3

- Repo docs: `AGENTS.md`, `README.md`, `docs/source/installation.rst`,
  `docs/{versioning,workflow,PR_GUIDELINES}.md`,
  `docs/agents/{README,doc-consistency,environment,preflight,review-lenses}.md`
- Skills: `.claude/skills/{commit-and-pr,execute-single-task,review-cycle,review-plan,review-pr,review-respond,task-pipeline}/SKILL.md`;
  `.codex/skills/{commit-and-pr,execute-single-task,review-plan,review-pr,review-respond}/SKILL.md`
- Deleted: `requirements.txt`, `Dockerfile`, `hazma/_utils/__init__.py`;
  `setup.cfg`'s `[aliases]` section
- Packaging: `pyproject.toml` (the `[tool.maturin] exclude` comment only)
- Follow-ups: `docs/followups/todo/tracked-non-source-files-under-hazma.md`
  (new) and its row in `docs/followups/README.md`
- Project docs: `../../phases/phase-07-cutover.md` (Prerequisites; a
  disposition column on both Task 7.3 enumeration tables), this file and
  the task note

### Task 7.2

- Workflows: `.github/workflows/release.yml` (rewritten),
  `.github/workflows/ci.yml` (cargo cache in the `rust` and `test` jobs)
- Project docs: `../../phases/phase-07-cutover.md` (Prerequisites; Task
  7.3's enumeration extended), `../../learnings/phase-02-rust-scaffold.md`
  (two forward pointers settled), this file and the task note

### Task 7.1

- Build: `pyproject.toml`; `setup.py` and `MANIFEST.in` deleted;
  `hazma/__init__.py`; `rust/{Cargo.toml,build.rs}` (comments);
  `.github/workflows/ci.yml` (comments only, no step changed)
- Gates and tests: `scripts/agents/preflight.sh`;
  `test/test_no_cython_remains.py`; `test/test_core_mediator_tables.py`;
  `test/test_core_{neutrino,photon_rho,scalar_xs,vector_xs}.py`;
  `test/test_core_mediator_positron.py`; `test/conftest.py`
- Docs: `AGENTS.md`, `docs/{versioning,workflow,PR_GUIDELINES}.md`,
  `docs/agents/{preflight,doc-consistency,environment,review-lenses}.md`,
  `docs/followups/` (two moved to `done/`, 23 inbound refs repointed),
  the three `PLAN.md` closing paragraphs and two project
  `task-notes/README.md`

## Verification

- Clean-clone `pip install .`; sdist content check; wheel abi3-tag +
  import check on CPython 3.10 and newest; RTD/Sphinx build;
  `scripts/agents/preflight.sh --closing`.

### Task 7.2

- Both dispatched `release.yml` runs, with `publish` skipped in each;
  conclusions and the assertion output are pasted in the task note. PR #84
  then ran it a third time through the new `pull_request` trigger, where
  `Publish to PyPI` again reported `skipping` — the release gate holding
  on a real pull-request event, not only under `workflow_dispatch`.
- The two assertion scripts extracted verbatim from the workflow and run
  against locally built artifacts, then against eleven mutants — every
  failure branch fires and the compressed-manylinux shape passes.
- `ci.yml` dispatched on the branch (it does not run on a branch push):
  run 33284511292, **all eight jobs success**, both cache steps observed
  targeting `rust/target` and the five Linux matrix entries deriving one
  shared key.

### Task 7.1

- Bare `pytest -q`: **2231 passed, 15 skipped, 12 subtests passed**.
- Clean clone (`git ls-files` export, no `.git`) + `uv pip install .` on
  CPython 3.12: green, imports from outside the repo.
- `uv build`, then `uv pip install --no-binary hazma` of the sdist into a
  fresh CPython 3.10 venv: green, spectra evaluate.
- The `cp310-abi3` wheel built under CPython 3.14 imports under 3.10.
- sdist 415 → 264 files; wheel 221 → 227; wheel tag
  `cp314-cp314-macosx_26_0_arm64` → `cp310-abi3-macosx_11_0_arm64`.

## Open Questions

None outstanding — the phase and the project are closed.

- **Answered by Task 7.2 (2026-08-29): no aarch64 or Windows wheels.**
  The support surface is unchanged and neither has a user asking for it,
  and each is one matrix row whenever that changes. Task 7.4 moved the
  record from `PLAN.md`'s Scope into
  [`../../../../docs/followups/todo/wheels-for-aarch64-and-windows.md`](../../../../docs/followups/todo/wheels-for-aarch64-and-windows.md),
  because closing the plan made that Scope bullet archival. The decision
  is unchanged.
- **Answered by Task 7.3 (2026-08-29): the four tracked non-source files
  under `hazma/` stay.** `requirements.txt` and the `Dockerfile` went;
  the four did not, because two of them are a superseded detector
  response with distinct provenance rather than backups. The decision and
  its reasoning are in
  [`../../../../docs/followups/todo/tracked-non-source-files-under-hazma.md`](../../../../docs/followups/todo/tracked-non-source-files-under-hazma.md)
  so the next packaging change does not re-derive them.

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).

## Handoff to Next Task

**There is no next task.** Phase 07 is Complete, and with it the
cython-to-rust project — all eight phases, all 33 tasks, shipped as
hazma 3.0.0 on 2026-08-29.

Read [`../../learnings/project-retrospective.md`](../../learnings/project-retrospective.md)
first and
[`../../learnings/phase-07-cutover.md`](../../learnings/phase-07-cutover.md)
for the packaging contract. Those two replace this file and the four
Phase 07 task notes for every later reader
([ADR-0002](../../../../docs/adrs/ADR-0002-read-phase-learnings-not-closed-task-notes.md));
[`../README.md`](../README.md)'s Handoff routes the common follow-on
questions.

**The one thing a release manager still owes:** `publish` has never run.
It is gated on `github.event_name == 'release'`, and every observation so
far — three dispatches and one `pull_request` run — correctly skipped it.
Cutting the 3.0.0 release is the first time that job executes, so watch
it rather than assuming it (`docs/agents/lessons.md`
`[unrun-workflow-cannot-close-a-criterion]`).
