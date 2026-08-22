---
phase: 07
title: Packaging cutover and close
status: Not started
---

# Phase 07: Packaging cutover and close

## Goal

Remove the setuptools/Cython toolchain, make maturin the build backend,
modernize the release pipeline to abi3 wheels, sweep the docs, and
close the project (version bump + CHANGELOG per `PLAN.md`).

## Prerequisites

- Phase 06 complete (zero Cython).
- Current packaging facts (verify still true at execution):
  build-system requires `numpy`, `cython`, `setuptools`, `scipy>=1.13`;
  version is dynamic via `attr: hazma.VERSION`; pytest config moved to
  pyproject in Phase 01; release.yml uses cibuildwheel
  (cp310–cp314 × {linux x86_64, macos arm64} = 10 wheels, sdist job,
  PyPI trusted publishing).

## Tasks

### Task 7.1: Backend switch to maturin

**Task note:** [`../task-notes/phase-07/task-7.1-maturin-backend.md`](../task-notes/phase-07/task-7.1-maturin-backend.md)
**Depends on:** —

**Exit criteria:**

- `[build-system] requires = ["maturin>=1.x"]`; `[tool.maturin]`
  mixed-layout config (python packages + `rust/Cargo.toml`,
  `module-name = "hazma._core"`); setuptools-rust, `setup.py`, and the
  cython/scipy/numpy build requirements deleted.
- Version becomes static in `[project]`; `hazma.VERSION` reads
  `importlib.metadata.version` (attribute preserved — it is public
  API); `scripts/agents/preflight.sh --closing` version check updated
  to the new source of truth.
- `uv pip install -e .` and plain `pip install .` build from a clean
  clone; import smoke green; package-data globs pruned to what pure
  Python still reads (positron/neutrino CSVs etc.); `MANIFEST.in`
  deleted (maturin sdist includes `rust/` + data via its own config —
  verify sdist contents explicitly).

### Task 7.2: Release pipeline

**Task note:** [`../task-notes/phase-07/task-7.2-release-pipeline.md`](../task-notes/phase-07/task-7.2-release-pipeline.md)
**Depends on:** Task 7.1

**Exit criteria:**

- release.yml rebuilt on maturin (PyO3/maturin-action or cibuildwheel's
  maturin support — pick and record): **2 abi3 wheels** (manylinux
  x86_64, macOS arm64) + sdist; trusted-publishing job preserved;
  wheel abi3 tags and importability verified in the workflow
  (`CIBW_TEST_COMMAND`-equivalent import check on the oldest supported
  CPython, 3.10, and the newest).
- Decision recorded (one line in task note): whether to add
  linux aarch64 / Windows wheels now that they are cheap — default no,
  matching current support surface.
- ci.yml: drop per-version rebuild caching of Cython; add
  cargo caching; matrix unchanged.

### Task 7.3: Documentation sweep

**Task note:** [`../task-notes/phase-07/task-7.3-docs-sweep.md`](../task-notes/phase-07/task-7.3-docs-sweep.md)
**Depends on:** Task 7.1

**Exit criteria:**

- `AGENTS.md` rewritten where it states Cython facts (layout tree,
  "Editing a .pyx requires a rebuild", layering §1, commands);
  `docs/agents/` env notes updated (Rust toolchain requirement, uv +
  editable-rebuild loop); `CLAUDE.md`/skills references checked by the
  doc-consistency checklist.
- README / docs/source install instructions updated (Rust toolchain
  only needed for source builds; wheels cover normal installs);
  Sphinx/RTD build verified against the maturin-built package.
- Stale artifacts removed: `requirements.txt`, `Dockerfile` (both
  contradict pyproject today) — or updated if kept deliberately.

### Task 7.4: Close the project

**Task note:** [`../task-notes/phase-07/task-7.4-close.md`](../task-notes/phase-07/task-7.4-close.md)
**Depends on:** Tasks 7.1–7.3

**Exit criteria:**

- `CHANGELOG.md` entry: the migration summary + the aggregated
  numerical-drift table from `../task-notes/numerical-impact.md`
  (per-function max shifts), naming this project slug.
- `VERSION` bumped per `PLAN.md` `version_bump` (re-check the level
  against actual recorded drift and the Task 0.5 outcome);
  `scripts/agents/preflight.sh --closing` green.
- Project retrospective written to
  `../learnings/project-retrospective.md` incl. §5 follow-on seeds
  (candidates surfaced so far: constants-table consolidation as a
  declared numerical change; free-threaded abi3t wheels when the
  ecosystem settles; relic-density ODEs to Rust if ever justified —
  each gets a `docs/followups/todo/` stub if still relevant at close).
- `PLAN.md` `status: Complete`; `projects/README.md` row moved to
  Completed.

## Exit Criteria

- All tasks complete; a release candidate builds, tests, and publishes
  from CI; no Cython, setuptools, or cibuildwheel-version-matrix
  residue anywhere.
- Phase learnings written to `../learnings/phase-07-cutover.md`.
