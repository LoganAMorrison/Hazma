---
phase: 07
title: Packaging cutover and close
status: Complete
---

# Phase 07: Packaging cutover and close

## Goal

Remove the setuptools/Cython toolchain, make maturin the build backend,
modernize the release pipeline to abi3 wheels, sweep the docs, and
close the project (version bump + CHANGELOG per `PLAN.md`).

## Prerequisites

- Phase 06 complete (zero Cython).
- Current packaging facts, as of Task 7.1 (2026-08-27): build-system
  requires `maturin` alone, and `[tool.maturin]` carries the mixed
  layout; `setup.py` and `MANIFEST.in` are deleted; the version is
  static in `[project] version`, with `hazma.VERSION` reading it back
  from `importlib.metadata`; pytest config moved to pyproject in
  Phase 01.
- Release pipeline, as of Task 7.2 (2026-08-29): release.yml is built on
  `PyO3/maturin-action@v1` and produces one `cp310-abi3` wheel per
  platform plus the sdist, replacing the cibuildwheel matrix
  (cp310–cp314 × {linux x86_64, macos arm64} = 10 wheels) it carried
  until then. The trusted-publishing job is unchanged and still gated on
  `github.event_name == 'release'`; the workflow also runs on pull
  requests that touch `release.yml` or `pyproject.toml`.
- Documentation, as of Task 7.3 (2026-08-29): no live instruction doc
  states a Cython fact any more. What survives
  `rg -n 'Cython|\.pyx|\.pxd|cythoniz'` over `AGENTS.md`, `README.md`,
  `docs/` and the two skill trees is project-slug citations and
  explicitly historical sentences, plus the deliberately historical
  `docs/agents/lessons-examples.md` and `docs/followups/`.
  `requirements.txt`, `Dockerfile`, `setup.cfg`'s `[aliases]` and the
  vestigial `hazma/_utils/` package are deleted.

## Tasks

### Task 7.1: Backend switch to maturin

**Task note:** [`../task-notes/phase-07/task-7.1-maturin-backend.md`](../task-notes/phase-07/task-7.1-maturin-backend.md)
**Depends on:** —

**Exit criteria:**

- `[build-system] requires = ["maturin>=1.x"]`; `[tool.maturin]`
  mixed-layout config (python packages + `rust/Cargo.toml`,
  `module-name = "hazma._core"`); setuptools-rust and `setup.py`
  deleted.

  **Amended by Task 6.4 (2026-08-27):** this bullet also owed "the
  cython/scipy/numpy build requirements deleted". Task 6.4 deleted the
  last `.pyx`, which left those three declared but read by nothing, so
  it removed them in the same pass rather than shipping a false
  requirement. `setup.py` survives that task stripped to the single
  `RustExtension`; deleting the file is still this task's.
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

  **Narrowed by Task 7.1 (2026-08-27):** the abi3 tag itself is no longer
  outstanding. maturin reads the crate's `abi3-py310` feature and already
  emits `hazma-2.1.0-cp310-abi3-<platform>.whl`, verified cross-version
  (built under CPython 3.14, installed and imported under 3.10). What this
  task owes is producing the *two platform* wheels in CI and verifying the
  tag and the import there — not making the tag correct. Under
  setuptools-rust the wheel was mistagged `cp314-cp314` despite carrying a
  correct `_core.abi3.so`, which is the failure this criterion was written
  against.
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

  **Narrowed by Task 7.1:** the *build-mechanism* half of this sweep is
  done — the version's source of truth, the backend, and the sdist/wheel
  machinery are corrected in `AGENTS.md`, `docs/versioning.md`,
  `docs/workflow.md`, `docs/agents/{preflight,doc-consistency,environment,review-lenses}.md`
  and `docs/PR_GUIDELINES.md`, because Task 7.1's own diff falsified them.

  What remains is the Cython-fact sweep proper. Enumerated rather than
  described, from `rg -n 'Cython|\.pyx|\.pxd|cythoniz'` over both files
  after Task 7.1 (line numbers are that task's; re-derive before
  editing):

  | File | Line | What is wrong | Task 7.3 |
  | --- | --- | --- | --- |
  | `AGENTS.md` | 20 | "historically Cython, currently mid-migration" — true only until this project closes; Task 7.4 is the last moment it can stay | rewritten |
  | `AGENTS.md` | 50 | layout tree: `_utils/  # Cython helpers (boost.pyx, constants.pxd, …)` — the directory holds no such file | row deleted with the package |
  | `AGENTS.md` | 68 | layering §1 names "the `.pyx` under `_utils/`, `spectra/`, …" | rewritten |
  | `AGENTS.md` | 90 | commands block: `pip install -e .  # build the Cython + Rust extensions in place` | rewritten |
  | `AGENTS.md` | 120–134 | the whole "Editing a `.pyx` or `.pxd` requires a rebuild" paragraph, including its closing "Confirm the same way as for Cython" | folded into the `.rs` paragraph |
  | `AGENTS.md` | 171 | "Never commit generated C/C++. `setup.py` cythonizes on build" — names a file Task 7.1 deleted | bullet deleted |
  | `environment.md` | 38–43 | "Editing a `.pyx` / `.pxd` and re-running pytest tests the OLD kernel", whose second sentence still says `setup.py` compiles them | deleted |
  | `environment.md` | 51–55 | "`pip install -e .` needs Cython, NumPy, and a C compiler" — false since Task 6.4, and now sits directly above the corrected `maturin`/cargo paragraph | folded into the cargo paragraph |
  | `environment.md` | 67 | "exactly like a `.pyx`" inside the `.rs` rebuild note | rewritten |
  | `environment.md` | 77–84 | "Deleting a `.pyx` does not make its module unimportable" — the stale-`.so` note; its *mechanism* is still true of `.rs`, so rewrite rather than delete | rewritten for `.rs` |
  | `environment.md` | 106–107 | "Never hand-edit generated `.c` / `.cpp`. They are cythonize output" | deleted |

  **Extended by Task 7.2 (2026-08-29):** three more sites, in the skills
  rather than in those two files, and invisible to the grep above because
  they name setuptools instead of Cython. Each tells a reviewer to check
  new package data against a table `pyproject.toml` no longer has:

  | File | Line | What is wrong | Task 7.3 |
  | --- | --- | --- | --- |
  | `.claude/skills/review-plan/SKILL.md` | 217–219 | "New `*.dat` / `*.csv` under `hazma/` must be registered in `[tool.setuptools.package-data]`" — maturin ships the whole `hazma/` tree, so there is nothing to register | rewritten |
  | `.claude/skills/review-pr/SKILL.md` | 107–109 | the same claim, as a review finding to look for | bullet deleted |
  | `.codex/skills/review-pr/SKILL.md` | 30 | "package data registration" in the lens summary — the pointer to the claim above | pointer deleted |

  Derived from `rg -n 'package-data|package data' .claude/ .codex/`; the
  `.codex/` copies carry only that one pointer, not the claim itself.

- README / docs/source install instructions updated (Rust toolchain
  only needed for source builds; wheels cover normal installs);
  Sphinx/RTD build verified against the maturin-built package.
- Stale artifacts removed: `requirements.txt`, `Dockerfile` (both
  contradict pyproject today), `docs/source/installation.rst`'s
  `python setup.py install` line, and `setup.cfg`'s `[aliases]` section
  (a setuptools-only feature, inert since Task 7.1) — or updated if kept
  deliberately. Also decide whether the four tracked editor leftovers
  under `hazma/` (`{A_eff,energy_res}/gecco.dat.bak`, `_gev.py.bak`,
  `form_factors/notes.org`) should be deleted; Task 7.1 excluded them
  from the distribution but left the files alone.

### Task 7.4: Close the project

**Task note:** [`../task-notes/phase-07/task-7.4-close.md`](../task-notes/phase-07/task-7.4-close.md)
**Depends on:** Tasks 7.1–7.3

**Exit criteria:**

- `CHANGELOG.md` entry: the migration summary + the aggregated
  numerical-drift table from `../task-notes/numerical-impact.md`
  (per-function max shifts), naming this project slug.
- `[project] version` in `pyproject.toml` bumped per `PLAN.md`
  `version_bump` (re-check the level against actual recorded drift and
  the Task 0.5 outcome; Task 7.1 moved the source of truth off
  `hazma/__init__.py`, and `preflight.sh --closing` reads the new one);
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

- All tasks complete; no Cython, setuptools, or
  cibuildwheel-version-matrix residue anywhere.
- A release candidate **builds and tests from CI**: `release.yml`
  produces both `cp310-abi3` wheels and the sdist and passes that
  workflow's own wheel-tag, sole-extension and cross-version import
  assertions, and its `publish` job's release gate is observed holding on
  a non-release event.
- Phase learnings written to `../learnings/phase-07-cutover.md`.

### Revision of the release clause (Task 7.4, 2026-08-29)

**The second bullet previously read "a release candidate builds, tests,
and *publishes* from CI." It was unsatisfiable, and circularly so.**
`release.yml`'s `publish` job is gated `if: github.event_name ==
'release'`; a GitHub release needs the `3.0.0` tag; that tag exists only
once the closing PR merges — and the closing PR is what this criterion
gates. Holding closure until an upload is observed is therefore a
deadlock rather than a stricter gate: the version bump could never land,
so the release could never be cut, so `publish` could never run.

The clause is narrowed to what closure can actually attest, and **the
upload is reassigned rather than dropped** — it is now an explicit
release-manager handoff, recorded in `../task-notes/phase-07/README.md`'s
Handoff and in the closing PR. This is a narrowing of a gate, so the
residual risk is stated plainly rather than buried: **trusted publishing
under `PyO3/maturin-action` has never executed.** The 3.0.0 release is
its first run, and it should be watched rather than assumed.

**Met on 2026-08-29 (Task 7.4), as revised.** All four task rows are
Complete and the frontmatter reads `status: Complete`. The residue clause
is asserted rather than inspected: `test/test_no_cython_remains.py` fails
on any Cython source, any setuptools build script, and any Cython entry
in the build requirements, and `release.yml` carries no version matrix.
The build-and-test clause is met by four observed `release.yml` runs —
three `workflow_dispatch` and one `pull_request` (the closing PR's own,
triggered by its `pyproject.toml` edit) — each producing both wheels and
the sdist, passing the workflow's assertions, and reporting `publish` as
`skipping`, which is the release gate holding.

The original phrasing was an instance of `docs/agents/lessons.md`
`[unrun-workflow-cannot-close-a-criterion]`, and this revision extends
that class: when dispatching cannot reach the job because the criterion
is structurally unsatisfiable, the fix is to revise the criterion and
reassign the observation — not to qualify a "Met" that is not met.
