# Working Memory: Phase 07 — Packaging cutover and close

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Phase:** 07
**Status:** In progress (Task 7.1 complete 2026-08-27)
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
| 7.2 | Release pipeline (abi3 wheels) | 7.1 | Not started | [task-7.2-release-pipeline.md](task-7.2-release-pipeline.md) |
| 7.3 | Documentation sweep | 7.1 | Not started | [task-7.3-docs-sweep.md](task-7.3-docs-sweep.md) |
| 7.4 | Close the project | 7.1–7.3 | Not started | [task-7.4-close.md](task-7.4-close.md) |

## Exit Criteria

- All rows Complete; phase file frontmatter `status: Complete`;
  release candidate publishes from CI.
- Phase learnings at `../../learnings/phase-07-cutover.md` and the
  project retrospective at `../../learnings/project-retrospective.md`.

## Inputs Reviewed

- `../../phases/phase-07-cutover.md`; `../README.md`; the drift table
  in `../numerical-impact.md` (moved out of `../README.md`'s "Numerical
  impact so far" section on 2026-08-21) — input to the CHANGELOG.

## Findings

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

- Add aarch64/Windows wheels now that they are cheap? (Task 7.2
  records the call; default no.)
- Should the four tracked editor leftovers under `hazma/` be deleted from
  the repository, not just excluded from the distribution? Handed to Task
  7.3 alongside `requirements.txt` and the `Dockerfile`.

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).

## Handoff to Next Task

**Tasks 7.2 (release pipeline) and 7.3 (docs sweep) are both unblocked
and share no files.** Read `../../PLAN.md`, `../README.md`, this file,
then the phase file — whose Prerequisites block and whose 7.2/7.3 exit
criteria Task 7.1 rewrote against the post-cutover tree, so they no
longer need re-verifying by hand.

**Currently safe to assume:**

- **maturin is the whole build.** `[build-system] requires` is
  `["maturin>=1.5,<2.0"]`; `[tool.maturin]` carries `python-source = "."`,
  `manifest-path`, `module-name`, `exclude`, `include`. No `setup.py`, no
  `MANIFEST.in`, no `setuptools`. `test/test_no_cython_remains.py` asserts
  it, and nothing else in `test/` reads a build script any more.
- **The wheel is already `cp310-abi3` and portable across CPythons.**
  Task 7.2 owes CI verification of the tag and the two platform wheels,
  not the tag itself.
- **The sdist is 264 files** — `hazma/` + `rust/` + pyproject +
  README/LICENSE/CHANGELOG — and source-installs into a fresh CPython
  3.10 venv. maturin honors `.gitignore` for both artifacts, so build
  output stays out; an untracked *unignored* file does not.
- **The version lives in `pyproject.toml`'s `[project] version`.** Task
  7.4's bump edits that line; `preflight.sh --closing` already reads it.
  Its tooling tendrils were swept: `docs/versioning.md`,
  `docs/workflow.md`, `docs/agents/{preflight,doc-consistency}.md`, and
  the three project `PLAN.md` closing paragraphs.
- **`pip install -e .` builds release now**, so a `rules.md` rule 12
  benchmark from an editable tree is sound again.

**Currently risky / unknown:**

- **`preflight.sh --closing` is vacuous on a branch cut before Task 7.1
  merged**: `origin/master`'s `pyproject.toml` has no `[project] version`
  to compare against, so the gate reports "bump unverifiable" — a FAIL,
  not a false pass. It resolves once 7.1 is on master, which is before
  7.4 runs.
- **An exact assertion against a compiled kernel may be scoped to the
  cargo profile.** Task 7.1 found and fixed one; the suite is green, but
  a newly written bit-equality claim should be checked against both
  profiles before being trusted.
- **`release.yml` still has no pull-request trigger**, so Task 7.2 cannot
  measure its own rewrite without an explicit dispatch
  (`../../../../docs/agents/lessons.md`
  `[unrun-workflow-cannot-close-a-criterion]`).
