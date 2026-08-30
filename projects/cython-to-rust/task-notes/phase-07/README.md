# Working Memory: Phase 07 — Packaging cutover and close

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Phase:** 07
**Status:** In progress (Tasks 7.1–7.2 complete; 7.2 on 2026-08-29)
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
  conclusions and the assertion output are pasted in the task note.
- The two assertion scripts extracted verbatim from the workflow and run
  against locally built artifacts, then against eleven mutants — every
  failure branch fires and the compressed-manylinux shape passes.

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

- **Answered by Task 7.2 (2026-08-29): no aarch64 or Windows wheels.**
  The support surface is unchanged and neither has a user asking for it;
  `PLAN.md`'s Scope keeps them as a cheap follow-up, and each is one
  matrix row whenever that changes.
- Should the four tracked editor leftovers under `hazma/` be deleted from
  the repository, not just excluded from the distribution? Handed to Task
  7.3 alongside `requirements.txt` and the `Dockerfile`.

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).

## Handoff to Next Task

**Task 7.3 (documentation sweep) is the only unblocked task**; 7.4 waits
on it. Read `../../PLAN.md`, `../README.md`, this file, then the phase
file — whose Prerequisites block carries both the Task 7.1 packaging
facts and the Task 7.2 release-pipeline facts, and whose 7.3 exit
criteria enumerate the remaining stale sites rather than describing them.
Re-derive the line numbers in that enumeration before editing.

**Currently safe to assume:**

- **maturin is the whole build.** `[build-system] requires` is
  `["maturin>=1.5,<2.0"]`; `[tool.maturin]` carries `python-source = "."`,
  `manifest-path`, `module-name`, `exclude`, `include`. No `setup.py`, no
  `MANIFEST.in`, no `setuptools`. `test/test_no_cython_remains.py` asserts
  it, and nothing else in `test/` reads a build script any more.
- **The release pipeline is maturin's too, and it has been observed to
  run.** `release.yml` builds one `cp310-abi3` wheel per platform plus the
  sdist on `PyO3/maturin-action@v1`; the tag, the sole-`.abi3.so` claim,
  the 3.10/3.14 imports and the sdist's build inputs are all asserted in
  the workflow. Task 7.2's criteria are closed against a dispatched run,
  not against the file.
- **`release.yml` now has a `pull_request` trigger**, filtered to
  `release.yml` and `pyproject.toml`. An edit to either is measured by an
  ordinary PR check; an edit anywhere else still needs
  `gh workflow run release.yml --ref <branch>`. `publish` stays gated on
  `github.event_name == 'release'`, which is what makes both safe.
- **The sdist is 264 files** — `hazma/` + `rust/` + pyproject +
  README/LICENSE/CHANGELOG — and source-installs into a fresh CPython
  3.10 venv. maturin honors `.gitignore` for both artifacts, so build
  output stays out; an untracked *unignored* file does not.
- **The version lives in `pyproject.toml`'s `[project] version`, and it
  is on `origin/master`.** Task 7.4's bump edits that line;
  `preflight.sh --closing` reads it and is no longer vacuous (Task 7.1
  merged in PR #83).
- **`pip install -e .` builds release now**, so a `rules.md` rule 12
  benchmark from an editable tree is sound again.

**Currently risky / unknown:**

- **An exact assertion against a compiled kernel may be scoped to the
  cargo profile.** Task 7.1 found and fixed one; the suite is green, but
  a newly written bit-equality claim should be checked against both
  profiles before being trusted.
- **A format assertion written against one platform's artifact encodes
  that platform's shape.** Task 7.2's wheel-tag check passed locally and
  on macOS and rejected a correct manylinux wheel, because a filename's
  tag fields are compressed *sets* and only the Linux wheel has more than
  one member. Both halves of the matrix have to run.
- **`--paths` on `preflight.sh` feeds black, isort and ruff directly**, so
  naming a `.yml` file there makes them parse it as Python and the gate
  goes red on a clean tree. Pass source paths (or the `hazma test`
  default) and use `--md` for markdown.
