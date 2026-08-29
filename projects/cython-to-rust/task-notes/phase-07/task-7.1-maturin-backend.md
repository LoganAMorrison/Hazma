# Task 7.1: Backend switch to maturin

**Date:** 2026-08-27
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-07-cutover.md` (Task 7.1);
`../../PLAN.md` (Scope, Numerical impact, Closing this project)
**Related ADRs:** ADR-0001 (framework choice — names maturin as the
packaging half of the decision)
**Depends On:** Phase 06 complete (zero Cython, one extension)

## Objective

Replace the setuptools + setuptools-rust build with maturin, move the
version's source of truth to `pyproject.toml` so the backend can read it
without importing the package it has not built, and delete `setup.py` and
`MANIFEST.in` along with the machinery that read them.

## Exit Criteria

From the phase file's Task 7.1 block, as amended by Task 6.4:

- `[build-system] requires = ["maturin>=1.x"]`; `[tool.maturin]`
  mixed-layout config (python packages + `rust/Cargo.toml`,
  `module-name = "hazma._core"`); setuptools-rust and `setup.py` deleted.
  (The cython/scipy/numpy requirements were already gone — Task 6.4.)
- Version static in `[project]`; `hazma.VERSION` reads
  `importlib.metadata.version`, attribute preserved;
  `scripts/agents/preflight.sh --closing` version check updated to the
  new source of truth.
- `uv pip install -e .` and plain `pip install .` build from a clean
  clone; import smoke green; package-data globs pruned to what pure
  Python still reads; `MANIFEST.in` deleted, with sdist contents verified
  explicitly.

## Inputs Reviewed

- `../../PLAN.md`; `../README.md` (handoff); `../phase-06/README.md`
  (handoff — the four Phase-07 risks); `../../phases/phase-07-cutover.md`
- `../../learnings/phase-06-mediator-spectra.md`; `../../rules.md`
- `pyproject.toml`, `setup.py`, `MANIFEST.in`, `setup.cfg`,
  `rust/Cargo.toml`, `rust/build.rs`, `.github/workflows/ci.yml`
- `scripts/agents/preflight.sh` (gate 10), `docs/versioning.md`
- `docs/followups/done/sdist-ships-generated-c-and-docs.md` and
  `.../editable-installs-build-the-rust-extension-in-debug.md` (read
  under `todo/`; both named this task as their window and both are closed
  here, so the paths above are their post-move ones)
- `docs/agents/environment.md`, `docs/agents/lessons.md`

## Findings

- **Every one of the four risks the Phase 06 handoff listed was real**,
  and a fifth was not listed: five per-module twin tests also read
  `setup.py` (`test_core_{neutrino,photon_rho,vector_xs,scalar_xs}.py`
  and `test_core_mediator_positron.py`), not just
  `test_no_cython_remains.py`'s `MANIFEST.in` check. All five asserted
  "this module is out of the build declaration". That claim now belongs
  to `test_no_cython_remains.py`'s tree-wide
  `test_the_build_requirements_name_no_cython_toolchain`: with no
  `setuptools` in `[build-system] requires`, no `Extension` can exist for
  any module, so the per-module restatements were dropped rather than
  repointed.
- **The setuptools-rust wheel was never abi3-tagged.** Despite
  `py_limited_api=True`, the baseline wheel came out
  `hazma-2.1.0-cp314-cp314-macosx_26_0_arm64.whl` containing a correctly
  named `_core.abi3.so`. maturin reads the crate's `abi3-py310` feature
  and tags the wheel `cp310-abi3-macosx_11_0_arm64`. Verified across
  versions: that wheel, built under CPython 3.14, installs and imports
  under CPython 3.10. Task 7.2 inherits a solved tagging problem.
- **`[tool.setuptools.package-data]` had drifted in both directions.** It
  shipped `*.pyd` (the Windows binary suffix) where it meant `.pxd`, and
  it omitted `hazma/form_factors/vector/testdata/*.json` — the
  parametrization for `_pi_pi_test.py` and friends, which are `.py` files
  inside `hazma/` and so already shipped. The wheel carried those tests
  without the data they need.
- **The `maturin>=1.5` floor is measured, not read off a changelog:**
  1.5.1, 1.7.8, 1.9.6 and 1.15.0 each build this configuration from the
  clean-clone export and each emit the same
  `hazma-2.1.0-cp310-abi3-macosx_11_0_arm64.whl`.
- **Editable installs are release builds under maturin.** The debug
  default belonged to `setuptools_rust.build_rust`'s
  `debug = self.inplace or self.debug`; maturin's PEP 517 hooks build
  release unconditionally, and only the separate `maturin develop` CLI
  (which this repo never invokes) defaults to debug. Measured:
  `uv pip install -e .` leaves `rust/target/release/` and no `debug/`,
  and `thermal_cross_section(x=0.5)` runs at **35.8 us** against the
  follow-up's 1866 us debug figure.
- **That profile change surfaced a latent test defect, and the test was
  wrong rather than the build.** `test_core_mediator_tables.py`'s grid
  comparison asserted bit-equality against `numpy.logspace` scoped by
  *platform*, but exactness there depends on the cargo **profile**: under
  `[profile.release]`'s `lto = true` / `codegen-units = 1` the grid moves
  one ulp at 4 of 500 abscissae at m = 550 MeV and 1 of 500 at
  m = 900 MeV. Proved independent of this task's diff by running
  origin/master's tree against each extension in turn (`rust/src` is
  byte-identical across the two branches): **70 passed** with the debug
  `.so`, **7 failed, 63 passed** with the release one. Wheels have always
  been release builds, so the release values are the ones users have
  always received; the bit-equal branch was green only because the
  documented dev loop and CI both installed editable.

## Decisions and Implementation Notes

- **`[project] version` is the source of truth; `hazma.VERSION` reads it
  back** via `importlib.metadata.version("hazma")`. Both `VERSION` and
  `__version__` are preserved as public API. No fallback sentinel: a
  hazma that is importable but not installed cannot compute anything (the
  compiled `_core` only exists after an install), so `PackageNotFoundError`
  is a better outcome than a version string that lies.
- **`preflight.sh --closing` isolates the `[project]` table before
  matching `version`**, so a `version` under any `[tool.*]` table cannot
  answer for it. Unit-checked against four inputs: current tree (2.1.0), a
  bumped copy (3.0.0), `origin/master`'s dynamic-version file (correctly
  unreadable), and a decoy whose only `version` sits under `[tool.ruff]`
  (correctly unreadable).
- **maturin's package sweep replaces twelve package-data globs.** It
  ships the `hazma/` directory, so runtime data is included by being in
  the package. What that sweep excludes was **measured, not assumed** —
  the first draft of this config carried `hazma/_core.abi3.so` and
  `hazma/**/*.so` entries that turned out to do nothing, and the comment
  explaining them asserted a mechanism maturin does not use. Probed by
  planting files and rebuilding: maturin honors `.gitignore`, so a stray
  `_stale_probe.abi3.so` reaches neither artifact with `exclude = []`;
  an untracked *unignored* file reaches both; and the four **tracked**
  editor leftovers reach both unless excluded. So `exclude` is
  `["hazma/**/*.bak", "hazma/**/*.org"]` and nothing more, and the guard
  test asserts those rather than a no-op. Excluded from the distribution
  rather than deleted from the repository, which is a separate question.
- **The grid comparison moved to the one-ulp budget the module already
  derived**, on every platform, rather than keeping an exact branch that
  encoded a debug build's arithmetic. The interp comparison keeps its
  platform split — measured unaffected by the profile in the same run.
  `rules.md` rule 2's spirit: the justification is in
  `assert_matches_numpy_grid`'s docstring, this note, and the reopened
  risk bullet of the closed follow-up.
- **`test_no_cython_remains.py`'s repo walk now skips `site-packages`.**
  The documented dev loop builds `.venv` inside the checkout and numpy
  ships 26 `.pxd` headers there, so the walk reported a working
  environment as a Cython regression. Skipping by that directory name
  rather than by `.venv` catches every virtualenv layout.
- **Two follow-ups closed** — both named this task as their window.
  `sdist-ships-generated-c-and-docs` is answered item by item by
  `[tool.maturin]`; `editable-installs-build-the-rust-extension-in-debug`
  is resolved by the backend swap itself. Its "the two profiles are
  numerically identical" risk note is **corrected in place**, since Task
  5.3's relic-density evidence holds for the functions it measured and
  does not generalize.
- **Deliberately left to Task 7.3:** `requirements.txt`, `Dockerfile`,
  `docs/source/installation.rst` (which still says
  `python setup.py install`), `setup.cfg`'s now-inert `[aliases]`
  section, `AGENTS.md`'s "Never commit generated C/C++ — `setup.py`
  cythonizes on build" convention, and the remaining Cython-fact
  paragraphs in `docs/agents/environment.md` (the `.pyx` rebuild note and
  the stale-`.so` note). All are recorded in the phase file's 7.3 block.
  What this task did patch in durable docs is only what its own diff
  falsified: the version mechanism, the build backend, and the
  sdist/wheel machinery. **PR #83 review** asked for the deferral to be
  checked rather than trusted: it was an undercount, so the phase file's
  7.3 block now carries the eleven-row inventory the sweep produced
  instead of a prose gesture at "two remaining `.pyx` paragraphs".

## Files Changed

Build and packaging:

- `pyproject.toml` — static `[project] version`; `[build-system]` on
  maturin alone; `[tool.setuptools.*]` replaced by `[tool.maturin]`
  (`python-source`, `manifest-path`, `module-name`, `exclude`, `include`)
- `setup.py`, `MANIFEST.in` — deleted
- `hazma/__init__.py` — `VERSION` reads `importlib.metadata.version`
- `rust/Cargo.toml`, `rust/build.rs` — comments naming setuptools-rust as
  the installer, and the "wheels stay CPython-tagged" note, repointed
- `.github/workflows/ci.yml` — three comments repointed; no step changed

Gates and tests:

- `scripts/agents/preflight.sh` — gate 10 reads `[project] version` from
  `pyproject.toml`
- `test/test_no_cython_remains.py` — `setup.py` and `MANIFEST.in` tests
  replaced; shared `_toml_array` helper; `site-packages` skipped in the walk
- `test/test_core_{neutrino,photon_rho,scalar_xs,vector_xs}.py`,
  `test/test_core_mediator_positron.py` — `setup.py` reads removed
- `test/test_core_mediator_tables.py` — `assert_matches_numpy_grid` on the
  one-ulp budget everywhere; comparator test split into grid and interp halves
- `test/conftest.py` — `collect_ignore` emptied

Docs:

- `docs/versioning.md`, `docs/workflow.md`, `docs/agents/preflight.md`,
  `docs/agents/doc-consistency.md`, `docs/agents/environment.md`,
  `docs/agents/review-lenses.md`, `docs/PR_GUIDELINES.md`, `AGENTS.md`
- `docs/followups/` — two entries moved to `done/` with resolutions, index
  rows moved, 23 inbound references repointed;
  `todo/moved-followups-leave-dangling-inbound-paths.md` re-swept
- `projects/{_template,cython-to-rust,parity-pinned-defect-repair}/` —
  closing-PR instructions repointed at `[project] version`

## Verification

- `pytest -q` (bare, the whole suite) — **2231 passed, 15 skipped,
  12 subtests passed in 24.96s**. Before the two test repairs the same
  command gave `12 failed, 2219 passed`: five reading the deleted
  `setup.py`, seven the profile-sensitive grid comparison. The count is
  unchanged from `origin/master`, which gives the same
  `2231 passed, 15 skipped, 12 subtests` — this task removed one test
  (`test_setup_py_builds_neither`) and split one comparator test in two.

  The pass/skip **split** is environment-dependent, so the local figure
  is not the portable statement; the collected total (2246) is. CI is,
  and it is identical either side of the cutover:

  | | Linux, py3.10–3.14 | macOS, py3.14 |
  | --- | --- | --- |
  | `origin/master` (run 33142305371) | 2229 passed, 17 skipped | 2230 passed, 16 skipped |
  | this branch (run 33233765164) | 2229 passed, 17 skipped | 2230 passed, 16 skipped |

  The gradient — 15 skipped locally, 16 on macOS CI, 17 on Linux CI — is
  the parity corpus's environment guard (`test_parity.py:297` skips when
  the interpreter, NumPy, SciPy or platform drift from the manifest) plus
  the platform-scoped modules, and it predates this task.
- `pytest test/test_no_cython_remains.py -q -o addopts=""` — `4 passed`.
- `pytest test/test_core_mediator_tables.py -q -o addopts=""` —
  `71 passed` (70 before; one comparator test split in two).
- `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`,
  `cargo test --no-default-features` — via `preflight.sh`; see its table.
- **Clean-clone plain install:** `git ls-files` exported to a fresh
  directory (no `.git`, no build state), `uv pip install .` into a new
  CPython 3.12 venv — exit 0, `import hazma, hazma._core` green from
  outside the repo.
- **sdist source install:** `uv build`, then
  `uv pip install --no-binary hazma dist/hazma-2.1.0.tar.gz` into a fresh
  CPython **3.10** venv — exit 0; `hazma.VERSION == "2.1.0"`,
  `dnde_photon_muon([1.0, 10.0], 200.0)` evaluates from outside the repo.
- **abi3 across versions:** the `cp310-abi3` wheel built under CPython
  3.14 installs and imports under CPython 3.10.
- **Artifact inventories**, built from a tree carrying an editable
  install's `hazma/_core.abi3.so`:

  | artifact | setuptools baseline | maturin | note |
  | --- | --- | --- | --- |
  | sdist files | 415 | 264 | `docs/`, `test/`, `notebooks/`, egg-info, `setup.cfg`, `requirements.txt` all dropped |
  | wheel files | 221 | 227 | +6 `testdata/*.json`; the 4 `.bak`/`.org` leftovers excluded |
  | wheel tag | `cp314-cp314-macosx_26_0_arm64` | `cp310-abi3-macosx_11_0_arm64` | one wheel per platform, not per CPython |
  | `.so` in wheel | 1 | 1 | the freshly built one, not the tree's stale copy |
  | `.so`/`.bak`/`.org`/egg-info in sdist | — | none | |

- **Editable-install profile:** `rust/target/release/` present,
  `rust/target/debug/` absent.
- Deferred by design: release.yml is untouched (Task 7.2 owns it, and it
  has no pull-request trigger, so a change there cannot be measured
  without a dispatch — `lessons.md`
  `[unrun-workflow-cannot-close-a-criterion]`).

**Preflight** — `scripts/agents/preflight.sh --paths "hazma test" --md
"<35 changed docs>"`:

```text
PASS   black --check           hazma test
FAIL   isort --check-only      run `isort hazma test` and re-check
FAIL   ruff check              see output below
PASS   cargo fmt --check       rust/
PASS   cargo clippy            rust/
PASS   cargo test              rust/
PASS   pytest                  2231 passed, 15 skipped, 12 subtests passed
PASS   import hazma            version 2.1.0
PASS   markdownlint            <35 changed docs>
SKIP   version bump            not a closing PR (pass --closing)
PASS   forbidden tokens        none added
```

The two FAILs are the trunk condition
[`preflight-isort-ruff-red-on-trunk.md`](../../../../docs/followups/todo/preflight-isort-ruff-red-on-trunk.md)
records, and the "is this mine?" analysis it asks every task in flight to
redo was run rather than assumed: `isort --check-only hazma test` names
files under `hazma/` that this diff does not contain, and
`ruff check hazma test` reports **6091 errors across 171 files**, of
which the intersection with this task's nine changed `.py` files is
**empty** (`comm -12` of the two sorted lists). Scoped to those nine
files, `black --check`, `isort --check-only`, `ruff check` and
`ruff check --isolated --select E9,F63,F7,F82` are all clean.

Note the `--paths` scoping: passing `scripts` as well turns black red
too, because `scripts/` carries unformatted Python. No Python file under
`scripts/` is in this diff — the change there is to a shell script.

## Numerical impact

**No public value changes.** The diff touches no kernel: `git diff
origin/master --stat -- rust/src` is empty, so the extension's arithmetic
is byte-identical to the trunk's.

The one mechanism that could have moved a number is the editable build's
cargo profile (debug → release). Measured directly, by running the same
public-surface sweep against a debug and a release `hazma._core` built
from identical sources: **16 arrays, 7,206 values, bit-equal** — the ten
`dnde_*` spectra at m = 900 MeV over 400 log-spaced energies, both
mediator models' total photon and positron spectra, and three
`thermal_cross_section` points each.

The profile does move one intermediate: `mediator_tables`' log-spaced
grid, by exactly **1 ulp** at 5 of 1000 sampled abscissae (4 of 500 at
m = 550 MeV, 1 of 500 at m = 900 MeV). It does not propagate to any
public value in the sweep above, and it is not new — every published
wheel has always been a release build. Nothing is appended to
`../numerical-impact.md`: no function's published output moved, in either
direction, at any tolerance.

## Open Questions

- **Should the four tracked editor leftovers be deleted from the
  repository** rather than only excluded from the distribution
  (`hazma/gamma_ray_data/{A_eff,energy_res}/gecco.dat.bak`,
  `hazma/vector_mediator/_gev.py.bak`,
  `hazma/vector_mediator/form_factors/notes.org`)? Out of scope here —
  deleting tracked files under `hazma/` is a source question, not a
  packaging one — and harmless while excluded. Task 7.3's stale-artifact
  pass is the natural place, alongside `requirements.txt` and the
  `Dockerfile` it already owns.
- **Three follow-up slugs still have dangling inbound paths** from
  earlier tasks' notes, recorded in
  `docs/followups/todo/moved-followups-leave-dangling-inbound-paths.md`,
  whose evidence block this task re-swept and corrected from two to three.
  Not this task's slugs; its own 23 references were all repointed.

## Plan Impact

**Impact Level:** Phase file patched.

`../../phases/phase-07-cutover.md`'s Task 7.2 and 7.3 blocks each carried
a statement this task made wrong, and both are patched here rather than
deferred: 7.2's "wheel abi3 tags ... verified in the workflow" now records
that the tagging itself is already done and that what remains is the CI
verification, and 7.3's docs list is narrowed to what is actually left
after this task's sweep. The Prerequisites block's packaging facts are
re-stated post-cutover. No ADR: ADR-0001 already names maturin as the
packaging decision, and nothing here departs from it.

`../../PLAN.md`'s "Closing this project" paragraph is patched for the
moved version source of truth — it named `hazma/__init__.py`, which no
longer holds a number. Its parenthetical already anticipated this ("note
Task 7.1 relocates the version's source of truth").

## Stale-state sweep

Each command run against `claude/cython-to-rust/task-7.1-maturin-backend`
at the end of the task.

### Identifier sweep — `setuptools-rust`, `setup.py`, `MANIFEST.in`

```text
$ rg -n 'setuptools-rust' <live files; history archives and closed
    phase notes excluded>
docs/agents/environment.md:59         KEPT  (dated: "Task 7.1; from Phase 02 until then")
pyproject.toml:75                     KEPT  (names what was removed)
test/test_no_cython_remains.py:105    KEPT  (same)
test/test_core_mediator_tables.py:236 KEPT  (the profile finding's cause)
phases/phase-02-rust-scaffold.md:13,27   KEPT  (Phase 02 canon; accurate history)
adrs/ADR-0001-...:37,67               KEPT  (accepted ADR; accurate history)
PLAN.md:97                            KEPT  (Phase 02 row; accurate history)
task-notes/{README,numerical-impact,phase-07/*}.md  KEPT/EDITED per file
```

No hit states the *live* build wrongly. `setup.py` / `MANIFEST.in`
similarly: every remaining occurrence either asserts their absence
(`test/test_no_cython_remains.py`), records that they went
(`test/conftest.py`, the five twin tests, `docs/agents/environment.md:185`),
or is an oracle-restore instruction against an old revision
(`test/parity/oracles/README.md`, where `git show 1b022d4:setup.py` still
resolves). Two remaining live-and-wrong occurrences are **deliberately
left to Task 7.3** and named in its phase-file block: `AGENTS.md:171` and
`docs/source/installation.rst:34`.

### Version-mechanism sweep

```text
$ rg -n 'attr = \{ attr|attr: hazma.VERSION|VERSION` in `hazma/__init__' .
    (history archives and Phase 00-05 notes excluded)
  none
```

### Follow-up path sweep (the loop `moved-followups-...md` prescribes)

```text
$ for p in $(rg -oN --no-filename 'docs/followups/(todo|done)/[a-z0-9-]+\.md' \
      projects/ docs/ hazma/ test/ rust/ README.md CHANGELOG.md | sort -u); do
    [ -f "$p" ] || echo "DANGLING: $p"; done
DANGLING: docs/followups/todo/cross-section-prefactor-threshold-cancellation.md
DANGLING: docs/followups/todo/legacy-parameters-width-exponent-bug.md
DANGLING: docs/followups/todo/oracle-restore-revisions-for-the-mediator-decay-pyx.md
```

Three, all pre-existing and none this task's slugs — they are the two the
open follow-up already lists plus one that accrued since, which this task
added to that file's evidence block. This task's own two slugs produced a
fourth DANGLING mid-sweep (a citation inside this note); EDITED, and the
loop is clean of them now.

### Forward-looking phrase sweep

```text
$ rg -n '(Task [0-9.]+ will|will be added|still pending|today: ?stub)' <changed files>
docs/agents/doc-consistency.md:94,167   KEPT  (the checklist's own text)
phase-05/task-5.3, phase-02/task-2.{1,2}  KEPT  (other tasks' pasted sweep blocks)
```

### Line-number citation sweep

```text
$ python scripts/agents/check_doc_citations.py <35 changed/new .md>
docs scanned: 35
in-repo citations checked: 12
external citations skipped: 9   (8 pre-existing .pyx/.pxd + setup.py:83,
                                 the latter in a closed Phase 05 note —
                                 same class as the .pyx ones, KEPT)
out-of-range or ambiguous: NONE
exit=0
```

### Count sweep

| Claim location | Command | Actual | Status |
| --- | --- | --- | --- |
| §Verification "2231 passed, 15 skipped, 12 subtests" | `pytest -q` | `2231 passed, 15 skipped, 12 subtests passed` | OK |
| §Verification "unchanged from origin/master" | same, in an `origin/master` worktree | `2231 passed, 15 skipped, 12 subtests passed` | OK |
| §Verification sdist 415 → 264 | `tar tzf … \| grep -v '/$' \| wc -l` | 415 (setuptools) / 264 (maturin) | OK |
| §Verification wheel 221 → 227 | `unzip -Z1 … \| wc -l` | 221 / 227 | OK |
| §Verification wheel tag | `basename dist/*.whl` | `hazma-2.1.0-cp310-abi3-macosx_11_0_arm64.whl` | OK |
| §Verification "1 `.so` in wheel" | `unzip -Z1 … \| grep -c '\.so$'` | 1 | OK |
| §Findings "35.8 us" | `python -m timeit` on `thermal_cross_section(0.5, …)` | `35.8 usec per loop` | OK |
| §Findings maturin floor 1.5.1/1.7.8/1.9.6/1.15.0 | `maturin build` per version on the clean-clone export | 4/4 built `cp310-abi3` | OK |
| §Findings "1 ulp at 4 of 500 / 1 of 500" | ulp diff of `photon_tables` vs `np.logspace` | 550 MeV: 4, 900 MeV: 1, max 1 ulp | OK |
| §Findings "70 passed (debug) vs 7 failed, 63 passed (release)" | `pytest test/test_core_mediator_tables.py` on `origin/master` with each `.so` | as stated | OK |
| §Findings "26 `.pxd` under `.venv`" | `rglob` count under `.venv` | 26 | OK |
| §Files Changed "23 inbound references repointed" | `git diff origin/master \| grep -c '^+.*followups/done/<slug>'` | 23 | OK |
| §Numerical impact "16 arrays, 7,206 values" | the sweep script's own output | `16 arrays, 7206 values` | OK |
| §Numerical impact "`rust/src` byte-identical" | `git diff origin/master --stat -- rust/src` | empty | OK |
| `[tool.maturin] exclude` = 2 globs / 4 files | `sed -n '/^exclude = \[/,/^]/p' pyproject.toml` | `hazma/**/*.bak`, `hazma/**/*.org` | OK |

### Numerical-impact statement

**No public value changes.** `git diff origin/master --stat -- rust/src`
is empty, so the kernels are byte-identical to the trunk's. The one
mechanism that could have moved a number — the editable build's cargo
profile going debug → release — was measured directly: a 16-function,
7,206-value sweep of the public spectra, mediator totals and
`thermal_cross_section` is **bit-equal** between a debug and a release
`hazma._core`. The profile does move `mediator_tables`' grid by 1 ulp at
5 of 1000 sampled abscissae; that intermediate reaches no public value in
the sweep, and release is what every published wheel has always shipped.
Nothing appended to `../numerical-impact.md`.

### Exit Criteria → evidence mapping

| Exit criterion | Evidence |
| --- | --- |
| `[build-system] requires = ["maturin>=1.x"]` | `pyproject.toml:80`; `test_no_cython_remains.py::test_the_build_requirements_name_no_cython_toolchain` asserts `== {"maturin"}` |
| `[tool.maturin]` mixed-layout config | `pyproject.toml` `python-source = "."`, `manifest-path`, `module-name = "hazma._core"`; proved by the wheel placing `hazma/_core.abi3.so` |
| setuptools-rust and `setup.py` deleted | `git status`: `D setup.py`; `test_no_cython_remains.py::test_no_setuptools_build_script_remains` |
| Version static in `[project]` | `pyproject.toml:21`; `uv build` stamps `hazma-2.1.0` |
| `hazma.VERSION` reads `importlib.metadata`, attribute preserved | `hazma/__init__.py`; `preflight.sh` gate 8 prints `version 2.1.0`; sdist-installed venv prints `2.1.0 2.1.0` for `VERSION, __version__` |
| `preflight.sh --closing` updated | gate 10 reads `[project] version`; reader unit-checked on 4 inputs incl. a `[tool.ruff]` decoy |
| `uv pip install -e .` builds | exit 0; `hazma._core.__file__` inside the worktree; `rust/target/release/` |
| plain `pip install .` from a clean clone | `git ls-files` export (no `.git`), CPython 3.12 venv, exit 0, imports from outside the repo |
| import smoke green | `preflight.sh` gate 8 PASS; CI's seven-module import run manually on the sdist venv |
| package-data pruned to what pure Python reads | twelve setuptools globs deleted; the six `testdata/*.json` the in-package tests read are now shipped; four editor leftovers excluded |
| `MANIFEST.in` deleted | `git status`: `D MANIFEST.in`; `test_the_distribution_sweep_ships_no_editor_leftovers` asserts absence |
| sdist contents verified explicitly | the inventory table in §Verification, plus a source install into a fresh CPython 3.10 venv |

## Handoff to Next Task

**Task 7.2 (release pipeline) and Task 7.3 (docs sweep) are both
unblocked** and share no files. Read `../../phases/phase-07-cutover.md`
(patched here), `../README.md`, then this note.

**Now safe to assume:**

- **maturin is the whole build.** `[build-system] requires = ["maturin>=1.5,<2.0"]`,
  `[tool.maturin]` carries `python-source = "."`, `manifest-path`,
  `module-name`, `exclude` and `include`. No `setup.py`, no `MANIFEST.in`,
  no `setuptools` anywhere in the build path. `test_no_cython_remains.py`
  asserts all of it.
- **The wheel is already `cp310-abi3` and already portable across
  CPythons** — verified 3.14-built, 3.10-installed. Task 7.2's abi3 work
  is *verifying it in CI*, not producing it. maturin also sets
  `macosx_11_0` rather than the builder's own OS version.
- **The sdist carries only `hazma/`, `rust/`, `pyproject.toml`, README,
  LICENSE, CHANGELOG and PKG-INFO** — 264 files against 415. Verified by
  source-installing it into a fresh 3.10 venv. maturin honors
  `.gitignore` for both artifacts, so build output stays out without
  configuration; a file that is untracked *and* unignored does reach
  them, so build a release from a clean tree.
- **`pip install -e .` now builds release**, so a `rules.md` rule 12
  benchmark taken from an ordinary editable tree is sound again. The
  reinstall costs more than it did; `[profile.release]` still has
  `lto = true` and `codegen-units = 1`.
- **The version lives in `pyproject.toml`'s `[project] version`.** Task
  7.4's closing bump edits that line, not `hazma/__init__.py`.
  `preflight.sh --closing` reads it there already. Note the gate is
  *vacuous on a branch cut before this task merged* — `origin/master`'s
  `pyproject.toml` has no `[project] version` to compare against, so the
  gate correctly reports "bump unverifiable" rather than a false pass.
  Once this lands on master, the baseline resolves.

**Still risky for the rest of Phase 07:**

- **A bit-equality assertion against a compiled kernel may be scoped to
  the build profile, not just the platform.** This task found one
  (`test_core_mediator_tables.py`) and fixed it. Any other test that
  compares a Rust result exactly against a NumPy or Python reference was
  green under debug and is now measured under release. The full suite is
  green, so none is currently failing — but a *newly written* exact
  assertion should be checked against both profiles before being trusted.
- **`release.yml` still has no pull-request trigger.** Task 7.2 rewrites
  it and cannot measure the rewrite without an explicit dispatch.
- **Task 7.3's list is smaller than the phase file used to imply but not
  empty:** `requirements.txt`, `Dockerfile`,
  `docs/source/installation.rst` (`python setup.py install`),
  `setup.cfg`'s inert `[aliases]`, `AGENTS.md`'s layout tree and `.pyx`
  rebuild paragraphs, and `docs/agents/environment.md`'s two remaining
  Cython-fact notes. The build-mechanism half of that sweep is done.
