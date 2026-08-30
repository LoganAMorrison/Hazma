# Phase 07 Learnings: Packaging cutover and close

**Phase:** 07 — Packaging cutover and close
**Closed:** 2026-08-29
**Tasks:** 7.1 (maturin backend), 7.2 (release pipeline), 7.3 (docs
sweep), 7.4 (close)

This file replaces the phase's four task notes and its
`task-notes/phase-07/README.md` for every later reader
([ADR-0002](../../../docs/adrs/ADR-0002-read-phase-learnings-not-closed-task-notes.md)).

## 1. Implementation Reality Check

The phase delivered what it promised. `pyproject.toml` is the only build
entry point, `[build-system] requires` is `["maturin>=1.5,<2.0"]`, a
release publishes two `cp310-abi3` wheels and an sdist from
`PyO3/maturin-action@v1`, no live instruction document states a Cython
fact, and the project closes at 3.0.0. Four of four tasks landed; no ADR
was needed, because ADR-0001 had already fixed the framework and every
packaging question this phase answered was an implementation of it.

What the plan did not anticipate is that **three of the four tasks found
their own criterion had been written against an anticipated tree rather
than the real one**, and in each case the criterion was smaller than the
work:

- **Task 7.1** inherited a bullet Task 6.4 had already half-discharged:
  6.4 deleted the last `.pyx`, which stranded the cython/numpy/scipy
  build requirements, so it removed them rather than shipping a false
  requirement, and 7.1's criterion was amended in place rather than
  quietly satisfied.
- **Task 7.2**'s criterion opened by asking for Cython caching to be
  dropped from `ci.yml`. There had never been any: `rg -in 'cython'
  .github/workflows/` returns six hits on `origin/master`, all prose
  inside comments. The clause was written in the August 2026 analysis
  against a shape the repository never had.
- **Task 7.3**'s enumeration found 14 of 34 stale sites, because it was
  scoped to the two files that first stated the fact. Run over
  `README.md`, `docs/` and both skill trees as well, the same pattern
  returned 20 more.

The generalization, and the one worth carrying past this project:
**a criterion scoped to a file is a criterion scoped to a guess.** Phase
02 learned that "it exists" and "it is load-bearing" differ; Phase 03,
that a plan's model of an external artifact is a hypothesis. This phase's
version is about *text*: a claim written once and pasted into seven
skill files has to be swept by its wording, not by the file that
originated it.

## 2. Critical Context for Future Work

- **`pyproject.toml` is the only build entry point.** `setup.py`,
  `MANIFEST.in`, `setup.cfg`'s `[aliases]`, `requirements.txt` and the
  `Dockerfile` are all deleted. `[tool.maturin]` carries
  `python-source = "."`, `manifest-path`, `module-name = "hazma._core"`,
  `exclude` and `include`. `test/test_no_cython_remains.py` asserts the
  build-requirement and build-script halves so they cannot regress
  unnoticed.
- **The version lives in `pyproject.toml`'s `[project] version`**, and
  `hazma.VERSION` reads it back out of `importlib.metadata`. That
  inversion is what lets the backend learn the version without importing
  a package it has not built. `preflight.sh --closing` parses the
  `[project]` table specifically, so a `version` under any `[tool.*]`
  table cannot answer for it.
- **`pip install -e .` now builds release, not debug.** maturin's PEP 517
  hooks build release unconditionally; only the `maturin develop` CLI
  (which this repo never invokes) defaults to debug. `rules.md` rule 12's
  benchmarks are therefore sound from an ordinary editable tree — but
  still run them from *outside* the repository, or `hazma` resolves to
  the worktree instead of site-packages.
- **`release.yml` runs on pull requests** that touch `release.yml` or
  `pyproject.toml`, so a packaging edit is measurable without cutting a
  release. `publish` stays gated on `github.event_name == 'release'`.
  This closes the `[unrun-workflow-cannot-close-a-criterion]` hole that
  Phase 02 opened and Task 7.1 inherited.
- **maturin honors `.gitignore` for both sdist and wheel**, so build
  output cannot leak into a release from a dirty tree. A file that is
  untracked **and** unignored still gets in, so build releases from a
  clean tree.
- **Nothing builds the docs.** There is no `.readthedocs.yaml` in the
  repository — the published docs come from RTD's default detection — and
  neither CI nor `preflight.sh` runs Sphinx. A broken docs build turns
  nothing red, so check it by hand when a change could plausibly affect
  it.

## 3. Quirk Log & Edge Cases

- **A wheel filename's last three fields are compressed tag *sets*, not
  tags.** Each is dot-separated and the wheel carries their cross
  product, one `Tag:` line per member. The manylinux wheel is
  `cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64` — one platform
  field standing for two tags — while the macOS wheel is one tag per
  field. An assertion comparing field against `Tag:` line as a single
  string passes on macOS and rejects a correct Linux wheel. **Both halves
  of a two-platform matrix must run before a format assertion is
  believed**; a locally built wheel is the macOS shape only.
- **`twine check` cannot fail an unbuildable sdist.** It reads metadata.
  An archive missing `rust/Cargo.toml` passes it and then fails every
  `pip install --no-binary`, and nothing else notices, because both wheel
  jobs build from the checkout rather than from the sdist. The sdist
  needs its own install-and-import job.
- **The manylinux container ships its own Rust toolchain.** Phase 02's
  `CIBW_BEFORE_ALL_LINUX` rustup install, its `CIBW_ENVIRONMENT_LINUX`
  `PATH` edit, and the host `dtolnay/rust-toolchain` step that covered
  macOS all came out with cibuildwheel; `maturin-action` needs none of
  them.
- **`--paths` on `preflight.sh` feeds black, isort and ruff directly.**
  Naming a `.yml` or `.md` file there makes them parse it as Python and
  the gate goes red on a clean tree. Pass source paths (or the default
  `hazma test`) and route markdown through `--md`.
- **A `.bak` suffix is not evidence that a file is a copy.** Two of the
  four "editor leftovers" Task 7.3 examined are not backups:
  `A_eff/gecco.dat.bak` cites a different source than `A_eff/gecco.dat`
  and tabulates a different grid, starting at 0.1 MeV rather than 0.2.
- **`hazma/_utils/` outlived its contents.** Task 6.4 deleted the four
  Cython headers and left the package's empty `__init__.py`, so the
  directory was an importable no-op until Task 7.3 removed it. Every
  `._utils` hit elsewhere in the tree is a sibling package, not this one
  — which is what made the stale directory look load-bearing.

## 4. Test Infrastructure State

- **`test/test_no_cython_remains.py` is the standing anti-regression
  gate** for the whole cutover: no Cython source anywhere, no setuptools
  build script, no Cython toolchain in the build requirements, and no
  editor leftovers in the distribution sweep. It is why no test module
  reads a build script any more — five did, through `setup.py`, until
  Task 7.1.
- **`release.yml` asserts its own output**, and at two levels: the wheel
  filename's tag fields against the `.dist-info/WHEEL` `Tag:` lines, and
  `hazma/_core.abi3.so` as the *only* compiled object in the wheel. It
  also import-smokes the one `cp310-abi3` wheel under both CPython 3.10
  and 3.14 with `--no-deps`, which is the claim abi3 actually makes.
- **`preflight.sh --closing` is not vacuous any more.** It reads
  `[project] version` from both the working tree and the base ref and
  fails on a missing baseline rather than reporting PASS, then requires a
  matching `## [X.Y.Z]` section in `CHANGELOG.md`.
- **The full gate on the capturing platform at close:** bare `pytest -q`
  is 2231 passed / 15 skipped / 12 subtests, and
  `cargo test --no-default-features` is 258 passed (0 doc-tests).
  Re-derive rather than quoting — the cargo count was 249 as recently as
  Task 7.1's note and the pytest count moved four times inside this phase
  alone.

## 5. Follow-on seeds

All four are filed, and none blocks anything.

- [Consolidate the divergent constants tables](../../../docs/followups/todo/consolidate-the-two-constants-tables.md)
  — the project's most obvious deferred cleanup, and a declared numerical
  change. `rules.md` rule 4 forbade it while kernels were being ported;
  that constraint lifts at close.
- [Free-threaded `abi3t` wheels](../../../docs/followups/todo/free-threaded-abi3t-wheels.md)
  — waiting on PyO3 and on the dependency ecosystem, not on hazma. The
  crate's only mutable module state is already `Mutex`-guarded.
- [The relic-density Boltzmann solve in Rust](../../../docs/followups/todo/relic-density-odes-in-rust.md)
  — deliberately out of scope, and it stays out until a profile says
  otherwise.
- [Wheels for linux-aarch64 and Windows](../../../docs/followups/todo/wheels-for-aarch64-and-windows.md)
  — Task 7.2 decided against them and declined to file, on the grounds
  that `PLAN.md` §Scope recorded the option. Closing the project made
  that record archival, so Task 7.4 filed it with the decision unchanged.

Task 7.3 also filed
[four tracked non-source files under `hazma/`](../../../docs/followups/todo/tracked-non-source-files-under-hazma.md),
which is where the `gecco.dat.bak` question landed.
