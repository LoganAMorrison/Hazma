# The sdist ships cythonized `*.c`, `docs/`, `test/` and `notebooks/`

- **Added:** 2026-08-06
- **Source:** cython-to-rust Task 0.4 — the first `uv build --sdist` run
  in the project's history
- **Scope:** cross-cutting (packaging)
- **Status:** done (2026-08-27, cython-to-rust Task 7.1)
- **Resolution:** folded into Task 7.1, which is the window this file
  named as the cheapest one. `MANIFEST.in` is deleted with the setuptools
  backend and every question below is answered by `[tool.maturin]`:
  - **Item 1, the cythonized `*.c`:** moot. Task 6.4 deleted the last
    `.pyx`, so there is no transpiler to produce them.
  - **Item 2, `docs/` / `test/` / `notebooks/`:** all three are dropped.
    maturin's sdist carries the `hazma/` package, the `rust/` crate and
    the files `[tool.maturin] include` names, so shipping any of them
    would now be a deliberate addition rather than a sweep's by-catch,
    and none is needed to build or install from source. Measured
    2026-08-27 on a clean tree: the sdist goes from **415 files to 264**.
  - **Item 3, working-directory dependence:** mostly fixed, and the
    residue is named rather than assumed. maturin honors `.gitignore` for
    both artifacts, which is exactly the failure this item described — a
    `.pytest_cache/README.md` swept in despite `.gitignore:526` listing
    it, and a built tree's `*.c` and `*.so` alongside. Probed after the
    cutover: a stray `hazma/_stale_probe.abi3.so` reaches neither the
    wheel nor the sdist, with or without `[tool.maturin] exclude`.

    What still tracks the working directory is a file that is untracked
    **and** unignored: `hazma/UNTRACKED_JUNK.txt`, planted as a probe,
    reaches both. That is much narrower than `global-include`, and it is
    the reason this file's "build from a clean tree, and say so when
    quoting a file count" instruction stays good advice rather than
    becoming unnecessary.

    Separately, `[tool.maturin] exclude` carries the one class no ignore
    rule can: four **tracked** editor leftovers under `hazma/`
    (`{A_eff,energy_res}/gecco.dat.bak`, `_gev.py.bak`,
    `form_factors/notes.org`), all four of which ship with
    `exclude = []`.
    `test/test_no_cython_remains.py::test_the_distribution_sweep_ships_no_editor_leftovers`
    asserts those entries survive.
  - **Item 4, the `*.pyd` typo:** deleted with the whole
    `[tool.setuptools.package-data]` block. maturin ships the package
    directory, so the twelve per-directory globs it replaced have no
    successor to drift.

  Verified as this file prescribed: `uv build`, then
  `uv pip install --no-binary hazma dist/hazma-2.1.0.tar.gz` into a fresh
  CPython 3.10 venv, then an import smoke of the public entry points from
  outside the repo.

## Why

Task 0.4 pruned `.claude/`, `.codex/` and `projects/` out of the sdist —
103 files of agent scaffolding that `MANIFEST.in`'s repo-wide
`global-include *.md` was sweeping into a publishable tarball, taking it
from 501 files to 398. That was unambiguous. Four payload questions
surfaced in the same run (the last during PR #49's review) and are
**not** unambiguous, so they were left alone rather than decided inside
a dead-code-purge task:

1. **20 cythonized `*.c` files** (`global-include *.c`). They are never
   used: `cython` is in `[build-system] requires` and `setup.py` calls
   `cythonize()` unconditionally on the `.pyx`, so a source install
   regenerates them. Worse, they are *build-state dependent* — the sdist
   sweeps whatever the source tree happens to have lying around, and
   `AGENTS.md` says generated C is never the source of truth. (The
   observed count is stable at 20 only because the isolated sdist build
   itself runs `cythonize()` first.)
2. **`docs/` (47 files), `test/` (12) and `notebooks/` (2).** Shipping
   tests in an sdist is a defensible convention; shipping the Sphinx
   sources is more marginal. Neither is a defect, just unexamined.
3. **The sweep reaches untracked, `.gitignore`d junk.** setuptools'
   sdist walks the *filesystem*, not the git index, so anything matching
   `global-include` that happens to be lying around is shipped. A tree
   that has run `pytest` grows `.pytest_cache/README.md`, and
   `global-include *.md` duly puts it in the tarball — despite
   `.gitignore:526` listing `.pytest_cache/`. Observed while re-deriving
   this task's counts in review: the same tree gave 400 files dirty and
   398 clean. This is the sharp edge of item 1: the sdist's contents are
   a function of the working directory's state, so two people building
   "the same" release can ship different tarballs. Whatever replaces
   `global-include`, it should enumerate what belongs rather than sweep
   and subtract.
4. **`[tool.setuptools.package-data]`'s `"hazma"` entry lists `*.pyd`,
   not `*.pxd`** (`pyproject.toml`). `.pyd` is the Windows extension-
   binary suffix, which setuptools adds on its own; the Cython headers
   the neighbouring `*.pyx` glob implies are spelled `.pxd`. Almost
   certainly a typo. It is currently harmless — the wheel ships all 17
   `.pxd` anyway (via `MANIFEST.in` + `zip-safe = false`) and nothing
   downstream cimports hazma's headers — which is exactly why it has
   survived unnoticed.

## What

Decide, and encode the decision in whichever backend is live at the time:

- Drop `*.c` from `MANIFEST.in`'s `global-include`, unless there is a
  deliberate reason to ship pre-cythonized sources (there is not, while
  `cython` stays a hard build requirement).
- Keep or drop `docs/` and `notebooks/`; state which and why in a
  comment, so the next reader does not re-litigate it.
- Fix or delete the `*.pyd` package-data entry.
- Make the tarball independent of working-directory state — either
  enumerate includes explicitly instead of `global-include`, or `prune`
  the known artifact directories (`.pytest_cache`, `.ruff_cache`,
  `.venv`, `htmlcov`). Reconciling `MANIFEST.in` against `.gitignore`
  would catch the whole class at once.

Verify the same way Task 0.4 did: `uv build --sdist`, then install the
tarball into a fresh venv with `--no-binary hazma` and import-smoke the
public entry points. A source install that stops working is the failure
mode that matters. **Build from a clean tree**, and say so when quoting
a file count — item 3 makes the number depend on it.

## Entry points

- `MANIFEST.in` (the `global-include` line and the `prune` block Task
  0.4 added); `.gitignore:526` (`.pytest_cache/`), which the sdist
  sweep does not consult
- `pyproject.toml` — `[tool.setuptools.package-data]`,
  `[tool.setuptools.packages.find]`
- `projects/cython-to-rust/task-notes/phase-00/task-0.4-prune-build.md`
  — the measured before/after sdist inventories
- `projects/cython-to-rust/phases/phase-07-cutover.md` Task 7.1 (maturin
  backend) and Task 7.3 (stale `requirements.txt` / `Dockerfile`, which
  are already assigned there and deliberately untouched by Task 0.4)

## Risks / open questions

Sequencing is the whole risk. If this lands before Phase 07 it is two
lines of `MANIFEST.in`; if it lands after, `MANIFEST.in` is gone and the
same decisions have to be expressed through maturin's include/exclude
rules instead. Cheapest window is now; the honest alternative is to fold
it into Task 7.1 rather than leave it drifting between the two.
