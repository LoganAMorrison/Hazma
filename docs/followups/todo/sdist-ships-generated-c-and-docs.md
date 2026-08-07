# The sdist ships cythonized `*.c`, `docs/`, `test/` and `notebooks/`

- **Added:** 2026-08-06
- **Source:** cython-to-rust Task 0.4 — the first `uv build --sdist` run
  in the project's history
- **Scope:** cross-cutting (packaging)
- **Status:** open
- **Triggers / blockers:** decide before Phase 07 Task 7.1 rewrites the
  build backend on maturin — maturin's sdist is `Cargo.toml`-driven and
  will not read `MANIFEST.in`, so whatever is decided here has to be
  re-expressed there rather than carried over.

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
