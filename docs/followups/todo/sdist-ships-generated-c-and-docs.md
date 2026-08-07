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
101 files of agent scaffolding that `MANIFEST.in`'s repo-wide
`global-include *.md` was sweeping into a publishable tarball. That was
unambiguous. Three payload questions surfaced in the same run and are
**not** unambiguous, so they were left alone rather than decided inside a
dead-code-purge task:

1. **20 cythonized `*.c` files** (`global-include *.c`). They are never
   used: `cython` is in `[build-system] requires` and `setup.py` calls
   `cythonize()` unconditionally on the `.pyx`, so a source install
   regenerates them. Worse, they are *build-state dependent* — the sdist
   sweeps whatever the source tree happens to have lying around, and
   `AGENTS.md` says generated C is never the source of truth. (The
   observed count is stable at 20 only because the isolated sdist build
   itself runs `cythonize()` first.)
2. **`docs/` (46 files), `test/` (12) and `notebooks/` (2).** Shipping
   tests in an sdist is a defensible convention; shipping the Sphinx
   sources is more marginal. Neither is a defect, just unexamined.
3. **`[tool.setuptools.package-data]`'s `"hazma"` entry lists `*.pyd`,
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

Verify the same way Task 0.4 did: `uv build --sdist`, then install the
tarball into a fresh venv with `--no-binary hazma` and import-smoke the
public entry points. A source install that stops working is the failure
mode that matters.

## Entry points

- `MANIFEST.in` (the `global-include` line and the `prune` block Task
  0.4 added)
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
