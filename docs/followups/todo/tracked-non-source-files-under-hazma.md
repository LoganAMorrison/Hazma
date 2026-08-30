# Decide the fate of the four tracked non-source files under `hazma/`

- **Added:** 2026-08-29
- **Source:** `projects/cython-to-rust/phases/phase-07-cutover.md` Task 7.3
- **Scope:** commit
- **Status:** open
- **Triggers / blockers:** none — but the two `.dat.bak` files are a
  physics-provenance question, so the maintainer should make that call
  rather than an agent.

## Why

`hazma/` tracks four files that are not python, not data any module
reads, and not documentation the Sphinx build renders:

- `hazma/gamma_ray_data/A_eff/gecco.dat.bak`
- `hazma/gamma_ray_data/energy_res/gecco.dat.bak`
- `hazma/vector_mediator/_gev.py.bak`
- `hazma/vector_mediator/form_factors/notes.org`

cython-to-rust Task 7.1 kept them out of both distribution artifacts via
`[tool.maturin] exclude`, and Task 7.3 examined them and deliberately
kept them in the repository. They cost users nothing today, so this is
tidiness rather than a defect — but leaving them undecided means the next
packaging change re-litigates them.

## What

Task 7.3's reading of the four, which is where a decision should start:

- **The two `gecco.dat.bak` are not backups of the live files.** They
  carry a different cited provenance and different numbers.
  `A_eff/gecco.dat` is headed "From Alexander Moiseev Febuary 7th, 2022"
  and starts at 0.2 MeV; `A_eff/gecco.dat.bak` is headed "Taken from
  slides from Alexander Moiseev" and starts at 0.1 MeV. Deleting them
  discards a superseded detector response someone reproducing an older
  GECCO projection might want. That is a physics call.
- **`_gev.py.bak` is a pre-refactor snapshot.** It is the monolithic
  ancestor of the `hazma/vector_mediator/_gev/` package that replaced it,
  and git holds it. Nothing imports it.
- **`notes.org` is reference content, not cruft.** Nine lines of isospin
  decompositions for the two-meson states the form factors in the same
  directory build on. If it goes, the content is worth moving into a
  module docstring rather than dropping.

Whatever is decided, keep the two `[tool.maturin] exclude` globs and the
assertion in `test/test_no_cython_remains.py` — they are the guard
against a new tracked leftover shipping, and they are cheap. Update that
test's docstring and the `exclude` comment in `pyproject.toml` if the
population it describes changes.

## Entry points

- `pyproject.toml` — the `[tool.maturin] exclude` block and the comment
  above it that enumerates the four files.
- `test/test_no_cython_remains.py::test_the_distribution_sweep_ships_no_editor_leftovers`
- `projects/cython-to-rust/task-notes/phase-07/task-7.3-docs-sweep.md`
  — where the reading above was recorded.
