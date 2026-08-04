---
phase: 00
title: Dead-code purge
status: Not started
---

<!-- markdownlint-disable-file MD025 -- frontmatter title is the schema -->

# Phase 00: Dead-code purge

## Goal

Delete the ~6,500 lines of unbuilt, unimported, or broken-on-import
Cython (and its data) so later phases port only live code. After this
phase the build has ~19 extensions, zero C++, and no `.pyx` outside the
live surface. No user-visible behavior changes except the resolved
`hazma.gamma_ray` decision (Task 0.5).

## Prerequisites

- Read `../references/cython-inventory.md` (dead-code map + constants
  entanglement) and `../rules.md` rule 10 (verify-before-delete).

## Future Phases (read-only)

- Phase 01 builds the parity corpus over the survivors; keep anything a
  corpus entry point needs.

## Tasks

### Task 0.1: Relocate legacy constants header

**Task note:** [`../task-notes/phase-00/task-0.1-relocate-constants.md`](../task-notes/phase-00/task-0.1-relocate-constants.md)
**Depends on:** —

**Exit criteria:**

- `hazma/_decay/parameters.pxd` moved verbatim (byte-identical values)
  to `hazma/_utils/legacy_parameters.pxd`.
- The four live `include "../_decay/parameters.pxd"` sites (both
  `scalar_mediator/*_spec*.pyx`, both `vector_mediator/*_spec*.pyx`)
  repointed; `pip install -e .` rebuilds; import smoke passes.
- Do **not** merge values into `_utils/constants.pxd` (rules.md rule 4).

### Task 0.2: Delete the phase-space / gamma-ray slice

**Task note:** [`../task-notes/phase-00/task-0.2-delete-mc-slice.md`](../task-notes/phase-00/task-0.2-delete-mc-slice.md)
**Depends on:** Task 0.1, Task 0.5 (gamma_ray decision)

**Exit criteria:**

- Deleted: `hazma/_phase_space/`, `hazma/_gamma_ray/`,
  `hazma/deprecated/rambo.py`,
  `hazma/rh_neutrino/_rh_neutrino_fsr_four_body.{pyx,pyi}`.
- `hazma/gamma_ray.py` handled per Task 0.5's decision.
- Importer check re-run at delete time and quoted in the PR body.
- `versioning.md` call for the `deprecated/rambo.py` removal stated
  explicitly in the PR body.

### Task 0.3: Delete superseded per-particle kernels and helpers

**Task note:** [`../task-notes/phase-00/task-0.3-delete-superseded.md`](../task-notes/phase-00/task-0.3-delete-superseded.md)
**Depends on:** Task 0.1

**Exit criteria:**

- Deleted: `hazma/_positron/`, `hazma/_neutrino/`, `hazma/_decay/`
  (incl. `interpolation_data/` and backups), `hazma/__decay.py`,
  `hazma/__positron_spectra.py`, `hazma/__neutrino_spectra.py`,
  `hazma/spectra/_positron/_kaon.pyx`,
  `hazma/field_theory_helper_functions/` (both modules), the dead
  ~165-line half of `hazma/_utils/boost.pyx`, and
  `test/decay/` (already collect_ignored and importing a nonexistent
  module).
- `cross_section_prefactor` callers use `hazma.utils`;
  `minkowski_dot` given a pure-Python home and
  `hazma/experimental/axial_vector_mediator/avm_msqrd.py` repointed.
- Import smoke (`hazma.theory`, `hazma.limits`, `hazma.cmb`,
  `hazma.pbh`, both mediators, `hazma.spectra._photon._muon`) passes.

### Task 0.4: Prune build and packaging config

**Task note:** [`../task-notes/phase-00/task-0.4-prune-build.md`](../task-notes/phase-00/task-0.4-prune-build.md)
**Depends on:** Tasks 0.2, 0.3

**Exit criteria:**

- `setup.py` extension list matches the survivors exactly (count the
  built `.so`s in a fresh `pip install -e .`); no `cpp=True` groups
  remain.
- `pyproject.toml` package-data globs and `MANIFEST.in` no longer ship
  deleted directories; sdist builds and contains no deleted paths.
- CI green on the full matrix.

### Task 0.5: Decide and execute the `hazma.gamma_ray` fate

**Task note:** [`../task-notes/phase-00/task-0.5-gamma-ray-decision.md`](../task-notes/phase-00/task-0.5-gamma-ray-decision.md)
**Depends on:** — (decision precedes Task 0.2's delete)

**Exit criteria:**

- Investigated whether `hazma.spectra` n-body machinery supersedes
  `gamma_ray.gamma`/`gamma_point` for every documented use.
- One of: (a) `hazma/gamma_ray.py` reimplemented over
  `hazma.phase_space`/`hazma.spectra` (keeps project `version_bump:
  minor`), or (b) module deleted with an ADR-0003 recording the
  `major` implication, PLAN frontmatter updated accordingly.
- Docs referencing the module updated either way.

**Notes:** The module is broken on import today, so no working user
exists; the decision is about the public-name contract, not behavior.

## Exit Criteria

- All tasks complete; preflight green; CI green.
- `find hazma -name "*.pyx"` lists only the live surface (19 modules +
  `_utils/boost.pyx`).
- No `language="c++"` extension remains; `git grep -l "std::"` over
  `hazma/` is empty.
- Phase learnings written to `../learnings/phase-00-dead-code-purge.md`.
