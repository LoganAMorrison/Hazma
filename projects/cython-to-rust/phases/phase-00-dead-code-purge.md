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
phase the build has 20 extensions (19 kernel modules + the C-level-only
`_utils.boost`), zero C++, and no `.pyx` outside the live surface.

This phase carries the project's two `major`-version removals (both
covered by `PLAN.md` `version_bump: major`): `hazma/deprecated/rambo.py`
(any removal from `hazma/deprecated/` is `major` per
`docs/versioning.md`) and the broken-on-import `hazma.gamma_ray`
module (ADR-0003, Proposed — Task 0.2 is gated on its acceptance).
Everything else is behavior-invisible.

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
**Depends on:** Task 0.1, ADR-0003 accepted (via Task 0.5)

**Exit criteria:**

- Deleted: `hazma/_phase_space/`, `hazma/_gamma_ray/`,
  `hazma/deprecated/rambo.py`, `hazma/gamma_ray.py` (per ADR-0003),
  `hazma/rh_neutrino/_rh_neutrino_fsr_four_body.{pyx,pyi}`.
- Importer check re-run at delete time and quoted in the PR body.
- PR body states both `major` calls explicitly (`deprecated/rambo.py`
  per `versioning.md`; `gamma_ray` per ADR-0003) and notes they are
  absorbed by the project-level `version_bump: major`.

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

### Task 0.5: Ratify and execute ADR-0003 (`hazma.gamma_ray` removal)

**Task note:** [`../task-notes/phase-00/task-0.5-gamma-ray-decision.md`](../task-notes/phase-00/task-0.5-gamma-ray-decision.md)
**Depends on:** — (precedes Task 0.2's delete)

**Exit criteria:**

- ADR-0003 status flipped to Accepted by Logan (or, if rejected, the
  phase halts and the plan is revised — a rebuild is a *new feature*
  with no numerical oracle, since the module cannot run to produce a
  baseline; it would re-enter via `docs/followups/` with its own
  validation plan, not through this project).
- Confirmed (and recorded in the task note) that `hazma.spectra`'s
  n-body machinery (`spectra/_nbody.py` over `hazma.phase_space`)
  covers the documented `gamma`/`gamma_point` use cases.
- Docs referencing `hazma.gamma_ray` repointed to `hazma.spectra`.

**Notes:** The module is broken on import today (transitively imports
the deleted `hazma.rambo`), so no working user exists and no
behavior-preserving baseline can be captured — which is why rebuild is
not offered as an in-project branch (plan-review round 1).

## Exit Criteria

- All tasks complete; preflight green; CI green.
- `find hazma -name "*.pyx"` lists exactly the 20 surviving extension
  sources (8 `spectra/_photon` + 2 `spectra/_positron` + 3
  `spectra/_neutrino` + 6 mediator + `_utils/boost.pyx`), and a fresh
  `pip install -e .` builds exactly 20 `.so`s.
- No `language="c++"` extension remains; `git grep -l "std::"` over
  `hazma/` is empty.
- Phase learnings written to `../learnings/phase-00-dead-code-purge.md`.
