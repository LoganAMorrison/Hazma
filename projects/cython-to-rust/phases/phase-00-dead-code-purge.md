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
module (ADR-0003, Accepted 2026-08-04).
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
- All five `include "../_decay/parameters.pxd"` sites in **built**
  extensions repointed: the four live ones (both
  `scalar_mediator/*_spec*.pyx`, both `vector_mediator/*_spec*.pyx`)
  plus `_gamma_ray/gamma_ray_generator.pyx`, which is a live
  `Extension` in `setup.py` until Task 0.2 deletes it — skipping it
  breaks the build immediately. The two unbuilt `_decay/*.pyx` sites
  (`decay_electron.pyx`, `_decay_muon_bak.pyx`, spelled
  `include "parameters.pxd"`) are repointed too so no `.pyx`/`.pxd` in
  the tree carries a dangling include; `.pyx.bak` files are left alone.
- `pip install -e .` rebuilds; import smoke passes. Note
  `_gamma_ray.gamma_ray_generator` compiles but has never been
  importable (`from hazma import rambo`), so it is excluded from the
  smoke set — see ADR-0003.
- Do **not** merge values into `_utils/constants.pxd` (rules.md rule 4).

### Task 0.2: Delete the phase-space / gamma-ray slice

**Task note:** [`../task-notes/phase-00/task-0.2-delete-mc-slice.md`](../task-notes/phase-00/task-0.2-delete-mc-slice.md)
**Depends on:** Task 0.1 (ADR-0003 accepted 2026-08-04)

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
  Note: `hazma.utils`'s `cross_section_prefactor` builds the incoming
  momentum from `kallen_lambda`, which cancels at threshold, while the
  Cython twin used the factored form. The swap is therefore a
  **declared** numerical change near threshold (≤2.1e-7 relative within
  1e-7 of it; ≤5e-15 at `cme ≥ 1.1 ×` threshold) — record it in
  `task-notes/README.md` "Numerical impact so far", do not absorb it
  silently.
- The three config files that name the deleted sources are updated in
  **this** task, because the build and the test collection break the
  moment the sources go: `setup.py`'s `_positron` / `_neutrino` /
  `field_theory_helper_functions` extension groups; `test/conftest.py`'s
  unconditional `THIS_DIR.joinpath("decay").iterdir()`; and the three
  `hazma._decay.interpolation_data.*` entries in `pyproject.toml` and
  `MANIFEST.in`. Task 0.4 keeps the rest.
- Import smoke (`hazma.theory`, `hazma.limits`, `hazma.cmb`,
  `hazma.pbh`, both mediators, `hazma.spectra._photon._muon`) passes.

### Task 0.4: Prune build and packaging config

**Task note:** [`../task-notes/phase-00/task-0.4-prune-build.md`](../task-notes/phase-00/task-0.4-prune-build.md)
**Depends on:** Tasks 0.2, 0.3

**Exit criteria:**

- `setup.py` extension list matches the survivors exactly (count the
  built `.so`s in a fresh `pip install -e .`); no `cpp=True` groups
  remain. Tasks 0.2 and 0.3 each drop their own groups as they delete
  the sources, so what remains here is the `_gamma_ray` /
  `_phase_space` pair plus the final reconciliation against the
  survivor count.
- `pyproject.toml` package-data globs and `MANIFEST.in` no longer ship
  deleted directories; sdist builds and contains no deleted paths.
  (Task 0.3 already removed the `hazma._decay.interpolation_data.*`
  entries; this task confirms nothing else dangles and runs the sdist.)
- CI green on the full matrix.

### Task 0.5: Execute ADR-0003 (`hazma.gamma_ray` removal)

**Task note:** [`../task-notes/phase-00/task-0.5-gamma-ray-decision.md`](../task-notes/phase-00/task-0.5-gamma-ray-decision.md)
**Depends on:** — (precedes Task 0.2's delete)

**Exit criteria:**

- ~~ADR-0003 status flipped to Accepted by Logan~~ — **done
  2026-08-04.** A rebuild stays out of scope (a *new feature* with no
  numerical oracle, since the module cannot run to produce a baseline);
  it re-enters with its own validation plan via
  [`../../../docs/followups/done/msqrd-driven-fsr-generator.md`](../../../docs/followups/done/msqrd-driven-fsr-generator.md),
  filed 2026-08-04.
- Replacement status of the module's actual public API confirmed and
  recorded in the task note: `gamma_ray_decay` is superseded by
  `hazma.spectra.dnde_photon` (the n-body path in `spectra/_nbody.py`
  over `hazma.phase_space`); `gamma_ray_fsr` (Monte-Carlo FSR from a
  user `msqrd`) has **no direct replacement** — the nearest live
  equivalents are the Altarelli–Parisi approximations
  (`hazma.spectra.dnde_photon_ap_{fermion,scalar}`), and the removal
  is declared as replacement-free for the general-`msqrd` case in the
  CHANGELOG. (`gamma`/`gamma_point` are the *compiled* names the
  module wraps, not its public API.)
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
