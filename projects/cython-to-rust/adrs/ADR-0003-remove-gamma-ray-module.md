# ADR 0003: Remove the broken `hazma.gamma_ray` module

**Date:** 2026-08-03
**Status:** Accepted (signed off by Logan 2026-08-04)
**Scope:** Project-scoped (applies only within `projects/cython-to-rust/`).

## Context

`hazma/gamma_ray.py` is a public-named module that cannot be imported:
it (and the compiled `hazma._gamma_ray.gamma_ray_generator` it wraps)
does `from hazma import rambo`, and `hazma/rambo.py` was deleted long
ago (the export in `hazma/__init__.py` is commented out). Every current
release ships it in this broken state, so no working user code can
depend on it.

The original plan let Phase 00 Task 0.5 choose between deletion and
reimplementation over `hazma.phase_space`. Plan review (round 1)
correctly observed the reimplementation branch has no numerical oracle:
the module cannot run, so there is no behavior to pin a
"behavior-preserving rebuild" against, and the Phase 01 corpus cannot
supply a baseline because it postdates the deletion. A rebuild would
therefore be a *new feature* with a fresh validation burden — outside
this project's migration scope. Separately, the project's
`version_bump` is `major` regardless, because Phase 00 also deletes
`hazma/deprecated/rambo.py` and any removal from `hazma/deprecated/`
is `major` per `docs/versioning.md` — so deletion carries no
incremental versioning cost.

## Decision

Delete `hazma/gamma_ray.py` in Phase 00 (Task 0.2, gated on this ADR's
acceptance) along with the `hazma._gamma_ray` extensions. Do not ship
a shim. The module's public API and its replacement status:

- `gamma_ray_decay` — N-body decay photon spectrum: superseded by the
  live, tested `hazma.spectra.dnde_photon`
  (`hazma/spectra/_nbody.py` over `hazma.phase_space`).
- `gamma_ray_fsr` — Monte-Carlo FSR spectrum from a user-supplied
  matrix element: **removed without a direct replacement.** The
  nearest live equivalents are the Altarelli–Parisi approximations
  (`hazma.spectra.dnde_photon_ap_fermion` / `_ap_scalar`); a general
  `msqrd`-driven FSR generator would be a new feature, tracked at
  [`docs/followups/done/msqrd-driven-fsr-generator.md`](../../../docs/followups/done/msqrd-driven-fsr-generator.md).

(`gamma`/`gamma_point` are the compiled `_gamma_ray` names the module
wraps, not its public surface.) Docs that reference `hazma.gamma_ray`
are updated to point at `hazma.spectra`.

## Consequences

- **Positive:** removes an unimportable public module instead of
  shipping it broken again; Task 0.5 becomes an execution step rather
  than an open design question; no unpinnable "rebuild" enters the
  migration's parity scope.
- **Negative:** the name `hazma.gamma_ray` disappears — formally a
  breaking removal (`major`), and any downstream code that still
  *textually* imports it will fail at import with `ModuleNotFoundError`
  instead of today's confusing transitive error.
- **Mitigation:** the project is `major` already (deprecated-module
  deletion); the CHANGELOG names both removed functions with their
  replacement status (`gamma_ray_decay` → `hazma.spectra.dnde_photon`;
  `gamma_ray_fsr` → none; nearest: the Altarelli–Parisi
  approximations). If a maintained equivalent is ever wanted, it
  enters as a designed feature with its own validation plan —
  [`docs/followups/done/msqrd-driven-fsr-generator.md`](../../../docs/followups/done/msqrd-driven-fsr-generator.md)
  — not as part of this migration.

## Addendum (2026-08-04)

The follow-up above has since been implemented, ad-hoc and outside this
migration exactly as prescribed: `hazma.spectra.dnde_photon_fsr`
(repo-wide ADR-0001, with its own validation corpus). The Phase 00
CHANGELOG entry for this removal should therefore name
`hazma.spectra.dnde_photon_fsr` as `gamma_ray_fsr`'s replacement
instead of "none". The decision recorded here is unchanged.
