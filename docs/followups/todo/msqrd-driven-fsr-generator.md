# A maintained `msqrd`-driven Monte-Carlo FSR generator

- **Added:** 2026-08-04
- **Source:** `projects/cython-to-rust/adrs/ADR-0003-remove-gamma-ray-module.md`
  (Accepted 2026-08-04) — the ADR removes `hazma.gamma_ray.gamma_ray_fsr`
  and names this follow-up as the only route back
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** ripens when a user (or a model in this repo)
  actually needs FSR from a general squared matrix element and the
  Altarelli–Parisi approximations are not good enough. Technically
  blocked on nothing once `projects/cython-to-rust/` Phase 00 lands;
  deliberately *not* part of that migration, because it is a new
  feature with a fresh validation burden rather than a port.

## Why

ADR-0003 deletes `hazma/gamma_ray.py`. Of its two public functions,
`gamma_ray_decay` is superseded by the live, tested
`hazma.spectra.dnde_photon` (the n-body path in `hazma/spectra/_nbody.py`
over `hazma.phase_space`), but `gamma_ray_fsr` is **removed with no
direct replacement**. It computed a final-state-radiation photon
spectrum by Monte-Carlo integration over an (N+1)-body phase space using
a user-supplied squared matrix element:

```python
gamma_ray_fsr(photon_energies, cme, isp_masses, fsp_masses,
              non_rad, msqrd, nevents=1000)
```

The nearest live equivalents are the Altarelli–Parisi approximations
`hazma.spectra.dnde_photon_ap_fermion` / `dnde_photon_ap_scalar`. Those
are collinear/soft approximations keyed to a splitting function; they do
not accept an arbitrary `msqrd` and they are not valid across the full
photon-energy range for every process. So the deletion does narrow what
the library can do, even though nobody can be using the removed function
today (the module has been broken on import in every shipped release —
it transitively imports the long-deleted `hazma.rambo`).

This is filed as a follow-up rather than reinstated in the migration
because the module cannot run, so there is **no numerical oracle**: no
baseline can be captured from the old code to pin a "behavior-preserving
rebuild" against, and the cython-to-rust parity corpus (Phase 01)
postdates the deletion. Anything built here is therefore new physics
code that has to earn its own validation, which is exactly what the
migration's parity discipline is designed *not* to absorb.

## What

Design and implement a maintained FSR generator as a first-class feature
of `hazma.spectra`, not as a resurrection of the old module:

1. **Decide the surface.** Most likely a `dnde_photon_fsr(...)` (or a
   `hazma.spectra` entry point named to match the existing
   `dnde_photon_*` family) taking photon energies, CME, initial- and
   final-state masses, the non-radiative cross section or width, and a
   `msqrd` callable over the radiative phase-space point. Follow the
   repo's arrays-in/arrays-out contract and state units for every
   argument and for the returned `dN/dE` (MeV⁻¹).
2. **Build it on `hazma.phase_space`.** RAMBO and the three-body helpers
   already live there (`hazma/phase_space/_rambo.py`,
   `_three_body.py`); the old module's own Monte-Carlo layer is the part
   that rotted. Do not reintroduce a private phase-space implementation.
3. **Write the validation plan first.** Candidate oracles, in rough
   order of strength: (a) analytic FSR spectra for
   `chi chi -> l+ l- gamma` in a mediator model, which several `hazma`
   models already provide in closed form — a Monte-Carlo run should
   converge to those within its statistical error; (b) the
   Altarelli–Parisi approximations in the collinear limit where they are
   known to be valid; (c) the soft-photon theorem as `E_gamma -> 0`.
   Pin at least one, state the tolerance and the reason for it, and
   record the Monte-Carlo seed so the test is deterministic.
4. **Statistical error is part of the API.** The old function exposed
   only `nevents`. Decide whether the new one returns an error estimate
   or documents the scaling; a spectrum whose noise floor is invisible
   to the caller is a trap in limit-setting code.
5. **Check the kinematic edges** — threshold, the photon endpoint at
   `E_gamma -> (s - (Σm)²)/(2√s)`, equal masses, and the massless limit.
   The old implementation's behavior there was never characterized.

If, after scoping, the answer is that the Altarelli–Parisi
approximations cover every real use, close this out by moving it to
`done/` with that conclusion recorded — a documented "no" is a valid
resolution.

## Entry points

- `projects/cython-to-rust/adrs/ADR-0003-remove-gamma-ray-module.md` —
  the decision that created this gap, with the replacement-status table.
- `hazma/gamma_ray.py` — the removed implementation; read it from git
  history after `projects/cython-to-rust/` Phase 00 Task 0.2 deletes it
  (`git show <pre-deletion-ref>:hazma/gamma_ray.py`), together with
  `hazma/_gamma_ray/gamma_ray_generator.pyx`, which held the compiled
  `gamma` / `gamma_point` kernels it wrapped.
- `hazma/spectra/altarelli_parisi.py` — the nearest live equivalents
  (`dnde_photon_ap_fermion`, `dnde_photon_ap_scalar`).
- `hazma/spectra/_nbody.py`, `hazma/phase_space/` — the live n-body and
  RAMBO machinery any rebuild should sit on.
- Related project: `projects/cython-to-rust/` — Phase 00 Task 0.5
  executes ADR-0003; the CHANGELOG entry there declares
  `gamma_ray_fsr` replacement-free and should link here.

## Risks / open questions

- **Is it wanted at all?** No working user code can depend on the
  removed function, so there is zero demonstrated demand. Do not build
  this speculatively — wait for a concrete use case, and let that use
  case pick the oracle in step 3.
- **Language and phase.** If this lands after the Cython→Rust migration
  completes, the Monte-Carlo inner loop is a candidate for
  `hazma._core` rather than Python. That is a performance decision to
  make against a measured profile, not up front.
- **Interface drift.** The old signature took `non_rad` (the
  non-radiative rate) as a bare float. Whether the new surface should
  instead take the non-radiative process itself, and whether it should
  accept a `Theory` model directly, is an open design question that
  should be settled before any code is written.
