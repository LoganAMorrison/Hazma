---
status: In Progress
phased: false
version_bump: minor
deliverable: Neutral visible axial rates and stable particle spectra
created: 2026-10-12
---

# Visible axial spectra

**Structure:** Flat task list.

## Goal

Implement declared neutral visible axial currents, exclusive rates and
normalized photon, positron and neutrino yields in maintained Hazma.
Preserve current normalization, longitudinal pole physics and spin information
so the same model cards can support indirect and laboratory observables.

## Scope

The initial implementation adds current and unit contracts, covariant axial
tensors and the exact isospin LO three-pion current. Later tasks add coherent
neutral poles, model rates, secondary yields, polarized boosts and observation
adapters. Experimental mediator classes are excluded.

General fitted NLO singlet/strange sources, high-energy fragmentation,
standalone thermal evolution and laboratory matching are separate deliverables.
Unknown source directions must remain explicit; tau normalization alone does
not determine their differential shapes.

## Numerical impact

New additive public functions under `hazma.form_factors.axial` return declared
MeV quantities. Existing public outputs are expected to remain unchanged and
are measured at the first implementation. The closing task carries the minor
version bump; this initial task cluster does not close the project.

## Orientation

- [Current and tensor conventions](adrs/ADR-0001-current-and-tensor-contracts.md).
- [LO three-pion derivation](references/lo-three-pion.md).
- [Task status](task-notes/README.md).

## Task details

### Task 1 Current and unit contracts

Provide immutable light-current, matching and source-coverage cards, normalized
isovector/octet/singlet decomposition and explicit energy-power conversion.

Exit criteria:

- Traceful currents round-trip without dropping the singlet.
- Scalar and array conversion preserves rates, widths and spectral density.
- Input validation rejects undefined scales and nonfinite values.

### Task 2 Axial tensors and LO three pion amplitudes

Provide covariant Hermitian tensors, source projections and charged/neutral
three-pion currents including the pion-pole term in the exact isospin limit.
The declared LO support ends at 600 MeV by default; this is a restricted model
range rather than a precision hadronic claim. Unsupported flavor directions
raise instead of silently projecting away the original current.

Exit criteria:

- Independent current/Ward identities, symmetry and Lorentz checks pass.
- Integrated source densities agree with independent phase-space normalization.
- On-shell widths use transverse density; neutral three-pion longitudinal
  density remains distinct from physical helicity zero.
- Thresholds, shapes, broadcasting and unsupported input paths are tested.
- Full preflight and durable documentation checks pass.

### Task 3 Coherent longitudinal meson poles

Derive the Proca/current/Goldstone embedding and reuse only meson/cut machinery.
Verify Ward/contact cancellation, Schur complement, shared-cut positivity and
single invisible-cut insertion. Do not substitute a scalar portal propagator.

### Task 4 Rates and mediator model

Expose declared partial widths, sigma, sigma v, thresholds and lines through
TheoryAnn. Check threshold limits and separate particle from population factors.

### Task 5 Rest frame stable particle yields

Convolve signed parent histories using public decay kernels. Pin e+ only
counts, neutrino flavor ordering and primary stable lines independently.

### Task 6 Polarized cascade marginals

Use validated production density or an explicit supplied density. Check
angular normalization, boosts, endpoints and energy/multiplicity conservation.

### Task 7 Resolved radiative photons

Add invariant and polarized event paths with Ward/soft checks and explicit
infrared cut, uncertainty and overlap bookkeeping. State missing virtual terms.

### Task 8 Observation adapters

Connect gamma limits and CMB deposition with explicit kinetic-energy table
arguments, charge-summed injection and symmetric-Dirac population counting.

### Task 9 Fitted input overlays

Propagate measured covariance and declared matching/truncation directions.
Separate inclusive normalization from differential shape validation.

## Related

The research integration plan and independent package comparisons live in the
companion research repository at commit `3d9c6c7`. They motivate this project;
the runtime must not import that repository or depend on its version folders.
