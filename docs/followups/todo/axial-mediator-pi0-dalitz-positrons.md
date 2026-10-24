# The axial mediator's three-pion yields omit the π⁰ Dalitz e±

- **Added:** 2026-10-23
- **Source:** `projects/visible-axial-spectra/` Task 6, deferred by user
  decision
- **Scope:** project
- **Status:** open
- **Triggers / blockers:** a positron or CMB constraint on
  `AxialVectorMediator` that leans on the `pi0 pi0 pi0` channel, whose
  only e± source is this decay, or a comparison of the `pi pi pi0`
  e± yield below the muon-decay continuum, where the Dalitz pairs sit.

## Why

`AxialVectorMediator.rest_frame_yields` and `rest_frame_counts` give the
e⁺ and e± yields of the three-pion final states from the charged pions'
`π → μ → e` and `π → e` chains alone. The neutral pion's Dalitz decay
`π⁰ → γ e⁺ e⁻`, with branching fraction 1.17%, is left out. For
`pi0 pi0 pi0` that is the whole e± yield, so the model reports an exact
zero where the physical yield is about 0.035 e⁺ per annihilation. For
`pi pi pi0` it adds 0.0117 e⁺ per annihilation to a yield of about one,
1.2%, concentrated at low energies, where it may be a larger share of the
continuum.

Hazma has no π⁰ Dalitz e± kernel to reuse: `hazma.spectra._positron`
has no neutral-pion function.

## What

Add a Dalitz e± kernel in `rust/src/kernels/axial_yields.rs`, or in a
sibling module it imports: the Kroll-Wada pair-mass spectrum folded with
the three-body `π⁰ → γ e⁺ e⁻` kinematics gives the rest-frame e±
density, which the existing boost helpers of `polarized_muon` then fold
with `ThreePionParents::spectator_density` for the charged channel and
with the single-pion density for the neutral one. Each π⁰ contributes
one e⁺ and one e⁻ per Dalitz decay, so the `"positron"` product gains
`BR(γ e⁺ e⁻)` per π⁰ and the charge-summed product twice that. Pin the
count per π⁰ against the branching fraction and the pair-mass spectrum
against the Kroll-Wada form at a few masses. Record the moved
`pi0 pi0 pi0` and `pi pi pi0` e± yields in `CHANGELOG.md` under
`Changed`, since the published zero becomes a number.

## Entry points

- `rust/src/kernels/axial_yields.rs`: `Channel::charged_leptons`,
  `ThreePionParents::charged_pion_daughters` and the module's `# Decays`
  section, which records the omission.
- `hazma/axial_vector_mediator/_model.py`: `rest_frame_yields`, whose
  docstring states that the Dalitz pairs are not included.
- `projects/visible-axial-spectra/task-notes/task-6-rest-frame-yields.md`:
  the decision and its reasoning.

## Risks / open questions

The radiative corrections to the Dalitz spectrum (Mikaelian and Smith)
change its shape by a few percent; the leading-order Kroll-Wada form is
enough for the first version if the docstring says so.
