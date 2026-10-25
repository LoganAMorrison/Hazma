# Visible axial spectra working memory

**Status:** In Progress
**Project:** visible-axial-spectra
**Date:** 2026-10-12

## Tasks

| # | Task | Depends on | Status | Task note |
| --- | --- | --- | --- | --- |
| 1 | Current and unit contracts | — | Complete | `task-1-contracts.md` |
| 2 | Axial tensors and LO three pion amplitudes | 1 | Complete | `task-2-lo-three-pion.md` |
| 3 | Rust port of the Task 1-2 numerics | 1, 2 | Complete | `task-3-rust-port.md` |
| 4 | Coherent longitudinal poles | 3 | Complete | `task-4-coherent-longitudinal-poles.md` |
| 5 | Rates and mediator model | 3, 4 | Complete | `task-5-rates-and-mediator-model.md` |
| 6 | Rest frame yields | 5 | Complete | `task-6-rest-frame-yields.md` |
| 7 | Polarized cascades | 6; 14 for model-generated pair densities | Complete | `task-7-polarized-cascades.md` |
| 8 | Resolved photons | 6; 7 for polarized boosts | Not started | — |
| 9 | Observation adapters | 6, 7; standalone `_f_eff_helper` PR | Not started | — |
| 10 | Fitted input overlays (optional) | 5; 6-9 for derived yields | Not started | — |
| 11 | Remaining hadronic channels | 5 | Not started | — |
| 12 | UV model cards | 3 | Not started | — |
| 13 | Validity and domain guards | 5, 11 | Not started | — |
| 14 | Mediator pair channels | 5, 12 | Not started | — |
| 15 | Invisible meson decays | 4, 5 | Not started | — |
| 16 | Benchmarks, documentation and closing | 6-9, 11-15 | Not started | — |

Tasks 4 and 12 have no dependency on each other and can run in parallel.

## Findings and decisions

- [ADR-0001](../adrs/ADR-0001-current-and-tensor-contracts.md) defines the
  normalization, units and source-longitudinal versus helicity distinction.
- [ADR-0002](../adrs/ADR-0002-rust-kernels-and-numpy-oracle.md) fixes the Rust
  kernels in `hazma._core.axial` as the compute path and keeps the NumPy
  implementation as the test oracle.
- [ADR-0003](../adrs/ADR-0003-proca-embedding-of-longitudinal-meson-poles.md)
  records the Proca embedding of the longitudinal meson poles (Task 4).
- Exact isospin LO three-pion coverage is distinct from a fitted physical
  charged/neutral mass prescription or a general singlet source.
- Scope extended on 2026-10-17 with Tasks 10-15 and gates for Tasks 3-9.
  Reference values and their note labels are in
  [`scope-and-acceptance.md`](../references/scope-and-acceptance.md). The
  version bump stays `minor`.
- Restructured on 2026-10-17: compute moves to Rust kernels behind `hazma._core`
  with Python as the public surface (`PLAN.md` Implementation layering). The new
  Task 3 ports the Task 1-2 numerics and keeps the NumPy code as the test
  oracle; former Tasks 3-15 are now Tasks 4-16.
- `rust/src/kernels/axial_yields.rs` is a module the plan's layering table
  did not name. It folds the three-pion parents and the muon-pair and
  three-pion decays into yields, while `polarized_muon` was the named module.
  The neutrino paths of `polarized_muon` are closed forms, and the adaptive
  quadrature stays as the test oracle and integrates the endpoint layer,
  where the closed forms' terms cancel.
- Task 7 adds `rust/src/kernels/polarized_cascade.rs`, whose `Cascade` and
  `Yields` impl fold polarized parents into spectra. The pair yield takes its
  branching fractions from `partial_widths`, so it inherits ADR-0004's
  refusal while an unmodeled final state is open.

## Numerical impact so far

The first cluster adds public primitives, Task 5 adds the
`AxialVectorMediator` model, Task 6 adds its public yields, and Task 7 adds
its polarized cascades; none of them changes an existing function. The
recorded existing-output grid contains 164 values and is bit-identical after
integration, including after the Task 3 Rust port and after Tasks 5, 6 and 7
(`check_public_impact.py`: maximum absolute difference 0.0). Reproduce it
with
`.venv/bin/python
projects/visible-axial-spectra/references/check_public_impact.py`.

## Verification

After Task 5 the model suite `test/axial_vector_mediator` passed 40 tests
and the `axial_rates` Rust unit tests passed 9. Path-scoped preflight passed
with 2,920 tests, 16 skips and 127 subtests. Exact commands and literal
summaries are in the completed task notes. The literal gate output for Tasks
1-2 is `projects/visible-axial-spectra/task-notes/preflight.txt`, for Task 4
it is `projects/visible-axial-spectra/task-notes/preflight-task-4.txt`, and
for Task 5 it is
`projects/visible-axial-spectra/task-notes/preflight-task-5.txt`.

After Task 6 the path-scoped preflight on the final tree passed with `2932
passed, 16 skipped, 20 warnings, 127 subtests passed`. The model suite
`test/axial_vector_mediator` has 52 tests, and `polarized_muon` and
`axial_yields` carry 25 Rust unit tests. The literal commands and the timings
(neutrino yields 2.36 s to 0.22 s over 40 energies, neutrino counts 24.6 s to
2.5 s over 300 bins at 500 MeV) are in `task-6-rest-frame-yields.md`.

After Task 7 the path-scoped preflight passed with `2957 passed, 16 skipped,
127 subtests passed`. The cascade suite
`test/axial_vector_mediator/test_cascades.py` has 24 tests, and
`polarized_cascade` and `axial_yields` carry 25 Rust unit
tests. The literal commands are in `task-7-polarized-cascades.md` and the
gate output is `projects/visible-axial-spectra/task-notes/preflight-task-7.txt`.

## Open questions

Later tasks must reconcile external photon source prescriptions and identify
missing singlet input directions. These do not block LO isovector currents.

Over 704 bins of 0.1 MeV just above the three-pion threshold, the `pi pi pi0`
e⁺ counts sum to `1 − 5.4e-8`, while 353 bins of 0.2 MeV sum to one within
1e-12. The cause is not understood, so later tasks should check fine-grid bin
sums at threshold.

## Handoff to next task

Read PLAN, ADR-0001 and
`projects/visible-axial-spectra/adrs/ADR-0002-rust-kernels-and-numpy-oracle.md`,
then the task notes and `references/scope-and-acceptance.md`. Task 8
(resolved photons) should read the handoff in
`task-7-polarized-cascades.md` first. It builds on
`polarized_cascade::Cascade` and its `Yields` impl: a new channel needs an
`Analyzer` (or, for a multibody decay, energy and alignment densities) and a
`Decay` arm. The public contract is the "Polarized cascades" section of
`docs/source/axial_vector_mediator.rst`. Binned three-pion photons are slow
because radiative charged-pion photons are integrated over each bin inside
the fold: about 20 ms per bin for 500 bins of 1.8 MeV and about 150 ms per
bin for wide bins, with timings that vary with machine load. Task 14 should
pass its threshold helicity fractions as `numpy.diag([F₊, F₀, F₋])`.

Earlier handoffs still apply. `polarized_muon.rs` takes the effective
polarization `P`: the μ⁻ helicity, which is minus the μ⁺ helicity, pinned by
`spinor_oracle.fermion_polarization`. Two follow-ups sit under
`docs/followups/todo/`: `axial-mediator-pi0-dalitz-positrons.md` and
`axial-mediator-polarized-radiative-muon-photons.md`. Tasks 11 and 14 remove
their final states from `AxialVectorMediator._unmodeled_thresholds` and add
their channels to `list_annihilation_final_states`; Task 11 also adds vector
quark couplings. The new `hazma._core.axial_vector_mediator` submodule is
exempt from the parity corpus through
`test/parity/cases.py::_CORE_NEW_PHYSICS_MODULES`. Task 15 can take
`AxialMesonPoles.invisible_widths` as its residue input; see "Handoff to Next
Task" in `task-4-coherent-longitudinal-poles.md`. Neutral poles require the
axial embedding and cannot use a spin-zero mediator Breit-Wigner replacement.
