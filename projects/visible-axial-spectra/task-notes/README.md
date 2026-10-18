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
| 5 | Rates and mediator model | 3, 4 | Not started | — |
| 6 | Rest frame yields | 5 | Not started | — |
| 7 | Polarized cascades | 6; 14 for model-generated pair densities | Not started | — |
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

## Numerical impact so far

The first cluster adds public primitives. The recorded existing-output grid
contains 164 values and is bit-identical after integration, including after
the Task 3 Rust port. Reproduce it with
`.venv/bin/python
projects/visible-axial-spectra/references/check_public_impact.py`.

## Verification

After Task 4 the focused axial suite passed 324 tests and the Rust crate 324
tests, with clippy `-D warnings` clean. Path-scoped preflight passed with
2,880 tests, 16 skips and 127 subtests. Exact commands and literal summaries
are in the completed task notes. The literal gate output for Tasks 1-2 is
`projects/visible-axial-spectra/task-notes/preflight.txt`, and for Task 4 it
is `projects/visible-axial-spectra/task-notes/preflight-task-4.txt`.

## Open questions

Later tasks must reconcile external photon source prescriptions and identify
missing singlet input directions. These do not block LO isovector currents.

## Handoff to next task

Read PLAN, ADR-0001 and
`projects/visible-axial-spectra/adrs/ADR-0002-rust-kernels-and-numpy-oracle.md`,
then the task notes and `references/scope-and-acceptance.md`. Task 5 should
read ADR-0003 first: its rates must call `AxialMesonPoles.mixing_expansion`,
reject points above `MIXING_GUARD`, feed
`AxialFormFactorGammaGamma.spectral_functions` into the contact, and pass an
`InvisibleCut` when dark matter can be pair-produced below a pole. Task 15 can
take `AxialMesonPoles.invisible_widths` as its residue input. See "Handoff to
Next Task" in `task-4-coherent-longitudinal-poles.md`. Neutral poles require
the axial embedding and cannot use a spin-zero mediator Breit-Wigner
replacement.
