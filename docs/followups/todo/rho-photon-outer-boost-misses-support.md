# Rho photon outer boost can miss its physical support

- **Added:** 2026-09-18
- **Source:** parity-pinned-defect-repair Task 8 scope audit; earlier
  measurement in cython-to-rust Task 4.5
- **Scope:** cross-cutting (public photon spectra)
- **Status:** open
- **Triggers / blockers:** build on A3's inner-pion correction; coordinate
  with Task 9's separate rho rest-frame normalization correction.

## Why

Both rho photon spectra can return spurious tail zeros even after the
charged-pion angular integral samples its support. Their outer energy
integral spans a wide interval while the integrand survives near only one
end. Every initial quadrature abscissa can miss that interval.

This was measured in
`projects/cython-to-rust/task-notes/phase-04/task-4.5-photon-rho.md`.
The A3 oracle in `test/parity/oracles/data/A3.npz` corrects the inner pion
only. Matching it is evidence for Task 8, not a proof of the outer boost.
This item splits the still-open outer work from
[`charged-pion-photon-spectrum-misses-the-forward-cone.md`](charged-pion-photon-spectrum-misses-the-forward-cone.md),
so closing that repaired inner defect does not lose the remaining work.

## What

Restrict each rho's outer interval to the support of its constituent
channels, with an independent energy-variable reference and endpoint
checks. Measure all downstream public changes and declare their corpus
relations without regenerating the historical arrays or widening budgets.
Account for both the neutral-pion line and charged-pion continuum in the
charged rho. Keep the rest-frame B3 correction separate and compose any
overlapping declarations.

## Entry points

- `rust/src/kernels/photon_rho.rs`: `boosted` and daughter spectra.
- `test/test_core_photon_rho.py`: physics invariants.
- `test/parity/deltas.py`: A3 and future B3 composition.
- `projects/parity-pinned-defect-repair/PLAN.md`: Tasks 8 and 9.

## Risks / open questions

The committed A3 oracle cannot predict this additional correction; the
Cython twins are gone. Establish an independent quadrature reference
before changing the kernel. Re-measure tail onset and integrated yield
against an A3-corrected build rather than copying old defect magnitudes.
