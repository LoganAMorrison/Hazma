# Charged-pion photon spectrum returns exactly zero in the forward cone

- **Added:** 2026-08-17
- **Source:** `projects/cython-to-rust/task-notes/phase-04/task-4.4-photon-pion.md`
- **Scope:** cross-cutting (public spectrum values)
- **Status:** inner pion repaired in parity-pinned-defect-repair Task 8;
  retained here until Task 12's coordinated follow-up closeout
- **Triggers / blockers:** none for the inner repair; Task 2 captured its
  independent Cython oracle before deletion of the twins.

## Why

The original angular integral over `[-1, 1]` could miss every nonzero
abscissa when only a narrow forward cone remained. QUADPACK then returned
an apparently converged zero. The defect was inherited from Cython.
At a pion energy of 1396 MeV, a 900 MeV photon returned exactly zero;
Task 8 restores **3.585860e-7 MeV^-1**, matching the independent capture.

The original investigation and tables are preserved at revision
`19c054e3b9a5df9ce1c29b6c91aaae3423674c42` in this file. Its proposed
post-deletion corpus regeneration was superseded by ADR-0001's declared
relations; the committed corpus is never regenerated.

## What

Task 8 clips the angular interval using the widest channel edge,
`(m_pi^2 - m_e^2)/(2 m_pi) = 69.784260 MeV`. The legacy muon edge at
69.783458 MeV is narrower and would remove the electron channel's last
sliver. Existing integration tolerances and channel expressions remain.

The before/after corpus capture changes 6,359 positions across the pion,
rho and mediator cases, with an `A3+B4` composite where scalar FSR was
already repaired. Details and verification:
[`task-8-charged-pion-cone.md`](../../../projects/parity-pinned-defect-repair/task-notes/task-8-charged-pion-cone.md).

The rho's *outer* integral independently loses support. That part of the
original investigation has its own open item:
[`rho-photon-outer-boost-misses-support.md`](rho-photon-outer-boost-misses-support.md).
Neither the A3 capture nor this repair claims to correct it.

## Entry points

- `rust/src/kernels/photon_pion.rs`: angular support and quadrature.
- `test/test_core_photon_pion.py`: independent boost and endpoint checks.
- `test/parity/test_pion_repair.py`: corpus zeros and scalar composition.
- `test/parity/deltas.py`: A3 and A3+B4 declarations.

## Risks / open questions

Task 9 must compose its rho rest-frame correction with A3: the daughter
pion is already boosted when the rho is at rest, so the position sets
are not disjoint. The outer rho repair is outside both tasks.
