# The muon photon rest-frame endpoint used the wrong power of r

- **Added:** 2026-08-16
- **Source:** `projects/cython-to-rust/` Phase 04 Task 4.3.
- **Scope:** commit; Rust endpoint guard and declared corpus delta.
- **Status:** repaired by `parity-pinned-defect-repair` Task 7; retained in
  `todo/` for Task 12's coordinated move and inbound-link sweep.
- **Triggers / blockers:** none. Task 2 captured the corrected Cython
  oracle before the twin was deleted; Task 7 consumes that capture.

## Defect and repair

The rest-frame branch used `y >= 1 - sqrt(r)`, where
`y = 2 E_gamma / m_mu` and `r = (m_e/m_mu)^2`. The boosted branch already
used `1 - r`. This removed the last 0.254263793 MeV of rest-frame support.
[Kuno and Okada, hep-ph/9909265v1, Eq. (53)](https://arxiv.org/pdf/hep-ph/9909265)
confirms that `1 - r` is the endpoint; `1 - sqrt(r)` marks a change in the
allowed electron-energy range.

`rust/src/kernels/photon_muon.rs` now uses `ONE_MINUS_R` on both branches.
The obsolete endpoint constant and test-only rest-frame copy were removed.
The boost-integral test now integrates the production rest-frame kernel.
`test/parity/deltas.py` declares four moved positions of
`spectra.photon.muon/rest/values` against the Task 2 corrected Cython
capture. The historical corpus arrays and existing budgets are unchanged.

## Signed approximation

The paper's Eqs. (54)–(56) neglect mass-suppressed terms. The expression
becomes negative over the final 0.0197755 MeV below the endpoint, reaching
-6.43368e-9 MeV^-1. Task 7 retains and documents it to keep the rest and
boosted formulas consistent. This is an approximation limitation, not a
physical negative photon yield; see
[ADR-0002](../../../projects/parity-pinned-defect-repair/adrs/ADR-0002-retain-the-signed-muon-endpoint-approximation.md).

The restored interval adds a net 5.44538e-8 photons per decay. Its
negative part contributes -8.97482e-11 photons per decay. The six composed
corpus cases do not move: their pion daughter muons have energy
109.778 MeV, and the mediator corpus uses daughter energies at least
125 MeV, so none reaches the rest branch. The general two-body spectrum
with two muons at production threshold does move; above threshold both
muons are boosted and the repair does not act.

## Verification and remaining lifecycle work

- `test/test_core_photon_muon.py` checks the physical endpoint, restored
  interval, signed J+/J- reference values, and scalar/array agreement.
- `rust/src/kernels/photon_muon.rs` retains the boost-integral identity
  and boundary/NaN tests.
- [Task 7 evidence](../../../projects/parity-pinned-defect-repair/task-notes/task-7-photon-muon-endpoint.md)
  records before/after grids, mutation checks, and the preflight gate.
- Task 12 moves this repaired record to `done/` and pins the revision.
