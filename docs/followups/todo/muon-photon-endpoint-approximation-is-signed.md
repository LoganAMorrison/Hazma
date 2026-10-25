# the radiative-muon photon spectrum is negative just below its endpoint

- **Added:** 2026-10-24
- **Source:** PR #132 review round 2 (visible-axial-spectra Task 7)
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** a finite-mass calculation of the `μ → e ν ν̄ γ`
  photon spectrum. Resolving this item supersedes
  [ADR-0002](../../../projects/parity-pinned-defect-repair/adrs/ADR-0002-retain-the-signed-muon-endpoint-approximation.md)
  of `parity-pinned-defect-repair`.

## Why

`dnde_photon_muon` implements Kuno and Okada's Eqs. (54)–(56)
(hep-ph/9909265), which neglect mass-suppressed terms. ADR-0002 keeps
that signed expression on both the rest-frame and in-flight branches,
because flooring one branch would break the boost-integral identity
between them. The ADR names a nonnegative replacement as separate work,
and this item tracks it.

The tail reaches every caller. Measured with `hazma.spectra.dnde_photon_muon`
on a geometric scan of the last 1e-2 below each endpoint:

| Muon energy (MeV) | Negative over the last | Minimum (MeV⁻¹) | Negative part (photons per decay) |
| --- | --- | --- | --- |
| at rest (`m_μ(1 + 1e-9)`) | 0.022 MeV, 4.2e-4 of the endpoint | −6.3e-9 | −8.9e-11 |
| 150 | 0.080 MeV, 6.3e-4 | −8.4e-13 | −4.2e-14 |
| 475.19 | 0.294 MeV, 6.3e-4 | −1.9e-13 | −3.6e-14 |
| 1000 | 0.625 MeV, 6.3e-4 | −9.0e-14 | −3.5e-14 |

The yields are negligible, but a negative density and a non-positive
endpoint bin are visible to any caller that probes the edge. PR #132's
`muon_photons_reach_their_lab_endpoint` therefore asserts that its oracle
is nonzero rather than positive.

## What

1. Derive or adopt a finite-mass `μ → e ν ν̄ γ` photon spectrum that stays
   nonnegative up to the kinematic endpoint `y = 1 − r`, `r = (m_e/m_μ)²`.
   Pin it against an independent evaluation at high precision, as
   ADR-0002 pinned the signed form against the J+/J- representation.
2. Implement it on both branches of `rust/src/kernels/photon_muon.rs`
   (`dnde_photon_muon_rest_frame` and `dnde_photon_muon`), keeping the
   boost-integral identity between them.
3. Reassess every caller: `hazma.spectra.dnde_photon_muon` and its
   docstring, `photon_pion`, `photon_rho`, `scalar_decay_photon`,
   `mediator_tables`, `axial_yields` (`Channel::MuonPair`) and
   `polarized_cascade` (`Cascade::muon_photons`).
4. Write the superseding ADR and set `superseded_by` on ADR-0002. Declare
   in `test/parity/deltas.py`, under a new repair label, every
   parity-corpus position that moves beyond its case's budget
   ([ADR-0003](../../adrs/ADR-0003-corpus-repairs-are-declared-deltas.md)).
5. Tighten `muon_photons_reach_their_lab_endpoint` in
   `rust/src/kernels/polarized_cascade.rs` to assert a positive oracle.

## Entry points

- `rust/src/kernels/photon_muon.rs`: the signed-approximation note on
  `dnde_photon_muon_rest_frame` and the test
  `the_in_flight_signed_endpoint_residual_is_bounded`.
- `hazma/spectra/_photon/__init__.py`: the public docstring's statement
  of the limitation.
- `projects/parity-pinned-defect-repair/adrs/ADR-0002-retain-the-signed-muon-endpoint-approximation.md`
- `docs/followups/done/photon-muon-rest-frame-endpoint-uses-the-wrong-power-of-r.md`
- `test/test_core_photon_muon.py`
- `rust/src/kernels/polarized_cascade.rs`:
  `muon_photons_reach_their_lab_endpoint`.

## Risks / open questions

- A nonnegative form moves published spectra near every muon photon
  endpoint, by about 1e-10 photons per decay at rest and 1e-13 in flight.
  Whether any parity-corpus position moves beyond its budget, and so
  needs a declaration, is to be measured.
- `docs/followups/todo/axial-mediator-polarized-radiative-muon-photons.md`
  also replaces the muon photon kernel, for polarized muons. The two items
  should share one finite-mass calculation rather than land two
  replacements.
- Whether a published finite-mass result exists in a form Hazma can
  implement in closed form, or whether the spectrum needs a numerical
  phase-space integration, is open.
