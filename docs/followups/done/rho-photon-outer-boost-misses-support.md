# Rho photon outer boost can miss its physical support

- **Added:** 2026-09-18
- **Source:** parity-pinned-defect-repair Task 8 scope audit; earlier
  measurement in cython-to-rust Task 4.5
- **Scope:** cross-cutting (public photon spectra)
- **Status:** done. Repaired as parity roster entry `C3`, the third
  label issued under
  [ADR-0003](../../adrs/ADR-0003-corpus-repairs-are-declared-deltas.md).
- **Triggers / blockers:** built on A3's inner-pion correction. B3's
  rest-frame correction is untouched: the rest-frame branch integrates
  nothing, so no `rest` array moves and `A3+B3` keeps its declarations.

> **Resolved.** `boosted` in `rust/src/kernels/photon_rho.rs` clips the
> boost window above at the rho's rest-frame photon endpoint, and
> returns `0.0` without a quadrature when the window starts above it.
> Both rho photon spectra move. See "Resolution (measured)" below.

## Why

Both rho photon spectra could return spurious tail zeros even after the
charged-pion angular integral sampled its support. Their outer energy
integral spans a wide interval while the integrand survives near only one
end. Every initial quadrature abscissa can miss that interval.

This was measured in
`projects/cython-to-rust/task-notes/phase-04/task-4.5-photon-rho.md`.
The A3 oracle in `test/parity/oracles/data/A3.npz` corrects the inner pion
only. Matching it is evidence for Task 8, not a proof of the outer boost.
This item splits the outer work from
[`charged-pion-photon-spectrum-misses-the-forward-cone.md`](charged-pion-photon-spectrum-misses-the-forward-cone.md),
so closing that repaired inner defect did not lose the remaining work.

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

## Resolution (measured)

**The support.** Each rho's integrand is exactly zero above a rest-frame
endpoint, which `photon_rho.rs` now computes from its daughters' own
support in `photon_pion.rs`. For the neutral rho that is the charged
pion's `π → e ν γ` edge, `(m_π² − m_e²)/(2 m_π)`, boosted into the rho
frame: **374.6256 MeV**. For the charged rho it is the higher of two
edges. The charged pion's continuum reaches **375.4681 MeV** and the
neutral pion's `γγ` box tops out at **374.6598 MeV**, so the continuum
sets it. Clipping at the box would have dropped the continuum's last
0.8 MeV. `charged_pion_photon_endpoint` and `neutral_pion_photon_box`
are new `pub` helpers in `photon_pion.rs`. The spectrum now uses the
box helper itself, so the box and its edge share one `f32`-rounded `β`.

**Tail onset.** Measured against an A3-corrected build (this branch's
base, `875bfd55`) on a 4,000-point scan of `[0.2, 0.999999]` of the
lab-frame endpoint `E'_max γ(1 + β)`. The table gives the last nonzero
point as a fraction of that endpoint. The two species agree to four
places before the repair, because A3 already fixed the inner pion:

| `E_ρ / m_ρ` | before | after |
| --- | --- | --- |
| 1.05 | 0.9980 | 1.0000 |
| 1.5 | 0.9874 | 1.0000 |
| 2 | 0.9726 | 1.0000 |
| 3 | 0.9332 | 1.0000 |
| 5 | 0.8260 | 1.0000 |
| 10 | 0.5369 | 1.0000 |

Task 4.5's pre-A3 figures differed only for the neutral rho, whose inner
pion was then losing its own cone.

**Integrated yield.** The flat boost carries an isotropic source's photon
energy into the lab multiplied by `γ` exactly, so
`∫ E dN/dE dE / (γ ∫ E' f(E') dE')` must be 1. The rest-frame energy is
383.4667 MeV for the charged rho and 2.4008 MeV for the neutral one:

| rho | `E_ρ / m_ρ` | before | after |
| --- | --- | --- | --- |
| charged | 1.05 | 0.99999703 | 1.00000083 |
| charged | 2 | 0.99873657 | 1.00000049 |
| charged | 10 | 0.64927667 | 1.00000016 |
| neutral | 1.05 | 1.00000000 | 1.00000000 |
| neutral | 2 | 1.00000001 | 1.00000001 |
| neutral | 10 | 0.99367480 | 1.00000003 |

The charged rho lost **35%** of its photon energy at `10 m_ρ`, most of it
the top of the `π⁰` box. The neutral rho's source is soft, so the same
lost fraction of its range carried 0.63% of its energy.

**Independent references.** `test/test_core_photon_rho.py` gains
`TestTheBoostedTail`. Its reference integrates over `u = ln E'` with
scipy, with the box edges as break points and at `epsrel = 1e-8`. It
shares the pion spectra with the kernel and nothing else. It agrees with
the kernel to **5.3e-6** worst from `1e-3` to `0.9999` of the lab
endpoint at `1.05`, `2` and `10 m_ρ`. That is the kernel's own requested
`epsrel = 1e-5`: the inner quadrature's noise caps any reference there,
and scipy reports roundoff above `1e-8`. The energy identity above is
pinned at `2` and `10 m_ρ`. `photon_rho.rs` pins the endpoints and the
lab-frame edge in `cargo`. The existing same-variable scipy transcription
now clips too and still matches at 1e-9. Removing the clip turns 14 of
the module's Python tests and one `cargo` test red.

**Corpus.** Measured before and after the repair on this tree. 382 of
the two cases' 2,790 pinned values move, 191 per case: 21 in
`near_rest`, 60 in `boosted_mild` and 110 in `boosted_strong`. Nothing
moves at `rest`, whose branch integrates nothing, or at `rest_plus_eps`,
where no grid point's window straddles the endpoint. Nine per case, one
in `boosted_mild` and eight in `boosted_strong`, were `0.0` and rise to
between 6.7e-15 and 1.6e-4 MeV⁻¹. The other 182 per case move by 1.5e-11
to 4.8e-2 relative, either way, where the narrower interval re-subdivides
a quadrature that had already found the support. No value falls by more
than 8.3e-4 relative.

The twelve moved arrays were already declared under `A3/nested`, so they
become the composite `A3+C3`. A3's capture predicts the unclipped boost
over the repaired pion. C3 adds the clipped boost minus the unclipped one,
both through scipy's QUADPACK at the kernel's tolerances over the live
pion kernels, following C2's pattern. The unclipped half reproduces the
A3 capture to 4.5e-15 at every boosted position, zeros included, and the
composition matches the repaired kernel to **6.9e-14** worst. Both are
far inside the rho cases' nested 1e-9 budget, which is unchanged.
`EXPECTED_DECLARED_ARRAYS` stays 407, and `test/parity/data/` is
untouched.

**Downstream.** The rho photon spectra reach the public surface through
`hazma.spectra.dnde_photon`'s `"rho"` and `"rho0"` final states, and
through `RHNeutrino`'s `ν ρ⁰` channel. At `m_N = 1`, `2` and `5` GeV that
channel moves 24, 89 and 174 of 400 log-spaced photon energies, pointwise
by up to 9.9e-2. Its tail extends: the last nonzero point at 5 GeV moves
from 2,174.8 to 2,368.6 MeV. Its photon energy per decay moves by at most
8.1e-7, because the neutral rho's boost there is mild. `RHNeutrino`'s
`ℓ ρ±` channel does not move, because `dnde_l_rho` evaluates the
`(ℓ, K)` final state. That is a separate defect, filed as
[`rhn-charged-rho-channel-evaluates-the-kaon.md`](../todo/rhn-charged-rho-channel-evaluates-the-kaon.md).
No mediator model decays through a rho, and no other corpus case moves.
