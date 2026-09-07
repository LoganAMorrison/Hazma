# The charged pion's neutrino continuum loses its quadrature support

- **Added:** 2026-09-06
- **Source:** `projects/parity-pinned-defect-repair/task-notes/task-10a-neutrino-pion-line.md`
- **Scope:** cross-cutting (public spectrum values)
- **Status:** open
- **Triggers / blockers:** **corpus re-pinning only.** The Cython twin
  (`hazma/spectra/_neutrino/_pion.pyx`) was deleted in `cython-to-rust`
  Task 4.6, so there is no oracle to capture and no deletion wave to beat.
  The corrected values need none: the repair narrows an integration
  window to the support the integrand already declares, and the
  difference is a `scipy.integrate.quad` over the narrowed window, which
  runs against the committed arrays with no build. Belongs in
  [`projects/parity-pinned-defect-repair/PLAN.md`](../../../projects/parity-pinned-defect-repair/PLAN.md)
  alongside the other pinned defects, as a roster entry of its own.

## Why

`hazma.spectra.dnde_neutrino_charged_pion` builds its muon-decay
continuum by boosting the muon's own neutrino spectrum out of the pion
frame,

```text
dN/dE = 1/(2 β γ) ∫_{γE(1−β)}^{γE(1+β)} dE'  (dN/dE')_μ(E', E_μ^rf) / E' ,
```

and integrates over the whole window. But the integrand is zero above the
muon-decay endpoint in the pion rest frame — **69.783500 MeV**, measured
on a 200,001-point sweep of `dnde_neutrino_muon(·, ENG_MU_PI_RF)`, for
both flavor rows — so for a boosted pion almost all of that window
contributes nothing:

| `E_π` (MeV) | `E_ν` (MeV) | window (MeV) | width / support |
| --- | --- | --- | --- |
| 1395.7039 | 798.21 | [40.01, 1.592e+04] | 228x |
| 1395.7039 | 1369.1 | [68.63, 2.731e+04] | 390x |
| 5000 | 412.098 | [5.753, 2.952e+04] | 423x |

QUADPACK's first 21-point rule then places every abscissa outside the
support, the error estimate agrees that the constant zero it sampled is
exact, and `quad` returns `0.0` with a successful status. This is the
same failure mode as
[`charged-pion-photon-spectrum-misses-the-forward-cone.md`](charged-pion-photon-spectrum-misses-the-forward-cone.md)
(roster entry A3), one kernel over: there the lost support is a cone in
`cos θ`, here it is the tail of a boost window.

**What is lost is not a tail correction.** Re-integrating over the window
clipped at the endpoint recovers, at the three points above,
`3.047944e-04`, `1.535789e-08` and `5.037573e-04` MeV⁻¹ against a shipped
electron row of `8.857273e-08`, `8.857273e-08` and `2.460992e-08` — a
factor of 3,441, 0.17 and 20,470. Where the continuum is lost entirely
the spectrum is the prompt `π → e ν_e` line and nothing else, which is
three to four decades below the truth.

The parity corpus pins 14 such positions: 13 in the `values` array of the
`boosted_strong` block of `spectra.neutrino.charged_pion`
(`E_π = 1395.7039` MeV, over `E_ν ∈ [798.21, 1369.1]` MeV) and 1 in that
block's `scalar_values`. They are inside the 215 positions
`test/parity/deltas.py` declares for roster entry B5, and they are
exactly the positions where that declaration's relative drop is
`0.500000000000` — the doubled line was the whole value because the
continuum was zero.

Both implementations lose the window the same way, which is why parity is
green: `scipy.integrate.quad` over the unclipped window returns `0.0` too.
The defect is in the interval, not in the integrator.

## What

Clip the boost window to the integrand's own support before integrating,
in `rust/src/kernels/neutrino_pion.rs`'s `boost_window` or at its use in
`dnde_mu_numu`. **The sibling kernel already does this**, which is both
the precedent and the strongest evidence the omission is a slip:
`rust/src/kernels/positron_pion.rs:163-164` computes

```rust
let emin = (gamma * (-beta).mul_add(k, e)).max(ME);
let emax = (gamma * beta.mul_add(k, e)).min(EMAX_PI_RF);
```

— both ends clamped to the integrand's support. The neutrino twin clamps
only `emin`, and only against zero.

Pick the clip from the muon spectrum's endpoint rather than from the
`π → e ν_e` line's energy: they differ by 7.6e-4 MeV
(69.783500 against `ENU_E_PI_RF = 69.784260`), and it is the narrower one
that bounds the integrand. That is the same 8e-4 MeV distinction the A3
follow-up had to settle between `ENG_GAM_MAX_PIRG` and
`(m_π² − m_e²)/(2 m_π)`.

Then re-declare: the B5 entry in `test/parity/deltas.py` currently covers
those 14 positions with an `Additive` relation whose term is the line
alone. Once the continuum comes back they move again, by a different
mechanism, so the two either compose into one declaration or the corpus
positions have to be split between them —
`projects/parity-pinned-defect-repair/rules.md` rule 7.

## Entry points

- `rust/src/kernels/neutrino_pion.rs` — `boost_window`, and `boost_integral`'s
  `quad` call
- `rust/src/kernels/positron_pion.rs:163-164` — the sibling that clips
- `test/test_core_neutrino.py` — `reference_dnde_neutrino_charged_pion`
  and `reference_pion_continuum` both integrate over the same unclipped
  window, so the independent reference reproduces the defect and must be
  fixed with the kernel
- `test/parity/deltas.py` — the B5 declaration, whose `0.500000000000`
  positions are exactly the affected ones
- `test/parity/data/` — the `boosted_strong` block of
  `spectra.neutrino.charged_pion`
- Related project: `projects/parity-pinned-defect-repair/`
- Sibling defect, same mechanism:
  [`charged-pion-photon-spectrum-misses-the-forward-cone.md`](charged-pion-photon-spectrum-misses-the-forward-cone.md)

## Risks / open questions

- **How many other kernels boost a bounded spectrum over an unclipped
  window?** `positron_pion.rs` clips and `photon_pion.rs` is A3's own
  subject, but the sweep was not done — `neutrino_pion.rs` was found
  because a repair's magnitude looked wrong, not because anyone looked.
  Grep the `quad` call sites for an upper limit that is a boost endpoint
  rather than a support endpoint.
- **The muon-neutrino row is worse, and the corpus does not catch it.**
  The measurements above are on the electron row because that is where
  roster entry B5 made the zero visible, but the same `boost_integral`
  serves the muon row through `Flavor::Muon`, and there no prompt line
  masks the loss: at `E_π = 1395.7039`, `E_ν = 798.21` MeV the shipped
  muon row is **exactly `0.0`** where the clipped integral gives
  `4.736526e-04` MeV⁻¹. So the repair's blast radius is both rows, not
  one, and it is larger than the electron-row figures suggest. Whether
  the same hard zero reaches the muon row at other pinned positions has
  not been swept.
- **This moves published numbers by three to four decades on a band**, so
  it is `minor` at least under [`docs/versioning.md`](../../versioning.md)
  and needs a `CHANGELOG.md` entry with the magnitude — a much larger one
  than B5's own.
