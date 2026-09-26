# The charged pion's neutrino continuum loses its quadrature support

- **Added:** 2026-09-06
- **Source:** `projects/parity-pinned-defect-repair/task-notes/task-10a-neutrino-pion-line.md`
- **Scope:** cross-cutting (public spectrum values)
- **Status:** done. Repaired as parity roster entry `C2`, the second
  label issued under
  [ADR-0003](../../adrs/ADR-0003-corpus-repairs-are-declared-deltas.md).
- **Triggers / blockers:** corpus re-pinning only. The Cython twin
  (`hazma/spectra/_neutrino/_pion.pyx`) was deleted in `cython-to-rust`
  Task 4.6, so there was no oracle to capture. `parity-pinned-defect-repair`
  closed with 2.3.0 before this landed, so it takes a `C` label rather
  than a roster entry in that project's plan.

> **Resolved.** `boost_window` in `rust/src/kernels/neutrino_pion.rs`
> clips the window above at `neutrino_muon::max_energy(ENG_MU_PI_RF)`,
> the muon spectrum's own endpoint, and clips the lower limit to the
> upper one so that an empty window integrates to `+0.0`. Both rows of
> `dnde_neutrino_charged_pion` move. See "Resolution (measured)" below.

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
[`charged-pion-photon-spectrum-misses-the-forward-cone.md`](../done/charged-pion-photon-spectrum-misses-the-forward-cone.md)
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
  [`charged-pion-photon-spectrum-misses-the-forward-cone.md`](../done/charged-pion-photon-spectrum-misses-the-forward-cone.md)

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

## Resolution (measured)

**Kernel.** `neutrino_muon::max_energy(emu)` returns the smallest double
at which `dnde_neutrino_muon`'s own guard closes, so the spectrum is
exactly zero there and nonzero one ulp below. At `E_μ^rf` that is
**69.78356271700862 MeV**. The 200,001-point sweep above put it at
69.783500, and the precise figure is 6.97e-4 MeV below `ENU_E_PI_RF`
rather than 7.6e-4. `boost_window` clips at it, as `positron_pion.rs`
clips at `EMAX_PI_RF`. Both are pinned in the kernel's tests:
`the_clip_is_the_integrand_s_own_endpoint` and
`max_energy_is_the_boosted_support_s_edge`.

**Physics invariant.** Neutrino number per flavor, which the corpus
cannot see because it compares each parent energy only against itself.
`the_boost_conserves_neutrino_number_per_flavor` (Rust) and
`TestPhysics::test_the_pion_yields_one_muon_neutrino_from_each_of_two_sources`
(`test/test_core_neutrino.py`) now run at `E_π = 400`, `1395.7039` and
`5000` MeV. With the clip, both rows integrate to their branching
ratios within 5.5e-5 at 20,001 points. Over the shipped window, the
muon-decay continuum carried this fraction of its neutrinos, measured
by trapezoid on 4,001 points against scipy over the unclipped window:

| `E_π` (MeV) | electron row | muon row |
| --- | --- | --- |
| 400 | 0.99998 | 0.99981 |
| 1395.7039 | 0.939 | 0.889 |
| 5000 | 0.212 | 0.178 |

**Independent reference.** `test/test_core_neutrino.py`'s
`reference_pion_continuum` clips at its own transcription of the
endpoint, and `reference_dnde_neutrino_charged_pion` now takes its
continuum from it. The kernel's
`a_strongly_boosted_continuum_is_not_lost` pins the three positions in
the table under "Why", for both rows, against that reference to 1e-6.
The muon row at `E_π = 1395.7039`, `E_ν = 798.21` MeV is
4.736526e-04 MeV⁻¹, where 2.3.0 returned exactly `0.0`.

**Corpus.** The committed arrays are untouched. The moved positions are
declared in `test/parity/deltas.py` as `C2`: an `Additive` term equal to
scipy's continuum over the clipped window minus scipy's continuum over
the shipped one, both over the ported muon kernel, which reproduces the
Cython's integrand bit for bit.

- **Reach.** 430 positions in 6 arrays, both rows at the same 215
  energies B5 moves in the electron row: 70 in `near_rest`, 138 in
  `boosted_mild` and 222 in `boosted_strong`. Nothing moves at `rest`,
  where the rest-frame branch integrates nothing, or at `rest_plus_eps`,
  where no grid point's window straddles the endpoint.
- **Magnitude.** 402 positions move by 6.4e-12 to 9.3e-4 relative, in
  either direction, because the narrower interval re-subdivides a
  quadrature that had already found the support. The other 28, all in
  `boosted_strong`, are where the shipped quadrature returned `0.0`: 14
  muon-row values that were exactly `0.0`, and the 14 electron-row
  values B5 left holding the prompt line alone, which rise by 0.17x to
  3,441x. This is the "exactly 0.500000000000" band B5 recorded.
- **`B5+C2`.** C2 moves exactly the six arrays B5 declares, so all six
  compose, as `rules.md` rule 7 requires. The composition holds to B5's
  derived 3e-12 and measures 1.273e-15 worst relative over the 430
  positions.
- **Independent oracle.**
  `test/parity/test_delta_models.py::test_the_shipped_window_is_the_stored_continuum`
  requires the term's shipped half, plus the shipped lines, to equal the
  stored arrays at all 2,344 positions of the four boosted blocks. It
  holds to 2.2e-16 relative, and it recounts the 28 lost positions. No
  spectrum kernel is evaluated.
- **Revert check.** Restoring the unclipped upper limit fails exactly
  the three blocks that hold the six declared arrays in
  `pytest test/parity`.

**Other windows.** The sweep asked for under "Risks" found no other
energy-space boost integral with an unclipped window except
`photon_rho.rs`'s, which
[`rho-photon-outer-boost-misses-support.md`](../todo/rho-photon-outer-boost-misses-support.md)
already tracks. The same failure in the angular variable is filed as
[`mediator-decay-angular-windows-miss-their-support.md`](../todo/mediator-decay-angular-windows-miss-their-support.md).

**`NaN` inputs.** The clip is a comparison, not `f64::min`, because
`f64::min` discards a `NaN` operand. Written with it,
`dnde_neutrino_charged_pion(nan, 400.0)` returned the finite
`(0.006957016, 0.005797595, 0)` where 2.3.0 returned `(nan, nan, 0)`.
The clipped kernel returns `(nan, nan, 0.0)` for a `NaN` neutrino or pion
energy, pinned in the kernel's
`a_nan_input_stays_nan_through_the_clip` and in
`test/test_core_neutrino.py`'s `test_a_nan_neutrino_energy_stays_nan`.
`positron_pion.rs` has the same idiom from before this repair, filed as
[`positron-pion-clip-turns-nan-into-a-spectrum.md`](../todo/positron-pion-clip-turns-nan-into-a-spectrum.md).
