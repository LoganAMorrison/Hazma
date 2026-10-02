# The mediator decay spectra lose their support at large boost

- **Added:** 2026-09-25
- **Source:** the unclipped-window sweep in
  [`neutrino-pion-continuum-loses-its-quadrature-support.md`](../done/neutrino-pion-continuum-loses-its-quadrature-support.md)
- **Scope:** cross-cutting (public spectrum values)
- **Status:** done. Repaired as parity roster entry `C6`, the sixth
  label issued under
  [ADR-0003](../../adrs/ADR-0003-corpus-repairs-are-declared-deltas.md).
- **Triggers / blockers:** none.

> **Resolved.** All three kernels start the `cos θ` integral at
> `mediator_tables::cos_theta_min` of the widest selected channel's
> rest-frame endpoint, and skip it where that bound reaches `1`. Every
> mediator decay photon and positron spectrum moves wherever the mediator
> is in flight. See "Resolution (measured)" below.

## Why

Three kernels boost a mediator's rest-frame spectrum into the lab by
integrating over `cos θ ∈ [−1, 1]`, and none of them narrows that range
to the integrand's support:

- `rust/src/kernels/scalar_decay_photon.rs`, `spectrum_point`'s `quad`;
- `rust/src/kernels/vector_decay_photon.rs`, the same shape;
- `rust/src/kernels/mediator_decay_positron.rs`, the same shape.

The rest-frame energy is `E' = γE(1 − β cos θ)`, and each integrand is
zero above its rest-frame endpoint, about `m/2`. At a lab energy near
`γ m/2` the support is `cos θ ∈ [β, 1]`, about `1/(2γ²)` of the range.
QUADPACK's first 21-point rule then samples only zeros and accepts
`0.0`. This is the same failure as
[`charged-pion-photon-spectrum-misses-the-forward-cone.md`](../done/charged-pion-photon-spectrum-misses-the-forward-cone.md)
(roster entry `A3`), which `photon_pion.rs` repaired by raising its
lower limit to `cos_min`.

Measured on 2026-09-25 with `dnde_decay_s` (the scalar positron
spectrum), `m_s = 550` MeV, `pws = [0, 1, 0]` and mode `"mu mu"`, on 400
log-spaced energies from 1 MeV to `E_s`. The last nonzero energy was
956 MeV at `γ = 2`, 2,817 MeV at `γ = 10`, 1,801 MeV at `γ = 30`,
603 MeV at `γ = 100` and 200 MeV at `γ = 300`. The kinematic endpoint
grows like `γ m_s`, so everything above those energies is lost, and the
visible spectrum *shrinks* as the boost grows.

## What

For each kernel, raise the lower `cos θ` limit to where `E'` reaches
the integrand's endpoint, as `photon_pion.rs` does, and pin it with an
independent reference in the energy variable. Measure which parity
corpus positions move; the corpus's `boosted_strong` mediator blocks
may be affected at `γ = 10`, and that has not been counted. Declare
them under a new `C<n>` label.

## Entry points

- `rust/src/kernels/scalar_decay_photon.rs` — `spectrum_point`
- `rust/src/kernels/vector_decay_photon.rs` — `spectrum_point`
- `rust/src/kernels/mediator_decay_positron.rs` — `spectrum_point`
- `rust/src/kernels/photon_pion.rs` — `cos_min`, the precedent
- `hazma/spectra/boost.py` — `make_boost_function` integrates any
  caller's spectrum over an unclipped energy window, and could take the
  same fix through an optional upper support edge

## Risks / open questions

- **The endpoint differs by channel.** Each mode string adds different
  channels, and a mode's endpoint is its largest channel's, so the clip
  has to follow the modes rather than be one constant per kernel.
- **`hazma/vector_mediator/_gev` does not reach a large boost today.**
  Its `dnde_*_v_v` spectra call `make_boost_function` with
  `gamma = 2 m_V / e_cm`, which is at most 1 wherever the channel is
  open. That looks inverted, and it is unverified; it would need its own
  follow-up before `boost.py`'s window matters there.

## Resolution (measured)

**The support.** Each channel's endpoint is written once, beside the
kernel that owns it, and each kernel's `rest_frame_endpoint` takes the
widest over the channels its mode selects, as "Risks" asked:

- **FSR** ends at `x_max = 1 − 4μ²`. `mediator_tables::fsr_photon_endpoint`
  serves all four FSR functions.
- **A table** ends one cell past its last non-zero value, because linear
  interpolation carries that value down to zero across the next cell.
  `RestFrameTable::support_end` reads that edge off the data. It sits up
  to one grid step, 1.6% at 550 MeV, above the tabulated kernel's own
  endpoint, so clipping at the kernel's endpoint would have cut real
  support.
- **The `π⁰` box** ends at its top, from the new
  `photon_pion::neutral_pion_photon_endpoint`.
- **The scalar's muon**, which is not tabulated, ends at its forward-cone
  edge, from the new `photon_muon::photon_endpoint`.

At `m = 550` MeV these are 275.00 (`e e g`), 234.40 (`mu mu g`),
204.16 (`pi pi g`), 258.09 (`pi pi` table), 266.41 (vector `mu mu`
table), 257.30 (`π⁰` box at `m/2`) and 264.44 MeV (scalar `mu mu`). The
positron tables end at 258.21 (`pi pi`) and 264.80 MeV (`mu mu`). Every
table ends below `m/2`, so none is unbounded; `cargo` asserts that at
four masses.

The clip does not split the integral at the *narrower* channels' edges.
The sweeps below show that `total` still resolves them to the kernel's
tolerance, because the widest channel keeps the integrand non-zero
across the whole interval.

**Tail recovery.** Measured on a 400-point sweep from 1 MeV to `E_s`, with
the arguments under "Why". The table gives the last non-zero energy in
MeV:

| `γ` | before | after |
| --- | --- | --- |
| 2 | 956 | 973 |
| 10 | 2,817 | 5,268 |
| 30 | 1,801 | 15,716 |
| 100 | 603 | 52,072 |
| 300 | 200 | 155,357 |

The "after" column tracks the muon continuum's lab endpoint,
`γ(E' + β p') ≈ 0.96 γ m_s`, to the grid's resolution.
The scalar and vector photon `total`s already reached their endpoint
before the clip, because the electron FSR's support is the widest, and
still do. Their single channels did not.

**Conservation.** Two statements that owe nothing to either integrator.
The lab photon energy of an isotropic source is `γ` times its rest-frame
energy, and a boost conserves positron number. Measured by trapezoid on
4,001 log-spaced energies at 550 MeV:

| quantity | `γ` | before | after |
| --- | --- | --- | --- |
| scalar photon energy, all continua | 10 | 0.6988 | 1.0000018 |
| scalar photon energy, all continua | 30 | 0.0806 | 1.0000018 |
| vector photon energy, `e e g` | 30 | 0.2929 | 0.9999995 |
| vector photon energy, `mu mu` | 30 | 0.5454 | 0.9999950 |
| positrons, `total` | 10 | 0.9600 | 1.0000043 |
| positrons, `total` | 30 | 0.4808 | 1.0000046 |

At 900 MeV the positron figures are 0.4952 before and 1.0000039 after at
`γ = 30`. The residuals are the trapezoid's.

**Independent references.** The tests have two new classes, both named
`TestTheBoostedTail`, in `test/test_core_mediator_decay_photon.py` and
`test/test_core_mediator_positron.py`. Each has a reference,
`energy_reference`, that integrates over the rest-frame energy instead of
the angle. That form has no window to miss: the photon uses `u = ln E'`,
and the positron uses `t = ln p'`, which removes its `1/p'`. Both use
scipy at `epsrel = 1e-8`, with every channel's edges as break points, and
share only the rest-frame spectrum with the existing same-variable
reference.

- **Agreement.** The kernels agree with the reference to 6.9e-5 (photon)
  and 1.1e-5 (positron) worst, from `1e-3` to `0.999` of the lab endpoint
  at `γ = 2`, `10` and `30`, in every mode. The budget is 1e-3, because
  the kernel's `epsabs = 1e-10` governs where `dN/dE` falls to 1e-12: a
  60-point sweep measured 7.6e-4 there.
- **Energy and number identities.** The table above is pinned to 3e-5
  (photon) and 2e-5 (positron).
- **Same-variable references.** The existing references now clip at their
  own transcription of the endpoints. They agree with the kernels to
  2.4e-11 (photon) and 7.8e-15 (positron), inside their unchanged 1e-9.
- **Revert check.** Running the new classes against the master build fails
  68 of their 81 tests. The 13 that pass are the `γ = 2` points where
  master lost nothing.
- **`cargo`.** Each kernel gains `a_strongly_boosted_*_keeps_its_forward_tail`:
  at `γ = 30` every channel is non-zero at 0.5, 0.9 and 0.99 of its lab
  endpoint, and exactly zero above it. Restoring the unclipped window
  fails all three.

**Skipping the quadrature keeps the errors.** Above the lab endpoint the
integral is not evaluated. The integrand read the partial widths at every
node, so the kernels still read them there, and a short buffer still
raises `IndexError` at every energy.

**Corpus.** The committed arrays are untouched. The moved positions are
declared in `test/parity/deltas.py` as `C6`. Its term is scipy's clipped
quadrature minus its whole-window one, both at the kernel's options. Both
run over the kernel's own rest-frame spectrum, which is the kernel
evaluated at rest, and the Doppler factor is fused as the kernels fuse
it. Unfused, the recovered 1e-29 MeV⁻¹ tails missed by 9.4e-7.

- **Reach.** 14,580 positions in 265 arrays over all seven mediator
  entry points:

  | Kind | Positions per entry point |
  | --- | --- |
  | scalar photon | 1,184 |
  | vector photon | 3,658 |
  | scalar positron | 1,520 |
  | vector positron | 1,520 |

  Nothing moves at `rest`. Ten scalar photon positions and 77 vector
  photon positions move at `rest_plus_eps`, where `β = 1.4e-6` still
  straddles an endpoint. The set is exactly the arrays measured moving
  between a master build and this one.
- **Magnitude.** 37 scalar and 131 vector photon values were `0.0`. Another
  26 scalar photon values, 44 vector photon values and 87 scalar positron
  values rise by more than 100%, up to 6.8e5x (photon) and 1.4e5x
  (positron). The positron values were the `e⁺e⁻` line alone. The rest
  move by 4e-16 to 0.97 relative, either way, where the narrower interval
  re-subdivides. The largest fall is 10.5%, at
  `mv_550.boosted_mild.pi_pi` `E = 858` MeV, where 2.3.0 sat 11.6% above
  the energy-variable reference and the repaired kernel sits 0.15% below
  it.
- **Composites.** The composites are `A3+C6` (28 vector arrays),
  `A3+B4+C6` (14), `B4+C6` (6) and `A4+C1+C6` (96). The other 121 carry C6
  alone.
  - *The B4 composites.* B4's term reads the live kernel and so becomes
    half the *clipped* FSR. C6's term therefore clips the stored,
    half-size FSR integrand, and the two sum to the repaired spectrum
    exactly.
  - *Worst relative error by budget.* On the 1e-9 nested budget: 2.4e-12
    for C6 alone, 8.1e-13 for `A3+C6` and 4.3e-11 for `A4+C1+C6`. On
    B4's own 1e-3: 1.25e-5 for the B4 composites, below the 3.1e-4 B4
    measured before the clip. `EXPECTED_DECLARED_ARRAYS` goes from 407 to
    528.
- **Independent oracle.**
  `test/parity/test_delta_models.py::test_the_whole_window_is_what_c6_starts_from`
  requires the term's whole-window half, plus the shipped line, to equal
  the array C6 starts from at every position where the term is evaluated.
  That array is the stored one, or the A3 or A4 capture a composite
  builds on. It holds to 4.4e-15 (scalar photon), 2.4e-12 (vector photon)
  and 4.3e-11 (positron) over 22,029 positions. The test also recounts
  the positions the whole window lost outright: 66, 205 and 184 in the
  three models' cases, more than the stored zeros because a line can sit
  over a lost continuum. `test_each_model_moves_what_it_says_it_moves`
  pins each model's reach at the counts above.

**Downstream.** These kernels reach the public surface through
`ScalarMediator`'s `s s` photon and positron spectra and
`VectorMediator`'s `v v` and `π⁰ v` spectra. Those channels boost by
`γ = m_x / m_med` at threshold. At `γ = 10` the changes per annihilation
are these:

- **`m_x = 5.5` GeV, `m_med = 550` MeV.** `HiggsPortal`'s `s s` photon
  energy rises 52%, from 2,273 to 3,460 MeV, and its positrons 1.9%.
  `KineticMixing`'s `v v` photon energy rises 13% and its positrons 2.4%.
- **`m_x = 2.75` GeV, `m_med = 275` MeV.** The figures are 3.5%, 6.2%, 14%
  and 2.7%.

The vector's positron count is now 1.999 and 2.001 per annihilation at
the two points. Every open vector channel there except `π⁰γ`, whose
branching fraction is at most 1e-4, yields one positron per decay, so two
mediators carry two.

**Not taken.** `hazma/spectra/boost.py`'s `make_boost_function` is
unchanged. Its only in-tree consumers are `hazma/vector_mediator/_gev`'s
`v v` positron and neutrino spectra, and "Risks" was right that they
never boost: both invert `γ`, so they return zero wherever the channel is
open. That defect is filed as
[`vector-mediator-gev-vv-positron-and-neutrino-are-zero.md`](vector-mediator-gev-vv-positron-and-neutrino-are-zero.md).
The window matters there only once that is repaired. Separately, the
sweep found a residual that predates this repair: the vector `total` is
2.2e-4 from the energy-variable reference at `γ = 1.05` and 0.060 of its
endpoint. That point is the `π⁰` box's lower edge, a discontinuity
inside an interval the clip never touches. It is filed as
[`mediator-decay-boosts-lack-channel-break-points.md`](../todo/mediator-decay-boosts-lack-channel-break-points.md).
