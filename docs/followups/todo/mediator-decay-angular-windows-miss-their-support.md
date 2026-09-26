# The mediator decay spectra lose their support at large boost

- **Added:** 2026-09-25
- **Source:** the unclipped-window sweep in
  [`neutrino-pion-continuum-loses-its-quadrature-support.md`](../done/neutrino-pion-continuum-loses-its-quadrature-support.md)
- **Scope:** cross-cutting (public spectrum values)
- **Status:** open
- **Triggers / blockers:** none. Any corpus position it moves takes the
  next `C<n>` label under
  [ADR-0003](../../adrs/ADR-0003-corpus-repairs-are-declared-deltas.md).

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
