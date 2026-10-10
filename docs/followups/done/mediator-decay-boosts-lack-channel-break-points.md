# The mediator decay boosts integrate across their channels' edges

- **Added:** 2026-09-29
- **Source:** the energy-variable sweep in
  [`mediator-decay-angular-windows-miss-their-support.md`](../done/mediator-decay-angular-windows-miss-their-support.md)
- **Scope:** commit
- **Status:** done for the two photon kernels, as parity roster entry
  `C9` under
  [ADR-0003](../../adrs/ADR-0003-corpus-repairs-are-declared-deltas.md).
  The positron kernel is left as it was, because break points measured no
  gain there.

> **Resolved.** `mediator_tables::RestFrameSupport` records each selected
> channel's endpoint and the energies where it changes form, and the
> scalar and vector photon kernels pass every one of them to the `cos θ`
> quadrature as a break point. See "Resolution (measured)" below.

## Why

The three mediator decay kernels integrate `cos θ` from the widest
selected channel's support edge to `1`, with no break point inside that
interval. Their integrands are discontinuous or kinked there: the `π⁰`
box has a lower edge as well as an upper one, and every narrower
channel's endpoint is a kink in a mode that sums channels. QUADPACK's
error estimate is not reliable across a jump, so it can stop short of the
`epsrel = 1e-5` it was asked for.

Measured against an energy-variable reference with every edge as a break
point, the vector's `"total"` photon spectrum at `m_V = 550` MeV and
`γ = 1.05` is 2.2e-4 off at 0.060 of its lab endpoint, which is the `π⁰`
box's lower edge. The point is unchanged by the `C6` clip, since the
window there is already inside the support.

## What

Pass each selected channel's edges, mapped to `cos θ`, as interior
break points to `crate::quad`. That covers the `π⁰` box's lower edge,
every narrower channel's endpoint and the `1/E` tail threshold of the
photon tables. Measure what moves in the parity corpus, and declare it
under the next `C<n>` label.

## Entry points

- `rust/src/kernels/scalar_decay_photon.rs` — `spectrum_point`
- `rust/src/kernels/vector_decay_photon.rs` — `spectrum_point`
- `rust/src/kernels/mediator_decay_positron.rs` — `spectrum_point`, left
  unchanged (see "Resolution (measured)")
- `rust/src/kernels/mediator_tables.rs` — `cos_theta_min`, the mapping
  from a rest-frame energy to `cos θ`
- `test/test_core_mediator_decay_photon.py` — `energy_reference`, which
  already integrates with those break points

## Risks / open questions

- Break points change the adaptive partition at every energy whose
  window contains one, so the corpus movement will be broad and at the
  `1e-5` level. The repair is worth it only if the measured gain beats
  that churn.

## Resolution (measured)

**The kinks.** `RestFrameSupport::add` takes a channel's endpoint and its
interior kinks. It widens the integral's endpoint as `C6` did and keeps
every kink of an open channel, and `cos_theta_points` maps each one
through `cos_theta_min`. `quad` drops the points that are not strictly
inside the window, including the widest endpoint, which lands on the
lower limit exactly. The kinks are:

- **every selected channel's endpoint**, which is interior for every
  channel but the widest under the vector's `"total"` and the scalar's
  default modes;
- **a table's first abscissa**, `RestFrameTable`'s grid start, where the
  photon tables' `1/E` tail meets the interpolant (`add_table` adds both
  edges of a table);
- **the bottom of the `π⁰` box**, `photon_pion::neutral_pion_photon_box`.

The boosted muon spectrum's change of form at `x = (1 − r)/(1 + β)` was
checked and left out. At 110, 275 and 1500 MeV muon energies its
one-sided slopes there differ no more than two adjacent slopes on one
side do. The rest-frame spectrum vanishes at its own endpoint, so the
boosted one is smooth to first order there.

**The gain.** Each tree was swept against `energy_reference` at 550 and
900 MeV, `γ` of 1.001, 1.05, 1.5 and 10, every vector mode, the scalar's
default modes and the three positron modes, on 60 log-spaced energies
each:

| Kernel | Moved | Closer / farther | Worst error before | Worst error after |
| --- | --- | --- | --- | --- |
| scalar photon | 213 of 480 | 188 / 25 | 5.0e-4 | 2.4e-5 |
| vector photon | 325 of 3,360, by up to 4.8e-4 | 235 / 90 | median 4.3e-7 | median 1.4e-7 |
| positron | 62 of 1,440 | 36 / 26 | 3.1e-6 | 1.05e-5 |

The positron's only interior kink is the narrower table's edge under
`"total"`, and its moves stay at or below 1.1e-5 in either direction, so
the positron kernel keeps its unmarked quadrature. That spares 12 positron
arrays a declaration with nothing to show for it.

The case "Why" cites did not reproduce at that size on 2.3.0: at
`m_V = 550` MeV, `γ = 1.05`, `"total"`, 0.060 of the lab endpoint, the
kernel sat 2.4e-6 from the reference before and 9.3e-7 after. The largest
corrections are elsewhere. In the parity corpus, the `pi0 g` mode at
`m_V = 250` MeV, `γ = 2`, `E = 11.6` MeV moves 1.56%, from 1.54% below
the reference to agreement within 1e-15. There the `π⁰γ` width is 1.1e-5,
so `epsabs = 1e-10` governs the quadrature, and an error estimate that
straddled the box bottom met that floor early.
`test_core_mediator_decay_photon.py::TestTheChannelEdges` pins three
points at the kernel's own `epsrel`, `1e-5`. With the break points they
land within 5.3e-7 of the reference, and 2.3.0 missed them by 8.7e-5 to
5.1e-4.

**The corpus.** `C9` moves 2,360 of each vector photon entry point's
29,295 pinned values in 35 arrays: every `total`, `mu_mu` and `pi0_g`
array of `near_rest`, `boosted_mild` and `boosted_strong`, the open
`pi_pi` arrays at 550 and 900 MeV, and `total` at `rest_plus_eps` for 550
and 900 MeV. Every one is an array `C6` already moved, so it composes as
`C6+C9` (42 arrays over both entry points) or `A3+C6+C9` (28). The term
redoes the window's quadrature in scipy with and without the break
points, and its worst error is 2.4e-12 of the 1e-9 nested budget. The
scalar photon arrays move too, but only inside the 1e-3 budget their
`B4` composites already carry, so they need no declaration.

**A reference fix.** The test's `rest_frame_spectrum` put the `π⁰` box
bottom at `E_π − (top − E_π)`, which is `E_π(3 − β)/2` rather than
`E_π(1 − β)/2`. So `energy_reference` was breaking its integral at an
energy that is no kink, and it sat 2e-3 high at `m_V = 900` MeV, `γ = 1.05`,
`E = 9.1` MeV, where a closed-form integral of the box agrees with the
kernel. The test's `reference` now takes the same break points as the
kernel, and both references take the corrected bottom.
