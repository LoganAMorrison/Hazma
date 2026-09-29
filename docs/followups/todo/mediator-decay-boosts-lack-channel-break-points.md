# The mediator decay boosts integrate across their channels' edges

- **Added:** 2026-09-29
- **Source:** the energy-variable sweep in
  [`mediator-decay-angular-windows-miss-their-support.md`](../done/mediator-decay-angular-windows-miss-their-support.md)
- **Scope:** commit
- **Status:** open

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
- `rust/src/kernels/mediator_decay_positron.rs` — `spectrum_point`
- `rust/src/kernels/mediator_tables.rs` — `cos_theta_min`, the mapping
  from a rest-frame energy to `cos θ`
- `test/test_core_mediator_decay_photon.py` — `energy_reference`, which
  already integrates with those break points

## Risks / open questions

- Break points change the adaptive partition at every energy whose
  window contains one, so the corpus movement will be broad and at the
  `1e-5` level. The repair is worth it only if the measured gain beats
  that churn.
