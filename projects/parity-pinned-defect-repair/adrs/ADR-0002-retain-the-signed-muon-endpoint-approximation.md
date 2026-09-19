# ADR 0002: Retain the signed muon endpoint approximation

**Date:** 2026-09-18
**Status:** Accepted
**Scope:** Project-scoped; A2, Task 7.

## Context

The muon photon rest-frame guard used `1 - sqrt(r)` rather than `1 - r`,
where `r = (m_e/m_mu)^2`. Kuno and Okada,
[hep-ph/9909265v1, Eq. (53)](https://arxiv.org/pdf/hep-ph/9909265),
give the latter as the kinematic endpoint. The former separates two
ranges of electron energy; it is not the photon endpoint.

Their Eqs. (54)–(56) neglect mass-suppressed terms. Hazma implements that
approximation, and it becomes negative near the endpoint. Writing
`t = 1 - y`, its polynomial-logarithm bracket is

```text
t * [-17/2 - 3*t/4 + 16*t^2/3 - 55*t^3/12
     + (3 - 2*t^2 + 2*t^3) * log(t/r)]
```

At `t = r` the logarithm vanishes and the remaining polynomial is
negative. This is an analytic limitation, not rounding noise or a
physical negative photon yield. The Task 2 corrected Cython capture
includes it, and the in-flight closed form integrates the same signed
expression to `1 - r`.

## Decision

Use `1 - r` on both branches and retain the signed approximation.
Document the limitation in the public Python docstring and Rust kernel.
Test the positive restored interval and the negative tail against the
published J+/J- representation evaluated independently at 60 digits.
Run the boost-integral identity against the production rest-frame kernel.
Declare only the four moved corpus positions against the Task 2 capture.

Cutting at the expression's zero would substitute a non-kinematic edge.
Flooring the rest branch alone would make it inconsistent with the
unchanged in-flight expression. A uniformly nonnegative replacement would
require a separate finite-mass calculation and a new assessment of every
boosted/composed caller; it is outside this endpoint-guard repair.

## Consequences

- Rest-frame and boosted implementations use the same support and signed
  expression. The original corpus and captured oracle remain untouched.
- At rest, the last 0.0197755 MeV below 52.8279516 MeV contains negative
  values, with a minimum of approximately -6.43368e-9 MeV^-1. Users
  needing endpoint precision cannot interpret that tail as a physical
  yield; the public docstring states the limitation.
- The restored interval contributes a net 5.44538e-8 photons per decay;
  its negative portion contributes -8.97482e-11. Measurements and checks
  are in `../task-notes/task-7-photon-muon-endpoint.md`.
