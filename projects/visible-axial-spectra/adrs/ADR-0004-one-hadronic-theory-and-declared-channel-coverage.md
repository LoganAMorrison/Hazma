# ADR 0004 Use one hadronic theory per model and refuse totals it cannot complete

**Date:** 2026-10-17
**Status:** Accepted
**Scope:** Project-scoped within visible-axial-spectra.

## Context

`AxialVectorMediator` (Task 5) takes a hadronic theory selector: `chpt`, U(3)
LO ChPT with the WZW action, or `rcht`, HLS with an `a1` nonet. The two
describe the same final states with different dynamics, so a sum over them
double counts. Only `chpt` has providers when the model ships, and each
provider covers a declared range: three pions to 600 MeV, `gamma gamma` to
1600 MeV. Some final states the note lists (`sec:ax-match`) have no provider
at all, such as `eta pi pi` (Task 11) and the mediator pair `Z' Z'`
(Task 14). A total that silently drops an open channel is a fabricated zero,
which the plan's Task 5 gate forbids. ADR-0001 also requires the LO
three-pion provider to reject a source with octet or singlet directions,
while the hypercharge-aligned benchmark has both.

## Decision

- **One theory per instance.** The selector is a required keyword fixed at
  construction. A declared theory without providers, `rcht` today, raises
  `NotImplementedError`; an unknown name raises `ValueError`. Rates are
  never summed across theories.
- **Channels read the directions symmetry lets them see.** In the isospin
  limit G parity keeps the octet and singlet directions out of three pions,
  so the model passes the providers the T3 component of the card,
  `AxialCurrent.from_generator_coefficients(c3, 0, 0)`, with the card's
  scheme and scale. The provider's own rule is unchanged. A channel whose
  coupling vanishes by symmetry is an exact zero at every energy, so it is
  not range-checked.
- **A rate past its provider's range raises.** Each channel rate raises
  where its provider has no declared support, and so does a width that
  needs one; the `width` override then supplies the total.
- **Totals refuse open channels without a rate.** `annihilation_cross_sections`
  and `annihilation_sigma_v` raise while an unmodeled final state is open at a
  physical energy, `e_cm >= 2 mx`: `v v` above `2 M`, and `eta pi0 pi0` and
  `eta pi pi` for a card with octet or singlet directions. `partial_widths`
  does the same for the decays. Per-channel functions stay available, since
  each one is still exact.
- **The mixing guard binds every channel.** A point where
  `AxialMesonPoles.mixing_expansion` exceeds `MIXING_GUARD` is rejected for
  every rate, lepton channels included, because the dropped second insertion
  rescales the whole longitudinal propagator (ADR-0003).

## Consequences

- **Positive:** No public total can understate an open channel, and a later
  provider widens the model by removing an entry from the unmodeled set
  rather than by changing a number that was already returned.
- **Negative:** Until Task 14, a model with `M < m_chi` has no total
  annihilation rate at any energy, so the thermal average and the limit
  adapters raise for it. Above 600 MeV an isovector card has no three-pion
  rate, and a mediator there needs an explicit width.
- **Mitigation:** The error names the missing final state. Tasks 11 and 14
  remove their entries from `AxialVectorMediator._unmodeled_thresholds` when
  they land, and Task 13's domain table owns the declared ranges.
