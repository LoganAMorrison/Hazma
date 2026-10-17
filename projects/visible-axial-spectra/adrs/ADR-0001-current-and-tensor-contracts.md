# ADR 0001 Current and tensor contracts

**Date:** 2026-10-12
**Status:** Accepted
**Scope:** Project-scoped within visible-axial-spectra.

## Context

Visible axial sources have independent isovector, octet and singlet pieces.
Tensor projection parallel to total momentum is not the physical helicity-zero
polarization of an on-shell vector. Research modules also use both GeV and MeV,
while maintained Hazma public quantities use MeV.

## Decision

The current is `G_A=diag(gau,gad,gas)` in
`L=Z_mu qbar gamma^mu gamma5 G_A q`, with dimensionless real coefficients.
Use generators T3=diag(1,-1,0), T8=diag(1,1,-2)/sqrt(3),
T0=sqrt(2/3)I, normalized by Tr(Ta Tb)=2 delta_ab. Preserve the singlet trace
and record QCD-current convention, scale and provenance explicitly. A finite
singlet convention change is explicit and is not advertised as RG evolution.

Public four-momenta carry leading Lorentz axis in metric (+,-,-,-), measured
in MeV. For three particles, momenta have shape (4,3,...) and currents (4,...).
Covariant Hermitian tensors retain complex amplitudes before contraction.
The integrated source densities are A_T=(1/3) integral dPhi |H_spatial_CM|^2
and A_L=integral dPhi |H_time_CM|^2, with the identical-particle symmetry
factor included once. These are not the half-measure absorptive Gram cuts.

For LO three-pion currents H is dimensionless, and both source densities have
units MeV^2. On-shell spin-one widths are A_T/(2q), in MeV. The source parallel
to q contributes to nonconserved off-shell sources; physical on-shell helicity
zero still satisfies q dot epsilon=0 and is not assigned A_L as a branching
fraction.

Energy conversion is explicit through an integer energy power: multiply GeV
quantities by 1000^power to obtain MeV quantities. No spin, charge or population
factor is inferred from units. Spectral results are immutable, finite and
carry threshold, input status and validity metadata.

The first LO channels accept only exact isospin isovector currents. Unsupported
octet/singlet directions raise instead of silently changing the supplied
current. A common pion mass and decay constant define the LO action. The
600 MeV default support ceiling is a declared restricted model choice, not a
claim of measured or NLO precision. Later providers may declare other ranges
only with their own validation.

## Consequences

The new API can retain traceful cards before a matching provider exists for
all their directions. Strict LO shapes remain distinct from physical-input
hybrids and fitted source models. Future pole and spectrum implementations
must preserve these normalizations and return explicit coverage information.

The current helpers do not supply an arbitrary-current source model, a new
mediator rate model or a completed photon/positron/neutrino yield pipeline.
