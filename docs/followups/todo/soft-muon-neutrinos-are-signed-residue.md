# soft lab muon-decay neutrinos are signed rounding residue

- **Added:** 2026-10-24
- **Source:** PR #133 review round 2 (visible-axial-spectra Task 8)
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** none.

## Why

A muon's decay neutrinos have a rest-frame density proportional to `ω²` as
`ω → 0`, so a moving muon's lab density tends to a positive multiple of
`E²`. The polarized kernel instead returns rounding residue of either sign,
linear in `E`. For `AxialVectorMediator.mediator_decay_yields` with the
`_model()` of `test/axial_vector_mediator/test_radiation.py`, lab energy
600 MeV, the unpolarized density and `product="neutrino"`, the muon
neutrinos of `mu mu` are `-8.54e-129`, `-7.46e-179` and `-4.90e-229` MeV⁻¹
at `E = 1e-100`, `1e-150` and `1e-200` MeV, and those of `pi pi pi0` are
`-1.70e-128` and `-1.70e-178` at the first two. The values are far too
small to matter in a spectrum, but they are negative, and the
endpoint-bin positivity test does not reach this end.

Separately, `boost_range` in `rust/src/kernels/polarized_muon.rs` squares
the lab energy of a massless daughter, so below about `1.5e-154` MeV its
edges collapse to `[0, γE]`. That is why `pi pi pi0` returns exactly zero
at `1e-200` MeV. `Boost::massless_range` in
`rust/src/kernels/polarized_cascade.rs` computes the same edges without
squaring, for the resolved photons. Switching `boost_range` to it moves the
reference density 1e-11 below the endpoint in
`neutrino_endpoint_layer_matches_a_high_precision_reference` by 3.2e-5,
past that test's 3e-5 tolerance, because an ulp of the edge moves it that
much. PR #133 therefore left the shared helper alone.

## What

Make the soft lab density of `MuonDecay::dnde` (the closed-form branch
below `endpoint_layer`) and of its counts nonnegative, either by
evaluating the closed forms in a form that does not cancel as `E → 0` or
by switching to the quadrature branch there. Then give `boost_range` the
massless edges of `Boost::massless_range`. Re-derive the endpoint-layer
reference's tolerance from the edge's ulp sensitivity rather than
loosening it to fit. Add a public test that soft neutrino yields are
nonnegative and scale as `E²`.

## Entry points

- `rust/src/kernels/polarized_muon.rs`: `boost_range`, the massless call in
  `MuonDecay`'s neutrino density, and
  `neutrino_endpoint_layer_matches_a_high_precision_reference`.
- `rust/src/kernels/polarized_cascade.rs`: `Boost::massless_range`.
- Related project: `projects/visible-axial-spectra/`.
