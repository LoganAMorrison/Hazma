# The GeV vector mediator's `v v` positron and neutrino spectra are zero

- **Added:** 2026-09-29
- **Source:** the "Risks" note in
  [`mediator-decay-angular-windows-miss-their-support.md`](mediator-decay-angular-windows-miss-their-support.md),
  checked while resolving it
- **Scope:** cross-cutting (public spectrum values)
- **Status:** done.

> **Resolved.** Both spectra tabulate one mediator's branching-weighted
> decay spectrum at rest, double it, and boost it by `γ = e_cm / (2 m_V)`
> through `boost.dnde_boost_array`'s new `rest_energies` grid. The
> `V → f f̄` lines enter as boosted boxes. See "Resolution (measured)"
> below.

## Why

`VectorMediatorGeV`'s `v v` positron and neutrino spectra are exactly
`0.0` wherever `χχ → VV` is open. Measured at `m_x = 5` GeV, `m_V = 1`
GeV and `cme = 10.1` GeV, with every coupling 1: over 30 energies from
1 MeV to 5 GeV the largest `|dN/dE|` of
`positron_spectrum_funcs()["v v"]` and of both flavors'
`neutrino_spectrum_funcs(flavor)["v v"]` is `0.0`. The photon `v v`
spectrum at the same point is non-zero, peaking at 8.5e-3 MeV⁻¹.

The photon path boosts correctly: `spectra.py`'s `dnde_photon_v_v` takes
`γ = (cme / 2) / m_V` and a rest-frame spectrum evaluated at `m_V`. The
positron and neutrino paths do neither. `positron.py:874` and
`neutrino.py:977` set `gamma = 2.0 * self.mv / cme`, the inverse, and
return zeros when `gamma < 1`, which is exactly where the channel is open.
Behind that guard, the rest-frame spectrum they hand to
`make_boost_function` is evaluated at `cme` rather than at `m_V`, which a
corrected `gamma` would expose next.

## What

Follow the photon path in both: take `γ = cme / (2 m_V)` and evaluate
the rest-frame `V` decay spectrum at `m_V`. Pin the result against the
boost's own identities: the `v v` spectrum must carry twice the `V`'s
rest-frame positron and neutrino numbers at any `γ`. `make_boost_function`
integrates over an unclipped energy window, so measure whether it needs
the support clip the `hazma._core` mediator kernels gained under `C6`.

## Entry points

- `hazma/vector_mediator/_gev/positron.py:874` — `dnde_positron_v_v`
- `hazma/vector_mediator/_gev/neutrino.py:977` — `dnde_neutrino_v_v`
- `hazma/vector_mediator/_gev/spectra.py` — `dnde_photon_v_v`, the path
  that is right
- `hazma/spectra/boost.py` — `make_boost_function`

## Risks / open questions

- The parity corpus pins no `_gev` spectrum, so nothing gates the values
  this moves; the identities above have to be the evidence.

## Resolution (measured)

**Three defects, not two.** Behind the inverted `γ` and the rest-frame
spectrum evaluated at `e_cm`, both sums also added every channel with
unit weight. The photon path weights each by the mediator's branching
fraction, and both spectra now do the same, in
`_dnde_positron_v_v_rest_frame` and `_dnde_neutrino_v_v_rest_frame`.

**The boost.** Calling `make_boost_function` as before would evaluate
the rest-frame spectrum at every quadrature node. The n-body channels
cost 0.05 to 0.1 s per call whatever the array size, so that is
impractical. Instead each spectrum tabulates the rest frame once, on
`V_V_REST_FRAME_POINTS = 2000` log-spaced energies spanning its support:
`m_e` to `m_V / 2` for positrons, and `1e-6 m_V / 2` to `m_V / 2` for
neutrinos. `dnde_boost_array` then integrates the linear interpolant
exactly with `InterpolatedUnivariateSpline.integral`. Its new
keyword-only `rest_energies` argument decouples that grid from the lab
energies; without it the function is unchanged.

**The clip "What" asked about is not needed.** The interpolant is zero
outside its grid and its integral is analytic, so no window can miss the
support. The number identity below holds to the same residual at
`γ = 1.01`, `5.05` and `20`, which a window that lost support at large
boost could not do.

**The lines.** `V → e⁺e⁻` and `V → ν ν̄` are delta functions at `m_V / 2`
in the rest frame. They enter through `boost_delta_function`, as the
photon path's `π⁰γ` and `ηγ` lines do, with weight `4 BR`: two particles
per decay, as every continuum in `hazma.vector_mediator._gev` counts
particles and antiparticles together, and two mediators.

**Number and energy identities.** At `m_x = 5` GeV and `m_V = 1` GeV,
integrated by trapezoid on 20,001 log-spaced lab energies plus the two
edges of the line's box:

- *Leptophilic model* (quark couplings zero, so the exact yield per
  decay is `2 BR(e e) + 2 BR(μ μ)` for positrons and so on). The counts
  match twice that within 3.2e-6 (positrons), 1.6e-5 (`ν_e`), 1.8e-5
  (`ν_μ`) and 1e-14 (`ν_τ`, the line alone), at `γ = 5.05` and `γ = 20`.
  The mean energy per particle scales with `γ` within 7.7e-9.
- *Every coupling 1.* The positron count matches twice the
  branching-weighted sum of the dispatch table's channel yields within
  4.4e-6. The two sides must read one model instance: the `pi pi pi0`
  partial width is a phase-space integral, and it differed by 0.4%
  between two instances.
- *Grid.* Against a 32,000-point tabulation, the 2,000-point spectra
  agree pointwise within 1.2e-5 (positrons) and 5.7e-4 (`ν_μ`) wherever
  they exceed 1e-3 of their peak. One call on 200 energies takes about
  0.9 s, against 4.5 to 9 s at 32,000 points.

**Values.** At the "Why" point, the spectra now carry 2.647 electrons
and positrons and 2.294, 3.117 and 0.375 `e`-, `μ`- and `τ`-flavored
neutrinos per annihilation. They peak at 3.7e-3, 4.4e-3, 5.9e-3 and
7.6e-5 MeV⁻¹.

**Tests.** `test/vector_mediator/test_gev_v_v_spectra.py` pins the
identities above and the zero below `e_cm = 2 m_V`; all 17 of its tests
fail against master. `test/spectra/test_boost_array.py` pins the new
argument: an identical result when `rest_energies` equals `energies`,
interpolation at rest, and a narrow peak boosting to
`boost_delta_function` within 8.8e-12.

**Found along the way.** Two defects outside this item's scope:

- `dnde_photon_v_v` adds its `π⁰γ` and `ηγ` line boxes with weight 2
  rather than `2 BR`, and places the `π⁰γ` line with the charged pion's
  mass. At the "Why" point the boxes carry 4.00 of the spectrum's 6.09
  photons per annihilation, where weighted boxes would carry 0.023. Filed
  as
  [`vector-mediator-gev-vv-photon-lines-ignore-branching-fractions.md`](../todo/vector-mediator-gev-vv-photon-lines-ignore-branching-fractions.md).
- `VectorMediatorGeV`'s `e e` positron line and `ν ν` neutrino lines
  count one particle per annihilation, while its continua count
  particles and antiparticles together. Filed as
  [`vector-mediator-gev-lines-count-one-particle.md`](../todo/vector-mediator-gev-lines-count-one-particle.md).
