# Both rho photon spectra return the boost integrand at rest, not the spectrum

- **Added:** 2026-08-18
- **Source:** `projects/cython-to-rust/task-notes/phase-04/task-4.5-photon-rho.md`
- **Scope:** cross-cutting (public spectrum values)
- **Status:** open
- **Triggers / blockers:** **Blocked until after cython-to-rust Phase 06
  Task 6.4.** The parity corpus pins the wrong values by construction —
  both rho cases carry a `rest` block — so a repair fails the gate that
  governs the remaining kernel swaps. Fixing it needs a declared corpus
  regeneration, the same prerequisite the six other blocked defects share
  (`charged-pion-photon-spectrum-misses-the-forward-cone.md`,
  `boost-integral-drops-last-interior-cell.md`,
  `positron-muon-spectrum-normalization-inverted.md`,
  `eta-prime-two-photon-line-missing-factor-two.md`,
  `phi-photon-lines-use-the-daughter-meson-energy.md`,
  `photon-muon-rest-frame-endpoint-uses-the-wrong-power-of-r.md`).

## Why

`hazma.spectra.dnde_photon_charged_rho` and `dnde_photon_neutral_rho`
boost a rest-frame photon source out of the rho's frame with the flat
integral

```text
dN/dE = 1/(2 β γ) ∫_{γE(1−β)}^{γE(1+β)} dE'  f(E') / E'
```

where `f` is the sum of the daughters' spectra and the `1/E'` belongs to
the boost kernel, not to `f`. As `β → 0` the window collapses and the
right-hand side tends to `f(E)`. The shipped short circuit returns
`f(E)/E` instead — the *integrand*, `1/E` and all. It is MeV⁻² where the
other branch is MeV⁻¹.

The deleted `hazma/spectra/_photon/_rho.pyx` wrote it as

```python
# hazma/spectra/_photon/_rho.pyx:47-49 (charged: :113-115)
if erho - MRHO < DBL_EPSILON:
    return integrand_neutral_rho(e)
```

and `rust/src/kernels/photon_rho.rs`'s `boosted` reproduces it under
`projects/cython-to-rust/rules.md` rule 1.

Measured on this tree (Task 4.5), stepping from `E_ρ = m_ρ` to the very
next representable double — the charged rho, ratio of the two branches:

| `E_γ` (MeV) | at `E_ρ = m_ρ` (MeV⁻¹) | one ulp above | ratio |
| --- | --- | --- | --- |
| 13 | 5.040024e-04 | 6.552032e-03 | 13.000000 |
| 50 | 1.124352e-04 | 5.621762e-03 | 50.000000 |
| 200 | 2.728379e-05 | 5.456758e-03 | 200.000000 |
| 300 | 1.817474e-05 | 5.452422e-03 | 300.000000 |

The ratio is exactly `E_γ`, which is the spurious `1/E` coming back out.

## What

Return the rest-frame *spectrum* rather than the integrand: multiply the
short-circuit result by `e`, or equivalently factor the `1/E'` out of the
integrand and into the boost so the two branches share one expression for
`f`. Both `.pyx` call sites became one Rust helper in Task 4.5, so this
is a one-line change in `boosted` plus its docs, plus the two Rust unit
tests and the Python test that currently pin the defect.

The blast radius is narrow but sharp. The branch fires **only** at
`E_ρ == m_ρ` exactly: the guard `E_ρ − m_ρ < DBL_EPSILON` is absolute,
and one ulp at 775.26 MeV is 1.14e-13 — about 500x `DBL_EPSILON` — so no
other double reaches it. A caller who evaluates a rho spectrum exactly at
rest gets a number wrong by a factor of `E_γ`; anyone one ulp away gets
the right one. That makes it easy to miss and worth fixing.

The same short circuit shape appears in the other boosted kernels
(`photon_muon`, `photon_tables`, `positron_muon`), where the rest-frame
branch returns a genuine rest-frame *spectrum* — so this is a defect
specific to the rho, not a convention the library shares.

## Entry points

- `rust/src/kernels/photon_rho.rs` — `boosted`, and the two integrand
  functions that carry the `1/E`
- `test/test_core_photon_rho.py` —
  `TestPhysics.test_the_rest_frame_branch_returns_the_bare_integrand`
  pins the current behavior and its magnitude
- `rust/src/kernels/photon_rho.rs` tests —
  `the_rest_frame_branch_returns_the_bare_integrand`
- `test/parity/data/` — the `rest` blocks of `spectra.photon.charged_rho`
  and `spectra.photon.neutral_rho`
- Blocked on the same regeneration as
  [`parity-corpus-pins-ill-conditioned-points.md`](parity-corpus-pins-ill-conditioned-points.md)

## Risks / open questions

- **Is `E_ρ == m_ρ` reachable from the public API in practice?** A user
  writing `dnde_photon_charged_rho(egams, parameters.rho_mass)` hits it
  exactly, and `hazma/spectra/_nbody.py` can pass a parent energy that
  lands on the mass for a two-body final state at threshold. Worth
  checking before deciding the fix is `patch` rather than `minor`.
- **The repair moves a published number** by a factor of `E_γ` at one
  parent energy, so it needs a `CHANGELOG.md` entry stating exactly that
  and is `minor` at least under `docs/versioning.md`.
