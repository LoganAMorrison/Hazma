# The scalar mediator decay spectrum radiated half its FSR photons

- **Added:** 2026-09-06
- **Source:** a user report that a decaying scalar and a decaying vector of
  the same mass and lifetime gave `e⁺e⁻` FSR spectra differing by a factor
  of almost exactly two. The 2.1.0 Altarelli-Parisi fix had been verified
  against the *annihilation* FSR of both MeV models, which is correct; the
  mediator *decay* kernels behind `ScalarMediator.dnde_ss` and
  `VectorMediator.dnde_vv` were never compared to it.
- **Scope:** cross-cutting (a published number is wrong; the repair is
  gated by the parity corpus)
- **Status:** done — repaired in the pull request that files this entry,
  under the declared-delta mechanism
  [`projects/parity-pinned-defect-repair/references/corpus-repinning.md`](../../../projects/parity-pinned-defect-repair/references/corpus-repinning.md)
  specifies and `test/parity/deltas.py` now implements. Roster label
  **B4** in
  [`defect-blast-radius.md`](../../../projects/parity-pinned-defect-repair/references/defect-blast-radius.md).

## Why

`hazma._core.scalar_mediator.scalar_mediator_decay_spectrum` — the
bit-faithful Rust port of `hazma/scalar_mediator/scalar_mediator_decay_spectrum.pyx`,
unchanged since it was written in May 2018 — carries two closed-form
rest-frame FSR coefficients, `dnde_fsr_l_srf` (`e⁺e⁻γ`, `μ⁺μ⁻γ`) and
`dnde_fsr_cp_srf` (`π⁺π⁻γ`). Both are half the pair-summed spectrum.

The oracle is the annihilation side of the same model. `χχ → S* → f f̄ γ`
at `√s = m_S` and `S → f f̄ γ` at rest share a matrix element: the
dark-matter current factorizes out of the normalized photon spectrum, so
`ScalarMediator.dnde_xx_to_s_to_ffg(E, m_S, m_f)` and
`dnde_xx_to_s_to_pipig(E, m_S)` are the decay spectra. Those reproduce
the model-independent collinear limit (Eq. 4.6 of arXiv:1907.11846,
`hazma.spectra.dnde_photon_ap_fermion` summed over both legs) to under a
percent, and the vector twin's decay kernel reproduces *its* annihilation
side the same way. Measured on master `fd4c2dd7`, kernel at rest
(`eng_s = m_s`), ratio of decay kernel to annihilation closed form at the
same invariant mass, after removing the `α_legacy / α_PDG = 1.000292`
that separates the kernels' constant table from `hazma.parameters`:

```text
scalar  e e g    m_s =  10 MeV   0.5000000000  (pointwise, x = 0.01 .. 0.9)
scalar  e e g    m_s = 550 MeV   0.5000000000
scalar  mu mu g  m_s = 550 MeV   0.5000000000
scalar  pi pi g  m_s = 550 MeV   0.5000        (through the boost integral)
vector  e e g    m_v = 550 MeV   1.0000
vector  mu mu g  m_v = 550 MeV   1.0000
vector  pi pi g  m_v = 550 MeV   1.0000
```

The pointwise scalar ratios are `0.5` to ten digits at every `x`, so
this is a normalization and not a shape difference. A scalar and a
vector of the same mass and lifetime decaying to `e⁺e⁻` therefore
differed by exactly two in their FSR — the reported symptom.

What it reached: `ScalarMediator.dnde_ss` (the `s s` channel of
`χχ → SS`, open when `m_S < m_χ`) had its `e⁺e⁻γ`, `μ⁺μ⁻γ` and `π⁺π⁻γ`
components at half size, while its `π⁰`, `π⁺`, `μ` decay photons and
the `γγ` line were right; and anyone calling the kernel directly for a
decaying scalar. With the corpus's own couplings the total photon yield
per decay was low by 1.7–2.9%, and pointwise by up to a factor of two
where FSR is the only open channel at a given energy. The vector kernel
was never affected.

## What

1. `rust/src/kernels/scalar_decay_photon.rs`: both coefficients are
   multiplied by `PAIR_NORMALIZATION = 2.0`, applied last so that scaling
   by a power of two changes no rounding upstream of it. Two crate tests
   pin the corrected size against the Altarelli-Parisi pair limit.
2. `test/test_core_mediator_decay_photon.py`: the `.pyx` transcriptions
   carry the same factor, and three new physics tests pin the scalar
   FSR against `dnde_xx_to_s_to_ffg` / `dnde_xx_to_s_to_pipig` at rest,
   the vector FSR against its own annihilation side, and the scalar
   lepton FSR against the collinear limit.
3. The corpus pins the pre-repair arrays and is not regenerated
   (`projects/cython-to-rust/rules.md` rule 2). `test/parity/deltas.py`
   — the delta-declaration layer that project's Task 1 specified —
   declares all 30 `.default` arrays of
   `mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum` as
   `stored + fsr_old`, with `fsr_old` evaluated as half the repaired
   kernel's FSR-only spectrum. The 15 `mu_mu_only` arrays stay undeclared
   and still match bit for bit, which is the "moved only what it
   intended" half of the proof.
4. `CHANGELOG.md` records the shift under `Changed`.

## Entry points

- `rust/src/kernels/scalar_decay_photon.rs` — `PAIR_NORMALIZATION`,
  `dnde_fsr_l_srf`, `dnde_fsr_cp_srf`.
- `rust/src/kernels/vector_decay_photon.rs` — the twin that was right,
  `dnde_fsr_l_vrf`, `dnde_fsr_cp_vrf`.
- `hazma/scalar_mediator/_scalar_mediator_fsr.py` — the annihilation-side
  closed forms that are the oracle.
- `test/parity/deltas.py` — the B4 declaration and the layer itself.
- `test/parity/test_parity.py::_assert_declared_delta` — how the runner
  consumes a declaration.
- `projects/parity-pinned-defect-repair/` — the roster this joins as B4.

## Risks / open questions

- **The declared relation is held to 1e-3, not to the case's 1e-9.**
  The term is its own `cos θ` quadrature (`epsrel = 1e-5`) over a
  different integrand than the stored total, and the repaired total is
  a third; measured 3.1e-4 worst relative over all 4,305 declared
  positions, at `ms_550.boosted_strong`, `E = 2696` MeV, where the boost
  window is narrow and the integrator's own error estimate is what
  moves. Where the FSR is the only open channel the pre-repair value is
  a factor of two away, so the budget still separates repaired from
  unrepaired by three orders of magnitude; the staleness assertion in
  the runner is what makes a revert fail.
- **Users who patched a factor of two by hand** — into the kernel, into
  `dnde_ss`, or into `hazma.spectra.altarelli_parisi` — should remove it
  on upgrading. The Altarelli-Parisi module is per leg by design and
  `hazma.spectra.dnde_photon` already sums it once per charged particle.
- The `χχ → SS` curves in the Hazma paper were produced by the same
  `.pyx` and would carry the same understatement of the FSR.
