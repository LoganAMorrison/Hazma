---
status: In Progress
phased: false
version_bump: minor
deliverable: Axial-vector mediator model with rates, stable spectra and limits
created: 2026-10-12
---

# Visible axial spectra

**Structure:** Flat task list.

## Goal

Add a supported axial-vector (chiral U(1)') mediator model to maintained Hazma:
declared neutral visible currents, exclusive annihilation rates and widths,
normalized photon, positron and neutrino yields including polarized cascades,
and the existing indirect-detection and CMB limits. Preserve current
normalization, longitudinal pole physics and spin information so the same model
cards can support indirect and laboratory observables.

## Scope

**In scope:**

- Current, unit and source-coverage contracts and the exact-isospin LO
  three-pion current (Tasks 1-2).
- Coherent longitudinal pi0, eta and eta' poles through the Proca embedding
  (Task 3).
- `AxialVectorMediator(TheoryAnn)` with declared channels, sigma, sigma v,
  widths, lines, thresholds and resonances (Task 4), extended to the remaining
  hadronic channels of the declared theory (Task 10).
- UV-scale model cards: generic `G_V`, `G_A` with an anomaly report, the
  hypercharge-aligned type-I 2HDM and the secluded L_mu-L_tau model (Task 11).
- Enforced validity domains and an explicit unsupported-domain error (Task 12).
- Mediator-pair annihilation chi chibar -> Z'Z' (Task 13).
- Invisible pi0, eta, eta' -> chi chibar widths with caller-supplied limits
  (Task 14).
- Rest-frame yields, polarized cascades, resolved radiative photons and the
  gamma-ray and CMB adapters (Tasks 5-8).
- Shipped benchmark cards, a channel support matrix generated from code,
  documentation and the closing release (Task 15).
- Optional: fitted-input overlays with covariance (Task 9). Task 9 is not
  required to close the project; if it is not done it moves to a follow-on
  project.

**Out of scope:**

- The pseudo-scalar mediator. It is a follow-on project,
  `pseudo-scalar-mediator`, that reuses the contracts, yields, adapters and
  guards built here.
- Corrections to existing Hazma models. Each is a standalone PR outside this
  project; the list is in `references/scope-and-acceptance.md`.
- General fitted NLO singlet and strange sources, perturbative matching and
  fragmentation above the declared hadronic ranges, momentum-resolved or
  non-equilibrium relic evolution, dark-Higgs exchange outside named
  completions, and laboratory recasts other than invisible meson decays (NA64,
  E137, BaBar, K -> pi X, B -> K X, HiggsTools).
- Any runtime import of, or data dependence on, the research repository.

## Numerical impact

Additive. New public functions under `hazma.form_factors.axial` and a new model
package `hazma.axial_vector_mediator` return declared MeV quantities. Existing
public outputs must stay bit-identical. Every task reruns
`references/check_public_impact.py` against
`references/public-values-baseline.json` and records the result in its task
note. Corrections that move existing published numbers are out of scope
(standalone PRs). The closing task carries the minor version bump.

## Orientation

- [Current and tensor
  conventions](adrs/ADR-0001-current-and-tensor-contracts.md).
- [LO three-pion derivation](references/lo-three-pion.md).
- [Scope sources and acceptance references](references/scope-and-acceptance.md):
  benchmark values with their note labels, the channel validity table, and the
  map to the integration-plan PR numbers.
- [Task status](task-notes/README.md).

Each task lands as one PR. Tasks 1 and 2 land together as the first PR from the
branch `codex/visible-axial-spectra/contracts-lo-three-pion`.

## Task details

### Task 1 Current and unit contracts

Provide immutable light-current, matching and source-coverage cards, normalized
isovector/octet/singlet decomposition and explicit energy-power conversion.

Exit criteria:

- Traceful currents round-trip without dropping the singlet.
- Scalar and array conversion preserves rates, widths and spectral density.
- Input validation rejects undefined scales and nonfinite values.

### Task 2 Axial tensors and LO three pion amplitudes

Provide covariant Hermitian tensors, source projections and charged/neutral
three-pion currents including the pion-pole term in the exact isospin limit. The
declared LO support ends at 600 MeV by default; this is a restricted model range
rather than a precision hadronic claim. Unsupported flavor directions raise
instead of silently projecting away the original current.

Exit criteria:

- Independent current/Ward identities, symmetry and Lorentz checks pass.
- Integrated source densities agree with independent phase-space normalization.
- On-shell widths use transverse density; neutral three-pion longitudinal
  density remains distinct from physical helicity zero.
- Thresholds, shapes, broadcasting and unsupported input paths are tested.
- Full preflight and durable documentation checks pass.

### Task 3 Coherent longitudinal meson poles

**Objective:** Provide the longitudinal source density `A_L` of the C-even final
states from coherent pi0, eta and eta' exchange.

**Scope / implementation notes:** Derive the Proca/current/Goldstone embedding;
do not substitute a spin-zero mediator Breit-Wigner. Resum the U(3) meson block
through its Schur complement with LO eta-eta' mixing (theta_P = -15.4 degrees by
default) and add the lepton and charm triangles to gamma gamma (note
`eq:ax-gg`). Keep a first-order axial-pseudoscalar mixing guard (guard value
0.1, note `sec:cosmo`). New module `hazma/form_factors/axial/_poles.py`; tests
in `test/form_factors/axial/test_poles.py`.

**Deliverable / gate:** Ward/contact cancellation and Schur complement versus
full inverse; `k.H_gg` approaches the ABJ coefficient away from the poles (note
`fig:ax-checks`); the transverse gamma gamma density is exactly zero; residues
are recovered at each pole; a single invisible-cut insertion is verified
independently.

### Task 4 Rates and mediator model

**Objective:** Ship `AxialVectorMediator(TheoryAnn)` with declared channel cross
sections, sigma v, partial and total widths, lines, thresholds and resonances.

**Scope / implementation notes:** New package `hazma/axial_vector_mediator/`
mirroring `hazma/vector_mediator/`. Parameters are `m_chi`, `M_Zp`, `g_V^chi`,
`g_A^chi`, a Task 1 current card for quarks plus explicit lepton and neutrino
couplings, and an optional width override. A hadronic theory selector chooses
`chpt` (U(3) LO ChPT + WZW) or `rcht` (HLS + a1 nonet); a model instance uses
exactly one theory and never sums across theories. Use the note master formula
`eq:ax-master`: the width enters only the transverse term, the `A_L` s-wave is a
non-resonant contact `2 m^2 g_A^2 A_L/(s M^4)`, and `Gamma(Z' -> X) =
A_T(M^2)/(2M)`. Initial channels are e, mu, tau, nu (exact), three pions (Task
2) and gamma gamma (Task 3). Implement `annihilation_resonances` and
`annihilation_thresholds` as in the maintained models. Convert between sigma and
sigma v_Moller explicitly.

**Deliverable / gate:** At zero width the master formula reproduces the spin-1 f
fbar rate and an independent explicit-spinor evaluation with all vector and
axial couplings nonzero (the note reached 5e-16 and 2e-14, `sec:ax-match`);
threshold limits of sigma and sigma v agree; particle and population factors are
separate; no fabricated zero for an unavailable channel.

### Task 5 Rest frame stable particle yields

**Objective:** Convolve signed parent histories with public decay kernels to
give rest-frame photon, positron and neutrino yields per channel.

**Scope / implementation notes:** Charge-aware parent sums; the public positron
output counts e+ only, and the charge-summed electromagnetic injection is a
distinct result; neutrinos are nu + nubar per flavor with shape (3, N); analytic
primary lines. Include the polarized-muon kernel in pi -> mu nu and in chiral
mu+ mu- chains; Hazma 2.3's positron kernel is unpolarized (note `sec:d21`,
`sec:k22`).

**Deliverable / gate:** Michel reference multiplicities; e+ only versus e+-
identity; neutrino flavor counts; mean e+ energy per pi+ at rest
36.94069288635647 MeV with spin and 38.430221 MeV without it (note `sec:k22`);
count bins and differential spectra round-trip by bin width.

### Task 6 Polarized cascade marginals

**Objective:** Boost polarized mediator decays into lab-frame yields from a
declared or model-generated spin density.

**Scope / implementation notes:** Angular and event convolution of normalized
production marginals with F0 = 0, 1/3 and 1. A user-supplied density gives
conditional yields; a model-generated density needs the production amplitude of
its completion (Task 13 for Z'Z'). The inclusive yield is 2B, never B^2.

**Deliverable / gate:** Density positivity and trace; integral of D over solid
angle equals the identity; daughter energies sum correctly; beta -> 0 and
ultra-relativistic endpoints; the isotropic limit agrees with the maintained
boost helpers; the 600/450 and 900/500 MeV three-pion cascade cases reproduce
their reference spectra.

### Task 7 Resolved radiative photons

**Objective:** Add invariant and polarized event paths for radiative photons in
the new channels.

**Scope / implementation notes:** Photon Ward identity, soft eikonal limit,
explicit infrared cut, uncertainty and overlap bookkeeping against the
Altarelli-Parisi approximation. Disable the generic `include_fsr` path where
exact emission covers a channel. State missing virtual and recoil terms.

**Deliverable / gate:** Ward and soft-limit tests; no double counting between
the approximate and exact radiation paths; the restricted LO resolved fractions
at 450/500 MeV listed in `references/scope-and-acceptance.md` are reproduced
within the stated statistical and discretization error.

### Task 8 Observation adapters

**Objective:** Connect the new spectra to the existing gamma-ray limits and to
CMB energy deposition.

**Scope / implementation notes:** Reuse `TheoryAnn.total_conv_spectrum_fn`,
`binned_limit` and `unbinned_limit`. CMB deposition folds photon and
charge-summed electron injection with the kinetic-energy table argument, counts
lines once, and uses `p_ann = f_eff <sigma v>/(2 m_chi)` for symmetric Dirac
dark matter. Requires the standalone `_f_eff_helper` correction PR
(`hazma/theory/_theory_cmb.py`, `hazma/limits/_cmb.py`).

**Deliverable / gate:** Monoenergetic and e+- box tests; the spin-corrected
cascade efficiencies 0.3967 (600/450 MeV) and 0.3705 (900/500 MeV) of note
`tab:spd-cascade`; the line efficiencies 0.724 and 0.601 of note `sec:spd-feff`;
the secluded L_mu-L_tau efficiency 0.952771 and p_ann of 1.286% of the Planck
limit (note `sec:axl6-cmb`, after Task 13).

### Task 9 Fitted input overlays (optional)

**Objective:** Offer measured-input overlays with their covariance as an option
on top of the LO and resonance sources.

**Scope / implementation notes:** ALEPH normalization in its isovector measured
range, physical pole and lattice sensitivities, and declared matching and
truncation directions. Preserve raw negative unfolded bins and covariance; a
positive interpolation is a separate object. Never invent missing cross-source
covariance.

**Deliverable / gate:** Measured bin normalization and full covariance
projection verified; inclusive normalization is kept separate from differential
shape validation.

### Task 10 Remaining hadronic channels

**Objective:** Extend the model to the hadronic channel set of note
`sec:ax-match` under each declared theory.

**Scope / implementation notes:** pi+ pi-, K Kbar, pi0 gamma, eta gamma, eta'
gamma, pi+ pi- pi0 (vector isoscalar and axial isovector including the a1),
3pi0, eta pi pi, with the validity labels of
`references/scope-and-acceptance.md`. Reuse `hazma/form_factors/vector/` for
vector-current pieces where the definitions match. The `rcht` axial current uses
the a1 fit to tau -> 3 pi nu (m_A = 1230 MeV, Gamma_A = 420 MeV, kappa_A =
0.1539, beta = 0.9408).

**Deliverable / gate:** The pure-vector limit (G_A = 0) reproduces Hazma's
vector-mediator sigma v(pi+ pi-), Gamma(V -> pi pi) and Gamma(V -> pi0 gamma) to
3e-14 relative (note `sec:hazma-pi0g`); the a1 fit gives tau branching ratios
9.03% and 9.23%; the resonance-to-LO transverse ratio tends to 1 at low energy
(note `fig:ax-checks`).

### Task 11 UV model cards

**Objective:** Provide UV-scale constructors that map a chiral U(1)' completion
onto the low-energy couplings of Task 4.

**Scope / implementation notes:** A generic `G_V`, `G_A` card with an anomaly
report that warns by default and raises in strict mode; the hypercharge-aligned,
anomaly-free type-I 2HDM constructor (note `sec:ax-uv`); a warning that a single
Higgs doublet suppresses the axial couplings by M_Z'^2/M_Z^2; the embeddability
check of the minimal H1H2S completion (note `eq:uv-massrel`); and the secluded
L_mu-L_tau card with loop-induced kinetic mixing feeding e+ e-. No SymPy at
runtime.

**Deliverable / gate:** Anomaly coefficients vanish for the shipped anomaly-free
cards; the embeddability bound for the illustrative hypercharge card (M_Z' >=
15.55 GeV or |g_A^chi| <= 0.0268) is reproduced.

### Task 12 Validity and domain guards

**Objective:** Make the declared validity domain of every channel and theory
part of the API.

**Scope / implementation notes:** Per-channel ceilings and labels from the note
channel table; `UnsupportedDomainError` above the declared range and an
`indicative` or `qualitative` flag where the note says so; direct-annihilation
(q = sqrt(s)) versus cascade (q = M_Z') domain flags; the mixing guard of Task
3; a perturbativity warning for user couplings.

**Deliverable / gate:** One test per guard and flag; the support matrix of Task
15 is generated from these declarations.

### Task 13 Mediator pair channels

**Objective:** Add chi chibar -> Z'Z' through t- and u-channel chi exchange,
with dark-Higgs exchange only for named completions.

**Scope / implementation notes:** Expose the threshold helicity content of the
pair (note `eq:vv-threshold`) as the production density for Task 6.

**Deliverable / gate:** The secluded L_mu-L_tau benchmark gives sigma v =
4.35523e-26 cm^3/s (note `sec:axl6-cmb`); the helicity fractions sum to one and
are positive.

### Task 14 Invisible meson decays

**Objective:** Return BR(pi0, eta, eta' -> chi chibar) from the longitudinal
residue, with caller-supplied experimental limits.

**Scope / implementation notes:** Uses the Task 3 residues; no detector
simulation; limits are inputs.

**Deliverable / gate:** The NA62 bound kappa <= 5.34e-6 GeV^-2 at m_chi = 60 MeV
(note `sec:ax-inv`); the exclusion window 1.355 MeV < m_chi < m_pi0/2 - 5.5 eV
for the illustrative hypercharge card (note `sec:ax`).

### Task 15 Benchmarks, documentation and closing

**Objective:** Ship benchmark cards, a generated support matrix and user
documentation, then close the project.

**Scope / implementation notes:** JSON cards for the hypercharge-aligned
resonance, the secluded L_mu-L_tau model, the 600/450 and 900/500 MeV three-pion
cascades and the pi0 and eta line points, each with its current (not superseded)
reference values. Documentation page and tutorial examples run as doctests. The
Hazma kinetic-equilibrium relic is flagged as not applicable at narrow poles;
the note relic value for the resonance card is an external reference only.

**Deliverable / gate:** The resonance card reproduces Gamma_Z' and BR_inv with
like-for-like width ingredients (note `sec:c8`, `sec:c10`); all doctests pass;
`scripts/agents/preflight.sh --closing` passes with the minor bump and a
CHANGELOG entry naming this project.

## Dependencies

- Requires: the standalone `_f_eff_helper` correction PR before Task 8; a
  rebuilt Rust core on master before the first PR merges.
- Enables: the follow-on project `pseudo-scalar-mediator`.

## Related

The research integration plan and independent package comparisons live in the
companion research repository at commit `3d9c6c7`; the physics derivations and
verified reference values are in the mediator extension note (version 11). They
motivate this project; the runtime must not import that repository or depend on
its version folders.

## Change control

See
[`../../docs/workflow.md#adr-placement`](../../docs/workflow.md#adr-placement)
for when to write an ADR and where it lives. Patch the affected `PLAN.md` or
reference file when canonical behavior changes.

## Closing this project

The PR that flips this `PLAN.md` `status:` to `Complete` must also bump
`[project] version` in `pyproject.toml` per the `version_bump:` frontmatter and
add a `CHANGELOG.md` entry naming this project slug. Verify with
`scripts/agents/preflight.sh --closing`. See
[`../../docs/versioning.md`](../../docs/versioning.md).

### Anticipated ADRs

- Hadronic theory selector and the rule against summing theories (Task 4).
- sigma versus sigma v, population factors and stable-particle output
  conventions (Tasks 4-5).
- Proca embedding of the longitudinal meson poles (Task 3).
- Validity policy: error versus flag (Task 12).
