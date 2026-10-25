# Task 8: Resolved radiative photons

**Date:** 2026-10-24
**Project:** visible-axial-spectra
**Status:** Complete
**Plan References:** `../PLAN.md` Task 8, Numerical impact, Implementation
layering; `../references/scope-and-acceptance.md` acceptance table
**Related ADRs:** ADR-0001, ADR-0002, ADR-0004 (project-scoped)
**Depends On:** Task 6, Task 7

## Objective

Add invariant and polarized paths for resolved radiative photons in the new
channels, with an explicit infrared cut and bookkeeping that keeps exact
emission and the Altarelli-Parisi approximation from double counting.

## Exit Criteria

- Ward and soft-limit tests.
- No double counting between the approximate and exact radiation paths.
- The restricted LO resolved fractions at 450/500 MeV listed in
  `references/scope-and-acceptance.md` (about 9.12e-4 / 3.14e-3) are
  reproduced within the stated statistical and discretization error.
- `references/check_public_impact.py` reports existing public outputs
  bit-identical.

## Inputs Reviewed

- `../PLAN.md` Tasks 6-8, Numerical impact, Implementation layering.
- `../references/scope-and-acceptance.md` and `../references/lo-three-pion.md`.
- Research repository `~/Documents/personal/mev-gev-pseudo-scalar-axial-vector`
  at `3d9c6c7`: `version4/spectra/RESEARCH_NOTE.md` "Prompt radiation from the
  axial three-pion channel", `version4/spectra/cascades.py`
  (`radiative_msq`, `radiative_photon_counts`),
  `version4/hadronic/UV_SPECTRA_REVIEW.md`, and
  `integration_audit/hazma/build_api_map.py` (`radiation` entry).
- `rust/src/kernels/{axial_three_pi,axial_yields,polarized_cascade,polarized_muon}.rs`,
  `hazma/axial_vector_mediator/_model.py`, `hazma/spectra/altarelli_parisi.py`.
- Task 6 and Task 7 notes (via the task brief).

## Findings

- **The reference definition.** The "restricted LO resolved fractions" are
  `Γ(A → π⁺π⁻π⁰γ, ω > 1 MeV in the mediator frame)/Γ_Born` for an on-shell
  transverse mediator (`F₀ = 0`), from a RAMBO Monte Carlo with physical pion
  masses and `α = 1/137.035999`: `(9.12 ± 0.28)e-4` at 450 MeV and
  `(3.14 ± 0.10)e-3` at 500 MeV. The rest-frame total is independent of the
  helicity density.
- **The fractions are mass-sensitive near threshold.** An independent NumPy
  RAMBO estimate (2e6 events) gives 9.06(10)e-4 and 3.07(4)e-3 with
  physical masses but 1.31e-3 and 3.61e-3 with the model's common π⁰ mass.
  The kernel takes the π⁰ and π± masses separately; the acceptance tests use
  the physical ones and the model keeps its common-mass convention (Task 6).
- **The LO emission is exactly eikonal for both densities.** In sigma-model
  coordinates the current `σ∂φ₃ − φ₃∂σ` and the four-pion vertex
  `(φ·∂φ)²/2F²` see the charged fields only through `φ·∂φ ⊃ ∂(π⁺π⁻)` and
  `φ²`, and the trace `Tr([v, U]{a, U}†)` vanishes, so there is no
  axial-photon contact. Photons attach only to external π±, whose Born vertex
  depends on `q − p₀`; the amplitude is the Born current at
  `s̃ = (q − p₀)²` times the eikonal, transverse and source-longitudinal
  alike. The research note claimed this only for the on-shell transverse
  current and left off-shell longitudinal insertions open.
- `q·H = m₀²(m₀² − s̃)/(q² − m₀²)` holds for any charged mass, since
  `2q·p₀ = q² + m₀² − s̃` exactly, so both weights depend on `s̃` alone.
- The eikonal integrates in closed form over the pair's decay angles,
  `(4π/ω_c²)[(1 + β²)L/β − 2]`, which leaves a 2-D integral per photon energy.
  The soft limit `ω dN/dω → (α/π)⟨(1 + β²)L/β − 2⟩` is approached linearly,
  with slope about 50ω/Q at 450 MeV.
- The model never routes through `hazma.spectra.dnde_photon(include_fsr=…)`,
  and its Born yields hold no prompt radiation, so the plan's "disable the
  generic `include_fsr` path" has nothing to disable here; Task 9's adapters
  must keep it that way.

## Decisions and Implementation Notes

- **Scope (user decision, 2026-10-24):** `pi pi pi0` is exact, and `e e` and
  `mu mu` take the Altarelli-Parisi leading-log spectrum, labeled
  approximate. No channel takes both.
- **API (user decision):** separate methods `rest_frame_resolved_photons`,
  `rest_frame_resolved_photon_counts`, `mediator_decay_resolved_photons` and
  `mediator_decay_resolved_photon_counts`, each with a required `cut` in MeV
  on the rest-frame photon energy. Existing methods are unchanged, and the
  result adds to their photons per Born decay. Keys follow the Born methods'
  open states, with zeros for neutral states, and the same refusals
  (`tau tau`).
- **Kernel:** `rust/src/kernels/axial_radiation.rs` (the plan's module name).
  It uses fixed Gauss-Legendre rules, 48 in `s` after the square-root
  substitution and 32 in `cos θ₀`. Doubling both moves a spectrum by less
  than 1e-10 of its value through 600 MeV, the declared range. The emission
  integrals carry `ω′²` and the bins integrate in `ln ω`, so a cut as small
  as 1e-200 MeV neither overflows nor loses the `ln(1/cut)` growth. The lab
  fold reuses Task 7's `Boost`, the
  `Helicities::alignment` and `[f + (3F₀ − 1)hP₂]/2`, now `pub(super)`.
- **Polarized photons:** the on-shell 3π photon alignment density is `h`, with
  weight `[(n̂·p₀)² − |p₀|²/3]/2` against `|p₀|²/3`; there is no `P₁` term. AP
  photons are collinear with their lepton, so they take the `PairDecay`
  analyzer's tensor power (the vector powers cancel between the two leptons).
  The AP spectrum is clipped at the kinematic endpoint `(Q² − 4m²)/2Q`, where
  its bracket is still `ln 4 − 1`, not at `(Q² − e·m²)/2Q`, which lies past it.
  AP needs `√s ≫ 2m`: for `mu mu` its soft limit is 9.2 times the exact eikonal
  one at 215 MeV and 1.67 times at 250 MeV, and the docstring and docs say so.
  The tensor power is a choice made while implementing; the alternative, an
  isotropic boost, would drop a known alignment.
- `α` is passed from Python as `hazma.parameters.alpha_em`, as
  `_poles.py` does. The benchmark tests pass the reference's
  `1/137.035999`.
- **Not included, and stated in every docstring:** virtual corrections,
  which cancel `ln(cut)`; `O(α)` recoil and virtual corrections to the decay
  yields; non-log AP terms; spin correlations of the lepton emission.
  Feeding these photons into `spectra()` is Task 9's adapter work.

## Files Changed

- `rust/src/kernels/axial_radiation.rs` — new kernel and 17 unit tests.
- `rust/src/axial_vector_mediator.rs` — four PyO3 entry points and a shared
  `probes` helper.
- `rust/src/kernels.rs` — module registration and docs.
- `rust/src/kernels/polarized_cascade.rs`, `axial_three_pi.rs` —
  `pub(super)` on `Boost` fields, `Boost::cosine`, `Helicities::alignment`,
  `legendre2` and `gauss_legendre`.
- `hazma/axial_vector_mediator/_model.py` — four public methods, docstring
  pointers from the Born methods.
- `test/axial_vector_mediator/radiation_oracle.py`,
  `test/axial_vector_mediator/test_radiation.py` — NumPy RAMBO oracle and 20
  tests.
- `docs/source/axial_vector_mediator.rst`, `CHANGELOG.md` — user docs.

## Verification

- `cargo test --manifest-path rust/Cargo.toml --no-default-features
  --features test-probes axial_radiation`: 17 passed. These cover the
  pair-angle eikonal against explicit four-vectors (1e-10), the Born
  integrals against `ThreePion::spectral_functions` (1e-12), the soft limit
  (1e-5), the rule convergence (1e-10), the
  reference fractions (2σ), alignment positivity, support, bin round trips,
  lab count and mean-energy conservation, the isotropic and slow-mediator
  limits (1e-8), tiny cuts adding the soft logarithm, and the AP pair
  formula, its kinematic endpoint and its tensor.
- `pytest test/axial_vector_mediator/test_radiation.py -n 0`: 20 passed. These
  cover the Ward identity of the explicit two-diagram amplitude (1e-12) and its
  polarization sum (1e-10), the reference fractions (2σ of the reference, and 4σ
  of a 4e5-event oracle), the rest spectrum for transverse and longitudinal
  against the oracle (4σ per bin), the polarized lab cascade for `F₀ = 1` and
  `F₀ = 0` against the boosted oracle (4σ), the soft limit through the public
  method (1e-5), AP-only leptons against `dnde_photon_ap_fermion` (1e-12), the
  isotropic lepton cascade against `dnde_boost` (1e-5), shapes, cuts, and
  refusals including open `tau tau`.
- `env PATH="$PWD/.venv/bin:$PATH" scripts/agents/preflight.sh --paths
  "hazma/axial_vector_mediator test/axial_vector_mediator"`: RESULT: PASS,
  with pytest at `2977 passed, 16 skipped, 20 warnings, 127 subtests passed
  in 58.65s`, and black, isort, ruff (0 new), cargo fmt, clippy and cargo
  test all passing.
- Performance: 50 rest densities take 0.4 ms, 50 rest bins 5 ms, 50 lab
  densities 11 ms and 50 lab bins 14 ms, against 0.14 s for the 4e5-event
  NumPy oracle.

## Review outcome

PR review fixed: the AP endpoint (past the kinematic limit, now clipped),
the AP validity statement, overflow and logarithm loss at tiny cuts, the
Gauss-Legendre range statement, the slow-mediator tolerance (1e-4 to 1e-8)
and an open-`tau tau` refusal test. Not changed: the common π⁰ mass in the
public 3π fractions (the Task 6 convention, documented, with the reference
pinned through the kernel at physical masses) and the 2-8% Monte Carlo pins
(normalization is pinned separately by the soft limit, the Born integrals and
the pair-angle eikonal).

## Numerical impact

`.venv/bin/python projects/visible-axial-spectra/references/check_public_impact.py`:
`"maximum_absolute_difference": 0.0`, `"status": "unchanged on the recorded
grid"`. The change is additive: new methods only, and existing yields are
untouched.

## Open Questions

None.

## Plan Impact

**Impact Level:** None

The plan's module name and exit criteria hold. The finding that the LO
emission is eikonal for the longitudinal density too extends the research
note's claim but changes no task.

## Stale-state sweep

This block is folded by hand: rows collapse many `rg` hits and carry
dispositions. Commands ran on the final tree, after `git add -N .` and after
the last prose edit. Counts for this note include this block.

- **Identifier sweep.** `rg -c --hidden '<identifier>'
  projects/visible-axial-spectra/ docs/ README.md hazma/ test/ .claude/
  .codex/ rust/src CHANGELOG.md` for each new name:
  - `rest_frame_resolved_photon|mediator_decay_resolved_photon` hits
    `CHANGELOG.md` (2), `docs/source/axial_vector_mediator.rst` (3),
    `_model.py` (12), `axial_radiation.rs` (2), the radiation tests (16) and
    this note (3), all KEPT as current.
  - `axial_radiation` hits `PLAN.md` (1, the layering table row),
    `rust/src/axial_vector_mediator.rs` (2), `rust/src/kernels.rs` (2),
    `radiation_oracle.py` (1) and this note (3), all KEPT.
  - `three_pion_radiation|pair_radiation` hits `_model.py` (4),
    `rust/src/axial_vector_mediator.rs` (11) and the radiation tests (3),
    KEPT.
  - `_RADIATING_PAIRS|_PION_MASSES|_cascade_radiation` hits `_model.py` (9),
    KEPT.
  - `radiation_oracle` hits `rust/src/kernels.rs` (1), the radiation tests
    (1) and this note (1), KEPT.
- **Line-number citation sweep.** `rg -n
  'axial_radiation\.rs:[0-9]+|test_radiation\.py:[0-9]+|radiation_oracle\.py:[0-9]+|_model\.py:[0-9]+|kernels\.rs:[0-9]+|polarized_cascade\.rs:[0-9]+'
  projects/visible-axial-spectra/ docs/` returned no hits.
  `scripts/agents/check_doc_citations.py` over `CHANGELOG.md`, the Sphinx
  page, the task-notes README and this note reported `docs scanned: 4`,
  `in-repo citations checked: 0` and `out-of-range or ambiguous: NONE`.
- **Forward-looking phrase sweep.** `rg -n --hidden '(Task [0-9]+ will|will
  be added|still pending|today: ?stub|currently|In Progress)'` over this
  note, the task-notes README, `hazma/axial_vector_mediator`,
  `axial_radiation.rs`, the Sphinx page and `CHANGELOG.md` hits only
  `task-notes/README.md:3` (`**Status:** In Progress`, KEPT: Tasks 9 to 16
  are open).
- **Doc consistency edits made by this sweep.** The Related ADRs line cited
  ADR-0005, but the radiation kernels compute no `sigma v`; the transverse
  fraction is a ratio of rates, independent of the per-pair normalization
  (EDITED to ADR-0001, ADR-0002 and ADR-0004). The task-notes README gained
  the Task 8 status, finding, verification paragraph and handoff, and its
  numerical-impact paragraph now names Task 8.
- **Canonical-contract diff.** PLAN Task 8 asks for the Ward and soft-limit
  tests, no double counting between exact and approximate emission, an
  explicit cut, the stated missing virtual and recoil terms, and the 450/500
  MeV fractions; each is mapped below. Its sentence "Disable the generic
  `include_fsr` path where exact emission covers a channel" is satisfied
  vacuously, because the model's yields never call `dnde_photon(include_fsr=…)`
  (see Findings), so it is not now wrong. The layering table row names
  `axial_radiation`, as shipped. ADR-0001 (the transverse current of an
  on-shell mediator, and the source-longitudinal density) holds: both
  densities are covered. ADR-0002 (kernels in Rust, NumPy oracle) holds.
  ADR-0004 (refusal while a final state is unmodeled) holds, since both
  families use `_open_final_states` and refuse `tau tau`. ADR-0003 and
  ADR-0005 concern the longitudinal poles and `sigma v`, which this task
  does not touch. No repo-wide ADR in `docs/adrs/` concerns this surface.
  `references/scope-and-acceptance.md` row 8 lists the 450/500 MeV fractions
  and is not stale. Nothing in the plan or the ADRs is now wrong.
- **Count sweep.**

| Claim location | Command | Actual | Status |
| --- | --- | --- | --- |
| Verification, 17 Rust tests | `grep -c '#\[test\]' rust/src/kernels/axial_radiation.rs` | 17 | KEPT |
| Verification, 20 radiation tests | `.venv/bin/python -m pytest --collect-only -q -n 0 test/axial_vector_mediator/test_radiation.py` | `20 tests collected` | KEPT |
| README, model suite size | `.venv/bin/python -m pytest --collect-only -q -n 0 test/axial_vector_mediator` | `97 tests collected` | KEPT |
| Numerical impact, 164 values | `check_public_impact.py` (below) | maximum absolute difference 0.0 | KEPT |

- **Exit Criteria to test mapping.**

| Exit criterion | Test or artifact |
| --- | --- |
| Ward and soft-limit tests | `test_radiative_diagrams_obey_the_ward_identity_and_factorize`, `test_soft_limit_is_the_eikonal_radiator_over_the_born`, and the Rust soft-limit test |
| No double counting | `test_lepton_pairs_are_altarelli_parisi_alone` and the per-state dispatch in `_radiation` |
| 450/500 MeV fractions | `test_restricted_lo_fractions_reproduce_the_reference`, and the Rust reference-fraction test |
| Public outputs bit-identical | `references/check_public_impact.py` |

- **Numerical-impact statement.** No public value changes (verified:
  `.venv/bin/python
  projects/visible-axial-spectra/references/check_public_impact.py`, which
  reports a maximum absolute difference of 0.0 over 164 values). The four
  methods are new and existing yields are untouched.
- **Task-note self-consistency.** The `**Status:**` header reads `Complete`,
  matching the README's Tasks cell. Every file, function and class cited in
  Files Changed appears in `git diff --stat origin/master --`.

## Handoff to Next Task

- Task 9 adds the resolved photons to the gamma-ray and CMB adapters, with
  an explicit cut, and must not add `include_fsr` on top.
- `docs/followups/todo/axial-mediator-polarized-radiative-muon-photons.md`
  (secondary radiative decay photons) is untouched by this task and still
  open.
