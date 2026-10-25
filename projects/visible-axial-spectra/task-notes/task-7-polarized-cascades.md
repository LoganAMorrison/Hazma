# Task 7: Polarized cascade marginals

**Date:** 2026-10-23
**Project:** visible-axial-spectra
**Status:** Complete
**Plan References:** `../PLAN.md` "Task 7 Polarized cascade marginals",
"Implementation layering", "Numerical impact"
**Related ADRs:** ADR-0001, ADR-0002, ADR-0004 (project-scoped)
**Depends On:** Task 6; Task 14 for model-generated pair densities

## Objective

Boost polarized mediator decays into lab-frame yields from a declared spin
density, so that a mediator of known energy and helicity content gives the
photon, positron and neutrino yields of each decay channel.

## Exit Criteria

- Density positivity and trace are validated.
- The decay matrix `D` integrates over solid angle to the identity.
- Daughter energies sum correctly.
- The `β → 0` and ultra-relativistic endpoints are correct.
- The isotropic limit agrees with the maintained boost helpers.
- The 600/450 and 900/500 MeV three-pion cascade cases reproduce their
  reference spectra.
- Existing public outputs stay bit-identical
  (`references/check_public_impact.py`).

## Inputs Reviewed

- `../PLAN.md` Task 7, Task 14 and "Implementation layering" (via a brief).
- Research note `version3/note_v3/sec_cascade.tex` at `3d9c6c7`.
- Research reference `version3/integrated/extensions/cp_cascades.py` at
  `3d9c6c7` (through a summary).
- `task-6-rest-frame-yields.md` Findings, Open Questions and Handoff.
- `rust/src/kernels/axial_yields.rs`, `hazma/axial_vector_mediator/_model.py`.

## Findings

- **Only the diagonal matters.** A daughter's lab energy is invariant under
  rotations about the flight axis, so the off-diagonal density elements,
  which carry `e^{i(λ−λ′)φ}`, integrate away. Every spin-one decay matrix
  has the form `D = [1 + g S·n̂ + h(3P_n − 1)]/(4π)`, and the angular density
  is `[1 + g(F₊ − F₋)P₁ + h(3F₀ − 1)P₂]/2`.
- **Fermion pairs.** In units of `M`, the helicity amplitudes are
  `a = g_V + βg_A` for the helicities `(+½, −½)`, `b = g_V − βg_A` for
  `(−½, +½)` and `c₀ = √2 g_V m/M` for each equal pair. They reproduce
  explicit Dirac spinors to 5e-16 for the rate and for the fermion's
  rest-frame spin along `n̂` and `θ̂`, and the spin normal to the decay plane
  vanishes. The antifermion at `−n̂` carries the same rest-frame spin
  vector (CP), so its helicity is minus the fermion's. The mean helicity
  over angles equals Task 5's `A_h/A_T` to 1e-13.
- **Muon lab helicity.** The boost along the mediator's flight is not
  collinear with the muon, so the lab helicity is
  `γ[s_L(k + βe cos θ) − βm s_T sin θ]/|p_lab|`, which mixes the spin
  along `θ̂` into it. Against an oracle that boosts the spin four-vector
  numerically, the μ⁺μ⁻ e⁺ and e⁻ densities agree to 1e-7.
- **Three-pion alignment.** The transverse current `ε·p₀` gives the π⁰
  `h = f`. A charged pion inherits `⟨P₂(cos θ₀)⟩` weighted by `|p₀|²`.
  Since `p₀ cos θ₀ = −(p + Kt)/2` across the partner range, the moment is
  closed form,
  `(3L/8)(p² + K²/3) − ½∫|p₀|²`. It matches a brute-force angle integral
  to 1e-10 of `∫|p₀|²`.
- **Reference spectra.** With every pion at 134.9768 MeV, the research code
  (`cp_cascades.kernel_axial_3pi` at `3d9c6c7`, n = 2000, converged to
  2.1e-7 per bin) and Hazma's π⁰ photons agree within 1e-5 of the peak bin
  on all 2000 bins. The TV between the F₀ = 0 and F₀ = 1/3 spectra is
  0.0538 for 600/450 MeV and 0.0585 for 900/500 MeV at these masses,
  against the note's 0.050 and 0.057 at the physical charged-pion mass.
- **Lab second moments.** `pion_lab_second_moments_match_dalitz_moments`
  compares ⟨E²⟩ of the π⁰ and of each charged pion at E_A = 900 and
  3000 MeV, for F₀ = 0 and 1, against moments integrated by brute force
  over the Dalitz plot from momentum conservation. They agree to 5.5e-12
  at worst, while the alignment moves the charged moment by 1.8e-2 to
  2.3e-2 and wiring `CHARGED` to `spectator_alignment` fails at 1.8e-2.
- **Performance.** A μ⁺μ⁻ e± density takes 84 µs per energy, against 0.16 s
  for the Python spinor oracle. For 500 three-pion bins, e± take 0.6 s,
  neutrinos 1.2 s and photons 10.9 s. The radiative charged-pion photons
  dominate the photon time, as in Task 6, whose rest-frame kernel takes
  6.7 s on 300 bins.

## Decisions and Implementation Notes

- **Scope (user decision).** Every visible decay is covered: `π⁺π⁻π⁰`,
  `e⁺e⁻`, `νν̄` and `μ⁺μ⁻`, with each muon's lab helicity propagated from
  its mediator-frame spin.
- **Reference masses (user decision).** The 600/450 and 900/500 MeV
  references are the research code re-run with every pion at 134.9768 MeV.
  The Rust test reads them from
  `rust/src/kernels/testdata/three_pion_cascade.csv`.
- **Module.** `rust/src/kernels/polarized_cascade.rs` keeps the plan's
  name. It holds `Helicities`, `Analyzer`, `Boost`, `PairDecay`, `Decay`
  and `Cascade`. `axial_yields` gains a `Yields` trait, which its `Channel`
  and the new `Cascade` both implement, and the two alignment densities.
- **Public API.** The new methods are
  `mediator_decay_yields/counts(…, mediator_energy, *, product, density)`,
  per decay into each open state, and `mediator_pair_yields/counts`, the
  inclusive `2Σ_f B_f K_f` with `B_f` from `partial_widths`. The density is
  a 3×3 helicity matrix in the basis (+1, 0, −1); Python validates
  Hermiticity, the trace and positivity at 1e-12 and passes the diagonal on.
- **Mediator at rest.** `mediator_energy ≤ M` raises, because a mediator at
  rest decays into lines that a density cannot hold.
- **Review fixes.** The pair yield divides by `width_v`, so a `width`
  override rescales every branching fraction; `partial_widths` still
  raises while an unmodeled decay is open. The lab second-moment test pins
  the charged pions' alignment wiring, and the tolerances of the cascade
  tests state their reasons.
- **Second review round.** A radiative photon's lab edge rides on its
  parent as a massless daughter's box does, so every fold over a muon or
  charged pion breaks at `threshold_gamma(edge, E*, 0)` for each probe
  edge. Before, a probe within 1e-3 of the edge sampled only zeros; at
  450/600 MeV the μ⁺μ⁻ photon density at 468.5 MeV is now
  4.876e-15 MeV⁻¹ instead of 0. The same break point now guards Task 6's
  rest-frame three-pion fold. A `width` of zero with open decays makes the
  pair methods raise `ValueError` instead of dividing by zero.
- **Unchanged physics.** Radiative muon and charged-pion photons stay
  unpolarized, as in Task 6 (see
  `docs/followups/todo/axial-mediator-polarized-radiative-muon-photons.md`).
  Model-generated densities wait for Task 14.

## Files Changed

- `rust/src/kernels/polarized_cascade.rs` (new): the cascade kernel and its
  17 tests.
- `rust/src/kernels/testdata/three_pion_cascade.csv` (new): the reference
  spectra, with their provenance in the header.
- `rust/src/kernels/axial_yields.rs`: adds the `Yields` trait, the
  `charged_alignment` and `spectator_alignment` densities, the brute-force
  alignment test and the photon-endpoint break point with its test, and
  opens the response helpers to the sibling module.
- `rust/src/kernels.rs`, `rust/src/axial_vector_mediator.rs`: register the
  module and add `pair_cascade_yields` and `three_pion_cascade_yields`.
- `hazma/axial_vector_mediator/_model.py`: adds the four public methods,
  `_helicity_fractions`, `_visible_widths` and `_checked_edges`.
- `test/axial_vector_mediator/test_cascades.py` (new);
  `test/axial_vector_mediator/spinor_oracle.py`: `polarized_vector_decay`.
- `docs/source/axial_vector_mediator.rst`: the "Polarized cascades"
  section.
- `CHANGELOG.md`: the `Unreleased` entry for the four public methods.
- `docs/followups/todo/axial-mediator-polarized-radiative-muon-photons.md`:
  names the cascade's photon arms.
- `projects/visible-axial-spectra/task-notes/README.md`: the Tasks row and
  the handoff.
- `projects/visible-axial-spectra/task-notes/preflight-task-7.txt` (new):
  the gate log.
- `docs/agents/lessons.md`, `docs/agents/lessons-examples.md`: the
  `fold-blind-to-the-response-support-edge` class, and PR #132 under
  `new-divisor-zero-set`.

## Verification

- `cargo test --manifest-path rust/Cargo.toml --no-default-features
  --features test-probes -- polarized_cascade axial_yields`: 25 passed
  (17 + 8). `cargo clippy … -D warnings` and `cargo fmt --check` are clean.
- `pytest -n 0 test/axial_vector_mediator/test_cascades.py`: 24 passed.
- `env PATH="$PWD/.venv/bin:$PATH" scripts/agents/preflight.sh --paths
  "hazma/axial_vector_mediator test/axial_vector_mediator" --md "…"`:
  RESULT: PASS on every row; pytest `2957 passed, 16 skipped, 127 subtests
  passed in 53.36s` (`preflight-task-7.txt`).

## Numerical impact

In budget. `references/check_public_impact.py` reports 164 values
unchanged, with a maximum difference of 0.0. Before the second review
round, every value of `rest_frame_yields` and `rest_frame_counts` was
bit-identical to a built `origin/master` (`d109b65b`), and so were
`partial_widths` and `annihilation_sigma_v_funcs`, over 220 arrays. The
photon-endpoint break point then moves Task 6's rest-frame `pi pi pi0`
photon yields by at most 5e-10 of their peak, within the fold's 1e-8
tolerance: 400 densities and 200 bins at `q` of 420, 450, 520 and 590 MeV
and transverse fractions of 1 and 0.6, against the build before the
change. The `mu mu` rest-frame photons are bit-identical. Probes within
1e-3 of the charged pions' photon edge, which the fold had returned as
zero, now carry their sliver.

## Open Questions

None.

## Plan Impact

**Impact Level:** Task note only. The module name follows the plan. The
`Yields` trait and the four public methods are internal and additive
contracts that this note records.

## Stale-state sweep

This block is folded by hand: rows collapse many `rg` hits and carry
dispositions. Commands ran on the final tree, after the five untracked
files were added with `git add -N`, and after the last prose edit. Counts
for this note include this block.

- **Identifier sweep.** `rg -c --hidden '<identifier>'
  projects/visible-axial-spectra/ docs/ README.md hazma/ test/ .claude/
  .codex/ rust/src CHANGELOG.md` for each new name:
  - `mediator_decay_yields|mediator_decay_counts|mediator_pair_yields|mediator_pair_counts`
    hits `CHANGELOG.md` (2), the radiative-photon follow-up (1),
    `docs/source/axial_vector_mediator.rst` (2), `_model.py` (19), the
    cascade tests (21), `polarized_cascade.rs` (2) and this note (3),
    all KEPT as current.
  - `polarized_cascade` hits the radiative-photon follow-up (2, EDITED to
    name the cascade's photon arms), `PLAN.md` (1, the layering table row),
    the task-notes README (3), `rust/src/axial_vector_mediator.rs` (2),
    `rust/src/kernels.rs` (2) and this note (12), all KEPT.
  - `_helicity_fractions|_visible_widths|_checked_edges|_inclusive_pair`
    hits `_model.py` (11) and this note (1), KEPT.
  - `pair_cascade_yields|three_pion_cascade_yields` hits `_model.py` (2),
    `rust/src/axial_vector_mediator.rs` (5) and this note (1), KEPT.
  - `charged_alignment|spectator_alignment` hits `axial_yields.rs` (6),
    `polarized_cascade.rs` (5) and this note (1), KEPT.
  - `polarized_vector_decay` hits `spinor_oracle.py` (1), the cascade
    tests (2) and this note (1), KEPT.
  - `three_pion_cascade\.csv` hits `polarized_cascade.rs` (1) and this
    note (3), KEPT.
- **Line-number citation sweep.** `rg -n
  'polarized_cascade\.rs:[0-9]+|axial_yields\.rs:[0-9]+|_model\.py:[0-9]+|spinor_oracle\.py:[0-9]+|test_cascades\.py:[0-9]+|kernels\.rs:[0-9]+'
  projects/visible-axial-spectra/ docs/` returned no hits.
  `scripts/agents/check_doc_citations.py` over the five touched Markdown
  documents (`CHANGELOG.md`, the follow-up, the task-notes README, this
  note and the Sphinx page) reported `docs scanned: 5`, `in-repo citations
  checked: 0` and `out-of-range or ambiguous: NONE`.
- **Forward-looking phrase sweep.** `rg -n --hidden '(Task [0-9]+ will|will
  be added|still pending|today: ?stub|currently|In Progress)'` over this
  note, the task-notes README, `hazma/axial_vector_mediator`, the two
  kernels, the Sphinx page, `docs/followups/todo/` and `CHANGELOG.md` hits
  only `task-notes/README.md:3` (`**Status:** In Progress`, KEPT: Tasks 8 to
  16 are open) and unrelated follow-ups under `docs/followups/todo/` (KEPT).
- **Doc consistency edits made by this sweep.** The Related ADRs line cited
  ADR-0005 and the README cited it as a reliance, but the cascade computes
  no `sigma v`; only the pair yield's use of `partial_widths` ties it to
  ADR-0004's refusal of unmodeled totals (EDITED to ADR-0001, ADR-0002 and
  ADR-0004). `CHANGELOG.md` had no entry for the four public methods (ADDED
  under `Unreleased`). The radiative-photon follow-up recorded the omission
  for Task 6 only (EDITED to cover `Cascade::muon_photons` and
  `Cascade::charged_pion_photons`).
- **Canonical-contract diff.** PLAN Task 7 asks for density positivity and
  trace, the identity of the integral of `D`, daughter energy sums, both
  endpoints, the isotropic limit and the two three-pion references; each is
  mapped below. The "declared or model-generated spin density" clause is met
  for a declared density, and the model-generated half waits on Task 14 as
  the plan says. The inclusive yield is `2 Σ_f B_f K_f`, never `B_f^2`. The
  layering table row names `polarized_cascade`, as shipped. ADR-0001 (on-shell
  helicity zero has `q·ε = 0` and takes no `A_L`, so the cascade uses the
  transverse current alone), ADR-0002 (kernels in Rust, NumPy oracle) and
  ADR-0004 (the pair yield inherits the refusal while `v v` or `eta pi pi` is
  open) hold; ADR-0003
  and ADR-0005 concern the longitudinal rates and `sigma v`, which this task
  does not touch. No repo-wide ADR in `docs/adrs/` concerns spectra
  generation beyond ADR-0001 (FSR generator), which this task does not use.
  Nothing in the plan or the ADRs is now wrong.
- **Count sweep.**

| Claim location | Command | Actual | Status |
| --- | --- | --- | --- |
| Verification, 25 Rust tests (17 + 8) | `cargo test --manifest-path rust/Cargo.toml --no-default-features --features test-probes -- polarized_cascade axial_yields` | `running 25 tests`, `25 passed` | EDITED |
| Files Changed, 17 tests in `polarized_cascade.rs`; `axial_yields.rs` gains two | `grep -c '#\[test\]' rust/src/kernels/polarized_cascade.rs rust/src/kernels/axial_yields.rs`; `git show origin/master:rust/src/kernels/axial_yields.rs \| grep -c '#\[test\]'` | 17 and 8; 6 on master | EDITED |
| Verification, 24 cascade tests | `.venv/bin/python -m pytest --collect-only -q -n 0 test/axial_vector_mediator/test_cascades.py` | `24 tests collected` | EDITED |
| Verification, 2957 passed | `preflight-task-7.txt` | `2957 passed, 16 skipped, 20 warnings, 127 subtests passed` | EDITED |
| Reference CSV, 500 bins at two mediators | `grep -vc '^#' rust/src/kernels/testdata/three_pion_cascade.csv` | 1001 lines: a column header and 1000 rows | KEPT |
| Numerical impact, 164 values | `check_public_impact.py` (below) | 164 unchanged | KEPT |

- **Exit Criteria to test mapping.**

| Exit criterion | Test or artifact |
| --- | --- |
| Density positivity and trace | `test_densities_are_validated` |
| `D` integrates to the identity | `decay_matrix_integrates_to_the_identity`, `physical_decay_matrices_are_positive` |
| Daughter energies sum correctly | `pair_daughters_are_normalized_and_share_the_energy`, `pion_lab_densities_are_normalized_and_share_the_energy` |
| `β → 0` and ultra-relativistic endpoints | `pair_endpoints_at_rest_and_ultra_relativistic`, `slow_three_pions_reproduce_the_rest_frame` |
| Isotropic limit | `test_isotropic_electrons_are_the_boosted_line`, `test_isotropic_muons_are_the_boosted_rest_frame`, `test_isotropic_three_pions_are_the_boosted_rest_frame`, `isotropic_pair_is_the_boosted_line` |
| 600/450 and 900/500 MeV references | `three_pion_cascades_reproduce_the_reference_spectra` |
| Public outputs bit-identical | `references/check_public_impact.py` |

- **Numerical-impact statement.** No released public value changes
  (verified: `.venv/bin/python
  projects/visible-axial-spectra/references/check_public_impact.py`, which
  reports a maximum absolute difference of 0.0 over 164 values). Task 6's
  unreleased rest-frame `pi pi pi0` photons move by at most 5e-10 of their
  peak, inside the fold tolerance (see Numerical impact). The new methods
  return values that no earlier release did.
- **Task-note self-consistency.** The `**Status:**` header reads `Complete`,
  matching the README's Tasks cell. Every file, function and class cited in
  Files Changed appears in `git diff --stat origin/master --`.

## Handoff to Next Task

Task 8 (resolved photons) can build on `polarized_cascade::Cascade` and its
`Yields` impl. A new channel needs an `Analyzer` (or, for a multibody
decay, energy and alignment densities) and a `Decay` arm. The
public contract is the "Polarized cascades" section of
`docs/source/axial_vector_mediator.rst`. Task 14 should pass its threshold
helicity fractions as `numpy.diag([F₊, F₀, F₋])`. Binned three-pion
photons are slow, because radiative charged-pion photons are integrated
over each bin inside the fold: about 20 ms per bin for 500 bins of 1.8 MeV
and about 150 ms per bin for wide bins. Timings vary with machine load.
