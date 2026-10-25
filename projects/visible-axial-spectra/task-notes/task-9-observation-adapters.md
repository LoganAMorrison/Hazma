# Task 9: Observation adapters

**Date:** 2026-10-25
**Project:** visible-axial-spectra
**Status:** Complete
**Plan References:** `../PLAN.md` Task 9, Numerical impact, Implementation
layering; `../references/scope-and-acceptance.md` acceptance table
**Related ADRs:** ADR-0004, ADR-0005 (project-scoped)
**Depends On:** Task 6, Task 7, Task 8; the `_f_eff_helper` correction
(PRs #134, #135)

## Objective

Connect the new axial-vector-mediator spectra to the existing gamma-ray limits
and to CMB energy deposition.

## Exit Criteria

These are the gates of `../PLAN.md` Task 9 as patched in this task (see
Decisions).

- Monoenergetic and e+- box tests.
- The model's cascade efficiencies 0.4021864 (600/450 MeV) and 0.3750662
  (900/500 MeV) for a transverse pair decaying into `pi pi pi0`.
- The line efficiencies 0.7241273 and 0.6053788 (note `sec:spd-feff`, read
  with tables linear in E).
- The secluded L_mu-L_tau conditional efficiency 0.9544662; p_ann of about
  1.288% of the Planck limit (the note's 1.286% scaled to linear-in-E tables)
  moves to Task 14.
- `references/check_public_impact.py` reports existing public outputs
  bit-identical.

## Inputs Reviewed

- `../PLAN.md` Task 9, Task 14, Implementation layering, Dependencies.
- `../task-notes/README.md` and the Task 8 handoff in
  `task-8-resolved-photons.md`.
- ADR-0002 (kernels in Rust, NumPy oracle), ADR-0004 (refusal while a final
  state is unmodeled) and ADR-0005 (rates per pair; consumers apply the
  population factor).
- `../references/scope-and-acceptance.md` acceptance table.
- `hazma/theory/_theory_cmb.py` and `hazma/cmb.py` (the `f_eff` fold and
  tables).

## Findings

- **Note labels.** `tab:spd-cascade`, `sec:spd-feff` and `sec:axl6-cmb` exist
  in no `.tex` under `~/Documents/personal/mev-gev-pseudo-scalar-axial-vector`
  (versions 3-26). The values trace to `version4/spectra/RESEARCH_NOTE.md`
  (cascades, lines), `version6/axial_secluded/cmb_results.json` (L_mu-L_tau),
  and `paperB_draft/b1_conventions_ledger.csv` row B1-45, which names
  0.3967/0.3705 as the muon-polarized replacements of 0.4022/0.3750. No script
  reproduces 0.3967/0.3705; their codex worktree no longer holds them.
- **Cascade fold of the current model.** Folding `mediator_pair_yields`
  (isovector card, transverse density, B(3π) = 1) with Hazma's tables gives
  0.402186 at 600/450 MeV and 0.375066 at 900/500 MeV (16000-point log grid;
  4000 points agree to 2e-6). Split at 600/450 MeV: photons 0.23446 against
  the research fold's 0.23046, e± 0.16773 against 0.17171.
- **Why the totals match the unpolarized values.** The e± carry the
  polarized π → μν decay: e± energy over charged-pion energy is 0.2638 in the
  lab against 0.2647 at rest polarized and 0.2777 in the unpolarized research
  spectra. The pion kinematics differ instead: the Task 6 mass hybrid injects
  1224.6 MeV per 1200 MeV pair and gives the π⁰ about 405 MeV against 397.8.
  The excess raises f_eff by about as much as the polarization lowers it.
- **Table interpolation.** The research fold
  (`version4/resources/deposition.py`) reads the Slatyer tables linearly in
  log E; `hazma.cmb.f_eff_g` and `f_eff_ep` are linear in E. Log-E reading of
  Hazma's own table reproduces the note's line values 0.723585 and 0.600670
  and its transverse L_mu-L_tau box 0.952770 (note 0.9527711); linear-in-E
  reading gives 0.7241273, 0.6053788 and 0.9544662.
- **Resolved photons need no cut in the TheoryAnn hooks.** In the
  centre-of-mass frame `rest_frame_resolved_photons` at E is identical for
  every cut below E (checked at cuts 1e-3, 1 and 1.9 MeV, E from 2 to 200 MeV,
  all three radiating states): the cut only masks the support.
- **Existing paths already folded the direct channels.** `TheoryCMB.f_eff`,
  `cmb_limit`, `binned_limit` and `total_conv_spectrum_fn` reach the model
  through `_spectrum_funcs`, `_positron_spectrum_funcs` and the line hooks,
  and the `_f_eff_helper` correction (PRs #134, #135) already doubles the
  positrons and reads the kinetic energy. Only prompt radiation and the
  mediator pair were missing.

## Decisions and Implementation Notes

- **Cascade and line gates pin the model's own values (user decision).** The
  plan's 0.3967/0.3705 assume energy-conserving pion kinematics that the
  Task 6 mass hybrid does not have, and its 0.724/0.601 and 0.952771 assume
  log-E interpolation; changing `hazma.cmb`'s tables would move every model's
  published f_eff. The gates are 0.4021864, 0.3750662, 0.7241273, 0.6053788
  and 0.9544662, each also checked against an independent fold.
- **L_mu-L_tau split (user decision).** The conditional efficiency is pinned
  here with a hand-supplied transverse density; p_ann needs sigma v(Z'Z')
  and moves to Task 14's gate as about 1.288% of Planck, the note's 1.286%
  (log-E tables) times 0.9544662/0.9527711 for Hazma's linear-in-E tables.
- **Photon hooks.** `_spectrum_funcs` maps each state to
  `_final_state_photons`, the continuum plus `_radiation` with the cut at the
  smallest requested energy. No `include_fsr` exists or is added, so nothing
  is counted twice. Positron hooks are unchanged: the model has no resolved
  e± beyond the lines.
- **`mediator_pair_f_eff(*, density, x_kd=1e-4)`.** It uses the `e_cm` that
  `f_eff` uses, `2 m_chi (1 + v_cmb^2/2)`, gives each mediator half of it, and
  returns `2 sum_f B_f D_f / (2 m_chi)` with `D_f` the deposited energy of one
  decay. It holds no prompt radiation, like the pair yields it folds, and
  raises when `e_cm <= 2M`.
- **Rust `deposition` kernel.** `deposited_energy` integrates
  `E dN/dE f(E - m)` with `polarized_muon::integrate` at epsrel 1e-8, breaking
  at every table node and at `Cascade::box_edges` (the two ends of an e± or ν
  box; muon and pion decays smooth theirs). The table is read with
  `crate::interp::interp`, which is NumPy's clamped linear interpolation and
  so equals `f_eff_ep`'s floor; photons start at the table's first node as
  `TheoryCMB` does. The dispatch gains `pair_cascade_deposition` and
  `three_pion_cascade_deposition`, sharing the new `pair_cascade` and
  `three_pion_cascade` constructors with the yield entry points.
- **Performance.** One `pi pi pi0` pair efficiency takes 2.9 s (both
  products); a 4000-point Python grid fold of the yields takes 18 s. The e±
  boxes take milliseconds.
- **Counts were rejected as the fold's input.** `mediator_pair_counts` costs
  about 8 s per 500 bins per product for three pions, and a midpoint fold of
  counts is only second order.

## Files Changed

- `rust/src/kernels/deposition.rs` (new) — the fold and its closed-form
  tests.
- `rust/src/kernels/polarized_cascade.rs` — `Cascade::box_edges` and its
  test.
- `rust/src/kernels.rs` — module registration and doc paragraph.
- `rust/src/axial_vector_mediator.rs` — the two deposition entry points and
  the shared cascade constructors.
- `hazma/axial_vector_mediator/_model.py` — `_final_state_photons`,
  `_cascade_deposition`, `mediator_pair_f_eff`, `_DEPOSITED`, class and
  method docstrings (the `pi pi pi0` energy excess, the resolved photons
  already in `spectra`).
- `test/axial_vector_mediator/test_observation_adapters.py` (new) — 11 tests.
- `docs/source/axial_vector_mediator.rst` — "Gamma-ray limits and the CMB".
- `CHANGELOG.md` — Unreleased entry.
- `../PLAN.md` Task 9 and Task 14 gates; `../references/scope-and-acceptance.md`
  acceptance rows.

## Verification

- `cargo test --manifest-path rust/Cargo.toml --no-default-features --features
  test-probes -- box_edges deposition`: 5 passed; the narrow-box test's
  tolerance is relative, `(width / center)^2`. `cargo clippy ... -D
  warnings`: clean.
- `.venv/bin/python -m pytest -n 0 -q
  test/axial_vector_mediator/test_observation_adapters.py`: 11 passed in
  7.0s. Oracles: the photon line against `f_eff_g` itself (1e-14); the
  transverse and isotropic e± boxes and the L_mu-L_tau box against scipy
  angular and energy integrals with the exact tensor power (1e-8); the
  `mu mu` plus neutrino pair against a scipy fold of `mediator_pair_yields`
  with break points at the table nodes (1e-8, the kernel's `epsrel`; the
  measured agreement is 3e-10); the `e e`
  photons and `binned_limit(comptel_diffuse)` against `SingleChannelAnn("e
  e")` (1e-13, 1e-12); the spectra hook against `rest_frame_yields` plus
  `rest_frame_resolved_photons` at cut 1 MeV (1e-14), with `mu mu` open.
- `env PATH=".venv/bin:$PATH" scripts/agents/preflight.sh --paths
  "hazma/axial_vector_mediator test/axial_vector_mediator" --md "<the four
  Markdown files>"`: RESULT: PASS across all eleven rows; pytest 3026 passed,
  16 skipped, 127 subtests passed; ruff 0 new.
- The cascade pins agree with a 16000-point trapezoid over
  `mediator_pair_yields` (0.4021865, 0.3750662) to 1.2e-7 at both points.

## Numerical impact

`.venv/bin/python projects/visible-axial-spectra/references/check_public_impact.py`:
164 existing outputs, maximum absolute difference 0.0, "unchanged on the
recorded grid", hazma imported from this worktree. That script evaluates only
`dnde_photon_neutral_pion`, `dnde_positron_muon`, `dnde_neutrino_muon` and
`VectorFormFactorPiPi.width`, so it cannot observe this diff, and its result
is not the reason no released value moves. The reason is that the diff touches
only `AxialVectorMediator`, which is unreleased (absent from tag 2.2.0),
`Cascade::box_edges` and two new entry points. The unreleased
`AxialVectorMediator.spectra` and `total_spectrum` gain the resolved photons
of `e e`, `mu mu` and `pi pi pi0`. The `pair_cascade_yields` and
`three_pion_cascade_yields` refactor only moved construction into shared
helpers, and the existing cascade tests pin their values.

## Open Questions

- The note's spin-corrected 0.3967/0.3705 remain unreproduced by any script.
  A common-mass cascade would test them, but the mass hybrid is a Task 6 user
  decision, so nothing here changes it.

## Plan Impact

**Impact Level:** Update phase file or `../rules.md` (flat project: `../PLAN.md`
and `../references/scope-and-acceptance.md`)

`../PLAN.md` Task 9's gate now names the model's values with the two
convention differences, and Task 14's gate takes the p_ann item. The
acceptance table follows. No ADR: the decisions bind only these gates.

## Review fixes

PR #136 review round 1 (reviewers A and E) added the `mu mu` plus neutrino
pair-efficiency test and a `mu mu` channel in the resolved-photon test, wrote
the `pi pi pi0` 2% energy excess into the docs, corrected three stale
statements about `spectra` and the resolved photons, restated Task 14's p_ann
gate as about 1.288%, made the narrow-box Rust tolerance relative, and
corrected the numerical-impact reasoning. The nested line-energy dicts in
`SingleChannelAnn` are deferred to
`docs/followups/todo/single-channel-nested-line-dicts.md`.

## Stale-state sweep

This block is folded by hand: rows collapse many `rg` hits and carry
dispositions. Commands ran on the final tree, after `git add -N` of the
untracked files and after the last prose edit. Counts for this note include
this block.

- **Identifier sweep.** `rg -c --hidden '<identifier>'
  projects/visible-axial-spectra/ docs/ README.md hazma/ test/ .claude/
  .codex/ rust/src CHANGELOG.md` (excluding the saved `*sweep.txt` and
  `stale-state.json` receipts) for each new name:
  - `mediator_pair_f_eff` hits `CHANGELOG.md` (1), the Sphinx page (1),
    `_model.py` (2), the task-notes README (1), `deposition.rs` (1), the
    adapter tests (7) and this note (6), all KEPT as current.
  - `_final_state_photons` hits `_model.py` (2) and this note (3), KEPT.
  - `_cascade_deposition` hits `_model.py` (4), `rust/src/axial_vector_mediator.rs`
    (4, the two `*_cascade_deposition` entry points) and this note (6),
    KEPT.
  - `_DEPOSITED` hits `_model.py` (2) and this note (2), KEPT.
  - `pair_cascade_deposition|three_pion_cascade_deposition` hits `_model.py`
    (2), `rust/src/axial_vector_mediator.rs` (4) and this note (3), KEPT.
  - `box_edges` hits `polarized_cascade.rs` (4), `axial_vector_mediator.rs`
    (1), the task-notes README (1) and this note (6), KEPT.
  - `deposited_energy` hits `deposition.rs` (5), `axial_vector_mediator.rs`
    (3) and this note (2), KEPT.
  - `fn pair_cascade\b|fn three_pion_cascade\b` hits
    `rust/src/axial_vector_mediator.rs` (2, the shared constructors), KEPT.
- **Line-number citation sweep.** `rg -n
  '_model\.py:[0-9]+|deposition\.rs:[0-9]+|test_observation_adapters\.py:[0-9]+|polarized_cascade\.rs:[0-9]+|axial_vector_mediator\.rs:[0-9]+'
  projects/visible-axial-spectra/ docs/` (excluding the receipts) returned no
  hits. `scripts/agents/check_doc_citations.py` over `CHANGELOG.md`, the
  Sphinx page, `PLAN.md`, `scope-and-acceptance.md`, the task-notes README
  and this note reported `docs scanned: 6`, `in-repo citations checked: 0`
  and `out-of-range or ambiguous: NONE`.
- **Forward-looking phrase sweep.** `rg -n --hidden '(Task [0-9]+ will|will
  be added|still pending|today: ?stub|currently|In Progress)'` over this
  note, the task-notes README, `hazma/axial_vector_mediator`,
  `deposition.rs`, `rust/src/axial_vector_mediator.rs`, the Sphinx page and
  `CHANGELOG.md` hits only `task-notes/README.md:3` (`**Status:** In
  Progress`, KEPT: Tasks 10 to 16 are open).
- **Retired-gate sweep.** `rg -n --hidden
  '0\.3967|0\.3705|0\.724\b|0\.601\b|0\.952771|1\.286%|0\.4022|0\.3750\b'`
  over `projects/`, `docs/`, `hazma/`, `test/`, `.claude/`, `.codex/`,
  `CHANGELOG.md` and `rust/src` (the herwig4dm and CSV data hits are
  unrelated digits and are not listed) hits the note values only as
  explained references: `PLAN.md` Task 9 gate and Task 14 gate,
  `scope-and-acceptance.md` rows 24-26, 28 and 29, the adapter test
  comments, and this note's Findings, Decisions, Open Questions and
  Handoff. All KEPT: each names the superseded value beside the pinned
  one, and p_ann stands in Task 14's gate as about 1.288% (1.286% is the log-E
  reading). This note's Exit
  Criteria (EDITED) now list the patched gates, where they had listed the
  original ones.
- **Doc consistency edits made by this sweep.** The note's Exit Criteria
  and Inputs Reviewed, which still carried the pre-decision gates and one
  source (EDITED). The task-notes README handoff, which still read as Task 9's
  kickoff and now points at Task 10 (EDITED), and its numerical-impact
  paragraph, which was re-wrapped (EDITED).
- **Canonical-contract diff.** PLAN Task 9 asks for the explicit reuse of
  `total_conv_spectrum_fn`, `binned_limit` and `unbinned_limit` (satisfied
  through the `_spectrum_funcs` hook and `test_electron_pair_gamma_rays_match_the_single_channel_model`),
  a deposition fold with the kinetic-energy table argument, lines counted
  once and `p_ann = f_eff <sigma v>/(2 m_chi)` (the existing `TheoryCMB`
  path, per the `_f_eff_helper` correction of PRs #134 and #135), and the
  gate, which this diff patched on the user's decision to pin the model's
  own values. The patched Task 9 gate (0.4021864, 0.3750662, 0.7241273,
  0.6053788, 0.9544662) equals the pins in
  `test_three_pion_cascade_efficiency`,
  `test_a_photon_line_deposits_the_table_at_its_energy` and
  `test_secluded_l_mu_l_tau_conditional_efficiency`; the Task 14 gate and
  the acceptance rows (24-26, 28, 29) carry the same numbers and the split
  of p_ann to Task 14. The layering table row for Task 9 names `deposition`
  in `hazma._core.axial_vector_mediator` with `TheoryAnn` adapters, as
  shipped. Task 16's benchmark cards ask for "current (not superseded)
  reference values", which the acceptance table now supplies. ADR-0001
  (units and densities) holds: the pair efficiency is a dimensionless ratio
  of a deposited energy in MeV to `2 m_chi`. ADR-0002 (Rust kernels, NumPy
  oracle) holds: the fold is in Rust and the tests use scipy integrals as
  the independent oracle, though no NumPy copy of the fold is kept. ADR-0004
  holds: `mediator_pair_f_eff` refuses through
  `_pair_branching_fractions` while a final state is unmodeled, and the
  spectra hooks still refuse a total with an open channel. ADR-0005 holds:
  the photon hook adds no population or multiplicity factor. ADR-0003
  concerns the longitudinal poles and is untouched. No repo-wide ADR in
  `docs/adrs/` concerns this surface. Nothing in the plan or the ADRs is now
  wrong.
- **Count sweep.**

| Claim location | Command | Actual | Status |
| --- | --- | --- | --- |
| Verification and README, 11 adapter tests | `.venv/bin/python -m pytest --collect-only -q -n 0 test/axial_vector_mediator/test_observation_adapters.py` | `11 tests collected` | EDITED from 10 |
| Verification and README, 5 Rust tests | `cargo test --manifest-path rust/Cargo.toml --no-default-features --features test-probes -- box_edges deposition` | `5 passed` | KEPT |
| Files Changed, 4 + 1 split of those 5 | `grep -c '#\[test\]' rust/src/kernels/deposition.rs` | 4, plus `box_edges_bound_the_electron_support` | KEPT |
| Verification, 11 passed | `.venv/bin/python -m pytest -n 0 -q test/axial_vector_mediator/test_observation_adapters.py` | `11 passed` | EDITED from 10 |
| Verification and README, 3026 passed | path-scoped `preflight.sh` of the review-fix pass | `3026 passed` | EDITED from 3025 |
| Numerical impact, 164 values | `check_public_impact.py` (below) | maximum absolute difference 0.0 | KEPT |

- **Exit Criteria to test mapping.**

| Exit criterion | Test or artifact |
| --- | --- |
| Monoenergetic and e+- box tests | `test_a_photon_line_deposits_the_table_at_its_energy` (monoenergetic), `test_a_transverse_electron_box_folds_to_its_angular_integral`, `test_an_isotropic_electron_box_is_flat`, and the Rust `deposition` closed-form tests |
| Cascade efficiencies 0.4021864 and 0.3750662 | `test_three_pion_cascade_efficiency` |
| Line efficiencies 0.7241273 and 0.6053788 | `test_a_photon_line_deposits_the_table_at_its_energy` |
| L_mu-L_tau conditional efficiency 0.9544662 | `test_secluded_l_mu_l_tau_conditional_efficiency` |
| `mu mu` and neutrino pair efficiency against an independent fold | `test_the_dimuon_and_neutrino_pair_efficiency_folds_the_pair_yields` |
| Public outputs bit-identical | `references/check_public_impact.py` |

- **Numerical-impact statement.** No existing public value changes
  (verified: `.venv/bin/python
  projects/visible-axial-spectra/references/check_public_impact.py`, which
  reports a maximum absolute difference of 0.0 over 164 values, hazma
  imported from this worktree), though that script does not observe this
  diff (see Numerical impact). The unreleased
  `AxialVectorMediator.spectra` and `total_spectrum` gain the resolved
  photons of `e e`, `mu mu` and `pi pi pi0`, and `mediator_pair_f_eff` is
  new; both belong to the unreleased model.
- **Task-note self-consistency.** The `**Status:**` header reads `Complete`,
  matching the README's Tasks cell. The Exit Criteria checklist above is the
  patched gate set and matches the mapping table. Every file, function and
  class cited in Files Changed appears in `git diff --stat origin/master --`
  after `git add -N`.

## Handoff to Next Task

- Task 14 can route `"v v"` through `mediator_pair_f_eff`'s fold once it
  supplies the pair's density and sigma v; p_ann of about 1.288% of Planck
  (linear-in-E tables) is its gate.
- `_spectrum_funcs` photons now include prompt radiation; a channel added
  later must give `_radiation` a branch or a zero.
- The pair efficiency holds no prompt radiation; adding it would need a
  mediator-frame cut and the O(alpha) recoil the decay yields lack.
