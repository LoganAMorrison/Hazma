# Task 4: Coherent longitudinal meson poles

**Date:** 2026-10-17
**Project:** visible-axial-spectra
**Status:** Complete
**Plan References:** Task 4 and "Implementation layering" in projects/visible-axial-spectra/PLAN.md
**Related ADRs:** ADR-0001, ADR-0002, ADR-0003 (project-scoped)
**Depends On:** Task 3

## Objective

Provide the longitudinal source density `A_L` of the C-even final states from
coherent pi0, eta and eta' exchange through the Proca/current/Goldstone
embedding, with the U(3) meson block resummed by its Schur complement and the
lepton and charm triangles added to gamma gamma.

## Exit Criteria

The plan's deliverable and gate, verbatim: "Ward/contact cancellation and Schur
complement versus full inverse; `k.H_gg` approaches the ABJ coefficient away
from the poles (note `fig:ax-checks`); the transverse gamma gamma density is
exactly zero; residues are recovered at each pole; a single invisible-cut
insertion is verified independently."

- The Ward/contact cancellation holds, and the Schur complement matches the
  full inverse of the meson block.
- `k.H_gg` approaches the ABJ coefficient away from the poles.
- The transverse gamma gamma density is exactly zero.
- The residue at each pole is recovered.
- A single invisible-cut insertion is verified independently.
- The default LO eta-eta' mixing angle is theta_P = -15.4 degrees, and the
  first-order axial-pseudoscalar mixing guard is 0.1.
- The compute lives in a Rust kernel behind `hazma._core.axial`, with a NumPy
  oracle under `test/` (ADR-0002), and existing public values stay
  bit-identical (`references/check_public_impact.py`).

## Inputs Reviewed

- `../PLAN.md` Task 4, "Implementation layering", "Numerical impact", and
  "Related", through a subagent brief.
- ADR-0001 and ADR-0002, through the same brief.
- `task-3-rust-port.md`, through the same brief.
- Mediator extension note, `version3/note_v3/sec_ax.tex`, `sec_ax_inv.tex`
  and `sec_cosmo.tex`, in the companion research repository
  (`mev-gev-pseudo-scalar-axial-vector`) at commit `3d9c6c7`.
- That repository's `version3/integrated/ax_xsec.py` (gamma gamma, triangle and
  guard code) and `extensions/handoff_reference/HANDOFF.md` (quadratic kernel,
  invisible-meson amplitudes), at the same commit.

## Findings

- The note's labels resolve to `version3/note_v3/` at `3d9c6c7`, not to a
  `version11` folder; `version11` holds integration code only.
- Combining `eq:ax-gg` with `eq:ax-master` gives
  `A_L(gamma gamma) = s alpha^2 |A(s)|^2/(8 pi^3)`. The closed form reproduces
  the research Berends-Giele evaluation (`ax_xsec.py`, model `chpt`) to at most
  1.4e-15 relative at eight energies, once that code is given Hazma's
  `alpha_em = 1/137.04` and lepton masses.
- The ABJ limit follows from completeness of the U(3) generators:
  `sum_P Tr(G_A lambda_P) N_c Tr(lambda_P Q^2) = 2 N_c Tr(G_A Q^2)`.
- The Ward identity is `s c^T D^-1 c = 2 f^2 Tr(G_A^2)` for massless, stable
  mesons. The second mixing insertion is the Schur correction
  `D_a = M^2 - s c^T D^-1 c`, and the note's `epsilon_P` is its per-pole size.
- The invisible cut moved from the Proca line onto the mesons is
  `i s beta kappa^2 c c^T/(8 pi)` with `kappa = 2 m_chi g_A^chi/M^2`. It
  couples pi0, eta and eta', so it must enter the full matrix.
- Rust has three `alpha_em` values (`pdg`, `legacy`) and Python a third
  (`hazma/parameters.py`). The kernels therefore take `alpha` and the triangle
  masses from Python's `hazma.parameters`, as the three-pion kernel takes
  its pion mass.
- An oracle that forms `K_aa - D_a` cancels `M^2` against itself (1e-6 absolute
  error at `M = 1e5` MeV). `axial_oracle.source_self_energy` takes the Schur
  ratio of the source's own mixing column instead.
- The only C-even channel with LO decay amplitudes here is gamma gamma. With
  exact isospin, `3 pi0` reaches only the pi0 pole, which Task 2's current
  already contains. `eta pi pi` is "qualitative" in `scope-and-acceptance.md`.

## Decisions and Implementation Notes

- **Design (ADR-0003).** The Proca contact carries the poles, and one mixing
  insertion is kept behind an exported guard. The meson block is inverted
  whole, and the cut is inserted once.
- **Kernel and oracle algorithms.** The kernel `rust/src/kernels/axial_poles.rs`
  works in the physical basis with a pivoted 3x3 complex solve. The oracle
  inverts the full 4x4 kernel over `(a, pi3, eta8, eta0)` in the generator
  basis, so basis covariance is tested along with the Schur identity.
- **Guard placement.** The guard is `AxialMesonPoles.mixing_expansion` plus
  `MIXING_GUARD`, and `spectral_functions` does not enforce it. The form
  factor is mediator-free, and Task 5's rates know `M`. The guard uses the
  visible widths only, as the research code did, which makes it conservative.
- **Triangle couplings.** `TriangleCouplings` requires all four couplings
  (e, mu, tau, c), so an absent triangle is a declared zero. The charm mass is
  `hazma.parameters.charm_quark_mass`, 1270 MeV, the research value.
- **Support.** It is set at 1600 MeV, from the note's 1.6 GeV statement for
  the axial current (`sec:ax`, "Benchmarks and validity"). This is a
  restricted-support declaration, not an accuracy claim.
- **Scope.** No `3 pi0` or `eta pi pi` pole channel is added (see Findings).
  Their decay amplitudes would be a new task, not a widening of this one.

## Files Changed

- `rust/src/kernels/axial_poles.rs`: new PyO3-free kernel and its unit tests.
- `rust/src/axial.rs`, `rust/src/kernels.rs`: entry-point registration and
  the module roster.
- `hazma/form_factors/axial/_poles.py`, `__init__.py`: the public surface
  `AxialMesonPoles`, `AxialFormFactorGammaGamma`, `InvisibleCut`,
  `TriangleCouplings` and `MIXING_GUARD`.
- `test/form_factors/axial/axial_oracle.py`: full-kernel and Dirac-trace
  oracle.
- `test/form_factors/axial/test_poles.py`: 36 tests.
- `projects/visible-axial-spectra/adrs/ADR-0003-proca-embedding-of-longitudinal-meson-poles.md`,
  with a link from `PLAN.md`.
- `docs/source/form_factors.rst` and `CHANGELOG.md` (Unreleased, Added).

## Verification

- `cargo test --manifest-path rust/Cargo.toml --no-default-features
  --features test-probes`: 322 passed (8 new), with clippy `-D warnings`
  clean.
- `.venv/bin/python -m pytest -n 0 -q test/form_factors/axial`: 312 passed.
- With the worktree `.venv/bin` on PATH,
  `scripts/agents/preflight.sh --paths "hazma/form_factors/axial
  test/form_factors/axial"`, with `--md` naming the four touched Markdown
  files, gives `RESULT: PASS` on every row. The pytest row reads
  `2868 passed, 16 skipped, 20 warnings, 127 subtests passed`, and ruff
  reads `0 new, 2 fixed`.
- Tolerances measured on macOS/arm64:

| comparison | measured max | asserted |
| --- | --- | --- |
| research Berends-Giele `A_L`, 8 energies | 1.4e-15 rel | 4e-15 |
| Schur complement vs 4x4 full inverse, amplitude and self-energy | 3.0e-13 rel | 1e-11 |
| meson-block cut vs Proca-line Dirac-trace cut | 1.7e-12 rel | 1e-11 |

- Performance on N = 1e5 energies: the kernel takes 51 ns per point (83 ns
  with the cut). The oracle's 4x4 solve takes 15.6 us per point, about 300
  times slower.

## Numerical impact

`.venv/bin/python projects/visible-axial-spectra/references/check_public_impact.py`
reports `maximum_absolute_difference: 0.0` and "unchanged on the recorded
grid". The change is additive.

## Open Questions

None.

## Plan Impact

**Impact Level:** ADR required

ADR-0003 records the Proca embedding, which the plan anticipated, and
`PLAN.md`'s Orientation now links it. No task order or acceptance criterion
changed.

## Stale-state sweep

Run against the branch with untracked files marked by `git add -N .`. The
tables below are folded by hand from the commands named in each row.

- **Identifier sweep.** `rg -n --hidden 'axial_poles|AxialMesonPoles|
  AxialFormFactorGammaGamma|MIXING_GUARD|InvisibleCut|TriangleCouplings'
  projects/visible-axial-spectra/ docs/ README.md hazma/ test/ .claude/
  .codex/ .github/` hits only this diff's files, `PLAN.md`'s layering table
  row 4 (KEPT: already names `axial_poles`), and `task-3-rust-port.md`'s
  handoff (KEPT: a record of what Task 3 expected). No hit in `.claude/`,
  `.codex/`, `.github/` or `README.md`.
- **Submodule roster sweep.** `rg --hidden -n 'axial' rust/src/lib.rs
  hazma/_core.pyi .github/workflows` finds the `served` sets in `ci.yml` and
  `release.yml`, which list submodules (KEPT: `axial` was already added and
  Task 4 adds a kernel module, not a submodule), and the comments in
  `lib.rs` and `_core.pyi`, which describe `axial` as holding "axial current
  kernels" and enumerate no contents (KEPT).
- **Line-number citation sweep.** `check_doc_citations.py --changed-vs
  origin/master` reports no docs to check (no changed Markdown cites
  `file:line`).
- **Forward-looking phrase sweep.** The Task 3 phrase regex over the project,
  `hazma/form_factors/axial`, the new kernel and the Sphinx page hits only
  status headers and this note's handoff (KEPT).
- **Count sweep.**

| Claim location | Command | Actual | Status |
|---|---|---|---|
| Verification "312 passed" | `pytest -n 0 -q test/form_factors/axial` | 312 | KEPT |
| Files Changed "36 tests" | `pytest --collect-only test/form_factors/axial/test_poles.py` | 36 | KEPT |
| Verification "8 new" Rust tests | `rg -c '#\[test\]' rust/src/kernels/axial_poles.rs` | 8 | KEPT |
| README "322 Rust tests" | `cargo test` (preflight) | 322 | KEPT |
| lib.rs "six per-domain submodules" | `dir(hazma._core)` modules | six | KEPT |

- **Numerical-impact statement.** No public value changes (verified:
  `.venv/bin/python projects/visible-axial-spectra/references/check_public_impact.py`,
  `maximum_absolute_difference: 0.0`, "unchanged on the recorded grid"). The
  change is additive: no existing function or Rust entry point was modified.
- **Exit Criteria to test mapping.**

| Exit criterion | Satisfied by |
|---|---|
| Ward/contact cancellation | `test_chiral_meson_poles_cancel_the_axial_contact`, `test_massive_poles_leave_the_pcac_remainder`, `test_proca_contraction_with_a_longitudinal_current_has_no_mediator_pole`, Rust `chiral_pole_sum_cancels_the_contact_at_every_energy` |
| Schur complement versus full inverse | `test_schur_complement_matches_the_full_kernel_inverse` |
| `k.H_gg` approaches the ABJ coefficient | `test_amplitude_approaches_the_abj_coefficient_away_from_the_poles`, Rust `amplitude_approaches_the_abj_coefficient_far_from_the_poles` |
| Transverse gamma gamma density is zero | `test_transverse_gamma_gamma_density_is_exactly_zero` |
| Residue at each pole | `test_residue_is_recovered_at_each_pole`, Rust `residue_is_recovered_at_each_pole` |
| Single invisible-cut insertion | `test_invisible_cut_inserted_once_matches_the_mediator_line_cut`, `test_invisible_widths_reproduce_the_note_benchmarks`, Rust `invisible_cut_adds_its_width_to_each_pole_once` |
| theta_P = -15.4 degrees, guard 0.1 | `test_benchmark_mixing_strengths_match_the_note`, `test_mixing_expansion_reproduces_the_note_example` |
| Rust kernel, NumPy oracle, public values unchanged | `axial_oracle.py` comparisons in `test_poles.py`; `check_public_impact.py` |

- **Task-note self-consistency.** `**Status:** Complete` matches the README
  Tasks row, and every file in Files Changed appears in `git diff --stat
  origin/master`.

## Handoff to Next Task

- Task 5 should read ADR-0003 first. Its rates must call
  `AxialMesonPoles.mixing_expansion` and reject points above `MIXING_GUARD`.
- Task 5 should feed `AxialFormFactorGammaGamma.spectral_functions` into the
  `2 m^2 g_A^2 A_L/(s M^4)` contact. When dark matter can be pair-produced
  below a pole, it should pass an `InvisibleCut`.
- Task 15 can take `AxialMesonPoles.invisible_widths` as its residue input.
  The note benchmarks it pins (`kappa` 4e-6 GeV^-2, BR 2.4696175e-9, and
  the hypercharge BR 3.95e-6) are already tested here.
