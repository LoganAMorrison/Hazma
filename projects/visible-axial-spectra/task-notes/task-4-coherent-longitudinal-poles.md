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
  works in the physical basis. The invisible cut is a rank-one update along
  the source direction `a_P = Tr(G_A lambda_P)`, so Sherman-Morrison gives
  `a^T D^-1 b = a^T Delta^-1 b/(1 + i gamma a^T Delta^-1 a)` in closed form
  with no linear solve, and a meson with `a_P = 0` is skipped exactly. Both
  sums are multiplied through by the coupled `Delta` nearest zero, so an
  open cut keeps a zero-width pole finite and only a singular `D` raises. The
  heavy-fermion triangle `1 - tau arcsin^2` is summed as a series for
  `s/(4 m^2) <= 1/4`, where the subtraction would cancel. The oracle
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

- `rust/src/kernels/axial_poles.rs`: new PyO3-free kernel and its unit tests,
  with the Sherman-Morrison cut and the triangle series.
- `rust/src/axial.rs`, `rust/src/kernels.rs`: entry-point registration and
  the module roster.
- `hazma/form_factors/axial/_poles.py`, `__init__.py`: the public surface
  `AxialMesonPoles`, `AxialFormFactorGammaGamma`, `InvisibleCut`,
  `TriangleCouplings` and `MIXING_GUARD`.
- `test/form_factors/axial/axial_oracle.py`: full-kernel and Dirac-trace
  oracle.
- `test/form_factors/axial/test_poles.py`: 48 tests.
- `projects/visible-axial-spectra/adrs/ADR-0003-proca-embedding-of-longitudinal-meson-poles.md`,
  with a link from `PLAN.md`.
- `docs/source/form_factors.rst` and `CHANGELOG.md` (Unreleased, Added).
- `docs/agents/lessons.md` and `lessons-examples.md`: the review lesson on
  closed forms that cancel.

## Verification

- `cargo test --manifest-path rust/Cargo.toml --no-default-features
  --features test-probes`: 324 passed (10 new), with clippy `-D warnings`
  clean.
- `.venv/bin/python -m pytest -n 0 -q test/form_factors/axial`: 324 passed.
- With the worktree `.venv/bin` on PATH, the gate below gives
  `RESULT: PASS` on every row. Its `--md` list is the seven Markdown files
  in `git diff origin/master...HEAD --name-only -- '*.md'`. The literal
  output is in
  `projects/visible-axial-spectra/task-notes/preflight-task-4.txt`; the
  pytest row reads `2880 passed, 16 skipped, 20 warnings, 127 subtests
  passed`.

```sh
scripts/agents/preflight.sh --paths "hazma/form_factors/axial test/form_factors/axial" \
    --md "$(git diff origin/master...HEAD --name-only -- '*.md' | tr '\n' ' ')"
```

- Tolerances measured on macOS/arm64:

| comparison | measured max | asserted |
| --- | --- | --- |
| research Berends-Giele `A_L`, 8 energies | 1.4e-15 rel | 4e-15 |
| Schur complement vs 4x4 full inverse, amplitude and self-energy | 3.0e-13 rel | 1e-11 |
| meson-block cut vs Proca-line Dirac-trace cut | 1.7e-12 rel | 1e-11 |
| four triangles vs 40-digit mpmath, 1e-6 to 1600 MeV (dense grid measured; 28 points pinned inline) | 3.2e-15 rel | 1e-14 |

- Performance on N = 1e5 energies: the kernel takes 51 ns per point (83 ns
  with the cut). The oracle's 4x4 solve takes 15.6 us per point, about 300
  times slower.

## Numerical impact

`.venv/bin/python projects/visible-axial-spectra/references/check_public_impact.py`
reports `maximum_absolute_difference: 0.0` and "unchanged on the recorded
grid". The change is additive.

## Review

Findings from a generalist (A) and a numerics (E) reviewer on PR #128.

| # | Finding | Verdict | Resolution |
|---|---|---|---|
| E1, A4 | Triangle `1 - tau asin^2` cancels for `s << m^2` (charm 29% off at 1e-4 MeV) | Accepted | Series for `x <= 1/4`; all four fermions pinned to an inline 40-digit mpmath table, since CI's test job does not install the regeneration-only `mpmath` |
| E3 | 3x3 pivoted solve ill-conditioned when the cut dominates | Accepted | Sherman-Morrison closed form; solve removed |
| E4, A1 | A decoupled zero-width pole gives `0*inf = NaN` | Accepted | Skipped exactly |
| E2 | Per-pole guard can understate the Schur correction up to 3x | Accepted | Documented in `mixing_expansion` and ADR-0003 |
| E5 | The kept cut is smaller than the dropped Schur term by `Pi_aa/M^2` | Accepted | Documented in `InvisibleCut` and ADR-0003 |
| E6 | Conditioning figure "~1e6" wrong | Accepted | Recomputed: 1.4e5 at q = 135.0 MeV |
| A2 | With a cut the amplitude depends on `M` through `kappa` | Accepted | Documented in `amplitude` and ADR-0003 |
| A3 | `spectral_functions` does not point at the guard | Accepted | Docstring paragraph |
| A6 | `dm_mass = 0` is a silent no-op cut | Accepted | `InvisibleCut` docstring |
| A5 | Provenance string over 88 columns | Rejected | Matches `_three_pi.py`'s style; black and ruff pass |

The heavy-triangle reference test also tightened to rel 1e-10. No existing
public value moves, so the numerical-impact result stands.

A second generalist review (head `39e2bd42`) found one blocking issue, P2:
dividing by the bare `Delta_P` before Sherman-Morrison raised at a pole of
zero visible width even when an open cut makes `D` invertible. For example,
`pion_width = 0`, source `(1e-3, -1e-3, 0)` and `InvisibleCut(20, 1, 2500)`
at `q = m_pi` raised. The exact value is `-1.8893751744e10 i`. The fix,
accepted, multiplies both sums through by the coupled `Delta` nearest zero.
`test_open_cut_keeps_a_stable_pole_finite` pins all three zero-width poles,
at two cut strengths, against a physical-basis direct inverse. That inverse
is good to 6e-16 against 50-digit mpmath, and the kernel to at most 4.9e-14,
under a 2e-13 tolerance. The generator-basis 4x4 oracle is unusable there,
with a condition number up to 6e10. A Rust test and an aligned-pion
closed-form test cover the same limit, and with the cut closed the pole
still raises.

A third generalist review (head `4f5496ce`) found no code issue and one
blocking documentation issue, P2: the verification records predated the
final head. The README pointed at a `preflight.txt` holding the Task 1-2
run, the citation sweep claimed no documents were scanned, and the PR body
quoted 2,873 Python and 323 Rust tests. The fix, accepted, records the
final-head gate in `preflight-task-4.txt`, leaves `preflight.txt` as the
Task 1-2 record its notes cite, replaces the `--md` placeholder with the
command that derives the list, and pastes the checker's literal result.

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
- **Line-number citation sweep.** `scripts/agents/check_doc_citations.py
  --changed-vs origin/master` scans the seven changed documents and checks
  one in-repo citation, which resolves by suffix. It skips one external
  citation, `hazma/_utils/boost.pyx`, and reports no out-of-range or
  ambiguous citation.
- **Forward-looking phrase sweep.** The Task 3 phrase regex over the project,
  `hazma/form_factors/axial`, the new kernel and the Sphinx page hits only
  status headers and this note's handoff (KEPT).
- **Count sweep.**

| Claim location | Command | Actual | Status |
|---|---|---|---|
| Verification "324 passed" | `pytest -n 0 -q test/form_factors/axial` | 324 | EDITED |
| Files Changed "48 tests" | `pytest --collect-only test/form_factors/axial/test_poles.py` | 48 | EDITED |
| Verification "10 new" Rust tests | `rg -c '#\[test\]' rust/src/kernels/axial_poles.rs` | 10 | EDITED |
| README "324 Rust tests" | `cargo test` (preflight) | 324 | EDITED |
| Verification and README "2880 passed" | `preflight.sh` pytest row (`preflight-task-4.txt`) | 2880 | EDITED |
| Verification "seven Markdown files" | `git diff origin/master...HEAD --name-only -- '*.md' \| wc -l` | 7 | EDITED |
| Citation sweep "seven changed documents" | `check_doc_citations.py --changed-vs origin/master` | 7 | EDITED |
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
| Single invisible-cut insertion | `test_invisible_cut_inserted_once_matches_the_mediator_line_cut`, `test_invisible_widths_reproduce_the_note_benchmarks`, Rust `invisible_cut_broadens_an_isolated_pole_by_its_width` |
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
