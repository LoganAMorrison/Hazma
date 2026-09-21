# Task 10: Repair A4 — positron-muon normalization

**Date:** 2026-09-20
**Project:** parity-pinned-defect-repair
**Status:** Complete
**Plan References:** `../PLAN.md`, Task 10 and "Numerical impact"; `../rules.md`
**Related ADRs:** ADR-0001 (project-scoped)
**Depends On:** Tasks 1 and 2

## Objective

Multiply the Michel positron spectrum by its normalization `R_FACTOR`
instead of dividing by it, and declare the correction on the corpus
against Task 2's A4 Cython capture without rewriting any stored array.

## Exit Criteria

- [x] `rust/src/kernels/positron_muon.rs` multiplies by `R_FACTOR` in both
  the rest-frame and in-flight expressions.
- [x] Declared deltas on all six cases of the A4 row of
  `../references/defect-blast-radius.md`: `spectra.positron.muon`,
  `spectra.positron.charged_pion`, and the four mediator positron entry
  points (`dnde_decay_s`, `dnde_decay_s_pt`, `dnde_decay_v`,
  `dnde_decay_v_pt`), and nowhere else.
- [x] The repaired value reproduces Task 2's oracle within each case's
  existing budget; no tolerance is widened.
- [x] The analytic normalization test that found the defect (Task 4.1's)
  asserts the corrected constant rather than the inversion, and the
  Michel spectrum integrates to 1 over its support to a stated tolerance.
- [x] Before/after measurement over the whole corpus and the public API;
  `CHANGELOG.md`, the follow-up, the README's numerical-impact section
  and the references updated.
- [x] Reverting the repair turns the gate red; preflight green;
  `git diff --stat -- test/parity/data` empty.

## Inputs Reviewed

- `../PLAN.md` (Scope, Numerical impact, Task 10), `../rules.md`, the
  working-memory README (Findings, Numerical impact, Handoff).
- `../references/defect-blast-radius.md` (A4 row and coverage arithmetic).
- `docs/followups/todo/positron-muon-spectrum-normalization-inverted.md`.
- `test/parity/oracles/data/manifest.json` (A4 entry) and
  `test/parity/oracles/patches/A4-positron-muon-normalization.patch`.
- `test/parity/deltas.py`, `test/parity/oracle_reference.py`,
  `test/parity/test_parity.py`, `test/parity/tolerances.py`.
- Task 9's note for the before/after capture recipe; Task 8's A3 variants
  as the per-budget precedent.

## Findings

- **The measured reach is exactly Task 2's prediction**: 21,975 values in
  the six A4 cases and nothing else across all 623 corpus blocks. The
  count lands in 178 of the capture's 260 value arrays; the other 82 are
  structurally untouched (pion `rest` returns zero, mediator `e_e` never
  reaches the muon kernel, `pi_pi` is closed at a 250 MeV mediator).
- **The patched Cython's operation order is bit-reproducible.** Writing
  `(-2 root) poly * N` and `numerator * N / (β + β)` reproduces the A4
  capture bit for bit at all 1,370 `spectra.positron.muon` values, so
  that case keeps `rtol = 0`. The follow-up's open question on operation
  order is answered by this.
- **The consumers' residuals are the port's, not the repair's.** Against
  the capture the pion sits at 5.30e-15 and the mediators at 3.55e-12
  (scalar) and 6.40e-12 (vector), the same order as the unrepaired port
  against the corpus on the same arrays (5.49e-15, 2.33e-12, 1.50e-12).
- **A pion exactly at rest returns zero**, a pre-existing kernel property
  (`rust/src/kernels/positron_pion.rs`, `dnde_positron_charged_pion`), so
  A4 does not reach the pion's `rest` block at all.

## Decisions and Implementation Notes

- **Three model variants, one per case budget** — `A4` (`EXACT_RTOL`,
  muon), `A4/pion` (`PORTED_QUAD_RTOL`), `A4/nested`
  (`PORTED_NESTED_RTOL`). One relation at the loosest budget would widen
  the muon's bit-for-bit contract, which `../rules.md` rule 2 forbids;
  A3 and `A3/nested` are the precedent.
- **Keys are generated as a product, not transcribed.** The 178 arrays
  are blocks × muon-fed channels × mediator masses; `_A4_BLOCKS` and
  `_A4_MEDIATOR_CHANNELS` in `test/parity/deltas.py` state the
  exclusions with their reasons. `EXPECTED_DECLARED_ARRAYS` (343) still
  makes the size show up in a diff, and a mutation that adds `e_e` or the
  pion `rest` block fails the staleness rule (Verification).
- **The follow-up stays in `todo/`**, marked repaired, for Task 12's
  single relocation sweep — the convention Tasks 4–9 and 10a followed.
- **Review round 1 (PR #99).** The `CHANGELOG.md` entry names
  `SingleChannelAnn("mu mu")` and `KineticMixingGeV` as downstream
  consumers; both route their `mu mu` positrons through
  `2 * dnde_positron_muon` alone (ratio to that expression measured
  exactly 1 on this build), so they rise by exactly `R_FACTOR²`. A
  reviewer's observation that `RHNeutrino` positron spectra are all zero
  is a pre-existing late-binding closure in
  `TheoryDec.positron_spectrum_funcs`, unmoved by this task and filed as
  `docs/followups/todo/decaying-theory-positron-channels-share-the-last-closure.md`.
- **No `PLAN.md`, rules or ADR change.** The Task 10 gate and the
  Numerical-impact bullet (`R_FACTOR²`) are accurate as written.

## Files Changed

- `rust/src/kernels/positron_muon.rs` — multiply by `R_FACTOR` in both
  expressions, in the capture's operation order; module doc; the rest and
  in-flight norm tests assert 1 and rule out `1/N²`.
- `rust/src/kernels/positron_pion.rs` — module doc; the boost-conservation
  test expects `BR_μ + BR_e`.
- `rust/src/kernels/neutrino_muon.rs` — doc comments that described the
  positron sibling as still inverted.
- `test/test_core_positron_muon.py`, `test/test_core_positron_pion.py`,
  `test/test_core_neutrino.py` — the same flip on the Python side; module
  docstrings.
- `test/parity/deltas.py` — `_A4`, `_A4_PION`, `_A4_NESTED`, the 178 keys,
  and three `DELTA_MODELS` entries.
- `test/parity/test_parity.py` — `EXPECTED_DECLARED_ARRAYS` 165 → 343.
- `CHANGELOG.md` — the `[Unreleased]` entry with the magnitude.
- `docs/followups/todo/positron-muon-spectrum-normalization-inverted.md`
  — status, renamed tests, the operation-order answer.
- `../references/defect-blast-radius.md` — A4's measured reach.
- `README.md` (working memory) and this note.

## Numerical impact

Measured from two builds of this worktree in one environment: the
editable install at trunk `8124b277` (defective), then the repaired tree
after `uv pip install -e . --config-settings build-args="--features
test-probes"`. Import paths were asserted inside the worktree. No corpus
or oracle array was regenerated.

**Corpus** (all 623 blocks, 181,191 evaluated values including scalar
probes):

| Case | Values moved | Arrays moved | Max abs shift (MeV⁻¹) | Oracle residual (max rel) |
| --- | --- | --- | --- | --- |
| `spectra.positron.muon` | 502 | 10 of 10 | 1.416302e-05 | 0 (bit-equal, 1,370 values) |
| `spectra.positron.charged_pion` | 525 | 8 of 10 | 1.158413e-05 | 5.296e-15 |
| `…scalar.positron.dnde_decay_s` | 5,237 | 40 of 60 | 7.965369e-06 | 3.546e-12 |
| `…scalar.positron.dnde_decay_s_pt` | 5,237 | 40 of 60 | 7.965369e-06 | 3.546e-12 |
| `…vector.positron.dnde_decay_v` | 5,237 | 40 of 60 | 3.348919e-06 | 6.400e-12 |
| `…vector.positron.dnde_decay_v_pt` | 5,237 | 40 of 60 | 3.348919e-06 | 6.400e-12 |
| **total** | **21,975** | **178** | | |

Every moved value rises. The after/before ratio runs from 1 + 8e-15 to
1.000374206650, median exactly `R_FACTOR² = 1.000374206647938`; the ratio
is below `R_FACTOR²` only where a line the muon kernel does not feed
(`π → e ν`, the mediators' `e e`) shares the position. No value outside
these six cases moves, and no grid, exception or NaN pattern changes.

**Public API**, 2,001-point log grid from 0.01 MeV to 5 GeV, NaN-aware:

| Function | Parent | Moved | After/before ratio | Max abs shift (MeV⁻¹) |
| --- | --- | --- | --- | --- |
| `dnde_positron_muon` | `m_μ`, `1.5 m_μ`, `5 m_μ` | 707, 854, 1,056 | exactly 1.000374206647938 | 1.416254e-05 (rest) |
| `dnde_positron_charged_pion` | `1.5 m_π`, `5 m_π` | 896, 1,099 | 1.0000299 – 1.000374206648 | 4.892869e-06 |
| `dnde_positron_charged_pion` | `m_π` | 0 | — (returns zero at rest) | 0 |
| `dnde_positron`, `["mu","mu"]`, `["mu","e"]` | cme 500, 2000 MeV | 937–1,155 | exactly `R_FACTOR²` | 5.480161e-06 |
| `dnde_positron`, `["pi","pi"]` | cme 500, 2000 MeV | 930, 1,154 | 1.00013875 – 1.00037421 | 7.632233e-06 |
| `HiggsPortal.total_positron_spectrum` | `mx=400, ms=1000`, cme 900 | 1,031 | 1.00034144 – 1.00037421 | 1.265910e-06 |
| `KineticMixing.total_positron_spectrum` | `mx=400, mv=1000`, cme 900 | 1,030 | 1.0000000061 – 1.00036806 | 1.840095e-07 |

Unchanged: the nine other `dnde_positron_*` meson spectra (three kaons,
η, η′, ω, φ, both ρ) at rest, `1.5 m` and `5 m`, and
`dnde_neutrino_muon` / `dnde_neutrino_charged_pion` at `5 m`.

**Physics invariant** (`../rules.md` rule 4): the repaired Michel
spectrum integrates to one positron per decay.

```text
scipy.integrate.quad(dnde_positron_muon, m_e, endpoint, epsrel=1e-13)
E_mu = m_mu      0.9999999999999996
E_mu = 1.5 m_mu  1.0000000000000013
E_mu = 5 m_mu    0.9999999999999717
```

Shipped: `1/R_FACTOR² = 0.999626`.

**Reproduction.** The corpus capture is the capture script in
`task-9-rho-rest-frame.md`, Reproduction, run unchanged against each
build. The comparison differs from Task 9's only in what it asserts:
moved values are confined to the six A4 cases and all rise, and the
repaired capture is compared against `oracle_reference._blocks("A4")`
array by array. The public grid evaluates each `dnde_positron_*` at
`np.geomspace(1e-2, 5e3, 2001)` MeV with the parents in the table, and
compares NaN-aware (the tabulated spectra at rest return `NaN` at 600 of
the 2,001 points in both builds).

## Verification

Preflight, with the worktree's `.venv/bin` first on `PATH` and no
`--tests`, so pytest ran the full suite:

```sh
scripts/agents/preflight.sh --paths "test/parity/deltas.py test/parity/test_parity.py test/test_core_positron_muon.py test/test_core_positron_pion.py test/test_core_neutrino.py" --md "CHANGELOG.md docs/followups/todo/positron-muon-spectrum-normalization-inverted.md projects/parity-pinned-defect-repair/references/defect-blast-radius.md projects/parity-pinned-defect-repair/task-notes/README.md projects/parity-pinned-defect-repair/task-notes/task-10-positron-muon-norm.md"
```

```text
PASS black, isort (0 new), ruff (0 new), cargo fmt, cargo clippy, cargo test
PASS pytest: 2314 passed, 16 skipped, 1 warning, 37 subtests passed in 26.41s
PASS import hazma (version 2.2.0), markdownlint, forbidden tokens
SKIP version bump: not a closing PR
RESULT: PASS
```

The gate rows are condensed; the pytest line and the result are verbatim.
This note's final sections were written after that run and rechecked with
markdownlint alone.

Targeted runs on the repaired build:

```text
pytest test/parity -q                                   702 passed, 1 skipped
pytest test/test_core_positron_muon.py test/test_core_positron_pion.py test/test_core_neutrino.py -n 0
                                                        110 passed
pytest test/test_theory_aggregation.py -q               69 passed
cargo test ... -q positron                              31 passed
cargo test ... -q neutrino_muon                         10 passed
python test/parity/generate.py --check                  corpus OK: 41 cases / 1580 arrays match the manifest
python test/parity/oracles/capture.py --check           oracles OK: 4 defects / 940 arrays match the manifest
git diff --stat -- test/parity/data test/parity/oracles test/parity/tolerances.py   (empty)
```

What the tests cover: the Michel normalization at rest and in flight
(Rust Simpson and trapezoid; Python trapezoid at four muon energies),
each of which also rules out `1/N²`; the pion's boost conservation at
`BR_μ + BR_e`; the neutrino sibling's unchanged norm; and the corpus
comparison of all 178 declared arrays against the A4 capture, with every
undeclared position still held to the stored value.

**Mutations.**

- *Revert*: the two kernel lines restored to division, the editable
  install rebuilt, then `pytest test/parity test/test_core_positron_muon.py
  test/test_core_positron_pion.py -q` gave `176 failed, 578 passed, 1
  skipped`: 169 corpus blocks (5 muon, 4 pion, 160 mediator) that fail
  the A4 relation, and the 7 normalization tests. Restoring the repair and
  rebuilding gave `754 passed, 1 skipped`.
- *Widen*: adding `e_e` to the 250 MeV mediator channels and the pion's
  `rest` block to the allowlist fails `test_entry_point_matches_corpus` on
  each added block (stale declaration), as the staleness rule requires.

## Open Questions

None. The mediator positron line's separate electron-velocity defect
([`mediator-positron-line-misses-the-electron-velocity.md`](../../../docs/followups/todo/mediator-positron-line-misses-the-electron-velocity.md))
is not on this roster and is not touched here.

## Plan Impact

**Impact Level:** None. The Task 10 gate in `../PLAN.md` is met as
written: six cases declared, the capture reproduced, the Task 4.1
normalization test flipped, the Michel integral 1. Checked against
ADR-0001 (a `Reference` relation to a Task 2 capture, no widened budget)
and ADR-0002 (photon muon only; not touched).

## Stale-state sweep

```text
$ rg -n "carries_the_inverted_normalization|shipped_inverted_normalization" -g '!projects/cython-to-rust/**' .
(no occurrences)
$ rg -n "\b165\b" test/parity projects/parity-pinned-defect-repair (excluding task notes)
task-notes/README.md: (`EXPECTED_DECLARED_ARRAYS` 165 → 343)   -- the Files Changed roll-up, intended
$ rg -n "A4 repair is next|Task 10.*Not started" projects/parity-pinned-defect-repair
(no occurrences)
$ git diff -U0 | grep -iE "for now|as discussed|per the plan|as requested|in this task|FIXME|breakpoint\(|pdb|print\("
(no occurrences)
$ Counter(d.repair for d in deltas.DECLARED_DELTAS.values())
343: A1 44, A1+B1 6, A1+B2 6, A2 1, A3 62, A3+B3 4, A3+B4 20, A4 178, B4 10, B5 6, B6 6
```

- **Numerical-impact statement:** `dnde_positron_muon` rises by exactly
  `R_FACTOR² = 1.000374206647938` (+0.0374%) at every nonzero value, and
  its consumers by up to that factor; 21,975 corpus values move. Logged
  in the README's "Numerical impact so far" and in `CHANGELOG.md`.
- The Tasks-table cell, this note's status, the README handoff (343
  arrays, A4 landed) and the follow-up status agree.
- The roster table in `../references/defect-blast-radius.md` still
  labels the Group A twins "(live)" for A1–A4; that stale prose predates
  this task and is Task 11's reconciliation.

## Handoff to Next Task

- All ten roster repairs have landed. Task 11 (reconcile the superseded
  sequencing prose) is next, then Task 12, which renames the
  `[Unreleased]` CHANGELOG heading, moves the repaired follow-ups to
  `done/` in one sweep, and bumps the version.
- `test/parity/deltas.py` declares 343 arrays; A4 overlaps no other
  repair, so it composes with nothing.
- Task 12's aggregation reads A4's figures from the README's "Numerical
  impact so far" and the table above.
