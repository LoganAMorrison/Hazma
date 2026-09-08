# Task 4: Repair A1 — the boost integral window

**Date:** 2026-09-07
**Project:** parity-pinned-defect-repair
**Status:** Complete
**Plan References:** `../PLAN.md` "Task 4", "Numerical impact"; `../rules.md`
rules 1–7, 10, 11
**Related ADRs:** `../adrs/ADR-0001-corpus-repairs-are-declared-deltas.md`
**Depends On:** Task 1 (the delta layer), Task 2 (the Cython oracle)

## Objective

Cover `[x[ihigh-1], x[ihigh]]` and stop dropping the table's final row in
`boost_integrate_linear_interp`, and declare the resulting delta on the
seven tabulated photon cases the A1 row names — and on nothing else.

## Exit Criteria

- Declared deltas on all seven cases of the A1 row of
  `../references/defect-blast-radius.md`, and on no other corpus array.
- The repaired value reproduces Task 2's Cython oracle within the
  function's existing budget.
- The measured sign matches `../PLAN.md`'s Task 4 gate: `rest_plus_eps`
  down at all 1,156 positions, the three boosted blocks up at 2,997 of
  2,998, `rest` unmoved.
- Every other corpus case unchanged against its original stored array.
- A physics invariant the corpus cannot supply (`../rules.md` rule 4).
- `git diff --stat -- test/parity/data` empty (`../rules.md` rule 1).

## Inputs Reviewed

- `../PLAN.md` (Task 4, Numerical impact, Scope), `../rules.md`, the head
  of `task-notes/README.md`.
- `../references/defect-blast-radius.md` §A1; `../references/corpus-repinning.md`
  is implemented by `test/parity/deltas.py`, which was read instead.
- `../adrs/ADR-0001-corpus-repairs-are-declared-deltas.md`.
- `../task-notes/task-2-cython-oracles.md` §A1 and the per-block sign table.
- `test/parity/oracles/patches/A1-boost-integral-window.patch` — the
  repair, stated exactly, on the Cython the oracle was captured from.
- `docs/followups/todo/boost-integral-drops-last-interior-cell.md`.
- `rust/src/boost.rs`, `test/test_core_boost.py`,
  `test/test_core_photon_tables.py`, `test/parity/{deltas,test_parity,
  tolerances,stability}.py`, `test/parity/oracles/capture.py`.
- `docs/agents/lessons.md` — `[exemption-wider-than-its-mechanism]`,
  `[measured-tree-vs-imported-module]`, `[test-name-claims-an-unmade-assertion]`.

## Findings

- **The repair is two lines of algorithm, not three.** The follow-up asks
  to "re-derive the upper partial-cell term so the two do not overlap".
  It needs no re-derivation: that term already anchors on `x[ihigh]`, so
  making the interior sum inclusive of `x[ihigh]` is enough to make the
  two meet exactly. The Task 2 patch had already reached the same
  conclusion and says so in its own comment; the "What" section of the
  follow-up is the stale half.
- **The repaired Rust reproduces the Cython oracle bit for bit at all
  10,045 positions**, and the unrepaired Rust reproduced the corpus bit
  for bit at the same 10,045. So this platform contributes no drift to
  either comparison, and the whole 4,154-position move is the repair.
  That also settles the FMA question the single-cell branch raises: the
  Cython's generated C contracts the new branch exactly as it contracts
  the two partial-cell terms beside it, so spelling it with `mul_add` in
  Rust — as the siblings are — is what reproduces it.
- **`ilow > ihigh` can only mean `ihigh == ilow - 1`.** `ilow` is the
  first node at or above `lb` and `ihigh` is at least `first - 1` where
  `first` is the first node at or above `ub`; with `lb <= ub` that gives
  `ihigh >= ilow - 1`. The `usize` underflow the branch looks like it
  invites cannot happen either: `first == 0` requires `ub == x[0]`
  exactly, and the 1e-6 edge test keeps `ihigh` at 0 there.
- **The two `rest`-block exclusions are one fact, and it is now
  measured.** No caller reaches the integral at `β = 0` — all seven
  short-circuit at `E − m < DBL_EPSILON` — so all 1,841 `rest` positions
  are undeclared and still gated at the case's own budget. That closes
  the README's open question, which Task 2 had answered from the capture
  and this task confirms against the repaired kernel.
- **The yield invariant needs a table that vanishes at `x[0]`.** The
  below-table `1/E` tail is an extrapolation *below* the tabulated
  support and is still reproduced from the Cython rather than repaired,
  so it adds photons the table does not contain. With `y[0] = 0` the tail
  term is identically zero and `∫dE dN/dE = ∫dx y(x)` holds to the outer
  rule's discretization error. A table with a nonzero first row would
  have been testing the extrapolation instead.
- **A `Reference` relation reading a committed capture needs the entry
  point to identify its case**, because `Delta` is one object per roster
  label (`test_parity.test_every_declaration_points_at_a_delta_model`
  compares by `id`) while a capture is keyed by case. The corpus
  manifest's `entry_point` cannot do it — it records where each case was
  *captured* from, which for all seven is a `.pyx` the port deleted.
  `cases.build_cases()` is the live authority, and reading it keeps the
  case list derived rather than transcribed.

## Decisions and Implementation Notes

- **The single-cell case is an `else` branch, not an early return.** The
  Cython patch returns early; structuring it as `if ilow > ihigh { … }
  else { … }` around the three existing terms makes the mutual exclusion
  structural instead of implied, and keeps one `/(2γβ)` at the end.
  Bit-identical either way — control flow only.
- **`oracle_reference.py` is a new sibling module rather than a function
  in `deltas.py`.** A1–A4 all need exactly this reader, and it is file
  I/O against a committed capture rather than a closed form, which is the
  kind of thing `thermal_reference.py` is already a precedent for.
- **The relation's `rtol` is `tolerances.TABULATED_RTOL` (1e-12), not
  zero.** The measured agreement is exact, but the capture is one
  platform's; a libm that moves the boost integral must be held here
  exactly as tightly as the undeclared positions of the same array are,
  and no tighter (`../rules.md` rule 2 forbids the other direction).
- **The follow-up stays in `docs/followups/todo/`**, matching B5's
  precedent: the project's Exit Criteria repoint all inbound links in one
  sweep at Task 12. Its status line, its two dangling test references and
  its superseded "regenerate the corpus" prescription are corrected in
  place.
- **`TestDroppedInteriorCell` is renamed, not deleted.** Its two tests
  assert the same two hand-computable cases at their correct values, so
  the name had to move with the assertion
  (`[test-name-claims-an-unmade-assertion]`).
- **`a_flat_table_boosts_to_the_log_of_the_window`'s budget tightened
  1e-4 → 1e-9.** Its own doc comment said the 1e-4 existed to let the
  dropped cell through. With the cell covered, what remains is the
  composite trapezoid error `(h²/12)(f′(ub) − f′(lb))` = 9.6e-10
  absolute, 5.1e-10 relative, measured.

## Files Changed

- `rust/src/boost.rs` — the repair (an inclusive interior sum and the
  single-cell closed form), a rewritten `# Faithfulness notes`, the
  renamed coverage test, a new yield invariant, and the tightened
  flat-table budget.
- `test/parity/oracle_reference.py` (new) — the Group A captures as a
  `deltas.Reference`.
- `test/parity/deltas.py` — the `A1` declaration, its 56 keys, `A1` in
  `DELTA_MODELS`, and the module's relation roster, which had not
  mentioned `Reference` since Task 13 added it.
- `test/parity/test_parity.py` — `EXPECTED_DECLARED_ARRAYS` 42 → 98.
- `test/parity/tolerances.py` — the `TABULATED` class's account of the
  algorithm, which now has a third closed form.
- `test/test_core_boost.py` — `integrate_reference` repaired in lockstep
  (it is the independent oracle for the kernel), `TestDroppedInteriorCell`
  → `TestWindowCoverage` inverted, `TestTrapezoidSummation`'s expectation,
  and two doc cross-references.
- `test/test_core_photon_tables.py` — `MIN_THRESHOLD_DIVERGENCE` →
  `MAX_THRESHOLD_MISS` + `THRESHOLD_FRACTIONS`, and the divergence pin
  inverted into the convergence acceptance test over all seven channels.
- `CHANGELOG.md` — an `[Unreleased] Changed` entry with the magnitudes,
  and a pointer on the 2.2.0 `Known issues` entry.
- `docs/followups/todo/boost-integral-drops-last-interior-cell.md` —
  status, the two renamed tests, and the superseded regeneration.
- `../PLAN.md` — the "Numerical impact" bullet, which still said the
  shift was one-signed.
- `../task-notes/README.md` — status, findings, numerical impact, open
  questions, files changed, handoff.

## Numerical impact

**Public values move, by design.** The measurement is the corpus itself,
taken against a build carrying the defect rather than against the stored
arrays, per the README's recipe: the defective build reproduced the
stored corpus at **all 10,045** A1 positions, so every difference below
is the repair.

```sh
# The whole corpus, 41 cases / 179,695 pinned values:
pytest test/parity -q                      → 687 passed, 1 skipped
```

Only the 4,154 declared positions moved; every other case is still
compared against its stored array under its own budget, and passed.

| Block | γ | moved / pinned | up | down |
| --- | --- | --- | --- | --- |
| `rest` | 1 | 0 / 1841 | 0 | 0 |
| `rest_plus_eps` | 1 + 1e-12 | 1156 / 2051 | 0 | 1156 |
| `near_rest` | 1.05 | 1130 / 2051 | 1130 | 0 |
| `boosted_mild` | 2 | 1028 / 2051 | 1028 | 0 |
| `boosted_strong` | 10 | 840 / 2051 | 839 | 1 |

| Case | moved / pinned | max abs shift | rest |
| --- | --- | --- | --- |
| `spectra.photon.eta` | 560 / 1435 | 6209.45 | 0 |
| `spectra.photon.eta_prime` | 552 / 1435 | 2502.46 | 0 |
| `spectra.photon.omega` | 633 / 1435 | 481699 | 0 |
| `spectra.photon.phi` | 551 / 1435 | 3435.51 | 0 |
| `spectra.photon.charged_kaon` | 631 / 1435 | 483975 | 0 |
| `spectra.photon.long_kaon` | 631 / 1435 | 659277 | 0 |
| `spectra.photon.short_kaon` | 596 / 1435 | 587910 | 0 |

Magnitudes, in the two regimes the sign splits over:

- `rest_plus_eps`: the shipped value is a **median 9,768x** and up to
  **360,507x** too high, and every one of its 1,156 positions falls.
- The three boosted blocks rise, by a **median 3.34e-02 at γ = 1.05,
  2.19e-03 at γ = 2 and 7.09e-05 at γ = 10**, and by up to 98.7%.
- Public limit, off the corpus grid: at a parent one part in 1e12 above
  rest, all seven channels now agree with their own rest-frame spectrum
  to better than 1% over `E_γ` from `m/20` to `3m/10` (worst 8.5e-3, eta
  at `m/20`). The follow-up measured 6,500x to 33,000x before.

**Unchanged:** every other corpus case, including all four
`mediator_spectra.*.photon.*` — the tabulated seven are not on their
composition path, which `pytest test/parity` asserts case by case rather
than this task inferring. `git diff --stat -- test/parity/data` is empty.

`../PLAN.md`'s `version_bump: minor` is unchanged and correct: published
numbers move, no public name, signature, return shape or documented unit
does.

## Verification

Built in the worktree, and confirmed to be what was measured:

```sh
uv pip install --python .venv/bin/python -e . \
    --config-setting build-args="--features test-probes"
python -c "import hazma._core; print(hazma._core.__file__)"
  → .../parity-pinned-defect-repair-36ba90/hazma/_core.abi3.so
```

```text
cargo test --manifest-path rust/Cargo.toml --no-default-features \
    --features test-probes          → test result: ok. 263 passed; 0 failed
pytest test/parity -q               → 687 passed, 1 skipped
pytest test/test_core_boost.py -q   → 51 passed
pytest test/test_core_photon_tables.py -q → 190 passed, 1 skipped
pytest -q                           → 2286 passed, 15 skipped, 1 warning,
                                       37 subtests passed
scripts/agents/preflight.sh --paths … --md …
                                    → RESULT: PASS, all eleven rows
```

All 15 skips are pre-existing and none is A1's: 11 "known to be broken"
form-factor tests, the charged kaon's absent monochromatic line, and the
parity corpus's declared-budget skip (`numpy 2.5.1 → 2.5.3`,
`scipy 1.18.0 → 1.18.1`, a macOS point release). A bare `pytest -q`
reports 24 instead, because `python -m pytest` leaves `.venv/bin` off
`PATH` and `test/agents/test_lint_delta.py` skips its nine subprocess
tests when it cannot find `ruff` and `isort`.

What the new and changed tests cover:

- **The two hand-computable cases**, at their correct values, in both
  languages: `boost::tests::a_clamped_window_covers_the_tables_final_row`
  and `a_window_inside_one_cell_is_the_window`;
  `TestWindowCoverage`'s three tests, each also asserting the port equals
  `integrate_reference`.
- **The physics invariant** (`../rules.md` rule 4):
  `boost::tests::the_boost_conserves_the_tabulated_yield` — the boosted
  spectrum integrated over the lab energy returns the table's own
  integral, at three boosts. Owes nothing to the Cython or to the corpus.
- **The public acceptance test** the follow-up proposed:
  `test_a_barely_moving_parent_converges_to_its_rest_frame_spectrum`,
  over all seven channels at four energies each.
- **The declaration**, through `test_entry_point_matches_corpus` on 28
  blocks: the relation's prediction at the declared positions, the stored
  value at every other position of the same array, and the staleness rule
  that fails a declaration whose repair is reverted.
- **The reduction order**, unchanged: `TestTrapezoidSummation` still
  compares against the live `np.trapezoid`, now over the inclusive slice.

**Test validity (stash-proof).** The kernel half of `rust/src/boost.rs`
was replaced with `git show HEAD:rust/src/boost.rs`'s, keeping the new
tests, and the tree rebuilt. Every layer turned red:

```text
cargo test … boost::   → 4 failed  (both coverage tests, the yield
                          invariant, the tightened flat-table budget)
pytest test/test_core_boost.py test/test_core_photon_tables.py \
       test/parity -q  → 56 failed, 872 passed, 2 skipped
```

grouped as 28 `test_entry_point_matches_corpus` blocks (7 cases x 4
declared blocks — the declaration is not stale), 7
`test_a_barely_moving_parent_converges_to_its_rest_frame_spectrum`, 7
`test_the_tabulated_port_is_the_fused_reference`, 5
`test_the_interior_sum_is_numpys_trapezoid`, 3 `TestWindowCoverage`, and
5 more `TestBoostIntegrateLinearInterp` sensitivity checks. The kernel was
then restored and `cmp`-verified byte-identical before rebuilding.

**Nothing deferred.**

## Open Questions

- **Do B1's and B2's positions overlap A1's on `eta_prime` and `phi`?**
  Still open, and now sharper: A1 declares `MOVED` over all four non-rest
  blocks of both cases, so Tasks 5 and 6 cannot simply add keys — every
  array B1 or B2 moves is already declared under A1, which `../rules.md`
  rule 7 forbids leaving as an overlap. The two must collapse into one
  composite declaration per array, or B1/B2 must be shown to move only
  positions A1 leaves alone, which `MOVED` makes impossible to express as
  a second key. This is the one structural consequence this task hands
  forward.
- **Should the `rest` blocks stay undeclared once B1 and B2 land?** B1
  never reaches `eta_prime[rest]` and B2 reaches neither of phi's rest
  blocks (Task 3), and A1 reaches no `rest` block at all, so all seven
  `rest` blocks should still be compared against the stored array
  afterwards. Worth re-checking rather than assuming when Tasks 5 and 6
  measure.

## Plan Impact

**Impact Level:** Update `PLAN.md`.

`PLAN.md`'s "Numerical impact" bullet for the boost integral asserted the
shift was "systematic and one-signed (they are currently always slightly
low)" — the follow-up's reading, which describes only the away-from-
threshold regime. Task 4's Task Details block already carried the
correction Task 2 measured; the summary bullet did not, and is now
patched to the measured figures. No task ordering, gate or interface
changed, and no ADR is needed: ADR-0001's mechanism absorbed a
capture-backed `Reference` relation without amendment.

## Stale-state sweep

### Identifier sweep

`rg -n --hidden '<id>' projects/ docs/ README.md
hazma/ test/ rust/ .claude/ .codex/ CHANGELOG.md | cut -d: -f1 | sort |
uniq -c`, run after the last prose edit. This note names each removed
identifier once or twice in order to report it, so its own hits are
listed and excluded.

| Identifier | Hits outside this note | Disposition |
| --- | --- | --- |
| `TestDroppedInteriorCell` | 1 — `projects/cython-to-rust/.../task-3.4-interp-boost.md` | EDITED — renamed everywhere live; the survivor is the frozen project, which Task 11 owns |
| `the_last_interior_cell_is_dropped` | 0 | DELETED |
| `MIN_THRESHOLD_DIVERGENCE` | 0 | DELETED |
| `test_the_boost_integral_still_diverges_near_threshold` | 0 | DELETED |
| `TestWindowCoverage` | 5 — `test/test_core_boost.py` (3), the follow-up (2) | KEPT — definition, two cross-references, two repointed follow-up entries |
| `oracle_reference` | 9 — `deltas.py` (3), `test/parity/README.md` (2), `../task-notes/README.md` (4) | KEPT — the new module and everything that names it |
| `MAX_THRESHOLD_MISS` / `THRESHOLD_FRACTIONS` | 2 / 3 — `test/test_core_photon_tables.py` | KEPT — definition plus uses |
| `A1_CASES` | 2 — `deltas.py`, the count command in `../references/defect-blast-radius.md` | KEPT |
| `a_clamped_window_covers_the_tables_final_row`, `a_window_inside_one_cell_is_the_window`, `the_boost_conserves_the_tabulated_yield` | 1 each — `rust/src/boost.rs` | KEPT — definition only |

### Line-number citation sweep

```text
$ python scripts/agents/check_doc_citations.py <the 7 docs this task touched>
docs scanned: 7
in-repo citations checked: 0
external citations skipped: 2
  hazma/_utils/boost.pyx (2)
out-of-range or ambiguous: NONE
```

The two skipped are the follow-up's citations of the deleted Cython, kept
deliberately as the historical record of where the defect lived.

### Forward-looking phrase sweep

```text
$ rg -n '(Task [0-9]+ will|will be added|still pending|today: ?stub|In Progress)' \
    task-4-boost-window.md test/parity/{oracle_reference,deltas}.py \
    rust/src/boost.rs test/test_core_{boost,photon_tables}.py
(no matches)
```

### Count sweep

| Claim location | Command | Actual | Status |
| --- | --- | --- | --- |
| `test_parity.EXPECTED_DECLARED_ARRAYS = 98` | `python -c "…; print(len(deltas.DECLARED_DELTAS))"` | 98 | OK |
| "56 for A1" (`../task-notes/README.md`, this note) | `python -c "…; print(sum(1 for d in deltas.DECLARED_DELTAS.values() if d.repair=='A1'))"` | 56 | OK |
| "0 of its 1841 `rest` positions" (`../references/defect-blast-radius.md`) | the command printed beside it there | 1841 | OK — was 1750, re-derived and corrected |
| "10,045 pinned positions" | the per-block table's column sum, `1841 + 4 x 2051` | 10045 | OK |
| "4,154 move" | `impact.py` over the two builds; equals Task 2's independent total | 4154 | OK |
| "70 arrays" in the A1 capture | `python -c "print(len(np.load('test/parity/oracles/data/A1.npz').files))"` | 70 | OK — 7 cases x 5 blocks x 2 suffixes |
| `git diff --stat -- test/parity/data` empty (rule 1) | that command | 0 lines | OK |

### Numerical-impact statement

the grid is the parity corpus itself,
all 41 cases and 179,695 pinned values, evaluated live and compared
against the stored arrays by `pytest test/parity -q` (687 passed, 1
skipped). Public values moved: 4,154 of the 10,045 belonging to the seven
tabulated photon cases, in the four non-`rest` blocks of each, down in
`rest_plus_eps` and up in the three boosted blocks. No other case moved —
every one of them is still compared against its stored array under its
own budget and passed. Magnitudes are in `## Numerical impact` above and
in `CHANGELOG.md`'s `[Unreleased]` entry.

### Exit Criteria → what satisfies it

| Criterion | Satisfied by |
| --- | --- |
| Declared deltas on all seven A1 cases and no other array | `deltas.DECLARED_DELTAS`'s 56 `A1` keys; `test_parity.EXPECTED_DECLARED_ARRAYS = 98`; `test_entry_point_matches_corpus` green on all 41 cases |
| Reproduces Task 2's oracle within the existing budget | `cmp_a1.py` — 0 of 10,045 positions differ from the capture; the relation carries `TABULATED_RTOL` |
| The measured sign matches the Task 4 gate | the per-block table in `## Numerical impact`: 1156 down, 2997 of 2998 up, `rest` 0 |
| Every other corpus case unchanged | `pytest test/parity -q` → 687 passed, 1 skipped |
| A physics invariant | `boost::tests::the_boost_conserves_the_tabulated_yield`, plus `test_a_barely_moving_parent_converges_to_its_rest_frame_spectrum` over seven channels |
| `git diff --stat -- test/parity/data` empty | that command, 0 lines |

### Task-note self-consistency

`**Status:** Complete` matches the
Tasks-table cell in `../task-notes/README.md`. Every file named in §Files
Changed appears in `git diff origin/master --stat` (14 files, two of them
new), and every symbol cited in §Findings and §Decisions —
`boost_integrate_linear_interp`, `oracle_reference.captured`,
`cases.build_cases`, `deltas.DECLARED_DELTAS`, `DELTA_MODELS`,
`Reference`, `tolerances.TABULATED_RTOL`, `TestWindowCoverage`,
`a_flat_table_boosts_to_the_log_of_the_window` — resolves in the tree.

## Handoff to Next Task

**Tasks 5 (B1) and 6 (B2) are unblocked and have a new constraint.**

- Read this note's first Open Question before anything else. A1 declares
  `MOVED` on all four non-rest blocks of `spectra.photon.eta_prime` and
  `spectra.photon.phi`, both suffixes — the eight arrays B1 and B2 land
  on are already declared. `../rules.md` rule 7 requires one composite
  declaration per shared array, not a second key; `deltas.DECLARED_DELTAS`
  maps one key to one `Delta`, so a second key is not expressible anyway.
  The composite's relation is the capture *plus* the closed-form term
  (`test/parity/oracles/data/A1.npz` carries the boost repair alone,
  since the Cython twin it was captured from has B1's and B2's defects).
- `test/parity/oracle_reference.py` is the reader for A2, A3 and A4 as
  well: `oracle_reference.captured("A2")` needs nothing new, and resolves
  its own case list from the oracle manifest.
- `EXPECTED_DECLARED_ARRAYS` is 98. Re-derive it with
  `python -c "import sys; sys.path.insert(0,'test/parity'); import deltas;
  print(len(deltas.DECLARED_DELTAS))"` rather than adding to it by hand.
- The measurement recipe held: the defective build reproduced the stored
  corpus bit for bit on all 10,045 A1 positions, so live-vs-stored and
  before-vs-after agreed here. That is this platform, not a general fact —
  keep capturing twice.
- Tasks 7 (A2), 8 (A3) and 10 (A4) are the other three `Reference`
  declarations and now have a worked precedent end to end.
