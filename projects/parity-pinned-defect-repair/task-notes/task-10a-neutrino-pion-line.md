# Task 10a: Repair B5 — the charged pion's doubled neutrino line

**Date:** 2026-09-06
**Project:** parity-pinned-defect-repair
**Status:** Complete
**Plan References:** `../PLAN.md` Task 10a, Scope, Numerical impact;
`../rules.md` rules 1-7, 10, 11
**Related ADRs:** `../adrs/ADR-0001-corpus-repairs-are-declared-deltas.md`
**Depends On:** Task 1 (the delta-declaration layer). Not Task 2 — B5's
Cython twin was deleted before this roster entry existed, so there is no
oracle to capture.

## Objective

Make `dnde_e_nue` the sole source of the `π → e ν_e` line in
`rust/src/kernels/neutrino_pion.rs`, so each prompt line is counted once,
and declare the corpus positions the change moves.

## Exit Criteria

- The `delta_e` binding is gone from `dnde_mu_numu` and from its returned
  `electron` field; `dnde_e_nue` is unchanged.
- A declared delta on `spectra.neutrino.charged_pion` and no other case,
  covering exactly the positions the boosted line reaches. `rest` and
  `rest_plus_eps` still match their stored arrays under the case's own
  budget.
- `git diff --stat -- test/parity/data` empty (rule 1).
- No tolerance widened (rule 2).
- An independent oracle: the closed form, owing nothing to the kernel
  (rule 3).
- A physics invariant: the electron-neutrino row integrates to
  `BR_μ + BR_e`, and the muon row is bit-identical (rule 4).
- The two Rust tests and the two Python tests that pinned the defect are
  renamed as well as re-pointed
  (`docs/agents/lessons.md` `[test-name-claims-an-unmade-assertion]`).
- The magnitude recorded in `CHANGELOG.md` and in the project's
  "Numerical impact so far" (rule 10).

## Inputs Reviewed

- `docs/followups/todo/neutrino-pion-electron-line-counted-twice.md` —
  the defect, its entry points, its predicted magnitude.
- `../PLAN.md` (all sections), `../rules.md`,
  `../references/corpus-repinning.md` §"Proof obligations",
  `../references/defect-blast-radius.md`.
- `../task-notes/README.md` — Tasks table, Findings, Open Questions,
  Handoff.
- `rust/src/kernels/neutrino_pion.rs` — module docs, `dnde_mu_numu`,
  `dnde_e_nue`, `dnde_neutrino_charged_pion`, the test module.
- `rust/src/boost.rs` — `boost_beta`, `boost_delta_function`.
- `test/parity/deltas.py`, `test/parity/test_parity.py`
  (`_assert_declared_delta`), `test/parity/tolerances.py`,
  `test/parity/cases.py` (`_spectrum_case`, `Block`).
- `test/test_core_neutrino.py` — the module docstring and `TestPhysics`.
- `docs/agents/environment.md` — the build recipe and the rebuild trap.

## Findings

- **The corpus radius is one case and three of its five blocks.** 215 of
  the 4,305 pinned values of `spectra.neutrino.charged_pion` move: 35 in
  `near_rest`, 69 in `boosted_mild`, 111 in `boosted_strong`, and none in
  `rest` or `rest_plus_eps`. All 215 are in the electron row and all move
  down. `spectra.neutrino.muon` does not move, which makes B5 disjoint
  from every other roster entry and retires this file's claim that
  neither `spectra.neutrino.*` case had a defect on its path.

- **`rest_plus_eps` is unmoved for a different reason than `rest`.** At
  `E_π = m_π` the kernel takes the near-rest branch, which drops both
  prompt lines. One epsilon above it the branch is the boosted one and
  the line is computed — but `β` is ~1e-8, so the boost window
  `[γE(1−β), γE(1+β)]` is narrower than the grid's spacing and no
  sampled energy straddles `ENU_E_PI_RF`. The block is therefore evidence
  that the declaration is minimal, not evidence that the branch is
  guarded.

- **The local magnitude the follow-up recorded is three orders of
  magnitude too small.** It measured 0.06% at `E_π = 200` MeV, from three
  points on the low-energy plateau. Above the muon-decay continuum's
  support the doubled line is the *entire* spectrum, so the repair halves
  it exactly: the relative drop is `0.500000` at 6 of 174 grid points at
  `E_π = 200` MeV and at 241 of 824 at `E_π = 5000` MeV. Integrated the
  follow-up's figure stands — one `BR_e` per pion, 0.0123% of the row.

- **The measurement has to isolate the repair from the platform drift.**
  Comparing the live tree against the stored corpus reports 556 moved
  values across four blocks, including the muon row; almost all of that
  is the ulp-level drift the case's own budget already absorbs
  (`numpy 2.5.1 → 2.5.3`, `scipy 1.18.0 → 1.18.1`, a macOS point
  release). Capturing the blocks twice — from a build carrying the defect
  and from the repaired build — gives 215 and only the electron row.

- **Those 50% positions expose a second, pre-existing defect.** Where the
  drop is exactly one half, the muon-decay continuum's quadrature
  returned exactly `0.0`: the integrand's support ends at
  `ENU_E_PI_RF = 69.784260` MeV while the boost window is up to 423x
  wider, so every QUADPACK abscissa falls outside it. That is the same
  lost-support failure as A3 and it is untouched by this repair, in the
  charged pion's *neutrino* kernel rather than its photon one. Filed as
  [`neutrino-pion-continuum-loses-its-quadrature-support.md`](../../../docs/followups/todo/neutrino-pion-continuum-loses-its-quadrature-support.md).

## Decisions and Implementation Notes

- **B5 is a new roster entry, not a new project.** The follow-up was
  filed 2026-08-20, the day after this plan was drawn, and its "Triggers
  / blockers" bullet already named this plan as its home — but no row was
  ever added, so `PLAN.md` said "eight defects" against a population of
  nine. Repairing it anywhere else would have needed the delta layer that
  lives here anyway.

- **Numbered `10a`, between Tasks 10 and 11.** `scripts/agents/resolve_task.py`
  sorts on `(\d+(?:\.\d+)*)\s*([a-z]*)`, so a letter suffix orders after
  its number without renumbering eleven existing tasks and their inbound
  citations. The position is forced from both sides: Task 12 has to close
  last, and Task 11's prose sweep moves this follow-up to `done/`.

- **The declared term is a closed form written from the physics, not a
  transcription of `boost_delta_function`.** Rule 3 wants an oracle the
  repair does not share; the boosted line is a rectangle of height
  `BR_e / (2 γ β E_ν^rf)` over `E ∈ (E_rf/(γ(1+β)), E_rf/(γ(1−β)))`,
  which is four lines of numpy and needs no probe. It disagrees with the
  kernel's own `boost_delta_function` by an FMA in the `γ` fold, worth
  under 1.5e-15 of the compared value.

- **The relation's `rtol = 3e-12` is derived, not fitted.** The
  comparison denominator `stored + term` is at most 2x smaller than
  `stored` — the doubled line cannot exceed the value it is part of — so
  the case's own `PORTED_QUAD_RTOL = 1e-12` is amplified by at most two,
  plus the closed form's ulp-level disagreement. Measured worst over the
  215 positions: 1.494e-15.

- **`the_boost_conserves_neutrino_number_per_flavor` cannot see this
  repair, and now says so.** Its 3e-3 budget is 24x `BR_e` itself, so it
  pins the continuum's normalization rather than the line count. The
  discriminating Rust assertion is the new window-edge step check in
  `the_electron_line_comes_from_one_half_only`; the discriminating Python
  one is the continuum-subtracted plateau ratio.

## Files Changed

- `rust/src/kernels/neutrino_pion.rs` — the repair, a module-doc section
  for the deliberate divergence from the `.pyx`, three re-pointed doc
  comments, and two renamed/re-pointed tests.
- `test/parity/deltas.py` — the `B5` declaration, its closed-form term,
  and `B5` added to `REPAIRS`.
- `test/parity/test_parity.py` — `EXPECTED_DECLARED_ARRAYS` 30 → 36.
- `test/test_core_neutrino.py` — the module docstring's declared-defect
  section, `reference_dnde_neutrino_charged_pion` (the independent
  oracle), and two renamed/re-pointed `TestPhysics` tests.
- `CHANGELOG.md` — a new `[Unreleased]` section with the magnitude.
- `docs/followups/todo/neutrino-pion-electron-line-counted-twice.md` —
  status, and the corrected local magnitude.
- `docs/followups/todo/neutrino-pion-continuum-loses-its-quadrature-support.md`,
  `docs/followups/README.md` — the second defect this task surfaced.
- `../PLAN.md`, `../references/defect-blast-radius.md`, `../task-notes/README.md`.

## Verification

Built with

```sh
uv pip install --python .venv/bin/python -e . \
    --config-setting build-args="--features test-probes"
```

confirmed importing from the worktree
(`hazma/_core.abi3.so` under `.claude/worktrees/focused-lovelace-7b108a`).

```text
cargo test --manifest-path rust/Cargo.toml --no-default-features \
    --features test-probes neutrino_pion
  → test result: ok. 15 passed; 0 failed; 246 filtered out

pytest test/parity -q                → 668 passed, 1 skipped
                                       (baseline before the repair: the same)
pytest test/test_core_neutrino.py -q → 58 passed

scripts/agents/preflight.sh --paths "test/parity/deltas.py
    test/parity/test_parity.py test/test_core_neutrino.py" --md "<8 docs>"
  → RESULT: PASS, every row green;
    pytest 2246 passed, 15 skipped, 12 subtests passed
```

What those cover: the kernel's own branch structure and boost arithmetic
(15 Rust tests, of which `the_electron_line_comes_from_one_half_only` is
the repair's); the whole 41-case corpus including the six newly declared
arrays and the two blocks of the same case that must *not* move; and the
Python layer's independent recomputation, plateau ratios, per-flavor
yields, at-rest branch, broadcasting and finiteness.

**Mutation pair (rule 6, proof obligation 2).** Restoring the `delta_e`
term and re-running:

```text
cargo test ... neutrino_pion
  → the_electron_line_comes_from_one_half_only ... FAILED
    "the mu nu_mu half steps by 0.00000032812679632122553 across the
     electron line's window edge, so it is carrying a copy of the line
     (0.00000032812679632122553)"
  → 14 passed; 1 failed
```

The step is exactly one line's height, which is the assertion's own
statement. `the_boost_conserves_neutrino_number_per_flavor` passes with
the defect restored, which is why its doc comment now says what its
budget can and cannot see.

Deferred: `scripts/agents/check_doc_citations.py` over the touched docs
is Task 11's gate, not this one's; the preflight run below is this task's.

## Numerical impact

**One public function moves: `hazma.spectra.dnde_neutrino_charged_pion`,
electron row only, downward only.** Measured by evaluating every public
`hazma.spectra.dnde_neutrino_*` and the n-body `dnde_neutrino` on
`np.geomspace(1e-4, 1e5, 2001)` from a build carrying the defect and from
the repaired build, and diffing:

| Function | Result |
| --- | --- |
| `dnde_neutrino_charged_pion` | moves at `E_π` = 200, 400, 1000, 5000 MeV; unchanged at `E_π = m_π` |
| `dnde_neutrino_muon` | identical at 105.658, 200, 1000 MeV |
| the nine tabulated `dnde_neutrino_*` | identical at `m` and `2 m` |
| `dnde_neutrino(..., ["pi", "pi"])` | moves; `["mu", "mu"]` identical |

| `E_π` (MeV) | positions moved | band (MeV) | median drop | max drop |
| --- | --- | --- | --- | --- |
| 200 | 174 / 2001 | 28.44 – 170.8 | 0.0108% | 50.0000% |
| 400 | 331 / 2001 | 12.68 – 387.3 | 0.0055% | 50.0000% |
| 1000 | 513 / 2001 | 4.937 – 994.3 | 0.0048% | 50.0000% |
| 5000 | 824 / 2001 | 0.9806 – 4955 | 0.0047% | 50.0000% |

Integrated, the electron-neutrino yield falls by
`BR(π → e ν_e) = 1.230e-4` per pion: 1.000141 → 1.000018 at
`E_π = 200` MeV and 0.988680 → 0.988556 at 1000 MeV by trapezoid on that
grid, i.e. −0.0123% both times. The muon row's integral is unchanged to
the last bit. On the corpus grids the same repair moves 215 of 4,305
pinned values, all in the electron row, all down, at a relative drop
between 4.716e-5 and exactly 0.500000.

## Open Questions

- **Does the muon-decay continuum's lost quadrature support reach other
  kernels?** `positron_pion.rs` boosts the same muon spectrum over the
  same kind of window. Filed as
  [`neutrino-pion-continuum-loses-its-quadrature-support.md`](../../../docs/followups/todo/neutrino-pion-continuum-loses-its-quadrature-support.md),
  which names that as its own first question rather than assuming it.
- **Should `[Unreleased]` stay, or fold into Task 12's version heading?**
  This task added one because 2.2.0 is released and a moved published
  number cannot wait for the close. Task 12 renames it; `preflight.sh
  --closing` greps for `## [<new version>]` and would fail on
  `[Unreleased]`.

## Plan Impact

**Impact Level:** Plan and reference patched.

`PLAN.md` gains Task 10a's canonical shape and counts nine defects rather
than eight (frontmatter `deliverable`, Goal, Scope, Numerical impact, The
defects, and Task 12's follow-up count).
`references/defect-blast-radius.md` gains the B5 radius section and
roster row, and its coverage arithmetic goes 25 slots / union 20 /
untouched 21 → 26 / 21 / 20. No ADR: the delta layer, the relation
vocabulary and the proof obligations all applied unchanged, which is
ADR-0001 working rather than a new decision.

## Stale-state sweep

Run against this branch after the last prose edit.

### Identifier sweep

```sh
rg -n --hidden '<identifier>' projects/ docs/ README.md hazma/ test/ \
    rust/src/ .claude/ .codex/
```

Listed by file rather than by hit count, since this note is itself one of
the files the command matches (`[measurement-taken-before-the-task-ended]`).

| Identifier | Files | Disposition |
| --- | --- | --- |
| `the_electron_line_is_counted_by_both_halves` (removed) | `PLAN.md`, this note, the B5 follow-up | KEPT — no source file carries it; each of the three names it as the pre-repair name beside the new one |
| `test_the_electron_line_is_counted_twice_and_the_muon_line_once` (removed) | `PLAN.md`, this note, the B5 follow-up | KEPT — same three, same reason |
| `the_electron_line_comes_from_one_half_only` (new) | `neutrino_pion.rs`, `PLAN.md`, this note, the B5 follow-up | EDITED — the live name reaches every doc that names the old one |
| `test_each_prompt_line_is_counted_exactly_once` (new) | `test_core_neutrino.py`, `PLAN.md`, this note, the B5 follow-up | EDITED — same |
| `_pion_electron_line`, `_B5` (new) | `deltas.py`, this note | KEPT |
| `EXPECTED_DECLARED_ARRAYS` | `test_parity.py`, `task-notes/README.md`, this note, `task-1-delta-declarations.md` | KEPT — Task 1's note records `= 30` as what Task 1 landed and is dated history |

`delta_e` returns 0 hits in `rust/src/kernels/neutrino_pion.rs`.

### Line-number citation sweep

```sh
rg -n -e 'neutrino_pion\.rs:[0-9]+' -e 'deltas\.py:[0-9]+' \
   -e 'test_parity\.py:[0-9]+' -e 'test_core_neutrino\.py:[0-9]+' \
   -e 'positron_pion\.rs:[0-9]+' projects/ docs/
```

The only citations into files this task touched are ten
`test_parity.py:<line>` references in seven `projects/cython-to-rust/`
task notes. This task's edit to that file replaces one line in place
(`git diff origin/master --stat` shows `2 +-`), so no line number moved.
The two new `positron_pion.rs:163-164` citations are into a file this
task does not touch, and were read back to confirm they are the two
clipping lines.

`scripts/agents/check_doc_citations.py` over the eight touched docs:
`docs scanned: 8`, `in-repo citations checked: 0`,
`external citations skipped: 2` (`hazma/spectra/_neutrino/_pion.pyx`,
deleted in `cython-to-rust` Task 4.6), `out-of-range or ambiguous: NONE`.
It bounds-checks `file:line` but not `file:line-line`, which is why the
`positron_pion.rs` range was checked by hand.

All 119 relative markdown links in the eight touched docs resolve, and
the repo-wide dangling-follow-up sweep returns the same three
pre-existing slugs it returned before this task
(`cross-section-prefactor-threshold-cancellation`,
`legacy-parameters-width-exponent-bug`,
`oracle-restore-revisions-for-the-mediator-decay-pyx`), none of them
this task's.

### Forward-looking phrase sweep

`rg -n '(Task [0-9]+ will|will be added|still pending|today: ?stub)'` over
`projects/parity-pinned-defect-repair/` and the two follow-ups: no
occurrences.

### Count sweep

| Claim location | Command | Actual | Status |
| --- | --- | --- | --- |
| "215 of 4,305", this note, `deltas.py`, `defect-blast-radius.md`, README | diff of the two corpus captures | 215 moved, 4305 pinned | OK |
| "35 in near_rest, 69 in boosted_mild, 111 in boosted_strong" | same | `{'near_rest': 35, 'boosted_mild': 69, 'boosted_strong': 111}` | OK |
| "36 declared arrays", `EXPECTED_DECLARED_ARRAYS` | `len(deltas.DECLARED_DELTAS)` | 36 (30 B4 + 6 B5) | OK |
| "41 cases, 18 `cross_sections.*`", `defect-blast-radius.md` | the manifest command that file quotes | 41, 18 | OK |
| "union 21, untouched 20", `defect-blast-radius.md` | `7+1+6+6+1 = 21`; `18 + 2 = 20`; `21 + 20 = 41` | 41 | OK |
| "worst 1.494e-15", `deltas.py`'s `why` | `max(abs(live - (stored+term))/abs(stored+term))` over the declared positions | 1.494e-15 | OK |
| "14 continuum-zero positions", the new follow-up | count of declared positions where `after == 0.5 * before` | 14 | OK |
| "69.783500 MeV", the new follow-up | 200,001-point sweep of `dnde_neutrino_muon(., ENG_MU_PI_RF)` | 69.783500, both rows | OK |
| "15 Rust tests", this note | `cargo test ... neutrino_pion` | 15 passed | OK |
| "668 passed, 1 skipped", this note | `pytest test/parity -q` | 668 passed, 1 skipped | OK |
| "58 passed", this note | `pytest test/test_core_neutrino.py -q` | 58 passed | OK |
| "no tests lost" | `grep -c "def test_"` vs `git show origin/master:<file>` | 32/32, 26/26; `#[test]` 15/15 | OK |

### Numerical-impact statement

Every public `hazma.spectra.dnde_neutrino_*` and the n-body
`dnde_neutrino`, on `np.geomspace(1e-4, 1e5, 2001)` (401 points for the
n-body), evaluated from a build carrying the defect and from the repaired
build. **One function moves**: `dnde_neutrino_charged_pion`, electron row
only, downward only, at four of the five parent energies (unchanged at
`E_π = m_π`), median drop 4.7e-5 to 1.1e-4 and maximum exactly
0.500000000000. `dnde_neutrino` moves for `["pi", "pi"]` and not for
`["mu", "mu"]`. The other eleven functions are bit-identical. Full table
under `## Numerical impact` above.

### Exit Criteria → test mapping

| Exit criterion | What satisfies it |
| --- | --- |
| `delta_e` gone; `dnde_e_nue` unchanged | `git diff origin/master -- rust/src/kernels/neutrino_pion.rs`; `the_electron_line_comes_from_one_half_only` |
| Declared delta on one case, exactly the positions the line reaches | `test_entry_point_matches_corpus[spectra.neutrino.charged_pion[*]]`, 5 blocks; the 36 other cases compared unchanged |
| `rest` / `rest_plus_eps` still match stored | the same two block tests, undeclared and green |
| `git diff --stat -- test/parity/data` empty | ran; empty |
| No tolerance widened | `git diff --stat -- test/parity/tolerances.py` empty |
| Independent oracle (rule 3) | `deltas._pion_electron_line` and `test_core_neutrino.reference_dnde_neutrino_charged_pion`, both closed forms |
| Physics invariant (rule 4) | `test_each_prompt_line_is_counted_exactly_once` (1e-6, continuum subtracted with scipy); `test_the_pion_yields_one_muon_neutrino_from_each_of_two_sources` |
| Tests renamed, not only re-pointed | the identifier sweep above |
| Magnitude in `CHANGELOG.md` and "Numerical impact so far" | both written |

### Task-note self-consistency

`**Status:** Complete` matches the Tasks-table cell in
`task-notes/README.md` (`10a … **Complete**`). Every file named in
§Files Changed appears in `git diff origin/master --stat` (12 files, two
of them new). Every function and constant cited in §Findings and
§Decisions — `dnde_mu_numu`, `dnde_e_nue`, `boost_delta_function`,
`boost_integral`, `_pion_electron_line`, `PORTED_QUAD_RTOL`,
`EXPECTED_DECLARED_ARRAYS`, `positron_pion.rs`'s clip — exists in the
tree at the spelling used.

## Handoff to Next Task

- **B5 is closed and disjoint.** Nothing in Tasks 3-10 shares an array
  with it, so it constrains no ordering. Task 11 has one more follow-up
  to sweep to `done/` than its Scope paragraph originally counted, and
  Task 12 one more magnitude to aggregate — both already patched.
- **Two blocks of `spectra.neutrino.charged_pion` are deliberately
  undeclared.** `rest` and `rest_plus_eps` must keep matching their
  stored arrays; a future repair that moves them has not found a wider
  B5, it has found something else.
- **The isolate-the-repair recipe is reusable** and Tasks 4, 7, 8 and 10
  will need it: capture the corpus blocks from a build with the defect,
  restore, rebuild, capture again, diff. Comparing against the stored
  corpus instead reports the platform drift as if it were the repair — it
  showed 556 moved values here where the repair moves 215.
