# Task 3: Closed-form delta models for the three twin-less defects

**Date:** 2026-09-06
**Project:** parity-pinned-defect-repair
**Status:** Complete
**Plan References:** `../PLAN.md` Task 3 (and Tasks 5, 6, 9, which consume
this); `../rules.md` rules 1, 2, 3, 4, 5, 6, 11;
`../references/corpus-repinning.md` ("Declaration schema", "Relations")
**Related ADRs:** `../adrs/ADR-0001-corpus-repairs-are-declared-deltas.md`
**Depends On:** Task 1 (the delta-declaration layer)

## Objective

Give B1, B2 and B3 a delta model that is stated in closed form, checked
against the committed corpus rather than against either the kernel that
will be repaired or a Cython twin, and expressed in the declaration
schema so that Tasks 5, 6 and 9 declare rather than derive.

## Exit Criteria

- A model in `test/parity/deltas.py` for each of B1, B2, B3, carrying its
  relation, its budget and the measurement that justifies the budget.
- A test that each model reproduces the **shipped** corpus value from the
  corrected form plus the named defect — falsifiable on this tree, with
  neither the repaired Rust nor a Cython twin.
- The `mpmath` reference for the one closed form that is analytic, in the
  shape of `test/parity/reference.py`.
- `pytest test/parity` green with the collected count accounted for, and
  `git diff --stat -- test/parity/data` empty (`../rules.md` rule 1).

## Inputs Reviewed

- `../PLAN.md` — Tasks 3, 5, 6, 9; "Numerical impact"; "The defects".
- `task-notes/README.md` — Tasks table, Findings, Numerical impact so far.
- `../rules.md` — all eleven.
- `../references/corpus-repinning.md` — declaration schema and relations.
- `../references/defect-blast-radius.md` — the B1, B2, B3 rows.
- `docs/followups/todo/eta-prime-two-photon-line-missing-factor-two.md`,
  `phi-photon-lines-use-the-daughter-meson-energy.md`,
  `rho-rest-frame-branch-returns-the-integrand.md`.
- `rust/src/kernels/photon_tables.rs` (`dnde`, `branch`, the line weights
  and energies), `rust/src/kernels/photon_rho.rs` (`boosted`),
  `rust/src/boost.rs` (`boost_beta`, `boost_delta_function`),
  `rust/src/constants.rs` (`pdg`).
- `test/parity/{deltas,test_parity,cases,tolerances,reference,stability}.py`,
  `test/test_core_photon_tables.py`, `test/parity/README.md`.
- `docs/agents/lessons.md`, `docs/agents/environment.md`.

## Findings

- **A declaration cannot land before its repair.** ADR-0001's staleness
  rule — a declared array that still equals the corpus fails — is
  enforced in `test_parity._assert_declared_delta`, so keying B1, B2 or
  B3 into `DECLARED_DELTAS` on this tree turns the gate red. The models
  therefore ship in a new `deltas.DELTA_MODELS`, and Tasks 5, 6 and 9 add
  the keys. `PLAN.md`'s Task 3 gate said "a declaration in
  `test/parity/deltas.py`" and has been patched to say which half lands
  when.
- **B1 does not reach the `rest` block.** `photon_tables::dnde` adds line
  terms only on `Branch::InFlight`; the rest-frame arm returns the bare
  table. So the eta-prime delta is 189 positions over six arrays, none of
  them in `rest`.
- **B2 does not reach either rest block.** At `beta = 1.414e-06` both
  windows are 2.8e-06 wide in relative energy and the grid steps over
  them: 0 positions in `rest` and `rest_plus_eps`, 305 over six arrays
  elsewhere. The corpus anchors the `M/2` line images (`_half_mass`) but
  not the phi's, which is why B1's `rest_plus_eps` is sampled and B2's is
  not.
- **The phi's two shipped lines sit above the spectrum's own endpoint**,
  where the boosted continuum is *exactly* zero. That is what makes B2
  readable off the corpus at all: the top of the stored array is a
  two-tread staircase and nothing else.
- **`_parent_beta`'s `E - M < DBL_EPSILON` arm is unreachable at these
  masses**, which is what `photon_tables::branch` says of its own copy:
  `DBL_EPSILON` is an absolute 2.2e-16 MeV and one ulp at 957.78 MeV is
  1.1e-13, so the interval holds no double but `M` itself. Kept, to
  mirror the kernel; `test_a_parent_at_rest_carries_no_boosted_line`
  asserts the unreachability rather than pretending to exercise it.
- **None of the three forms earns an arbitrary-precision reference.**
  Against 60-digit `mpmath`: the two-body energies are correct to 15.6
  to 16.2 decimal digits and a boosted line's `height × width` is `1.0`
  to within one ulp, against the ~33 digits `reference.py`'s four
  kernels lose to `atan` cancellation. Keeping `mpmath` out of the test
  path also keeps true what `pyproject.toml`, `stability.py` and
  `reference.py` all three assert about it.
- **A model test that recomputes the model tests nothing.** The first
  draft asserted against constants and expressions spelled out in the
  test module; three of ten mutations to `deltas.py` — including
  changing B1's weight to `2·BR` and B3's transform to `stored / E` —
  left it green. Every model test now drives `DELTA_MODELS[...]`.

## Decisions and Implementation Notes

- **A relation now returns the repaired array, not an additive term.**
  `Additive.expected` and `Exact.expected` both answer "what should this
  array be", and the runner derives the moved mask from
  `predicted != stored` instead of `term != 0`. Two reasons. B3's
  relation is a transform of the stored array, and forcing it through
  `stored + term` would cost two roundings where the repaired kernel
  spends one — the prediction stops being bit-exact for no gain. And
  `predicted != stored` is the more direct spelling of what `MOVED`
  means: a term too small to move a float64 no longer counts as having
  moved it, which is also what the repaired kernel will do.
- **`Exact` is the new relation, and the spec already named it.**
  `corpus-repinning.md`'s relation table lists `Exact(f)` with the rho
  `rest` case as its example; this is that class, with the stored arrays
  (abscissae included) passed in so a transform can be a function of the
  grid.
- **The models read `hazma._core.boost`, not a re-rounded copy.** The
  window edges decide which positions `MOVED` resolves to, and the
  kernel computes them with `mul_add`; a NumPy transcription disagrees at
  the anchor points either side of an edge, which is exactly where the
  corpus samples. `boost_delta_function` is not what B1 or B2 repairs, so
  reading it is not circular — and the corpus, not that function, is what
  the models are checked against.
- **Constants are spelled out and cited, not imported.**
  `hazma.parameters` carries the masses but no branching ratios, and
  `test/test_core_photon_tables.py` already sets the convention of
  transcribing `rust/src/constants.rs`'s `pdg` table so a consolidation
  of the two constant tables cannot move a test with the code.
- **ADR-0001 states the contract, so it moved with it** (review round 1).
  The ADR defined `MOVED` as "resolved against the term at comparison
  time" and called one extra kernel evaluation per declared array an
  unconditional cost. Both are the pre-`Exact` contract: the runner now
  resolves against the predicted array, and a closed-form transform
  evaluates no kernel. Its Decision and Consequences sections are
  rewritten to the relation-general form, and the staleness bullet now
  carries the timing that follows from it — a declaration cannot precede
  its repair.
- **The relation budgets are bounds, not measurements, and say so.**
  Neither B1/B2's 1e-11 nor B3's 1e-9 can be measured until the repair
  lands: each is set to the case's own budget with headroom, so Tasks 5,
  6 and 9 tighten rather than widen (`../rules.md` rule 2).

## Files Changed

- `test/parity/deltas.py` — the `Exact` relation, `expected` on both
  relations, the B1/B2/B3 models and their helpers, `DELTA_MODELS`.
- `test/parity/test_delta_models.py` — new. The non-circular gate: 17
  tests, no kernel evaluated.
- `test/parity/test_parity.py` — the runner moves to the prediction
  protocol; the shape tests sweep `DELTA_MODELS`; two new tests.
- `test/parity/README.md` — the new module, the model/declaration split,
  the re-derived collected count.
- `projects/parity-pinned-defect-repair/PLAN.md` — Task 3's gate, its
  `mpmath` scope note, and the stale `hazma/_utils/boost.pyx` reference.
- `projects/parity-pinned-defect-repair/adrs/ADR-0001-corpus-repairs-are-declared-deltas.md`
  — the relation contract, relation-general (review round 1).
- `docs/agents/lessons.md`, `docs/agents/lessons-examples.md` —
  `[sweep-excluded-the-canonical-directory]` gains `adrs/` as a live site
  and PR #92 as a citation (review round 1).
- `projects/parity-pinned-defect-repair/references/corpus-repinning.md` —
  when a declaration may be written; `Ratio` → `Exact` in the schema
  example.
- This note.

## Measurements

Every figure below is from a run against this branch; commands in
`## Verification`.

**B1 — the eta-prime line's weight, read off the stored arrays.** The
step across the top of each line's boosted window, over the modelled
plateau `w / (2 γ β e0)`:

| Case | Weight the corpus reads | Correct weight | Worst deviation |
| --- | --- | --- | --- |
| `spectra.photon.eta` | `2 BR` = 0.7882 | `2 BR` | 0.0 |
| `spectra.photon.long_kaon` | `2 BR` = 1.094e-03 | `2 BR` | 2.3e-14 |
| `spectra.photon.short_kaon` | `2 BR` = 5.26e-06 | `2 BR` | 0.0 |
| `spectra.photon.eta_prime` | **`1 BR` = 2.307e-02** | `2 BR` | 4.9e-13 |
| `spectra.photon.phi` (η line) | `BR` at 656.942002472385 MeV | same weight, 362.5189975276151 MeV | 0.0 |
| `spectra.photon.phi` (η′ line) | `BR` at 959.6459594437648 MeV | same weight, 59.815040556235125 MeV | 0.0 |

22 (line, block) pairs; worst over all of them **4.9e-13**. The residue
is the continuum's own change across the straddling step, which the
anchor cluster holds to 1e-09 relative for the four `M/2` lines and to
zero for the phi's two.

**B2 — the staircase.** Above the phi's true photon endpoint the stored
spectrum is `h_η + h_η′`, then `h_η′`, then exactly `0.0`. At
`boosted_mild` the three read 5.744387e-06, 1.871064e-08 and 0.0 (63
positions, all exactly zero), against modelled treads of 5.744387e-06
and 1.871064e-08 — a match to the last bit. The model's repaired array
is identically zero across all of it.

**B3 — the rho rest block.** `stored × E_γ` against the same case's
`rest_plus_eps` block, which runs the correct branch at
`beta = 1.414e-06`:

| Case | Paired positions | Both non-zero | Worst relative | Median |
| --- | --- | --- | --- | --- |
| `spectra.photon.charged_rho` | 255 of 255 | 170 | 6.8e-11 | 2.9e-11 |
| `spectra.photon.neutral_rho` | 255 of 255 | 170 | 1.8e-09 | 3.2e-11 |

Zero patterns match exactly. The unrepaired array agrees at **no**
position: it differs by a factor of `E_γ`, and the nearest grid point to
`E_γ = 1` MeV is 0.9959 MeV.

**Reach of each model**, re-derived after the last edit:

| Model | Cases | Arrays moved | Positions moved | Direction |
| --- | --- | --- | --- | --- |
| B1 | `spectra.photon.eta_prime` | 6 | 189 | all up — a second copy of a positive line |
| B2 | `spectra.photon.phi` | 6 | 305 | 233 up, 72 down — the yield is relocated, not changed |
| B3 | both rho `rest` blocks | 4 | 350 | 198 up, 152 down — the factor is `E_γ`, which crosses 1 MeV |

**Arbitrary precision**, one-off, `mpmath` at 60 dps: `(M² ∓ m²)/(2M)`
is correct to 15.6–16.2 digits for both phi lines; `E_γ + E_Y − M` is
`0.0` exactly at 60 dps and at most 1.1e-13 in float64;
`E_γ² − (E_Y² − m²)` is `4.5e-56` at 60 dps; a boosted line's
`height × width` is `1.0` exactly at 60 dps and to one ulp in float64.

## Verification

Environment: `uv venv --python 3.12 .venv`, `--group dev`,
`numpy==2.5.1`, then
`uv pip install -e . --config-setting build-args="--features test-probes"`.
`hazma` and `hazma._core` both resolve inside this worktree. Provenance
is `exact=False` (scipy 1.18.1 against the manifest's 1.18.0, plus 41
served Rust kernels), so the declared per-case budgets are in force —
the expected mid-port state.

```sh
pytest                     # 2265 passed, 15 skipped, 12 subtests passed
pytest test/parity -q      # 687 passed, 1 skipped
pytest test/parity --collect-only -q   # 688 tests collected
git diff --stat -- test/parity/data    # (empty)
```

Baseline at `origin/master`, measured on this same environment by
restoring the four files and holding the new one aside:
`2246 passed, 15 skipped, 12 subtests passed`, `test/parity` collecting
`669`. So this task adds **19 tests and no skips** — 17 in
`test_delta_models.py`, 2 in `test_parity.py`.

What the 17 cover: six parametrized line-weight readings against the
corpus (three correct siblings as controls, three defects); the sweep's
own coverage count; B1's weight and its factor-of-two falsification;
B2's staircase, its model's zero-above-the-endpoint statement and its
falsification at the corrected energies; B3's two cases, each with the
ratio identity, the model's guard, and the no-coincidence falsification;
the rest-frame arm; the line normalization; and two-body kinematics. The
two in `test_parity.py` are `Exact` through `_assert_declared_delta` and
the rule that a declaration must point at a `DELTA_MODELS` entry.

**Mutation harness** (`../rules.md` rule 6): ten single-edit mutations of
`deltas.py`, each applied to a `cmp`-verified clean baseline and reverted
and re-verified after
(`[mutation-harness-poisons-its-own-baseline]`). All ten turn
`test_delta_models.py` red:

| Mutation | Result |
| --- | --- |
| B1 weight `1·BR` → `2·BR` | 1 failed |
| B1 weight `1·BR` → `0.5·BR` | 1 failed |
| B1 line energy `M/2` → `M/3` | 2 failed |
| B1 term ignores the block's boost | 3 failed |
| B2 shipped energy uses the photon form | 2 failed |
| B2 relocates only the eta line | 2 failed |
| B2 term drops the subtraction | 2 failed |
| B3 transform `× E` → `/ E` | 2 failed |
| B3 transform `× E` → `× E²` | 2 failed |
| B3 rest-frame guard dropped | 3 failed |

An eleventh, dropping `_parent_beta`'s `DBL_EPSILON` guard, stays green
and provably must: at these masses no representable double lies in the
interval it covers, so no input distinguishes the two spellings. That is
the finding above, and the test asserts it rather than the guard.

Lint on the touched Python: `black --check`, `isort --check-only` and
`ruff check` over `test/parity/` are all clean (12 files unchanged, all
checks passed).

## Numerical impact

**No public value changes.** The diff contains no library or build file:
`git diff origin/master --name-only -- hazma rust pyproject.toml | wc -l`
returns `0`, and the full suite passes with the same 15 skips as the
baseline. Nothing is added to the project's "Numerical impact so far" —
the three repairs this task models are Tasks 5, 6 and 9, and each will
record its own figures there (`../rules.md` rule 10).

## Open Questions

- **Will the relation budgets survive their repairs?** B1 and B2 carry
  1e-11 and B3 1e-9, each derived from the case's own budget rather than
  measured against a repaired kernel, which cannot exist yet. Tasks 5, 6
  and 9 measure the real figure; `../rules.md` rule 2 says the answer is
  to tighten, and a repair that needs a *wider* budget has found
  something this task did not model.
- **Does A1 (Task 4) disturb B1's or B2's declared positions?** Both
  land on arrays the boost-window repair also moves, and rule 7 forbids
  overlapping declarations. The line terms and the boost integral are
  disjoint mechanisms — one is `boost_delta_function`, the other
  `boost_integrate_linear_interp` — but the position sets have not been
  intersected. Tasks 5 and 6 inherit that as a gate, not an assumption.

## Plan Impact

**Impact Level:** Update `PLAN.md` and `references/corpus-repinning.md`.

Three canonical statements were wrong and are patched in this task rather
than deferred. `PLAN.md` Task 3's **gate** asked for a declaration in
`test/parity/deltas.py`, which ADR-0001's own staleness rule forbids
before the repair lands; it now names the model/declaration split and
where each half goes. Its **B1 scope bullet** described
`hazma/_utils/boost.pyx` as "still live", which `cython-to-rust` Task 6.4
has since deleted. Its **`mpmath` scope note** asked for a reference the
measurement shows these forms do not need, and now records the digit
count instead of the instruction. `corpus-repinning.md` gains the timing
rule under "How the runner uses it" and has `Ratio(...)` corrected to
`Exact(...)` in its schema example — the class this task implemented for
the case that example names.

No ADR. The relation protocol change is an implementation of ADR-0001's
decision, not a change to it: the corpus is still never rewritten, a
declaration still names positions and a relation, and a stale
declaration still fails.

## Stale-state sweep

Run against this branch after the last prose edit.

### Identifier sweep

`rg -n --hidden '<id>' projects/ docs/ README.md hazma/ test/ .claude/
.codex/`, hit counts:

```text
DELTA_MODELS 27   Exact 39   TransformFn 3   _parent_beta 9
_boosted_line 11  _on_value_grids 3   _photon_energy 10  _daughter_energy 8
_rho_rest_frame_spectrum 2   _eta_prime_line_second_copy 2
_phi_lines_relocated 2   PHI_LINE_DAUGHTERS 2   test_delta_models 14
EXPECTED_REACH 4   EXPECTED_PLATEAU_READINGS 2
```

Every hit outside the six changed files is a mention of the ordinary
English word (`Exact` matches "exact" in `reference.py`, the cython-to-rust
notes and `docs/source/spectra.rst`), not the class. The three sweeps that
could have found stale prose:

| Pattern | Hits | Disposition |
| --- | --- | --- |
| `relation\.term\(` outside `deltas.py` | 3 | KEPT. Two are `test_delta_models.py` driving `Additive.term`, which is still a field. The third is `task-1-delta-declarations.md:118`, a dated measurement script in a closed note — it still runs and still prints 3,065. |
| `Ratio\(` under `projects/`, `docs/` | 1 | EDITED. The schema example in `references/corpus-repinning.md` now reads `Exact(...)`; the surviving hit is this note describing that edit. |
| `still live` under this project | 2 | EDITED. `PLAN.md`'s Task 3 B1 bullet, plus this note describing the edit. `hazma/_utils/` is gone from the tree. |
| `term=term,` in `test/` | 0 | The old `_assert_declared_delta` keyword has no callers left. |

### Line-number citation sweep

```sh
scripts/agents/check_doc_citations.py <the five changed .md files>
#   docs scanned: 5
#   in-repo citations checked: 0
#   external citations skipped: 0
#   out-of-range or ambiguous: NONE
```

Zero because this task's docs cite symbols and file paths, not
`file:line`. The one code-to-test citation added — `deltas._B1`'s `why`
naming `test_delta_models.py::test_a_tabulated_line_carries_the_weight_the_corpus_stores`
— was written stale as
`test_every_tabulated_line_reproduces_its_corpus_plateau`, caught in
self-review, EDITED, and verified by running the named node id (6 passed).

### Forward-looking phrase sweep

`rg -n '(Task [0-9]+ will|will be added|still pending|today: ?stub)'` over
the four changed Python and Markdown files under `test/parity/`: **0
hits**. Forward references to Tasks 5, 6 and 9 exist only in project
documents, where they are the plan.

### Count sweep

| Claim location | Command | Actual | Status |
| --- | --- | --- | --- |
| Note, Verification | `pytest -q` | `2265 passed, 15 skipped, 12 subtests passed` | OK |
| Note, Verification | `pytest test/parity -q` | `687 passed, 1 skipped` | OK |
| Note, Verification | `pytest test/parity --collect-only -q` | `688` | OK |
| Note, Verification (baseline) | same two, files restored to `origin/master` | `2246 passed, 15 skipped, 12 subtests`; `669` | OK — +19 tests, +0 skips |
| `test/parity/README.md` "plus 27 others … reports 650" | `pytest test/parity/test_parity.py --collect-only -q` | `650` | OK |
| `test/parity/README.md` "623 of them" | manifest block count | `623` | OK |
| Note, "17 tests" | `pytest test/parity/test_delta_models.py --collect-only -q` | `17` | OK |
| `EXPECTED_DECLARED_ARRAYS` | `len(deltas.DECLARED_DELTAS)` | `30` = `30` | OK — no key added |
| `EXPECTED_PLATEAU_READINGS = 22` | the test itself | `22` | OK |
| `EXPECTED_REACH` B1/B2/B3 | the test itself | `(6, 189)`, `(6, 305)`, `(4, 350)` | OK |
| Note, B2 sign split | model, re-derived | 233 up, 72 down | EDITED — first written 207/98 |
| Note, B1 per-line worst | model, re-derived | 0.0 / 2.3e-14 / 0.0 / 4.9e-13 / 0.0 / 0.0 | EDITED — first written from a hand-computed height |
| Note, B3 worst | model, re-derived | 6.8e-11 charged, 1.8e-09 neutral | OK |

### Numerical-impact statement

**No public value changes.** No grid was evaluated because no public code
path is reachable from this diff:

```sh
git diff origin/master --name-only -- hazma rust pyproject.toml | wc -l   # 0
git diff --stat -- test/parity/data | wc -l                              # 0
```

The full suite passes with the same 15 skips as the `origin/master`
baseline measured on this environment.

### Exit Criteria to test mapping

| Exit criterion | What satisfies it |
| --- | --- |
| A model per defect, with relation, budget, measurement | `deltas.DELTA_MODELS["B1"/"B2"/"B3"]`; shape-gated by `test_parity.py::test_every_delta_model_is_a_roster_repair_with_its_evidence` |
| A test reproducing the shipped value from the corrected form plus the defect | `test_delta_models.py`, 17 tests; falsification arms inside the three per-defect tests; ten mutations of `deltas.py` all turn it red |
| The `mpmath` reference where the form is analytic | Measured unnecessary (15.6–16.2 digits, `height × width` = 1.0 to one ulp) and `PLAN.md`'s scope note patched to record the measurement instead of the instruction |
| `pytest test/parity` green, collected count accounted for | `687 passed, 1 skipped`; 669 → 688 collected, +19 accounted for above |
| `git diff --stat -- test/parity/data` empty | Empty (row above) |

### Review round 1 — the contract sweep

The round's one blocking finding was `ADR-0001` still defining `MOVED`
against an additive term. Swept as a class rather than fixed at the line
(`doc-consistency.md` §11):

```sh
rg -n --hidden 'against the term|term is non-zero|non-zero positions of the term|term does not move|the term moves|extra kernel evaluation|relation whose term' \
    projects/ docs/ hazma/ test/ .claude/ .codex/ README.md CHANGELOG.md
```

#### Pre-fix occurrences

18 hits in 10 files. Triaged by role, per §11's instruction-versus-record
rule:

| Site | Role | Disposition |
| --- | --- | --- |
| `adrs/ADR-0001…md:43-45` | Durable decision, in force | EDITED — the finding |
| `adrs/ADR-0001…md:58-62` | Same, one hunk further down | EDITED — the reviewer did not cite it; "one extra kernel evaluation per declared array" is false of an `Exact` transform, which evaluates none |
| `references/defect-blast-radius.md:174` | Live spec, B4-specific | KEPT — B4's relation is `Additive` and its term is unchanged |
| `task-notes/task-1-delta-declarations.md:62,67,68,122` | Closed task note | KEPT — a record of what Task 1 built, in its own review round's voice |
| `task-notes/README.md:198`, `deltas.py:513`, `docs/followups/done/scalar-decay-fsr-half-normalized.md:83` | B4 measurements | KEPT — still true of `Additive` |
| `test_parity.py:659` | Comment on the synthetic fixture's local `term` | KEPT — that array still exists and still moves three positions |
| `test_delta_models.py:167,298,341-343`, `deltas.py:105` | Live code and `Additive`'s own docstring | KEPT — `Additive.term` is unchanged |

#### Post-fix occurrences

13 hits, none in `adrs/`. Every survivor is one of the KEPT rows above,
plus three new ones in `docs/agents/lessons-examples.md` and two in this
note that **quote** the old wording — rewriting a record of what went
stale would falsify it.

### Task-note self-consistency

`**Status:** Complete` matches the Tasks-table cell in
`task-notes/README.md` and the mapping table above. Every file named in
§Files Changed appears in `git diff origin/master --stat` or as one of the
two created files (`test/parity/test_delta_models.py`, this note); every
symbol named in §Findings and §Decisions — `Exact`, `expected`,
`DELTA_MODELS`, `_parent_beta`, `_boosted_line`, `_photon_energy`,
`_daughter_energy` — is defined in that diff.

## Handoff to Next Task

**Tasks 5 (B1), 6 (B2) and 9 (B3) each inherit a finished model.** The
repair is: fix the kernel, then add the `DECLARED_DELTAS` keys pointing
at `deltas.DELTA_MODELS["Bn"]` — not a new `Delta` — and bump
`EXPECTED_DECLARED_ARRAYS` in `test_parity.py`. The keys, measured on
this branch, are

- **B1** (`spectra.photon.eta_prime`): `rest_plus_eps.values`,
  `near_rest.values`, `boosted_mild.{values,scalar_values}`,
  `boosted_strong.{values,scalar_values}` — six arrays, 189 positions.
  **Not** `rest`, which takes the branch that adds no line.
- **B2** (`spectra.photon.phi`): `near_rest.{values,scalar_values}`,
  `boosted_mild.{values,scalar_values}`,
  `boosted_strong.{values,scalar_values}` — six arrays, 305 positions
  (233 up, 72 down).
  **Neither** rest block; no grid point falls in either window there.
- **B3**: `rest.{values,scalar_values}` of `spectra.photon.charged_rho`
  and of `spectra.photon.neutral_rho` — four arrays, 350 positions.

Re-derive those counts after Task 4 lands rather than pasting them
(`../rules.md` rule 11): B1 and B2 share their arrays with the A1 boost
repair, and `test_delta_models.EXPECTED_REACH` is where the re-derivation
lives.

**Safe to assume:** the models reproduce the committed corpus, and the
mutation table above says what breaks them. `test_delta_models.py`
evaluates no kernel, so it stays green through every repair and keeps
saying the same thing — the corpus pinned the defect, and this is the
shape of it.

**Risky:** the three relation budgets are bounds rather than
measurements (see Open Questions), and B1's and B2's disjointness from
A1's positions is argued, not measured.
