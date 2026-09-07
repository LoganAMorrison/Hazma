# Working Memory: Repair the parity-pinned numerical defects

**Date:** 2026-08-19 (created)
**Project:** parity-pinned-defect-repair
**Status:** In Progress
**Plan References:** `../PLAN.md` (all sections)
**Related ADRs:** `../adrs/ADR-0001-corpus-repairs-are-declared-deltas.md`
(Task 1); one more anticipated, see `../PLAN.md`
**Depends On:** none. **Constrains** `cython-to-rust` Tasks 4.6, 6.2,
6.3 and 6.4 — see Open Questions.

## Objective

Track cumulative context and live task status across all thirteen tasks so
any agent picking up work mid-project has the facts, decisions and open
questions needed to start without re-discovering them.

## Tasks

Canonical task *shape* lives in `../PLAN.md` under "Task Details". This
section tracks live *status*.

| # | Task | Depends on | Status | Task Note |
|---|------|------------|--------|-----------|
| 1 | Delta-declaration layer | — | **Complete** — landed in PR #87 with the B4 repair; ADR-0001 | `task-1-delta-declarations.md` |
| 2 | Capture the corrected-value oracles | — | **Complete** | `task-2-cython-oracles.md` |
| 3 | Closed-form delta models (B1–B3) | 1 | **Complete** — models in `deltas.DELTA_MODELS`, keys land with the repairs | `task-3-closed-form-deltas.md` |
| 4 | Repair A1 — boost integral window | 1, 2 | Not started | `task-4-boost-window.md` |
| 5 | Repair B1 — η′ line weight | 3, 4 | Not started | `task-5-eta-prime-line.md` |
| 6 | Repair B2 — φ line energies | 3, 4 | Not started | `task-6-phi-lines.md` |
| 7 | Repair A2 — muon photon endpoint | 1, 2 | Not started | `task-7-photon-muon-endpoint.md` |
| 8 | Repair A3 — charged-pion forward cone | 7 | Not started | `task-8-charged-pion-cone.md` |
| 9 | Repair B3 — rho rest-frame branch | 3, 8 | Not started | `task-9-rho-rest-frame.md` |
| 10 | Repair A4 — positron-muon normalization | 1, 2 | Not started | `task-10-positron-muon-norm.md` |
| 10a | Repair B5 — charged-pion neutrino line | 1 | **Complete** | `task-10a-neutrino-pion-line.md` |
| 11 | Reconcile the superseded sequencing prose | 4–10a | Not started | `task-11-prose-reconciliation.md` |
| 12 | Close — aggregate the drift, bump | 11 | Not started | `task-12-close.md` |

```text
1 ──┬──► 3 ──┬──────────► 5 ──┐
    │        │                │
    ├──► 4 ──┴────────────────┼──► 11 ──► 12
    │        └──► 6 ──────────┤
    ├──► 10a ─────────────────┤
2 ──┼──► 7 ──► 8 ──► 9 ───────┤
    └──► 10 ──────────────────┘
```

Task 2 had no upstream dependency and the project's only hard external
deadline. It is done: every Group A oracle is captured and committed
under `test/parity/oracles/`, so `cython-to-rust` Tasks 4.6, 6.2, 6.3 and
6.4 may now run in any order without stranding a repair. Nothing else in
this project is time-critical.

## Exit Criteria

- All thirteen tasks complete; all nine defects repaired — the eight
  follow-ups under `docs/followups/todo/` moved to `docs/followups/done/`
  with inbound links repointed and the revision pinned (B4's already
  lives there; B5's is repaired but deliberately still in `todo/`, so the
  repoint sweep happens once).
- No live document still sequences any of the seven original repairs
  "after Phase 06 Task 6.4".
- `git diff --stat -- test/parity/data` empty across the whole project.
- Closing PR bumps `[project] version` in `pyproject.toml` per
  `PLAN.md`'s
  `version_bump: minor` and adds a `CHANGELOG.md` entry naming this
  project slug, carrying the aggregated per-defect shifts. See
  [`../../../docs/versioning.md`](../../../docs/versioning.md).

## Inputs Reviewed

- `../PLAN.md`, `../rules.md`, all three `../references/*.md`.
- The seven follow-ups under `docs/followups/todo/` — the defects, their
  measured magnitudes, and their entry points.
- `projects/cython-to-rust/rules.md` rules 1–3 (parity discipline),
  `projects/cython-to-rust/task-notes/README.md` ("Numerical impact so
  far" and Findings), `projects/cython-to-rust/phases/phase-04-spectra-kernels.md`
  and `phase-06-mediator-spectra.md` (the deletion schedule).
- `test/parity/README.md` — the corpus's own account of what it pins,
  what it compares, and when not to regenerate.
- `docs/agents/lessons.md` — the classes this project is most exposed to
  are listed under Findings.

## Findings

- **The corpus cannot be regenerated wholesale, and has not been able to
  since Phase 04 Task 4.1 (2026-08-11), the first wrapper swap.**
  `test/parity/generate.py` calls
  `cases.assert_no_rust_core()` first, which raises once `hazma._core`
  *serves* any kernel. So "one declared regeneration after Task 6.4",
  which five of the seven follow-ups proposed, was never an available
  move — not merely a mistimed one.
- **The four Group A twins are `cdef`-only.** None of
  `hazma/spectra/_photon/_pion.pyx`, `hazma/spectra/_photon/_muon.pyx`,
  `hazma/spectra/_positron/_muon.pyx` or `hazma/_utils/boost.pyx`
  defines a top-level `def`, so they are reachable from Python solely
  through `__pyx_capi__`. `test/test_core_boost.py` already drives
  `hazma._utils.boost` that way — that harness is the model for Task 2,
  not something to reinvent.
- **The deadline is not one date, it is three.** Task 4.6 (the only
  Phase 04 task left) strands A4's `spectra.positron.charged_pion`
  capture; Tasks 6.2/6.3 strand the mediator-spectra captures; only the
  remainder waits for 6.4. `../references/defect-blast-radius.md` has
  the table.
- **`boost_integrate_linear_interp` reaches only the seven tabulated
  photon spectra.** Its former Cython call sites were
  `_photon/{_eta,_eta_prime,_kaon,_omega,_phi}.pyx`, all deleted in Task
  4.2; `rust/src/kernels/photon_tables.rs` is now the sole consumer. The
  boost-window repair therefore does not touch the muon, pion, rho,
  positron, neutrino or mediator paths, which is narrower than the
  follow-up's "cross-cutting" scope line suggests.
- **Two repair pairs land on the same arrays.** A3 and B3 both move both
  rho cases (the rho quads over the pion, which quads over the muon);
  A1 shares `spectra.photon.eta_prime` with B1 and `spectra.photon.phi`
  with B2. `../rules.md` rule 7 is the constraint that falls out of it.
  This bullet named A2 and A3 as the first pair until Task 2 measured
  A2's radius at a single case that is neither rho — its defect is behind
  an at-rest guard no composed caller reaches. A2 now overlaps nothing.
- **B5 is disjoint from everything else, and reaches one case.** The
  charged pion's neutrino path uses `boost_delta_function` and
  `super::neutrino_muon`; no other roster entry touches either. Task 10a
  measured 215 of the 4,305 pinned values of
  `spectra.neutrino.charged_pion` moving, in three of its five blocks and
  in the electron row only. That retires
  `../references/defect-blast-radius.md`'s claim that neither
  `spectra.neutrino.*` case had a defect on its path.
- **Measure a repair against a build carrying the defect, not against the
  stored corpus.** Task 10a's live-vs-stored comparison reported 556
  moved values across four blocks including the muon row; the repair
  moves 215, in three blocks, in one row. The rest is the ulp-level
  platform drift the case budget already absorbs (`numpy 2.5.1 → 2.5.3`,
  `scipy 1.18.0 → 1.18.1`, a macOS point release). Capture the blocks
  twice — defective build, then repaired build — and diff those.
- **A declared position at exactly 0.500000 is a second defect, not a
  rounding.** B5's factor-of-two positions are where the muon-decay
  continuum's own quadrature returns a hard zero: its integrand's support
  ends at 69.7835 MeV and the boost window runs hundreds of times wider,
  so every QUADPACK abscissa falls outside it — A3's failure mode, in the
  neutrino kernel. `rust/src/kernels/positron_pion.rs:163-164` clips both
  ends of the same kind of window and is the precedent for the fix. Filed
  as
  [`neutrino-pion-continuum-loses-its-quadrature-support.md`](../../../docs/followups/todo/neutrino-pion-continuum-loses-its-quadrature-support.md);
  it moves published numbers by three to four decades on a band, far more
  than B5 itself.
- **A declaration cannot be written before its repair lands.**
  ADR-0001's staleness rule fails a declared array that still equals the
  corpus, so Task 3's three models sit in `deltas.DELTA_MODELS` and Tasks
  5, 6 and 9 add the `DECLARED_DELTAS` keys. `PLAN.md` Task 3's gate and
  `references/corpus-repinning.md` are patched to say so.
- **Two of the three Group B defects miss a rest block, and B2 misses
  both.** `photon_tables::dnde` adds line terms only in flight, so B1
  never reaches `spectra.photon.eta_prime[rest]`; and at
  `beta = 1.414e-06` the phi's two windows are 2.8e-06 wide in relative
  energy with no grid point inside, so B2 reaches neither rest block. The
  corpus anchors the `M/2` line images but not the phi's, which is the
  whole difference.
- **The phi's shipped lines sit above the spectrum's own endpoint**,
  where the boosted continuum is exactly zero — which is why B2 is
  readable straight off the committed arrays as a two-tread staircase.
- **A model test that recomputes the model tests nothing.** Three of ten
  mutations to `test/parity/deltas.py` left the first draft of
  `test_delta_models.py` green, including B1's weight doubled and B3's
  transform inverted. Every model test now drives `DELTA_MODELS[...]`
  rather than a local re-spelling. Related to
  `[test-name-claims-an-unmade-assertion]` but distinct: the name was
  right and the assertion was real; it just was not about the code under
  test.
- **Lesson classes this project is most exposed to**, from
  `docs/agents/lessons.md`: `[exemption-wider-than-its-mechanism]` (a
  declaration written wider than the mechanism that earned it),
  `[platform-scoped-oracle-asserted-globally]` (the Task 2 capture is a
  locally compiled oracle — declare the scope from the corpus manifest's
  platform, never probe for it), `[mutation-harness-poisons-its-own-baseline]`
  (Task 2 patches and reverts `.pyx` files repeatedly),
  `[measured-tree-vs-imported-module]` (what the capture imports and what
  it hashes must be proven the same tree), and
  `[test-name-claims-an-unmade-assertion]` (several existing tests are
  named for the defect they pin and need renaming, not just
  re-pointing).

## Numerical impact so far

**Two public values have moved: B4, in PR #87, and B5, in Task 10a**
(bullets below). Task 2
shipped no library behavior —
its four `.pyx` patches exist only inside the capture and are reverted.
What it produced is the *measurement* each of Tasks 4, 7, 8 and 10 will
be judged against: how far the corrected value sits from the committed
corpus array, taken from patched Cython rather than from the Rust that
will be repaired. Full per-case table in
`task-2-cython-oracles.md`; the headline per defect, on the corpus
grids:

| Defect | Repair task | Cases moved | Positions moved | Largest relative shift | Direction |
| --- | --- | --- | --- | --- | --- |
| A1 boost window | 4 | 7 of 7 predicted | 4154 | ~1.0 (shipped up to 9,800× high near threshold) | **both** — down in `rest_plus_eps`, up in the boosted blocks |
| A2 muon endpoint | 7 | **1** of 7 predicted | 4 | n/a (`0.0` → negative) | down; all four values become negative |
| A3 pion cone | 8 | 6 of 6 predicted | 6359 | 7.77 | both |
| A4 positron norm | 10 | 6 of 6 predicted | 21,975 | `0.000374207` uniformly | up, at every position |
| B5 pion neutrino line | 10a — **landed** | 1 of 1 | 215 | exactly 0.5 | down, at every position |

Three things in there are corrections to the plan rather than
confirmations of it, and Tasks 4, 7 and 8 inherit them: A1's sign is not
one-signed, A2 reaches one corpus case instead of seven, and A2's whole
delta is four negative values replacing zeros. `PLAN.md`'s Task 4, 7 and
8 gates and `references/defect-blast-radius.md` have been patched to
match; the task note carries the evidence.

The physics invariants each patched build was asked for, while it still
existed (`../rules.md` rule 4) — none of these is re-runnable, so they
are recorded rather than re-derivable:

- **A1**: both of the follow-up's hand-computable cases land on the
  closed form (1.933333 and 3.500000, against shipped 1.266667 and
  53.497500).
- **A2**: the O(α) rest-frame formula's own zero is at 52.808176 MeV,
  0.019774 MeV below the kinematic endpoint it is now guarded at.
- **A3**: `dnde_photon_charged_pion(900, 1396)` = 3.585860e-07 MeV⁻¹
  against the follow-up's predicted 3.586e-07; yield 0.0808/0.0807/0.0806
  photons per decay at `E_π` = 1000/1396/5000 MeV.
- **A4**: the Michel spectrum integrates to 1.000000000000 (one ulp) at
  rest and at both boosts; shipped, 0.999625933330.
- **B5** (repaired in Task 10a): `hazma.spectra.dnde_neutrino_charged_pion`
  loses one `BR(π → e ν_e) = 1.230e-4` from its electron-neutrino row per
  pion, 0.0123% of the row integrated. The muon and tau rows are
  bit-identical, `spectra.neutrino.muon` and the nine tabulated
  `dnde_neutrino_*` do not move, and `dnde_neutrino` moves only for final
  states containing a charged pion. Two grids, both taken from a build
  carrying the defect against the repaired one. On
  `np.geomspace(1e-4, 1e5, 2001)` at `E_π` = 200, 400, 1000, 5000 MeV the
  median pointwise drop is 4.7e-5 to 1.1e-4 and the maximum is exactly
  `0.500000000000`. On the corpus grids it moves 215 of the case's 4,305
  pinned values, from 4.716e-5 up to that same `0.500000000000` at 14 of
  them — the positions where the muon-decay continuum's quadrature returns
  zero. Declared relation held to 3e-12 (measured 1.494e-15).
  Details in `task-10a-neutrino-pion-line.md`.
- **B4** (repaired in PR #87, outside the task sequence): the scalar
  decay kernel's FSR-only spectrum at rest was 0.5000000000 × the
  annihilation-side `dnde_xx_to_s_to_ffg` / `dnde_xx_to_s_to_pipig` in
  every channel; now 1.000. `scalar_mediator_decay_spectrum`, all 15
  `.default` blocks, both arrays: 3,065 of 4,305 positions are declared
  (the FSR term is non-zero there), 2,874 of them move by more than
  0.1%, pointwise up to exactly 2×; photon yield per decay +1.7% to
  +2.9% with the corpus couplings. Declared relation held to 1e-3
  (measured 3.1e-4); the other 1,240 positions stay at the case budget.
  Counts from the command in `task-1-delta-declarations.md`. Details in
  `docs/followups/done/scalar-decay-fsr-half-normalized.md`.

**Task 3 moved nothing** — it models B1, B2 and B3 rather than repairing
them, and its diff contains no library or build file. What it measured
off the committed corpus, for Tasks 5, 6 and 9 to be judged against:
B1's eta-prime plateau reads `1 · BR_ETAP_TO_A_A` where its three correct
siblings read `2 · BR` (22 line/block readings, worst 4.9e-13); B2's two
phi lines sit at 656.942002472385 and 959.6459594437648 MeV, above the
spectrum's own endpoint, as a staircase matching the modelled treads to
the last bit; and B3's `stored × E_γ` reproduces the same case's
`rest_plus_eps` block to 6.8e-11 (charged rho) and 1.8e-09 (neutral).
Reach: B1 six arrays / 189 positions, B2 six / 305, B3 four / 350.

Tasks 4–10 each move a published spectrum by design, and each records the
function, the grid and the max shift here in its own PR (`../rules.md`
rule 10). Task 12 aggregates this section into the `CHANGELOG.md` entry —
it does not reconstruct it. B4 and B5 have each written their own
`CHANGELOG.md` entry already, B5's under `[Unreleased]` because 2.2.0 is
released; Task 12 renames that heading rather than re-deriving it, and
`preflight.sh --closing` greps for `## [<new version>]`, so it must.

## Decisions and Implementation Notes

- **The corpus is extended, not regenerated.** The committed arrays stay
  as the record of what 2.1.0 shipped; each repair adds a declared delta
  against them. Rationale and schema in
  `../references/corpus-repinning.md`; decided in
  `../adrs/ADR-0001-corpus-repairs-are-declared-deltas.md`.
- **B4 was found and repaired outside the task sequence** (PR #87). A
  user report led to the scalar decay kernel's FSR, which no roster
  entry covered; the repair could not merge without the Task 1 layer,
  so the layer landed with it. Its case is inside A3's set, so Task 8
  inherits a rule 7 obligation against a declaration that already
  exists.
- **B5 was rostered rather than repaired ad-hoc** (Task 10a). Its
  follow-up was filed 2026-08-20, one day after this plan, and its
  "Triggers / blockers" bullet already named this plan as its home — but
  no row was added, so the plan counted eight defects against a
  population of nine. It is numbered **Task 10a** because
  `scripts/agents/resolve_task.py` sorts on `(\d+(?:\.\d+)*)\s*([a-z]*)`,
  so a letter suffix inserts between Tasks 10 and 11 without renumbering
  eleven tasks and their citations; Task 12 has to close last and Task 11
  sweeps this follow-up, so both sides of the position are forced.
- **B5's follow-up stays in `todo/` until Task 12** even though the
  repair has landed, so the inbound-reference repoint
  (`docs/workflow.md#follow-ups`, and
  `docs/followups/todo/moved-followups-leave-dangling-inbound-paths.md`)
  happens once for all eight rather than eight times. Its `Status:` line
  records the repair and says so.
- **A relation returns the repaired array, not an additive term**
  (Task 3). `Additive.expected` and `Exact.expected` both answer "what
  should this array be", and the runner derives the moved mask from
  `predicted != stored`. It is what lets B3 be `Exact` — a transform of
  the stored array, bit-exact where `stored + term` would cost a second
  rounding — and it makes `MOVED` mean exactly "float64 moves here".
- **Group B needs no `mpmath` reference** (Task 3). `reference.py` exists
  for kernels that lose ~33 digits to an `atan` cancellation; measured at
  60 dps, these three forms lose under half a digit. The committed corpus
  is the oracle instead, and `mpmath` stays out of the test path, as
  `pyproject.toml`, `stability.py` and `reference.py` all three say.
- **The seven follow-ups' "Risks" sections were deliberately left
  standing** when their "Triggers / blockers" bullets were corrected, so
  the correction and the plan that justifies it would land in one
  reviewable place first. Each corrected bullet says so and points here.
  Task 11 sweeps them.
- **The Group A deadline binds on the oracle capture, not on the
  repair** (PR #72 review round 1). The first draft of the four Group A
  blocker bullets said "fix BEFORE Task 6.4", which reads as an
  instruction to land the Cython fix, the Rust fix and the corpus change
  together before the deletion — the thing this plan deliberately
  decomposes. Under the delta mechanism the repair is legal on a tree
  with ported kernels at any time; what cannot follow the deletion wave
  is capturing the corrected values from the twin. The bullets now say
  that, and point at Task 2 (capture) and the specific repair task
  separately.
- **`references/defect-blast-radius.md` is the canonical case
  enumeration; `PLAN.md` quotes it by row** (same review). The reference
  originally brace-elided its case lists, and the plan's gates then said
  "both mediator photon cases" against a population of three and "both
  mediator positron cases" against four — each mediator ships a
  `dnde_decay_*` and a `dnde_decay_*_pt` entry point. Every list is now
  written out with a count, and each repair gate names every case.

## Files Changed

The change that created this project touched only
`docs/followups/todo/*.md` (seven blocker bullets, one hunk per file),
`projects/README.md` (the Active Projects row) and
`projects/parity-pinned-defect-repair/` itself. Task 2 added
`test/parity/oracles/`. Task 1 landed in PR #87 with the B4 repair:
`test/parity/deltas.py` (new), `test/parity/test_parity.py`,
`test/parity/README.md`, `rust/src/kernels/scalar_decay_photon.rs`,
`test/test_core_mediator_decay_photon.py`,
`hazma/scalar_mediator/_scalar_mediator_spectra.py`, `CHANGELOG.md`,
`docs/followups/`, and this project's plan, rules, references, ADR and
notes — see `task-1-delta-declarations.md`.

### Task 3

`test/parity/deltas.py` (the `Exact` relation, the `expected` protocol,
the B1/B2/B3 models, `DELTA_MODELS`), `test/parity/test_delta_models.py`
(new), `test/parity/test_parity.py`, `test/parity/README.md`,
`PLAN.md` (Task 3's gate and two scope notes),
`references/corpus-repinning.md`, and `task-3-closed-form-deltas.md`. No
library or build file, and `test/parity/data/` untouched.

### Task 10a

`rust/src/kernels/neutrino_pion.rs`, `test/parity/deltas.py`,
`test/parity/test_parity.py`, `test/test_core_neutrino.py`,
`CHANGELOG.md`, `docs/followups/todo/neutrino-pion-electron-line-counted-twice.md`,
`docs/followups/todo/neutrino-pion-continuum-loses-its-quadrature-support.md`
(new), `docs/followups/README.md`, `../PLAN.md`,
`../references/defect-blast-radius.md`, this file, and
`task-10a-neutrino-pion-line.md` (new). `test/parity/data/` untouched.

## Verification

**PR #87** (Task 1 + B4) was gated with the bare `preflight.sh` over its
Python paths and markdown; the pytest row read `2241 passed, 15 skipped`
before this round's tests were added — the Task 1 note has the final
figures.

**The scaffolding change itself** (the seven corrected blocker bullets +
this project tree; no task of this project had run yet) was gated with:

```sh
scripts/agents/preflight.sh \
    --paths "test/parity/generate.py test/parity/cases.py test/parity/stability.py" \
    --md "$(git show --name-only --format= HEAD | grep '\.md$' | tr '\n' ' ')"
```

`RESULT: PASS` — every row green, `pytest` at `1810 passed, 15 skipped`
on a tree cleaned of stale `.c`/`.so` and rebuilt with
`uv pip install -e .`. Two notes for whoever repeats it. `--paths` names
Python the diff does not contain, because the diff contains none and
omitting the flag selects the `hazma test` directory form that is red on
the trunk for reasons unrelated to any branch. And the gate does not run
`scripts/agents/check_doc_citations.py` — that was run separately over
all 18 changed docs (15 in-repo citations, none out of range), together
with a script confirming all 52 relative markdown links in those docs
resolve.

- Tasks 1–3: `pytest test/parity` (collected count unchanged),
  `python test/parity/generate.py --check`, the new oracle `--check`.
- Tasks 4–10: `pytest test/parity` plus `pytest test/test_theory_aggregation.py`,
  the per-task mutation pair (revert → red, widen → red), and
  `cargo test --manifest-path rust/Cargo.toml --no-default-features`.
- Task 11: `scripts/agents/check_doc_citations.py` over every touched
  doc, with paths passed explicitly while fixes are uncommitted.
- Task 12: `scripts/agents/preflight.sh --closing`.

Every one of these needs a built tree
(`uv pip install -e . --config-setting build-args="--features
test-probes"` — uv spells the flag singular where pip spells it
`--config-settings`); a non-editable install leaves no extension where
the corpus insists on measuring one, and one built without the feature
has no `hazma._core` test probes for the suite to import.

## Open Questions

- **Does the boost integral run at β = 0?** If the tabulated kernels
  short-circuit at rest, A1's declaration excludes every `rest` block
  and the case count in `../references/defect-blast-radius.md` shifts.
  Measured in Task 4, not assumed.
- **Do A2's and A3's declared positions on the two rho cases actually
  overlap, or only appear to?** They are different mechanisms (an
  endpoint guard and a lost quadrature support) reaching the same arrays
  through the same nesting. `../rules.md` rule 7 forces the question to
  be answered rather than absorbed.
- **What happens if `cython-to-rust` reaches Task 4.6 before Task 2
  lands?** The A4 `spectra.positron.charged_pion` oracle becomes
  unrecoverable from anything but the repaired Rust. The fallback is a
  closed-form model (the defect is an overall factor, so the delta may
  be expressible as one) — but that has to be established, not assumed,
  and the loss recorded rather than papered over.
- **How many kernels boost a bounded spectrum over an unclipped window?**
  Task 10a found `neutrino_pion.rs` doing it and `positron_pion.rs`
  clipping correctly, but only because a repair's magnitude looked wrong.
  The sweep of the remaining `quad` call sites has not been done —
  [`neutrino-pion-continuum-loses-its-quadrature-support.md`](../../../docs/followups/todo/neutrino-pion-continuum-loses-its-quadrature-support.md)
  carries it as its own first question.
- **Will Group B's relation budgets survive their repairs?** B1 and B2
  carry `rtol` 1e-11 and B3 1e-9, each derived from the case's own budget
  rather than measured against a repaired kernel, which cannot exist yet.
  Tasks 5, 6 and 9 measure the real figure and tighten; a repair needing
  a *wider* one has found something Task 3 did not model (`rules.md`
  rule 2).
- **Do B1's and B2's positions overlap A1's on the same arrays?** Both
  land on arrays Task 4 also moves, and rule 7 forbids an overlap. The
  mechanisms are disjoint — `boost_delta_function` against
  `boost_integrate_linear_interp` — but the position sets have not been
  intersected. Tasks 5 and 6 inherit that as a gate.
- **Should the Task 2 oracles stay committed after `cython-to-rust`
  closes?** They are the last evidence that a repaired value was ever
  checked against a non-Rust implementation. Anticipated ADR.

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).

## Handoff to Next Task

**For the next agent starting any task in this project:**

1. Read `../PLAN.md` end-to-end once — especially "The premise this
   project corrects" — then this file, then `../rules.md`.
2. Read the task's detail block in `../PLAN.md` and the references its
   detail names.
3. Check "Open Questions" above.
4. Build first: nothing is prebuilt on a fresh worktree, and stale
   generated `.c`/`.cpp` must be cleaned before the build
   (`docs/agents/environment.md`).

**Currently safe to assume:**

- The four Group A `.pyx` twins are present and buildable on this tree,
  verified at `3e01590`.
- `test/parity/data/` is intact — `python test/parity/generate.py --check`
  verifies it in under a second with no build.
- B4 (PR #87) and B5 (Task 10a) are the repairs that have changed a
  library value; every other corpus array is still compared against its
  stored value. `test/parity/deltas.py` declares 36 arrays, and
  `test_parity.EXPECTED_DECLARED_ARRAYS` is the literal that makes a
  change to that number show up in a diff.
- The measurement recipe every remaining repair task needs: capture the
  corpus blocks from a build carrying the defect, restore the repair,
  rebuild, capture again, and diff *those* — not the live tree against
  the stored corpus, which reports the platform drift as if it were the
  repair (Findings).
- B1, B2 and B3 have finished, corpus-checked delta models in
  `deltas.DELTA_MODELS`. Tasks 5, 6 and 9 fix the kernel, add the
  `DECLARED_DELTAS` keys pointing at those models, and bump
  `EXPECTED_DECLARED_ARRAYS`; the key lists and position counts are in
  `task-3-closed-form-deltas.md`'s handoff, to be re-derived after
  Task 4 rather than pasted.

**Currently risky / unknown:**

- The blast-radius table is derived from the composition graph, not
  measured. Treat it as where to look, never as what you will find. B5 is
  the sharpest case so far: the table's coverage arithmetic said both
  `spectra.neutrino.*` cases were untouched, and one of them was not.
- A repair's own follow-up can understate its magnitude by orders of
  magnitude. B5's predicted 0.06% local shift measured at a factor of
  two, because the follow-up sampled only the plateau. Re-derive every
  figure before quoting it in a `CHANGELOG.md` entry.
- Every deadline in this plan depends on `cython-to-rust`'s pace, which
  this project does not control and must not assume.
