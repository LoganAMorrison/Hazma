# Project Retrospective: parity-pinned-defect-repair

**Closed:** 2026-09-24, Task 12, shipped as hazma 2.3.0.
**Span:** 2026-08-19 to 2026-09-24. Fourteen tasks (1–13 and 10a), with
B4 repaired ahead of the sequence in PR #87.
**Sources:** `../task-notes/README.md` (Findings, Decisions, "Numerical
impact so far") and the per-task notes it links. Durable decisions are in
[ADR-0001](../adrs/ADR-0001-corpus-repairs-are-declared-deltas.md),
[ADR-0002](../adrs/ADR-0002-retain-the-signed-muon-endpoint-approximation.md)
and [ADR-0003](../adrs/ADR-0003-keep-the-group-a-oracle-captures-committed.md).

This file replaces the task notes for later readers. Open a note only
when a section below points to it for a specific detail.

## 1. Implementation Reality Check

**The deliverable held; the roster and the magnitudes did not.** The plan
was drawn for seven defects. It closed with ten. B4 was found from a user
report and repaired in PR #87, before the layer existed, so the layer
landed with it. B5's follow-up was filed the day after the plan and named
the plan as its home, but no row was ever added (Task 10a). B6 was
corpus-pinned and had nowhere else to go (Task 13). All ten were
repaired, and `test/parity/data/` is byte-identical to where it started:
`git diff --stat f11415a^ HEAD -- test/parity/data` prints nothing.

**The premise was right, and it was also more urgent than it looked.**
The plan's central correction was that waiting for `cython-to-rust` Task
6.4 would destroy the only independent oracle. Task 2 captured all four
Group A oracles on 2026-08-19, ahead of every deletion wave. Nothing
after that was time-critical. The larger surprise, recorded in
`../references/the-premise.md`, was that the corpus had been impossible
to regenerate since Phase 04 Task 4.1. The planned "one declared
regeneration after Task 6.4" had never been available at all.

**Every magnitude the follow-ups predicted was re-measured, and several
changed.**

- A1's sign was not uniform. Near threshold the shipped values were
  thousands of times too high.
- A2 reached one corpus case rather than seven, and all four positions
  it moves become negative.
- B5's predicted 0.06% local shift was a factor of 2 at the top of the
  spectrum.
- B1's 0.63% was 0.603%, and is better quoted as the window-independent
  `+BR`.
- B6's "0.5%–5%" was up to 100% in ⟨σv⟩ and −99.9% in `relic_density`.

The per-defect table is in `CHANGELOG.md` `[2.3.0]`, with its grids in
`../task-notes/README.md`.

**The mechanism grew four times, each time because a real repair needed
it.**

- `Reference` (Task 13): a quadrature that was never resolved has no
  closed-form delta.
- `Composed` (Task 5): two repairs land on one array, and there is only
  one key per array.
- An absolute floor (Task 6): a relocated line cancels a term the
  capture never held.
- Per-case model variants (Tasks 8 and 10): these keep each case's own
  budget.

ADR-0001 was amended for each. No tolerance in `tolerances.py` was
widened (`../rules.md` rule 2).

## 2. Critical Context for Future Work

- **The corpus is extended, never regenerated.** A repair that moves a
  pinned value declares it in `test/parity/deltas.py`. Each key is
  `(case, block, suffix)` and maps to one relation. Every undeclared
  position is still held to the stored array under its
  `tolerances.py` budget. At close, 343 arrays are declared; re-derive
  the count with `Counter(d.repair for d in deltas.DECLARED_DELTAS.values())`.
  `test_parity.EXPECTED_DECLARED_ARRAYS` is the literal that makes a
  change in the count show up in review.
- **Picking a relation.** Use `Additive` when the physics names the
  term, `Exact` when a closed form transforms the stored array,
  `Reference` when only a second implementation can say what the value
  should be, and `Composed` when a second repair lands on an array that
  is already declared. All four answer `expected(fn, block, stored)` and
  return the *repaired* array. The runner takes the moved mask from
  `predicted != stored`.
- **A declaration must describe a change.** ADR-0001's staleness rule
  fails any declared array that still equals the corpus. A model
  therefore cannot be declared before its repair lands. It waits in
  `DELTA_MODELS` (Task 3).
- **A composite covers only the arrays that every named repair moves.**
  A1 alone keeps the others. Otherwise the label is wider than its
  mechanism.
- **`deltas.REPAIRS` is a closed set of ten.** A repair filed after this
  project has no label scheme yet. See §5.
- **The Group A captures are load-bearing.** 321 of the 343 declared
  arrays read `test/parity/oracles/data/` through
  `test/parity/oracle_reference.py`, which maps `(module, qualname)`
  through `cases.build_cases()`, not through the manifest's
  `entry_point`. The manifest records where each case was captured,
  and for Group A that is a deleted `.pyx`. ADR-0003 keeps the captures.
- **ADR-0002.** The muon rest-frame spectrum keeps its signed O(α) tail.
  This preserves the in-flight boost identity, and the public docstring
  states the limitation.

## 3. Quirk Log & Edge Cases

- **Measure a repair against a build that still has the defect, not
  against the stored corpus.** On B5, a live-vs-stored diff reported 556
  moved values in four blocks. The repair moves 215, in three blocks.
  The rest was ulp-level platform drift that the budget already absorbs.
  Capture twice (defective build, then repaired build) and diff those.
- **A Group A capture corrects its own defect and nothing else.**
  `oracles/defects.py` patches one `.pyx` per defect. `A1.npz` therefore
  still carries B1's bare `BR`, which is why `A1 capture + B1 term` is
  the repaired value and not a double count.
- **A relocation cannot be declared exactly.** Where the shipped line
  was most of the value, a plateau of 3.1e-5 over a continuum of 3.3e-21,
  the prediction reads exactly zero. `A1+B2` declares a floor of 1e-20,
  and a test caps every floor below the array's own zero floor.
- **The blast-radius table is a reading of the composition graph, not a
  measurement.** Every measurement came back narrower, or reached
  somewhere the table said nothing reached (B5).
- **A defect copied verbatim from a `.pyx` can have call sites the port
  never touched.** Two of B6's four sites were pure Python.
- **Zeroing `epsabs` is half of a convergence fix.** The limit has to be
  reachable too: at `quad`'s default of 50 subdivisions, 33 of 540
  thermal positions came back flagged.
- **A position that moves by exactly 0.500000 is a second defect, not
  rounding.** In B5 those positions are where the muon continuum's
  quadrature returns a hard zero: the window is wider than the support,
  which is A3's failure mode. That is filed separately (§5).
- **A model test that recomputes the model tests nothing.** Three of
  ten mutations to `deltas.py` left the first draft of
  `test_delta_models.py` green. Every model test now drives
  `DELTA_MODELS[...]` itself.

## 4. Test Infrastructure State

- `test/parity/deltas.py` has the four relations, `DECLARED_DELTAS`,
  `DELTA_MODELS` and `REPAIRS`. `test/parity/test_parity.py` is the
  runner plus the shape tests: no overlap, no stale declaration,
  `EXPECTED_DECLARED_ARRAYS` and `EXPECTED_FLOORED_ARRAYS`.
  `test/parity/test_delta_models.py` checks each model against the
  stored arrays without evaluating a kernel.
- `test/parity/oracles/` holds the Task 2 captures. It self-checks with
  `python test/parity/oracles/capture.py --check`, which needs no
  build. `test/parity/test_oracles.py` checks that the roster and the
  capture agree.
- `test/parity/thermal_reference.py` is B6's reference: scipy QUADPACK
  on the same integrand at `epsrel = 1e-12`.
- `python test/parity/generate.py --check` verifies the stored corpus
  in under a second.
- The mutation pair per repair: reverting the repair turns the gate
  red, and widening the declaration turns it red too. Use it for any
  later declaration.

## 5. Follow-on seeds

Each substantive seed has a live follow-up. Four were filed during the
project, and one is filed with this retrospective.

- **The rho's outer boost can miss its support** (Task 8):
  [`rho-photon-outer-boost-misses-support.md`](../../../docs/followups/todo/rho-photon-outer-boost-misses-support.md).
  The inner pion repair (A3) left it standing. It needs an independent
  reference beyond the A3 capture.
- **The charged pion's neutrino continuum loses its quadrature support**
  (Task 10a):
  [`neutrino-pion-continuum-loses-its-quadrature-support.md`](../../../docs/followups/todo/neutrino-pion-continuum-loses-its-quadrature-support.md).
  It moves published numbers by three to four decades on a band. That
  is far more than B5 itself. The file also carries the sweep of every
  `quad` call site that boosts a bounded spectrum over an unclipped
  window.
- **The φ omits its direct `φ → π⁰γ` line** (Task 6):
  [`phi-omits-its-direct-pi0-photon-line.md`](../../../docs/followups/todo/phi-omits-its-direct-pi0-photon-line.md).
  It was deliberately not folded into B2, because B2 relocates the
  yield and this would raise it.
- **Every decaying-theory positron channel evaluates the last channel**
  (PR #99 review, Task 10):
  [`decaying-theory-positron-channels-share-the-last-closure.md`](../../../docs/followups/todo/decaying-theory-positron-channels-share-the-last-closure.md).
- **The delta layer outlives the project that owns its ADR** (filed with
  this retrospective):
  [`delta-declaration-layer-outlives-its-project.md`](../../../docs/followups/todo/delta-declaration-layer-outlives-its-project.md).
  It covers promoting ADR-0001 to `docs/adrs/` and deciding how a later
  repair joins the closed `deltas.REPAIRS`. The first three seeds above
  will each need that.

B6's review also filed
[`thermal-fallback-upper-limit-collapses-at-x-25.md`](../../../docs/followups/todo/thermal-fallback-upper-limit-collapses-at-x-25.md)
(PR #91 review round 1). It is a separate defect in the pure-Python
fallback's upper limit and is still open.

## 6. Process notes

- **Closing all the follow-ups in one move paid off.** Holding every
  repaired follow-up in `todo/` until Task 12 meant their inbound
  references were repointed once instead of eight times: 159 path
  occurrences across 41 files, plus eight index rows and seven relative
  links between `todo/` and `done/`. The cost was eight `Status:` lines
  that read "repaired, retained here" for up to 18 days (B5, repaired
  2026-09-06).
- **Rule 10 made the close an aggregation.** Each repair wrote its own
  function, grid and maximum shift into "Numerical impact so far" and
  its own `CHANGELOG.md` entry. Task 12 renamed the heading and added a
  summary table, and re-derived nothing.
