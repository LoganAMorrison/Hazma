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
| 4 | Repair A1 — boost integral window | 1, 2 | **Complete** | `task-4-boost-window.md` |
| 5 | Repair B1 — η′ line weight | 3, 4 | **Complete** — declared as the composite `A1+B1` | `task-5-eta-prime-line.md` |
| 6 | Repair B2 — φ line energies | 3, 4 | Not started | `task-6-phi-lines.md` |
| 7 | Repair A2 — muon photon endpoint | 1, 2 | Not started | `task-7-photon-muon-endpoint.md` |
| 8 | Repair A3 — charged-pion forward cone | 7 | Not started | `task-8-charged-pion-cone.md` |
| 9 | Repair B3 — rho rest-frame branch | 3, 8 | Not started | `task-9-rho-rest-frame.md` |
| 10 | Repair A4 — positron-muon normalization | 1, 2 | Not started | `task-10-positron-muon-norm.md` |
| 10a | Repair B5 — charged-pion neutrino line | 1 | **Complete** | `task-10a-neutrino-pion-line.md` |
| 11 | Reconcile the superseded sequencing prose | 4–10a, 13 | Not started | `task-11-prose-reconciliation.md` |
| 12 | Close — aggregate the drift, bump | 11 | Not started | `task-12-close.md` |
| 13 | Repair B6 — thermal quadrature convergence | 1 | **Complete** | `task-13-thermal-quadrature.md` |

```text
1 ──┬──► 3 ──┬──────────► 5 ──┐
    │        │                │
    ├──► 4 ──┴────────────────┼──► 11 ──► 12
    │        └──► 6 ──────────┤
    ├──► 10a ─────────────────┤
    └──► 13 ──────────────────┤
2 ──┼──► 7 ──► 8 ──► 9 ───────┤
    └──► 10 ──────────────────┘
```

Task 2 had no upstream dependency and the project's only hard external
deadline. It is done: every Group A oracle is captured and committed
under `test/parity/oracles/`, so `cython-to-rust` Tasks 4.6, 6.2, 6.3 and
6.4 may now run in any order without stranding a repair. Nothing else in
this project is time-critical.

## Exit Criteria

- All fourteen tasks complete; all ten defects repaired — the follow-ups
  under `docs/followups/todo/` moved to `docs/followups/done/` with
  inbound links repointed and the revision pinned (B4's and B6's already
  live there; B5's is repaired but deliberately still in `todo/`, so the
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
- **A defect can live in the integrator rather than in a closed form,
  and the roster had no shape for one** (Task 13). B6's stored values
  are not the true ones transformed; they are the true ones unresolved,
  so no `Additive` or `Exact` relation can state the delta. The layer
  grew a `Reference` relation, which answers Task 3's `expected`
  protocol alongside `Additive` and `Exact`.
- **Zeroing a tolerance is only half a convergence fix.** B6's kernels
  also needed `THERMAL_LIMIT` above `quad`'s default of 50: at 50, 33 of
  the 540 thermal positions exhausted the subdivision table and came
  back flagged. A criterion that binds has to be reachable.
- **A capture-backed `Reference` needs the entry point to name its
  case.** `Delta` is one object per roster label — `test_parity`'s
  `test_every_declaration_points_at_a_delta_model` compares by `id` — so
  a repair spanning seven cases cannot close a case name over its
  relation. The corpus manifest's `entry_point` cannot resolve it either:
  it records where each case was *captured* from, which for every Group A
  case is a `.pyx` the port deleted. `cases.build_cases()` is the live
  authority, and `test/parity/oracle_reference.py` maps
  `(module, qualname)` through it for A2, A3 and A4 as well.
- **A declaration written as `MOVED` over a whole block is a wall for the
  next repair that lands on the same array.** A1 covers all four non-rest
  blocks of `eta_prime` and `phi`, which is every array B1 and B2 move.
  There is no second key to add — one key, one `Delta` — so those repairs
  compose their relation with A1's rather than declaring beside it.
  Foreseen by `../rules.md` rule 7; what was not foreseen is that the
  mechanism makes the wrong answer inexpressible rather than merely
  forbidden.
- **A physics invariant may have to switch off an unrepaired
  faithfulness item to be about the repair.** The boost's yield identity
  `∫dE dN/dE = ∫dx y(x)` fails by 5.6% on any table with a nonzero first
  row, because the below-table `1/E` tail — still reproduced from the
  Cython, not repaired — adds photons the table does not contain. A table
  vanishing at `x[0]` zeroes that term and leaves the identity exact.
- **Four call sites shared B6's defect and only two were in Rust.** The
  two pure-Python ones (`hazma/relic_density/_thermal_functions.py`, the
  GeV vector model) were never ported, so a Rust-only repair would have
  left them. Worth checking for the same shape on any defect inherited
  verbatim from a `.pyx`.

- **A Group A capture carries every *other* defect its case had.**
  `oracles/defects.py` patches exactly one `.pyx` per defect, so
  `oracles/data/A1.npz` was taken from a restored Cython η′ that still
  wrote a bare `BR`. That is what makes `A1 capture + B1 term` the
  repaired value rather than a double-count, and the same holds for φ and
  B2 (Task 5). A capture is a corrected value for *its* defect only.

## Numerical impact so far

**Five public values have moved: B4 in PR #87, B5 in Task 10a, B6 in
Task 13, A1 in Task 4 and B1 in Task 5** (bullets below).

**A1 — all seven tabulated photon spectra.** Measured on the corpus's
own grids (10,045 positions, 35 blocks) against a build carrying the
defect, which reproduced the stored arrays bit for bit, so the whole move
is the repair. 4,154 positions move and **the sign splits by block**:

| Block | γ | moved / pinned | Magnitude |
| --- | --- | --- | --- |
| `rest` | 1 | 0 / 1841 | no caller reaches the integral at β = 0 |
| `rest_plus_eps` | 1 + 1e-12 | 1156 / 2051, all **down** | shipped a median **9,768x** and up to **360,507x** too high |
| `near_rest` | 1.05 | 1130 / 2051, all up | median +3.34e-02, max +98.7% |
| `boosted_mild` | 2 | 1028 / 2051, all up | median +2.19e-03 |
| `boosted_strong` | 10 | 840 / 2051, 839 up | median +7.09e-05 |

Off the corpus grid, the limit that says the window is now covered: at a
parent one part in 1e12 above rest all seven agree with their own
rest-frame spectrum to better than 1% over `E_γ` from `m/20` to `3m/10`,
against the 6,500x–33,000x the follow-up measured. The repaired kernel
reproduces Task 2's Cython oracle **bit for bit** at all 10,045
positions. Details: `task-4-boost-window.md`.

**B1 — `dnde_photon_eta_prime`, and `dnde_photon` on any final state
carrying an η′.** Measured before and after on the same worktree by
reverting the constant, rebuilding the editable install, and diffing the
two captures. The spectrum rises at every parent energy above rest and is
unchanged at `E = M_η′`, where the kernel takes its rest-frame arm and
adds no line: 7/601 grid points move at `E = 1.001 M`, 137/601 at
`1.5 M`, 325/601 at `5 M`, all upward, by 7.7e-04 to 1.0 relative. The
n-body path moves with it (36/201 on `["etap", "etap"]`, 45/201 on
`["etap", "eta"]`, 0/201 on `["eta", "eta"]`). The yield rises by
**exactly `BR_ETAP_TO_A_A` = 0.02307 photons per decay** at any boost —
a boosted δ-function integrates to its own weight — which is 0.603% of
the repaired 3.8291 over `1e-3 ≤ E ≤ E_parent` at `E_parent = 2 M`.
`PLAN.md`'s pre-repair 0.63% was over a narrower total; quote the yield
rather than the percentage. On the corpus: 189 positions over 6 of this
case's 10 value arrays, none in either `rest` array and none at the
scalar probe of `rest_plus_eps` or `near_rest`. The four `X → γγ` line
integrals now all equal their declared weight to a ratio of
1.0000000000. `spectra.photon.{eta,long_kaon,short_kaon,omega,phi}` are
bit-identical across the rebuild. Details: `task-5-eta-prime-line.md`.

**B6 — both mediator `thermal_cross_section`, and `relic_density`
through them.** Measured on the two corpus cases' own grids (570
positions, 6 blocks, `x = mx/T` from 0.1 to 1000) against
`test/parity/thermal_reference.py`, scipy's QUADPACK on the same
integrand at `epsrel = 1e-12`:

| Quantity | Grid | Shipped error | After |
| --- | --- | --- | --- |
| ⟨σv⟩, scalar + vector | 570 corpus positions | up to **1.00 relative** (both `closed_resonance` blocks); per-block medians 7.2e-6 to 8.1e-2 | within 3.6e-9 of the reference |
| `relic_density`, semi-analytic | the 6 pinned model points | −91.85% to +2.04% | re-pinned |
| `relic_density`, Boltzmann | the 6 pinned model points | −99.88% to +2.40% | re-pinned |

539 of the 570 stored positions move. This is **the largest correction
in the project by four orders of magnitude**, and the only one that is
not a spectrum: freeze-out abundance goes as 1/⟨σv⟩, so a ⟨σv⟩ that
retained none of the true value at the closed-resonance points moved the
relic density there by two orders of magnitude. Cost, measured on this
worktree: 12 `relic_density` solves go 1.31 s → 5.4 s, and a single
`thermal_cross_section` call 4.6–48 µs → 21–136 µs. Task 2
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
| A1 boost window | 4 — **landed** | 7 of 7 predicted | 4154 | ~1.0 (shipped a median 9,768× and up to 360,507× high near threshold) | **both** — down in `rest_plus_eps`, up in the boosted blocks |
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
it does not reconstruct it. B4, B5, B6, A1 and B1 have each written their
own `CHANGELOG.md` entry already, all but B4's under `[Unreleased]` because
2.2.0 is released; Task 12 renames that heading rather than re-deriving
it, and
`preflight.sh --closing` greps for `## [<new version>]`, so it must.

## Decisions and Implementation Notes

- **Group A repairs are `Reference` relations reading the Task 2
  capture** (Task 4), not `Additive` or `Exact`: each defect is a lost or
  doubled term of a quadrature, which no transform of the stored array
  rebuilds. `test/parity/oracle_reference.py` is the single reader for
  all four.
- **Two repairs on one array declare one `Composed` relation, labelled
  `A1+B1`** (Task 5). `../rules.md` rule 7's first real case: A1 and B1
  move 18 of the same positions on `spectra.photon.eta_prime`, so neither
  disjointness nor a second key was available. `deltas.Composed` predicts
  a base relation's array with each further repair's `Additive` term on
  top, `deltas.repair_labels` splits the label back into roster entries,
  and `REPAIRS` stays the closed set of ten. Amends ADR-0001.
- **A composite covers only the arrays *every* named repair moves**
  (Task 5): two of A1's eight `eta_prime` arrays keep A1's declaration
  alone, because B1 moves nothing at their scalar probe. Naming B1 there
  would be a label wider than its mechanism.
- **A relation's budget is the case's own where the port is bit-faithful**
  (Task 4): A1 carries `tolerances.TABULATED_RTOL` = 1e-12 against a
  measured 0.0, because the capture is one platform's and the declared
  positions must not be held looser *or* tighter than the undeclared ones
  beside them.
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
- **Two tasks generalized the relation protocol concurrently, and the
  merge kept one** (Tasks 3 and 13). Task 13 shipped
  `term_for(fn, block, suffix, pinned)` returning the additive term;
  Task 3 shipped `expected(fn, block, stored)` returning the repaired
  arrays, and merged to master first. `Reference` was rewritten onto
  `expected` and `term_for` is gone. Worth knowing that the discarded
  shape existed for a reason: Task 13 reached it after an `expected`-like
  first attempt round-tripped `pinned + term` and turned the unpinnable
  positions' stored `NaN` into a `NaN` term, breaking three B4 blocks.
  Task 3's version avoids that by dropping unpinnable positions from the
  predicted array rather than from a derived term.
- **B6's budget is set from the kernel's own tolerance, not from the
  local measurement** (Task 13). The repaired kernels agree with the
  reference to 3.6e-9 here, but they are only held to `epsrel = 1.49e-8`,
  and a platform whose libm steers QUADPACK to a different accepted
  partition may land anywhere inside that. The declaration's `rtol` is
  6.7x the bound, not 28x the measurement.

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

### Task 4 (A1)

`rust/src/boost.rs` (the repair, the rewritten faithfulness notes, the
renamed coverage test, a yield invariant, a tightened budget),
`test/parity/oracle_reference.py` (new — the Group A captures as a
relation, and the reader A2/A3/A4 reuse), `test/parity/deltas.py`,
`test/parity/test_parity.py` (`EXPECTED_DECLARED_ARRAYS` 42 → 98),
`test/parity/tolerances.py`, `test/test_core_boost.py`,
`test/test_core_photon_tables.py`, `CHANGELOG.md`,
`docs/followups/todo/boost-integral-drops-last-interior-cell.md`,
`../PLAN.md`, `../references/defect-blast-radius.md`, this file, and
`task-4-boost-window.md` (new). `test/parity/data/` untouched.

### Task 5 (B1)

`rust/src/kernels/photon_tables.rs` (the doubled weight, its doc comment,
the doubled folded-constant bit pattern, the renamed weight test),
`test/test_core_photon_tables.py` (`SPECTRA["eta_prime"]`'s weight and the
renamed `TestPhysics` test), `test/parity/deltas.py` (the `Composed`
relation, `repair_labels`, `_A1_B1`, six re-pointed keys),
`test/parity/test_parity.py` (composite labels in the model shape test),
`CHANGELOG.md`,
`docs/followups/todo/eta-prime-two-photon-line-missing-factor-two.md`,
`../adrs/ADR-0001-corpus-repairs-are-declared-deltas.md`,
`../references/corpus-repinning.md` (the Relations table, which named two
relations that were never built and neither that were), this file, and
`task-5-eta-prime-line.md` (new). `EXPECTED_DECLARED_ARRAYS` stays 98 —
this task re-points keys rather than adding them. `test/parity/data/`
untouched.

### Task 13 (B6)

`rust/src/kernels/vector_xs.rs`, `rust/src/kernels/scalar_xs.rs`
(`THERMAL_EPSABS`, `THERMAL_LIMIT`, and the composite-rule test),
`hazma/relic_density/_thermal_functions.py`,
`hazma/vector_mediator/_gev/thermal_cross_section.py`,
`test/parity/thermal_reference.py` (new), `test/parity/deltas.py`,
`test/parity/test_parity.py`, `test/test_relic_density.py`,
`test/test_core_quad.py`, `CHANGELOG.md`, `docs/followups/` (the
follow-up moved to `done/`, plus a new `todo/` entry), plus this
project's `PLAN.md`, `references/defect-blast-radius.md` and these
notes — see `task-13-thermal-quadrature.md`.

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

- **Do B1's and B2's declared arrays collapse into A1's?** Answered for
  B1 (Task 5): they do, through a new `deltas.Composed` relation and the
  label `A1+B1`, and they had to — A1 and B1 move 18 of the same
  positions. Two of A1's eight `eta_prime` arrays keep A1's declaration
  alone, because B1 reaches neither scalar probe. Task 6 repeats the
  shape for `A1+B2` and re-derives which of φ's arrays B2 actually moves
  rather than assuming six.
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
- **Will Group B's relation budgets survive their repairs?** B1's did,
  and then some: the composite measured 2.1e-16 against the repaired
  kernel — one ulp — where the standalone model reserved 1e-11. It is
  declared at the case's own 1e-12 rather than tightened to what it
  measures, for A1's reason (the capture is one platform's). B2 and B3
  still carry 1e-11 and 1e-9 unmeasured; Tasks 6 and 9 measure. A repair
  needing a *wider* budget has found something Task 3 did not model
  (`rules.md` rule 2).
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

- The four Group A `.pyx` twins are gone; `test/parity/oracles/data/`
  is what stands in for them, and `test/parity/oracle_reference.py` is
  the reader that turns a capture into a `deltas.Reference`. Task 4 is
  the worked precedent for A2, A3 and A4 — they need no new machinery.
- `test/parity/data/` is intact — `python test/parity/generate.py --check`
  verifies it in under a second with no build.
- B4 (PR #87), B5 (Task 10a), B6 (Task 13), A1 (Task 4) and B1 (Task 5)
  are the repairs that have changed a library value; every other corpus
  array is still compared against its stored value.
  `test/parity/deltas.py` declares 98 arrays — 30 for B4, 6 for B5, 6 for
  B6, 50 for A1 alone and 6 for the composite `A1+B1` — and
  `test_parity.EXPECTED_DECLARED_ARRAYS` is the literal that makes a
  change to that number show up in a diff. Re-derive the split with
  `Counter(d.repair for d in deltas.DECLARED_DELTAS.values())`.
- The measurement recipe every remaining repair task needs: capture the
  corpus blocks from a build carrying the defect, restore the repair,
  rebuild, capture again, and diff *those* — not the live tree against
  the stored corpus, which reports the platform drift as if it were the
  repair (Findings). On this worktree Task 4's defective build
  reproduced the stored corpus bit for bit on all 10,045 A1 positions,
  so the two agreed; that is this platform, not a general fact.
- B2 and B3 have finished, corpus-checked delta models in
  `deltas.DELTA_MODELS`, and B3's arrays are still undeclared, so Task 9
  adds keys and bumps `EXPECTED_DECLARED_ARRAYS` in the ordinary way.
  **Task 6 cannot**: A1 already declares every array B2 moves, and one
  key holds one `Delta`, so it composes as `A1+B2` the way Task 5 did for
  B1 — `task-5-eta-prime-line.md` is the worked precedent, and the
  position counts in `task-3-closed-form-deltas.md` are to be re-derived
  against the repaired kernel rather than pasted. Task 5's held exactly;
  that is one case, not a rule.
- The delta layer now carries four relations. A new repair picks
  `Additive` when the physics names the term, `Exact` when a closed form
  transforms the stored array, `Reference` when only a second
  implementation can say what the value should be, and `Composed` when a
  second repair moves an array the first already declares; all four
  answer `expected`, and the runner needs no change for any of them.

**Currently risky / unknown:**

- The blast-radius table is derived from the composition graph, not
  measured. Treat it as where to look, never as what you will find. B5 is
  the sharpest case so far: the table's coverage arithmetic said both
  `spectra.neutrino.*` cases were untouched, and one of them was not.
- A repair's own follow-up can misstate its magnitude in either
  direction. B5's predicted 0.06% local shift measured at a factor of
  two, because the follow-up sampled only the plateau; B1's predicted
  0.63% measured at 0.603%, because a *percentage of a yield* depends on
  the integration window while the absolute shift does not. Re-derive
  every figure before quoting it in a `CHANGELOG.md` entry, and prefer
  the window-independent statement where the physics offers one.
- Every deadline in this plan depends on `cython-to-rust`'s pace, which
  this project does not control and must not assume.
