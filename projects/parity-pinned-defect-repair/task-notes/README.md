# Working Memory: Repair the parity-pinned numerical defects

**Date:** 2026-08-19 (created)
**Project:** parity-pinned-defect-repair
**Status:** In Progress
**Plan References:** `../PLAN.md` (all sections)
**Related ADRs:** none yet — two anticipated, see `../PLAN.md`
**Depends On:** none. **Constrains** `cython-to-rust` Tasks 4.6, 6.2,
6.3 and 6.4 — see Open Questions.

## Objective

Track cumulative context and live task status across all twelve tasks so
any agent picking up work mid-project has the facts, decisions and open
questions needed to start without re-discovering them.

## Tasks

Canonical task *shape* lives in `../PLAN.md` under "Task Details". This
section tracks live *status*.

| # | Task | Depends on | Status | Task Note |
|---|------|------------|--------|-----------|
| 1 | Delta-declaration layer | — | Not started | `task-1-delta-declarations.md` |
| 2 | Capture the corrected-value oracles | — | **Complete** | `task-2-cython-oracles.md` |
| 3 | Closed-form delta models (B1–B3) | 1 | Not started | `task-3-closed-form-deltas.md` |
| 4 | Repair A1 — boost integral window | 1, 2 | Not started | `task-4-boost-window.md` |
| 5 | Repair B1 — η′ line weight | 3, 4 | Not started | `task-5-eta-prime-line.md` |
| 6 | Repair B2 — φ line energies | 3, 4 | Not started | `task-6-phi-lines.md` |
| 7 | Repair A2 — muon photon endpoint | 1, 2 | Not started | `task-7-photon-muon-endpoint.md` |
| 8 | Repair A3 — charged-pion forward cone | 7 | Not started | `task-8-charged-pion-cone.md` |
| 9 | Repair B3 — rho rest-frame branch | 3, 8 | Not started | `task-9-rho-rest-frame.md` |
| 10 | Repair A4 — positron-muon normalization | 1, 2 | Not started | `task-10-positron-muon-norm.md` |
| 11 | Reconcile the superseded sequencing prose | 4–10 | Not started | `task-11-prose-reconciliation.md` |
| 12 | Close — aggregate the drift, bump | 11 | Not started | `task-12-close.md` |

```text
1 ──┬──► 3 ──┬──────────► 5 ──┐
    │        │                │
    ├──► 4 ──┴────────────────┼──► 11 ──► 12
    │        └──► 6 ──────────┤
2 ──┼──► 7 ──► 8 ──► 9 ───────┤
    └──► 10 ──────────────────┘
```

Task 2 had no upstream dependency and the project's only hard external
deadline. It is done: every Group A oracle is captured and committed
under `test/parity/oracles/`, so `cython-to-rust` Tasks 4.6, 6.2, 6.3 and
6.4 may now run in any order without stranding a repair. Nothing else in
this project is time-critical.

## Exit Criteria

- All twelve tasks complete; all seven follow-ups moved to
  `docs/followups/done/` with inbound links repointed and the revision
  pinned.
- No live document still sequences any of the seven repairs "after
  Phase 06 Task 6.4".
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

**No public value has moved yet.** Task 2 shipped no library behavior —
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

Tasks 4–10 each move a published spectrum by design, and each records the
function, the grid and the max shift here in its own PR (`../rules.md`
rule 10). Task 12 aggregates this section into the `CHANGELOG.md` entry —
it does not reconstruct it.

## Decisions and Implementation Notes

- **The corpus is extended, not regenerated.** The committed arrays stay
  as the record of what 2.1.0 shipped; each repair adds a declared delta
  against them. Rationale and schema in
  `../references/corpus-repinning.md`; candidate for a project-scoped
  ADR at Task 1 (`../PLAN.md`, Anticipated ADRs).
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

_None yet — no task started._ The change that created this project
touched only `docs/followups/todo/*.md` (seven blocker bullets, one hunk
per file), `projects/README.md` (the Active Projects row) and
`projects/parity-pinned-defect-repair/` itself. No library file, test or
build input is in that diff.

## Verification

**The scaffolding change itself** (the seven corrected blocker bullets +
this project tree; no task of this project has run yet) was gated with:

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

Every one of these needs a built tree (`uv pip install -e .`); a
non-editable install leaves no extension where the corpus insists on
measuring one.

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
- Nothing in this project has changed a library value yet.

**Currently risky / unknown:**

- The blast-radius table is derived from the composition graph, not
  measured. Treat it as where to look, never as what you will find.
- Every deadline in this plan depends on `cython-to-rust`'s pace, which
  this project does not control and must not assume.
