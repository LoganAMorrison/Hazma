# Task 7.4: Close the project

**Date:** 2026-08-29
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../PLAN.md` §"Closing this project", §"Numerical
impact", §"Scope"; `../../phases/phase-07-cutover.md` Task 7.4 and
§"Exit Criteria"; `../../rules.md` rules 1–4, 12
**Related ADRs:** ADR-0001, ADR-0002, ADR-0003 (all Accepted; none
amended here)
**Depends On:** Tasks 7.1, 7.2, 7.3

## Objective

Close the cython-to-rust project: aggregate the running numerical record
into a `CHANGELOG.md` release section, bump `[project] version` to the
declared `major` level, synthesize the Phase 07 learnings and the project
retrospective, file the retrospective's follow-on seeds, and flip the
project's status in `PLAN.md` and `projects/README.md`.

## Exit Criteria

From `../../phases/phase-07-cutover.md` Task 7.4 and §"Exit Criteria",
plus `../../PLAN.md` §"Closing this project":

- `CHANGELOG.md` entry: the migration summary plus the aggregated
  numerical-drift table from `../numerical-impact.md` (per-function max
  shifts), naming this project slug.
- `[project] version` in `pyproject.toml` bumped per `PLAN.md`
  `version_bump` — re-checked against the recorded drift and the Task 0.5
  outcome rather than inherited from the frontmatter — and
  `scripts/agents/preflight.sh --closing` green. (Task 7.1 moved the
  version's source of truth off `hazma/__init__.py`; the gate reads
  `pyproject.toml`.)
- Project retrospective at `../../learnings/project-retrospective.md`
  including §5 follow-on seeds. The three candidates the phase file names
  — constants-table consolidation as a declared numerical change,
  free-threaded `abi3t` wheels, relic-density ODEs to Rust — each get a
  `docs/followups/todo/` stub if still relevant at close, and **all** of
  `docs/followups/todo/` is cross-checked for entries this project
  sourced.
- `PLAN.md` `status: Complete`; `projects/README.md` row moved from
  Active to Completed with the Shipped date.
- Phase 07 closure (from §"Exit Criteria" of the phase file): all task
  rows Complete, phase file frontmatter `status: Complete`, phase
  learnings at `../../learnings/phase-07-cutover.md`, and — per
  `.claude/skills/execute-single-task` Step 8 — the project README's
  Phase 07 Findings / Decisions / Files Changed / Verification entries
  moved verbatim into the `history-*.md` archives.

## Inputs Reviewed

- `../../PLAN.md` — all sections; §"Closing this project" is the checklist.
- `../numerical-impact.md` — the whole 859-line log; the input to the
  CHANGELOG table, not reconstructed from memory.
- `../../phases/phase-07-cutover.md` — Prerequisites, Task 7.4, §"Exit Criteria".
- `README.md` (this phase's working memory) — Tasks table, Findings, Handoff.
- `../README.md` (project working memory) — Phases table, Handoff, Open Questions.
- `../../learnings/phase-0{0..6}-*.md` — §1 and §5 of each, for the retrospective.
- `../../rules.md` — rules 1 (reproduce defects), 2 (no corpus
  regeneration), 3 (declare drift above 1e-12), 4 (constants bit-parity),
  12 (measure performance).
- `docs/versioning.md`, `docs/followups/README.md` + `todo/`,
  `projects/README.md`, `CHANGELOG.md` `[Unreleased]`.
- `scripts/agents/preflight.sh` gate 10 (the `--closing` version gate).
- `task-7.2-release-pipeline.md` §Decisions — the recorded aarch64 /
  Windows call.

## Findings

- **The `[Unreleased]` section is Phase 00's, and it is the release
  section.** Its `Added` / `Removed` / `Changed` blocks were written by
  Tasks 0.2, 0.3 and 0.5 and were explicitly left as "the settled wording
  for the Phase 07 aggregate" (`../numerical-impact.md`, Task 0.2). So
  closing is a promotion of that heading plus the Phase 02–06 material,
  not a fresh section written beside it.
- **The declared `major` survives the re-check, and nothing numerical
  drives it.** The largest drift in the whole port is
  `scalar_mediator_decay_spectrum` at **5.3327e-12** relative — a
  `patch`-level number under `docs/versioning.md`. `major` is carried
  entirely by Phase 00's two API removals (`hazma.gamma_ray`,
  `hazma.deprecated.rambo`), exactly as `PLAN.md` §"Numerical impact"
  predicted. The one genuinely user-visible numerical change is Task
  0.3's threshold repair (1.3e-4 within 1e-10 of threshold, plus three
  edge behaviors), and it too is already written up in `[Unreleased]`.
- **Exactly seven entry points moved past `rules.md` rule 3's 1e-12
  threshold, and all seven are mediator spectra that moved for one
  reason.** Task 6.2's three and Task 6.3's four are `crate::quad`
  against scipy's QUADPACK. The next largest is Task 4.5's
  `dnde_photon_charged_rho` at 1.5e-13 — the same cause, below the
  threshold — and every remaining entry point is bit-equal or moves by
  ≤5.5e-15. So the CHANGELOG table wants two honest columns, worst
  relative shift and cause, rather than one row per function repeating
  the same sentence eleven times.
- **The project ends with fourteen tolerance budgets tightened and none
  widened**, and neither opening budget (`QUAD_RTOL` 1e-8,
  `NESTED_RTOL` 1e-6) has a holder: `QUAD_RTOL` emptied at Task 5.2,
  `NESTED_RTOL` at Task 6.3. That is the strongest single statement the
  closing entry can make about the port's fidelity, and it is checkable
  from `test/parity/tolerances.py` rather than from prose.
- **Twelve live 2.1.0 defects were surfaced by the port and reproduced
  under rule 1**, and the count has to be derived rather than read off:
  the log's running ordinals drift (Task 5.3 calls the unconverged
  thermal quadrature "the eleventh" where Task 5.1 had already filed it
  as the ninth). Counting the filed follow-ups instead gives one from
  Task 3.4, seven from Phase 04, three from Phase 05 and one from Task
  6.3. They are the closing entry's most valuable content for
  a user: none is introduced here, but a user reading only "we rewrote
  the compiled layer" would not learn that `thermal_cross_section` has
  been 0.5%–5% wrong across freeze-out the whole time. `Fixed` is the
  wrong section for them — nothing was fixed — so they go under a named
  `Known issues` block that points at `docs/followups/todo/`.
- **The aarch64 / Windows question needs a stub after all**, and this is
  a change of circumstance rather than a re-litigation of Task 7.2's
  call. Task 7.2 declined to file one because "`PLAN.md`'s Scope already
  records it as a cheap follow-up" — true while `PLAN.md` was a live
  document. Closing it makes that record archival, and
  `docs/followups/todo/` is the repo's live backlog by construction
  (`docs/followups/README.md`: "`ls todo/` is the live backlog at a
  glance"). The decision itself is unchanged and the stub records it as
  a deliberate no.

## Decisions and Implementation Notes

- **Promote `[Unreleased]` to `## [3.0.0] — 2026-08-29` rather than open
  a new section.** Phase 00's blocks are already the settled wording and
  every later phase's material is additive to them. A second section
  would split one release across two headings and break the
  `preflight.sh --closing` `grep "[${NEW_VER}]"` contract's intent.
- **One drift table, keyed by entry point, with the cause column
  carrying the explanation.** Per-function prose for 41 entry points
  would be unreadable and 27 of the rows would say "bit-equal". The
  table lists every entry point that moved at all and states the
  bit-equal remainder as a count.
- **`Known issues` is a new CHANGELOG section name**, outside Keep a
  Changelog's six. The file's own header enumerates the six, so the
  header is amended in the same edit rather than the section being
  smuggled in. The alternative — filing twelve reproduced defects under
  `Fixed` — would be false, and under `Changed` would imply the port
  moved them.
- **Three seeds filed, and a fourth for the platform matrix.** The
  constants consolidation, the free-threaded wheels and the relic-density
  ODE port are the three the phase file names; each is still live at
  close and each gets a `todo/` stub. The aarch64 / Windows stub is the
  fourth, for the reason in Findings.
- **Review round 1 (blocking finding 1): the release criterion is
  revised, not qualified.** The first version of this task annotated
  §"Exit Criteria" with "Met … with one clause qualified" while the same
  paragraph documented the clause as unmet, and `../README.md` said "All
  met" — a contradiction the reviewer was right to block on. The
  reviewer offered two remedies; **the first is unavailable.** "Keep
  closure pending until the release publish is observed" deadlocks:
  `publish` is gated on `github.event_name == 'release'`, a release needs
  the `3.0.0` tag, and that tag exists only after this closing PR merges,
  so closure would gate on an event that closure itself enables. Taking
  the second remedy — formally revise the criterion — is also what
  `execute-single-task` Step 7 prescribes for a gate sentence that is
  factually wrong. The revision narrows the clause to what closure can
  attest and reassigns the upload rather than dropping it; because it is
  a narrowing, the residual risk is stated in the phase file rather than
  buried: trusted publishing under `maturin-action` has never executed.
- **Review round 1 (blocking finding 2): the file-set counts are
  re-derived, not patched.** The reviewer cited one stale count ("19
  files: 12 modified, 7 created"). It was correct when written and went
  stale when `/commit-and-pr` added three `.claude/skills/` edits at the
  commit boundary — which means every *sibling* file-set count went stale
  the same way, not just the cited one, and this round's own fixes then
  moved them again. Rather than chase them, every count was re-derived
  from `git diff` once after all content edits were frozen
  (`doc-consistency.md` §11 rule 1, "sweep last"): **24 files, 17
  modified and 7 created**; 23 `.md` in the diff; 20 of those outside
  `.claude/skills/`. Two distinct quantities were being conflated — the
  `.md` count in the diff and the count of documents this task authored —
  so both are now named explicitly wherever they appear.
- **`hazma/experimental/axial_vector_mediator/__init__.py` stays
  broken.** Phase 00's learnings flagged it and deliberately filed
  nothing, because `experimental/` is not a public surface
  (`docs/versioning.md`). Closing the project does not change that, and
  fixing it here would be scope creep into a tree the lint gate excludes.

## Files Changed

- `CHANGELOG.md` — `[Unreleased]` promoted to `[3.0.0]`; the migration
  summary, the aggregated drift table and the `Known issues` block added;
  the section list in the file header amended for `Known issues`.
- `pyproject.toml` — `[project] version` `2.1.0` → `3.0.0`.
- `projects/cython-to-rust/PLAN.md` — `status: Complete`; Phase 07 row
  filled in.
- `projects/README.md` — `cython-to-rust` row moved Active → Completed.
- `projects/cython-to-rust/phases/phase-07-cutover.md` — frontmatter
  `status: Complete`, and §"Exit Criteria" revised: the release clause
  narrowed to build-and-test plus an observed release gate, with
  §"Revision of the release clause" recording why and what risk that
  leaves (review round 1).
- `docs/agents/lessons.md` and `docs/agents/lessons-examples.md` — the
  `[unrun-workflow-cannot-close-a-criterion]` class extended to cover a
  structurally unsatisfiable criterion, cited to this PR (review round 1).
- `projects/cython-to-rust/learnings/phase-07-cutover.md` — new.
- `projects/cython-to-rust/learnings/project-retrospective.md` — new.
- `projects/cython-to-rust/task-notes/README.md` — Phases table, Status,
  Handoff, Decisions, Open Questions; Phase 07 sections archived out.
- `projects/cython-to-rust/task-notes/phase-07/README.md` — Task 7.4 row,
  Status, Findings, Handoff.
- `projects/cython-to-rust/task-notes/history-findings.md` and
  `history-decisions.md` — Phase 07's one cross-phase entry each, moved
  in verbatim under a `## Phase 07 (moved 2026-08-29 at project close)`
  heading. `history-files-changed.md` and `history-verification.md` are
  untouched: the project README recorded no cross-phase entry in either
  section for this phase.
- `projects/cython-to-rust/task-notes/numerical-impact.md` — Task 7.4
  entry (no public value changes) and the closing pointer.
- Four new follow-up stubs under `docs/followups/todo/`:
  `consolidate-the-two-constants-tables.md`,
  `free-threaded-abi3t-wheels.md`, `relic-density-odes-in-rust.md`,
  `wheels-for-aarch64-and-windows.md`.
- `docs/followups/README.md` — four rows under Open.
- `docs/versioning.md` — the illustrative `[project] version` snippet,
  which still quoted `2.1.0`.
- `.claude/skills/{commit-and-pr,execute-single-task,task-pipeline}/SKILL.md`
  — the closing-PR instructions still told the next agent to bump
  `VERSION` in `hazma/__init__.py`. Repointed at `pyproject.toml`'s
  `[project] version`.
- `projects/cython-to-rust/task-notes/phase-07/task-7.4-close.md` — this note.

## Verification

- **`scripts/agents/preflight.sh --closing --md "<the 23 changed .md>"`**
  — the full gate, with every markdown file in the diff passed to
  `--md`. Rows:

  | Gate | Result |
  | --- | --- |
  | `black --check` | PASS (`hazma test`) |
  | `isort --check-only` | **FAIL** — pre-existing, see below |
  | `ruff check` | **FAIL** — pre-existing, see below |
  | `cargo fmt --check` | PASS |
  | `cargo clippy` | PASS |
  | `cargo test` | PASS |
  | `pytest` | PASS — `2231 passed, 15 skipped, 12 subtests passed` |
  | `import hazma` | PASS |
  | `markdownlint` | **FAIL** — pre-existing, see below |
  | `version bump` | PASS — `2.1.0 → 3.0.0 + CHANGELOG entry` |
  | `forbidden tokens` | PASS — none added |

- **All three FAIL rows are pre-existing, and none is this task's.**
  For isort and ruff the diff contains **zero** `.py` files, so both
  read bytes identical to `origin/master`.

  ```sh
  git diff origin/master --name-only -- '*.py' | wc -l   # -> 0
  ```

  Measured anyway on this tree: `isort --check-only hazma test` reports
  **72** ERROR lines and `ruff check hazma test` **6091** errors, which
  is the condition
  [`docs/followups/todo/preflight-isort-ruff-red-on-trunk.md`](../../../../docs/followups/todo/preflight-isort-ruff-red-on-trunk.md)
  exists for.

  markdownlint is red on exactly one file, `.claude/skills/task-pipeline/SKILL.md`,
  which this task edited by two lines that touch no fence. Its lint state
  is **unchanged**: 7 errors on this tree and 7 on `origin/master`, the
  same five rules with the same contexts (4 MD031 on blockquoted fences,
  1 MD032, 2 MD036). That is the condition
  [`markdownlint-skips-skill-file-shapes`](../../../../docs/followups/todo/markdownlint-skips-skill-file-shapes.md)
  tracks — markdownlint was never run over `.claude/skills/`, so the
  skill trees have never been clean. **All 20 documents this task
  authored or rewrote lint clean** on their own:

  ```sh
  markdownlint --dot <the 20 non-skill .md in the diff>   # exit 0
  ```

  No other gate is red.

- **`pytest -q` covers the numerical gates this task must not disturb:**
  `test/parity` at its declared budgets on the capturing platform, and
  `test/test_theory_aggregation.py`'s model-layer identities. Both are
  inside the 2231.

- **`cargo test --manifest-path rust/Cargo.toml --no-default-features`**
  — `test result: ok. 258 passed; 0 failed` on the lib target, plus 0
  doc-tests. **The 249 the project working memory carried was two tasks
  stale**; both learnings files now say 258 and say to re-derive it.

- **The version resolves after a rebuild.** `uv pip install -e .` then
  `python -c "import hazma; print(hazma.VERSION, hazma.__version__)"` →
  `3.0.0 3.0.0`. Worth running explicitly: `hazma.VERSION` reads the
  *installed* metadata, so preflight's import-smoke row reported
  `version 2.1.0` from a tree whose `pyproject.toml` already said 3.0.0.

- **The drift table was derived, not transcribed.** The 14-moved /
  27-bit-equal split is read out of `test/parity/tolerances.py` — 11
  budgets whose rationale says "Tightened from" plus 3 `_pt` twins saying
  "Tightened with its twin", against 41 cases — and cross-checked against
  every per-task entry in `../numerical-impact.md`. The two sources agree
  on all 14 entry points. `QUAD_RTOL` and `NESTED_RTOL` were confirmed to
  have zero holders by the same script.

- **Deferred, with reason:** the `publish` job of `release.yml` is still
  unexercised. It is gated on `github.event_name == 'release'` and cannot
  run before an actual release is cut, so this task cannot close it; the
  phase README's Handoff names it as the one thing a release manager
  still owes. Sphinx was not re-run — no page in this diff is under
  `docs/source/`.

## Numerical impact

**No public value changes**, and the diff proves it rather than a sweep
doing so: one line of `pyproject.toml` (the version) and markdown.

```sh
git diff origin/master --name-only -- '*.py' '*.rs' '*.pyx' '*.pxd' \
    '*.csv' '*.dat' '*.npy' '*.toml'
# -> pyproject.toml    (single hunk: version = "2.1.0" -> "3.0.0")
```

No code path, constant, table or signature is reachable, so no grid
evaluation applies. Recorded as the closing entry in
[`../numerical-impact.md`](../numerical-impact.md), which also states the
one user-visible non-numerical move (`hazma.VERSION` → `3.0.0`) and the
aggregate the CHANGELOG carries.

**Version level re-checked, not inherited.** `PLAN.md`'s
`version_bump: major` holds: the largest drift anywhere in the port is
5.3327e-12 relative, `patch`-level under `docs/versioning.md`, and
`major` rests on Phase 00's removal of `hazma.gamma_ray` and
`hazma.deprecated.rambo` — the latter `major` by the
`hazma/deprecated/` rule specifically.

## Open Questions

None. The project is closed and every question the phase READMEs carried
is either answered in place or filed as a follow-up.

Two things a reader should know are *unresolved by design* rather than
open:

- **`release.yml`'s `publish` job has never executed.** Three dispatches
  and one `pull_request` run all skipped it correctly, which is what the
  `github.event_name == 'release'` gate is for. Cutting 3.0.0 is the
  first execution — watch it (`docs/agents/lessons.md`
  `[unrun-workflow-cannot-close-a-criterion]`).
- **Twelve reproduced 2.1.0 defects ship in 3.0.0.** That is
  `rules.md` rules 1 and 2 working as intended, not an oversight; the
  CHANGELOG's `Known issues` section says so to users and
  `projects/parity-pinned-defect-repair/` sequences the repairs.

## Plan Impact

**Impact Level:** Phase file patched (plus the closure edits `PLAN.md`
mandates).

- `../../PLAN.md`: `status:` → `Complete`; the Phase 07 row filled in;
  the §Scope bullet on Windows / linux-aarch64 repointed at the follow-up
  Task 7.4 filed, because closing the plan makes a Scope bullet an
  archival record rather than a live one.
- `../../phases/phase-07-cutover.md`: frontmatter `status:` → `Complete`;
  §"Exit Criteria" **revised**, not merely annotated. Its release clause
  read "a release candidate builds, tests, and *publishes* from CI",
  which no closing PR can satisfy — see §"Revision of the release clause"
  in that file, and the review-round bullet in §Decisions above. The
  clause now asks for build-and-test plus an observed release gate; the
  upload is reassigned to the release manager with its residual risk
  stated. `../README.md`'s Exit Criteria copy was revised in the same
  pass.
- No ADR. Nothing about architecture, invariants, interfaces, task
  ordering, units or normalization changed; all three project ADRs were
  Accepted and none needed amending. The canonical-contract diff was run
  over `PLAN.md`, all eight phase files and the three ADRs; the two
  `PLAN.md` clauses above were the only sentences that had become
  factually wrong.

## Stale-state sweep

Run against `claude/cython-to-rust/task-7.4-close` after every prose edit
was frozen, then re-run once and reproduced. Hand-folded where noted;
everything else is pasted output.

### Identifier and status sweep

```sh
rg -n -i 'cython-to-rust.*(in progress|active)|(in progress|active).*cython-to-rust' \
   --glob '!projects/cython-to-rust/task-notes/history-*' .
```

Six hits, all **KEPT**. Three are inside closed Phase 01/06 task notes
recording *their own* sweeps at the time; the other three are in this
note — §Files Changed describing the move, this block quoting the
pattern, and this block's own prose about the other project
(doc-consistency rule 3: the citing doc counts as a match). No live
status claim survives.

```sh
rg -n '(Task [0-9.]+ will|will be added|still pending|In Progress|Not started)' \
   $(git diff origin/master --name-only | grep '\.md$')
projects/README.md:43: ... parity-pinned-defect-repair ... | In Progress |
task-7.4-close.md: <this block, twice — the quoted pattern and the
                    quoted hit above; line numbers omitted because each
                    edit to this block moves its own citation>
```

Three hits, all **KEPT**. The real one is the other project, which
genuinely is in progress; the `cython-to-rust` row moved to the Completed
table. The other two are this block quoting itself.

### Link and index sweep

Every relative markdown link in all 23 changed/created `.md` files
resolves (script over `git diff origin/master --name-only`):
`checked 23 markdown files; all relative links resolve`.

`docs/followups/README.md`'s Open table against `docs/followups/todo/`:
`listed but missing: none / on disk but unlisted: none / counts: listed
27 on disk 27`.

### Line-number citation sweep

```sh
python scripts/agents/check_doc_citations.py --changed-vs origin/master <23 docs>
docs scanned: 23
in-repo citations checked: 5
  resolved by exact: 4
  resolved by suffix: 1
external citations skipped: 16
out-of-range or ambiguous: NONE
```

The 16 skipped are `.pyx` / `.pxd` paths inside archived history text —
the condition
[`citation-checker-skips-deleted-inrepo-files`](../../../../docs/followups/todo/citation-checker-skips-deleted-inrepo-files.md)
describes. The checker only reads `.py:<line>`, so the
`rust/src/kernels/mediator_tables.rs:320` citation this task introduced
was bounds-checked by hand: that line is
`slot: Mutex<Option<(u64, Arc<T>)>>`. Its `hazma/parameters.py:205`
citation *is* covered, and is one of the four resolved above —
`alpha_em: float = 1.0 / 137.04`.

### Count sweep

| Claim location | Command | Actual | Status |
| --- | --- | --- | --- |
| CHANGELOG "27 … bit-for-bit … other 14" | `tolerances.BUDGETS`: count `"Tightened from"` + `"Tightened with its twin"` | 11 + 3 = **14** moved, 41 − 14 = **27** | OK |
| CHANGELOG "neither … opening budget … claimed" | count holders of `QUAD_RTOL`, `NESTED_RTOL` | **0** and **0** | OK |
| CHANGELOG "all 16 mediator cross sections other than the two ⟨σv⟩" | untouched-budget roster, `cross_sections.*` rows | 11 scalar + 5 vector = **16** | OK (an earlier draft said 17) |
| CHANGELOG drift table, 11 rows | each figure against `../numerical-impact.md` | all 11 match to the quoted precision | OK |
| CHANGELOG "at most 5.4e-12" | max of the table | 5.3327e-12 | OK |
| CHANGELOG "twelve defects" | `docs/followups/todo/` entries sourced by this project | 1 (T3.4) + 7 (P04) + 3 (P05) + 1 (T6.3) = **12** | OK (the log's own ordinals disagree; see §Findings) |
| Retrospective/Phase-07 "`cargo test` … 258" | `cargo test --manifest-path rust/Cargo.toml --no-default-features` | `ok. 258 passed; 0 failed` | OK (was 249 in working memory — **EDITED** in both files) |
| Retrospective/Phase-07 "`pytest -q` 2231/15/12" | preflight's pytest row | `2231 passed, 15 skipped, 12 subtests passed` | OK |
| Constants stub "twelve names disagree" | `the_two_tables_disagree_where_the_cython_says_they_do` | **12** entries | OK |
| Constants stub "two α values differ by 2.6e-4" | `(1/137 − 1/137.035999084) / (1/137.035999084)` | 2.627e-4 | OK |
| `docs/versioning.md` version snippet | `grep '^version' pyproject.toml` | `3.0.0` | **EDITED** — snippet still said 2.1.0 |
| Skills naming the version's home | `rg -n 'hazma/__init__\.py' .claude/skills/ .codex/skills/` | 3 hits before, **0** after | **EDITED** — see below |
| `task-pipeline/SKILL.md` lint delta | `markdownlint --dot` on this tree vs `git show origin/master:` copy | **7 errors both sides**, same rules and contexts | KEPT — pre-existing |

### Numerical-impact statement

**No public value changes (verified:
`git diff origin/master --name-only -- '*.py' '*.rs' '*.pyx' '*.pxd'
'*.csv' '*.dat' '*.npy' '*.toml'` → `pyproject.toml` only, whose single
hunk is `version = "2.1.0"` → `"3.0.0"`).** No code path, constant,
table or signature is reachable from this diff, so no grid evaluation
applies. `pytest -q` is `2231 passed, 15 skipped, 12 subtests passed`,
which includes `test/parity` at its declared budgets and
`test/test_theory_aggregation.py`. Logged as the closing entry in
[`../numerical-impact.md`](../numerical-impact.md).

### Exit Criteria → artifact mapping

| Criterion | Artifact |
| --- | --- |
| CHANGELOG entry with the aggregated drift table, naming the slug | `CHANGELOG.md` §`[3.0.0]` — lede names `cython-to-rust` and links its `PLAN.md`; the 11-row table under `Changed`; `preflight.sh --closing` asserts the `## [3.0.0]` section exists |
| Version bumped per `version_bump`, level re-checked | `pyproject.toml:23` `3.0.0`; re-check recorded in §Numerical impact; `preflight.sh --closing` row `version bump  2.1.0 → 3.0.0 + CHANGELOG entry` |
| `preflight.sh --closing` green | all rows PASS except the two documented trunk reds (below) |
| Retrospective incl. §5 seeds, three named candidates | `../../learnings/project-retrospective.md` §5 — all three filed, plus a fourth; `docs/followups/todo/` cross-checked whole (27 = 27 above) |
| `PLAN.md status: Complete` | `head -2 projects/cython-to-rust/PLAN.md` → `status: Complete` |
| `projects/README.md` row moved with Shipped date | Completed table row, `2026-08-29 (hazma 3.0.0)` |
| Phase 07 closed: rows, frontmatter, learnings, history sweep | phase README's four rows Complete; `phases/phase-07-cutover.md` frontmatter `status: Complete`; `../../learnings/phase-07-cutover.md`; the two Phase 07 cross-phase entries appended verbatim to `../history-{findings,decisions}.md` under a `## Phase 07 (moved 2026-08-29 at project close)` heading |

### Preflight

```text
PASS   black --check           hazma test
FAIL   isort --check-only      run `isort hazma test` and re-check
FAIL   ruff check              see output below
PASS   cargo fmt --check       rust/
PASS   cargo clippy            rust/
PASS   cargo test              rust/
PASS   pytest                  2231 passed, 15 skipped, 12 subtests passed
PASS   import hazma            version 3.0.0
FAIL   markdownlint            (task-pipeline/SKILL.md only; unchanged)
PASS   version bump            2.1.0 → 3.0.0 + CHANGELOG entry
PASS   forbidden tokens        none added
```

**The two FAIL rows are the trunk's, not this task's**, and the proof is
stronger than a stash-and-rerun: `git diff origin/master --name-only --
'*.py' | wc -l` is **0**, so both linters read bytes identical to
`origin/master`. On this tree they report 72 isort ERROR lines and 6091
ruff errors — the standing condition
[`preflight-isort-ruff-red-on-trunk`](../../../../docs/followups/todo/preflight-isort-ruff-red-on-trunk.md)
tracks. A first run also had markdownlint red on one MD012 in
`../numerical-impact.md` (a doubled trailing blank line this task
introduced); fixed, and green above.

### Out-of-scope edit taken deliberately

Three skill files told a closing PR to bump `VERSION` in
`hazma/__init__.py` — a file that has not held the version since Task
7.1 moved it to `pyproject.toml`. They survived Task 7.3's sweep for the
same reason the three sites Task 7.2 found did: they name the *old
mechanism* rather than the word the grep looked for. This PR is the
first closing PR since the cutover, so it is the first thing to falsify
them, and leaving them means the next project close follows a wrong
instruction. Three one-line repoints, taken here rather than deferred;
`rg -n 'hazma/__init__\.py' .claude/skills/ .codex/skills/` now returns
nothing.

### Task-note self-consistency

`**Status:** Complete` in the header, matching the phase README's Task
7.4 row and this note's §Verification. Every file named in §Files Changed
appears in `git diff origin/master --stat` (**24 files: 17 modified, 7
created**), and every file in that diff is named in §Files Changed —
re-checked mechanically after the last content edit. Review round 1 cited
this line reading "19 files: 12 modified, 7 created", which was true when
written and went stale when `/commit-and-pr` added three
`.claude/skills/` edits at the commit boundary; the round's own fixes
then added two more files. **Every file-set count in this note is
re-derived from `git diff` at the frozen tree rather than patched at the
one line review cited** — the `--md` argument, the link sweep, the
citation sweep and the lint count all moved with it. The two remaining
`18`s were replaced by 20: that is the *non-skill* document count, a
different number from the 23 `.md` in the diff. Three
figures written before measurement were corrected in place rather than
left to stand: "eleven defects" → twelve, "33 rows would say bit-equal"
→ 27, and "seven of the eight entry points past 1e-12" → exactly seven.

## Handoff to Next Task

**There is no next task.** The cython-to-rust project is Complete —
eight phases, 33 tasks, 2026-08-03 to 2026-08-29, shipped as hazma
3.0.0.

Read [`../../learnings/project-retrospective.md`](../../learnings/project-retrospective.md)
first; it replaces this note and the other 32 for every later reader
([ADR-0002](../../../../docs/adrs/ADR-0002-read-phase-learnings-not-closed-task-notes.md)).
[`../README.md`](../README.md)'s Handoff routes the common follow-on
questions, and
[`../../learnings/phase-07-cutover.md`](../../learnings/phase-07-cutover.md)
carries the packaging contract.

**Safe to assume:**

- `[project] version` is `3.0.0` and `CHANGELOG.md` has a matching
  `## [3.0.0] — 2026-08-29` section. `preflight.sh --closing` is green on
  both.
- `projects/README.md` lists cython-to-rust under **Completed**; the only
  Active project is `parity-pinned-defect-repair`.
- Four follow-on seeds are filed in `docs/followups/todo/` with index
  rows: constants consolidation, free-threaded wheels, relic-density
  ODEs, aarch64/Windows wheels.

**Still risky:**

- **Cutting the release is not this task's, and it is not automatic.**
  Nothing here tags or publishes. The first `publish` execution is
  unobserved (above).
- **`rules.md` rule 4 expires with the project but the divergence does
  not.** The two constants tables are still split in
  `rust/src/constants.rs` and a cargo test still asserts it. Anyone
  consolidating them is making a declared numerical change and should
  reuse `parity-pinned-defect-repair`'s declared-delta mechanism rather
  than regenerating the corpus.
