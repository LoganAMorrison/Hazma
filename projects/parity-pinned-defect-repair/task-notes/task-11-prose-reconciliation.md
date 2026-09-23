# Task 11: Reconcile the superseded sequencing prose

**Date:** 2026-09-22
**Project:** parity-pinned-defect-repair
**Status:** Complete
**Plan References:** `../PLAN.md`, Task 11, "The premise this project
corrects" and "Dependencies"; `../rules.md`
**Related ADRs:** ADR-0001 (project-scoped), cited as the decision that
replaced the regeneration
**Depends On:** Tasks 4–10a and 13

## Objective

Leave no live document still telling a reader to wait for
`cython-to-rust` Task 6.4, and retire the twin-liveness claims that went
false when Task 6.4 deleted the last Cython.

## Exit Criteria

- [x] No live document sequences a repair "after Phase 06 Task 6.4" or
  proposes a corpus regeneration as the repair route. Population
  re-derived at execution time, sweeping the behavior words as well as
  the task id (`[settling-a-deferral-has-two-sweeps]`).
- [x] The known live copy in
  `projects/cython-to-rust/phases/phase-03-numerics-foundation.md` is
  corrected through that project's change control.
- [x] The twin-liveness claims in `../references/defect-blast-radius.md`
  (`(live)` roster cells, the Group A lead-in) and
  `../references/corpus-repinning.md` (`already drives
  hazma._utils.boost`) are corrected, sweeping on `still live`, `(live)`
  and `still supplies`.
- [x] The seven follow-ups' "Risks" sections no longer propose "one
  declared regeneration after Phase 06 Task 6.4".
- [x] `scripts/agents/check_doc_citations.py` run over every touched doc,
  and the sweep commands pasted after the last prose edit
  (`[gate-green-is-not-citations-green]`,
  `[sweep-block-written-from-intent]`).

## Inputs Reviewed

- `../PLAN.md` (Task 11, the premise, Scope, Dependencies, Task 12).
- `README.md` (this directory) and `../rules.md`.
- All three `../references/*.md`.
- `docs/agents/lessons.md`: the four classes the Task 11 spec names, plus
  `[touched-doc-inherits-its-citations]`.
- The eight roster follow-ups under `docs/followups/todo/`.
- `projects/cython-to-rust/phases/phase-0{3,4,6}-*.md`, its
  `rules.md` Process section, and its `references/*.md`.
- `git log` for the deletion commits of every Group A twin.

## Findings

- **The population was wider than the plan's one known copy and
  narrower than the raw grep.** `grep -rn "Task 6\.4" --include="*.md"`
  hit 54 tracked files before this note existed; the note itself makes
  it 55. Triage by file role
  (`[sweep-excluded-the-canonical-directory]`) leaves 102 live `.md`
  files, and the behavior and liveness sweeps over those led to edits in
  12 of them. Every other hit is a task note, a learnings file, a `done/`
  follow-up, or a past-tense statement that is still true.
- **Three of the seven follow-ups had already been rewritten by their own
  repairs** (A2, A3, B3: the photon-muon, charged-pion and rho files).
  Only A1, A4, B1 and B2 still carried the stale sequencing, plus B5's
  future-tense "when Phase 06 ports them".
- **The stale text was not confined to "Risks".** Each of A1, A4, B1 and
  B2 also had a "What" step asking for a corpus regeneration, and each
  called its siblings blocked by "the same blocker"; A1's and A4's
  sibling lines added "one declared corpus regeneration after Phase 06
  Task 6.4". All of them were swept.
- **`the-premise.md` re-derives cleanly.** Claims 1 and 5 are history,
  claim 4 still holds, and claim 6 is confirmed: Task 2 (`1a304d02`,
  2026-08-19) beat every wave, the last of which was `f479b231`
  (2026-08-27).
- **`../PLAN.md` Task 12's gate miscounted the follow-ups.** It read "the
  original seven less B5's and B6's, which are already in `done/`". B5's
  follow-up is still in `todo/` by decision (`README.md`, Decisions), and
  neither B5 nor B6 was among the original seven. There are eight to
  move.

## Decisions and Implementation Notes

- **The phase-03 correction is clerical, not canonical.** It sits in a
  Findings bullet of a Complete phase, and no exit criterion depends on
  it. It was corrected in place with an `**Amended by ...**` note quoting
  the old text, which is the shape `phase-06-mediator-spectra.md` and
  `phase-07-cutover.md` already use. No ADR was written.
  `projects/cython-to-rust/rules.md`'s Process section defines no further
  change control for a closed project.
- **The references stay history where they say so.** `the-premise.md` is
  a snapshot at `3e01590`, so it gained a dated re-derivation section
  rather than rewritten claims. `defect-blast-radius.md`'s deletion
  schedule gained a status paragraph and past tense, not a rewrite.
- **Deleted-source citations in touched follow-ups are pinned to a
  revision** (`[touched-doc-inherits-its-citations]`): `f479b231^` for
  the Task 6.4 files and `e2698eb6^` for `_neutrino/_muon.pyx`. The Why
  sections of the η′, φ and neutrino-pion files were already pinned
  (`665aed5`, `ed1fa20`), so they were left alone.
- **`projects/cython-to-rust/references/*` `.pyx` inventory rows stay
  unpinned.** Their subject is the Cython that was replaced, and pinning
  every row is Task 12's move-and-pin sweep or nobody's. What a row
  claims about the present is corrected, though: every Status cell of
  `numerics-replacements.md`'s `scipy.integrate.quad` table now names
  the task, kernel and commit that ported its call site, and the commit
  that deleted the `.pyx`.
- **Review fixes (PR #101, round 1).** The first pass edited only the
  photon-pion row of that table; the five sibling rows still read
  `Cython — Task 4.6`, `Cython — Phase 05` and `Cython — Phase 06`, a
  status vocabulary the liveness sweep did not include. The widened
  status sweep below found those five and one more stale claim of the
  same kind in `test/parity/tolerances.py`'s `QUAD` docstring ("Phase 06
  is where the next one arrives"), and all six were corrected. The
  count-sweep table's `repaired` command was wrong (it printed 397, not
  10) and is replaced, and `corpus-repinning.md`'s "Since … so" sentence
  is rewritten. Both classes are recorded in `docs/agents/lessons.md`.
- **B5's open question stays open.** The Rust port of the mediator
  positron modules (`mediator_decay_positron.rs`) has not been checked
  for a doubled `π → e ν` line. The follow-up now says so in the present
  tense rather than deferring it to a phase that has run.

## Files Changed

- `docs/followups/todo/boost-integral-drops-last-interior-cell.md`,
  `eta-prime-two-photon-line-missing-factor-two.md`,
  `phi-photon-lines-use-the-daughter-meson-energy.md`,
  `positron-muon-spectrum-normalization-inverted.md` — "Triggers /
  blockers" rewritten to the settled state, "What" regeneration steps
  replaced by the declared delta, sibling and "Risks" lines corrected,
  deleted-source citations pinned.
- `docs/followups/todo/neutrino-pion-electron-line-counted-twice.md` —
  the mediator positron re-check is stated against the Rust port.
- `projects/cython-to-rust/phases/phase-03-numerics-foundation.md` —
  the boost bullet corrected, with an "Amended by" note.
- `projects/cython-to-rust/phases/phase-04-spectra-kernels.md` — the
  capi-survivor exception put in past tense.
- `projects/cython-to-rust/references/numerics-replacements.md` — every
  Status cell of the `scipy.integrate.quad` table marked ported, with the
  kernel and the landing and deletion commits.
- `../PLAN.md` — a one-sentence status on the premise's deadline, a
  "Discharged" line under Dependencies, and Task 12's follow-up count.
- `../references/defect-blast-radius.md` — Group A lead-in, four roster
  Twin cells, all ten Serving-kernel cells marked repaired, and the
  deletion schedule's tense.
- `../references/corpus-repinning.md` — the Task 2 protocol's harness
  sentence and the defect count (nine to ten).
- `../references/the-premise.md` — "Re-derived after the port finished".
- `test/parity/test_oracles.py` — one module-docstring verb, "deletes" to
  "deleted".
- `test/parity/tolerances.py` — the `QUAD` budget-class docstring no
  longer says Phase 06 will bring an unported case.
- `docs/agents/lessons.md` and `docs/agents/lessons-examples.md` — PR
  #101 added to `[settling-a-deferral-has-two-sweeps]` and
  `[sweep-block-written-from-intent]`, with worked examples.
- `README.md` (this directory) and this note.

## Numerical impact

No public value changes (verified:
`git diff --stat origin/master -- hazma rust test/parity/data test/parity/oracles`
prints nothing). The only non-Markdown edits are docstring prose in two
test-suite modules, `test/parity/test_oracles.py` and
`test/parity/tolerances.py`; no constant or test changed.

## Verification

- `scripts/agents/check_doc_citations.py` over the 16 touched `.md`
  files, paths passed explicitly: see the sweep block.
- `scripts/agents/preflight.sh --paths "test/parity/test_oracles.py
  test/parity/tolerances.py" --md "<the 16 touched .md files>"`: see the
  sweep block for the result row.
- Nothing is deferred except B5's open mediator re-check, which is
  recorded in its follow-up.

## Open Questions

- None new. The mediator positron re-check lives in
  [`neutrino-pion-electron-line-counted-twice.md`](../../../docs/followups/todo/neutrino-pion-electron-line-counted-twice.md).

## Plan Impact

**Impact Level:** Task note only, plus a clerical patch to `../PLAN.md`.
Task 12's gate sentence counted the follow-ups wrongly and now names the
eight it moves. No task shape, ordering or ADR changed, and the
`cython-to-rust` phase-03 amendment is clerical.

## Stale-state sweep

Run on the working tree after the last prose edit, against
`origin/master` = `20db28f5`. `/tmp/t11/live.txt` is the live-doc
population, built by the first command.

**Live-doc population** (`[sweep-excluded-the-canonical-directory]`):

```sh
git ls-files "*.md" | grep -v -e "/task-notes/" -e "/learnings/" \
    -e "^docs/followups/done/" -e "lessons-examples.md" -e CHANGELOG.md \
    > /tmp/t11/live.txt   # 102 files
git ls-files "*.md" | xargs grep -l "Task 6\.4" | wc -l   # 55, this note included
```

**Behavior and liveness sweep** over the live population:

```sh
xargs grep -n -i -E "after (Phase 06 )?Task 6\.4|until (after )?(Phase 06 )?Task 6\.4|blocked until|declared (corpus )?regeneration|one (declared )?regeneration|still live|\(live\)|still supplies|still has a live|already drives|already uses" \
    < /tmp/t11/live.txt | cut -d: -f1,2
```

23 hits, every one KEPT:

| Hits | Why it stays |
| --- | --- |
| `docs/agents/lessons.md:144` | the lesson that names the phrases |
| `docs/followups/todo/eta-prime-two-photon-line-missing-factor-two.md:137` | quotes the proposal and says it "was never an available move" |
| `projects/cython-to-rust/phases/phase-03-numerics-foundation.md:233,234` | the "Amended by" note quoting the replaced text |
| `projects/cython-to-rust/phases/phase-04-spectra-kernels.md:28`, `references/numerics-replacements.md:121` | EDITED to past tense; they still match "until Task 6.4" |
| `projects/parity-pinned-defect-repair/PLAN.md:35,58` | the premise section, quoting what the follow-ups said |
| `PLAN.md:198` | "already uses", about `test/parity/stability.py`, unrelated |
| `PLAN.md:278` | Task 3's already-corrected "was still live; ... has since deleted it" |
| `PLAN.md:575,576,586,587,589,592,597` | this task's own spec, quoting what it sweeps |
| `references/defect-blast-radius.md:384` | EDITED; now "was still live" in the dated schedule |
| `references/the-premise.md:10,14,78,96` | the snapshot at `3e01590`, quoting the follow-ups |
| `references/the-premise.md:129` | the new re-derivation, in past tense |

**Task-id sweep** — `xargs grep -n "Task 6\.4" < /tmp/t11/live.txt`:
every hit outside the rows above is past tense and true (`f479b231`
deleted the files), a heading (`phase-06-mediator-spectra.md:85`), or
`cython-inventory.md:140`, a reference that declares itself a snapshot of
2.1.0. KEPT.

**Status-vocabulary sweep** (review round 1), per
`docs/agents/doc-consistency.md` §11. The liveness phrases above miss a
tracking table's own Status wording, so this sweep adds it. Pre-fix
occurrences come from the committed tree at `fe603cb0`, cut to 110
columns:

```sh
git grep -n -E 'Cython — (Task|Phase)|Cython - (Task|Phase)|still Cython|not yet ported|to be ported|unported' \
    HEAD -- projects docs hazma test .claude .codex README.md CHANGELOG.md
```

```text
projects/cython-to-rust/references/numerics-replacements.md:123:| `spectra/_positron/_pion.pyx:58` | cosθ | `
projects/cython-to-rust/references/numerics-replacements.md:124:| `spectra/_neutrino/_pion.pyx:124,127` | ener
projects/cython-to-rust/references/numerics-replacements.md:125:| scalar `thermal_cross_section` (`:1370` regi
projects/cython-to-rust/references/numerics-replacements.md:126:| vector `thermal_cross_section` (`:615` regio
projects/cython-to-rust/references/numerics-replacements.md:127:| 4 × mediator spectrum modules | cosθ ∈ [
projects/cython-to-rust/task-notes/README.md:218:  `cases.PORTED_ENTRY_POINTS` would keep unported kernels bit
projects/cython-to-rust/task-notes/README.md:268:  measurement, and the neutrino kernels are unported and unch
projects/cython-to-rust/task-notes/phase-04/README.md:421:  `spectra.photon.charged_pion` takes; the two unpor
projects/cython-to-rust/task-notes/phase-04/task-4.1-positron-muon.md:436:  difference scoped to `PORTED_ENTRY
projects/cython-to-rust/task-notes/phase-04/task-4.5-photon-rho.md:290:  taken by the two ρ cases; the seven
test/parity/tolerances.py:107:    for the next unported member rather than a live budget, and Phase 06
```

| Hit | Action |
| --- | --- |
| `numerics-replacements.md:123-127` | EDITED: each Status cell names its porting task, kernel, landing commit (`e2698eb6`, `6df9cfd8`, `aa6ab98b`, `75947619`, `c384aff3`) and the commit that deleted the `.pyx` |
| `test/parity/tolerances.py:107` | EDITED: `QUAD_RTOL` is the starting point "for a newly ported quadrature-backed case"; the Phase 06 prediction is gone, matching the `QUAD_RTOL` constant's own comment |
| the five `cython-to-rust/task-notes/` hits | KEPT: task notes of a Complete project, past-tense records of their run |

The same table's lead-in, "Keep this table current as Phases 04–06
land", is KEPT: it is the rule under which the rows above were just
brought current, and it binds no further port.

Post-fix, the same pattern over the working tree:

```sh
rg -n --hidden 'Cython — (Task|Phase)|Cython - (Task|Phase)|still Cython|not yet ported|to be ported|unported' \
    projects/ docs/ hazma/ test/ .claude/ .codex/ README.md CHANGELOG.md \
    | sort | grep -v task-11-prose
```

```text
docs/agents/lessons-examples.md:802:  table's own status vocabulary, `Cython — Task 4.6`, `Cython — Phase 05`
docs/agents/lessons-examples.md:803:  and `Cython — Phase 06`, which no liveness phrase matches; two reviewers
projects/cython-to-rust/task-notes/README.md:218:  `cases.PORTED_ENTRY_POINTS` would keep unported kernels bit-exact for
projects/cython-to-rust/task-notes/README.md:268:  measurement, and the neutrino kernels are unported and unchecked.
projects/cython-to-rust/task-notes/phase-04/README.md:421:  `spectra.photon.charged_pion` takes; the two unported `QUAD` cases keep
projects/cython-to-rust/task-notes/phase-04/task-4.1-positron-muon.md:436:  difference scoped to `PORTED_ENTRY_POINTS` — would keep the unported
projects/cython-to-rust/task-notes/phase-04/task-4.5-photon-rho.md:290:  taken by the two ρ cases; the seven unported mediator-spectrum cases
```

The `grep -v` drops this note, which quotes the pattern. The two new
hits are the lesson that quotes the retired phrasing. Over
the 102-file live population the same pattern returns nothing.

**Grammar sweep** for the "Since … so" construction, over every touched
`.md`, with `P='\bSince\b[^;]{0,120}?, so\b'`. Pre-fix, each file read
at `HEAD` with `git show HEAD:<file> | rg -c -U --pcre2 "$P"` matched
once, in `corpus-repinning.md` (lines 136-137). Post-fix,
`git diff --name-only origin/master -- "*.md" | xargs rg -n -U --pcre2 "$P"`
prints nothing.

**Line-number citation sweep:**

```sh
python3 scripts/agents/check_doc_citations.py \
    $(git diff --name-only origin/master -- "*.md"; git ls-files -o --exclude-standard -- "*.md")
```

```text
docs scanned: 16
in-repo citations checked: 1
  resolved by suffix: 1
external citations skipped: 30
out-of-range or ambiguous: NONE
```

The one in-repo citation is `_scalar_mediator_spectra.py:72` in
`docs/agents/lessons-examples.md`, an existing entry this change does not
touch. The 30 external skips are all into deleted `.pyx`/`.pxd`. Each one in a
touched follow-up is pinned to a revision (`f479b231^`, `e2698eb6^`,
`665aed5`, `ed1fa20`) and was read back at that revision with `git show
<rev>:<path> | sed -n <line>p`. The ones in
`projects/cython-to-rust/references/numerics-replacements.md` and
`phase-03-numerics-foundation.md` are inventory rows naming what the
port replaced (Decisions).

**Forward-looking phrase sweep** — `rg -n "(Task [0-9]+ will|will be
added|still pending|today: ?stub|currently|In Progress)"` over the same
14 files: `PLAN.md:2` and `README.md:5` (`In Progress`, correct until
Task 12), `README.md:589` ("concurrently"), and
`neutrino-pion-electron-line-counted-twice.md:73` ("currently pin the
defect", followed by "**Done, 2026-09-06.**"). All KEPT.

**Count sweep:**

| Claim location | Command | Actual | Status |
| --- | --- | --- | --- |
| Findings, "54 tracked files … 55" | `git ls-files "*.md" \| xargs grep -l "Task 6\.4" \| wc -l`, then again with `grep -v task-11-prose` before `xargs` | 55, 54 | OK |
| Findings, "102 live `.md` files" | `wc -l < /tmp/t11/live.txt` | 102 | OK |
| Findings, "edits in 12 of them" | `git diff --name-only origin/master -- "*.md" \| grep -v -e task-notes -e docs/agents/lessons \| wc -l` | 12 | OK |
| Files Changed, "all ten Serving-kernel cells" | `grep -c '\*\*repaired\*\*' references/defect-blast-radius.md` | 10 | OK |
| the-premise, `find` prints `0` | `find hazma -name "*.pyx" -o -name "*.pxd" \| wc -l` | 0 | OK |
| Every cited commit exists | `git cat-file -t` on each of `f479b231 e2698eb6 1a304d02 75947619 c384aff3 20db28f5 6df9cfd8 aa6ab98b` | 8 × `commit` | OK |

**Numerical-impact statement:** no public value changes (verified:
`git diff --stat origin/master -- hazma rust test/parity/data
test/parity/oracles | wc -l` prints `0`).

**Exit Criteria → artifact:**

| Criterion | Satisfied by |
| --- | --- |
| No live doc sequences a repair after Task 6.4 | the behavior sweep above, 23 hits all KEPT |
| phase-03 copy corrected | `phase-03-numerics-foundation.md`, "Amended by" note |
| Twin-liveness claims corrected | `defect-blast-radius.md` roster and schedule; `corpus-repinning.md` protocol |
| Seven "Risks" sections | A1's and A4's rewritten; B1's and B2's were already corrected; A2, A3, B3 carry none |
| Citations checked, sweep pasted | this block |

**Task-note self-consistency:** `**Status:** Complete`, every Exit
Criterion checked, and every file in Files Changed appears in `git diff
--stat origin/master` or is this new note.

**Preflight** (after the review-round fixes):
`scripts/agents/preflight.sh --paths "test/parity/test_oracles.py
test/parity/tolerances.py" --md "<the 16 touched .md files>"` printed
`RESULT: PASS`, exit 0, with every gate green. The pytest row read
`2320 passed, 16 skipped, 1 warning, 37 subtests passed in 56.82s`, and
the version bump was SKIP because this is not a closing PR. Only this
note was edited after that run, so `markdownlint --dot` was re-run on
it, and it passed.

## Handoff to Next Task

- Task 12 is next and last. Read `../PLAN.md` Task 12, then this
  directory's `README.md` "Numerical impact so far".
- Safe to assume: no live document sequences a repair after Task 6.4,
  and the eight roster follow-ups in `todo/` all read as repaired.
- Still to do at close: move those eight to `done/`, repoint their
  inbound links, and rename the `[Unreleased]` CHANGELOG heading to the
  bumped version.
