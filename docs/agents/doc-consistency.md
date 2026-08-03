# Doc-consistency checklist

The canonical checklist for keeping durable docs consistent with what a
PR ships. Stale-sibling doc drift — counts, status, identifiers, units,
and contract phrases that disagree across files — is a top review-defect
class and drives most round 2–4 churn. This is the one copy; skills point
here instead of restating it. Any Blocking divergence found here is a
REQUEST CHANGES verdict: a durable-doc contradiction is a blocking
severity anchor.

## Two audiences, one list

- **Implementers run it pre-PR (shift-left).** Before you commit, walk
  every check below that your diff touches — cheaper than a review round.
  `execute-single-task` and `review-respond` drive you through this list.
- **Reviewer D verifies it** against the PR head as the primary review
  activity — re-run the commands, don't trust the prose. See
  [`review-lenses.md`](review-lenses.md) for the roster and verdict rule.

## The checks

Each check names the exact command where one exists; run commands against
the current branch head, never the ambient checkout.

1. **Reproduce every cited fact.** For every count, command, file path,
   and identifier mentioned in a durable doc the diff touches or
   references — task note, working-memory README, phase file, ADRs,
   learnings docs, README, `CHANGELOG.md`, follow-ups, and the PR body —
   re-run the listed command and re-grep the listed identifier. Treat the
   task note's `## Verification` section as **regenerated, not curated**:
   run each command, paste output verbatim, replace the section as a
   unit; never hand-edit individual numbers. Any command that no longer
   reproduces (wrong path, removed file, renamed test) must be fixed or
   removed. External-behavior assertions ("matches the published value in
   arXiv:1907.11846 Table 2", "reproduces PPPC4DMID within 1%") need a
   primary-source citation in the same paragraph — a DOI, an arXiv id
   plus equation number, a permalink, or a command + output.

2. **Units and physical claims.** Every docstring, ADR, and plan sentence
   that states a unit, a normalization, or a frame (`MeV⁻¹`, `per
   annihilation`, `lab frame`, `dN/dE` vs `E dN/dE`) must agree with what
   the code returns. Confirm against the code, not against a sibling
   doc — sibling docs are where the wrong unit propagates.

3. **Canonical-contract gates.** The phase-file gate text and ADR
   sections must agree with what the PR shipped. If the PR changed the
   contract, patch the phase file or ADR in this same PR — do not defer
   the contract patch to a follow-up. On a project-closing PR (a
   `status:` flipped to `Complete`), walk every gate bullet in PLAN /
   phase file / ADRs and record one line of evidence per gate (test name,
   measurement, command output). Block a closing PR with any open
   canonical gate.

4. **Intra-document coherence.** Inventory every count and arithmetic
   claim in a touched doc — the prose sentences, not just the
   authoritative table — plus every qualitative prose claim (mechanism,
   failure mode, or derivation stated in words), and reconcile each
   against the authoritative section. Prose counts must agree with their
   tables; breakdowns must sum to their totals; described mechanisms must
   match sibling sections. Do **not** `rg` for the authoritative value
   and stop — that finds the correct number, never the stale one; the bug
   is a sentence saying `12 channels` while the table says `14`, caught
   only by reading the prose against the table. When a fact is corrected,
   re-derive it from first principles rather than confirming it against
   an adjacent corrected copy.

5. **Four-corner cross-document consistency.** Whenever the diff touches
   any one of `{PLAN.md, projects/<slug>/adrs/*,
   projects/<slug>/task-notes/*, projects/<slug>/learnings/*,
   docs/followups/*}`, open every other file in that set and `rg` the
   changed concept. The PLAN summary line, phase README row, task-note
   `Status`, working-memory Tasks row, ADR §Consequences vs §Body, and
   follow-up status must all agree on status and concept; cite
   line-numbered evidence.

6. **Version-bump and CHANGELOG consistency (closing PRs).** If the diff
   flips a `projects/<slug>/PLAN.md` `status:` to `Complete`, run
   `scripts/agents/preflight.sh --closing` to verify that `VERSION` in
   `hazma/__init__.py` moved relative to the trunk and that
   `CHANGELOG.md` carries a matching `## [X.Y.Z] — YYYY-MM-DD` section
   naming the project slug. Also confirm the bump **level** matches the
   `version_bump:` frontmatter and that the frontmatter itself is still
   right per [`../versioning.md`](../versioning.md) — a project that
   ended up moving a published number is `minor`, not `patch`, even if it
   started as one.

7. **Docstring / inline-comment sweep on touched source files.** For
   every `.py` / `.pyx` file the diff touches, read the module docstring
   and the docstring above every public function or class, and flag any
   sentence that describes the pre-change state — a `Parameters` section
   listing a removed argument, a `Returns` section with the old units, a
   "currently only supports two-body final states" surviving the change
   that added three-body. The same sweep applies to task-note section
   comments that say "Task N will…" / "still pending" / "today: stub".

8. **Sphinx surface.** If the diff renames, moves, or removes a public
   object, grep `docs/source/` for it. An `automodule` / `autofunction`
   directive pointing at a gone symbol breaks the published docs build
   without failing any test here.

9. **Follow-up file mechanics.** If the PR closes a follow-up, confirm
   the row in `docs/followups/README.md` flipped from Open to Done, that
   the file was `git mv`d from `todo/` to `done/` (a closed status still
   under `todo/` is blocking), that every inbound reference was repointed
   (`rg -l 'followups/todo/<slug>\.md'` shows none stale), and that every
   function name, module, or file path the stub cites still exists. If
   the PR opens a follow-up, confirm the stub lives under `todo/`,
   follows `docs/followups/_template.md`, has an index row, and its
   references are valid. A closing-task retrospective must cross-check
   **all** of `docs/followups/todo/` for entries sourced from this
   project — not only the task's own diff.

10. **PR title/body sanity.** Validate the title with

    ```bash
    scripts/agents/check_pr_title.py "<pr-title>"
    ```

    (Conventional Commits; scope `^[a-z0-9-]+$`, ≤10 chars, no trailing
    `.` — `phase-space` is 11 chars, rejected on length, so use `phase`.)
    Skip this while the title is still the pipeline placeholder. Then
    verify every count, file name, and identifier in the PR body
    reproduces against today's diff — stale bodies that name removed
    functions or pre-fix test counts are a recurring blocking finding.

11. **Stale-sibling sweep procedure.** Before fixing any factual claim —
    a count, identifier, command, line number, unit, or qualitative prose
    claim — run the class-wide sweep; do not point-fix the cited line:
    `rg -n '<old-value>' projects/ docs/ hazma/ test/ README.md
    CHANGELOG.md`. Paste the output under `### Pre-fix occurrences`.
    Apply the fix to every listed occurrence, or explicitly justify each
    one you skip. Re-run the same `rg` and paste under
    `### Post-fix occurrences`. For **numeric** fixes, sweep on the bare
    digit (`\b<old>\b`), not the surrounding phrase — numbers feel
    localized but rarely are. Qualitative prose claims get the same
    before/after treatment. This is what prevents a fix from introducing
    an adjacent stale contradiction.

12. **New artifacts count too.** A brand-new test, comment, or doc that
    embeds a stale fact is the most common escape. Sweep new files, not
    only edited ones: for each `??` entry from `git status --short`,
    `git add -N <path>` so it appears in the diff, or open it and walk it
    as a fresh creation. Re-derive every embedded count from a canonical
    command rather than trusting the literal.

## The sweep block (forcing function)

Describing what to check is not enough — the forcing function is pasting
actual output. Implementers append this block to the task note (above
`## Handoff to Next Task`) before committing, running each command
against the current branch. Skimming or omitting it is the single biggest
cause of REQUEST CHANGES verdicts.

Under a `## Stale-state sweep` heading, paste output for each sub-sweep
(mark each hit KEPT / EDITED / DELETED):

- **Identifier sweep** — every new/renamed/removed name and every
  identifier cited in §Files Changed / §Decisions / §Findings:
  `rg -n '<identifier>' projects/<slug>/ docs/ README.md hazma/ test/`.
- **Line-number citation sweep** — `file:line` citations of touched
  files: `rg -n '<file_basename>\.py:[0-9]+' projects/<slug>/ docs/`. To
  bounds-check them all mechanically, run
  `scripts/agents/check_doc_citations.py --changed-vs origin/master`
  (add `--map <repo/relative/path.py>` for each short-form basename it
  reports as ambiguous) and paste its summary.
- **Forward-looking phrase sweep** — `rg -n '(Task [0-9]+ will|will be
  added|still pending|today: ?stub|currently|In Progress)'
  projects/<slug>/ hazma/`.
- **Count sweep** — a table re-deriving every numeric claim from a
  canonical command (`Claim location | Command | Actual | Status`).
- **Numerical-impact statement** — for any diff that can reach a public
  function: the grid you evaluated, whether values moved, and by how
  much. `No public value changes (verified: <command>)` is a valid row;
  silence is not.
- **Exit Criteria → test mapping** — a table naming the test or artifact
  that satisfies each Exit Criterion bullet; a missing row means not
  done.
- **Task-note self-consistency** — the `**Status:**` header, §Exit
  Criteria checkboxes vs the mapping table, and that every
  function/class/file/API cited in §Files Changed / §Decisions /
  §Findings appears in `git diff --stat origin/master --` or a created
  file.

Three rules govern the block itself, in this order:

1. **Sweep last.** Freeze every prose edit — including the `lessons.md`
   ledger append that `review-respond` asks for — before re-deriving any
   sweep. A ledger append is a content edit that enters the diff, so it
   moves both the line numbers the sweep cites *and* the doc set a
   `--changed-vs` run covers.
2. **Then prove it is a fixed point.** Re-run every command once more
   after pasting and require it to reproduce. But *"reproduce" is not
   always "byte-identical"*: `rg` walks directories in parallel, so a
   multi-directory sweep returns the same rows with the same line numbers
   in a **different order** run to run. Compare `sort`ed captures for
   those, and claim byte-identity only of the deterministic commands — an
   explicit file list, `rg -c`, `wc -l`, `git diff`,
   `check_doc_citations.py`.
3. **Label anything hand-folded.** Collapsing many `rg` hits into one row
   per file, or appending KEPT/EDITED dispositions, is fine and usually
   clearer — but say the block is folded, and record the citing doc
   itself as a match when the command matches it. "Pasted from the
   command's real output" over a hand-annotated table is how a fabricated
   row survives four reviewers.

For a zero-touch sweep (a one-line fix with no identifier churn), rows
may legitimately read "no occurrences" — but still produce the block to
prove each command ran. Do not skip it because "this PR is too small":
two-line deletions routinely shift stale references downstream.

Read [`lessons.md`](lessons.md) before running this checklist and check
the diff against each recurring class listed there.
