# Review-lessons ledger

A living ledger of review findings that are *class-shaped* — mistakes
that could recur on unrelated tasks, not one-off typos. It is the
feedback loop that otherwise goes missing: the same finding classes recur
across PRs with nothing capturing the lesson in between.

## Contract

- **Append** (`review-respond`, the lessons step): when a review —
  especially a verification or external round — catches a mistake that is
  class-shaped, add a one-line entry in the same commit, citing the
  PR(s). If an existing entry already covers the class, add the new PR to
  its citation list rather than duplicating.
- **Read** (`execute-single-task`, before writing code; every reviewer,
  per the review-lenses baseline duties): read this file before working
  and check the diff against each listed class.
- **Promote and prune**: when an entry stabilizes, fold it into a
  `docs/agents/` checklist, `AGENTS.md`, or a lint rule, then delete it
  here. Keep this file under ~30 entries — it is a working set, not an
  archive. Promoted lessons live in their destination; this ledger only
  holds the classes not yet encoded elsewhere.

## Format

One line per class: `- [class] one-line rule (PR #N, PR #M)`. The
`[class]` tag is a short kebab-case slug so recurrences are easy to match
and merge.

Every entry must cite at least one real PR. **Do not add an entry from
intuition** — an uncited "lesson" is a guess wearing a citation's
clothes, and it costs every future reviewer the time to check it. If you
believe a class is worth pre-empting but have no PR for it, put it in the
relevant `docs/agents/` checklist as a check, not here as a lesson.

## Ledger

- [ported-file-stale-reference] A file copied in from another repo carries
  that repo's references — workflow paths, CI actions, internal design docs
  — and they read as authoritative here while pointing at infrastructure
  that does not exist. Grep every ported file for paths and tool names and
  confirm each resolves in *this* repo before shipping it (PR #18:
  `check_pr_title.py` claimed to mirror `.github/workflows/pr_linter.yaml`,
  which exists only upstream).
- [normalized-id-in-path] An id that is normalized for display ("Phase 2")
  must not be interpolated into a filesystem path when the on-disk name is
  the un-normalized form (`phase-02/`). The two agree for most values and
  diverge exactly at the padded ones, so a spot check passes and the bug
  ships. When a value has both a display form and a path form, name them
  separately and test a value where they differ (PR #18:
  `resolve_phase.py`'s kickoff prompt sent agents to `task-notes/phase-2/`).
