# The review-lessons ledger is past its working-set cap

- **Added:** 2026-08-19
- **Source:** PR #72 review round 1 — appending two class-shaped entries
  took the ledger from 36 to 38 against its own "under ~30" contract
- **Scope:** cross-cutting (`docs/agents/lessons.md` is read before every
  task and by every reviewer)
- **Status:** open
- **Triggers / blockers:** none. Independent of any project; the only
  cost of waiting is that every agent and every reviewer keeps paying the
  read.

## Why

[`docs/agents/lessons.md`](../../agents/lessons.md) opens with its own contract:
*"Keep this file under ~30 entries — it is a working set, not an
archive"*, with a **Promote and prune** clause — when an entry
stabilizes, fold it into a `docs/agents/` checklist, `AGENTS.md`, or a
lint rule, then delete it from the ledger.

The file is at **38 entries** and 42,746 bytes — `grep -c '^- \['` and
`wc -c` over `docs/agents/lessons.md`. It has been appended to
on most recent PRs and pruned on none of them, which is the expected
failure mode: appending is a required step of `review-respond`, pruning
is nobody's step. The cost is not cosmetic. The ledger is on the
mandatory reading list for `execute-single-task` before writing code and
for every reviewer under the review-lenses baseline duties, so each entry
past the useful set is a tax on every task in the repo, and a long
ledger is read less carefully than a short one — which is exactly the
failure the entries exist to prevent.

Several entries look ripe. `[unpinned-formatter-version]` describes a
divergence that was fixed by deleting the duplicate pin, and the
invariant now lives in `pyproject.toml`'s `[dependency-groups]` — a
sentence in `AGENTS.md` would carry it. `[wheel-tag-vs-extension-abi]`
is a packaging fact that belongs beside the packaging gate.
`[elided-doc-paths]` is enforceable by
`scripts/agents/check_doc_citations.py`, which already resolves
ambiguous basenames — the lesson could become the checker's error text.
Those are candidates to assess, not a decided list.

## What

A promote-and-prune pass over `docs/agents/lessons.md`:

1. Triage each of the 38 entries into *still a live working-set class*,
   *stabilized — promote and delete*, or *superseded by a gate that now
   catches it*.
2. For each promotion, write the rule into its destination
   (`docs/agents/preflight.md`, `doc-consistency.md`,
   `review-lenses.md`, `AGENTS.md`, or a checker's message) and delete
   the ledger entry — the contract says promoted lessons live in their
   destination, not in both places.
3. Land back under ~30.

Two constraints the pass has to respect. An entry's PR citations are the
evidence that the class is real, so a promotion carries them to the
destination rather than dropping them. And a few entries have grown into
multi-shape essays (`[measurement-taken-before-the-task-ended]` and
`[sibling-copies-of-a-fixed-claim]` are each several hundred words with
four or five distinct sub-shapes) — those are candidates for splitting
into a `docs/agents/` reference of their own rather than for deletion,
since the sub-shapes are individually load-bearing.

Worth considering alongside: make the append step in
`.claude/skills/review-respond/SKILL.md` and its `.codex/` twin state the
cap, so the next agent that appends past it is the one that notices.

## Entry points

- `docs/agents/lessons.md` — the ledger; its `## Contract` section is
  the authority on the cap and the promote-and-prune clause
- `docs/agents/preflight.md`, `docs/agents/doc-consistency.md`,
  `docs/agents/review-lenses.md`, `AGENTS.md` — the promotion
  destinations the contract names
- `scripts/agents/check_doc_citations.py` — where `[elided-doc-paths]`
  would land if it becomes a checker message rather than a lesson
- `.claude/skills/review-respond/SKILL.md`, `.codex/skills/review-respond/SKILL.md`
  — the skills that mandate the append and do not mention the cap

## Risks / open questions

- Pruning is lossy in a way appending is not: an entry deleted without a
  faithful promotion silently removes a class every future reviewer was
  checking against. Prefer moving text to deleting it, and land the pass
  as one reviewable diff rather than trickling deletions into unrelated
  PRs.
- The cap is "~30", not 30. The goal is a set an agent will actually
  read, so the pass should be judged on whether the survivors are the
  classes that still recur — not on hitting a number.
