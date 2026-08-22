# ADR 0002: Phase learnings replace a closed phase's task notes

**Date:** 2026-08-21
**Status:** Accepted (implemented by the PR that introduced it, branch
`claude/agent-context-cost-955b49`, same date)

## Context

Agent task execution in this repo is gated by a mandatory reading list
(`execute-single-task` Step 4 in both skill trees, and the kickoff prompt
`scripts/agents/resolve_phase.py` emits). Before this decision the list
sent an agent to the project working-memory README on every task *and*
to the learnings of every closed phase, while the README had accreted
every closed phase's Findings, Decisions, Files Changed and Verification
entries — the same history the learnings condense. Each phase learnings
file opens with "Read this instead of the notes; the notes are history",
yet nothing in the contract let an agent skip the accretions.

Measured 2026-08-21 at `c57ce4f` with tiktoken's `o200k_base` encoding
(an approximation of Claude's tokenizer — the ratios are what matter):

| item | before | after |
| --- | ---: | ---: |
| `projects/cython-to-rust/task-notes/README.md` | 2,574 lines / 47,690 tokens | 508 / 9,034 |
| whole `projects/cython-to-rust/task-notes/` tree | 35 files / 332,691 tokens | 40 / 334,747 |
| `projects/cython-to-rust/learnings/` | 6 files / 20,436 tokens | unchanged |
| `docs/agents/lessons.md` | 660 lines / 11,606 tokens | 210 / 3,517 |
| `execute-single-task` Step 4 list for cython-to-rust Task 5.3 (20 files) | 116,632 tokens | 71,195 |

The measurement script is in the introducing PR's body; it reads each
path from `origin/master` and from the branch and sums
`len(enc.encode(text))`. The learnings tree is one sixteenth of the notes
tree it condenses, and the README alone was 41% of the mandatory list.

Three `execute-single-task` transcripts (cython-to-rust Tasks 4.6, 5.1
and 5.2, 2026-08-20/21; sessions `48cf9c6b…`, `a9238d11…` and
`f533cd02…` under `~/.claude/projects/`) were also audited from their
`usage` fields and tool payloads. Each single-prompt run ended between
513k and 644k tokens of context. In the Task 5.2 run (74.7k → 513k):
tool results were 151k tokens, of which the mandatory documents above
were about 23k (the agent read two slices of the README, 777 tokens, and
never opened `lessons.md`); ad-hoc `sed -n` sweeps through `.pyx`,
generated `.c`, `.rs` and test files were about 75k; the agent's own
heredoc payloads — task-note and working-memory chunks of 1–7k tokens
each, plus code — were 71k over the session; the remainder was retained
reasoning. Tasks 4.6 and 5.1 had the same shape (heredoc payloads of
140k and 88k tokens). The reading list is therefore a real but bounded
lever; decisions 3 and 5 address the larger drivers.

## Decision

1. **Learnings are the reading contract for a closed phase.** An agent
   reads `learnings/phase-XX-*.md` for every closed upstream phase and
   does not open that phase's `task-notes/phase-XX/` notes or README
   unless a learnings entry, the current handoff, or a citation sends it
   to a specific note for a specific detail. Task notes of the *current*
   phase stay in the list — the previous task's `## Handoff to Next Task`
   and `## Open Questions` first. Flat projects: the same rule for the
   preceding task's note.
2. **The project working-memory README is a head file** of roughly 5k
   tokens: the status table(s), open questions, the rolling handoff, and
   pointers. When a section outgrows that it moves *wholesale and
   verbatim* to a sibling file and the heading stays as a pointer:
   `## Numerical impact so far` → `task-notes/numerical-impact.md`; a
   closed phase's Findings, Decisions, Files Changed and Verification
   entries → `task-notes/history-<section>.md`, swept at phase close once
   the learnings file exists. Nothing is deleted, summarised or reordered
   by a sweep, and each archive file records its source lines and commit.
3. **Task notes carry a length budget**, stated in the template and the
   skill: `## Findings` + `## Decisions` under ~100 lines together, the
   whole note under ~500, with the pasted-evidence sections (measurement
   tables, Verification summary lines, Numerical impact, the Stale-state
   sweep block, Handoff) exempt. No gate is weakened; prose is what gets
   compressed.
4. **`docs/agents/lessons.md` holds one line per class** with its PR
   citations; the worked examples live verbatim in
   `docs/agents/lessons-examples.md` under a `### <class>` heading, and
   the append contract writes both.
5. **Context discipline is part of the skill:** delegate survey reads to
   a subagent and take back the conclusion, not the file; never echo a
   generated artifact into the transcript; write a file once and do not
   read it back.

Applied to `cython-to-rust` on 2026-08-21: README lines 58–892,
1492–1658, 1659–1953 and 1954–2179 (at `c57ce4f`) moved to
`task-notes/history-{findings,decisions,files-changed,verification}.md`;
lines 893–1491 moved to `task-notes/numerical-impact.md`; every live
citation of the README's "Numerical impact so far" section was repointed
(`PLAN.md`, `rules.md`, the phase files, `task-notes/phase-07/README.md`,
the `parity-pinned-defect-repair` PLAN). A line-level check — every
non-blank line of the old README still exists under `task-notes/` —
passed with 0 of 2,051 moved lines missing. Other projects adopt the
split when their README outgrows the budget; the templates carry the
instructions.

## Consequences

- **Positive:** the mandatory list for the next cython-to-rust task
  drops from 116,632 to 71,195 tokens (−39%), the README from 47,690 to
  9,034 and `lessons.md` from 11,606 to 3,517; every future phase close
  has a sweep step that keeps the head file bounded; the note budget and
  the context-discipline rules address the two drivers the transcripts
  actually show; nothing historical is deleted and every moved entry is
  one `git show` from its original position.
- **Negative:** a detail that lives only in a closed phase's task note is
  one hop further away — the agent follows a pointer from the learnings
  file or greps `task-notes/` for it, instead of meeting it in the
  README; a reviewer checking a historical claim reads the archive. The
  head file still carries the verbatim Handoff and Open Questions (4,735
  and 2,421 tokens at `c57ce4f`), so it lands at ~9k rather than 5k until
  the next task rewrites the handoff under the new budget. Two
  `test/parity/*.py` docstrings cite the README's section by name; they
  still resolve because the heading remains as a pointer, and they were
  left alone because `test/` was outside the introducing PR's scope.
- **Mitigation:** phase learnings are already contracted to carry
  everything durable (`docs/workflow.md` §Learnings; the learnings
  template's §2 "Critical Context for Future Work"), and this ADR makes
  that contract load-bearing — a learnings author now knows the notes
  will not be read; each archive file states its source lines and commit;
  the README keeps every moved section's heading as a pointer so no
  inbound citation breaks; the archive files sit beside the README so the
  moved text's relative links still resolve; and they are plain markdown,
  so `rg` over `task-notes/` finds what it always found.
