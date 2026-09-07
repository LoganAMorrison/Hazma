# `.claude/skills/*/SKILL.md` was never in markdownlint's scope

- **Added:** 2026-08-06
- **Source:** PR #48 review round 1 (first PR to pass a skill file to the
  markdownlint gate)
- **Scope:** cross-cutting
- **Status:** done
- **Resolved:** 2026-09-06 in
  [PR #88](https://github.com/LoganAMorrison/Hazma/pull/88), by option 2
  (normalize the docs) rather than the option 1 this file recommended —
  see **Resolution** below for why the recommendation was wrong.

## Why

`.markdownlint.jsonc` (added by
[`markdownlint-config-for-templates`](../done/markdownlint-config-for-templates.md))
encodes the repo's canonical shapes, but it was written against `docs/`
and `projects/` — the two trees that PR's sweep covered. The skills under
`.claude/skills/` were never linted, so nothing checked whether the
config accommodates *their* shapes.

PR #48 was the first PR to edit a skill file and therefore the first to
pass one to preflight's markdownlint gate. It fails, and not because of
anything that PR changed:

```text
$ markdownlint --dot .claude/skills/*/SKILL.md .codex/skills/*/SKILL.md
.claude/skills/review-plan/SKILL.md:12    MD036/no-emphasis-as-heading
.claude/skills/review-plan/SKILL.md:20    MD036/no-emphasis-as-heading
.claude/skills/task-pipeline/SKILL.md:14  MD036/no-emphasis-as-heading
.claude/skills/task-pipeline/SKILL.md:21  MD036/no-emphasis-as-heading
.claude/skills/task-pipeline/SKILL.md:318 MD032/blanks-around-lists
.claude/skills/task-pipeline/SKILL.md:336 MD031/blanks-around-fences
.claude/skills/task-pipeline/SKILL.md:356 MD031/blanks-around-fences
.claude/skills/task-pipeline/SKILL.md:361 MD031/blanks-around-fences
.claude/skills/task-pipeline/SKILL.md:366 MD031/blanks-around-fences
```

Nine errors in two files, none of them introduced by PR #48 —
`task-pipeline/SKILL.md` is not in that diff at all. The practical
effect is that any PR touching a skill file inherits a red gate row it
did not cause, which is exactly the dynamic that gets a gate ignored
rather than satisfied.

The two rules are arguably the config's problem, not the docs':

- **MD036** fires on `**When to use this skill**`, which this file called
  "the bolded section label every `SKILL.md` opens with". That was the
  load-bearing claim for option 1, and it is false: six of the eight
  `.claude` skills spell it `## When to use this skill` as a real
  heading, and only `review-plan` and `task-pipeline` use bold. The
  `MD041` precedent cited here does not transfer, because that one
  accommodates a frontmatter format the harness imposes, whereas this is
  two files deviating from a convention the repo already keeps.
- **MD031 / MD032** fire inside blockquoted example blocks, where the
  fences and lists are quoted content being *shown*, not structure.

## What

Decide one of:

1. **Relax per path.** Extend `.markdownlint.jsonc` with the two rules
   scoped to the skill trees, alongside the existing `MD041` note. This
   is the cheapest and matches the precedent already in the file.
2. **Normalize the skill docs.** Promote the bolded labels to real
   headings and add the blank lines. Larger diff, and it edits the
   agent-tooling contract — a heading change is visible to anything that
   navigates these files by section.

Then bring the skill trees into the swept set for real: run
`markdownlint --dot` over `.claude/skills/` and `.codex/skills/` once,
fix or relax whatever else surfaces, and say in
[`docs/agents/preflight.md`](../../agents/preflight.md)'s markdownlint
gate that the skill trees are in scope — otherwise the next agent
rediscovers this.

## Entry points

- `.markdownlint.jsonc` (the `MD041` block is the precedent for a
  skill-file-scoped relaxation)
- `.claude/skills/review-plan/SKILL.md`,
  `.claude/skills/task-pipeline/SKILL.md`
- `docs/agents/preflight.md` (the markdownlint gate, which documents
  what the gate covers)
- Prior art: [`markdownlint-config-for-templates`](../done/markdownlint-config-for-templates.md)

## Risks / open questions

Option 2 changes heading structure in files the agent harness reads. If
any skill or doc references a `SKILL.md` section by its bolded label,
that reference has to move with it — sweep before choosing it.

## Resolution

Option 2, and the count above is why: relaxing `MD036` for the skill
trees would have bent a rule to fit two outliers rather than bringing
them in line with the other fourteen files. The four bolded labels in
`review-plan/SKILL.md` and `task-pipeline/SKILL.md` are now `##`
headings, matching their siblings and sitting above the `## Inputs` those
files already had.

The risk this file flagged — that something might reference a section by
its bolded label — was swept before the change and did not materialize:
`rg -n --hidden 'When to use this skill|When NOT to use this skill'`
returned only the labels themselves and this file.

The remaining five errors were `MD031`/`MD032` inside Phase C's
blockquoted subagent prompt, where a fence or list sat flush against the
line above. They took quoted blank lines (`>`), not bare ones — a bare
blank line would have split one 79-line blockquote into several, which
was checked after the fix rather than assumed.

`markdownlint --dot .claude/skills/*/SKILL.md .codex/skills/*/SKILL.md`
now exits 0, and the markdownlint gate in
[`preflight.md`](../../agents/preflight.md) says the skill trees are in
scope so the next agent does not have to rediscover that they are.
