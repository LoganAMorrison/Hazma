# `.claude/skills/*/SKILL.md` was never in markdownlint's scope

- **Added:** 2026-08-06
- **Source:** PR #48 review round 1 (first PR to pass a skill file to the
  markdownlint gate)
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** none — touches lint config and agent docs
  only, no library code. Ripens the next time any PR edits a skill file.

## Why

`.markdownlint.jsonc` (added by
[`markdownlint-config-for-templates`](../done/markdownlint-config-for-templates.md))
encodes the repo's canonical shapes, but it was written against `docs/`
and `projects/` — the two trees that PR's sweep covered. The skills under
`.claude/skills/` were never linted, so nothing checked whether the
config accommodates *their* shapes.

PR #48 was the first PR to edit a skill file and therefore the first to
pass one to preflight gate 6. It fails, and not because of anything that
PR changed:

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

- **MD036** fires on `**When to use this skill**`, the bolded section
  label every `SKILL.md` opens with. The config already special-cases
  these files once (`MD041`'s `front_matter_title` accepts their `name:`
  frontmatter) on the grounds that the format "is defined by the
  harness, not by us" — the same argument applies here.
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
[`docs/agents/preflight.md`](../../agents/preflight.md) gate 6 that the
skill trees are in scope — otherwise the next agent rediscovers this.

## Entry points

- `.markdownlint.jsonc` (the `MD041` block is the precedent for a
  skill-file-scoped relaxation)
- `.claude/skills/review-plan/SKILL.md`,
  `.claude/skills/task-pipeline/SKILL.md`
- `docs/agents/preflight.md` (gate 6, which documents what the gate covers)
- Prior art: [`markdownlint-config-for-templates`](../done/markdownlint-config-for-templates.md)

## Risks / open questions

Option 2 changes heading structure in files the agent harness reads. If
any skill or doc references a `SKILL.md` section by its bolded label,
that reference has to move with it — sweep before choosing it.
