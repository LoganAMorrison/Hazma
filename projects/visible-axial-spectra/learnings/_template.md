# <Phase X | Project> Learnings: <Name>

<!--
For phased projects, create one learnings file per phase when the phase
closes: `phase-XX-<slug>.md`.

For flat projects, create one retrospective when the project wraps up:
`project-retrospective.md` (or similar).

Learnings are durable memory for future work that touches the same
area — not a status log. Synthesize from task notes; don't just copy
them.
-->

## 1. Implementation Reality Check

<Summarize how execution matched or diverged from the plan. Link to any
ADRs generated during this phase or project.>

## 2. Critical Context for Future Work

<List types, variable names, invariants, or internal API boundaries
established here that future tasks should respect.>

- <Rule or contract 1>

## 3. Quirk Log & Edge Cases

<Document language-level fights, dependency quirks, or specification
ambiguities resolved here, so future work doesn't repeat the same
mistakes.>

- <Quirk 1>

## 4. Test Infrastructure State

<Note new test harnesses, fixtures, mocks, or specific commands
established here that should be reused.>

- <Testing tool or standard 1>

## 5. Follow-on seeds

<List substantive deferred items the project surfaced — work that's
out of scope here but worth picking up later. Each seed gets one or
two paragraphs explaining what it is, what triggers it, and which
files / facets / ADRs are involved.

For every seed, also drop a stub in `../../../docs/followups/` (one
file per seed; from `projects/<slug>/learnings/` that resolves to
the repo's `docs/followups/`). The retrospective entry is the
historical record; the followup file is the live actionable form.
See `../../../docs/workflow.md#follow-ups` for the lifecycle.>

- <Seed 1>
