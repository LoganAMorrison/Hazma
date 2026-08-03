---
name: review-respond
description: "Synthesize existing Hazma PR review feedback, assess each comment on merit, implement justified in-scope fixes, run the preflight gate, and provide verification-ready responses. Use when reviews already exist; do not use to collect reviews."
---

# Respond to review feedback

Act as the implementing engineer. Accept neither comments nor rejections on
faith: assess each against the current PR head, task scope, ADRs, tests, and
measurements. Use the caller-supplied worktree and branch; otherwise resolve
the PR/branch and its true merge base.

## Assess every comment

Read applicable project context and
[`docs/agents/lessons.md`](../../../docs/agents/lessons.md), then classify
each comment as exactly one of:

|Category|Meaning|Action|
|--------|-------|------|
|`fix`|Valid and in scope|Implement it.|
|`fix-partial`|Valid concern; proposal is wrong|Make the smaller correct fix.|
|`acknowledge`|Valid but out of scope|File or link a follow-up.|
|`reject`|Incorrect or already handled|Explain with command evidence.|

A numerics comment requires a measurement before accepting or rejecting it.
For an acknowledged follow-up, use the lifecycle in
[`docs/workflow.md`](../../../docs/workflow.md), including existing follow-ups
and open-PR deduplication.

## Implement and verify accepted changes

Fix the whole class, not only the cited line. Before changing a factual claim,
perform the before/after stale-sibling sweep required by
[`docs/agents/doc-consistency.md`](../../../docs/agents/doc-consistency.md).
Re-measure affected public paths and update durable records and the PR body if
needed. Rebuild after Cython edits and run the preflight gate before staging.

For class-shaped review findings, update the lessons ledger in the same commit
only when a real PR citation is available. In a pipeline or review cycle,
commit and push the approved fixes after the gate; standalone use leaves them
uncommitted for `$commit-and-pr`.

## Report

Return one block per reviewer:

```text
## Response to <Reviewer ID>

### Accepted
| # | Comment Summary | Category | Action Taken |
|---|---|---|---|

### Acknowledged (deferred)
| # | Comment Summary | Reason | Follow-up file |
|---|---|---|---|

### Rejected
| # | Comment Summary | Reason (with evidence) |
|---|---|---|

### Verdict Requested: <ACCEPT | ITERATE>
```

Then give a combined count, files modified, literal test summary,
numerical-impact result, and outstanding questions. Never do drive-by cleanup,
silently revert a valid fix, or claim a green run without its summary line.
