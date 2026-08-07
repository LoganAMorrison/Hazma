---
name: review-pr
description: Review a PR that implements one plan task against its spec in projects/<slug>/, as a single lens-focused reviewer. Distinct from generic /code-review (working-tree diffs), /review-plan (pre-implementation plans), and /review-cycle (multi-reviewer orchestration).
---

**Role:** Act as a rigorous code reviewer for a PR that implements one
task from a project's plan (or an ad-hoc change). You are **one**
reviewer running **one** lens and producing **one** structured verdict.
Synthesis across lenses and iteration to convergence is `/review-cycle`'s
job. Your job is to determine whether the task was completed correctly,
with depth dictated by your assigned lens.

## When to use this skill

- The user asks to review a PR (by number or URL) against its plan task.
- Called by `/review-cycle` for each reviewer agent (one lens per
  invocation).

## When NOT to use this skill

- **Working-tree diffs / uncommitted changes** → `/code-review`. This
  skill reviews PRs; push the branch or open a draft PR first.
- **A project plan not yet implemented** → `/review-plan`.
- **Driving multiple reviewers to convergence** → `/review-cycle`.

## Inputs

Required:

- **PR** — a number, URL, or pushed branch name.

Optional:

- **lens** — `default`, `completeness`, `logic`, `doc-consistency`, or
  `numerics`. Default is `default` (the generalist pass). Each lens maps
  to one reviewer in the roster; see
  [`review-lenses.md`](../../../docs/agents/review-lenses.md).

## Workflow

### Step 1: Gather PR context

Normalize the input to a PR number: a number is used directly; a URL
yields its trailing number (`/pull/(\d+)`); a branch resolves via
`gh pr list --head <branch> --json number --jq '.[0].number'` (if none,
ask the user to open one, even a draft).

Fetch metadata:
`gh pr view <N> --json title,body,state,baseRefName,headRefName,files,additions,deletions,commits`

**Fetch the PR head fresh (baseline duty).** Do not review against the
ambient checkout. Delete any stale ref, then
`git fetch origin pull/<N>/head:refs/remotes/origin/pr/<N>`, and verify
the fetched SHA against `gh pr view`. Treat `gh pr diff <N>` as possibly
**truncated** — if the diff is large or looks clipped, read the changed
files from the fetched ref. Full recipe in
[`review-lenses.md`](../../../docs/agents/review-lenses.md) under
Baseline duties.

### Step 2: Resolve the project slug

Parse the head branch. `<agent>` is `claude` or `codex` — **parse both**:

- `<agent>/<slug>/<task-slug>` → project work; the first segment after
  the prefix is the slug.
- `<agent>/<desc>` (no second `/`) → ad-hoc. Skip project-spec reading;
  review against `AGENTS.md`, `docs/PR_GUIDELINES.md`, and the diff.

A branch starting with neither prefix is treated as ad-hoc — note it.

### Step 3: Read the task specification (project work only)

1. [`lessons.md`](../../../docs/agents/lessons.md) — read first and check
   the diff against each listed class (baseline duty).
2. `projects/<slug>/PLAN.md` — scope, frontmatter (`phased`,
   `version_bump`), the **Numerical impact** section, the task row.
3. `projects/<slug>/rules.md` if present.
4. **Phased only:** the target phase file, for the exact task definition,
   Exit Criteria, and Prerequisites.
5. The task note (`task-notes/task-N-<slug>.md` or
   `task-notes/phase-XX/task-X.Y-<slug>.md`). `_template.md` files are
   reference material — skip them.
6. ADRs referenced by the task or touching the same area:
   `projects/<slug>/adrs/` and `docs/adrs/`.
7. Upstream phase learnings or prior task notes where they help judge
   this implementation.

For ad-hoc PRs, skip 2–7.

### Step 4: Read the changed files in full

Do **not** review only the diff. For every file with non-trivial changes,
read the full file (or at minimum the surrounding module) so you can
judge integration, breakage, and omissions.

Common "diff omits" patterns in this repo:

- A new public object not exported from the package `__init__.py` (or
  missing from `__all__`) — users cannot reach it.
- A new final state / channel added to one dispatch table but not its
  siblings (photon but not positron; the function map but not the
  documented channel list).
- A `.pyx` / `.pxd` change with no evidence of a rebuild, so the cited
  test results came from the old extension.
- A renamed or removed public object still referenced by `docs/source/`
  — the published Sphinx build breaks without any test failing.
- Package data (`*.dat`, `*.csv`) added under `hazma/` but not registered
  in `[tool.setuptools.package-data]` in `pyproject.toml`, so it is
  missing from the installed wheel.
- A new test file placed where the run that is cited as green never
  collects it — `test/conftest.py`'s `collect_ignore`, or under `test/`
  when only a bare `pytest` was run (`setup.cfg` scopes that to `hazma`).

### Step 5: Evaluate based on your assigned lens

Spend ~90% of your effort on your assigned lens. If you notice a clearly
blocking issue outside it, note it briefly under "Cross-cutting", but do
not hunt outside your area — that is another reviewer's job.

**Baseline duties for EVERY lens** (before your lens-specific work):

- **PR-body claims reproduce.** Cross-check every count, file name, and
  identifier in the body against today's diff.
- **Zero-collection guard.** `pytest` exits 5 on zero tests collected;
  a `-k` filter matching nothing exits 0 with `no tests ran`. Read the
  `N passed` line before trusting any cited green. Note also that a bare
  `pytest` collects a *different* suite from `pytest test`, because
  `setup.cfg`'s `testpaths` is `hazma`.
- **Empirical execution.** When the diff edits a docstring example, a
  README snippet, or claims a user-visible behavior, RUN it and paste the
  output. Static review does not catch a wrong number.
- **Rebuild awareness.** If the diff touches `.pyx` / `.pxd` /
  `setup.py`, confirm the cited results came from a rebuilt tree.

The full baseline duties and the **per-lens FOCUS rubric** for your lens
live in [`review-lenses.md`](../../../docs/agents/review-lenses.md) —
read your section there; it carries the load-bearing detail. Summaries:

- **`default` (A — Generalist):** all dimensions with equal weight, plus
  public-surface hygiene (exports, NumPy docstrings, **units stated**),
  layering per `AGENTS.md`, the scalar/array broadcasting contract, no
  stray debug, and PR title/body per `docs/PR_GUIDELINES.md`.
- **`completeness` (B):** does the diff satisfy each Exit Criterion, with
  a pinning test per bullet, and stay within scope? A test that would
  still pass under a plausible regression is *not* a pinning test.
- **`logic` (C):** adversarial correctness, edge cases, error handling,
  and test validity. Single-symbol mutation test; one adversarial
  boundary case the PR's tests miss (threshold, endpoint, zero/equal
  masses, empty array, scalar-vs-array, NaN, integration across a
  singularity).
- **`doc-consistency` (D):** run
  [`doc-consistency.md`](../../../docs/agents/doc-consistency.md) as your
  review pass and report each check with line-numbered evidence; any
  contradiction is blocking. Includes units-in-docstrings, the Sphinx
  surface, and version-bump + CHANGELOG on closing PRs.
- **`numerics` (E):** **did any published value move?** Run the affected
  public functions before and after on a representative grid and diff.
  A value that moved without acknowledgement in the PR body or CHANGELOG
  is blocking regardless of test status. Then: dimensional analysis
  against `hazma/parameters.py`, limiting cases, floating-point
  stability, integration/interpolation ranges, tolerance justification,
  and performance (vectorization, the Cython boundary, measured claims).

### Step 6: Produce the review

**Verdict rule (apply deterministically):**

- **Any Blocking finding ⇒ REQUEST CHANGES.**
- **Zero Blocking findings ⇒ APPROVE** (list non-blocking suggestions).
- **COMMENT** only for a standalone advisory review — never inside
  `/review-cycle` or `/task-pipeline`.

Severity anchors: **Blocking** = correctness bug, a number that moves
without being acknowledged, CI-breaking change, spec/Exit-Criterion
violation, durable-doc contradiction, or a missing mandated gate.
**Non-blocking** = style, optional improvement, deferred-scope
suggestion. On verification rounds mark each original comment
**RESOLVED**, **PARTIALLY RESOLVED**, or **UNRESOLVED**.

**Focused lens template** (`completeness`, `logic`, `doc-consistency`,
`numerics`):

```text
## Verdict

<APPROVE | REQUEST CHANGES | COMMENT>

One-sentence summary of the overall assessment.

## Issues (<lens> Focus)

### Blocking

<Numbered. File path, line number, what's wrong, what should change.
Empty if none.>

### Non-blocking

<Numbered. Empty if none.>

## Cross-cutting

<Clearly blocking issues outside your lens. Omit the section entirely if
there are none. Keep entries brief.>

## Lens-Specific Observations

<completeness: Exit-Criterion → test map (pass/fail per bullet) + scope.
logic: test-validity and edge-case assessment, naming the regression each
key test would catch. doc-consistency: per-check results with
line-numbered evidence. numerics: the before/after comparison you ran —
function, grid, command, and result — plus stability and performance
assessment.>
```

**Default / generalist template** (`default`):

```text
## Verdict

<APPROVE | REQUEST CHANGES | COMMENT>

One-sentence summary.

## Task Completeness

<Exit Criteria checklist with pass/fail. Verify required artifacts exist
(task note filled in, ADR if needed, phase file / rules.md / PLAN.md
updated if canonical behavior changed). Ad-hoc PRs: assess against the
PR's own Summary.>

## Issues

### Blocking
### Non-blocking

## Numerics

<Did anything the library returns change? What did you check, and how?
"No public code path touched" is a valid answer; silence is not.>

## Scope

<Did the PR stay within task boundaries? Any drift?>

## Testing

<Test quality and coverage relative to the task requirements — including
whether the tests pin values or only shapes.>

## Conventions

<Code style, `projects/<slug>/rules.md` compliance, ADR alignment, and
PR title/body format per `docs/PR_GUIDELINES.md`.>
```

## When a step can't complete

Degrade gracefully — review what you can, surface the gap, never
fabricate the missing piece.

- **Diff too large to read in one pass.** Do not silently skim. Read the
  highest-risk files first (behavior-changing code, then tests, then
  docs), then list the unread surfaces under a **Residual Risks**
  heading.
- **Task spec missing.** Do not hallucinate a spec. Review as ad-hoc and
  flag the missing spec as a finding.
- **A re-run command fails** (no compiler, no built extensions, missing
  dependency). Report it as a finding stating what you could not verify —
  do not invent a result and do not treat reviewer-env noise as a code
  defect.

## Guardrails

- Do not skim. Read every line of the diff (or degrade explicitly).
- Do not approve by default. Start skeptical and let the code convince
  you. Apply the verdict rule mechanically — the same findings must yield
  the same verdict across fan-out reviewers.
- Do not conflate "tests pass" with "correct." In a physics library a
  green suite with a loose tolerance is the most common way a wrong
  number ships.
- Do not request stylistic changes that contradict existing patterns —
  and do not cite `hazma/experimental/` as a pattern; it is outside the
  lint gate and outside the public surface.
- Do not suggest scope expansion beyond the task spec.
- Be specific: file, line, what's wrong, what should change.
- If you are uncertain whether something is a bug, say so explicitly
  rather than silently passing it.
- `_template.md` files are reference material. Do not flag their absence
  from listings or their placeholders as missing content.
