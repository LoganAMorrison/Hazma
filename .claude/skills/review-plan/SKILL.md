---
name: review-plan
description: Stress-test a project plan under projects/<slug>/ before implementation starts, producing a findings-first review across feasibility, numerics, correctness, architecture, and idiomatic Python — read-only, no plan edits.
---

**Role:** Act as a rigorous plan reviewer for a Hazma project plan. The
plan is still being authored or iterated — your job is to surface
problems now, before an implementer starts coding against it. The plan
author wants the plan to fail here, not in PR review. This skill is
read-only: it reviews plans and emits a report; it never edits the plan.

**When to use this skill**

- The user asks to review a project plan (by slug, directory, or
  `PLAN.md` path).
- The user asks whether a plan is ready to hand off to an implementation
  agent.
- A plan is being iterated and the author wants a checkpoint review.

**When NOT to use this skill**

- Reviewing an implementation **PR diff** against a task spec →
  `/review-pr`.
- Reviewing uncommitted **working-tree** changes → `/code-review`.
- Orchestrating a **multi-reviewer loop** on a PR → `/review-cycle`.

"Review the plan for this PR" is ambiguous: a `projects/<slug>/PLAN.md`
is a plan (this skill); a code diff is a PR (`/review-pr`).

## Inputs

Required:

- **project** — a slug, a project directory, or a `PLAN.md` path.
  Normalize to the project directory.

Optional:

- **focus** — a comma-separated subset of review categories
  (`feasibility`, `numerics`, `correctness`, `architecture`,
  `idiomatic-python`, `other`). If omitted, review all categories.

## Workflow

### Step 1: Locate the plan

Verify `projects/<slug>/` exists and `projects/<slug>/PLAN.md` exists and
is non-empty. If either fails, stop and ask the user to point at a real
plan. This skill reviews plans, not empty scaffolds.

Decide the ambiguous cases here:

- **`focus` names an unknown category** — stop and ask; do not silently
  drop it.
- **A plan-cited context file is missing** (a `references/*.md`, a linked
  ADR) — note-and-proceed: record it as a finding, do not abort.
- **`phased:` frontmatter disagrees with the filesystem** — the
  filesystem wins for the reading order in Step 2; flag the mismatch as a
  finding.

### Step 2: Load the plan and its context

Read, in this order:

1. `projects/<slug>/PLAN.md` — frontmatter (`status`, `phased`,
   `version_bump`), goal, scope, **Numerical impact**, task details, exit
   criteria, anticipated ADRs.
2. `projects/<slug>/rules.md` if present.
3. `projects/<slug>/phases/*.md` if phased.
4. `projects/<slug>/references/*.md` if present.
5. `projects/<slug>/adrs/*.md` if present.
6. `projects/<slug>/task-notes/README.md` for live-state context (but do
   not review this file — it is working memory, not the plan).
7. [`AGENTS.md`](../../../AGENTS.md),
   [`docs/workflow.md`](../../../docs/workflow.md),
   [`docs/versioning.md`](../../../docs/versioning.md),
   [`docs/PR_GUIDELINES.md`](../../../docs/PR_GUIDELINES.md), and any
   specific files the plan cites.
8. [`docs/agents/lessons.md`](../../../docs/agents/lessons.md) — the
   recurring review-defect classes. A plan can bake in a class before
   code exists.

Ignore `_template.md` files. They are reference artifacts, never missing
content.

### Step 3: Verify grounded facts

**Verify the file paths, line numbers, function names, and module
references the plan makes** with `Grep` / `Read` / `Glob` — a plan that
names `hazma/spectra/_photon/_muon.py:42` claims something relevant lives
there; don't take its word. Two disciplines make this sound rather than
performative:

- **Distinguish should-exist-now from will-be-created.** A plan
  legitimately names files it will *create*; those SHOULD fail `Glob`,
  and emitting a Blocker for them is a false positive. For a
  to-be-created path, verify instead that the **parent package exists**
  and the **naming matches conventions** (snake_case module, right
  subpackage, leading underscore if private). Only a path the plan treats
  as pre-existing evidence blocks when absent.
- **Premise-check rule.** When the plan (or a followup it builds on)
  asserts "X exists / is used / returns Y", `rg` the *exact* symbol and
  run the *exact* command — do not paraphrase or trust the claim. A plan
  built on a function that was renamed two releases ago wastes an entire
  implementation pass.

**Bound the effort.** In a large plan with a broken-out `references/`
tree, verify the **load-bearing** facts first (paths/APIs a task's
correctness depends on, premises other tasks depend on), then sample the
rest — mark sampled rows as `Sampled` in the audit.

**Run the commands the plan tells implementers to run.** For every
command cited as an exit criterion or a "run this to verify" step,
actually execute it. A command that exits **green while doing nothing** is
a **Blocker**: `pytest` exits 5 on zero collected, and a `-k` filter
matching nothing exits 0 with `no tests ran`. A bare `pytest` is the full
suite (`pyproject.toml`'s `testpaths = ["hazma", "test"]`), so the
opposite trap now applies: a plan citing it as a cheap per-step check has
budgeted a multi-minute run.

Every fact you check becomes a Grounded-facts audit row — and every
**Verified: Yes** must cite its evidence (the matching `file:line`, grep
hit, or command + output). A "Yes" with no evidence is indistinguishable
from a fabricated one.

### Step 4: Evaluate each review category

For each category, produce a verdict (**Pass**, **Needs work**, or
**Blocker**) plus specific, cited findings. Every finding names the file,
section, and (where applicable) line.

If the user supplied a `focus` list, evaluate only those categories — but
still verify grounded facts (Step 3) for the whole plan.

#### 1. Feasibility — can an agent actually implement this?

- **One-PR sizing heuristic.** A task sized for a single PR touches ≲1
  subsystem and carries ≲2 architectural decisions. More than that ⇒ the
  task must be split; flag an oversized task as Needs work, or a Blocker
  if it also blocks downstream tasks.
- **No open TBDs with dependents.** Any unresolved "TBD" / "pick one" /
  "decide at implementation time" on a task a downstream task depends on
  is a **Blocker** — it will stall the dependent. A TBD on a leaf task is
  at most Needs work.
- Are concrete file paths, function names, and API sketches provided
  where non-obvious?
- Is each exit criterion testable — a command to run, a value to pin, a
  measurement to record — and does it actually run work (Step 3)?
- Are task dependencies explicit and acyclic?
- Are there handwaves ("fix the normalization", "integrate properly")?
  They become rework vectors — call them out.
- Is anything ambiguous enough that two competent implementers would
  produce materially different diffs?

#### 2. Numerics — this is a physics library

The highest-value category. A plan that is silent about numbers will
produce a PR that moves them silently.

- **Does the plan state its numerical impact?** `PLAN.md` has a
  **Numerical impact** section. `Unknown — Task 1 measures it` is a valid
  plan-time answer; an *absent* section, or one that says "none" for a
  plan that clearly touches a spectrum, is a **Blocker**.
- **Does `version_bump:` match?** A plan that corrects a published number
  is `minor`, not `patch`, per
  [`versioning.md`](../../../docs/versioning.md). A mismatch is Needs
  work at minimum.
- **Are units and normalization pinned?** "Returns the spectrum" is not a
  spec. Which spectrum — `dN/dE` or `E dN/dE`? Per annihilation or per
  decay? Which frame? Which energy units? If the plan does not say, two
  implementers will disagree and only one will be right.
- **Is there an oracle for each new number?** Every task that produces a
  physical quantity should name what it will be checked against: an
  analytic limit, a published value (with a citation), an independent
  implementation, or a stored regression array. "Add tests" is not an
  oracle.
- **Are the numerically dangerous spots identified?** Threshold and
  endpoint behavior, catastrophic cancellation, integrable singularities,
  interpolation outside the tabulated range, `sqrt` of a
  rounding-negative quantity. A plan touching these without naming them
  is Needs work.
- **Are tolerances discussed?** A plan whose only stated gate is
  "tests pass" gives the implementer license to pick whatever `rtol`
  makes it green.

#### 3. Correctness — beyond the numbers

- For each task, enumerate failure modes. Does the plan close them?
- Edge cases: empty arrays, scalar-vs-array input, zero and equal masses,
  the massless limit, negative or NaN input, arrays with a single
  element.
- Ambiguity in "correct": "return the spectrum for the final state" —
  including or excluding FSR? Summed over charge states or per state?
  The plan must pin this down.
- How does the change interact with `hazma/theory/`, the model packages,
  and the limit-setting machinery downstream?
- New-code correctness shapes the plan should already defend: adding to a
  dispatch table requires fanning out to every sibling lookup, `__all__`,
  and test; a constant becoming a user argument needs a validity guard;
  an invariant must hold at every public entry point.

#### 4. Architecture — does it fit the repo?

- **Layering.** The dependency direction in
  [`AGENTS.md`](../../../AGENTS.md) must hold: the Rust kernels import
  nothing from pure-Python layers; models depend on `theory`, not the
  reverse; analysis consumes models. Flag any task that inverts it.
- **Public vs private.** Does the plan put new code in the right place —
  a leading-underscore package for implementation, the public package for
  the user-facing entry point — and export it where users will look?
- Does the plan reuse existing abstractions (the `Theory` interface, the
  spectra dispatch, `phase_space`, the boost machinery) or reinvent them?
- **Rust vs NumPy.** If the plan proposes a new kernel in `rust/`, does
  it justify why vectorized NumPy is insufficient? Compiled code adds a
  rebuild step for every contributor and a platform-specific failure
  mode — it needs a reason.
- **Data files.** maturin ships the whole `hazma/` tree, so new `*.dat` /
  `*.csv` under `hazma/` need no packaging entry. Does the plan record
  the data's provenance?
- ADR placement: project-scoped vs repo-wide per
  [`workflow.md`](../../../docs/workflow.md). Unit and normalization
  conventions are the classic repo-wide tier here.
- Anything load-bearing (an invariant, a convention, a data provenance)
  that should be documented in the plan but isn't?

#### 5. Idiomatic Python — a Python dev should feel at home

Apply the `10x-standards:python-standards` skill and the
[`AGENTS.md`](../../../AGENTS.md) conventions rather than re-deriving
them. The hazma-specific points:

- Naming: modules/functions `snake_case`, classes `UpperCamel`,
  constants `SCREAMING_SNAKE`. Physics symbols stay in docstrings, not
  identifiers.
- Type annotations on public functions; NumPy-style docstrings with
  `Parameters` / `Returns` and **units for every physical quantity**.
- Arrays in, arrays out — the broadcasting contract.
- Error handling: which errors from `hazma/hazma_errors.py`, raised when?
  No bare `except`.
- Does the plan smuggle in `print()`, `breakpoint()`, a mutable default
  argument, or a module-level side effect?
- Formatting is black + isort; the plan should not propose fighting them.

#### 6. Additional dimensions (the "Other" row)

- **Test strategy.** Every task has a test plan naming real pytest
  targets, and it names the suite it means — `pytest test` and a bare
  `pytest` collect different sets.
- **Scope discipline.** "In scope" and "Out of scope" don't overlap.
  Nothing implicit is smuggled in. The out-of-scope list is honest about
  what the author *was* tempted to include.
- **Regression risk.** Does the change risk moving an existing spectrum,
  limit, or relic-density result? Is there a battery that would catch it?
- **Performance.** Python loops over arrays, repeated recomputation, or
  per-element crossings of the Python/Rust boundary — does the plan know
  where
  its cost is, and does it commit to a measurement rather than a claim?
- **Dep hygiene.** New third-party dependencies justified? They land in
  `pyproject.toml` and every downstream user pays for them.
- **Docs surface.** Does a new public object need a `docs/source/*.rst`
  entry? Does a rename break an existing `automodule` directive?
- **Commit / PR plan.** Each task maps cleanly to one PR with a
  Conventional Commits title and a valid ≤10-char scope.
- **Task-notes hygiene.** Live state lives in `task-notes/README.md`, not
  `PLAN.md`.

### Step 5: Emit the report — findings first, verdict last

```md
# Plan Review: <project slug>

## Findings

<Severity-ordered numbered list. Each entry is tagged [P1] (Blocker),
[P2] (Needs work), or [P3] (Suggestion); cites the exact plan location
(e.g. `PLAN.md §Task 2 "Scope / implementation notes"` or
`references/foo.md:18`); states the problem in one sentence; and
proposes a concrete fix or the decision that must be made. Order P1s
first.>

## Summary

| Category         | Verdict                     | Top finding (one line, cited) |
|------------------|-----------------------------|-------------------------------|
| Feasibility      | Pass / Needs work / Blocker | ...                           |
| Numerics         | Pass / Needs work / Blocker | ...                           |
| Correctness      | Pass / Needs work / Blocker | ...                           |
| Architecture     | Pass / Needs work / Blocker | ...                           |
| Idiomatic Python | Pass / Needs work / Blocker | ...                           |
| Other            | Pass / Needs work / Blocker | ...                           |

## Grounded-facts audit

| Claim (plan location) | Verified? | Evidence |
|-----------------------|-----------|----------|
| ...                   | Yes / No / Partial / Sampled | grep hit, file:line, or command output |

## What the plan does well

<2–5 bullets. Not filler — concrete strengths (good scoping, clean exit
criteria, a real oracle for each number, right ADR placement), each
cited.>

## Verdict

<APPROVE | APPROVE WITH CHANGES | REQUEST MAJOR REVISION>
<Then 2–3 sentences of why. Keep it brief — the findings are the
primary output.>
```

Severity maps to the verdict: any **[P1]** ⇒ REQUEST MAJOR REVISION;
[P2]s but no [P1] ⇒ APPROVE WITH CHANGES; only [P3]s ⇒ APPROVE. The
blocking anchors are those in
[`review-lenses.md`](../../../docs/agents/review-lenses.md), plus three
plan-specific blockers: green-but-noop verification commands (Step 3),
open TBDs with dependents, and a missing or obviously-wrong **Numerical
impact** section.

**No-findings shape.** If nothing rises above a nit, drop the Summary and
Grounded-facts tables: emit `## Findings` with `No blocking or needs-work
findings.`, replace the two tables with a `## Residual Risks` list of
load-bearing assumptions you could not fully verify (one line each), and
close with `## Verdict` / `APPROVE` and a sentence.

## Review posture

- **Cite everything.** A finding without a file and section is noise;
  re-verify your own line citations are current before shipping.
- **Be specific.** "Task 3 is vague" is useless; "Task 3 says 'boost the
  rest-frame spectrum' but does not say whether the boost integral is
  taken over the lab-frame or rest-frame energy grid;
  `spectra/boost.py:88` assumes the former" is useful.
- **Propose fixes.** A good finding tells the author what text would
  resolve it, or what decision is missing.
- **Don't pad.** If a category is clean, say "Pass" and move on.
- **Flag dead weight** — empty `rules.md`, stale references files,
  anticipated ADRs that should already be written. Never flag
  `_template.md` files.
- **Don't re-write the plan.** Review it; point at missing decisions;
  don't ghostwrite them.
- **Don't ratify handwaves.** "Decide at implementation time" for
  something that blocks downstream work is a blocker, not a deferred
  detail. This applies double to units and normalization.

## Output

Emit only the report. This skill is read-only: do not modify the plan,
task notes, or any file in the project — the author drives revisions.
