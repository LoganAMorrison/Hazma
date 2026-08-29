# Review lenses

The canonical reviewer roster, lens rubrics, selection rules, verdict
rule, and baseline duties for every PR review in this repo. Applies to
Claude Code, Codex, and any coding agent.

Skills point here instead of restating the roster. `review-cycle` selects
reviewers from this table; `review-pr` runs a single lens from it;
`task-pipeline` delegates its review phase to `review-cycle`. When this
file conflicts with a skill's inline prose, this file plus
[`AGENTS.md`](../../AGENTS.md) win. Edit the roster or a rubric here once
— never fork a copy into a skill.

## Reviewer roster

Each reviewer's identity — letter, role, agent-specific capability target, and
`--lens` flag — is fixed below. A single prompt template spawns the selected
subset, substituting these fields. The Model column applies to Claude; Codex
inherits the active model.

|ID|Role|Model|Effort|Codex|Lens|
|--|----|-----|------|-----|----|
|A|Generalist|`sonnet`|medium|inherit|none|
|B|Completeness|`sonnet`|medium|inherit|`completeness`|
|C|Logic|`sonnet`|high|inherit|`logic`|
|D|Doc consistency|`sonnet`|medium|inherit|`doc-consistency`|
|E|Numerics|`opus`|high|inherit|`numerics`|

- **A — Generalist:** Use almost always as the broad safety net for
  completeness, correctness, numerics, and conventions.
- **B — Completeness:** Use for explicit Exit / Acceptance Criteria,
  multi-bullet objectives, or a phase-file gate.
- **C — Logic:** Use for runtime behavior, control flow, error handling,
  broadcasting, interpolation, integration bounds, or non-trivial branching.
  Pure docs, fixtures, and renames usually do not need C.
- **D — Doc Consistency & Canonical Contract:** Use when a diff touches
  durable docs, a public docstring, or claims that must reproduce.
- **E — Numerics & Performance:** Use whenever a change can move a number or
  touches a hot loop or Cython boundary. This is the highest-value physics
  lens.

**Agent models and effort.** The roster is calibrated for the current
generation:

- **A–D default to `sonnet`** (Sonnet 5). They are pattern-matching and
  cross-referencing passes; Sonnet handles them at full quality.
- **E defaults to `opus`** (Opus 5). Deciding whether a spectrum is
  *right* — not merely finite — is the hardest judgment in this repo and
  the one worth the strongest model.
- Claude's orchestrator MAY upgrade **A** to `opus` for high-risk or
  project-closing PRs, and MAY downgrade **D** to `haiku` for a docs-only
  diff with no numeric claims. Claude implementation and review-response
  agents are `opus`.
- Codex reviewers inherit the active Codex model unless the caller has
  authorized an override. Use the table's effort target, and raise E to
  `xhigh` for a changed published spectrum or limit. Codex implementation
  and review-response agents use the active model with high effort.

**Reasoning effort.** Pass the `Effort` column when the runner supports
it. `high` for C and E, `medium` elsewhere; raise E to `xhigh` on a diff
that changes a published spectrum or a limit. Do not raise effort as a
substitute for giving the reviewer the right context.

## Selection rules

Selection happens **once per pipeline run**, at the start of the review
phase, and the chosen set is used for every round (convergence
comparisons assume a stable set — do not add or drop reviewers between
rounds).

- **Default-include A and D.** The set must be non-empty and must include
  at least one reviewer that broadly covers correctness for the touched
  area — in practice **A**. Drop A only when another lens fully covers
  the diff (e.g. a doc-only PR where D suffices) and you can articulate
  why.
- **Include B** when the task has explicit Exit / Acceptance Criteria, a
  phase-file gate, or multi-bullet objectives.
- **Include C** when the diff changes runtime behavior.
- **Always include E when the diff can move a number** — any change under
  `hazma/` that is not purely a rename, a docstring, or a type
  annotation. When in doubt, include E; the cost of a missed numerical
  regression in a published library is not symmetric with the cost of one
  extra reviewer.
- **Always include D on project-closing PRs** — D verifies the version
  bump and the CHANGELOG entry (see the D rubric).
- **Bias toward inclusion when uncertain.** An extra reviewer costs one
  parallel subagent; a missed lens costs a shipped regression. But do not
  select a reviewer just to fill the slate — a lens with no plausible
  findings dilutes the round with low-signal "LGTM" comments.
- **Project-closing PRs: prefer the full roster.** Parallel independent
  reviews of one PR produce largely non-overlapping findings, so the
  redundancy is worth it on the highest-stakes diffs.

Record the chosen set as `SELECTED_REVIEWERS` (e.g. `{A, C, D, E}`) with
a one-sentence justification per chosen and per omitted reviewer; surface
it in the first round's PR comment for the audit trail.

## Verdict rule

Canonical, for every reviewer:

- **Any Blocking finding ⇒ REQUEST CHANGES.**
- **Zero Blocking findings ⇒ APPROVE** (list non-blocking suggestions, if
  any).
- **COMMENT is not used inside orchestrated loops** (`review-cycle`,
  `task-pipeline`) — only for standalone advisory reviews.

Severity anchors:

- **Blocking** = correctness bug, **a number that moves without being
  acknowledged**, CI-breaking change, spec / Exit-Criterion violation,
  durable-doc contradiction, or a missing mandated gate.
- **Non-blocking** = style, optional improvement, or a deferred-scope
  suggestion.

Re-review vocabulary (verification rounds): mark each original comment
**RESOLVED**, **PARTIALLY RESOLVED**, or **UNRESOLVED**.

## Baseline duties (every reviewer)

These apply regardless of lens.

- **Fresh-eyes PR-head recipe.** Fetch the PR ref fresh each round —
  delete any stale ref, then
  `git fetch origin pull/<N>/head:refs/remotes/origin/pr/<N>`. Verify the
  head SHA against `gh pr view`. Never test against the ambient checkout.
  Treat `gh pr diff` output as possibly truncated — fall back to the
  fetched ref.
- **Zero-collection guard.** `pytest` exits 5 on zero tests collected,
  and a `-k` filter matching nothing exits 0 with `no tests ran`. Assert
  a real `N passed` count before trusting a cited green.
- **Empirical execution.** When the diff adds or edits a docstring
  example, a README snippet, or claims a user-visible behavior, RUN it
  and paste the output. Static review does not catch a wrong number.
- **Rebuild awareness.** If the diff touches `rust/` or the build
  configuration in `pyproject.toml`, confirm the cited test results came
  from a tree that was
  rebuilt. A green run against a stale extension proves nothing — and on
  the Rust side `cargo test` is not a rebuild, since it works out of
  `rust/target/` while Python imports the `hazma/_core.abi3.so` that only
  `pip install -e .` refreshes.
- **Read [`lessons.md`](lessons.md) first**, then check the diff against
  each listed recurring class.
- **Apply the verdict rule** (above), including the re-review vocabulary
  on verification rounds.
- **New-code correctness shapes** — flag when the diff:
  - adds a branch to a dispatch table (a final-state → spectrum-function
    map, a channel list) without fanning it out to every sibling lookup,
    `__all__`, and test;
  - turns a hard-coded constant into a user-supplied argument without a
    validity guard;
  - defends an invariant on only one entry point rather than every public
    one;
  - writes to a cache or a module-level table before validating.

## Per-lens FOCUS rubrics

### A — Generalist

Evaluate all dimensions — completeness, correctness, numerics, and
conventions — with equal weight. Specifically confirm:

1. **Public-surface hygiene.** New public objects are exported where
   users will look for them (`hazma/spectra/__init__.py`, the model
   package `__init__`), documented with NumPy-style docstrings, and have
   **units stated** for every physical quantity.
2. **Layering.** The dependency direction in
   [`AGENTS.md`](../../AGENTS.md) holds — Cython kernels import nothing
   from pure-Python layers; models depend on `theory`, not the reverse.
3. **Broadcasting contract.** A new spectrum-shaped function accepts both
   a scalar and an array energy and has a test for both.
4. **No stray debug.** No `breakpoint()`, `pdb`, or `print()` added to
   library code.
5. **PR title/body** conform to
   [`../PR_GUIDELINES.md`](../PR_GUIDELINES.md), and project work carries
   an accurate `## Project` section.

### B — Completeness

Focus on task-spec satisfaction, Exit Criteria, scope discipline, and
test coverage. **Build an explicit Exit-Criterion → test map:** copy each
Exit / Acceptance Criterion bullet from the task spec verbatim, then cite
the specific `pytest` node id, fixture, or regression array that pins it.
Bullets with no pinning test are **blocking**.

A cited test that **would still pass under a plausible regression**
counts as *not* pinned — e.g. the assertion is `result > 0` and the bug
returns a wrong positive number; the tolerance is loose enough to absorb
the defect; only one final state of a multi-channel change is covered;
the test asserts shape but never a value.

Also flag any behavior delivered outside the task spec (scope creep) or
any spec clause the diff narrows without amending the PLAN / phase file.

### C — Logic

Act adversarially — correctness bugs, edge cases, error handling, and
**test validity**.

- **Single-symbol mutation test.** For the most behavior-changing new or
  modified test in the diff, name the specific regression in the
  production code that would flip its assertions. If you cannot name one
  ("any error path", "any wrong value", "I'm not sure"), raise the test
  as **blocking**. Mentally invert one comparison, drop a factor of 2,
  swap two arguments, or return the un-normalized value, and ask whether
  the test catches it.
- **Adversarial boundary case.** For the most behavior-changing function
  in the diff, construct one boundary case the PR's own tests do not
  cover. In this repo the productive ones are: energy exactly at
  threshold; energy above the kinematic endpoint; zero or equal masses;
  the massless limit; an empty array; a scalar where an array was
  assumed (and vice versa); a negative or NaN input; and integration
  limits that cross a singularity. If you can describe a concrete failing
  case, raise it as **blocking** with the reproducer.
- **Error handling.** Missing propagation, a bare `except`, a swallowed
  `RuntimeWarning` from NumPy, or a silent `nan_to_num` that hides a real
  divergence.

### D — Doc consistency & canonical contract

Run [`doc-consistency.md`](doc-consistency.md) as the reviewer pass —
that file is the canonical checklist (count / command / identifier
reproduction, canonical-contract gate evidence, intra-document coherence,
four-corner cross-document consistency, version-bump and CHANGELOG
consistency on closing PRs, the docstring sweep, follow-up file
mechanics, PR title / body sanity, and the stale-sibling sweep). Report
each check's result with line-numbered evidence; any contradiction is
**blocking**.

Two repo-specific additions:

- **Units in docstrings** must match what the code returns. A docstring
  saying `MeV^-1` over a function that returns `GeV^-1` is blocking.
- **Sphinx docs** under `docs/source/` that reference a renamed or
  removed public object must be updated in the same PR.

### E — Numerics & performance

The highest-value lens in this repo. Focus first on **whether the numbers
are right**, then on how fast they are computed.

**Numerical correctness:**

- **Did any published value move?** Identify every public function the
  diff can reach. For each, run it before and after on a representative
  grid and diff the arrays. If a value moved and neither the PR body nor
  `CHANGELOG.md` says so, that is **blocking** — regardless of whether
  tests pass. A loose tolerance absorbing a real shift is exactly the
  failure this lens exists to catch.
- **Dimensional analysis.** Check the units of every new expression
  against the docstring and against `hazma/parameters.py`. A missing
  factor of `c`, `ħ`, or a MeV/GeV mix-up is the classic defect here.
- **Limits and special cases.** Does the new expression reduce correctly
  in the massless limit, the non-relativistic limit, or at threshold? Ask
  for the check if the PR does not show it.
- **Floating-point stability.** Catastrophic cancellation in a difference
  of nearly equal terms; `exp` of a large negative argument; division by
  a quantity that vanishes at the kinematic edge; `sqrt` of a value that
  can go slightly negative from rounding. Flag `np.errstate` suppression
  and `nan_to_num` that hide rather than handle these.
- **Integration and interpolation.** Are the limits right and justified?
  Is the quadrature appropriate for an integrand with an endpoint
  singularity? Does an interpolator extrapolate silently outside its
  tabulated range?
- **Tolerances.** Every `isclose` / `allclose` tolerance in a new test
  should have a stated reason. An unexplained `rtol=1e-2` is a
  non-blocking finding at minimum; one that is loose enough to hide the
  bug the test claims to guard is blocking.

**Performance:**

- Weight severity by how hot the path is: a per-energy-point inner loop
  over a large grid matters; one-time setup does not.
- Python-level loops over NumPy arrays where a vectorized expression
  exists; repeated recomputation of an invariant inside a loop; array
  copies where a view suffices; `np.append` in a loop.
- **The Cython boundary.** Crossing from Python into a `.pyx` kernel once
  per array element is the expensive pattern; crossing once per array is
  the cheap one.
- **Verify claimed properties.** When the diff claims a speedup, ask for
  the measurement — a command and two numbers, on a stated grid size.
  "Should be faster" is not a measurement, and an unmeasured perf claim
  in a docstring or PR body is a finding.
