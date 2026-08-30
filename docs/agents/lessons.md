# Review-lessons ledger

A living ledger of review findings that are *class-shaped* — mistakes
that could recur on unrelated tasks, not one-off typos. It is the
feedback loop that otherwise goes missing: the same finding classes recur
across PRs with nothing capturing the lesson in between.

This file holds the **rules**; the **worked examples** behind them — what
the PR did, what review caught, the command that exposed it — live in
[`lessons-examples.md`](lessons-examples.md) under a `###` heading per
class. They were split on 2026-08-21
([ADR-0002](../adrs/ADR-0002-read-phase-learnings-not-closed-task-notes.md)): before
that, every entry carried its examples inline and the ledger ran 660
lines / ~11.6k tokens, read on every task by every implementer and
reviewer. Nothing was dropped — each example moved verbatim.

## Contract

- **Append** (`review-respond`, the lessons step): when a review —
  especially a verification or external round — catches a mistake that is
  class-shaped, add a one-line entry here **and** its worked example under
  a matching `### <class>` heading in
  [`lessons-examples.md`](lessons-examples.md), in the same commit,
  citing the PR(s). If an existing entry already covers the class, add
  the new PR to its citation list here and append the new example there
  rather than duplicating.
- **Read** (`execute-single-task`, before writing code; every reviewer,
  per the review-lenses baseline duties): read this file before working
  and check the diff against each listed class. Open a class's section
  in `lessons-examples.md` only when its one-line rule is not enough to
  act on — the examples are evidence, not the contract.
- **Promote and prune**: when an entry stabilizes, fold it into a
  `docs/agents/` checklist, `AGENTS.md`, or a lint rule, then delete it
  here and its section in `lessons-examples.md`. Keep this file under
  ~30 entries — it is a working set, not an archive. Promoted lessons
  live in their destination; this ledger only holds the classes not yet
  encoded elsewhere.

## Format

One entry per class: `- [class] one-line rule (PR #N, PR #M)`. The
`[class]` tag is a short kebab-case slug so recurrences are easy to match
and merge; the same slug heads the class's section in
`lessons-examples.md`. Keep the rule to the action a reader must take;
the story of how it was learned belongs in the examples file.

Every entry must cite at least one real PR. **Do not add an entry from**
**intuition** — an uncited "lesson" is a guess wearing a citation's
clothes, and it costs every future reviewer the time to check it. If you
believe a class is worth pre-empting but have no PR for it, put it in the
relevant `docs/agents/` checklist as a check, not here as a lesson.

## Ledger

- [ported-file-stale-reference] A file copied in from another repo carries that
  repo's references (workflow paths, CI actions, design docs) and reads as
  authoritative here; grep every ported file for paths and tool names and
  confirm each resolves in *this* repo before shipping it (PR #18).
- [normalized-id-in-path] An id normalized for display ("Phase 2") must not be
  interpolated into a path whose on-disk form is padded (`phase-02/`); name the
  display form and the path form separately and test a value where they differ
  (PR #18).
- [derived-count-not-rederived] Every count in a plan, note or PR body — your
  own artifacts included — is re-derived from source with a stated command at
  write time, and the command is quoted next to the number so the next reader
  re-runs it instead of trusting it. A count describing *your own diff* goes
  stale the moment the diff grows, so give it a row in the count sweep like any
  other — an uncounted count is one nothing re-checks (PR #35, #59, #84).
- [measurement-taken-before-the-task-ended] Re-run every measurement against the
  final tree after your last edit; take both halves of a before/after on the
  same tree and environment; derive breakdowns from the command so the parts
  sum; and never let "both", "each" or "the two" stand in for an enumerated
  count (PR #49, #55, #64, #67, #68, #72).
- [partial-historical-labeling] Label the *section* as historical, not one line
  of it — decide per claim what it is a statement about, date the block, and
  head a task note's §Files Changed / §Numerical impact with the task's own PR
  number (PR #64, #65).
- [flat-vs-sectioned-numbering] A document that restarts numbering per section
  but is cited by a flat index has two schemes and no key; put the mapping in
  the cited document and annotate new citations with both forms (PR #55).
- [artifact-inventory-depends-on-cwd-state] A claim about what a built artifact
  contains must state that it came from a clean tree — filesystem-walking
  packagers ship untracked junk that `.gitignore` hides from `git status` (PR
  #49).
- [wheel-tag-vs-extension-abi] A wheel's tag is per-distribution: claim abi3
  *wheels* only after the last version-specific extension is gone, and verify
  extension-level abi3 via the `.abi3.so` filename instead (PR #35).
- [docstring-section-not-reconciled] Treat a docstring as one artifact — when a
  function's contract changes, reconcile summary, `Parameters`, `Returns`,
  `Notes` and `Raises` together; never add one section and leave the rest
  describing the old behavior (PR #37).
- [unpinned-formatter-version] A version a gate depends on lives in exactly one
  file (`pyproject.toml`'s `[dependency-groups]`, installed with `--group
  lint`); a workflow or doc that restates it is the bug, and `preflight.sh` runs
  whatever is on `PATH` (PR #37).
- [stale-ci-capability-claim] After editing `.github/workflows/`, grep
  `docs/agents/` and the skills for Python versions, tool names and "CI
  does/does not check X" claims and re-derive each from the workflow file (PR
  #33).
- [status-encoding-path-reference] A `docs/followups/` path encodes the item's
  status, so resolving one invalidates every inbound reference; repoint to the
  new path — never strip the segment, `docs/followups/<slug>.md` resolves to
  nothing — and sweep with `rg -oN --no-filename
  'docs/followups/[A-Za-z0-9_./-]*\.md' -g '*.md' . | sort -u` plus `[ -e ]` on
  each hit. Run the sweep repo-wide, not over `--paths`: the reference that
  goes stale is in a file the moving PR never touches. A stale reference
  surviving elsewhere in the tree is evidence the sweep was skipped there,
  never a convention to copy (PRs #44, #81).
- [degenerate-sample-count] An API whose contract includes a statistical error
  estimate validates its sample-count input — type and `>= 2` — at the public
  entry point, or a degenerate count flows to every consumer as a finite value
  with `error=nan` (PR #41).
- [touched-doc-inherits-its-citations] Editing any line of a durable doc puts
  every fact it cites into scope: run `check_doc_citations.py` over the docs you
  touched, pin historical evidence to a commit, cite full paths (never
  basenames) whenever your own diff removes a file, and remember that a ledger
  entry about a bad citation is itself a citation (PR #42, #43, #65, #67, #74).
- [changed-vs-sees-only-commits] A `--changed-vs <ref>` tool diffs committed
  history, so on an uncommitted tree it scans zero files and prints a
  success-shaped line; before believing any gate, confirm it reported a non-zero
  *scope* — docs scanned, tests collected, files linted (PR #42).
- [sqrt-hides-factor-signs] Refactoring `sqrt(P)` into `sqrt(A*B)` silently
  widens the accepted domain where two negative factors multiply; check every
  factor's sign across the whole unphysical domain and pair each sign-changing
  factor with a strictly positive partner so one root per boundary goes NaN (PR
  #43).
- [sibling-copies-of-a-fixed-claim] A repo fact is written in several durable
  docs — and in `docs/agents/` *and* the skills; fix by re-reading the whole
  enclosing artifact, then sweep repo-wide keyed on the *claim* (reflow the file
  to one line first, alternate the synonyms), and when a count changes mid-PR
  re-derive every number derived from it (PR #48, #50, #60, #63).
- [elided-doc-paths] Write every citation as a full repository-relative path —
  `.../foo.py` is unresolvable once the basename is not unique — and never quote
  a bad citation in citation form, because the checker parses the example (PR
  #43, #67).
- [measured-tree-vs-imported-module] A provenance tool must prove that what it
  imports and what it measures are the same tree: assert each imported module's
  `__file__` lies under the measured root and record the resolved path in the
  output (PR #50).
- [exactness-untestable-on-one-platform] A bit-equality assertion that only ever
  runs on the machine that wrote it is untested, not verified; ask which *libm*
  computed each derived quantity (`geomspace`, `logspace` go through `log10` and
  a `power`) and give it a tight derived budget off-platform (PR #52).
- [gate-disabled-stays-green] Removing, narrowing or conditionally skipping a
  check cannot turn CI red, so verify a gate by observing it *execute* —
  collected/passed counts, the env it echoed — and remember GitHub Actions'
  `cond && '' || 'flag'` yields `'flag'` for both outcomes (PR #52, #53).
- [renumbered-list-orphans-its-references] Inserting an item into a numbered
  list falsifies every prose reference to the items after it, wherever they
  live; sweep `rg -n '[Gg]ate [0-9]|[Ss]tep [0-9]|item [0-9]'` across `docs/`,
  `.claude/`, `.codex/` and `docs/followups/`, and prefer naming the target over
  re-pinning a number (PR #56).
- [unrun-workflow-cannot-close-a-criterion] A workflow with no `pull_request`
  trigger is invisible to PR checks; dispatch it (`gh workflow run <file> --ref
  <branch>`, with publishing jobs gated) and paste the job conclusions before
  marking a criterion met (PR #56).
- [marker-count-vs-outcome-count] A count of declaration sites and a count of
  runtime outcomes are different numbers; take outcomes from `pytest -rs` and
  sites from `rg`, state which one you mean, and give both when they differ (PR
  #53).
- [hand-written-population-in-a-derived-check] Derive the population a check
  runs over with the same `rg`/parse step that produces the values — never a
  hand-typed list, one sampled member per group, or the grids your own sweep
  looped over; count what the gate compares (PR #58, #73, #75).
- [bound-parameter-sized-the-allocation] A parameter that *bounds* work is not a
  prediction of it, so never size a buffer by one up front; grow on demand, and
  make the guard test ask for a size no allocator can satisfy so a revert fails
  loudly (PR #60).
- [signed-zero-lost-by-a-derived-formula] A wrapper that does arithmetic on an
  upstream routine's outputs inherits argument cases the upstream guarded —
  `-0.0` above all; sweep both signs of zero at every order and branch, and
  re-apply any guard the upstream applied before its arithmetic (PR #59).
- [platform-scoped-oracle-asserted-globally] A test against a locally compiled
  oracle asserts a property of that build; declare the scope from
  `test/parity/data/manifest.json` (never probe for it), compare bit-for-bit
  there, hold a *measured*, peak-scaled budget elsewhere, and ask whether your
  platform could even have produced a number that would have stopped you (PR
  #61, #63, #68).
- [settling-a-deferral-has-two-sweeps] Settling a deferred decision leaves stale
  text in two disjoint populations — pointers carrying the task id, and
  statements of the old behavior carrying none; sweep the behavior words and
  identifiers as well as the task id (PR #62).
- [numstat-over-a-directory] `git diff --numstat -- '<dir>/'` counts edits to
  surviving files too; measure deletions with `--diff-filter=D` and re-derive
  sub-counts from the corrected total (PR #66).
- [gate-green-is-not-citations-green] `preflight.sh` does not run
  `check_doc_citations.py` — run it yourself whenever the diff touches a `.md`,
  with explicit paths while fixes are uncommitted, because `--changed-vs` takes
  its file list from committed history (PR #67).
- [test-name-claims-an-unmade-assertion] A test that captures a signal and never
  asserts on it advertises a check it does not perform; assert where the signal
  is reachable, say in the docstring where it is not, and rename so the roster
  stops claiming the check (PR #68).
- [stale-group-membership-claim] After editing a `[dependency-groups]` group,
  `rg` its name and member list out of `docs/agents/` and `AGENTS.md` and
  re-derive each enumeration (PR #69, #71).
- [sweep-excluded-the-canonical-directory] Sweep `projects/` *in* and triage by
  file role — a task note is history, a `references/` file is spec unless it
  self-declares a snapshot, the working-memory `## Phases` table is live status
  — and give a canonical per-site table a Status column (PR #70).
- [exemption-wider-than-its-mechanism] Scope a carve-out to the positions its
  mechanism actually reaches — a declared allowlist with the measurement beside
  each row — and pin the allowlist's shape with a test so it cannot drift
  outward (PR #71).
- [sweep-block-written-from-intent] Write the stale-state sweep block last, by
  pasting command output after every prose edit is frozen; "I remember fixing
  that" is a claim to re-check, not evidence. The command you paste must be the
  one that produced the output beside it — a narrowed run written up under a
  repo-wide command reads as a clean sweep and is unreproducible. The same rule
  binds any inventory of what a task is *deferring* — enumerate it from the
  grep, not from memory (PR #71, #78, #83, #84).
- [sign-copied-from-a-defect-description] A delta quoted from a bug report
  carries the *defect's* sign and the repair's is the opposite with the same
  magnitude; restate the endpoints, or say "magnitude", when you copy a figure
  across the fix boundary (PR #72).
- [deadline-bound-to-the-wrong-artifact] A deadline that exists because a
  *resource* is about to disappear binds on capturing that resource, not on
  finishing the work that consumes it; name the artifact and the wave that
  strands it (PR #72).
- [correction-record-names-the-wrong-doc] A note recording *where* a fixed
  claim lived is itself a claim; naming one location for a claim a repo-wide
  `rg` returns several times silently caps the next sweep. Paste the sweep and
  its hit count into the note instead of a location (PR #76).
- [phase-handoff-outlives-its-question] A phase README's `## Handoff to Next
  Task` is written *before* the phase starts, as a list of open worries; the
  phase's first task answers several of them and closing one in the task note
  does not close it in the README. Re-read that block against what you shipped
  before calling the sweep done (PR #79).
