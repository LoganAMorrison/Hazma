# Review-lessons ledger

A living ledger of review findings that are *class-shaped* — mistakes
that could recur on unrelated tasks, not one-off typos. It is the
feedback loop that otherwise goes missing: the same finding classes recur
across PRs with nothing capturing the lesson in between.

## Contract

- **Append** (`review-respond`, the lessons step): when a review —
  especially a verification or external round — catches a mistake that is
  class-shaped, add a one-line entry in the same commit, citing the
  PR(s). If an existing entry already covers the class, add the new PR to
  its citation list rather than duplicating.
- **Read** (`execute-single-task`, before writing code; every reviewer,
  per the review-lenses baseline duties): read this file before working
  and check the diff against each listed class.
- **Promote and prune**: when an entry stabilizes, fold it into a
  `docs/agents/` checklist, `AGENTS.md`, or a lint rule, then delete it
  here. Keep this file under ~30 entries — it is a working set, not an
  archive. Promoted lessons live in their destination; this ledger only
  holds the classes not yet encoded elsewhere.

## Format

One line per class: `- [class] one-line rule (PR #N, PR #M)`. The
`[class]` tag is a short kebab-case slug so recurrences are easy to match
and merge.

Every entry must cite at least one real PR. **Do not add an entry from
intuition** — an uncited "lesson" is a guess wearing a citation's
clothes, and it costs every future reviewer the time to check it. If you
believe a class is worth pre-empting but have no PR for it, put it in the
relevant `docs/agents/` checklist as a check, not here as a lesson.

## Ledger

- [ported-file-stale-reference] A file copied in from another repo carries
  that repo's references — workflow paths, CI actions, internal design docs
  — and they read as authoritative here while pointing at infrastructure
  that does not exist. Grep every ported file for paths and tool names and
  confirm each resolves in *this* repo before shipping it (PR #18:
  `check_pr_title.py` claimed to mirror `.github/workflows/pr_linter.yaml`,
  which exists only upstream).
- [normalized-id-in-path] An id that is normalized for display ("Phase 2")
  must not be interpolated into a filesystem path when the on-disk name is
  the un-normalized form (`phase-02/`). The two agree for most values and
  diverge exactly at the padded ones, so a spot check passes and the bug
  ships. When a value has both a display form and a path form, name them
  separately and test a value where they differ (PR #18:
  `resolve_phase.py`'s kickoff prompt sent agents to `task-notes/phase-2/`).
- [derived-count-not-rederived] A count that appears in a plan (extensions,
  entry points, files) must be re-derived from source with a stated command
  at write time, not carried over from analysis prose — three counts in one
  plan drifted (32→"19" survivors vs 20 actual; "43" corpus entry points vs
  41 consumed; "44 .pyx/.pxd" conflating 44 .pyx + 33 .pxd), and each read
  as authoritative until review recounted (PR #35).
- [wheel-tag-vs-extension-abi] A wheel's tag is per-distribution, not
  per-extension: while any version-specific extension (e.g. Cython) remains
  in the package, wheels stay CPython-tagged no matter how many extensions
  use the limited API — claim abi3 *wheels* only after the last
  version-specific extension is gone, and verify extension-level abi3 via
  the `.abi3.so` filename instead (PR #35).
- [docstring-section-not-reconciled] Adding a section to an existing
  docstring (a `Raises`, a new parameter, a changed unit) without re-reading
  the rest of it ships a self-contradictory doc — the new text is right and
  the old sections still describe the previous behavior, which reads as
  authoritative. Treat a docstring as one artifact: when a function's
  contract changes, reconcile summary, `Parameters`, `Returns`, `Notes`, and
  `Raises` together (PR #37: a `Raises: Always` was added to a removal stub
  whose `Returns`/`Notes` still promised a spectrum — and carried two older
  defects, "gamma ray" prose in the positron module and a
  `hazma.phase_space_generator.rambo` path that never existed).
- [unpinned-formatter-version] A formatter gate is only meaningful at the
  version CI runs, and `preflight.sh` invokes whatever `black` is on `PATH`.
  A pin written down in two places will drift, and the drift is invisible:
  `pyproject.toml`'s dev extra (`<27.0`) once admitted a newer major than
  the Lint job's own literal pin (`<25.0`), so a locally-clean
  `black --check` still failed CI on lines nobody meant to touch, and a
  locally-red one could be a phantom (PR #37: black 26.5.1 reformatted a
  `warnings.warn` into a style black 24.10.0 rejects; the same skew had
  earlier been recorded as "preflight is red on master", which it was not).
  Fixed by deleting the duplicate rather than syncing it — the pins live
  only in `pyproject.toml`'s `[dependency-groups]`, and both CI and you
  install `--group lint`. The lesson generalizes past black: a version a
  gate depends on belongs in exactly one file, and a workflow that
  re-states it is the bug.
- [stale-ci-capability-claim] A change to `.github/workflows/` silently
  falsifies every prose description of what CI does. Those descriptions
  live in `docs/agents/` and the skills, not next to the workflow, so
  nothing fails when they rot — and agents then skip a gate CI no longer
  covers, or run one it now does. After editing a workflow, grep the
  Python versions, tool names, and "CI does/does not check X" claims out
  of `docs/agents/` and `.claude/skills/` and re-derive each from the
  workflow file (PR #33: the matrix had gone 3.10–3.12 → 3.10–3.14 and
  `black --check` had been re-enabled, while three docs still said
  otherwise).
- [degenerate-sample-count] An API whose contract includes a statistical
  error estimate must validate its sample-count input at the public entry
  point: a `ddof=1` sample deviation is undefined below two samples, and a
  degenerate count otherwise flows to *every* internal consumer — the
  per-energy numerator and the Monte-Carlo non-radiative denominator alike —
  as a finite value with `error=nan` plus a warning nobody reads. Validate
  the type too: a non-integral count otherwise dies deep inside NumPy with
  a message that names neither the argument nor the fix (PR #41).
- [touched-doc-inherits-its-citations] Editing *any* line of a durable doc
  puts **every** fact that doc cites into your PR's scope, however
  mechanical your own edit was. A diff that only stripped markdownlint
  pragmas from `references/cython-inventory.md` inherited a
  `boost.pyx:427,447,456` citation into a file a later purge had cut from
  461 to 241 lines — stale since `e94fb21`, caught by review, not by the
  author. Run `check_doc_citations.py` over the docs you touched, not the
  ones you wrote, and pin historical evidence to a commit
  (``lines 427, 447, 456 as of `e94fb21^` ``) so a later deletion cannot
  falsify it (PR #42).
- [changed-vs-sees-only-commits] A `--changed-vs <ref>` tool diffs
  *committed* history, so running it on an uncommitted tree scans zero
  files and prints a success-shaped line — `check_doc_citations.py
  --changed-vs origin/master` answered "no docs to check" mid-session and
  was read as a pass; the real run after committing failed. Same family as
  `markdownlint` exiting 0 on a glob that matches nothing. For any gate,
  confirm it reported a non-zero *scope* (docs scanned, tests collected,
  files linted) before believing its verdict (PR #42).
