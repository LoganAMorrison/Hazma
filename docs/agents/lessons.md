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
  version CI runs. `preflight.sh` invokes whatever `black` is on `PATH`, and
  `pyproject.toml`'s dev extra (`<27.0`) admits a newer major than the Lint
  job pins (`<25.0`) — so a locally-clean `black --check` can still fail CI,
  on lines you never meant to touch, and a locally-red one can be a phantom.
  Check the workflow's pin before trusting or reporting any black/ruff/isort
  result, and reproduce a lint failure with CI's exact version before
  "fixing" it (PR #37: black 26.5.1 reformatted a `warnings.warn` into a
  style black 24.10.0 rejects; the same skew had earlier been recorded as
  "preflight is red on master", which CI's black says it is not).
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
