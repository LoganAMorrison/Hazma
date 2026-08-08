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
- [measurement-taken-before-the-task-ended] A number measured *correctly*,
  with a real command, still goes stale if the task's own later output
  changes what the command measures. Re-run every measurement against the
  final tree before handing off, and measure both halves of a before/after
  pair on the *same* tree — taking "before" early and "after" late compares
  two different trees while looking rigorous. Watch for the self-referential
  case in particular (PR #49: an sdist file count of `498 → 397` was taken
  right after the `MANIFEST.in` fix, but the follow-up documenting the sdist
  payload then landed under the un-pruned `docs/` and became part of that
  payload; the true figures on the final tree were `501 → 398`). Distinct
  from [derived-count-not-rederived] above, which is about numbers never
  derived at all; this one is about a derivation that expired.
- [artifact-inventory-depends-on-cwd-state] Any claim about what a built
  artifact *contains* must state that it came from a clean tree. A
  filesystem-walking packager (setuptools' sdist, Docker build context)
  ships untracked junk that `.gitignore` hides from `git status`, so the
  same commit yields different artifacts depending on what the working
  directory has accumulated (PR #49: an identical tree produced 400 files
  dirty and 398 clean, the difference being `.pytest_cache/README.md`
  swept in by `global-include *.md` despite `.gitignore:526`).
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
- [status-encoding-path-reference] A `docs/followups/` path encodes the
  item's status in a directory segment, so resolving one invalidates every
  inbound reference. Stripping the segment to make the reference
  status-neutral is not a fix — `docs/followups/<slug>.md` resolves to no
  file at all, which is worse than the stale-but-real path it replaced.
  Repoint to the new path and say the old one in prose if the history
  matters. `check_doc_citations.py` will not catch this: it bounds-checks
  `.py`/`.pyx`/`.pxd` line citations only, so nothing verifies that a
  markdown path reference resolves. Sweep with
  `rg -oN --no-filename 'docs/followups/[A-Za-z0-9_./-]*\.md' -g '*.md' .
  | sort -u` and test each hit with `[ -e ]` (PR #44: two Phase-00
  task-note records were left naming a status-stripped path).
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
  ``boost.pyx lines 427, 447, 456 as of `e94fb21^` `` citation into a file
  a later purge had cut from 461 to 241 lines — stale since `e94fb21`,
  and itself unpinned here until PR #43 tripped over it; caught by
  review, not by the author. Run `check_doc_citations.py` over the docs
  you touched, not the ones you wrote, and pin historical evidence to a commit
  (``lines 427, 447, 456 as of `e94fb21^` ``) so a later deletion cannot
  falsify it (PR #42, #43 — #43 ran the checker over only the two docs a
  reviewer named; three further elisions in the same file resolved by
  suffix and would have rotted later).
- [changed-vs-sees-only-commits] A `--changed-vs <ref>` tool diffs
  *committed* history, so running it on an uncommitted tree scans zero
  files and prints a success-shaped line — `check_doc_citations.py
  --changed-vs origin/master` answered "no docs to check" mid-session and
  was read as a pass; the real run after committing failed. Same family as
  `markdownlint` exiting 0 on a glob that matches nothing. For any gate,
  confirm it reported a non-zero *scope* (docs scanned, tests collected,
  files linted) before believing its verdict (PR #42).
- [sqrt-hides-factor-signs] Refactoring `sqrt(P)` into `sqrt(A*B)` for
  conditioning silently widens the accepted domain: a product of two
  negative factors is positive, so the root returns a plausible number
  where the quantity is undefined. When the polynomial has more than one
  real root, the region *outside* the outermost pair is exactly where
  this hides — check the sign of every factor across the whole unphysical
  domain, not only the interval you sampled. Pair each sign-changing
  factor with a strictly positive partner so one root per boundary goes
  NaN (PR #43: the Källén `λ(s, m1², m2²)` turns positive again below
  `|m1 - m2|`, so `two_body_momentum(1.0, 10.0, 1.0)` returned 48.99
  against a threshold of 11 — pre-existing, but a new docstring had just
  promised NaN below threshold, converting it into a false contract).
- [sibling-copies-of-a-fixed-claim] A repo-fact claim is usually written
  down in more than one durable doc, and fixing the copy you happened to
  open leaves the others reading as authoritative. The two shapes that
  bite: a *list* whose head item you corrected while items 2–n still
  describe the old tree, and a claim duplicated across
  `docs/agents/` and `.claude/skills/*/SKILL.md`. Fix by re-reading the
  whole enclosing artifact, then `rg` the claim's distinctive phrase
  repo-wide — not just the file the reviewer cited (PR #48: `AGENTS.md`'s
  Layering list had item 1 updated but items 2–6 still named the deleted
  `gamma_ray.py` and `_decay`; and `test/conftest.py excludes
  test_gamma_ray.py` was fixed in `docs/agents/review-lenses.md` while
  three skills kept saying it; PR #50: a phase file's
  `51 passed / 20 skipped` survived because Task 0.1 had fixed the
  *sibling* copy in the project README and never swept the class, and a
  single task note carried two different byte counts for the same
  directory). Generalizes [stale-ci-capability-claim] from workflows to
  any repo fact, and applies *within* one file as much as across
  several.
- [elided-doc-paths] A `.../foo.py` shorthand in a durable doc is not
  mechanically resolvable and fails `scripts/agents/check_doc_citations.py`
  the moment the basename is not unique in the repo — and the elision is
  most tempting exactly where a doc lists many siblings from one package.
  Write every citation as a full repository-relative path (PR #43:
  `.../widths.py` and `.../utils.py` were ambiguous with 2 and 4
  candidates). Pairs with [touched-doc-inherits-its-citations] above.
- [measured-tree-vs-imported-module] A tool that records provenance must
  prove that what it *imports* and what it *measures* are the same tree,
  or it will record a falsehood with full confidence. Import resolution
  follows `sys.path`; a digest, a file walk, or a `REPO_ROOT` constant
  follows the filesystem, and a site-packages install silently separates
  the two. The failure is invisible precisely because every artifact
  looks internally consistent. Tie them together in code — assert each
  imported module's `__file__` lies under the measured root — and record
  the resolved path in the output, so a *past* run stays auditable and
  not just a future one guarded (PR #50: the parity-corpus generator
  hashed every `.pyx` under the repo while importing `hazma` from the
  environment, so a stale or Rust-enabled install could have produced
  values that `kernel_digest` did not describe).
- [exactness-untestable-on-one-platform] A bit-equality assertion that
  only ever runs on the machine it was written on is not a verified
  invariant, it is an untested one, and the first CI matrix that reaches
  it will fail wholesale rather than subtly. Before asserting exactness
  on any derived quantity, ask which *libm* computed it: `numpy.geomspace`
  and `logspace` go through `log10` and a `power`, so a grid built from
  them is not "arithmetic on constants" across platforms even though it
  is within one. Give such a quantity a tight, derived budget in the
  off-platform mode and keep bit-equality only where the whole
  environment matches (PR #52: the parity runner compared abscissae with
  `assert_array_equal` in *both* modes on that premise; enabling the
  suite in CI failed all 623 blocks on Linux/glibc, every one of them by
  exactly one ulp and none of them on a value budget). The general
  shape: a gate whose strictest path is unreachable in the environment
  that wrote it buys no safety until something else runs it.
- [gate-disabled-stays-green] Removing, narrowing, or conditionally
  skipping a check cannot turn CI red, so a green run is not evidence the
  check still runs. Verify a gate by observing it *execute* — the job's
  collected/passed counts, or the env it echoed — not by observing the
  absence of failure. Two traps land here together: GitHub Actions'
  `&&`/`||` return values rather than booleans and treat `''` as falsy,
  so `cond && '' || 'flag'` yields `'flag'` for **both** outcomes of
  `cond` and the empty branch is unreachable; and a skip expressed as
  `--ignore` disappears from the summary line entirely rather than
  showing up as a skip count (PR #52 added
  `PARITY: ${{ runner.os == 'macOS' && '' || '--ignore=test/parity' }}`
  to scope the parity corpus to its capturing platform, which instead
  disabled it on every entry including macOS; all seven checks passed for
  two PRs, and PR #53 caught it only by noticing the job reported `380
  passed` where a run including the corpus collects ~1019).
- [marker-count-vs-outcome-count] A count of *declaration sites* and a
  count of *runtime outcomes* are different numbers, and prose that says
  "N skips" silently picks one. A marker on a parametrized class yields
  many skipped tests; a `skipif` whose condition is false yields none. Do
  not derive either number by reading decorators — take outcomes from
  `pytest -rs` and sites from `rg`, state which one you mean, and give
  both when they differ (PR #53: a phase file claimed "three unrelated
  skips survive" while enumerating five sites, and the task note
  attributed 13 skipped tests to 5 sites plus a `skipif` that never
  fired; the true split was 5 sites → 13 tests as 5 + 5 + 3).
