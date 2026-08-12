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
  as authoritative until review recounted (PR #35). It applies just as
  much to *your own* artifacts, where the temptation to count in your head
  is strongest because you just wrote the thing: a task note claimed "53
  tests in 7 classes" and "15 passed (7 new)" against 53 in 8 and 8 new,
  copied into three durable docs and a PR body, all measured on the same
  tree and all wrong, while the numbers they sat beside were correct
  (PR #59). Quote the command next to the number — `pytest --collect-only
  -q | awk -F'::' '{print $2}' | sort | uniq -c`, `grep -c '#\[test\]'` —
  so the next reader re-runs it instead of trusting it.
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
  derived at all; this one is about a derivation that expired. A second
  shape: the measurement expires because the *environment* changed, not the
  tree — and then both figures are true, of different environments, which is
  why neither looks wrong in isolation (PR #55: a wheel recorded as
  `cp313-cp313-…` in one section and `cp312-cp312-…` in another, because the
  venv was rebuilt on the corpus's capturing interpreter partway through).
  When a measured artifact carries the environment in its name, record the
  invariant and the mechanism (`cp<XY>`, never `abi3`, because a Cython
  extension remains) rather than one run's value.
- [flat-vs-sectioned-numbering] A document whose items restart numbering in
  each section, cited elsewhere by a flat index, has two schemes and no key —
  so a correct citation reads as a dangling one and a reviewer's "fix" breaks
  the other twenty. Put the mapping in the cited document, and annotate new
  citations with both forms (PR #55: `projects/cython-to-rust/rules.md` has
  Parity 1–3, Constants 1, Licensing 1, Rust 1–4, Process 1–3, while the plan
  and phase files cite `rule 4`, `rule 8`, `rule 10`, `rule 12`; the flat
  scheme was documented only in a parenthetical inside one Phase 00 task
  note). Applies to any ADR, checklist, or rules file with numbered sections.
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
  directory; PR #60: a wrong call-site count was swept by grepping the
  *paired* phrases it usually appeared in — `eleven`, `six of the
  eleven` — which fixed twelve copies and missed two more that carried
  the bare number word alone, one of them in the same file's own
  "all twelve occurrences were swept" record. Key the sweep on the
  **claim** — every numeral or number word within N characters of
  `call site`, in either order — not on the phrasing the number happened
  to arrive in, and never let the sweep's own summary assert a
  completeness its pattern cannot support. Round 2 of the same PR showed
  that claim-keyed is still not enough while the pattern is *line*-keyed:
  three more copies survived, two of them wrapped across a newline
  (`the four\n  spectra/mediator-spectrum calls`) and one using a synonym
  (`those six sites` where the anchor was `call site`) — including two in
  the canonical reference the earlier round claimed to have swept, which
  then held three different counts of the same thing. **Reflow the file to
  one line before matching, and alternate the synonyms** —
  `re.sub(r"\n\s*(//|#|\*)?\s*", " ", text)` then a single regex over the
  result — because prose wraps wherever the formatter put it and a
  line-oriented `rg` cannot see a claim that straddles the break.
  PR #63 adds the *temporal* shape: a later round of the same PR changed
  a test count from 45 to 47, and the verification records written in the
  earlier round — a phase README, two task notes, and the PR body — kept
  reporting the superseded figure alongside its derived siblings
  (`1422` full-suite passes, `+45`, `17 scoped`). **A count you change
  mid-PR invalidates every number derived from it, not just its own
  copies**, so sweep the arithmetic (baseline ± delta, per-mode splits)
  and not only the digit; and treat the PR body as a durable record with
  the rest, since review reads it as one).
  Generalizes [stale-ci-capability-claim] from workflows to
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
- [renumbered-list-orphans-its-references] Inserting an item into a numbered
  list silently falsifies every prose reference to the items after it — and
  those references live outside the list, often outside the file, so
  renumbering the list itself looks like the whole job. Sweep
  `rg -n '[Gg]ate [0-9]|[Ss]tep [0-9]|item [0-9]'` (or the local ordinal
  noun) across `docs/`, `.claude/`, `.codex/` and `docs/followups/` after
  any insertion, and prefer *naming* the target over re-pinning a number
  that will shift again at the next insertion (PR #56: three cargo gates
  inserted as 4–6 renumbered markdownlint from 6 to 9, leaving
  `docs/agents/preflight.md`'s "Markdown rules" section opening "Gate 6
  runs against the committed `.markdownlint.jsonc`" — pointing at
  `cargo test` — plus three live references in an open follow-up and two
  in a resolved one; caught by review, and the list renumbering itself had
  been done correctly). Distinct from [flat-vs-sectioned-numbering], which
  is about two schemes coexisting without a key; this is one scheme whose
  indices moved.
- [unrun-workflow-cannot-close-a-criterion] A workflow with no
  `pull_request` trigger is invisible to PR checks, so an exit criterion
  phrased against it stays unmeasured no matter how green the PR is —
  and "wired, and the recipe is documented upstream" is an argument, not
  evidence. Dispatch it (`gh workflow run <file> --ref <branch>`) and
  paste the job conclusions; check first that any publishing job is gated
  (`if: github.event_name == 'release'`) so the dispatch is build-only.
  Never mark the task Complete over the gap (PR #56: `release.yml`'s
  cibuildwheel job carried two of Task 2.2's exit criteria and had never
  run; the dispatched run passed on both platforms with `publish`
  skipped, and the assertion step reported `5 wheel(s) carry
  hazma/_core.abi3.so` per OS).
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
- [hand-written-population-in-a-derived-check] A count can be *measured*
  with a real command and still be wrong, because the command was run
  against a hand-written list of what to measure rather than against the
  population enumerated from source. Derive the population with the same
  `rg`/parse step that produces the values, so the check cannot silently
  under-enumerate; a pairing table typed by hand is the population, and
  it is exactly as trustworthy as typing. The failure hides when the
  wrong answer collides with a different, correct number nearby (PR #58:
  "all twelve masses in `constants.pxd` are bit-equal to
  `hazma/parameters.py`'s" was checked by a script — over a
  hand-written 12-pair list that omitted `MASS_KL` and `MASS_KS`. There
  are 14. Twelve was also the correct count of spectra extensions that
  `include` the header, two paragraphs up, so every internal
  cross-check agreed). Distinct from [derived-count-not-rederived],
  where no derivation happened, and from
  [measurement-taken-before-the-task-ended], where a correct derivation
  expired: here the derivation ran, on the final tree, over the wrong
  domain.
- [bound-parameter-sized-the-allocation] A parameter that *bounds* work is
  not a prediction of it, so sizing a buffer by one turns a caller's
  permissive input into an allocation request. `scipy.integrate.quad`
  accepts `limit` up to `INT_MAX`; the QUADPACK port allocated its five
  `limit`-length workspaces up front, so `limit = INT_MAX` reserved ~80 GiB
  before the first panel was evaluated — and Rust **aborts** on allocation
  failure rather than unwinding, so it can never surface as a Python
  exception, against an exit criterion of "never panics across FFI"
  (PR #60). Two traps around it: the eager allocation is *invisible* on a
  platform whose allocator overcommits (peak RSS moved 18.2 → 18.6 MB on
  macOS, so a passing test proved nothing), and a first-round fix that
  merely rejects values *above* the range leaves the same hazard directly
  below it. Grow on demand instead, and make the guard test ask for a size
  no allocator can satisfy — a mutation reverting to eager sizing must
  fail loudly (`memory allocation of 36028797018963976 bytes failed`,
  SIGABRT) rather than depending on the host's memory policy.
- [signed-zero-lost-by-a-derived-formula] A wrapper that reaches its answer
  by *arithmetic on* an upstream routine's outputs, rather than by calling
  it, inherits an argument case the upstream never had — and `-0.0` is the
  one that hides, because `-0.0 < 0.0` is false, so IEEE routes it to the
  *zero* branch and a `+0.0` test says nothing about it. Sweep both signs
  of zero against the oracle, at every order or branch the formula runs,
  not just the one a reviewer names (PR #59: `bessel_kn` built `Kₙ` from
  the recurrence `K_{m-1} + (2m/x)·K_m` on cephes `k0`/`k1` seeds; at
  `x = -0.0` the seeds are `+∞` and `2m/x` is `-∞`, so every order from 2
  up returned `NaN` where scipy returns `+∞`. Orders 0 and 1 return the
  seeds directly and were always right, which is exactly why the bug
  survived an edge-case test that checked `+0.0` and `-1.0`). Generalizes
  past zero: any guard the upstream applies *before* its arithmetic has to
  be re-applied by a caller that does arithmetic *after* it.
- [platform-scoped-oracle-asserted-globally] A test that compares a port
  against a *locally compiled* oracle — the Cython twin, NumPy, anything
  the platform's C compiler built — is asserting a property of that
  build, not only of the port. Fused multiply-add is the usual culprit:
  `-ffp-contract=on` is the C default, so an oracle compiled for a target
  with hardware FMA computes different numbers than one without, and a
  bit-for-bit assertion that is exactly right on the reference platform
  fails everywhere else (PR #61: 19 assertions in
  `test/test_core_{interp,boost}.py` passed on macOS/arm64 and failed on
  Linux/x86-64, where neither NumPy nor the Cython contracts). **Declare
  the scope from the platform; do not probe for it.** Read the capturing
  platform out of `test/parity/data/manifest.json` so the scope cannot
  drift from the corpus's, compare bit-for-bit there, and hold a measured
  budget everywhere else. Probing — evaluate the compiled oracle against
  an unfused transcription, conclude "this build contracts", skip where it
  does not — is the trap this lesson used to recommend, and it fails in
  both directions: a probe sees one contraction mechanism and is blind to
  the rest, so it claims bit-equality on a build that diverges for another
  reason (PR #63, `test_core_positron_muon.py`, twice) *and* silently
  voids the whole comparison when the one mechanism it knows is absent
  (`test_core_boost.py`, 19 claims skipped on every non-macOS CI entry
  from PR #61 until 2026-08-12 — and the capturing platform cannot see
  either failure, because there the probe answers correctly by accident).
  Scale the off-platform budget to the **peak** of the compared array
  rather than applying a pointwise `rtol`: the worst relative gap between
  two roundings lands at whatever cancellation point the domain contains,
  and an `rtol` wide enough for that hides real defects, while against the
  peak a wrong branch or dropped term still lands at O(1). Measure the
  budget — build the tree for the other platform and compare — rather than
  guessing it, and give it a test that proves it still rejects a real
  error, since on the capturing platform nothing else exercises it. The
  tell you are about to make this mistake: the oracle is something you
  compiled rather than something you pinned.
- [settling-a-deferral-has-two-sweeps] When a task *settles* something an
  earlier task deferred, the stale text lives in two disjoint populations
  and grepping one feels like finishing. The **pointers** say "Task N
  decides this" and all carry the task id, so they sweep in one `rg`; the
  **statements of the pre-decision behavior** carry no such token — they
  are ordinary prose asserting the old contract as current fact, and they
  are the ones a future reader will act on (PR #62: `Task 3.5 decides` was
  swept across four durable docs and the phase file, while
  `projects/cython-to-rust/learnings/phase-02-rust-scaffold.md` §2 and the
  project working memory's Findings still said a 0-d array must be
  `float64`, that everything invalid raises `ValueError`, and that
  `map_unary` was the only helper — every one of which the same PR
  changed). Sweep the **behavior words** as well as the task id: the
  identifiers the settled thing is named by (`map_unary`), and the
  distinctive phrases of the *superseded* rule (`still enforces dtype`,
  `anything else → ValueError`). Note that a doc predicting its own
  supersession ("a Task N decision that changes any of them turns a named
  test red") is a pointer *and* a statement, and closing the loop on it is
  cheap. Sibling of [sibling-copies-of-a-fixed-claim]; the difference is
  that the two populations share no token, so one pattern cannot find
  both.
