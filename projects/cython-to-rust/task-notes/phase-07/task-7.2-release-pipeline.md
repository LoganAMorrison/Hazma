# Task 7.2: Release pipeline

**Date:** 2026-08-29
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-07-cutover.md` (Task 7.2);
`../../PLAN.md` (Scope, Phases)
**Related ADRs:** ADR-0001 (framework choice, names maturin and abi3-py310)
**Depends On:** Task 7.1 (maturin backend)

## Objective

Rebuild `.github/workflows/release.yml` on maturin so a release publishes
two abi3 wheels and an sdist instead of a ten-entry cibuildwheel matrix,
verify the wheel tag and cross-CPython importability inside the workflow,
and give `ci.yml` a cargo cache.

## Exit Criteria

Copied from the phase file's Task 7.2 block, including its Task 7.1
narrowing.

- `release.yml` rebuilt on maturin (PyO3/maturin-action or cibuildwheel's
  maturin support — pick and record): **2 abi3 wheels** (manylinux
  x86_64, macOS arm64) + sdist; trusted-publishing job preserved; wheel
  abi3 tags and importability verified in the workflow
  (`CIBW_TEST_COMMAND`-equivalent import check on the oldest supported
  CPython, 3.10, and the newest).
  - Narrowed by Task 7.1: the tag itself is already correct
    (`cp310-abi3`, verified locally cross-version). What this task owes
    is producing the *two platform* wheels in CI and verifying the tag
    and the import **there**.
- Decision recorded (one line in this note): whether to add
  linux aarch64 / Windows wheels now that they are cheap — default no,
  matching the current support surface.
- `ci.yml`: drop per-version rebuild caching of Cython; add cargo
  caching; matrix unchanged.
- The workflow is *observed to run*, not merely wired
  (`docs/agents/lessons.md` `[unrun-workflow-cannot-close-a-criterion]`):
  a dispatch on this branch with the publish job gated, and the job
  conclusions pasted into the Verification section below.

## Inputs Reviewed

- `../../PLAN.md` — Scope, Numerical impact, Phases table.
- `../../phases/phase-07-cutover.md` — Prerequisites and the Task 7.2
  block.
- `../README.md` (project) and `README.md` (phase 07) — handoff, open
  questions.
- `../../rules.md` — rules 6–9 (Rust conventions), rule 12 (measured
  claims).
- `task-7.1-maturin-backend.md` — `## Handoff to Next Task` only.
- `../../learnings/phase-02-rust-scaffold.md` — the cibuildwheel/rustup
  recipe and the "release.yml never runs" finding.
- `../../learnings/phase-06-mediator-spectra.md` §1–2 — zero Cython,
  corpus is the only pin.
- `docs/agents/lessons.md` — whole ledger;
  `[unrun-workflow-cannot-close-a-criterion]`,
  `[stale-ci-capability-claim]`, `[gate-disabled-stays-green]` and
  `[wheel-tag-vs-extension-abi]` bind here.
- `docs/agents/environment.md` — packaging section (wheel vs sdist
  machinery under maturin).
- `.github/workflows/{ci,release}.yml`, `pyproject.toml`,
  `rust/Cargo.toml` — the tree being changed.

## Findings

- **A wheel filename's last three fields are compressed tag *sets*, not
  tags.** Each is dot-separated and the wheel carries their cross product,
  one `Tag:` line per member (PEP 425). The manylinux wheel is
  `cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64` — one platform
  field standing for two tags — while the macOS wheel is one tag per
  field. An assertion comparing the field against the `.dist-info/WHEEL`
  lines as a single string therefore passes locally, passes on macOS, and
  rejects a correct Linux wheel, which is exactly what run 33283941251
  did. The local mutation harness could not have caught it: every wheel
  this machine builds is the macOS shape.
- **The criterion's first ci.yml clause had nothing to drop.** "Drop
  per-version rebuild caching of Cython" was written in the August 2026
  analysis against an anticipated shape; no such caching was ever in the
  file. `rg -in 'cython' .github/workflows/` returns six hits on
  `origin/master` and every one is prose inside a comment: three are the
  project slug `cython-to-rust` dating a design note, three say Task 6.4
  deleted the last Cython extension. No step, flag or cache key mentions
  Cython. The only caching `ci.yml` has is `actions/setup-python`'s
  `cache: pip`, which caches wheel downloads keyed on `pyproject.toml`
  and is unrelated. Recorded rather than silently satisfied.
- **cibuildwheel had nothing left to iterate over.** Its per-version
  matrix is its purpose, and `abi3-py310` removes the versions: the ten
  wheels it built were five copies per platform of one shared object.
  That is what makes this a replacement rather than a port of the same
  job to a new driver.
- **The manylinux container ships its own Rust toolchain.** The
  `CIBW_BEFORE_ALL_LINUX` rustup install and the `CIBW_ENVIRONMENT_LINUX`
  `PATH` edit that Phase 02 worked out, and the host
  `dtolnay/rust-toolchain` step that covered macOS, all come out with
  cibuildwheel. `maturin-action` needs neither.
- **`twine check` cannot fail an unbuildable sdist.** It validates
  metadata. An archive missing `rust/Cargo.toml` would pass it and then
  fail every `pip install --no-binary` — and nothing else in the pipeline
  would catch that, because both wheel jobs build from the checkout, not
  from the sdist.

## Decisions and Implementation Notes

- **`PyO3/maturin-action@v1` over cibuildwheel's maturin support**, the
  choice the phase file asked to record. cibuildwheel's value here was
  the CPython matrix and the manylinux container; abi3 removes the first
  and maturin-action supplies the second. Keeping cibuildwheel would mean
  narrowing `CIBW_BUILD` to a single version to avoid rebuilding the same
  wheel five times — configuring a tool to not do the thing it is for.
- **No aarch64 or Windows wheels** (the decision the criterion asks for,
  default kept). `PLAN.md` puts them out of scope and Task 7.2 is where
  the call is made explicit: the support surface is unchanged, the two
  platforms are the two CI tests on, and neither target has a user asking
  for it. They are cheaper under maturin than they were — a matrix row
  each — which is exactly why this stays a deliberate no rather than an
  oversight. `docs/followups/todo/` gets no stub: `PLAN.md`'s Scope
  already records it as a cheap follow-up.
- **`--no-deps` for the two import checks.** The claim under test is that
  one `cp310-abi3` wheel loads on both ends of the range it advertises.
  `hazma/__init__.py` imports only the standard library, so
  `import hazma._core` needs none of numpy/scipy/matplotlib/scikit-image,
  and installing them would let a third-party wheel gap on the newest
  CPython fail this job for a reason that is not hazma's. The
  full-dependency install smoke already exists in `ci.yml`'s
  `Import smoke test`, on every matrix entry.
- **Two assertions, not one.** The old step checked only that
  `hazma/_core.abi3.so` was inside each wheel, which is the
  extension-level claim; `docs/agents/lessons.md`
  `[wheel-tag-vs-extension-abi]` exists because the distribution-level
  claim can disagree with it, and under setuptools-rust it did. The new
  step asserts the filename tag set, the `.dist-info/WHEEL` `Tag:` lines,
  and that the abi3 object is the *only* compiled object — a
  `_core.cpython-312-*.so` appearing beside it would mean the feature
  stopped applying.
- **`--locked` on the wheel build**, so the published artifact is a build
  of the committed `rust/Cargo.lock` rather than of whatever resolves that
  day. Verified the lockfile is current:
  `cargo metadata --manifest-path rust/Cargo.toml --locked` succeeds.
- **A path-filtered `pull_request` trigger**, scoped to `release.yml` and
  `pyproject.toml`. It resolves the class this task had to work around —
  `docs/agents/lessons.md` `[unrun-workflow-cannot-close-a-criterion]`,
  which Phase 02 hit on the same file — at the cost of two rare paths
  rather than every PR. Rust source is deliberately excluded: `ci.yml`
  already compiles the crate on both operating systems for every PR, so
  adding `rust/**` here would buy a duplicate build. The `publish` job's
  existing `if: github.event_name == 'release'` gate is what keeps the new
  trigger from uploading, and it was checked before dispatching.
- **`manylinux: auto` rather than a pinned policy.** The criterion names
  "manylinux x86_64" without a policy; pinning one here would assert a
  container the action may move off. The resulting platform tags are
  printed and compared against the filename instead of assumed — they came
  back `manylinux_2_17_x86_64` and `manylinux2014_x86_64`.
- **One cargo cache per OS across the whole Python matrix.** abi3 links
  against the limited API, so the cargo artifacts do not vary with the
  interpreter and five matrix entries would otherwise fill five caches
  with the same objects. `Swatinem/rust-cache@v2` keys on `runner.os` and
  the toolchain already; `workspaces: rust` points it at the crate's own
  `[workspace]` root. pip builds in-tree, so both the `pip install .` and
  the `pip install -e .` in that job populate and reuse `rust/target`
  rather than a temporary copy.

## Files Changed

- `.github/workflows/release.yml` — rewritten on maturin: two wheel
  matrix rows, the tag/extension assertion, the two import checks, the
  sdist build-input assertion, the new `pull_request` trigger.
- `.github/workflows/ci.yml` — `Swatinem/rust-cache@v2` in the `rust` and
  `test` jobs. Matrix, steps and commands otherwise unchanged.
- `projects/cython-to-rust/phases/phase-07-cutover.md` — Prerequisites
  gains the post-7.2 release-pipeline facts; Task 7.3's enumeration gains
  the three `[tool.setuptools.package-data]` sites in the skills, which
  its Cython-worded grep could not have found.
- `projects/cython-to-rust/learnings/phase-02-rust-scaffold.md` — the two
  forward pointers this task discharged, settled in place rather than
  deleted (the rustup-in-container recipe, the missing PR trigger).
- `projects/cython-to-rust/task-notes/phase-07/README.md` and this note —
  status, findings, handoff.

## Verification

### The workflow, observed running

`docs/agents/lessons.md` `[unrun-workflow-cannot-close-a-criterion]`:
these criteria are closed against dispatched runs, not against the file.
`publish`'s `if: github.event_name == 'release'` gate was read before
each dispatch, and `publish` skipped in both.

`gh workflow run release.yml --ref claude/cython-to-rust/task-7.2-release-pipeline`

| Run | Wheel (manylinux) | Wheel (macOS) | sdist | publish |
| --- | --- | --- | --- | --- |
| [33283941251](https://github.com/LoganAMorrison/Hazma/actions/runs/33283941251) | failure | success | success | skipped |
| [33284053186](https://github.com/LoganAMorrison/Hazma/actions/runs/33284053186) | success | success | success | skipped |

The first run's failure is the compressed-tag-set finding above: the
build was correct and the assertion was wrong. It is kept in this table
rather than replaced by the green one, because it is the evidence that
the assertion can fail.

Assertion output from the green run (`gh run view 33284053186 --log`):

```text
Build wheel (manylinux-x86_64)  wheel: hazma-2.1.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
Build wheel (manylinux-x86_64)  platform tags: ['manylinux_2_17_x86_64', 'manylinux2014_x86_64']
Build wheel (manylinux-x86_64)  WHEEL Tag: ['cp310-abi3-manylinux2014_x86_64', 'cp310-abi3-manylinux_2_17_x86_64']
Build wheel (manylinux-x86_64)  extension: hazma/_core.abi3.so
Build wheel (manylinux-x86_64)  Import 3.10  2.1.0 /opt/hostedtoolcache/Python/3.10.21/x64/lib/python3.10/site-packages/hazma/_core.abi3.so
Build wheel (manylinux-x86_64)  Import 3.14  2.1.0 /opt/hostedtoolcache/Python/3.14.7/x64/lib/python3.14/site-packages/hazma/_core.abi3.so
Build wheel (macos-arm64)       wheel: hazma-2.1.0-cp310-abi3-macosx_11_0_arm64.whl
Build wheel (macos-arm64)       platform tags: ['macosx_11_0_arm64']
Build wheel (macos-arm64)       WHEEL Tag: ['cp310-abi3-macosx_11_0_arm64']
Build wheel (macos-arm64)       extension: hazma/_core.abi3.so
Build wheel (macos-arm64)       Import 3.10  2.1.0 /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/hazma/_core.abi3.so
Build wheel (macos-arm64)       Import 3.14  2.1.0 /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/hazma/_core.abi3.so
Build sdist                     hazma-2.1.0.tar.gz: 264 members, all build inputs present
```

Two wheels, both `cp310-abi3`, both imported on 3.10 and 3.14, each
carrying `hazma/_core.abi3.so` and no other compiled object. 264 sdist
members matches the count Task 7.1 measured locally.

### The assertions, proved able to fail

A check that cannot fail is not a check. Both `python3 - <<'PY'` blocks
were extracted **from the workflow file** by regex — not retyped — run
against locally built artifacts, then against mutants built by rewriting
the archives:

| Mutant | Exit | Message |
| --- | --- | --- |
| macOS wheel, unmodified | 0 | `extension: hazma/_core.abi3.so` |
| manylinux compressed tag set | 0 | `extension: hazma/_core.abi3.so` |
| one `Tag:` line dropped | 1 | `WHEEL declares Tag: [...], filename expands to [...]` |
| a `cp311-cp311` `Tag:` line added | 1 | same |
| filename retagged `cp314-cp314` | 1 | `filename tag is cp314-cp314, expected cp310-abi3` |
| `Tag:` rewritten to `cp314-cp314` | 1 | `WHEEL declares Tag: [...], filename expands to [...]` |
| `_core.cpython-312-darwin.so` added | 1 | `expected hazma/_core.abi3.so as the only compiled object` |
| two wheels in `dist/` | 1 | `expected exactly one abi3 wheel in dist/, found 2` |
| no wheels in `dist/` | 1 | `... found 0: []` |
| sdist, unmodified | 0 | `264 members, all build inputs present` |
| sdist without `rust/Cargo.lock` | 1 | `missing ['hazma-2.1.0/rust/Cargo.lock']` |
| sdist without `rust/Cargo.toml` | 1 | `missing ['hazma-2.1.0/rust/Cargo.toml']` |
| two sdists in `dist/` | 1 | `expected one sdist, found [...]` |

### Local gates

```text
$ scripts/agents/preflight.sh --paths "hazma test" \
    --md "<the four markdown files this diff touches>"
PASS   black --check           hazma test
FAIL   isort --check-only      run `isort hazma test` and re-check
FAIL   ruff check              see output below
PASS   cargo fmt --check       rust/
PASS   cargo clippy            rust/
PASS   cargo test              rust/
PASS   pytest                  2231 passed, 15 skipped, 12 subtests passed in 29.52s
PASS   import hazma            version 2.1.0
PASS   markdownlint            <the four files>
SKIP   version bump            not a closing PR (pass --closing)
PASS   forbidden tokens        none added
```

The two red rows are inherited, not introduced:
`docs/followups/todo/preflight-isort-ruff-red-on-trunk.md` records both
gates as failing on unmodified trunk code, and
`git diff origin/master --name-only | grep -E '\.py$|\.pyx$|\.rs$'`
returns nothing, so this diff cannot have moved either count. The
markdownlint row was red on the first run — four `MD013` line-length
violations in this note — and the lines were rewrapped rather than the
rule relaxed.

- `cargo metadata --manifest-path rust/Cargo.toml --locked` succeeds, so
  `--locked` in the wheel build cannot fail on a stale lockfile.
- `uv build --wheel` / `uv build --sdist` locally:
  `hazma-2.1.0-cp310-abi3-macosx_11_0_arm64.whl`, `hazma-2.1.0.tar.gz`.

### Measured cost

Wall clock for the whole workflow, `createdAt` to `updatedAt`:

| Run | Driver | Wheels | Wall |
| --- | --- | --- | --- |
| [31297673951](https://github.com/LoganAMorrison/Hazma/actions/runs/31297673951) (2026-08-09, Task 2.2) | cibuildwheel | 10 | 16m 30s |
| [33284053186](https://github.com/LoganAMorrison/Hazma/actions/runs/33284053186) (2026-08-30) | maturin-action | 2 | 1m 18s |

Not a controlled comparison — different days, different runner images,
and the cibuildwheel run installed each of its ten wheels with full
dependencies while these two import-check with `--no-deps`. The
order-of-magnitude is nevertheless the ten-to-two wheel count: the
cibuildwheel Linux job alone ran 16m 25s of its 16m 30s.

### Deferred

- The sdist is asserted to *contain* its build inputs, not to build. A
  source install compiles the crate from scratch and Task 7.1 verified it
  locally (`uv pip install --no-binary hazma` into a fresh 3.10 venv);
  adding it here would put minutes of `cargo build` on every packaging PR
  to re-check something the wheel jobs already prove compiles.

## Numerical impact

None. The diff touches two GitHub Actions workflow files and four project
documents; no `hazma/` module, no `rust/` source, no test. Verified:
`git diff origin/master --stat` lists no path under `hazma/`, `rust/` or
`test/`, and the bare `pytest` in the preflight table above reports the
same **2231 passed, 15 skipped, 12 subtests passed** as Task 7.1.

## Open Questions

- None outstanding. The aarch64/Windows question the criterion posed is
  answered in §Decisions (no, deliberately), and the phase README's Open
  Questions entry is rewritten to state the answer rather than keep
  posing it.

## Plan Impact

**Impact Level:** Phase file patched.

Two patches, both because this task's own diff falsified or narrowed
canonical text. No ADR is needed: ADR-0001 already names maturin and
`abi3-py310`, and nothing about that changed.

- **Prerequisites** carried "**Not yet touched:** release.yml still uses
  cibuildwheel (cp310–cp314 × {linux x86_64, macos arm64} = 10 wheels
  …)". Replaced with the post-7.2 facts, dated, in the same shape as the
  Task 7.1 bullet beside it.
- **Task 7.3's exit criteria** gained three sites its own derivation
  could not have found. That enumeration came from
  `rg -n 'Cython|\.pyx|\.pxd|cythoniz'` over `AGENTS.md` and
  `environment.md`; the three stale `[tool.setuptools.package-data]`
  claims live in the skills and say "setuptools", not "Cython", so
  neither the pattern nor the file set reaches them. Extending the table
  is cheaper than letting 7.3 miss them, and Step 7's canonical-contract
  rule says to patch it here rather than defer.

Also settled, in `../../learnings/phase-02-rust-scaffold.md`: the two
forward pointers this task discharged — the rustup-in-the-container
recipe it said Task 7.1 would inherit (7.2 retired it instead), and the
missing pull-request trigger. Settled in place with a date rather than
deleted, so the phase's record of what it learned stays intact.

## Stale-state sweep

Run against this branch after every prose edit was frozen.

### Identifier sweep

Task notes are excluded throughout — they are history (ADR-0002).

```sh
rg -n 'cibuildwheel|CIBW_' \
  projects/cython-to-rust/{phases,learnings}/ \
  projects/cython-to-rust/PLAN.md docs/ .github/ README.md AGENTS.md
```

```text
learnings/phase-02-rust-scaffold.md:133,135,136   KEPT   historical; the entry now carries "Settled by Phase 07 Task 7.2"
.github/workflows/release.yml:7                   EDITED new header naming what the job replaced
phases/phase-07-cutover.md:26                     EDITED new Prerequisites bullet, same sentence
phases/phase-07-cutover.md:69,73                  KEPT   Task 7.2's criterion as written; the spec, not a claim
phases/phase-07-cutover.md:182                    KEPT   phase Exit Criteria "no cibuildwheel-version-matrix residue"
docs/agents/lessons-examples.md:471               KEPT   worked example for [unrun-workflow-cannot-close-a-criterion]
```

`rg -n 'pypa/cibuildwheel' .github/ docs/ AGENTS.md projects/cython-to-rust/{phases,learnings}/`
→ no occurrences. The action pin this diff removes is cited nowhere but
closed task notes.

```sh
rg -n 'PyO3/maturin-action|Swatinem/rust-cache' \
  .github/ projects/cython-to-rust/{phases,learnings}/ docs/
```

→ 7 hits: 4 in the two workflows (the `uses:` lines) and 3 in project
docs this diff wrote (the phase file's Prerequisites and Task 7.2
criterion, the phase-02 learnings settlement).

### Line-number citation sweep

```text
$ python scripts/agents/check_doc_citations.py \
    projects/cython-to-rust/phases/phase-07-cutover.md \
    projects/cython-to-rust/learnings/phase-02-rust-scaffold.md \
    projects/cython-to-rust/task-notes/phase-07/README.md \
    projects/cython-to-rust/task-notes/phase-07/task-7.2-release-pipeline.md
docs scanned: 4
in-repo citations checked: 0
external citations skipped: 1
  hazma/spectra/_neutrino/_muon.pyx (1)
out-of-range or ambiguous: NONE
```

The skipped `.pyx` is pre-existing in the phase-02 learnings and is the
known `docs/followups/todo/citation-checker-skips-deleted-inrepo-files.md`
behavior, not something this diff introduced. The line numbers this diff
*does* add — Task 7.3's three skill sites — are stated with their
derivation command in the phase file, so the next reader re-derives them
rather than trusting them.

### Forward-looking phrase sweep

Over the six touched files:

```sh
rg -n '(Task 7\.[0-9] will|will be added|still pending|Not yet touched|still has no pull-request|Not started)'
```

```text
task-notes/phase-07/README.md:22   KEPT  | 7.3 | ... | Not started |
task-notes/phase-07/README.md:23   KEPT  | 7.4 | ... | Not started |
```

Both are live status in the Tasks table and correct. "Not yet touched"
and "still has no pull-request trigger", the two phrases this task
falsified, return nothing.

### Count sweep

| Claim location | Command | Actual | Status |
| --- | --- | --- | --- |
| §Findings "six hits on `origin/master`" | `rg -in 'cython' <origin/master>/.github/workflows/` | 6 (ci.yml 2, release.yml 4) | OK |
| §Findings "no step, flag or cache key mentions Cython" | same command, each hit read | all six begin `#` | OK |
| §Handoff "four hits, all dated historical notes" | `rg -in 'cython\|cibuildwheel\|CIBW' .github/workflows/` | 4 | OK |
| phase README "eleven mutants" | mutation-table rows minus the two unmodified baselines | 13 − 2 = 11 | OK |
| §Verification "264 members" | the workflow's own output; `uv build --sdist` locally | 264 in both | OK |
| §Verification 16m 30s → 1m 18s | `gh run view <id> --json createdAt,updatedAt` | 05:53:39→06:10:09; 00:45:05→00:46:23 | OK |
| §Verification "Linux job alone ran 16m 25s" | same, `jobs[].startedAt/completedAt` | 05:53:43→06:10:08 | OK |
| Exit criterion "2 abi3 wheels" | the green run's job list, plus the in-workflow assertion | 2 wheel jobs, one wheel each | OK |

### Numerical-impact statement

**No public value changes (verified: `git diff origin/master --stat`
lists no path under `hazma/`, `rust/` or `test/`).** The diff is two
workflow files and four project documents. The bare `pytest` in the
preflight table reports the same **2231 passed, 15 skipped, 12 subtests
passed** as Task 7.1, from a `pip install -e .` tree whose
`hazma._core.__file__` resolves inside this worktree. No entry is owed to
`../numerical-impact.md` and none was appended.

### Exit Criteria → evidence mapping

| Criterion | Evidence |
| --- | --- |
| release.yml rebuilt on maturin; driver picked and recorded | `PyO3/maturin-action@v1`; §Decisions bullet 1 |
| 2 abi3 wheels (manylinux x86_64, macOS arm64) + sdist | run 33284053186: three build jobs green, tags asserted in-workflow |
| trusted-publishing job preserved | `publish` unchanged — `environment: pypi`, `id-token: write`, `pypa/gh-action-pypi-publish@release/v1` |
| wheel abi3 tags verified in the workflow | `Assert the wheel is a single cp310-abi3 distribution`; output pasted above |
| importability on oldest (3.10) and newest CPython | `Import the wheel on CPython 3.10` / `3.14`, both platforms; output pasted above |
| aarch64/Windows decision recorded | §Decisions bullet 2 — no; the phase README's Open Questions entry now answers rather than poses it |
| ci.yml: drop per-version Cython caching | none existed; §Findings bullet 2 carries the derivation |
| ci.yml: add cargo caching | `Swatinem/rust-cache@v2` in the `rust` and `test` jobs |
| ci.yml: matrix unchanged | `git diff origin/master -- .github/workflows/ci.yml` adds only the two cache steps |
| workflow observed to run, publish gated | two dispatches; `publish: skipped` in both |

### Task-note self-consistency

`**Status:** Complete` matches the phase README's Tasks-table cell for
7.2. Every file named in §Files Changed appears in
`git diff origin/master --stat`: `.github/workflows/{ci,release}.yml`,
`phases/phase-07-cutover.md`, `learnings/phase-02-rust-scaffold.md`,
`task-notes/phase-07/README.md`, and this note as a created file.

## Handoff to Next Task

**Task 7.3 (documentation sweep) is next, and is the only thing blocking
7.4.** Read `../../PLAN.md`, `../README.md`, `README.md` (phase 07), then
the phase file's Task 7.3 block — whose enumeration this task extended.
Re-derive its line numbers before editing; the originals are Task 7.1's.

**Currently safe to assume:**

- **The release pipeline is done and has been observed running.** Two
  `cp310-abi3` wheels plus the sdist, on `PyO3/maturin-action@v1`, with
  the tag, the sole-`.abi3.so` claim, the 3.10/3.14 imports and the
  sdist's build inputs all asserted inside the workflow.
- **`release.yml` is no longer unmeasurable from a PR.** It runs on pull
  requests touching `release.yml` or `pyproject.toml`. An edit anywhere
  else still needs `gh workflow run release.yml --ref <branch>`, and
  `publish` stays gated on `github.event_name == 'release'`, which is
  what makes both safe.
- **`ci.yml` is otherwise untouched** — same matrix, same steps, same
  commands. Only `Swatinem/rust-cache@v2` was added, to the `rust` and
  `test` jobs.
- **No Cython or cibuildwheel residue remains in `.github/workflows/`**
  beyond prose in comments: `rg -in 'cython|cibuildwheel|CIBW'
  .github/workflows/` returns four hits, each a dated note explaining why
  a step looks the way it does.

**Currently risky / unknown:**

- **A format assertion written against one platform's artifact encodes
  that platform's shape.** This task's wheel-tag check passed locally and
  on macOS and rejected a correct manylinux wheel, because a filename's
  tag fields are compressed *sets* and only the Linux wheel has more than
  one member. Both halves of the matrix have to run — the same shape as
  `[exactness-untestable-on-one-platform]`, applied to a file format
  rather than to arithmetic.
- **`--paths` on `preflight.sh` feeds black, isort and ruff directly.**
  Naming a `.yml` there makes them parse it as Python: this task's first
  run reported `FAIL black --check` and `Found 331 errors` from ruff on
  an otherwise clean tree. Pass source paths or the `hazma test` default,
  and use `--md` for markdown.
- **An exact assertion against a compiled kernel may still be scoped to
  the cargo profile** (inherited from Task 7.1, unchanged here).
