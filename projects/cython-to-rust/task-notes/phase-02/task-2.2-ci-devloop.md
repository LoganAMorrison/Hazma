# Task 2.2: CI, preflight, and dev-loop documentation

**Date:** 2026-08-08
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-02-rust-scaffold.md` (Task 2.2);
`../../rules.md` rule 6 (Rust conventions 1)
**Related ADRs:** ADR-0001 (accepted)
**Depends On:** Task 2.1

## Objective

Make the Rust half of the build a *gated* part of the repo rather than an
accident of the runner image: pin a toolchain in CI, add the three cargo
gates to `scripts/agents/preflight.sh`, and write the `.rs` rebuild loop
into the durable docs so the next agent does not rediscover it.

## Exit Criteria

Copied from the phase file's Task 2.2 block:

- CI installs the Rust toolchain on both OS matrices; full matrix green;
  wheel-build job (`release.yml`) still succeeds with the hybrid build.
  **Hybrid wheels stay CPython-version-tagged** — what is verified here
  is extension-level only: each wheel contains `hazma/_core.abi3.so`,
  i.e. the Rust extension is built against the limited API
  (`abi3-py310`). Distribution-level abi3 wheel tags and the 2-wheel
  matrix are asserted in Phase 07, never earlier.
- `scripts/agents/preflight.sh` grows `cargo fmt --check`,
  `cargo clippy -- -D warnings`, `cargo test` (skipped gracefully when
  `rust/` absent, so pre-Phase-02 branches still preflight).
- `docs/agents/` env notes + `AGENTS.md` Commands section document the
  rebuild loop (when a `.rs` edit requires re-running the editable
  install vs. plain `cargo test`).

## Inputs Reviewed

- `../../PLAN.md`, `../README.md` (project working memory),
  `../../phases/phase-02-rust-scaffold.md`, `../../rules.md`.
- `README.md` (this phase's working memory) — Task 2.1's findings, the
  open question about CI's unpinned toolchain, and the three exact cargo
  spellings it names.
- `.github/workflows/ci.yml`, `.github/workflows/release.yml`,
  `scripts/agents/preflight.sh`, `pyproject.toml`'s `[build-system]`,
  `MANIFEST.in`, `rust/Cargo.toml`, `rust/build.rs`.
- `docs/agents/environment.md`, `docs/agents/preflight.md`,
  `docs/agents/lessons.md` (`[stale-ci-capability-claim]`,
  `[gate-disabled-stays-green]`, `[sibling-copies-of-a-fixed-claim]`,
  `[wheel-tag-vs-extension-abi]`).

## Findings

- **CI was green on borrowed luck.** Task 2.1's hybrid build passed all
  seven checks on PR #55 with no toolchain step anywhere, because the
  GitHub-hosted images ship a usable cargo and setuptools-rust finds it
  unconfigured. Nothing in the repo required that, and because
  `[build-system] requires` now names `setuptools-rust`, an image
  refresh without Rust would not have degraded the Rust half — it would
  have failed the *whole* build, on every matrix entry at once.
- **The two wheel platforms need two different toolchains.**
  cibuildwheel builds the macOS wheels on the runner itself, so a host
  `dtolnay/rust-toolchain` step covers them; it builds the Linux wheels
  inside a manylinux container that cannot see the host, so those need
  rustup installed *in* the container (`CIBW_BEFORE_ALL_LINUX`) and put
  on `PATH` for the build (`CIBW_ENVIRONMENT_LINUX`). Same shape as
  Task 0.4's wheel-vs-sdist lesson: two artifacts, two mechanisms, and
  fixing one has never fixed the other.
- **`release.yml` cannot be verified by a pull request.** Its triggers
  are `release: published` and `workflow_dispatch` only, so the exit
  criterion "wheel-build job still succeeds with the hybrid build" is
  invisible to this branch's checks however green they are. The
  consequence, made concrete by review round 1: such a criterion has to
  be closed by an explicit `gh workflow run … --ref <branch>`, which is
  safe here only because `publish` is gated on
  `github.event_name == 'release'` and therefore skips. §Verification
  carries the run.
- **`cargo build` publishes nothing to Python** — measured, not assumed.
  Cargo works out of `rust/target/`; the importable artifact is
  `hazma/_core.abi3.so`, and only `pip install -e .` puts it there.
  `python -c "import hazma._core; print(hazma._core.__file__)"` resolves
  to `<worktree>/hazma/_core.abi3.so`. This is the `.pyx` rebuild trap
  with an extra step, because the fast iteration command
  (`cargo test`) and the publishing command are now different commands —
  which is exactly what the exit criterion asked to be written down.
- **The rebuild-awareness claim had nine sibling copies.** `rg` over
  `docs/` and both skill trees found the `.pyx` / `.pxd` / `setup.py`
  triplet in `docs/agents/review-lenses.md`, six `.claude/skills/`
  files, and one `.codex/skills/` file. Fixing only the two documents
  the exit criterion names would have left seven authoritative-looking
  copies telling a reviewer that a `.rs` diff needs no rebuild —
  `docs/agents/lessons.md` `[sibling-copies-of-a-fixed-claim]` in its
  across-files shape.
- **Two CI-capability claims were already stale before this task
  touched them**, both about the parity corpus: `docs/agents/`'s
  preflight gate 4 said a bare `pytest` is "deliberately the same
  collection `.github/workflows/ci.yml` runs" and its matrix bullet said
  "the parity corpus included, on every entry". CI has run
  `--ignore=test/parity` everywhere except macOS since PR #52/#53. Both
  corrected here, since editing a durable doc puts its facts in scope
  (`[touched-doc-inherits-its-citations]`, `[stale-ci-capability-claim]`).

## Decisions and Implementation Notes

- **The cargo gates go before pytest, not after.** They cost seconds;
  the bare suite costs minutes, nearly all of it the parity corpus. A
  rustfmt diff should not wait behind it. Cost: three trailing gates
  renumbered in `preflight.sh` and `docs/agents/preflight.md`, which is
  the same churn either insertion point would have caused.
- **`--manifest-path rust/Cargo.toml` rather than `cd rust`.** The
  script anchors itself to `REPO_ROOT` so every gate runs against the
  worktree containing it; a `cd` inside one gate would leak into the
  ones after it.
- **Three rows, not one.** Each cargo command gets its own PASS/FAIL
  row, so a failure names which of the three failed without reading the
  captured output.
- **Absence rules mirror the existing table.** No `rust/` → SKIP
  (pre-Phase-02 branches must still preflight, per the exit criterion);
  `rust/` present but no `cargo` → WARN, the same "gate did not run"
  hole already used for a missing isort or ruff, and documented as such
  in `preflight.md`.
- **`cargo_gate()` reads its status as `if capture ...; then`** rather
  than the `$?` idiom its four Python-gate siblings use. Deliberate
  rather than inconsistent: shellcheck flags the older form (SC2181,
  style-only), and the counts are 5 on `origin/master` and 5 here — the
  new helper did not add a sixth. Rewriting the four existing sites
  would be unrelated churn in a task about Rust.
- **Two widenings past the literal exit criteria**, both recorded in the
  phase file rather than only here:
  1. *The cargo gates also run in CI*, as a dedicated `rust` job. The
     criterion asked only that CI install a toolchain. But
     `preflight.sh` is local discipline that nothing enforces, and
     Phases 03–06 land the entire numerics layer in Rust; a formatting
     and clippy gate that no PR check runs would be decorative by
     Phase 04. `rules.md` Rust rule 1 makes the three gates canonical,
     so this puts them somewhere that can fail.
  2. *The wheel assertion is a job step, not an eyeball.* The criterion
     says each wheel must contain `hazma/_core.abi3.so`. Since
     `release.yml` never runs on a PR, checking that by hand once would
     verify this branch and nothing after it. The step also fails on an
     empty `wheelhouse/`, because a check that verifies nothing and
     reports success is `[gate-disabled-stays-green]` waiting to happen.
- **The filename is the abi3 assertion.** PyO3's `abi3-py310` feature is
  what makes setuptools-rust install the cdylib as `_core.abi3.so`; a
  plain `_core.cpython-312-*.so` would mean the feature silently stopped
  applying. The *wheel* stays CPython-tagged and will until the last
  Cython extension goes in Phase 06 — `lessons.md`
  `[wheel-tag-vs-extension-abi]`, and the phase file says so twice.
- **`dtolnay/rust-toolchain@stable`** for the toolchain, matching how
  the repo already pins actions by tag (`pypa/cibuildwheel@v4.1.1`,
  `actions/checkout@v7`). The `rust` job installs a Python for one
  reason: `cargo test --no-default-features` links libpython for real.
- **Review round 1 (PR #56) landed two fixes**, both accepted:
  1. *An inserted gate renumbered the list and orphaned the prose that
     pointed at it.* `docs/agents/preflight.md`'s "Markdown rules"
     section still opened "Gate 6 runs against the committed
     `.markdownlint.jsonc`" — true before this task inserted the cargo
     gates, and pointing at `cargo test` after. The class fix went wider
     than the cited line: `rg 'Gate [0-9]|gate [0-9]'` found six more
     live references, and rather than re-pin them to numbers that will
     shift again at the next insertion, the markdownlint ones are now
     **named** (`the markdownlint gate`). New `lessons.md` entry
     `[renumbered-list-orphans-its-references]`.
  2. *A `Complete` status cannot sit on an unrun exit criterion.* The
     evidence mapping said `release.yml` had never run; the status said
     done. Resolved by running it — see §Verification — rather than by
     softening either. The reviewer's alternative (leave the task
     incomplete) was the right fallback and would have been taken had
     the dispatch failed.

## Files Changed

- `.github/workflows/ci.yml` — new `rust` job (fmt / clippy / test);
  `dtolnay/rust-toolchain@stable` before the build in every test-matrix
  entry.
- `.github/workflows/release.yml` — host toolchain step (macOS);
  `CIBW_BEFORE_ALL_LINUX` + `CIBW_ENVIRONMENT_LINUX` (manylinux
  container); `hazma._core` added to `CIBW_TEST_COMMAND`; new step
  asserting every wheel carries `hazma/_core.abi3.so` and failing on an
  empty `wheelhouse/`.
- `scripts/agents/preflight.sh` — `cargo_gate()` helper and gates 4–6;
  three trailing gates renumbered; header block and `--paths` help
  updated.
- `AGENTS.md` — Commands section gains the three cargo spellings and the
  `.rs` rebuild loop; the compiled-layer sentence, the layout tree
  (`rust/`) and Layering item 1 reconciled with the migration.
- `docs/agents/environment.md` — four new Build-and-imports entries
  (cargo required to build at all, `.rs` staleness, `cargo build` ≠
  rebuild, `--no-default-features`); CI matrix bullet corrected and the
  `rust` job described.
- `docs/agents/preflight.md` — gates 4–6 added with their absence rules,
  gates 7–11 renumbered, the import-smoke gate extended to `rust/`, the
  stale "parity corpus included" claims corrected, and the CI-floor,
  critical-path and WARN paragraphs reconciled.
- `docs/agents/README.md`, `docs/agents/review-lenses.md`,
  `.claude/skills/{task-pipeline,review-pr,commit-and-pr,execute-single-task,review-respond,review-cycle}/SKILL.md`,
  `.codex/skills/execute-single-task/SKILL.md` — the rebuild-awareness
  sweep.
- `projects/cython-to-rust/phases/phase-02-rust-scaffold.md` — canonical:
  Task 2.2 exit criteria gain the exact cargo spellings and a bullet
  recording both widenings.
- `projects/cython-to-rust/rules.md` — canonical: Rust rule 1 notes the
  gates now run in CI too and points at the phase file for the exact
  spellings.
- `projects/cython-to-rust/task-notes/README.md`,
  `projects/cython-to-rust/task-notes/phase-02/README.md`, and this note
  — bookkeeping.
- `projects/cython-to-rust/task-notes/phase-02/task-2.1-crate-skeleton.md`
  — one line: its "CI carries no Rust toolchain step (Task 2.2)" risk
  bullet struck through and dated, since a bald present-tense claim in a
  Handoff section reads as current no matter which note it sits in.

Added in review round 1:

- `docs/agents/lessons.md` — two entries,
  `[renumbered-list-orphans-its-references]` and
  `[unrun-workflow-cannot-close-a-criterion]`, both citing PR #56.
- `docs/followups/todo/markdownlint-skips-skill-file-shapes.md` (3
  occurrences) and `docs/followups/done/markdownlint-config-for-templates.md`
  (2) — ordinal references to the markdownlint gate, now named rather
  than numbered.

24 files: 23 modified, 1 added. Nothing under `hazma/` or `rust/`.

## Verification

Environment: the worktree was cleaned of stale build artifacts
(`find hazma -name '*.c' -o -name '*.cpp' -o -name '*.so' | xargs rm -f`)
and rebuilt from scratch — `uv venv --python 3.12` then
`uv pip install -e . --group dev` on CPython 3.12.12, macOS/arm64, the
corpus's capturing interpreter. `python -c "import hazma; …"` resolves
inside the worktree.

**The gate.** `scripts/agents/preflight.sh --paths "setup.py" --md
"<16 changed docs>"` → **RESULT: PASS**, all eleven rows:

```text
PASS   black --check           setup.py
PASS   isort --check-only      setup.py
PASS   ruff check              setup.py
PASS   cargo fmt --check       rust/
PASS   cargo clippy            rust/
PASS   cargo test              rust/
PASS   pytest                  1009 passed, 13 skipped, 6 warnings in 571.34s (0:09:31)
PASS   import hazma            version 2.1.0
PASS   markdownlint            AGENTS.md docs/agents/… (16 files)
SKIP   version bump            not a closing PR (pass --closing)
PASS   forbidden tokens        none added
```

`--paths setup.py` rather than the default: **this branch touches no
Python at all** (`git diff origin/master -- '*.py'` is empty), and the
default `hazma test` is the directory form that returns thousands of
pre-existing findings —
[`../../../../docs/followups/todo/preflight-isort-ruff-red-on-trunk.md`](../../../../docs/followups/todo/preflight-isort-ruff-red-on-trunk.md).
`setup.py` is a real, checked file that keeps the three Python rows
honest instead of omitted.

**What the pytest row covers, and what it proves.** 1022 collected;
`1009 passed, 13 skipped` is byte-identical to Task 2.1's closing state,
which is the evidence that this task moved no test outcome. Categories:
the in-package `*_test.py` modules, the `test/` tree (spectra,
mediators, `rh_neutrino`, theory aggregation, utils), and the 626-block
golden parity corpus over all 41 consumed entry points. The corpus ran
in **bit-equality mode** — `pytest -q test/parity -k "capturing_tree or
provenance or rust_core"` → `1 passed, 628 deselected`, i.e.
`test/parity/test_parity.py::test_running_on_the_capturing_tree` passed
rather than skipping, which is how the runner reports its mode.

**The three cargo gates, run directly as well as through the gate:**
`cargo fmt --manifest-path rust/Cargo.toml --check` → exit 0;
`cargo clippy --manifest-path rust/Cargo.toml --all-targets --
-D warnings` → exit 0; `cargo test --manifest-path rust/Cargo.toml
--no-default-features` → `test result: ok. 2 passed; 0 failed` plus 0
doc-tests, exit 0.

**Every branch of the new gate block was executed, not reasoned about:**

| Branch | How it was forced | Result |
| --- | --- | --- |
| `rust/` absent → SKIP | copied `preflight.sh` into a scratch tree with no `rust/` and ran it | three `SKIP … no rust/ crate in this tree` rows |
| `cargo` absent → WARN | `env PATH=/usr/bin:/bin bash scripts/agents/preflight.sh …` | three `WARN … cargo not installed` rows |
| gate fails → FAIL | appended `fn  badly_formatted( )->i32{42}` to `rust/src/kernels.rs`, ran, then `git checkout --` it | `FAIL cargo fmt --check` and `FAIL cargo clippy` (dead-code under `-D warnings`); `git status --short rust/` clean afterwards |
| all green → PASS | the gate run above | three `PASS … rust/` rows |

**The wheel / abi3 criterion, measured on the final tree.** `uv build
--wheel` produced `hazma-2.1.0-cp312-cp312-macosx_11_0_arm64.whl`, and
the *exact* script from `release.yml`'s new assertion step, run against
a `wheelhouse/` holding it, printed
`ok   hazma-2.1.0-…whl` / `1 wheel(s) carry hazma/_core.abi3.so`, exit
0. The same script against an empty `wheelhouse/` exits 1 with
`no wheels in wheelhouse/ — nothing was verified`. Note the wheel tag is
`cp312-cp312` because a 3.12 venv built it — the invariant is
`cp<XY>`, **never** `abi3`, while any Cython extension remains
(`docs/agents/lessons.md` `[wheel-tag-vs-extension-abi]`); the abi3
property is the `.so` filename inside.

**`build-sdist` deliberately got no toolchain, and that was checked:**
with cargo removed from `PATH` (`command -v cargo` → empty),
`uv build --sdist` succeeded — building an sdist packages sources and
compiles nothing. The tarball carries the crate (13 paths under
`hazma-2.1.0/rust/`: `Cargo.toml`, `Cargo.lock`, `build.rs`, `src/*.rs`).

**Workflow syntax:** both files parse (`yaml.safe_load`), `ci.yml` has
jobs `[lint, rust, test]`, the folded scalars render as the exact
commands documented in the phase file, and the toolchain step precedes
`Build and install hazma` in the test job.

**Static checks on the changed non-Python files:**
`shellcheck scripts/agents/preflight.sh` → 5 SC2181 (style) hits,
**the same 5 as `origin/master`** — the new helper added none.
`scripts/agents/check_doc_citations.py <21 docs>` → `docs scanned: 21`,
`in-repo citations checked: 11`, `out-of-range or ambiguous: NONE`
(non-zero scope, per `lessons.md` `[changed-vs-sees-only-commits]`).

**`release.yml`, observed** (added in review round 1 — the reviewer was
right that a Complete status could not sit on top of an unrun exit
criterion). `gh workflow run release.yml --ref
claude/cython-to-rust/task-2.2-ci-preflight-dev-loop-docs` →
[run 31297673951](https://github.com/LoganAMorrison/Hazma/actions/runs/31297673951),
**conclusion `success`**:

| Job | Conclusion | Evidence |
| --- | --- | --- |
| Build sdist | success | no toolchain step, as designed — an sdist packages sources and compiles nothing |
| Build wheels (macos-latest) | success | host `dtolnay/rust-toolchain` path |
| Build wheels (ubuntu-latest) | success | container path — `Running before_all…` then `Rust is installed now. Great!` |
| Publish to PyPI | **skipped** | `if: github.event_name == 'release'` held; a dispatch run publishes nothing |

The assertion step printed real output on both platforms — 10 wheels, the
unchanged CPython-tagged matrix the phase file predicts, each carrying
the abi3 extension:

```text
ok   hazma-2.1.0-cp310-cp310-macosx_11_0_arm64.whl
… cp311, cp312, cp313, cp314 …
5 wheel(s) carry hazma/_core.abi3.so

ok   hazma-2.1.0-cp310-cp310-manylinux_2_28_x86_64.whl
… cp311, cp312, cp313, cp314 …
5 wheel(s) carry hazma/_core.abi3.so
```

**"Full matrix green"** is likewise now observed rather than deferred:
all eight checks on [PR #56](https://github.com/LoganAMorrison/Hazma/pull/56)
passed — `Lint` 16s, `Rust (fmt, clippy, test)` 30s, and the six `Test`
entries 11m52s–19m50s. The `rust` job passing is what settles the one
risk this task could not test locally: `cargo test
--no-default-features` links libpython for real, and whether
`actions/setup-python`'s interpreter satisfies that on a runner was
reasoned, not measured. It does.

## Open Questions

- ~~**`release.yml` has never run with any of this.**~~ — **closed in
  review round 1**, which is where it belonged: a reviewer refused a
  `Complete` status sitting on top of an unrun exit criterion, and was
  right to. Dispatched run 31297673951 is green on both platforms with
  `publish` skipped; details in §Verification. The general lesson is
  worth keeping: a workflow with no pull-request trigger is invisible to
  PR checks, so its criteria need an explicit dispatch, not an argument
  from plausibility. Phase 07 Task 7.1 rewrites this job for maturin and
  inherits the same obligation.
- **The `Dockerfile` builds hazma from a fresh clone in an image with no
  cargo**, so it broke the moment Task 2.1 landed. No follow-up filed:
  `phases/phase-07-cutover.md` Task 7.3 already owns removing it
  alongside `requirements.txt`, and
  [`../../../../docs/followups/todo/sdist-ships-generated-c-and-docs.md`](../../../../docs/followups/todo/sdist-ships-generated-c-and-docs.md)
  records that assignment. Noted here so the breakage is on the record
  between now and then.

## Plan Impact

**Impact Level:** Phase file patched (no ADR).

Two canonical documents changed, both because this task's output made
their existing text incomplete rather than wrong:

- `phases/phase-02-rust-scaffold.md`, Task 2.2 exit criteria — the
  spellings that actually run (two carry a load-bearing flag the
  shorthand omits), plus a bullet recording the two widenings above, so
  the plan says what shipped instead of leaving it inferable from the
  diff. Same precedent as Task 2.1's patch to its own criteria.
- `rules.md` Rust rule 1 — the three gates now run in CI as well as
  preflight, and the rule points at the phase file rather than
  restating spellings that would then exist in two places.

No ADR: nothing here revises ADR-0001, and no numerical, interface or
ordering contract moved. `PLAN.md` needs no edit — its Phase 02 row
describes the phase's deliverable, which is unchanged.

## Stale-state sweep

Run against this branch after every prose edit was frozen.

**Identifier sweep** — the three names this task introduces:

```console
$ rg -n 'cargo_gate' projects/ docs/ scripts/ .github/
scripts/agents/preflight.sh:212,213,234,236,238        KEPT (definition + 3 calls)
projects/…/task-2.2-ci-devloop.md:117,162              KEPT (this note)
projects/…/phase-02/README.md:151                      KEPT (phase memory)

$ rg -n 'dtolnay/rust-toolchain' projects/ docs/ .github/
.github/workflows/ci.yml:50,88                         KEPT (rust job, test matrix)
.github/workflows/release.yml:24                       KEPT (macOS host)
docs/agents/environment.md:233                         EDITED (CI matrix bullet)
projects/…/phase-02/README.md:85,205                   EDITED
projects/…/task-2.2-ci-devloop.md:62,144,155           KEPT (this note)

$ rg -n 'CIBW_BEFORE_ALL_LINUX|CIBW_ENVIRONMENT_LINUX' projects/ docs/ .github/
.github/workflows/release.yml:28,41,43,46              KEPT
projects/…/phase-02/README.md:86,87                    EDITED
projects/…/task-2.2-ci-devloop.md:64,65,158            KEPT (this note)
```

**Line-number citation sweep** — no `file:line` citation into code was
added or invalidated; the mechanical check over all 21 touched docs,
re-run after the last prose edit:

```console
$ scripts/agents/check_doc_citations.py <21 touched docs>
docs scanned: 21
in-repo citations checked: 11
  resolved by exact: 8
  resolved by suffix: 2
  resolved by context: 1
external citations skipped: 1
out-of-range or ambiguous: NONE
```

**Markdown-link sweep** (the checker does not cover these —
`lessons.md` `[status-encoding-path-reference]`): every relative link in
every changed `.md` was resolved with `[ -e ]`. Two hits, both the same
pre-existing forward reference to the note Task 2.3 creates —
`phases/phase-02-rust-scaffold.md` and `task-notes/phase-02/README.md`
each carry one, and `git show origin/master:<file> | grep -c` returns 1
for both, so neither is this diff's. KEPT.

**markdownlint** over all 17 changed docs except
`.claude/skills/task-pipeline/SKILL.md` → exit 0. The invocation was
proved to have non-zero scope rather than trusted: markdownlint prints
its usage banner and exits **0** on arguments that match nothing (an
earlier run with a trailing empty argument did exactly that), so a
control file containing `#  bad` was linted with the same command shape
and returned `MD019` with exit 1.

**Forward-looking / stale-claim sweep:**

```console
$ rg -n '(Task [0-9.]+ will|will be added|still pending|today: ?stub)' \
    projects/cython-to-rust/task-notes/phase-02/ \
    projects/cython-to-rust/phases/phase-02-rust-scaffold.md \
    docs/agents/preflight.md docs/agents/environment.md AGENTS.md
(no hits outside task-2.1's own quoted sweep block)

$ rg -n 'no Rust toolchain|has no Rust|parity corpus included' projects/ docs/ .claude/
projects/…/task-2.1-crate-skeleton.md:675   EDITED — struck through, closed by this task
projects/…/phase-02/README.md:197           EDITED — Open Question closed
projects/…/task-notes/README.md:806         EDITED — risk bullet closed
projects/…/task-2.2-ci-devloop.md:93,174    KEPT (this note describes the fix)
docs/agents/preflight.md                    EDITED — both "parity corpus included" claims gone
```

**Gate-ordinal sweep (review round 1).** Inserting the cargo gates as
4–6 pushed markdownlint from gate 6 to gate 9, and prose elsewhere still
pointed at the old ordinals. The renumbering *inside* the list was
correct; what rotted were references from outside it.

### Pre-fix occurrences

```console
$ rg -n 'Gate 6|gate 6' docs/ .claude/ .codex/ AGENTS.md projects/ scripts/
docs/agents/preflight.md:106                      → markdownlint   EDITED
docs/followups/todo/markdownlint-skips-skill-file-shapes.md:21     EDITED
docs/followups/todo/markdownlint-skips-skill-file-shapes.md:68     EDITED
docs/followups/todo/markdownlint-skips-skill-file-shapes.md:77     EDITED
docs/followups/done/markdownlint-config-for-templates.md:48        EDITED
docs/followups/done/markdownlint-config-for-templates.md:102       EDITED
projects/…/phase-00/task-0.2-delete-mc-slice.md:390                KEPT
projects/…/phase-01/task-1.3-test-wiring.md:405                    KEPT
```

The two `KEPT` hits are dated task-note records of what a *past* PR ran,
not instructions to a future reader; the repo treats task notes as
history (`../README.md`: "the learnings are the distillation, the notes
are history"). Rewriting them would falsify the record. Everything
prescriptive was fixed, including the `done/` follow-up, whose line 102
made a present-tense claim about `preflight.sh`'s current behavior.

Fixed by **naming** the gate rather than renumbering it, so the next
insertion cannot rot the same text again.

### Post-fix occurrences

```console
$ rg -n 'Gate 6|gate 6' docs/ .claude/ .codex/ AGENTS.md projects/ scripts/
projects/…/phase-00/task-0.2-delete-mc-slice.md:390                KEPT
projects/…/phase-01/task-1.3-test-wiring.md:405                    KEPT

$ rg -n 'Gate [0-9]+|gate [0-9]+|Gates [0-9]' docs/agents/ docs/followups/ \
    scripts/agents/preflight.sh
scripts/agents/preflight.sh:138,156,174,192,243,268,281,317,364   1,2,3,4-6,7,8,9,10,11 — match the implementation
docs/agents/preflight.md:106                       gate 9 = markdownlint ✓
docs/agents/preflight.md:170                       gate 7 = pytest ✓
docs/followups/todo/preflight-isort-ruff-red-on-trunk.md:20,24     gates 2,3 = isort, ruff ✓ (unmoved)
```

**Count sweep:**

| Claim location | Command | Actual | Status |
| --- | --- | --- | --- |
| §Verification "1009 passed, 13 skipped" | `preflight.sh` pytest row | `1009 passed, 13 skipped … 571.34s` | OK |
| §Verification "1022 collected" | 1009 + 13 | 1022 | OK |
| §Verification "2 passed" cargo tests | `cargo test --manifest-path rust/Cargo.toml --no-default-features` | `ok. 2 passed; 0 failed` | OK |
| §Verification "5 SC2181, same as master" | `shellcheck … \| grep -c SC2181` on both trees | 5 and 5 | OK |
| §Findings "nine sibling copies" | `rg` for the `.pyx`/`.pxd` triplet | 1 in `docs/agents/`, 7 under `.claude/skills/`, 1 under `.codex/skills/` = 9 | OK |
| §Verification "21 docs scanned" | `check_doc_citations.py` summary, re-run last | `docs scanned: 21` | OK |
| §Verification "13 paths under `rust/`" | `tar tzf … \| grep '^hazma-2.1.0/rust/'` | 13 (incl. 2 directory entries) | OK |
| phase README "7 markdownlint errors, both trees" | `markdownlint --dot` on branch vs `origin/master` copy | 7 and 7 (task-pipeline only; other six 0/0) | OK |
| §Files Changed file list | `git diff origin/master --name-status` | 24 files: 23 `M` + 1 `A` | OK |

**Numerical-impact statement:** **No public value changes** (verified:
`git diff origin/master -- hazma rust` → 0 lines, and
`git diff origin/master -- setup.py pyproject.toml MANIFEST.in` → 0
lines). No library module, kernel, signature, constant or build *input*
is reachable from this diff, so no grid evaluation applies; the compiled
artifacts are the trunk's. Positive corroboration rather than absence
alone: the parity corpus ran in bit-equality mode over all 41 consumed
entry points and passed, and a wheel built from this branch carries
`hazma/_core.abi3.so`.

**Exit Criteria → evidence mapping:**

| Exit-criterion bullet | Evidence | Status |
| --- | --- | --- |
| CI installs the Rust toolchain on both OS matrices | `ci.yml:88`, a step in the `test` job, which runs `ubuntu-latest` ×5 and `macos-latest` ×1; YAML parsed and step order confirmed | Done |
| Full matrix green | PR #56: all eight checks pass — `Lint` 16s, `Rust (fmt, clippy, test)` 30s, six `Test` entries 11m52s–19m50s | Done |
| `release.yml` wheel job still succeeds with the hybrid build | Dispatched run 31297673951 → `success`; both `build-wheels` jobs and `build-sdist` green, `publish` skipped | Done |
| Each wheel contains `hazma/_core.abi3.so` | The assertion step's own output on both platforms: `5 wheel(s) carry hazma/_core.abi3.so` ×2. Locally it also exits 1 on an empty `wheelhouse/` | Done |
| Hybrid wheels stay CPython-tagged | All 10 released wheels are `cp310`–`cp314` × {`macosx_11_0_arm64`, `manylinux_2_28_x86_64`}; no `abi3` wheel tag anywhere | Done |
| `preflight.sh` grows the three cargo gates | Gates 4–6; the gate run above shows three `PASS … rust/` rows | Done |
| Skipped gracefully when `rust/` absent | Forced in a scratch tree → three `SKIP … no rust/ crate in this tree` rows | Done |
| `docs/agents/` env notes document the rebuild loop | `docs/agents/environment.md` — four new Build-and-imports entries; `docs/agents/preflight.md` gates 4–6 and gate 8 | Done |
| `AGENTS.md` Commands section documents the rebuild loop | Commands block gains the three cargo spellings; new "Editing a `.rs`" paragraph naming `cargo build` as *not* the rebuild | Done |

**Task-note self-consistency:** `**Status:** Complete` matches the phase
README's Tasks-table cell and the project README's Phases row; every
file named in §Files Changed appears in `git diff origin/master
--name-status` (24 files: 23 modified, 1 added — this note); no
function, flag or identifier cited in §Findings or §Decisions is absent
from the diff. The `Complete` status now rests on an evidence mapping
whose every row reads `Done` — which was review round 1's whole point.

## Handoff to Next Task

- **Read first:** `../README.md` (project working memory) and
  [`README.md`](README.md) (this phase's), then the phase file. Task 2.3
  is the last task in Phase 02 and depends only on Task 2.1 — nothing
  here blocks it.
- **Now safe to assume:** the three cargo gates run themselves, in
  `preflight.sh` (gates 4–6, ahead of pytest) and in CI's `rust` job;
  every CI entry installs a toolchain; and the `.rs` rebuild loop is
  documented in `AGENTS.md`, `docs/agents/environment.md`,
  `docs/agents/preflight.md` and every review skill. A future PR that
  cites `cargo test` output as evidence about Python behavior should now
  be challenged by the reviewer roster, not just by whoever remembers.
- **Nothing left risky in this task.** Both items that were open at
  hand-off closed with evidence: the `rust` job's `cargo test` step does
  link libpython through `actions/setup-python`'s interpreter on a
  runner (PR #56, 30s), and `release.yml` is green end to end on a
  dispatched run. What survives is a habit, not a risk — a workflow
  without a pull-request trigger has to be dispatched deliberately or
  its exit criteria stay unmeasured.
