# Task 6.4: Retire the capi survivors and the `_utils` headers

**Date:** 2026-08-27
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-06-mediator-spectra.md` (Task 6.4,
Exit Criteria); `../../PLAN.md` (Scope, Phases); `../../rules.md` rules 1,
4, 10
**Related ADRs:** ADR-0001, ADR-0002
**Depends On:** Tasks 6.2, 6.3

## Objective

Delete the last Cython in the tree — the four capi-survivor spectra
extensions and the four `hazma/_utils/` headers — strip `setup.py` to the
Rust extension alone, and close Phase 06. Nothing is swapped here: all 41
consumed entry points already run on `hazma._core`, so this is deletion
and build plumbing only.

## Exit Criteria

From the phase file's Task 6.4 block:

- `rg "cimport|__pyx_capi__|\.pxd"` over `hazma/` confirms zero
  consumers; then delete the four capi-survivor extensions
  (`_photon/_muon`, `_photon/_pion`, `_positron/_muon`,
  `_positron/_pion` `.pyx`+`.pxd`), `hazma/_utils/boost.{pyx,pxd}`,
  `constants.pxd`, `kinematics.pxd`, `legacy_parameters.pxd`, and the
  `spectra/_neutrino/_neutrino` struct module.
- `find hazma -name "*.pyx" -o -name "*.pxd"` returns **nothing**;
  `setup.py` builds only `hazma._core`; full suite + corpus green.

Phase-level, since this task closes Phase 06:

- Zero Cython in the tree; all 41 consumed entry points on `hazma._core`.
- Drift table complete in `../numerical-impact.md`.
- Phase learnings written to
  `../../learnings/phase-06-mediator-spectra.md`; phase file frontmatter
  `status: Complete`; `PLAN.md` Phases-table cell updated; the project
  README's Phase 06 entries swept into `history-*.md`.

Carried in from the Task 6.3 handoff:

- Decide the restore-revision follow-up rather than defer it again.
- State explicitly that `test/parity/oracles` needs no re-capture.
- `rm` the orphaned `.so`/`.c` beside each deleted source, or the next
  `pip install -e .` measures extensions nothing builds.

## Inputs Reviewed

- `../../PLAN.md`; `../README.md`; `README.md` (this phase);
  `../../rules.md`; `../../phases/phase-06-mediator-spectra.md` Task 6.4;
  `../../phases/phase-07-cutover.md` Task 7.1 (scope boundary).
- `task-6.3-positron-spectra.md` `## Handoff`.
- `docs/agents/lessons.md`; `docs/agents/environment.md`;
  `docs/agents/doc-consistency.md`.

## Findings

- **The `rg` sweep was clean, as the handoff predicted, but the *tests*
  were not.** No module under `hazma/` read a doomed file, so the library
  half of this task was a plain `git rm`. Six test modules were another
  matter: five imported a doomed extension at module scope and one parsed
  two of the deleted headers. That is where the whole of this task's risk
  sat, and the phase file's exit criteria do not mention it.
- **`test/test_core_boost.py` lost nothing by losing its oracle**, and
  that was designed in on 2026-08-12. Its `TestFusedArithmetic` already
  swept all five live photon tables x 6 boost regimes x 300 energies and
  40,000 delta-function draws **bit-for-bit** against
  `integrate_reference` / `delta_function_reference` with `mul_add=fma` —
  strictly stronger than the two `…matches_the_cython…` sweeps it
  replaced, because the reference is the same number on every platform
  while the Cython was the same number only on macOS/arm64. Nine further
  tests corroborated a hand-computed value *with* the Cython; those were
  repointed to the same reference rather than deleted.
- **Repointing to the reference exposed one real divergence, in Python
  rather than in the port.** At `beta == 1` both implementations pass the
  shared guard (`beta > 1.0`, not `>=`) and compute `gamma = 1/sqrt(0)`;
  Rust yields `+inf` under IEEE-754 so the height underflows to `0.0`,
  while Python raises `ZeroDivisionError`. The port is right and the
  transcription is simply undefined there — recorded in the test rather
  than papered over.
- **`test/test_core_constants.py` had to go, and its own docstring said
  so** ("when the last `.pxd` goes in Phase 06, delete the module"). It
  parsed `constants.pxd`, `legacy_parameters.pxd` and three `.pyx` and
  compared ~220 constants bit-for-bit against `rust/src/constants.rs`.
  **It was run green one last time against `origin/master` before the
  deletion — 21 passed** — so the transcription is verified correct at
  the moment its oracle disappeared.
- **The knowledge in that module survives; only the text-parsing does.**
  `constants.rs`'s own five `cargo` tests already assert the rule-4
  divergences, that `photon_pion` mixes the two tables in both
  directions, `R_FACTOR`'s provenance, and the const-folding. What is
  genuinely unpinned afterwards is a constant that **no entry point
  reads** — the corpus pins the rest through the 41 entry points. Stated
  in the module header rather than left implicit.
- **The restore-revision recursion is not real, and that unblocks the
  follow-up.** `capture.py` resolves a revision with
  `git show <rev>:<path>`, which does not care whether `<rev>` is a SHA
  or a `^` expression. A task that cannot know its own commit can still
  name a revision that **already exists** — here `1b022d4`, the
  `origin/master` this branch was cut from, where all twelve files it
  deletes are present in final form. Both 6.2's and 6.3's SHAs resolved
  normally (`7594761^`, `c384aff^`).
- **A complete roster needs the compile closure, not just the patched
  files.** Defect A3 patches `_photon/_pion.pyx` and A4
  `_positron/_muon.pyx`, but neither compiles without its `.pxd`, its
  cimported twin, `_utils/boost.{pyx,pxd}` and `constants.pxd`. The
  roster went 13 -> 29 entries for that reason, and every one was
  verified to resolve against git.
- **A consumer sweep scoped to `hazma/` and `test/` misses CI, and this
  one did.** The first push went red on every `Test` job:
  `.github/workflows/ci.yml`'s import smoke test named
  `hazma.spectra._photon._muon` explicitly, and `release.yml`'s
  `CIBW_TEST_COMMAND` named it too — both deliberately, to import *a
  compiled extension* rather than the pure-Python package, so a mistagged
  or broken wheel fails the build. `docs/agents/environment.md` carried
  the same import as its "confirm the `.so` is in your worktree" recipe.
  All three now name `hazma._core`, the only compiled module left.
  **The rule the phase file's `rg` gate implies is too narrow: a
  deletion sweep has to cover `.github/` and `scripts/` as well as the
  package and the suite**, and `pytest` cannot see the gap because the
  workflows are not Python the suite imports.
- **A test caught a bug in a test I wrote in the same pass.**
  `assert not path.glob(...)` is always false — `glob` returns a
  generator, which is truthy — so the replacement for
  `test_nothing_cimports_this_extension_any_more` passed vacuously in the
  wrong direction and failed loudly in the right one. `list(...)` is the
  fix, and the episode is the reason the new module asserts on a
  materialised `== []` rather than on truthiness.

## Decisions and Implementation Notes

- **`setup.py` is stripped, not deleted** — Phase 07 Task 7.1 owns its
  removal along with the maturin switch. It now imports only
  `setuptools` and `setuptools_rust` and declares one `RustExtension`.
- **`cython`, `numpy` and `scipy` leave `[build-system] requires` here**,
  even though Task 7.1's exit criteria also claim that deletion. They
  existed solely for the `.pyx` — numpy's headers, the compiler, and
  `scipy.special.cython_special.pxd` — and after this task nothing
  build-time reads any of them, so leaving them would be a false
  requirement. Task 7.1's bullet is patched to say so.
- **`MANIFEST.in`'s `global-include` drops `*.pyx *.pxd *.c`.** The `*.c`
  pattern never matched a tracked file (this repo commits no generated C)
  and only swept a local build's gitignored output into an sdist made
  from a dirty tree, so removing it fixes a real leak rather than tidying.
- **A new `test/test_no_cython_remains.py` makes the exit criterion
  executable.** Each swap task asserted its own twin's absence in its own
  module, which is right while twins remain; the tree-wide claim needed a
  home that outlives the project. Four tests, all asserting on sources
  and build declarations rather than on `ImportError`, and all four were
  confirmed to fail against `origin/master`.
- **The four `capsule` oracle Sources became `restored`.** With the
  `.pyx` deleted there is no `__pyx_capi__` to read. The `capsule` *kind*
  stays in `entry_points.py` for the same reason `live` does: the
  committed `data/manifest.json` records it for captures taken while the
  capsules existed.
- **Re-capture is left possible rather than declared impossible.**
  `oracles/README.md` said it "cannot be done at all once Task 6.4
  lands". With the roster complete it can, so the text now describes what
  it costs — restore every source in the closure, plus `setup.py` and
  `pyproject.toml` — instead of forbidding it.

## Files Changed

**Deleted (14 tracked Cython sources):**
`hazma/_utils/{boost.pyx,boost.pxd,constants.pxd,kinematics.pxd,legacy_parameters.pxd,kinematics.pyx.bak}`,
`hazma/spectra/_photon/{_muon,_pion}.{pyx,pxd}`,
`hazma/spectra/_positron/{_muon,_pion}.{pyx,pxd}`.
`kinematics.pyx.bak` was a tracked backup file no rule covered; it went
with the header it shadowed.

**Deleted (test):** `test/test_core_constants.py` — its oracle was two of
the deleted headers.

**New:** `test/test_no_cython_remains.py`.

**Build and CI:** `setup.py` (Cython half removed), `pyproject.toml`
(`[build-system] requires`, one stale comment), `MANIFEST.in`,
`.github/workflows/{ci,release}.yml` (both named a deleted module in an
import smoke test), `docs/agents/environment.md` (the same import, as a
recipe).

**Tests repaired:** `test/test_core_boost.py` (Cython oracle retired,
nine tests repointed to the in-module reference, docstring rewritten),
`test/test_core_{photon_muon,photon_pion,positron_muon,positron_pion}.py`
(twin classes and capsule assertions retired, docstrings rewritten),
`test/test_core_{dispatch,neutrino}.py` (stale comments).

**Parity/oracles:** `test/parity/oracles/defects.py` (`RESTORED_SOURCES`
13 -> 29), `test/parity/oracles/entry_points.py` (four `capsule` Sources
-> `restored`, docstring), `test/parity/oracles/README.md`,
`test/parity/cases.py`, `test/parity/README.md`.

**Rust (documentation only, no code):** `rust/src/{boost.rs,boost_probe.rs,lib.rs,constants.rs,kernels/neutrino_muon.rs}`.

**Other:** `hazma/_core.pyi`.

**Follow-up:** `docs/followups/todo/oracle-restore-revisions-for-the-mediator-decay-pyx.md`
-> `done/` with a `## Resolution`; `docs/followups/README.md` row moved;
three live inbound links repointed.

**Project bookkeeping:** this note, `../README.md` (phase),
`../../phases/phase-06-mediator-spectra.md` (status + Task 6.4 gate),
`../../phases/phase-07-cutover.md` (Task 7.1 bullet),
`../../PLAN.md` (Phases table),
`../../learnings/phase-06-mediator-spectra.md` (new),
`../README.md` + `../history-*.md` (phase-close sweep).

## Verification

Environment: macOS 26.5.2 / arm64, CPython 3.12.12, NumPy 2.5.1, SciPy
1.18.0 — the parity corpus's own capture platform, so `EXACT` cases run
in bit-equality mode. Rebuilt with `uv pip install -e .
--no-build-isolation` after the deletions; `hazma._core.__file__` and
`hazma.__file__` both resolve inside this worktree, and
`find hazma -name '*.so'` returns exactly one file, `_core.abi3.so`
(six before: five Cython + `_core`).

- `find hazma -name "*.pyx" -o -name "*.pxd"` -> **0** (13 before).
- `cargo fmt --manifest-path rust/Cargo.toml --check` -> clean.
- `cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings`
  -> clean.
- `cargo test --manifest-path rust/Cargo.toml --no-default-features` ->
  `258 passed` (unchanged from Task 6.3 — no Rust code changed, only
  doc comments).
- `pytest -q` -> **`2231 passed, 15 skipped, 12 subtests passed`**
  (2389/15/12 before). Every one of the 158 is accounted for, by
  `--collect-only` per module: `test_core_constants` 21 -> 0,
  `test_core_boost` 80 -> 50, `test_core_photon_muon` 53 -> 29,
  `test_core_photon_pion` 73 -> 30, `test_core_positron_muon` 47 -> 25,
  `test_core_positron_pion` 49 -> 27, and `test_no_cython_remains`
  0 -> 4. That is 162 retired and 4 added.
- `pytest test/parity -q` -> `658 passed, 1 skipped` (unchanged).
- Warnings went `8 -> 0`: every one was an `IntegrationWarning` raised
  inside a retired Cython-twin comparison, plus one pre-existing
  `SyntaxWarning` that xdist reports per worker.

**Test-validity checks.** All four tests in the new module were run
against `origin/master`'s versions of the files they read and all four
fail there: `setup.py` uses `cythonize`/`Extension`/`numpy`,
`[build-system] requires` is the five-entry list, and the `MANIFEST.in`
sweep carries `*.pyx *.pxd *.c`. The source sweep was additionally
proved by dropping a `hazma/_utils/_probe.pyx` into the tree, which
turned it red, and removing it again. `test_core_constants.py` was run
green (`21 passed`) against `origin/master` immediately before deletion.

**Restore-roster check.** All 29 `RESTORED_SOURCES` entries resolve:
each `git show <rev>:<path>` returns non-empty with exit 0.

## Numerical impact

**No public value changes**, measured rather than argued. A second
worktree was built at `origin/master` (`1b022d4`) with an identical
pinned environment, and 88 arrays were captured from both trees and
compared: every public `hazma.spectra.dnde_*` entry point over
`np.logspace(-3, 3, 401)` at two parent energies (250 and 1000 MeV), plus
`HiggsPortal` and `KineticMixing` `total_spectrum` and
`total_positron_spectrum` at `e_cm = 510` MeV. **0 of 88 moved**, with
the 72 numeric arrays bit-identical and the 16 that raise returning the
same exception type on both sides.

That is the expected result and the reason the measurement is worth
stating: nothing under `hazma/` imported any deleted file, so this task
could only have moved a number by breaking the build. Nothing is
appended to `../numerical-impact.md`.

Two harness notes, because the first reading was wrong. Encoding a raised
exception as `hash(type(exc).__name__)` made 16 arrays differ across runs
— `str` hashing is randomised per process — and the fix was to store the
name's bytes. And a `python -c` run from the repository root imports the
**cwd's** `hazma/`, not the installed one, so the trunk comparison has to
be driven from outside both trees.

## Open Questions

- **`test/test_core_constants.py`'s bit-equality gate over ~220
  constants does not survive, and nothing fully replaces it.** The
  corpus pins every constant that reaches one of the 41 entry points,
  and `constants.rs`'s `cargo` tests pin the structural relations and
  rule-4 divergences; a constant that no entry point reads is pinned by
  neither. That is recorded in `rust/src/constants.rs`'s module header,
  and the natural time to close it is whenever the two tables are
  consolidated — the separate declared change rule 4 still forbids.
- **The line term's missing `1/r` is still open and still post-6.4**
  ([the missing electron velocity](../../../../docs/followups/todo/mediator-positron-line-misses-the-electron-velocity.md)).
  Unchanged by this task.
- **Re-capturing `test/parity/oracles` is now possible but expensive.**
  Nothing needs it: the committed arrays cover A3 and A4 and change only
  if a patch does. Said explicitly here because the phase README asked
  this task to say it when it deleted the last `.pyx`.

## Plan Impact

**Impact Level:** Phase file patched (two files).

- `../../phases/phase-06-mediator-spectra.md` — Task 6.4's first exit
  criterion named "the `spectra/_neutrino/_neutrino` struct module" among
  the files to delete. Phase 04 Task 4.6 had already deleted it; the
  bullet is corrected to say so rather than listing a file that cannot be
  found. Phase frontmatter flipped to `status: Complete`.
- `../../phases/phase-07-cutover.md` — Task 7.1's second exit criterion
  read "setuptools-rust, `setup.py`, and the cython/scipy/numpy build
  requirements deleted". This task deleted the build requirements, so the
  bullet now names only what 7.1 still owes. Its Prerequisites block,
  which records the pre-cutover packaging facts, is corrected to the
  two-entry `requires` list.

No ADR: nothing canonical about architecture, interfaces, units or
ordering changed. The `[build-system]` edit shifts work *earlier* within
an ordering the plan already fixes, and is recorded in both phase files.

## Stale-state sweep

Every command run against this branch after the last edit.

| Check | Command | Result |
| --- | --- | --- |
| Cython sources | `find hazma -name "*.pyx" -o -name "*.pxd" \| wc -l` | `0` (13 on `origin/master`) |
| Built extensions | `find hazma -name "*.so"` | one file, `_core.abi3.so` (6 before) |
| Deleted tracked files | `git diff origin/master --diff-filter=D --name-only` | `14` matching `.pyx`/`.pxd`/`.bak` |
| Cython consumers | `rg "cimport\|__pyx_capi__\|\.pxd" hazma/` | no occurrences |
| Live imports of a deleted module | `rg` over `hazma/` and `test/` for the six deleted module paths | no occurrences |
| Restore roster | `len(defects.RESTORED_SOURCES)` | `29`, and all 29 resolve under `git show <rev>:<path>` |
| Capsule oracle sources | count of `kind == "capsule"` in `entry_points.SOURCES` | `0` |
| Moved follow-up, inbound links | `rg 'todo/oracle-restore-revisions'` | only historical `Files Changed` path listings; all three markdown links repointed to `done/` |
| Dangling `:class:`/`:meth:` refs | AST scan of the six changed test modules for references to names they no longer define | `0` — three `TestAgainstTheCythonTwin` refs were found and downgraded to literals |
| Forbidden tokens | `git diff origin/master -- '*.py' '*.rs'`, `+` lines, grepped for `TODO`/`FIXME`/`breakpoint()`/`pdb`/`print(` | no occurrences |
| Session language | same diff, grepped for "as discussed", "per the plan", "as requested", "for now", "in this task" | no occurrences |
| Deleted-module references, repo-wide | `rg` for an import of any deleted module across the whole tree, not just `hazma/` and `test/` | 5 hits, all provenance strings in captured manifests and the `PORTED_ENTRY_POINTS` / oracle rosters; the three *executable* ones in `.github/` and `docs/agents/` were found by CI and fixed |
| Present-tense survivor claims | `rg 'capi survivor\|capi-survivor\|four survivors'` outside `projects/` | 9 hits, all past-tense or explicitly historical after one fix in `test/test_core_mediator_positron.py` |
| Orphaned module constants | AST scan of the five rewritten test modules for upper-case module-level assignments with exactly one occurrence in the file | **21 found and removed**, each confirmed live on `origin/master` first so the cleanup is this task's own residue and not a drive-by |
| Rust gates | `cargo fmt --check`; `cargo clippy --all-targets -- -D warnings`; `cargo test --no-default-features` | clean; clean; `258 passed` |
| Suite | `pytest -q` | `2231 passed, 15 skipped, 12 subtests passed` |
| Corpus | `pytest test/parity -q` | `658 passed, 1 skipped` |
| Preflight | `preflight.sh --paths "<13 .py/.pyi>" --md "<13 .md>"` | **RESULT: PASS**, all eleven rows |

**Numerical-impact statement.** No public value changes, verified by
building `origin/master` in a second worktree with an identical pinned
environment and comparing 88 captured arrays across every public
`hazma.spectra.dnde_*` entry point and both mediator models' total photon
and positron spectra: **0 of 88 moved**. Nothing appended to
`../numerical-impact.md`.

**Two sweep findings worth naming.** `preflight.sh` passes `--paths`
verbatim to `black` and `ruff`, so a path list containing `.rs`,
`MANIFEST.in` or `pyproject.toml` fails both gates on parse errors that
read like real lint debt (12,319 "ruff errors" in one run) — pass Python
paths only and let the `cargo` rows cover `rust/`. And `ruff` caught a
portability bug the suite could not: `test_no_cython_remains.py` first
parsed `pyproject.toml` with `tomllib`, which is 3.11+, while
`requires-python` is `>=3.10` and CI's test matrix includes 3.10.
Replaced with a regex over the `[build-system]` array.

## Handoff to Next Task

**Phase 06 is closed and the Cython-to-Rust port is complete.** Phase 07
(packaging cutover and project close) is next, at Task 7.1. The direct
brief is [`README.md`](README.md)'s `## Handoff`; read
[`../../learnings/phase-06-mediator-spectra.md`](../../learnings/phase-06-mediator-spectra.md)
in place of this phase's four task notes.

**Now safe to assume:**

- **No Cython exists anywhere**, and four properties are asserted rather
  than remembered, in `test/test_no_cython_remains.py`: no `.pyx`/`.pxd`
  in the repository, no Cython in `setup.py`'s declarations,
  `[build-system] requires` is exactly `{setuptools, setuptools-rust}`,
  and no transpiler-output glob in `MANIFEST.in`.
- **`setup.py` is one `RustExtension`.** Phase 07 Task 7.1 deletes the
  file; its exit criteria were patched here, because 6.4 already removed
  the cython/numpy/scipy build requirements 7.1 used to owe.
- **The oracle roster is complete at 29 entries** and every revision
  resolves, so a re-capture stays possible if a patch ever changes.
  Nothing needs one.
- **No test in the tree executes against Cython.** Four mechanisms that
  used to are now frozen rosters or reference implementations, each with
  its provenance recorded in place.

**Still risky for Task 7.1:**

- **`MANIFEST.in` is on 7.1's deletion list and
  `test_no_cython_remains.py` reads it.**
  `test_the_sdist_manifest_sweeps_up_no_transpiler_output` will fail on
  a missing file. Decide there whether maturin's own include config
  carries the claim, and rewrite the test rather than dropping it.
- **`pyproject.toml`'s `[build-system]` is read by that module too**, so
  the maturin switch must update
  `test_the_build_requirements_name_no_cython_toolchain` in the same
  commit — it asserts an exact two-element set.
- **`AGENTS.md` and `docs/agents/` still state Cython facts** that are
  now false: the layout tree, "Editing a `.pyx` requires a rebuild",
  layering §1, and the commands block. Task 6.4 left them for Task 7.3,
  which owns that sweep — so 7.3 corrects live errors rather than tidying.
- **`preflight.sh` gates `black`/`ruff` on whatever `--paths` receives.**
  Pass Python paths only; the `cargo` rows already cover `rust/`.
