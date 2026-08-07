# Task 0.4: Prune build and packaging config

**Date:** 2026-08-06
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-00-dead-code-purge.md` (Task 0.4
and phase Exit Criteria); `../../PLAN.md` §Scope, §Numerical impact
**Related ADRs:** none directly (ADR-0003 was fully discharged by Tasks
0.5 and 0.2)
**Depends On:** Tasks 0.2, 0.3 — both Complete

## Objective

Close Phase 00 by reconciling the build and packaging config against the
surviving Cython surface: prove `setup.py`'s extension list is exactly
the survivors, delete the `make_extension` C++ branch that no caller can
reach any more, and run the sdist that no task in this phase had yet run.

## Exit Criteria

Copied from the phase file's Task 0.4 block before implementation; the
last two bullets were added *by* this task and are flagged as such.

- [x] `setup.py`'s extension list matches the survivors exactly, counted
  against the `.so` from a fresh `pip install -e .`.
- [x] `make_extension`'s now-unreachable `cpp=True` parameter and
  `language="c++"` branch removed — no caller has passed it since Task
  0.2.
- [x] `pyproject.toml` package-data globs and `MANIFEST.in` ship no
  deleted directory; **the sdist builds and contains no deleted path.**
- [x] *(scope addition, criterion amended in this PR)* the sdist ships no
  agent scaffolding.
- [x] *(scope addition, criterion amended in this PR)* no durable doc
  names `_build.py` as the build entry point.
- [x] Phase closure: `../../learnings/phase-00-dead-code-purge.md`
  written, phase file frontmatter `status: Complete`, `PLAN.md` Phases
  table row updated.
- [ ] CI green on the full matrix — cannot be checked pre-PR; verified
  locally on CPython 3.12 (build, sdist, wheel, both suites) and left to
  the PR run.

## Inputs Reviewed

- `../../PLAN.md` (all sections), `../README.md`, `README.md` (phase-00
  working memory), `../../phases/phase-00-dead-code-purge.md`,
  `../../rules.md`.
- `../../phases/phase-07-cutover.md` Tasks 7.1/7.3 — checked for overlap
  before widening scope. It already owns `requirements.txt` and
  `Dockerfile`, so this task left both alone.
- `docs/agents/environment.md`, `docs/agents/preflight.md`,
  `docs/agents/doc-consistency.md`, `docs/agents/lessons.md`.
- `setup.py`, `pyproject.toml`, `MANIFEST.in`, `setup.cfg`,
  `requirements.txt`, `Dockerfile`, `.github/workflows/ci.yml`.
- `git show 7a817f9` — the commit that deleted `_build.py` and added
  `setup.py`.

## Findings

- **The extension list was already correct; the reconciliation is the
  deliverable, not a fix.** Declared-in-`setup.py`, `.pyx`-on-disk and
  `.so`-built are the same 20-element set, with empty symmetric
  differences in both directions. Tasks 0.2 and 0.3 each dropped their
  own groups as they deleted sources — as the phase file predicted, a
  deletion task cannot defer them — so Task 0.4 found nothing left to
  remove. Recorded here because "no change" is a result that has to be
  *measured*: the criterion is a set equality, not an eyeball count of a
  literal list.
- **The `.so` count is not by itself evidence of reconciliation.** A
  stale `.so` from a prior build survives `pip install -e .` and inflates
  the count to match a list that is wrong. Clean
  (`find hazma -name '*.c' -o -name '*.cpp' -o -name '*.so' | xargs rm -f`)
  first, then compare the three sets, not the three counts.
- **`MANIFEST.in`'s `global-include` is repo-wide, and nothing scoped
  it.** The pre-task sdist carried 498 files, of which 101 were
  `.claude/` (18), `.codex/` (18) and `projects/` (65) — the agent
  skills, and this project's own plans, ADRs and task notes, inside a
  publishable tarball. The wheel has been clean since `7a817f9`
  restricted `[tool.setuptools.packages.find]` to `hazma*`; that fix
  never reached the sdist, because the two are built by different
  machinery. **A clean wheel is not evidence of a clean sdist.**
- **The sdist had never been built in this project.** Task 0.2's handoff
  says so explicitly (`build` was not in its venv). It is the only check
  in Phase 00 that could have caught the above, and it was the last one
  to run.
- **`_build.py` has not existed since 2026-08-02** — commit `7a817f9`
  deleted it and added `setup.py` in the same change, to fix wheels
  being mistagged `py3-none-any`. That predates this project. Thirteen
  durable docs still named it as the build entry point, including
  `AGENTS.md` (the tie-breaker) and `docs/agents/environment.md`'s
  **Build and imports** section — i.e. the exact sentence every agent
  reads before deciding whether a rebuild is required. `docs/versioning.md`
  additionally warned readers to leave alone a stale `VERSION` constant
  (`VERSION = "2.0.0-rc1"`, retrievable at `git show 7a817f9^:_build.py`)
  in a file that no longer exists.
- **The sdist regenerates the `.c` it ships.** Deleting every `.c` from
  the tree and rebuilding the sdist still produced 20 of them: the
  isolated sdist build runs `setup.py`, which calls `cythonize()`
  unconditionally, writing `.c` next to each `.pyx` before `global-include
  *.c` sweeps them in. So the tarball's contents are stable, but they are
  build-output rather than source. Deferred, with the `docs/`, `test/`
  and `*.pyd` questions, to
  [`../../../../docs/followups/todo/sdist-ships-generated-c-and-docs.md`](../../../../docs/followups/todo/sdist-ships-generated-c-and-docs.md).
- **The preflight `isort`/`ruff` rows behave as Task 0.5 predicted**, and
  this task narrowed the `ruff` debt: `setup.py` was one of the files
  carrying `UP006` (`List[str]` where `list[str]` works on Python ≥3.10).
  Modernising the annotation on the signature line this task was already
  rewriting dropped the `from typing import List` import with it.

## Decisions and Implementation Notes

- **Removed the C++ branch rather than keeping it "in case".** No caller
  passes `cpp`, `git grep -l 'std::' -- hazma/` is empty, and the phase
  Exit Criteria require no `language="c++"` extension to remain. The
  docstring records *why* it went (the C++ extensions left with
  `_gamma_ray/` and `_phase_space/` in Task 0.2) so the next reader does
  not reintroduce it. The docstring deliberately does not spell the
  literal `language="c++"` — a future `grep` for that string should
  return nothing, and a prose mention would be a false positive.
- **Modernised `List[str]` → `list[str]` in the same signature.** Two
  tokens on a line the task was rewriting anyway, on a repo whose floor
  is Python 3.10; it removes a configured-`ruff` finding rather than
  leaving one behind in a file this task owns. Not a behavior change —
  `setup.py` runs only at build time and the build was re-run after.
- **Pruned the agent scaffolding out of the sdist (scope addition).**
  `.claude/`, `.codex/` and `projects/` are not distribution content
  under any reading, the fix is three `prune` lines, and it was found by
  the sdist run this task was assigned to perform. Widened here and the
  criterion amended in the same PR, on **Task 0.2's precedent** — the
  alternative was to file a follow-up against a `MANIFEST.in` that Phase
  07 Task 7.1 deletes, which would have quietly dropped it.
- **Did not prune `docs/`, `test/`, `notebooks/` or the cythonized
  `*.c`.** Each is a judgment call with a real argument on both sides
  (shipping tests in an sdist is a convention, not a defect), and a
  dead-code-purge task is the wrong place to settle packaging policy.
  Follow-up filed with the measured inventories and a stated deadline —
  before Task 7.1, because maturin will not read `MANIFEST.in`.
- **Left `requirements.txt` and `Dockerfile` alone** even though both
  contradict `pyproject.toml` today (`numpy>=1.16.2` vs `numpy>=2.0`).
  `phase-07-cutover.md` Task 7.3 already names both. Touching them here
  would have been scope theft, not scope addition.
- **Swept all thirteen docs that named `_build.py` (scope addition)** —
  twelve by rename, one by deletion; see Files Changed. Phase 07 Task
  7.3 nominally covers `AGENTS.md`,
  `docs/agents/` and the skills — but it covers them for *Rust*, and it
  is six phases out. A wrong filename in the tie-breaker doc and in the
  rebuild-awareness rule would have misled every agent in Phases 01–06.
  A partial fix would have been worse than none, so the sweep is
  exhaustive over the *durable-doc surface* — `AGENTS.md`, `docs/`,
  `.claude/`, `.codex/`, `.github/` and the build config all come back
  empty. The name still appears 21 times under `projects/`, entirely in
  this task's own records describing the rename; those are dated history,
  not live references, and are deliberately left as written.
- **Deleted `docs/versioning.md`'s stale-`VERSION` blockquote instead of
  rewriting it.** A mechanical rename would have turned a true warning
  about `_build.py` into a false claim about `setup.py`, which has no
  `VERSION` at all. Same §11 stale-sibling pass caught the illustrative
  snippet three lines above it still reading `"2.0.2"` against the live
  `"2.1.0"`.

## Files Changed

### Build and packaging

- `setup.py` — `make_extension` loses the `cpp` parameter, the
  `language="c++"` / `-std=c++11` branch, and the now-single-use
  `include_dirs` local; annotations modernised to builtin generics, so
  `from typing import List` goes too. Docstring records why no C++ branch
  remains. Extension list untouched (it was already exact).
- `MANIFEST.in` — `prune .claude`, `prune .codex`, `prune projects`, with
  a comment explaining that the `global-include` above them is a
  repo-wide sweep and that the wheel is clean by a different mechanism.
- `pyproject.toml` — **unchanged.** Audited, nothing dangles; the one
  suspect entry (`*.pyd`) went to the follow-up.

### Durable docs

Thirteen files named `_build.py`
(`grep -rl '_build\.py' --include='*.md' .` outside `projects/`).
Twelve took the rename:

- `AGENTS.md`, `docs/PR_GUIDELINES.md`, `docs/agents/environment.md`,
  `docs/agents/preflight.md`, `docs/agents/review-lenses.md`,
  `.claude/skills/{commit-and-pr,execute-single-task,review-cycle,
  review-pr,review-respond,task-pipeline}/SKILL.md`,
  `.codex/skills/execute-single-task/SKILL.md` — `_build.py` →
  `setup.py` (one token each). `docs/PR_GUIDELINES.md` also had its
  table row re-padded, since the replacement is a character shorter.

The thirteenth is the exception the rename would have broken:

- `docs/versioning.md` — its only occurrence was the blockquote warning
  readers off a stale `VERSION`, so renaming would have produced a false
  claim about `setup.py`. Blockquote deleted; the `hazma/__init__.py`
  snippet three lines above it re-derived from the live file
  (`2.0.2` → `2.1.0`).
- `docs/agents/environment.md` additionally gained two new **Build and
  imports** entries — the wheel-is-not-sdist trap and the
  path-probe-is-not-a-build rule — per that file's own standing
  instruction to record a new trap in the same commit. (`lessons.md`
  was deliberately not touched: its format requires a merged PR
  citation, and it says to put an uncitable class in a `docs/agents/`
  checklist instead.)
- `docs/followups/todo/sdist-ships-generated-c-and-docs.md` (new) +
  an Open row in `docs/followups/README.md`.

### Project bookkeeping

- `projects/cython-to-rust/phases/phase-00-dead-code-purge.md` — Task 0.4
  exit criteria amended for the two scope additions; frontmatter
  `status: In Progress` → `Complete`.
- `projects/cython-to-rust/PLAN.md` — Phases-table row 00.
- `projects/cython-to-rust/learnings/phase-00-dead-code-purge.md` (new).
- `projects/cython-to-rust/task-notes/README.md` and
  `task-notes/phase-00/README.md` — status, findings, files-changed
  roll-up, handoff.
- This note.

## Verification

Environment: `uv venv --python 3.12` inside the task worktree; hazma
built with `uv pip install --python .venv/bin/python -e .` after
`find hazma -name '*.c' -o -name '*.cpp' -o -name '*.so' | xargs -r rm -f`.
`python -c "import hazma; print(hazma.__file__)"` resolves inside the
worktree, not to an installed copy.

**Extension reconciliation** — the three sets compared programmatically
(`setup.py` imported with a stubbed `setup()`; `.pyx` globbed; `.so`
globbed), not counted by eye:

```text
declared in setup.py : 20
.pyx sources on disk : 20
.so built in place   : 20
declared - pyx : none      pyx - declared : none
declared - so  : none      so - declared  : none
RECONCILED
git grep -l 'std::' -- hazma/ : empty
```

That is the 20 the phase Exit Criteria name: 8 `spectra/_photon` + 2
`spectra/_positron` + 3 `spectra/_neutrino` + 6 mediator +
`_utils/boost`.

**sdist** (`uv build --sdist`), before and after the `MANIFEST.in` prune:

| | before | after |
| --- | --- | --- |
| files in tarball | 498 | 397 |
| `.claude/` + `.codex/` + `projects/` | 101 | 0 |
| deleted Phase-00 paths | 0 | 0 |

The deleted-path probe is an anchored alternation over every path Phase
00 removed (`^hazma/_decay/`, `^hazma/_gamma_ray/`, `^hazma/_phase_space/`,
`^hazma/_positron/`, `^hazma/_neutrino/`,
`^hazma/field_theory_helper_functions/`, `^hazma/deprecated/`,
`^hazma/gamma_ray\.py$`, the three `hazma/__*.py` shims,
`^hazma/spectra/_positron/_kaon\.pyx$`, `_rh_neutrino_fsr_four_body`,
`_rh_neutrino_spectra\.py`, `^test/test_gamma_ray\.py$`, `^test/decay/`,
`^docs/source/(gamma_ray|rambo)\.rst$`) — no hits either way. An earlier
unanchored probe produced 70+ false positives on live paths
(`hazma/spectra/_positron/`, `hazma/phase_space/_rambo.py`,
`hazma/theory/_theory_gamma_ray_limits.py`); anchoring is load-bearing
for this check, not cosmetic.

**The sdist installs and runs.** Installed into a *fresh* venv with
`uv pip install --no-binary hazma dist/hazma-2.1.0.tar.gz`, then
imported and evaluated from outside the repo:

```text
dnde_photon_muon([1,10,50], 200 MeV) : [0.01769967 0.00142339 0.00012374]
dnde_positron_muon([1,10,50], 200)   : [6.42823262e-05 5.41334191e-03 8.67582062e-03]
ScalarMediator.total_spectrum(E, 550): [0.03022713 0.00252758 0.00531033]
```

`hazma.{theory,limits,cmb,pbh,utils,spectra,phase_space,relic_density}`,
both mediator packages and `hazma.spectra._photon._muon` all import. This
is the check that actually proves nothing dangles — a tarball can pass a
path probe and still fail to build.

**wheel** (`uv build --wheel`): 311 files, `hazma/` + `.dist-info` only,
20 `.so`, `Tag: cp312-cp312-macosx_11_0_arm64` and
`Root-Is-Purelib: false` (the mistagging `7a817f9` fixed has not
regressed). No deleted path.

**Tests**, both against the rebuilt worktree:

```text
$ .venv/bin/python -m pytest -q test
244 passed, 20 skipped in 291.62s (0:04:51)

$ .venv/bin/python -m pytest -q
57 passed, 10 skipped in 0.45s
```

Identical to Task 0.2's counts, which is the expected result for a diff
that changes no library code. The two commands are two *different*
suites, not a subset and a superset: `setup.cfg`'s `testpaths = hazma`
means a bare `pytest` collects only the in-package `*_test.py` modules
(`hazma/form_factors/`, `hazma/phase_space/`) and never enters `test/`.
Neither exited 5, and both summary lines are quoted above rather than
inferred from the exit status. What they cover: the 244 are the
spectra/mediator/positron/rh-neutrino suites plus `test/test_utils.py`'s
16 pinned `hazma.utils` cases; the 57 are form-factor and phase-space
unit tests. Note that **neither suite exercises the packaging change** —
no test imports `setup.py` or inspects a built distribution, which is
why the sdist install-and-run check above is the real gate here.

**Numerical impact: none.** See below.

## Numerical impact

**No public value changes.** 213 arrays — every compiled-backed public
entry point over `np.logspace(-2, 3, 200)` MeV (12 `dnde_photon_*`, 12
`dnde_positron_*`, 12 `dnde_neutrino_*`, each at parent energies 150 /
500 / 1500 MeV, plus both models' `spectra()`, `positron_spectra()`,
`annihilation_cross_sections()` and `thermal_cross_section()` at mediator
masses 200 / 550 / 1200 MeV) — **bit-for-bit identical** across the
change, max relative deviation `0.000e+00`, measured by dumping the grid,
`git stash`ing the diff, cleaning and rebuilding, dumping again, and
comparing:

```text
arrays compared       : 213
bit-for-bit identical : 213
max relative deviation: 0.000e+00
```

Expected, and the mechanism is worth stating rather than asserting: the
only executable change is the deletion of an `if cpp:` branch no call
site reaches, so every `Extension` object `setup.py` constructs is
identical to before and the compiled artifacts are the same. Everything
else in the diff is `MANIFEST.in` and prose.

Per the recipe in `../README.md`, `git add -A` was run after the
`git stash pop` so the staged tree matches the working tree.

## Open Questions

- **The sdist payload** — cythonized `*.c`, `docs/`, `test/`,
  `notebooks/`, and the `*.pyd`-for-`*.pxd` package-data entry. Filed as
  [`../../../../docs/followups/todo/sdist-ships-generated-c-and-docs.md`](../../../../docs/followups/todo/sdist-ships-generated-c-and-docs.md),
  deliberately deferred, with a stated deadline of Phase 07 Task 7.1
  (after which `MANIFEST.in` no longer exists to fix).
- **`preflight.sh` `isort`/`ruff` on the trunk** — still red for reasons
  unrelated to any single task, per
  [`../../../../docs/followups/todo/preflight-isort-ruff-red-on-trunk.md`](../../../../docs/followups/todo/preflight-isort-ruff-red-on-trunk.md).
  This task's delta is recorded under Verification in `../README.md`.
- **CI on the full matrix** is the one exit criterion that cannot be
  closed from a local worktree. Everything CI runs was run here on
  CPython 3.12; the 3.10–3.14 sweep and the Linux legs are the PR's job.

## Plan Impact

**Impact Level:** Update phase file (no ADR).

- `phases/phase-00-dead-code-purge.md` Task 0.4 exit criteria amended to
  name the two scope additions (the sdist prune, the `_build.py` doc
  sweep), rather than absorbing them silently into the diff — Task 0.2's
  precedent, now applied twice in this phase.
- Same file's frontmatter flipped to `status: Complete`; `PLAN.md`'s
  Phases table row 00 updated to match. No task ordering, interface,
  unit or normalization convention changed, so no ADR.
- Phase 07 Task 7.3's criteria were **not** patched. It still owes the
  Rust-facing rewrite of `AGENTS.md` and `docs/agents/`; this task only
  corrected a filename that was wrong independently of Rust.

## Stale-state sweep

Run against `claude/cython-to-rust/task-0.4-prune-build` at the end of
the task.

| Check | Command | Result |
| --- | --- | --- |
| No `_build.py` in the durable-doc surface | same grep over `AGENTS.md CLAUDE.md docs/ .claude/ .codex/ .github/ setup.py pyproject.toml MANIFEST.in setup.cfg` | no occurrences |
| `_build.py` survives only as history | `grep -rn '_build\.py' projects/ \| wc -l` | 21 — all in this task's own note, the phase README, the learnings and the amended criterion, each describing the rename. Dated records, not live references; a repo-wide grep therefore does **not** come back empty, and a sweep row claiming it does would be wrong. |
| No C++ wiring in build config | `grep -rn 'language\s*=\|c++\|std::' setup.py pyproject.toml MANIFEST.in` | no occurrences |
| No C++ in the package | `git grep -l 'std::' -- hazma/` | empty |
| Extension sets agree | reconcile script (above) | `RECONCILED`, 20/20/20 |
| sdist ships no deleted path | anchored probe over `tar tzf` listing | no occurrences |
| sdist ships no scaffolding | `cut -d/ -f1` on the listing | `.claude`/`.codex`/`projects` absent |
| `VERSION` claim re-derived | `grep -n VERSION hazma/__init__.py` vs `docs/versioning.md` | both `2.1.0` |
| Extension count claims | `find hazma -name '*.so' \| wc -l` | 20, matching every doc that states it |
| Task-0.4 row vs `**Status:**` vs phase frontmatter | read all three | all `Complete` |
| Forbidden tokens | `git diff origin/master -- '*.py'` for `breakpoint()`/`pdb`/`print()` | none added |
| **Numerical-impact statement** | 213-array before/after diff | **none** — bit-for-bit identical, `0.000e+00` |

## Handoff to Next Task

**Phase 00 is closed.** The next agent starts Phase 01 (golden parity
corpus) — read `../../PLAN.md`, then `../README.md`, then
`../../learnings/phase-00-dead-code-purge.md`, then
`../../phases/phase-01-parity-corpus.md`. Phase 00's per-task notes are
history; the learnings file is the distillation.

**Currently safe to assume:**

- The build is reconciled and reproducible: 20 `.pyx` → 20 declared
  `Extension`s → 20 `.so`, zero C++, no dangling package-data or
  `MANIFEST.in` entry. Re-derive with the clean-then-count recipe rather
  than quoting the number.
- The sdist and the wheel both build, and the sdist *installs and runs*
  in a fresh venv from outside the repo. That check now exists as a
  recipe in this note; Phase 07 Task 7.1 should re-run it against
  maturin.
- Every durable doc names `setup.py`, not `_build.py`, as the build
  entry point. The rebuild-awareness rules in the skills and in
  `docs/agents/review-lenses.md` are now actionable.
- The public compiled surface has not moved anywhere in Phase 00, with
  the two declared pure-Python helper drifts recorded in `../README.md`
  §"Numerical impact so far". **Phase 01's corpus pins the current
  values** — including the out-of-band `two_body_momentum` repair — and
  the Rust port must reproduce those.

**Currently risky / unknown:**

- CI has not run this diff. Local verification was CPython 3.12 on macOS
  arm64 only; the matrix is 3.10–3.14 on Linux plus macOS 3.14.
- The sdist payload question is open and time-boxed to before Task 7.1
  (see Open Questions). It is the only packaging item Phase 00
  deliberately left on the table besides `requirements.txt` /
  `Dockerfile`, which belong to Task 7.3.
