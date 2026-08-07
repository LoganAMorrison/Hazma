# Phase 00 Learnings: Dead-code purge

Synthesized at phase close (2026-08-06) from the five task notes in
`../task-notes/phase-00/`. Durable memory for Phases 01–07 and for
anyone who later deletes code in this repo — not a status log.

## 1. Implementation Reality Check

The phase delivered its headline target exactly: **32 extensions → 20**,
zero C++, no `.pyx` outside the live surface, and roughly −33,000 lines
across `hazma/` and `test/`. The 20 survivors are 8 `spectra/_photon` +
2 `spectra/_positron` + 3 `spectra/_neutrino` + 6 mediator +
`_utils/boost`, and `setup.py`'s declared list, the `.pyx` on disk and
the `.so` from a clean build are now a verified set equality, not three
numbers that happen to agree.

Where execution diverged from the plan:

- **Task ordering was not the numbered ordering.** The plan implied
  0.1 → 0.2 → 0.3 → 0.4 with 0.5 as a decision gate. What actually ran
  was 0.1 → 0.3 → 0.5 → 0.2 → 0.4, because ADR-0003's sign-off gated
  0.2's deletion while 0.3 had no such dependency. Phased plans should
  expect the sign-off-gated task to float.
- **Three of five tasks patched canonical criteria.** Task 0.1 corrected
  its own (four include sites → five built + two unbuilt), Task 0.2
  rewrote **Task 0.4's**, and Task 0.4 amended its own twice. This is the
  phase's most transferable process result: *the criteria written before
  a deletion are routinely wrong about what the deletion strands*, and
  patching them in the same PR is cheaper and more honest than absorbing
  the difference into the diff.
- **Task 0.4 was smaller than written and wider than written at once.**
  The extension-list reconciliation it was assigned turned out to be a
  no-op (Tasks 0.2/0.3 had already dropped every group, because a
  deletion task *cannot* defer its own `setup.py` edit — the build fails
  immediately on an `Extension` whose source is gone). But the sdist it
  was assigned to run — the first in the project's history — surfaced two
  defects the criteria never anticipated.

ADRs: **ADR-0003** (remove `hazma.gamma_ray`) was accepted 2026-08-04
with an Addendum the same day, and is **fully discharged** — non-deletion
steps in Task 0.5, the deletion in Task 0.2. ADR-0001 and ADR-0002 were
accepted before the phase and were not touched. Repo-wide ADR-0001
(FSR generator) flipped Proposed → Accepted in Task 0.5 as a side effect
of PR #41 merging.

## 2. Critical Context for Future Work

- **The live surface is 20 extensions; re-derive, do not quote.** Every
  count in this project's docs is a snapshot. The recipe is: clean
  (`find hazma -name '*.c' -o -name '*.cpp' -o -name '*.so' | xargs -r rm -f`),
  `pip install -e .`, then compare the *sets* — declared in `setup.py`,
  `.pyx` on disk, `.so` built. Comparing counts hides a stale `.so` that
  makes a wrong list look right.
- **`setup.py`, not `_build.py`, is the build entry point.** `_build.py`
  was deleted in `7a817f9` (2026-08-02), before this project began, but
  thirteen durable docs still named it until Task 0.4 swept them —
  twelve by rename, and `docs/versioning.md` by deleting a blockquote
  whose whole subject was a stale `VERSION` inside the vanished file. If
  you see `_build.py` referenced outside this project's dated records
  again, it is rot.
- **`hazma.utils` is the single home for `cross_section_prefactor` and
  `minkowski_dot`**; the legacy constants table is
  `hazma/_utils/legacy_parameters.pxd` and is now its only copy, kept
  deliberately divergent from `_utils/constants.pxd` per `../rules.md`
  rule 4.
- **`hazma/deprecated/` no longer exists as a package.** Removing
  `rambo.py` removed the directory with it, since there was no
  `__init__.py`. `AGENTS.md` and `docs/versioning.md` §6 were rescoped
  from "it stays importable" (a claim about the tree) to a policy about
  whatever is parked there next.
- **Four spectra extensions and the `_utils` headers are capi
  survivors.** They outlive their Phase 04 swap because the mediator
  spectrum `.pyx` cimport their `__pyx_capi__` symbols; Phase 06 Task 6.4
  is the only place they die. Task 0.3 also kept `boost_jac` /
  `boost_eng` in `_utils/boost.pyx` on that basis — they have zero
  cimporters but *are* declared in `boost.pxd`, i.e. published C-level
  API. That is a 6.4 call, not a 0.x one.
- **The public compiled surface did not move in this phase.** Two
  declared drifts exist, both in the Cython → pure-Python helper swap,
  both recorded in `../task-notes/README.md`. Separately, the
  `two_body_momentum` repair landed out-of-band before Phase 01, so
  **the corpus must pin the post-fix values.**

## 3. Quirk Log & Edge Cases

The five that cost real time, in the order a future deleter will hit
them:

- **Stale generated `.c`/`.cpp` silently poisons the build.** A worktree
  can inherit them from another environment; their mtimes suppress
  re-cythonization and the build dies deep inside generated code with a
  misleading error (Task 0.1 saw `no member named 'subarray' in
  '_PyArray_Descr'` from a `.cpp` generated against an older NumPy).
  Clean before every rebuild. This is the single most-repeated lesson of
  the phase.
- **`git rm -r` on a package leaves it importable.** An untracked
  `__pycache__` keeps the directory alive, and an empty directory on
  `sys.path` is a *namespace package* — `import
  hazma.field_theory_helper_functions` still succeeded right after the
  `git rm` (Task 0.3). `rm -rf` too, then re-run the negative import
  check. Any verify-after-delete that only reads the git index misses
  this.
- **A `git stash` round-trip un-stages a deletion** (Task 0.2). Stashing
  to baseline the linters against the trunk and popping restores removals
  as *unstaged*, so `git ls-files` still lists deleted paths and
  `scripts/agents/check_doc_citations.py` tracebacks with
  `FileNotFoundError`. `git add -A` after every pop. This bites precisely
  when following the documented recipe for proving preflight redness is
  pre-existing.
- **A clean wheel is not evidence of a clean sdist** (Task 0.4). The two
  are built by different machinery: `[tool.setuptools.packages.find]`
  scoped the wheel to `hazma*` back in `7a817f9`, but `MANIFEST.in`'s
  `global-include *.md` kept sweeping `.claude/`, `.codex/` and
  `projects/` — 101 files of agent scaffolding — into the tarball.
  Nobody noticed for four months because nobody ran `build --sdist`.
- **An unanchored path probe over a tarball listing is useless.** Task
  0.4's first pass matched `_positron`, `rambo` and `gamma_ray` as bare
  substrings and produced 70+ hits, every one a live path
  (`hazma/spectra/_positron/`, `hazma/phase_space/_rambo.py`,
  `hazma/theory/_theory_gamma_ray_limits.py`). Anchor with `^` and `$`.
  A probe that cries wolf gets ignored, which is worse than no probe.

Two more worth carrying:

- **Deleting a `.pyx` forces the matching `setup.py` edit in the same
  task** — a deletion task cannot hand its extension groups to a later
  cleanup task. `test/conftest.py` had the same property for
  `test/decay/`: its `iterdir()` was unconditional and raised at
  *collection*, taking the whole suite down.
- **`docs/source/` has orphan pages no toctree reaches** (nine documents
  from `index.rst`, plus four nested). Sphinx still builds an orphan, so
  "unlinked" is not "not shipped" — but deleting one breaks no
  navigation. Precedent set in Task 0.5 and reused in 0.2: an orphan page
  whose entire subject is being removed is deleted, not converted into a
  stub Sphinx cannot redirect from.

## 4. Test Infrastructure State

- **`test/conftest.py` now skips no test module at all.** Its
  `collect_ignore` holds only the repo's `setup.py`. Both entries that
  used to hide part of the suite went with the code they covered
  (`test/decay/` in 0.3, `test/test_gamma_ray.py` in 0.2). **This makes
  Phase 01 Task 1.3's suite merge simpler than planned.**
- **Two disjoint suites remain, for one unrelated reason.**
  `pytest -q test` → `244 passed, 20 skipped`; bare `pytest -q` →
  `57 passed, 10 skipped`, because `setup.cfg`'s `testpaths = hazma`
  keeps a bare run inside the package. Cite the command, never "the full
  suite". **Zero compiled-layer pinned tests run anywhere** — that gap is
  exactly what Phase 01 exists to close.
- **The phase's own regression harness is the before/after grid**, and it
  should be reused verbatim in Phases 04–06. Dump every compiled-backed
  public entry point over `np.logspace(-2, 3, 200)` MeV at three parent
  energies / three mediator masses, `git stash`, clean, rebuild, dump
  again, diff. Task 0.3 ran 171 arrays, Task 0.2 159, Task 0.4 213; all
  bit-for-bit identical. It caught nothing in this phase, which is the
  point — it is what licenses "no public value changes" as a *measured*
  claim rather than an assertion.
- **`test/test_utils.py`** (16 pinned cases) was added by Task 0.3 when
  `cross_section_prefactor` and `minkowski_dot` moved to pure Python. It
  is the pattern for pinning a relocated numeric helper.
- **New in this phase and worth keeping: install the sdist and run it.**
  `uv build --sdist`, then `uv pip install --no-binary hazma
  dist/*.tar.gz` into a *fresh* venv and import-smoke from outside the
  repo. A tarball can pass every path probe and still fail to build.
  Phase 07 Task 7.1 should re-run this against maturin.
- **`preflight.sh` cannot return zero for a file under `hazma/`** on the
  trunk (gates 2 and 3), while CI enforces only `black` plus
  `ruff --isolated --select E9,F63,F7,F82`. Every task in this phase
  inherited the two red rows and had to prove they were pre-existing.
  Tracked in
  [`../../../docs/followups/todo/preflight-isort-ruff-red-on-trunk.md`](../../../docs/followups/todo/preflight-isort-ruff-red-on-trunk.md).

## 5. Follow-on seeds

- **The sdist payload is unexamined.** Beyond the scaffolding Task 0.4
  pruned, the tarball still ships 20 cythonized `*.c` (never used — the
  build always re-cythonizes, and their presence makes the sdist
  build-output rather than source), `docs/` (46 files), `test/` (12) and
  `notebooks/` (2), and `pyproject.toml`'s package-data lists `*.pyd`
  where `*.pxd` was surely meant. Each is a judgment call rather than a
  defect, which is why a dead-code task did not settle them. **Trigger:
  before Phase 07 Task 7.1** — maturin does not read `MANIFEST.in`, so
  after the cutover the same decisions cost more to express. Filed as
  [`../../../docs/followups/todo/sdist-ships-generated-c-and-docs.md`](../../../docs/followups/todo/sdist-ships-generated-c-and-docs.md).
- **`requirements.txt` and `Dockerfile` contradict `pyproject.toml`**
  (`numpy>=1.16.2` against `numpy>=2.0`; a dev toolchain of flake8 and
  jupyter that the PEP 735 groups replaced). Deliberately untouched by
  Task 0.4 because `phases/phase-07-cutover.md` Task 7.3 already owns
  them. Noted here so the next reader does not re-derive that they are
  stale.
- **The constants tables are still deliberately divergent.**
  `_utils/legacy_parameters.pxd` versus `_utils/constants.pxd`, preserved
  bit-for-bit per `../rules.md` rule 4. Consolidation is a *declared
  numerical change* for after the port, and is already named in
  `PLAN.md` §Scope as out of scope. It remains the most obvious
  post-project cleanup.
- **The remaining `sqrt(kallen_lambda(...))` call sites** carved out of
  the `cross_section_prefactor` repair —
  [`../../../docs/followups/todo/kallen-under-sqrt-remaining-call-sites.md`](../../../docs/followups/todo/kallen-under-sqrt-remaining-call-sites.md).
  Phase 01 pins whatever these currently return, so fixing one afterwards
  means regenerating corpus entries. Worth resolving early or accepting
  consciously.
- **`hazma/experimental/axial_vector_mediator/__init__.py` is broken on
  the trunk** (`from hazma.theory import Theory`, but `hazma.theory`
  exports `TheoryAnn` / `TheoryDec`). Pre-existing, out of every lint
  gate, and excluded from import smoke throughout this phase. No
  follow-up filed — `experimental/` is explicitly not a public surface —
  but it will keep tripping import sweeps until someone deletes or fixes
  it.
