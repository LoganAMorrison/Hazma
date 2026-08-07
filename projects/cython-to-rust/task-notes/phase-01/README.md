# Working Memory: Phase 01 — Golden parity corpus

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Phase:** 01
**Status:** In Progress (Task 1.1 complete 2026-08-07)
**Plan References:** `../../phases/phase-01-parity-corpus.md`
**Related ADRs:** none
**Depends On:** Phase 00 complete

## Objective

Track live per-task status and phase-scoped findings for the parity
corpus.

## Tasks

| # | Task | Depends on | Status | Task Note |
| --- | ------ | ------------ | -------- | ----------- |
| 1.1 | Corpus specification + generator | — | **Complete (2026-08-07)** | [task-1.1-corpus-generator.md](task-1.1-corpus-generator.md) |
| 1.2 | Pytest runner + tolerance budgets | 1.1 | Not started — **next** | [task-1.2-parity-runner.md](task-1.2-parity-runner.md) |
| 1.3 | Wire both suites into one gate | 1.2 | Not started | [task-1.3-test-wiring.md](task-1.3-test-wiring.md) |
| 1.4 | Retire/regenerate legacy `.npy` suites | 1.2 | Not started | [task-1.4-legacy-npy.md](task-1.4-legacy-npy.md) |

## Exit Criteria

- All rows Complete; phase file frontmatter `status: Complete`.
- Phase learnings at `../../learnings/phase-01-parity-corpus.md`.

## Inputs Reviewed

- `../../phases/phase-01-parity-corpus.md`; `../README.md`;
  `../../references/numerics-replacements.md` (tolerance table);
  `../../references/cython-inventory.md` (entry-point table).

## Findings

- **The corpus exists and is self-checking.** 41 cases / 623 blocks /
  1,580 arrays / 179,695 pinned values, 2.9 MiB, generated from the
  pre-port Cython at `f025448` (kernel digest `f5e6e269be47`).
  `python test/parity/generate.py --check` re-verifies it in under a
  second and needs no built tree. Two full regenerations produced a
  byte-identical manifest.
- **Coverage is derived from the tree, not transcribed.**
  `assert_full_coverage` walks the surviving `.pyx` for top-level
  `def`s and fails both ways — an unpinned `def`, or a case naming a
  `def` that no longer exists. Later phases therefore cannot delete a
  Cython module without the corpus noticing. `assert_no_rust_core`
  enforces rules.md rule 2 in code.
- **Two entry points raise rather than return at a kinematic edge.**
  `sigma_xx_to_v_to_pipi` and `sigma_xx_to_v_to_pi0v` raise `TypeError`
  at exactly `e_cm = 2·mx`; three blocks carry `raises` records. Not in
  the inventory's bug list. Task 1.2's exit criteria were patched to
  require replaying them.
- **The two `thermal_cross_section` implementations disagree above
  `x = 300`** — scalar returns `0.0`, vector saturates at `xnew = 300`.
  Both behaviors are pinned; Phase 05 must reproduce both or declare the
  unification.
- **Edge values are contract.** 26 negatives in `spectra.photon.muon`,
  123 negatives + 5 infinities in `cross_sections.scalar.sigma_xl_to_xl`.
  Stored as returned; do not "fix" them during a port without declaring
  it.
- **`--check`'s independence from a built tree is fragile.** It holds
  only because `cases.py` imports `HiggsPortal` / `KineticMixing` inside
  its factory functions. Hoisting those imports would silently make the
  integrity check require a full build.

## Decisions and Implementation Notes

- Specification (`test/parity/cases.py`) and generator
  (`test/parity/generate.py`) are separate modules so Task 1.2's runner
  imports the call convention rather than re-deriving it — Task 1.1.
- Provenance is recorded twice: the git SHA rules.md rule 2 asks for
  (necessarily with `dirty: true`, since the generating commit does not
  exist yet) and a `kernel_digest` over all 44 `.pyx`/`.pxd`/photon-CSV
  files, which is what actually identifies the kernels — Task 1.1.
- The Cython version in the manifest is read off the generated `.c`
  header, not `importlib.metadata`: build isolation keeps Cython out of
  the runtime environment — Task 1.1.
- Raises are captured as `nan` plus a manifest record of index,
  argument and exception *type*; the message is deliberately omitted
  because Cython rewords its errors between releases — Task 1.1.
- Five parent energies per spectrum rather than the four required; the
  extra is `M·(1 + 1e-12)`, straddling the `E − M < DBL_EPSILON`
  rest-frame short-circuit — Task 1.1.

## Files Changed

### Task 1.1

- `test/parity/cases.py`, `test/parity/generate.py`,
  `test/parity/README.md` — new.
- `test/parity/data/*.npz` (41) + `test/parity/data/manifest.json` — new.
- `../../phases/phase-01-parity-corpus.md` — Task 1.2 exit criteria
  gained the raise-replay bullet.
- This file and `../README.md` — status bookkeeping.

Nothing under `hazma/` was touched.

## Verification

- Task 1.1: `python test/parity/generate.py` → `41 cases / 623 blocks /
  2937.3 KiB`; `--check` → `corpus OK: 41 cases / 1580 arrays`. Both
  existing suites unchanged from the Phase 00 baseline (`pytest -q` →
  `57 passed, 10 skipped`; `pytest -q test` → `244 passed, 20 skipped`).
  Full command log and the guard negative-tests are in the task note.
- Remaining: bare `pytest` green incl. the parity suite (Tasks 1.2/1.3).

## Open Questions

- Whether every stored value is bit-reproducible on the Linux CI matrix
  is **unverified** — the corpus was captured on macOS/arm64. Task 1.2
  answers it when it sets per-function budgets, which is the right place
  for the answer.
- Task 1.4's scope narrowed but is not decided: the 159 skipped `.npy`
  arrays under `test/scalar_mediator/` and `test/vector_mediator/`
  overlap the cross-section and mediator-spectrum cases now pinned here,
  so the redundant-vs-complementary call has a concrete comparison
  target.

## Plan Impact

**Impact Level:** Update phase file (Task 1.1).

`../../phases/phase-01-parity-corpus.md`'s Task 1.2 exit criteria gained
a bullet requiring the runner to replay the manifest's `raises` records
rather than compare the stored `nan`. Patched in the task that found the
need. Nothing else canonical moved.

## Handoff to Next Task

**For the next agent working in Phase 01 (Task 1.2):** read
`test/parity/README.md`, then `test/parity/cases.py`'s module docstring,
then `task-1.1-corpus-generator.md`'s Findings and Handoff. The corpus
manifest records the generating git SHA — rules.md rule 2 — and the
`kernel_digest` that actually identifies the kernels.

**Currently safe to assume:**

- `test/` is green post-PR #31 and unchanged by Task 1.1; merging the
  suites in Task 1.3 is safe.
- Coverage of the 41 entry points does not need re-deriving —
  `assert_full_coverage` does it on every generation.
- `cases.py` is the single source of the call convention. Re-evaluate
  `block.array_call` / `block.scalar_call` against the live
  implementation rather than rebuilding argument tuples.
- The corpus pins the **post-fix** `two_body_momentum` values.

**Currently risky / unknown:**

- Task 1.2 must replay the three `raises` blocks, now an exit criterion.
- Do not hoist `cases.py`'s deferred model imports.
- Corpus grid density vs. repo-size budget is **settled**: 2.9 MiB
  against a ~10 MB ceiling, with `MAX_TOTAL_BYTES` failing generation
  above it.
- Cross-platform bit-reproducibility, per Open Questions.
