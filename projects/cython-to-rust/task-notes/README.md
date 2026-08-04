# Working Memory: cython-to-rust

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Status:** In Progress
**Plan References:** `../PLAN.md` (all sections)
**Related ADRs:** ADR-0001 (accepted), ADR-0002 (**proposed — needs
Logan's sign-off before Phase 03 Tasks 3.2/3.3**), ADR-0003
(**proposed — needs Logan's sign-off before Phase 00 Task 0.2**)
**Depends On:** none

## Objective

Track cross-phase context and live phase status for the Cython→Rust
migration so any agent picking up work mid-project starts from facts,
not re-discovery. Per-task status lives in each `phase-XX/README.md`.

## Phases

| # | Phase | Phase file | Working memory | Status |
| --- | ------- | ----------- | ---------------- | -------- |
| 00 | Dead-code purge | [phase-00-dead-code-purge.md](../phases/phase-00-dead-code-purge.md) | [phase-00/README.md](phase-00/README.md) | In Progress (0.1 done) |
| 01 | Golden parity corpus | [phase-01-parity-corpus.md](../phases/phase-01-parity-corpus.md) | [phase-01/README.md](phase-01/README.md) | Not started |
| 02 | Rust scaffold | [phase-02-rust-scaffold.md](../phases/phase-02-rust-scaffold.md) | [phase-02/README.md](phase-02/README.md) | Not started |
| 03 | Numerics foundation | [phase-03-numerics-foundation.md](../phases/phase-03-numerics-foundation.md) | [phase-03/README.md](phase-03/README.md) | Not started |
| 04 | Spectra kernels | [phase-04-spectra-kernels.md](../phases/phase-04-spectra-kernels.md) | [phase-04/README.md](phase-04/README.md) | Not started |
| 05 | Mediator cross sections | [phase-05-mediator-cross-sections.md](../phases/phase-05-mediator-cross-sections.md) | [phase-05/README.md](phase-05/README.md) | Not started |
| 06 | Mediator spectra | [phase-06-mediator-spectra.md](../phases/phase-06-mediator-spectra.md) | [phase-06/README.md](phase-06/README.md) | Not started |
| 07 | Cutover + close | [phase-07-cutover.md](../phases/phase-07-cutover.md) | [phase-07/README.md](phase-07/README.md) | Not started |

```text
00 ──► 01 ──► 02 ──► 03 ──► 04 ──► 06 ──► 07
                        └──► 05 ──┘
```

## Exit Criteria

- All eight phases Complete; zero `.pyx`/`.pxd` in the tree; all 41
  consumed entry points served by `hazma._core` (the 2 unconsumed
  `sigma_xx_to_all` exports dropped in Phase 05); maturin backend live.
- ADR-0002 and ADR-0003 accepted (or superseded).
- Closing PR bumps `VERSION` in `hazma/__init__.py` per `PLAN.md`'s
  `version_bump:` frontmatter and adds a `CHANGELOG.md` entry naming
  this project slug, with the aggregated drift table. See
  [`../../../docs/versioning.md`](../../../docs/versioning.md).

## Inputs Reviewed

- `../PLAN.md`, both `../references/*.md`, `../rules.md` — project
  contract and grounded facts.
- August 2026 analysis session: every `.pyx`/`.pxd` read in full by
  five parallel passes; findings distilled into the references.
- rust-cyphus crates (cloned + test-run 2026-08-03, rustc 1.96):
  results in `../references/numerics-replacements.md`.

## Findings

- The five-pass audit's load-bearing facts (dead-code evidence, entry
  points, cimport DAG, quad call sites, bugs) are **already promoted**
  into `../references/` — consult those, not this section, and
  re-verify line numbers at execution time (snapshot of 2.1.0).
- Test-infra fact agents keep tripping on: bare `pytest` collects only
  `hazma/**` (`setup.cfg` `testpaths`), while preflight runs
  `pytest -q test` — two disjoint suites until Phase 01 Task 1.3 merges
  them. Zero compiled-layer pinned tests run anywhere today.
- Local env fact: nothing is prebuilt on a fresh clone — build with uv
  (`uv venv`, `uv pip install -e .`) before preflight; expect preflight
  red on a clean tree otherwise. **And clean stale build artifacts
  first** — `find hazma -name '*.c' -o -name '*.cpp' -o -name '*.so' |
  xargs rm -f`. A worktree can inherit generated `.c`/`.cpp` produced
  under a different NumPy; their mtimes suppress re-cythonization and
  the build dies inside generated code with a misleading error
  (Task 0.1). Clean, the tree builds on Cython 3.2.9 / NumPy 2.5.1.
- **Preflight's Python gates are red on `origin/master` itself** —
  measured at the trunk in Task 0.1: `black --check hazma test` wants to
  reformat 34 files, `ruff check hazma test` reports 6844 errors, isort
  errors on several `test/` files. Do not read these as your regression;
  compare against the trunk baseline and judge only the delta. Two
  invocation traps, both hit in Task 0.1: passing `--paths` a `.pxd`
  makes black/ruff parse Cython as Python and fail, and passing it a
  _directory_ drags in that directory's pre-existing unformatted `.py`.
  Scope `--paths` to changed files, and omit it entirely when the diff
  has no Python.
- `hazma._gamma_ray.gamma_ray_generator` compiles but has never been
  importable on `master` (`from hazma import rambo`; `hazma/rambo.py`
  does not exist). It is still a live `Extension` in `setup.py`, so it
  must keep _compiling_ until Task 0.2 deletes it — but it can never be
  part of an import-smoke set.
- cyphus-diffeq (Hairer ODE ports) noted as possible future interest if
  relic density ever moves to Rust — out of scope here, candidate
  follow-up seed at close.

## Numerical impact so far

- **Task 0.1 (constants-header relocation): no public value changes.**
  All four mediator spectrum entry points and both model-level
  `total_spectrum` / `total_positron_spectrum` wrappers evaluated over
  `np.logspace(-2, 3, 200)` MeV at three mediator masses and every
  final-state mode — 64 arrays — before and after, **bit-for-bit
  identical** (max relative deviation 0.000e+00). Expected: `include`
  is a textual paste and the values moved verbatim.

(Per-function drift lines land here as Phase 04–06 swaps merge; the
Phase 07 CHANGELOG is assembled from this section — do not reconstruct
it from memory.)

## Decisions and Implementation Notes

- Rust + PyO3 + maturin over pybind11; single abi3 `hazma._core`;
  setuptools-rust coexistence during migration → ADR-0001 (Accepted).
- No GSL-derived (GPL-3) code in tree or dependency graph; cephes
  lineage (`spec_math`) for specfun; netlib-QUADPACK translation for
  the integrator; cyphus crates as out-of-repo oracles only →
  ADR-0002 (**Proposed**).
- Capi-survivor exception: four spectra Cython extensions outlive their
  Phase 04 swap until Phase 06 Task 6.4, because the mediator spectrum
  `.pyx` cimport their `__pyx_capi__` symbols — recorded in the
  Phase 04 file's Goal block.
- Constants bit-parity before consolidation → `../rules.md` rule 4.
- **Plan-review round 1 (2026-08-03)** forced four canonical changes:
  (1) `version_bump` → `major` (Phase 00 deletes
  `hazma/deprecated/rambo.py`; any `deprecated/` removal is `major`
  per versioning.md); (2) `hazma.gamma_ray` default is now _delete_
  via ADR-0003 — the rebuild branch was dropped because the module
  cannot run, so no behavior-preserving baseline exists; (3) hybrid
  wheels stay CPython-tagged through Phases 02–06, only `hazma._core`
  itself is abi3 (limited API); distribution-level abi3 tags are a
  Phase 07 assertion; (4) counts re-derived from source: 20 surviving
  extensions (not 19), 43 public defs / 41 consumed (corpus covers
  41; 2 `sigma_xx_to_all` dropped in Phase 05), 44 `.pyx` + 33 `.pxd`
  (77 files). QAGP breakpoint preprocessing (endpoint-coincident and
  out-of-interval points both occur live) added to Task 3.3's exit
  criteria.
- **Task 0.1 (2026-08-04)** patched the phase-00 file: its Task 0.1
  exit criterion named "the four live include sites", but
  `_gamma_ray/gamma_ray_generator.pyx:24` is a fifth site in a _built_
  `Extension`, so skipping it breaks `pip install -e .` before Task 0.2
  can delete the module. Criterion now names five built sites plus the
  two unbuilt `_decay/` extras.
- **Plan-review round 2 (2026-08-03)**, two completeness fixes: the
  inventory's boost-retirement claim corrected to Phase 06 Task 6.4
  (capi survivor `spectra/_positron/_pion.pyx:10` cimports the
  _linked_ `boost_delta_function`, so the compiled `_utils.boost`
  extension must outlive Phase 04); Task 0.5/ADR-0003 now name the
  module's real public API — `gamma_ray_decay` (superseded by
  `hazma.spectra.dnde_photon`) and `gamma_ray_fsr` (removed with no
  direct replacement; nearest are the Altarelli–Parisi
  approximations) — instead of the wrapped compiled names
  `gamma`/`gamma_point`.

## Files Changed

### Phase 00

- **Task 0.1** — `hazma/_decay/parameters.pxd` relocated to
  `hazma/_utils/legacy_parameters.pxd`; include repointed in seven
  `.pyx`; phase-00 file's Task 0.1 criterion corrected; follow-up filed
  for the `WIDTH_K`/`WIDTH_PI` exponent bug. Full list in
  [`phase-00/README.md`](phase-00/README.md).

## Verification

- Scaffolding PR: `scripts/agents/preflight.sh` (repo gate; no code
  changes).
- Later: per-phase Verification sections in `phase-XX/README.md`.

## Open Questions

- **ADR-0002 sign-off** (Logan): accept the license-clean-numerics
  decision, or deliberately take the GPL route? Gates Phase 03
  Tasks 3.2/3.3.
- **ADR-0003 sign-off** (Logan): confirm deletion of the
  broken-on-import `hazma.gamma_ray`. Gates Phase 00 Task 0.2; if
  rejected, the phase halts for a plan revision (rebuild would be a
  new feature via `docs/followups/`, not this project).
- Phase 05 parallelism: run 05 alongside 04 (no shared files) or keep
  strictly serial? Decide when Phase 04 starts, based on who's driving.
- ~~Whether the mediator cross-section `.pyx` include a constants
  header~~ — **closed by Task 0.1: they contain no `include` directive
  at all.**

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).

## Handoff to Next Task

**For the next agent starting any task in this project:**

1. Read `../PLAN.md` end-to-end, then this file, then the active
   phase's `phase-XX/README.md`.
2. Load the reference file(s) the phase's Prerequisites name — the
   references replace re-reading the Cython audit.
3. Check Open Questions above; Phase 03 must not start Tasks 3.2/3.3
   while ADR-0002 is unaccepted.

**Currently safe to assume:**

- The dead-code map and entry-point inventory were verified against
  2.1.0 (Aug 2026); the tree is unchanged apart from Task 0.1's
  header relocation.
- `test/` is green (52 passed / 20 skipped as of Task 0.1, 2026-08-04;
  51/20 at PR #31) — merging the suites in Task 1.3 is safe.
- The legacy constants table now lives at
  `hazma/_utils/legacy_parameters.pxd`; nothing under `hazma/_decay/`
  is included by a built extension, so Task 0.3 can delete the
  directory without include-path fallout.

**Currently risky / unknown:**

- `spec_math`'s `li2` argument convention vs scipy's `spence` is
  unverified — Task 3.2 pins it before anything depends on it.
