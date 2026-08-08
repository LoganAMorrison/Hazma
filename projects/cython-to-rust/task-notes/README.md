# Working Memory: cython-to-rust

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Status:** In Progress
**Plan References:** `../PLAN.md` (all sections)
**Related ADRs:** ADR-0001 (accepted), ADR-0002 (accepted 2026-08-04 —
Phase 03 Tasks 3.2/3.3 no longer gated), ADR-0003 (accepted
2026-08-04 with an Addendum the same day; non-deletion steps executed in
Task 0.5 on 2026-08-05, deletion executed in Task 0.2 on 2026-08-06 —
**fully discharged**)
**Depends On:** none

## Objective

Track cross-phase context and live phase status for the Cython→Rust
migration so any agent picking up work mid-project starts from facts,
not re-discovery. Per-task status lives in each `phase-XX/README.md`.

## Phases

| # | Phase | Phase file | Working memory | Status |
| --- | ------- | ----------- | ---------------- | -------- |
| 00 | Dead-code purge | [phase-00-dead-code-purge.md](../phases/phase-00-dead-code-purge.md) | [phase-00/README.md](phase-00/README.md) | **Complete (2026-08-06)** — all five tasks done; [learnings](../learnings/phase-00-dead-code-purge.md) |
| 01 | Golden parity corpus | [phase-01-parity-corpus.md](../phases/phase-01-parity-corpus.md) | [phase-01/README.md](phase-01/README.md) | **Complete (2026-08-08)** — all four tasks done; [learnings](../learnings/phase-01-parity-corpus.md) |
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
- ADR-0002 and ADR-0003 both accepted 2026-08-04.
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
- Test-infra fact agents kept tripping on, **settled in Task 1.3**: bare
  `pytest`, `preflight.sh`, and CI used to collect three different
  things (`hazma/**` from `setup.cfg`'s `testpaths`, `pytest -q test`,
  and the bare form respectively), so the parity corpus Task 1.2 landed
  gated nothing. pytest is now configured in `pyproject.toml` with
  `testpaths = ["hazma", "test"]` and all three run the bare command.
  Two consequences worth carrying forward: the suite costs 8m58s on the
  capturing machine under concurrent load because of `test/parity`, and
  it needs `pip install -e .` — a non-editable install leaves no
  extension in the tree the corpus insists on measuring.
- Local env fact: nothing is prebuilt on a fresh clone — build with uv
  (`uv venv`, `uv pip install -e .`) before preflight; expect preflight
  red on a clean tree otherwise. **And clean stale build artifacts
  first** — `find hazma -name '*.c' -o -name '*.cpp' -o -name '*.so' |
  xargs rm -f`. A worktree can inherit generated `.c`/`.cpp` produced
  under a different NumPy; their mtimes suppress re-cythonization and
  the build dies inside generated code with a misleading error
  (Task 0.1). Clean, the tree builds on Cython 3.2.9 / NumPy 2.5.1.
- **Preflight's black/isort verdict on `origin/master` — corrected in
  Task 0.3, then resolved.** Task 0.1 recorded "black wants to reformat
  34 files, isort errors on several `test/` files" as a property of the
  trunk. **It was not.** That was an unpinned newer black: CI pinned
  `black>=23.3,<25.0` while `pyproject.toml`'s dev extra allowed
  `<27.0`, and the two majors format differently, so at `cd0be2b` black
  24.10.0 called the trunk clean while black 26 wanted 33 files
  (PR #37). The divergence is now fixed — one pin, in
  `pyproject.toml`'s `[dependency-groups]` `lint` group, installed by CI
  and by you (`pip install --group lint`), with the repo reformatted to
  black 26.x. Install that group rather than a hand-picked version; see
  [`../../../docs/followups/done/black-pin-divergence-pyproject-vs-ci.md`](../../../docs/followups/done/black-pin-divergence-pyproject-vs-ci.md).
  The class is `[unpinned-formatter-version]` in
  `docs/agents/lessons.md`.
- **`ruff check hazma test` really is red on the trunk (6298 findings
  under ruff 0.16.1; 6844 was the count under the version current at
  Task 0.1 — the number tracks the linter, so re-measure rather than
  quote), and that does not block CI.** CI's ruff step is
  `ruff check --isolated --select E9,F63,F7,F82`, which deliberately
  ignores `pyproject.toml`'s much stricter config. Judge the configured
  form as a delta against the trunk; run the `--isolated` form to
  predict CI.
- Two `--paths` invocation traps, both hit in Task 0.1: passing it a
  `.pxd` makes black/ruff parse Cython as Python and fail, and passing
  it a _directory_ drags in that directory's pre-existing unformatted
  `.py`. Scope `--paths` to changed files, and omit it entirely when the
  diff has no Python.
- ~~`hazma._gamma_ray.gamma_ray_generator` compiles but has never been
  importable on `master`~~ — **gone as of Task 0.2 (2026-08-06)**, with
  the rest of `_gamma_ray/` and `_phase_space/`. The tree now has no
  extension that compiles but cannot import, and no C++ at all.
- **A stranded dependent belongs to the task that strands it** (Task
  0.2). Deleting `hazma/gamma_ray.py` would have left three files
  importing a module that no longer exists. All three went with it —
  `hazma/rh_neutrino/_rh_neutrino_spectra.py` (a legacy twin of the live
  `hazma/rh_neutrino/_spectra.py`, which already calls the ADR-0003
  replacement), the callerless `electron` helper in
  `hazma/spectra/_photon/__init__.py`, and `test/test_gamma_ray.py` — and
  the phase file's Task 0.2 exit criteria were patched to name them, so
  the widening is on the record rather than inferred from the diff.
  Rewriting the five `gamma_ray_decay` call sites to `dnde_photon` was
  explicitly rejected: the signatures, the FSR default and the three-body
  `msqrd` convention all differ, so it would have been an unoracled
  physics change in unreachable code — ADR-0003's own reasoning applied
  one level down.
- **A `git stash` round-trip un-stages a deletion** (Task 0.2), which
  makes `git ls-files` still list the removed paths and
  `scripts/agents/check_doc_citations.py` traceback on a
  tracked-but-absent file. `git add -A` after every pop. This bites
  precisely when following the recipe for proving preflight's isort/ruff
  redness is pre-existing.
- cyphus-diffeq (Hairer ODE ports) noted as possible future interest if
  relic density ever moves to Rust — out of scope here, candidate
  follow-up seed at close.
- **`gamma_ray_fsr` is no longer replacement-free** (Task 0.5). ADR-0003
  was written when the removal had no successor; its **Addendum
  (2026-08-04)** records that
  [`docs/followups/done/msqrd-driven-fsr-generator.md`](../../../docs/followups/done/msqrd-driven-fsr-generator.md)
  resolved ad-hoc as `hazma.spectra.dnde_photon_fsr` (repo-wide
  ADR-0001, PR #41, merged 2026-08-05). The settled replacement wording,
  which Phase 00's CHANGELOG entry and the Phase 07 aggregate both
  inherit: `gamma_ray_decay` → `hazma.spectra.dnde_photon`;
  `gamma_ray_fsr` → `hazma.spectra.dnde_photon_fsr`; **neither a
  drop-in** (`dnde_photon_fsr` takes the non-radiative *matrix element*
  rather than a rate float and has no `isp_masses`). The ADR's Decision
  body still reads "no direct replacement" by design — it is a dated
  record amended by its Addendum, and the forward-looking gate text was
  patched instead.
- **A clean wheel is not evidence of a clean sdist** (Task 0.4). Wheel
  contents come from `[tool.setuptools.packages.find]`, sdist contents
  from `MANIFEST.in`, and fixing one has never fixed the other. The
  sdist was shipping `.claude/`, `.codex/` and `projects/` — 103 files
  on this branch's final tree — because `global-include *.md` is a
  repo-wide sweep. Pruned: 501 → 398 files. **Phase 07
  Task 7.1 inherits the general lesson:** maturin has its own
  include/exclude machinery and reads neither of these files, so verify
  the tarball's contents directly after the cutover instead of assuming
  the wheel's cleanliness carries over.
- **The sdist install-and-run check is the real packaging gate** (Task
  0.4, new to this project): `uv build --sdist`, then
  `uv pip install --no-binary hazma dist/*.tar.gz` into a fresh venv and
  import-smoke from outside the repo. A path probe over `tar tzf` proves
  nothing dangles by *name*; only a source install proves the build
  works. Reuse it in Phase 07.
- **`_build.py` does not exist and has not since 2026-08-02** (`7a817f9`
  replaced it with `setup.py`). Thirteen durable docs still named it —
  including `AGENTS.md` and the rebuild-awareness rules every review
  skill points at — until Task 0.4 swept them. If a doc, skill or plan
  mentions it again, that text predates the sweep.
- **Sphinx orphan pages are a live doc-sweep hazard** (Task 0.5).
  `docs/source/index.rst` reaches nine documents; `limits.rst` and
  `models.rst` nest four more. Every other `docs/source/*.rst` is in no
  toctree — Sphinx still builds it, so an orphan is shipped-but-unlinked
  rather than absent. Task 0.5 deleted `gamma_ray.rst` on that basis;
  `rambo.rst` (documenting the long-gone `hazma.rambo.PhaseSpace`) is
  the same shape and is left to Task 0.2.

- **The parity corpus is live and self-checking** (Task 1.1). 41 cases /
  623 blocks / 1,580 arrays / 179,695 values, 2.9 MiB under
  `test/parity/data/`, captured from the pre-port Cython identified by
  kernel digest `f5e6e269be47`. The digest, not the manifest's git SHA,
  is the provenance record — the SHA is whatever was HEAD at generation
  time, always with `dirty: true`, whereas the digest certifies the
  `.pyx`/`.pxd`/CSV bytes the values actually came from. Generation also
  refuses to run unless every imported module resolves inside the
  repository, so an installed `hazma` cannot supply values the digest
  does not describe.
  `python test/parity/generate.py --check` re-verifies it in under a
  second **without a built tree**. Coverage of the 41 consumed entry
  points is *derived* from the tree by `assert_full_coverage`, not
  transcribed — so Phases 04–06 cannot delete a Cython module without
  the corpus objecting, and no later task needs to re-count.
- **Two live entry points raise instead of returning at threshold**
  (Task 1.1, not in the inventory's bug list):
  `sigma_xx_to_v_to_pipi` and `sigma_xx_to_v_to_pi0v` raise `TypeError`
  at exactly `e_cm = 2·mx` (Cython refusing a complex `**0.5` result).
  The scalar-mediator siblings do not. Pinned as `nan` plus a manifest
  `raises` record; Phase 05 ports them as-is per rules.md rule 1, and
  any repair is a separate declared change.
- **The two `thermal_cross_section` implementations disagree above
  `x = 300`** (Task 1.1): the scalar returns `0.0`
  (`hazma/scalar_mediator/_c_scalar_mediator_cross_sections.pyx:1401-1402`),
  the vector clips to `xnew = 300` and keeps evaluating
  (`hazma/vector_mediator/_c_vector_mediator_cross_sections.pyx:649`).
  Both behaviors are pinned. **Phase 05 must reproduce both or declare
  the unification** — a shared Rust helper is the obvious design and
  would silently move published numbers.
- **The corpus stops where Cython stops** (Task 1.4). Everything above
  the compiled boundary — `hazma/theory/__init__.py`'s dict assembly, the
  `"total"` sums, the branching-fraction division, the
  branching-fraction weighting of each spectrum, the line `bf`, plus both
  models' `partial_widths` — is pure Python, and no corpus case reaches
  it by construction (`cases.py` enumerates top-level `def`s in surviving
  `.pyx`). **A Phases 04–06 swap that repoints a kernel correctly but
  loses a branching-fraction weight passes the corpus and moves every
  published spectrum.** `test/test_theory_aggregation.py` is the gate for
  that layer: 21 identity-based tests, 0.6s, platform-independent. Run it
  either side of every swap.
- **Both mediator positron kernels return `nan` at exactly
  `0.510998928`** (Task 1.4) — the legacy `MASS_E` in
  `hazma/_utils/legacy_parameters.pxd:18`, against `0.5109989461` in
  `_utils/constants.pxd:5` and `hazma/parameters.py:50`. One point, not a
  window: a 2,000,001-point sweep of `[0.5109988, 0.5109990]` finds that
  single value with `0.0` on both sides, and the scalar and vector
  kernels agree. The corpus does **not** pin it (zero `nan` across 19,610
  pinned positron values), so a Rust port can land anywhere there and
  still pass. The constants divergence itself was already recorded
  (`../references/cython-inventory.md` §Bugs item 3); this consequence
  was not. **Phases 05/06 must reproduce the `nan` or declare the
  consolidation** —
  [`../../../docs/followups/todo/positron-spectrum-nan-at-legacy-electron-mass.md`](../../../docs/followups/todo/positron-spectrum-nan-at-legacy-electron-mass.md).
- **A worktree can inherit `.so` files whose source package is gone**
  (Task 1.1): this tree carried `_gamma_ray/` and `_phase_space/`
  extensions deleted in Task 0.2, giving 25 `.so` against `setup.py`'s
  20. Same class as the stale generated `.c` already in
  `docs/agents/environment.md`; the same clean-then-rebuild recipe
  fixes it, and any "N extensions" claim must be taken after it.

## Numerical impact so far

- **Task 0.1 (constants-header relocation): no public value changes.**
  All four mediator spectrum entry points and both model-level
  `total_spectrum` / `total_positron_spectrum` wrappers evaluated over
  `np.logspace(-2, 3, 200)` MeV at three mediator masses and every
  final-state mode — 64 arrays — before and after, **bit-for-bit
  identical** (max relative deviation 0.000e+00). Expected: `include`
  is a textual paste and the values moved verbatim.
- **Task 0.3 (dead-code purge): compiled surface unchanged; two
  declared drifts in the Cython→pure-Python helper swap.**
  - _Compiled spectra and cross sections: no change._ Every
    compiled-backed public entry point (12 `dnde_photon_*`, 2
    `dnde_positron_*`, 2 `dnde_neutrino_*` × 3 flavors, plus both
    models' `spectra` / `positron_spectra` /
    `annihilation_cross_sections` / `thermal_cross_section`) over
    `np.logspace(-2, 3, 200)` MeV — 171 arrays — **bit-for-bit
    identical** across the deletion and a full clean rebuild.
  - _`cross_section_prefactor` (Cython → `hazma.utils`):_ ≤**2.1e-7**
    relative within 1e-7 of the 2-body threshold, falling to ≤5e-15 at
    `cme ≥ 1.1 ×` threshold and ≤3.4e-16 well above it. Cause: the
    `hazma.utils` form builds `p` from `kallen_lambda`, which cancels at
    threshold; the deleted Cython twin used the factored product.
    Affects `hazma.deprecated.rambo` (public per `versioning.md` §6) and
    the broken-on-import `hazma.gamma_ray`. Seeded end-to-end check on
    `PhaseSpace.cross_section`: bit-identical at ordinary kinematics,
    1.8e-10 at threshold × (1+1e-7). Repair landed out-of-band —
    see the `two_body_momentum` entry below.
    [`docs/followups/done/cross-section-prefactor-threshold-cancellation.md`](../../../docs/followups/done/cross-section-prefactor-threshold-cancellation.md).
  - _`minkowski_dot` (Cython → `hazma.utils`):_ ≤**2.7e-14** relative
    over 1998 random four-vector pairs (≤3.2e-15 on on-shell momenta).
    Cause: the C compiler contracts `a*b - c*d` into an FMA. Only
    in-library consumer is `hazma/experimental/`, which
    `docs/versioning.md` excludes from the public surface.
  - Neither drift changes the project's `version_bump: major`, which the
    API removals already force.
- **Out-of-band (`two_body_momentum`, resolves the Task 0.3 follow-up):
  the two-body momentum is now computed from the factored form.** This
  reverses the `cross_section_prefactor` drift recorded above and goes
  past it: relative error against an exact-rational reference is ≤4.4e-16
  at every distance from threshold, versus 4e-2 at threshold for the
  `kallen_lambda` form. Values move by ≤2e-15 at `cme ≥ 1.1 ×`
  threshold (≤5e-16 at `≥ 2 ×`), 2.0e-13 at `1.01 ×`, and 1.3e-4 within
  1e-10 of threshold;
  at threshold itself `cross_section_prefactor` now returns `+inf`
  instead of a large finite number, and the entire below-threshold
  region is now NaN (λ turns positive again below `|m1 - m2|`, where
  both the Källén form and the first factored draft returned a finite,
  meaningless momentum). Also repointed
  `hazma.phase_space` two-body integration and `hazma.deprecated.rambo`.
  **Phase 01 corpus note:** this landed before the parity corpus is
  generated, so the corpus captures the fixed values and the Rust port
  must reproduce _these_, not the pre-fix ones.

- **Task 0.5 (execute ADR-0003's non-deletion steps): no public value
  changes.** The diff is durable docs plus one docstring hunk in
  `hazma/spectra/_photon/__init__.py`; no code path, signature, or
  constant is touched, so no grid evaluation applies.
- **Task 0.2 (delete the phase-space / gamma-ray slice): no public value
  changes.** Every compiled-backed public entry point over
  `np.logspace(-2, 3, 200)` MeV — the 12 `dnde_photon_*`, 2
  `dnde_positron_*` and 2 `dnde_neutrino_*` at three parent energies,
  plus both models' `spectra()` / `positron_spectra()` /
  `annihilation_cross_sections()` / `thermal_cross_section()` at three
  mediator masses — **159 arrays, bit-for-bit identical** across the
  deletion and a full clean rebuild (max relative deviation 0.000e+00).
  Expected: everything removed was unbuilt, unimported, or broken on
  import, and nothing surviving imports or cimports it. What *did* change
  is the **public API surface**, which is where this task's `major`
  weight sits: `hazma.gamma_ray` (both functions, each with a named
  non-drop-in replacement) and `hazma.deprecated.rambo` are gone, and the
  `### Removed` block under `CHANGELOG.md`'s `[Unreleased]` is the
  settled wording for the Phase 07 aggregate.

- **Task 0.4 (prune build and packaging config): no public value
  changes.** 213 arrays — 12 `dnde_photon_*`, 12 `dnde_positron_*` and
  12 `dnde_neutrino_*` over `np.logspace(-2, 3, 200)` MeV at parent
  energies 150 / 500 / 1500 MeV, plus both models' `spectra()`,
  `positron_spectra()`, `annihilation_cross_sections()` and
  `thermal_cross_section()` at mediator masses 200 / 550 / 1200 MeV —
  **bit-for-bit identical** across the change and a clean rebuild (max
  relative deviation 0.000e+00). Expected, and the mechanism is
  checkable rather than merely plausible: the only executable change is
  the removal of an `if cpp:` branch no call site reaches, so every
  `Extension` object `setup.py` builds is unchanged and the compiled
  artifacts are identical. **Phase 00 therefore closes with the public
  compiled surface exactly where it started**; the only declared drifts
  in the whole phase are Task 0.3's two pure-Python helper swaps and the
  out-of-band `two_body_momentum` repair, both above.

- **Task 1.1 (parity corpus generator): no public value changes**
  (verified: `git diff origin/master -- hazma` is empty). The diff adds
  only `test/parity/` and project bookkeeping, plus one bullet in the
  Phase 01 file; no library module, signature, constant or build input
  is touched, so no grid evaluation applies. Both suites reproduced the
  Phase 00 closing counts exactly (`pytest -q` → 57 passed / 10 skipped;
  `pytest -q test` → 244 passed / 20 skipped). What the task *did*
  produce is the baseline every later drift is measured against: 179,695
  pinned values across the 41 consumed entry points. Two pre-existing
  behaviors it recorded — the `TypeError` at `e_cm = 2·mx` and the
  `x > 300` thermal divergence — are observations, not drifts, and are
  under Findings above.

- **Tasks 1.1–1.3 (parity corpus, its runner, and the wiring): no public
  value changes** — none of the three touched `hazma/` (verified:
  `git diff origin/master -- hazma` empty on each). Task 1.2 additionally
  *proves* it for the whole compiled surface: `pytest -q test/parity` →
  `626 passed` with every one of the 41 entry points held to
  bit-equality against the corpus, on the environment that captured it.
  Task 1.3 re-ran that proof as part of the merged suite (bare
  `pytest -q` → 935 passed / 30 skipped, parity in exact mode) and is
  what makes it a standing gate rather than a manual run.

- **Task 1.4, 2026-08-08 (retire the legacy `.npy` suites): no public value
  changes** (verified: `git diff origin/master -- hazma` is empty — 0
  lines). The diff touches only `test/`, `docs/followups/` and
  `projects/`; no library module, signature, constant or build input is
  reachable from it, so no grid evaluation applies. What the task did
  produce is a *second* gate beside the corpus:
  `test/test_theory_aggregation.py` pins the pure-Python aggregation as
  identities (`total` is the channel sum, a branching fraction is a
  cross-section ratio, a spectrum is `bf × kernel`, a line's `bf` is its
  channel's) plus three two-body closed forms. Eleven implementation
  mutations confirm each class fires. Two pre-existing behaviors it
  *measured* — the `nan` at the legacy `MASS_E` and the rejected scalar
  energies — are observations, not drifts, and are under Findings and
  Open Questions.

(Per-function drift lines land here as Phase 04–06 swaps merge; the
Phase 07 CHANGELOG is assembled from this section — do not reconstruct
it from memory.)

## Decisions and Implementation Notes

- Rust + PyO3 + maturin over pybind11; single abi3 `hazma._core`;
  setuptools-rust coexistence during migration → ADR-0001 (Accepted).
- No GSL-derived (GPL-3) code in tree or dependency graph; cephes
  lineage (`spec_math`) for specfun; netlib-QUADPACK translation for
  the integrator; cyphus crates as out-of-repo oracles only →
  ADR-0002 (Accepted 2026-08-04 — Hazma stays MIT).
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
  `hazma/_gamma_ray/gamma_ray_generator.pyx` line 24 **as of `c6991a6`**
  was a fifth site in a _built_ `Extension`, so skipping it would have
  broken `pip install -e .` before Task 0.2 could delete the module.
  Criterion now names five built sites plus the two unbuilt `_decay/`
  extras. (Task 0.2 has since deleted that file; retrieve it with
  `git show c6991a6:hazma/_gamma_ray/gamma_ray_generator.pyx`.)
- **Plan-review round 2 (2026-08-03)**, two completeness fixes: the
  inventory's boost-retirement claim corrected to Phase 06 Task 6.4
  (capi survivor `hazma/spectra/_positron/_pion.pyx:10` cimports the
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
- **Task 0.3** — deleted `hazma/_decay/`, `hazma/_positron/`,
  `hazma/_neutrino/`, `hazma/field_theory_helper_functions/`, the three
  `hazma/__*.py` legacy shims, `spectra/_positron/_kaon.pyx`,
  `test/decay/`, and the dead half of `_utils/boost.pyx`;
  `minkowski_dot` given a pure-Python home in `hazma/utils.py` and all
  `common_functions` callers repointed there; `setup.py`,
  `test/conftest.py`, `pyproject.toml`, `MANIFEST.in` updated; nine
  durable docs swept; `test/test_utils.py` and a
  `cross_section_prefactor` follow-up added. 135 files, −29,337 lines.
  Full list in [`phase-00/README.md`](phase-00/README.md).
- **Task 0.2** — deleted `hazma/_gamma_ray/`, `hazma/_phase_space/`,
  `hazma/gamma_ray.py`, `hazma/deprecated/rambo.py` (emptying the
  package), `rh_neutrino/_rh_neutrino_fsr_four_body.{pyx,pyi}`,
  `rh_neutrino/_rh_neutrino_spectra.py`, `test/test_gamma_ray.py` and
  `docs/source/rambo.rst`; dropped the dead `electron` helper and three
  orphaned imports from `hazma/spectra/_photon/__init__.py`, two
  extension groups from `setup.py`, and the last `collect_ignore` entry
  from `test/conftest.py`. `CHANGELOG.md` gained the `### Removed` block;
  seven durable docs and four `docs/followups/` records were swept, with
  every citation into a deleted file pinned to `c6991a6`. 43 files
  (+816 / −4,413); under `hazma/` and `test/` alone, 25 files and
  **−4,023 lines against +6**. Full list in
  [`phase-00/README.md`](phase-00/README.md).
- **Task 0.4** — build/packaging reconciliation, closing the phase.
  `setup.py`'s `make_extension` lost its unreachable C++ branch (and,
  on the same signature, `List[str]` → `list[str]` plus a return type,
  taking the file from 5 configured-`ruff` findings to 0); `MANIFEST.in`
  gained `prune .claude` / `.codex` / `projects`, which took the sdist
  from 501 to 398 files; `pyproject.toml` audited and unchanged.
  The thirteen durable docs that still named `_build.py` were swept —
  twelve by rename, plus `docs/versioning.md`, whose sole occurrence sat
  inside an obsolete blockquote that was deleted outright (its `VERSION`
  snippet was re-derived at the same time). One follow-up filed
  ([sdist payload](../../../docs/followups/todo/sdist-ships-generated-c-and-docs.md)).
  Phase closed: learnings written, phase frontmatter
  `status: Complete`, `PLAN.md` Phases row updated. Full list in
  [`phase-00/README.md`](phase-00/README.md).
- **Task 0.5** — docs repointed off `hazma.gamma_ray` ahead of the
  delete: `docs/source/gamma_ray.rst` deleted (orphan page),
  `hazma/spectra/_photon/__init__.py`'s `electron` docstring repointed,
  `docs/PR_GUIDELINES.md`'s `limits` scope row pruned, and the stale
  "replacement-free" wording corrected in the phase file, `PLAN.md`, and
  `docs/followups/done/msqrd-driven-fsr-generator.md`. Repo-wide
  ADR-0001 flipped Proposed → Accepted (PR #41 merged). Full list in
  [`phase-00/README.md`](phase-00/README.md).

### Phase 01

- **Task 1.1** — new `test/parity/`: `cases.py` (the corpus
  specification, 41 cases / 623 blocks, plus the three guards),
  `generate.py` (generate / `--check`), `README.md`, and
  `data/` (41 `.npz` + `manifest.json`, 2.9 MiB). One canonical patch:
  `../phases/phase-01-parity-corpus.md`'s Task 1.2 exit criteria gained
  the raise-replay bullet. Nothing under `hazma/` touched. Full list in
  [`phase-01/README.md`](phase-01/README.md).
- **Tasks 1.2–1.3** — the runner (`test/parity/test_parity.py`,
  `tolerances.py`) and the wiring that made one bare `pytest` the whole
  gate (`pyproject.toml`'s `[tool.pytest.ini_options]`, the CI editable
  reinstall, `preflight.sh`'s empty `--tests` default). Nothing under
  `hazma/` touched. Full list in
  [`phase-01/README.md`](phase-01/README.md).
- **Task 1.4** — phase closure. Deleted 96 files: both skipped mediator
  test classes, their two `generate_test_data.py` producers, the 90
  `.npy` arrays under `test/{scalar,vector}_mediator/data/`, the 0-byte
  `test/positron/test_positron.py`, and `test/rh_neutrino/widths.py` (a
  matplotlib script). Renamed `test/rh_neutrino/integration.py` →
  `test_rh_neutrino_integration.py`. Added
  `test/test_theory_aggregation.py` (21 tests / 69 collected) and two
  `docs/followups/todo/` entries. Phase closed: learnings written, phase
  frontmatter `status: Complete`, `PLAN.md` Phases row updated. Nothing
  under `hazma/` touched. Full list in
  [`phase-01/README.md`](phase-01/README.md).

## Verification

- Scaffolding PR: `scripts/agents/preflight.sh` (repo gate; no code
  changes).
- Per-phase Verification sections live in `phase-XX/README.md`.
- **Phase 01 closing state (2026-08-08):** bare `pytest -q` →
  `1006 passed, 13 skipped` on the capturing environment (1019 collected:
  67 `hazma` + 952 `test`), parity suite included and in exact mode;
  `python test/parity/generate.py --check` → `corpus OK: 41 cases / 1580
  arrays`. Off macOS, CI runs `pytest --ignore=test/parity`. No skip
  anywhere in the repo is waiting on this project. The public compiled
  surface is still exactly where Phase 00 left it.
- **Phase 00 closing state (2026-08-06):** 20 `.pyx` ↔ 20 declared
  `Extension`s ↔ 20 `.so`, verified as a set equality; zero C++;
  `pytest -q test` → `244 passed, 20 skipped`, bare `pytest -q` →
  `57 passed, 10 skipped`; sdist and wheel both build, and the sdist
  installs and runs in a fresh venv from outside the repo. The public
  compiled surface is unchanged from where the phase started.

## Open Questions

- ~~**ADR-0002 sign-off** (Logan): accept the license-clean-numerics
  decision, or deliberately take the GPL route?~~ — **closed
  2026-08-04: accepted.** Hazma stays MIT and no GPL-3 crate enters the
  tree or the dependency graph, so **Phase 03 Tasks 3.2/3.3 are
  unblocked**: `spec_math` (cephes lineage) for specfun, a fresh
  netlib-QUADPACK translation for the integrator, cyphus as an
  out-of-repo oracle only.
- ~~**ADR-0003 sign-off** (Logan): confirm deletion of the
  broken-on-import `hazma.gamma_ray`~~ — **closed and fully
  discharged.** Accepted 2026-08-04; non-deletion steps executed in
  Task 0.5 on 2026-08-05 (replacement status recorded, docs repointed);
  the deletion itself executed in Task 0.2 on 2026-08-06, with its
  CHANGELOG entry. **Phase 00 closed the same day.**
  `gamma_ray_fsr`'s successor, `hazma.spectra.dnde_photon_fsr`, shipped
  via
  [`../../../docs/followups/done/msqrd-driven-fsr-generator.md`](../../../docs/followups/done/msqrd-driven-fsr-generator.md).
- ~~`cross_section_prefactor`'s threshold cancellation (found in Task
  0.3, filed as a follow-up): fix it **after** the port, or Phase 01's
  corpus pins the cancelling values?~~ — **closed: the repair landed
  out-of-band before Phase 01 started**, via `two_body_momentum`'s
  factored form (see "Out-of-band" under "Numerical impact so far").
  The question is moot; the corpus will pin the fixed values.
- Phase 05 parallelism: run 05 alongside 04 (no shared files) or keep
  strictly serial? Decide when Phase 04 starts, based on who's driving.
- **The sdist payload** (opened by Task 0.4, the first task in the
  project to build one): the tarball still ships 20 cythonized `*.c`,
  `docs/`, `test/` and `notebooks/`, and `pyproject.toml`'s package-data
  says `*.pyd` where `*.pxd` was surely meant. Deferred deliberately —
  judgment calls, not defects — but **time-boxed to before Phase 07 Task
  7.1**, because maturin does not read `MANIFEST.in` and the same
  decisions cost more to express afterwards. Filed as
  [`../../../docs/followups/todo/sdist-ships-generated-c-and-docs.md`](../../../docs/followups/todo/sdist-ships-generated-c-and-docs.md).
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
3. Check Open Questions above. No ADR sign-off is outstanding — all
   three project ADRs are Accepted, so no phase carries a decision gate.

**Currently safe to assume:**

- The dead-code map and entry-point inventory in
  [`../references/cython-inventory.md`](../references/cython-inventory.md)
  were verified against 2.1.0 (Aug 2026) and the file declares itself a
  snapshot. **Every row of its dead-code table is now done** — Task 0.1
  relocated the constants header, Task 0.3 executed the bulk, and Task
  0.2 finished it with `_gamma_ray/`, `_phase_space/`,
  `deprecated/rambo.py` and
  `rh_neutrino/_rh_neutrino_fsr_four_body.pyx`. Read that file for the
  **live surface** and the cimport DAG, which Phases 04–06 still need;
  read its headline counts as history.
- **20 extensions, 20 `.pyx`, 17 `.pxd`, zero C++**, and as of Task 0.4
  `setup.py`'s declared list is *verified* to be that same set, not
  merely the same size. Re-derive with the clean-then-rebuild recipe
  rather than quoting this; a stale `.so` makes a wrong list look right.
- **Phases 00 and 01 are both Complete** (2026-08-06 and 2026-08-08).
  Read
  [`../learnings/phase-00-dead-code-purge.md`](../learnings/phase-00-dead-code-purge.md)
  and
  [`../learnings/phase-01-parity-corpus.md`](../learnings/phase-01-parity-corpus.md)
  rather than their nine task notes — the learnings are the
  distillation, the notes are history. **Phase 02 (the Rust scaffold) is
  next.** No phase carries a decision gate.
- **The parity corpus is the gate from here on.** `python
  test/parity/generate.py --check` verifies it; `test/parity/cases.py`
  is the single source of every entry point's call convention. Do not
  regenerate it from a tree in which any kernel runs on Rust — rules.md
  rule 2, now enforced in code by `assert_no_rust_core`.
- **The build entry point is `setup.py`.** `_build.py` was deleted in
  `7a817f9` (2026-08-02) and Task 0.4 swept the thirteen durable docs
  that still named it, so `AGENTS.md`, `docs/`, the skills, `.github/`
  and the build config are all clean. The name still appears under
  `projects/` — that is this phase's own record of the sweep, not a live
  reference.
- **The sdist and wheel both build, and the sdist installs and runs** in
  a fresh venv from outside the repo (recipe in Task 0.4's note; reuse
  it in Phase 07). Neither ships a deleted path; neither ships the agent
  scaffolding any more.
- **The suites are merged and green on the capturing platform**: bare
  `pytest -q` → 1006 passed / 13 skipped at Phase 01 close (Task 1.4,
  2026-08-08), from 935/30 at Task 1.3. Task 1.4's delta: +69 aggregation
  tests, +2 from the `rh_neutrino` rename, −17 skips as the two legacy
  mediator classes left. Forcing budget mode drops one test to a skip
  rather than failing — the parity suite reports its mode that way.
  Re-derive rather than quoting; the historical series is in
  [`phase-01/README.md`](phase-01/README.md).
- **`test/test_theory_aggregation.py` is the model-layer gate the corpus
  cannot be** (Task 1.4): identities over `hazma/theory/`'s pure-Python
  aggregation, no golden data, 0.6s, and the only numerical gate in the
  repo that is not scoped to the capturing platform. **Phases 04–06 run
  it either side of every kernel swap.**
- **Off macOS the corpus does not reproduce**, so CI runs
  `pytest --ignore=test/parity` on every entry except macOS (339
  collected there vs 965). Linux fails ~70-75 blocks: mostly last-bit
  libm noise, but six are cancellation points where the pinned value
  flips sign. That is a corpus defect, it blocks the Phase 04-06 port
  gate as much as CI, and it is tracked in
  [`../../../docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md`](../../../docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md)
  — **read it before Phase 04**.
- The legacy constants table lives at
  `hazma/_utils/legacy_parameters.pxd` and is now its **only** copy.
  `hazma.utils` is the only home for `cross_section_prefactor` and
  `minkowski_dot`.
- **`hazma.gamma_ray` is gone, docs and all** (Task 0.5 swept, Task 0.2
  deleted). Surviving mentions are dated records — ADRs, follow-ups, task
  notes — plus the `CHANGELOG.md` `### Removed` block, which carries the
  settled replacement wording for the Phase 07 aggregate:
  `gamma_ray_decay` → `hazma.spectra.dnde_photon`, `gamma_ray_fsr` →
  `hazma.spectra.dnde_photon_fsr`, **neither a drop-in**. Do not
  reconstruct it from memory.

**Currently risky / unknown:**

- `spec_math`'s `li2` argument convention vs scipy's `spence` is
  unverified — Task 3.2 pins it before anything depends on it.
- ~~Phase 01's corpus will capture `cross_section_prefactor`'s
  near-threshold cancellation as if it were intended behavior.~~ — no
  longer a risk: the repair landed before Phase 01 (see "Out-of-band"
  under "Numerical impact so far"). The live obligation is the mirror
  image — **the corpus must pin the post-fix values, and the Rust port
  must reproduce those, not the pre-fix ones.**
- ~~**The corpus was captured on macOS/arm64** ... whether every stored
  value is bit-reproducible on the Linux CI matrix is unverified~~ —
  **measured in Task 1.3: it is not.** See the "Off macOS the corpus does
  not reproduce" bullet above; the corpus is scoped to its capturing
  platform until its follow-up lands.
- **Two Task 1.4 follow-ups ripen inside this project.** The
  [`MASS_E` `nan`](../../../docs/followups/todo/positron-spectrum-nan-at-legacy-electron-mass.md)
  before Phases 05/06 — `rules.md` rule 4 makes it a declared numerical
  change either way, and deciding after the swap costs a second one. The
  [scalar-energy contract](../../../docs/followups/todo/model-spectra-reject-scalar-energies.md)
  during 04–06 — `Theory.spectra` and `Theory.positron_spectra` reject the
  scalar energies their docstrings advertise, and the compiled half of
  that resolves itself if the port normalizes at the public boundary.
