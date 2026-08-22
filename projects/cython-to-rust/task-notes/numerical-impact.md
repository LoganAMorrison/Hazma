# Numerical impact so far: cython-to-rust

**Project:** cython-to-rust
**Moved:** 2026-08-21, from [`README.md`](README.md) §"Numerical impact
so far" — lines 893–1491 of that file at commit `c57ce4f`; verbatim,
in order, nothing summarised. Reproduce the move with

```sh
git show c57ce4f:projects/cython-to-rust/task-notes/README.md | sed -n '894,1491p'
```

This is the project's running record of every value the library
returns that the port has moved, and of every public path checked and
found unchanged. It is what `../PLAN.md` §"Closing this project" and
Phase 07's CHANGELOG aggregation are assembled from — **do not
reconstruct it from memory at close time.** The contract is unchanged
by the move: one entry per task that touched a public code path, giving
the function, the grid checked, and the result ("unchanged", or the
magnitude and direction of the shift), appended at the end in task
order (`../rules.md` rule 3; `.claude/skills/execute-single-task` step
6b). `README.md` keeps the `## Numerical impact so far` heading as a
pointer here so that every citation of that section still resolves.

---

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

- **Task 2.1, 2026-08-08 (Rust crate + setuptools-rust): no public value
  changes**, and for the first time in this project that is *measured at
  bit-equality* rather than argued from the diff. `git diff origin/master
  -- hazma` is one file, the non-executable `hazma/_core.pyi` (+19); the
  one new runtime artifact, `hazma/_core.abi3.so`, is imported by nothing
  under `hazma/`. The stronger statement: on the corpus's capturing
  environment (CPython 3.12.12, macOS/arm64) the parity suite ran in
  **bit-equality mode** — `rtol = 0` across all 41 consumed entry points,
  626 blocks, 1,580 arrays, 179,695 pinned values — and passed, inside a
  bare `pytest -q` of `1009 passed, 13 skipped` (1022 collected, +3 on
  Phase 01's 1019; the skip count is unchanged, which is what proves the
  mode). No ad-hoc grid sweep is reported because the corpus is a
  stricter grid than any of them. **That evidence only exists because the
  task fixed the mode switch its own deliverable would otherwise have
  broken** — see the served-vs-importable finding above; shipped without
  it, this line could have claimed no better than 1e-8.

- **Task 2.2, 2026-08-08 (CI, preflight, dev-loop docs): no public value
  changes** (verified: `git diff origin/master -- hazma rust` is empty —
  0 lines, and the tree was rebuilt from clean before anything was run).
  The diff is workflows, `preflight.sh`, durable docs, skills and project
  bookkeeping; no library module, kernel, signature, constant or build
  *input* is reachable from it, so no grid evaluation applies. The
  build's own inputs are untouched: `setup.py`, `pyproject.toml`,
  `MANIFEST.in`, `rust/` and every `.pyx` are byte-identical to the
  trunk, so the artifacts are too. Positive evidence rather than only
  absence: a wheel built from this branch carries `hazma/_core.abi3.so`
  inside a CPython-tagged wheel (`cp<XY>`, the interpreter that built
  it — never `abi3`, and see `lessons.md`
  `[wheel-tag-vs-extension-abi]`), which is Task 2.2's own
  extension-level criterion measured on the final tree.

- **Task 2.3, 2026-08-09 (cross-language plumbing test): no public value
  changes** (verified: `git diff origin/master -- hazma` is empty — 0
  lines, on a tree cleaned and rebuilt before anything was run). The diff
  is one new test module, one non-executable hunk in `rust/src/lib.rs`
  (`roundtrip`'s advertised `text_signature`, on the scaffold probe
  nothing under `hazma/` imports), and project bookkeeping; no library
  module, kernel, signature, constant or build *input* is reachable from
  it. Measured rather than only argued: the bare suite ran the parity
  corpus in **bit-equality mode** — `rtol = 0` across all 41 consumed
  entry points, 179,695 pinned values — and passed, at
  `1063 passed, 13 skipped` (+54 on Task 2.2's 1009, all of them the new
  module; the skip count is unchanged, which is what proves the mode).
  No ad-hoc grid sweep is reported because the corpus is a stricter grid
  than any of them. **Phase 02 therefore closes with the public compiled
  surface exactly where Phase 00 left it** — the whole phase's only
  change under `hazma/` across all three tasks is the non-executable
  `hazma/_core.pyi` stub.

- **Task 3.1, 2026-08-09 (constants module): no public value changes**
  (verified: `git diff origin/master -- hazma` is empty — 0 lines, on a
  tree cleaned and rebuilt before anything was run). The diff is one new
  Rust module that no Python imports and no Rust kernel calls, the
  `pub mod` line that admits it, one new test module, and project
  bookkeeping; no library module, kernel, signature, constant or build
  *input* under `hazma/` is reachable from it. Measured rather than only
  argued: the bare suite ran the parity corpus in **bit-equality mode** —
  `rtol = 0` across all 41 consumed entry points, 179,695 pinned values —
  and passed, at `1088 passed, 13 skipped` (+25 on Phase 02's 1063, all
  of them the new module; the skip count is unchanged, which is what
  proves the mode). What the task *did* produce is 224 constants that now
  exist in two places at once, and the argument that the second copy is
  bit-for-bit the first: 25 Python tests comparing source to source, five
  `cargo test` units, and a thirteen-mutation validity campaign. **Every
  Phase 04–06 drift line below this one is measured against Rust kernels
  reading these tables**, so a wrong value here would surface as a kernel
  bug rather than a constants bug — which is the whole reason the task
  refuses to trust its own transcription.

- **Task 3.2, 2026-08-09 (special functions): no public value changes**
  (verified: `git diff origin/master -- hazma` is one file,
  `hazma/_core.pyi`, and the hunk is a comment — no executable line under
  `hazma/` is reachable from this diff, on a tree rebuilt before anything
  was run). The rest is a new PyO3-free Rust module that no Python
  imports and no Rust kernel yet calls, its registration-only Python
  probe, two new test modules' worth of tests, and the parity corpus's
  served-kernel exemption. Measured rather than only argued: the bare
  suite ran the parity corpus in **bit-equality mode** — `rtol = 0`
  across all 41 consumed entry points, 179,695 pinned values — and
  passed, at `1154 passed, 13 skipped` (+66 on Task 3.1's 1088, all of
  them this task's tests; the skip count is unchanged, which is what
  proves the mode). **That evidence exists only because the task caught
  its own deliverable disabling the mode** — see the test-surface
  finding above; shipped unnoticed, every later Phase 03–06 line in this
  section would have been measured at 1e-8 instead of bit-equality.
  What the task *did* produce, numerically, is a Rust `spence`/`k1`/`kn`
  that tracks `scipy.special` to ≤ 4.0e-15 over every domain hazma
  reaches (per-sweep figures in the task note), against 5.1e-9 for the
  cephes `kn` it rejected. **Phase 04's muon photon kernel and Phase
  05's thermal ⟨σv⟩ are the first swaps whose drift lines will be
  measured against these**, so a wrong choice here would surface as a
  kernel bug rather than a specfun bug.

- **Task 3.3, 2026-08-10 (QUADPACK port): no public value changes**
  (verified: `git diff origin/master -- hazma` is one file,
  `hazma/_core.pyi`, and the hunk is a comment block — no executable line
  under `hazma/` is reachable from this diff, on a tree rebuilt before
  anything was run). The rest is a new PyO3-free Rust module that no
  Python imports and no Rust kernel yet calls, its registration-only
  Python probe, one new test module, and the parity corpus's served-kernel
  exemption. Measured rather than only argued: the bare suite ran the
  parity corpus in **bit-equality mode** — `rtol = 0` across all 41
  consumed entry points, 179,695 pinned values — and passed, at
  `1212 passed, 13 skipped` (+58 on Task 3.2's 1154, all of them this
  task's tests; the skip count is unchanged, which is what proves the
  mode). What the task *did* produce, numerically, is an integrator that
  reproduces `scipy.integrate.quad`'s subdivision on 4,456 of 4,461
  converged runs and its value to within 3.6e-2 of the requested
  tolerance. **Phase 04's spectra kernels and Phase 05's thermal ⟨σv⟩ are
  the first swaps whose drift lines will be measured against this**, so a
  wrong choice here would surface as a kernel bug rather than a
  quadrature bug — and the divergence regime (`limit` exhausted) is one
  no live call site enters today, asserted in
  `test/test_core_quad.py` rather than assumed.

- **Task 3.4, 2026-08-10 (interpolation + boost kernels): no public value
  changes** (verified: `git diff origin/master -- hazma` is one file,
  `hazma/_core.pyi`, and every line of the hunk is comment text — no
  executable line under `hazma/` is reachable from this diff, on a tree
  rebuilt before anything was run). The rest is two PyO3-free Rust
  modules that no Python imports and no Rust kernel yet calls, their
  registration-only probes, two new test modules, the parity corpus's
  served-kernel exemption, and one follow-up. Measured rather than only
  argued: the bare suite ran the parity corpus in **bit-equality mode** —
  `rtol = 0` across all 41 consumed entry points, 179,695 pinned values —
  and passed, at `1314 passed, 13 skipped` (+102 on Task 3.3's 1212, all
  of them this task's tests; skip count unchanged, and
  `tolerances.provenance` → `exact=True` checked directly rather than
  inferred).

  What the task *did* produce, numerically, is a foundation that
  reproduces the Cython **bit-for-bit** where the Cython is what the
  corpus records: zero mismatches on all seven live tables across six
  boost regimes × 400 energies, zero across 40,000 delta-function draws,
  and zero on the `np.interp` sweep — 20,304 abscissae for the 100-row
  eta table, 21,504 for the six 500-row tables (recorded as `20,204`
  until 2026-08-12; the sweep is `20,000 + 3n + 4`). **Phase 04's
  kaon/eta/omega/phi swaps are the first whose drift lines are measured
  against this.**

  **One drift is already known and lands with Phase 04, not here.** The
  Rust is bit-equal to the *contracted* (macOS/arm64) Cython on every
  platform, because `f64::mul_add` is fused unconditionally. On a target
  whose C compiler does not contract — baseline x86-64, which is what the
  Linux wheels are built for — today's Cython returns the unfused values,
  which differ from these by up to **3.6e-12** relative on the corpus
  grids. That is past rule 3's 1e-12 declaration threshold, so the Phase
  04 swap PR must state it. Nothing moves in this task, because nothing
  calls the new code. The alternative — plain arithmetic everywhere —
  was rejected because it misses the corpus by that same 3.6e-12 on
  *every* platform, which the 1e-12 `TABULATED` budget does not cover.

- **Task 3.5, 2026-08-11 (dispatch and error layer): no public value
  changes** (verified: `git diff origin/master -- hazma` is one file,
  `hazma/_core.pyi`, and every line of the hunk is comment text — no
  executable line under `hazma/` is reachable from this diff, on a tree
  rebuilt before anything was run). The rest is the PyO3 boundary module
  that no Python imports and no Rust kernel yet calls, its
  registration-only probe, one rewritten test module, the parity corpus's
  served-kernel exemption, and bookkeeping. Measured rather than only
  argued: the bare suite ran the parity corpus in **bit-equality mode** —
  `rtol = 0` across all 41 consumed entry points, 179,695 pinned values —
  and passed, at `1378 passed, 13 skipped` (+64 on Task 3.4's 1314, which
  is exactly `test/test_core_dispatch.py` growing from 54 tests to 118;
  skip count unchanged, and `tolerances.provenance` → `exact=True`
  checked directly).

  What the task *did* settle is a set of **user-visible behavior changes
  that land with Phases 04–06, not here** — no value moves, but the
  exception surface of 35 entry points does. Each is a widening or a
  type-only change and none can break a call that works today, and all of
  them belong in the Phase 07 CHANGELOG beside rule 9's assert
  tightening:

  - a 0-d array takes the scalar path everywhere (17 entry points raise
    `AssertionError` today — the 16 under `hazma/spectra/` plus
    `scalar_mediator_decay_spectrum`; the 18 cross sections already
    return a float);
  - a list or tuple is accepted everywhere (the 18 cross sections raise
    `AttributeError` today);
  - a rank error is a `ValueError` carrying the Cython assert's message
    **verbatim**, rather than an `AssertionError` that vanishes under
    `python -O`;
  - a dtype error keeps its `ValueError` but names the dtype, because the
    Cython has no single string to reproduce (`expected 'double'` in the
    spectra, `expected 'float64_t'` in the mediator modules);
  - `hazma/spectra/_neutrino/_muon.pyx:205`'s "Photon energies" becomes
    "Neutrino energies" (line number at `ed1fa20`; Task 4.6 deletes the
    file and ships that decision).

- **Task 4.1, 2026-08-11 (`dnde_positron_muon` → Rust — the first kernel
  swap): no public value changes.** The "before" is still in the tree:
  the pre-port Cython `cdef` `dnde_positron_muon_point`, reached through
  `_muon.pyx`'s `__pyx_capi__` now that its `def` is gone. Against it the
  Rust is **bit-for-bit identical** — `np.logspace(-2, 3, 200)` MeV at
  muon energies 150 / 500 / 1500 MeV (3 arrays, 600 values, max relative
  deviation 0.000e+00), and a wider 126,182-point sweep over 14 parent
  energies (rest, `+1e-16`, `+1e-9`, mildly and strongly boosted, `1e9`,
  below threshold, zero) on geometric, linear, random and
  edge-enumerated grids, **0 not bit-equal**. The corpus says it more
  strictly still: `spectra.positron.muon`'s declared budget is
  `EXACT_RTOL = 0.0`, so **the swap was gated at `rtol = 0` against its
  pre-port pins** — the gate did not weaken for the entry point being
  swapped. `git diff origin/master -- hazma` is four files, none of them
  another kernel.

  **What did change is the gate's mode, permanently.** From this swap
  `tolerances.provenance` reports `exact=False` and `effective_budget`
  returns the *declared* budget everywhere. Because the `EXACT` class's
  declared budget is itself `0.0`, **19 of the 41 cases lose nothing**;
  the other 22 loosen — `SPECFUN` (1) to 1e-13, `TABULATED` (7) to
  1e-12, `QUAD` (5) to 1e-8, `NESTED` (9) to 1e-6 — plus the abscissa
  comparison to 1e-13. All 41 still pass. **Two reasons are recorded in
  the skip message and only one is the swap:** the kernel digest also
  moved (`f5e6e269be47 -> fdbae2c19d87`), because removing a `def`
  changes the `.pyx` bytes the digest covers. So the flip was
  unavoidable in any task that touches a surviving `.pyx` at all. The
  tell in the suite is the skip count: **13 → 14**, and it stays there.

  Separately, and *not* a drift: this task **measured** that the shipped
  `dnde_positron_muon` is 0.0374% low against its own analytic
  normalization (see Findings). That is a pre-existing 2.1.0 defect the
  port reproduces, so no value moved — but it is the first entry the
  Phase 07 CHANGELOG will want to mention as a *known* wrong number
  rather than a changed one.

- **Task 4.2, 2026-08-12 (the seven tabulated photon spectra): no public
  value changes**, measured twice and from opposite directions.
  - _Against the Cython being replaced, before it was deleted._ All seven
    entry points × six parent energies (`E = M`, `M(1+1e-12)`, `1.05 M`,
    `2 M`, `10 M`, `1000 M`) × 8,000 photon energies each, half
    log-spaced and half log-uniform random over `[1e-5 M, 100 E]`:
    **336,000 points, 0 bitwise mismatches, max relative deviation
    0.000e+00**. This is the only form of against-the-Cython evidence
    this family gets, because unlike Task 4.1's capi survivor the five
    `.pyx` do not outlive the PR — so it was taken *before* the deletion
    and is recorded here rather than in a standing test.
  - _Against `origin/master` at the public API._ 665aed5 built in a
    scratch worktree with the same pinned environment, and the same
    script run on both: 12 `dnde_photon_*` × 4 parent energies, 2
    `dnde_positron_*` and 2 `dnde_neutrino_*` × 3 each, plus both models'
    `spectra()`, `positron_spectra()`, `annihilation_cross_sections()`
    and `thermal_cross_section()` — **97 arrays / 18,694 values,
    bit-for-bit identical**.
  - _One declared behavior change, at `NaN` inputs only._ The seven entry
    points with a `NaN` **photon** energy and a parent in flight returned
    `IndexError` and now return `NaN`; with a `NaN` **parent** energy
    they raised `AssertionError` and now raise `ValueError` (rule 9's
    tightening, which the port declares once). No finite input moves, and
    the corpus samples no `NaN` abscissa. **Belongs in the Phase 07
    CHANGELOG's behavior-change list, not its numerical one.**
  - Separately, and *not* a drift: this task **measured** two more
    pre-existing 2.1.0 defects (the η′ line's missing factor of two, the
    φ lines' daughter-meson energies). Reproduced, so no value moved —
    but like Task 4.1's normalization finding they are entries the
    Phase 07 CHANGELOG will want to mention as *known wrong* numbers
    rather than changed ones.

- **Task 4.3, 2026-08-16 (`dnde_photon_muon` → Rust, the only
  `spence`-bearing kernel): no public value changes** — but the first
  swap that had to *earn* that, and it moved a Phase 03 deliverable to do
  it.
  - _Against the Cython being replaced._ The pre-port `cdef`
    `dnde_photon_muon_point` is still in the tree behind
    `hazma/spectra/_photon/_muon.pyx`'s `__pyx_capi__` (capi survivor),
    and the Rust is
    **bit-for-bit identical** to it over 144,000 points: nine parent
    energies (`m_μ`, `m_μ(1+1e-12)`, `m_μ+1e-9`, 110, 150, 500, 1500,
    `1e5`, `1e9` MeV) × two 8,000-point grids each, one geometric and one
    uniform random. **0 mismatching doubles.** All five corpus blocks
    likewise show a difference of exactly zero, so the `SPECFUN` budget
    (1e-13) went unused.
  - _The first build was not bit-equal, and the reason is worth the
    space._ It differed at 11,306 of 70,000 points, max **3.15e-11**
    relative, concentrated at `E_μ = m_μ(1+1e-12)` — and **every one of
    the 24 failing corpus points was reproduced to a ratio of 1.000** by
    `(5/β)·Δspence·α/(3π E_μ)` alone. The kernel forms
    `(5/β)·(spence(x₋) − spence(x₊))` at `β = 1.4142764231806604e-06`, so
    `1/β ≈ 3.5e6` amplifies `spec_math`'s ≤2.0e-15 disagreement with
    `scipy.special.spence` by six orders of magnitude. The absolute size
    never exceeded **1.15e-14** on a block whose peak is 17.2.
  - _Fixed at the source, not at the budget._ `rust/src/special.rs` now
    transcribes cephes `spence` in-tree with the FP contraction scipy's C
    build uses (fused `polevl` Horner, fused
    `π²/6 − ln(x)·ln(1−x)`, fused `−0.5·z·z − y`), instead of calling
    `spec_math::Polylog::li2`. Same algorithm, same coefficients, fewer
    roundings — **0 mismatches against `scipy.special.spence` at 13,000
    points across all four branches**, where `spec_math` had 2289 of
    8,000 in the `(0,1)` arm alone. `SPECFUN` stayed at 1e-13; no budget
    was widened, so **rule 2 was not invoked**.
  - _`spence`'s only consumer inside hazma is this kernel_
    (`rg spence hazma/ rust/src` outside `special*.rs` returns
    `_photon/_muon.pyx:113`), so nothing else could have moved with it;
    `test/test_core_special.py`'s sweeps confirm the transcription tracks
    scipy at least as closely as `spec_math` did on every branch.
  - _No new behavior change._ The 0-d-array and rank-error divergences are
    the dispatch contract's, already declared for Task 4.1.
  - Separately, and *not* a drift: this task **measured** a fifth
    pre-existing 2.1.0 defect — `hazma/spectra/_photon/_muon.pyx:41`
    cuts the muon-rest-frame
    photon spectrum at `y = 1 − √r` where the kinematic endpoint (and the
    file's own in-flight branch, and
    `hazma/spectra/_photon/_pion.pyx`'s `ENG_GAM_MAX_MURF`) is
    `y = 1 − r`, leaving a hard zero over the top **0.2543 MeV** of the
    support where the spectrum is `5.34e-7 MeV⁻¹`, and a
    **discontinuity in `E_μ` at rest**. Reproduced, so no value moved —
    another entry the Phase 07 CHANGELOG will want under *known wrong*
    rather than *changed*.
- **Task 4.4 (`_photon/_pion` → Rust): one entry point bit-equal, the
  other moved by 2.6e-15 — below rule 3's declaration threshold.**
  - _`dnde_photon_neutral_pion`: no change at all._ Bit-equal to the
    Cython at all **1,305** corpus values and at **9,000** independently
    sampled points across nine parent energies, 0 mismatches. It is closed
    form, and reproducing the `.pyx`'s two `cdef float` truncations is
    what makes that possible — an all-`f64` spelling lands 8.5e-9 away.
  - _`dnde_photon_charged_pion`: ≤**2.618e-15** relative._ 317 of the
    1,500 pinned values are not bit-equal (worst block `boosted_mild` at
    2.618e-15; `rest` 3.540e-16, `rest_plus_eps` 2.981e-16, `near_rest`
    6.735e-16, `boosted_strong` 3.434e-16), and an independent 8,000-point
    sweep at eight parent energies gives 1,374 differences with a worst of
    6.499e-15. **Intended, and the reason the `QUAD` class exists**: the
    entry point moves from scipy's QUADPACK to the in-tree port, a
    different implementation of the same algorithm, so bit-equality was
    never available. Below 1e-12, so recorded rather than declared; the
    budget was *tightened* on this measurement, 1e-8 → 1e-12.
  - _No other public value moved._ The remaining 39 corpus cases are green
    at their own budgets and `test/test_theory_aggregation.py` is
    `69 passed` either side of the swap.
  - Separately, and *not* a drift: this task **measured** a sixth
    pre-existing 2.1.0 defect — the charged pion's `qagp` over `cos θ`
    returns exactly `0.0` in the narrow forward cone it never samples,
    making the spectrum a hard zero over the top ~25% of its support at
    `γ_π = 10` (0.041% of the yield there, 2.96% at `γ_π = 36`).
    Reproduced, so no value moved — another entry for the Phase 07
    CHANGELOG under *known wrong* rather than *changed*.

- **Task 4.5 (`_photon/_rho` → Rust): both entry points moved, both far
  inside budget, and the budget was *tightened*.** Measured against the
  live Cython twin before deleting it.
  - `hazma.spectra.dnde_photon_charged_rho`: worst **1.5e-13** relative
    over the 1,395 values the parity corpus pins (1,070 of them
    bit-equal), and worst **2.5e-11** over a denser 3,200-point
    off-corpus sweep at eight parent energies. The outlier is at
    `E_ρ = 5 m_ρ`, `E_γ = 431.99` MeV, where the π⁰ box's upper edge —
    a jump discontinuity — sits strictly inside the boost window and a
    single QUADPACK bisection decision can flip; scipy's own `abserr`
    there is 8.8e-08 on a value of 1.18e-02, so the two implementations
    differ **five decades below the error either admits to**. Beyond
    rule 3's 1e-12 threshold, hence declared here and in the PR body.
  - `hazma.spectra.dnde_photon_neutral_rho`: worst **3.2e-15**
    (corpus-pinned) and **4.9e-13** (off-corpus).
  - `test/parity/tolerances.py` gained `PORTED_NESTED_RTOL = 1e-9` and
    both cases moved to it from `NESTED_RTOL = 1e-6` — a **tightening**,
    the second per-case one in the project after Task 4.4's
    `PORTED_QUAD_RTOL`.
  - The project's declared numerical stress test therefore came in five
    decades inside its own budget. The nesting damps rather than
    amplifies, because the outer integral averages the inner one over a
    window.
  - **One reproduced defect, new to this task:** the `E_ρ − m_ρ <
    DBL_EPSILON` branch returns the boost *integrand*, so it is short by
    exactly a factor of `E_γ` (MeV⁻² where the spectrum is MeV⁻¹). The
    guard is absolute and one ulp at 775.26 MeV is 500x `DBL_EPSILON`, so
    it fires at `E_ρ == m_ρ` and at no other double. Reproduced, so no
    value moved — a Phase 07 CHANGELOG entry under *known wrong*, filed as
    [`rho-rest-frame-branch-returns-the-integrand.md`](../../../docs/followups/todo/rho-rest-frame-branch-returns-the-integrand.md).
  - **And a correction to Task 4.4's entry above:** the charged pion's
    lost forward cone does **not** merely propagate to the ρ, it deepens.
    A pure boost preserves the fraction of the endpoint at which the
    cliff sits (`γ(1−β)·γ(1+β) = 1`), so the inner 0.945 should hold at
    every ρ energy; measured, the charged ρ runs 0.9963 at `γ_ρ = 1.05`
    down to **0.5366** at `γ_ρ = 10` (neutral: 0.9420 → 0.5073). Also
    reproduced, also *known wrong* rather than *changed*, but the repair
    needs a restricted outer interval as well as the inner fix.

- **Parity-corpus stability follow-up (2026-08-18): no public value
  moves.** Nothing under `hazma/` changed; the diff is `test/parity/`,
  CI, and docs. What changed is what the corpus *asserts*: 494 of its
  179,695 pinned values (0.27%, all in four scalar cross-section cases)
  stop being compared, because they are cancellation residue rather than
  numbers any implementation reproduces. Verified by the corpus itself —
  `pytest test/parity -q` is `637 passed, 1 skipped` on the capturing
  platform with the `EXACT` class still at `rtol = 0` there.
  **This is a debt made visible, not created**: those four entry points
  were already returning wrong numbers, and Phase 07's CHANGELOG should
  say so under *known wrong* alongside the rho and pion entries above.
  The repair is
  [`scalar-elastic-cross-sections-cancel-in-atan-difference.md`](../../../docs/followups/todo/scalar-elastic-cross-sections-cancel-in-atan-difference.md),
  which moves published numbers and is therefore its own declared change.

- **Task 4.6 (`_positron/_pion` + the neutrino pair — closes Phase 04):
  three entry points moved, all three inside their declared budgets, and
  no other value moved.** Measured with the shipped wrappers against the
  stored corpus arrays on the capturing platform:
  - `spectra.positron.charged_pion` — **5.494e-15** worst relative over
    1,460 pinned values, 1,304 of them bit-equal; budget **tightened**
    `QUAD_RTOL` (1e-8) → `PORTED_QUAD_RTOL` (1e-12).
  - `spectra.neutrino.muon` — **bit-equal at all 3,795 pinned values**,
    and at all 9,600 points of a denser off-corpus sweep. Stays at
    `EXACT_RTOL` (0), achieved rather than assumed.
  - `spectra.neutrino.charged_pion` — **9.739e-16** worst relative over
    4,185 pinned values, 3,793 of them bit-equal; budget **tightened** to
    `PORTED_QUAD_RTOL` the same way.

  Off-corpus, where the corpus does not reach: 3.5e-13 for the positron
  pion (8 pion energies to 1e5 MeV), 2.3e-14 for the neutrino pion,
  bit-equal for the neutrino muon. The four already-ported kernels the
  diff can reach were re-measured as a control and are **unchanged** from
  their own tasks' figures (`spectra.photon.charged_pion` 2.618e-15,
  `spectra.photon.charged_rho` 1.511e-13, `spectra.positron.muon` and
  `spectra.photon.muon` bit-equal).
  - **One unpinned value repaired**, not drifted:
    `dnde_neutrino_charged_pion(0.0, epi)` returned `NaN` from the first
    version of the port and returns `(0, 0, 0)` now, which is what the
    Cython returns — `scipy.integrate.quad` short-circuits `a == b`
    without calling the integrand and `crate::quad` did not. The corpus
    grids start at `1e-5 m_π`, so no pinned value is involved.
  - **A seventh *known wrong* entry for the Phase 07 CHANGELOG:** the
    charged pion's `π → e ν` **neutrino** line is added twice
    (`_pion.pyx` sums two `cdef`s and both carry it; the muon row has no
    second copy). Measured by continuum subtraction at exactly 2.0000
    copies. The electron-neutrino yield is overweight by 1.23e-4 per pion
    (0.0123% integrated, 0.062% locally on the plateau at
    `E_π = 200` MeV). Reproduced, so no value moved; filed as
    [`neutrino-pion-electron-line-counted-twice.md`](../../../docs/followups/todo/neutrino-pion-electron-line-counted-twice.md).
  - **And a fact worth not "fixing":** `_neutrino/_muon.pyx` applies the
    Michel normalization the **right** way round — both its rows
    integrate to exactly one neutrino — while its `_positron/_muon.pyx`
    sibling divides and is 0.0374% low (Task 4.1's defect). The two files
    really do disagree and only one of them is wrong.

- **Task 5.1 (vector cross sections): five of six entry points
  bit-equal; one moves by 2.06e-14.** The five closed forms —
  `sigma_xx_to_v_to_{ff,pipi,pi0g,pi0v}` and `sigma_xx_to_vv` —
  reproduce the Cython **exactly** at all 5,811 values the corpus
  compares them on (5,814 stored, less 3 positions that stand in for a
  pinned raise), on the capturing platform, at `rtol = 0`. That was not
  free: two of them compiled through `double _Complex` (see Findings),
  and a real-arithmetic transliteration would have missed by up to
  9.0e-15.
  `cross_sections.vector.thermal_cross_section` moves by at most
  **2.0597e-14** relative over its 285 pinned values (64 bit-equal),
  worst at `open_resonance`, `x = 0.298`
  (`9.316997739611058e-08 → 9.316997739610866e-08`). The drift is the
  Bessel prefactor and weight rather than the integrator —
  `bessel_kn(2, ·)` agrees with scipy to 8.9e-16 and the prefactor
  squares it. Below rule 3's 1e-12 threshold, so no CHANGELOG line of
  its own. **Budget tightened, not widened:** that case goes from
  `QUAD_RTOL` (1e-8) to `PORTED_QUAD_RTOL` (1e-12), 49x headroom.
  - **A user-visible *string* change, and it is the only one:** the
    `TypeError` both complex channels raise at `e_cm = 2 m_x` keeps its
    type and loses Cython's wording, which advised
    `use 'cython.cpow(True)'` — a compiler directive that will not exist
    after Phase 07. The corpus records only the type.
  - **An eighth and ninth *known wrong* entry for the Phase 07
    CHANGELOG,** both filed and neither introduced here: two of the six
    channels **raise** at the annihilation threshold while the other
    four return `inf` or `nan`
    ([the `2 m_x` raise](../../../docs/followups/todo/vector-cross-sections-raise-at-the-two-mx-threshold.md)),
    and `thermal_cross_section` returns its integrator's *initial
    estimate* — 0.5%–5% off the true integral for every `x` above about
    5, i.e. across the whole freeze-out region
    ([the unconverged quadrature](../../../docs/followups/todo/thermal-cross-section-quadrature-never-converges.md)).
    Both are reproduced under rule 1, so no value moves; the second is
    the more consequential, because relic abundance goes as 1/⟨σv⟩.

- **Task 5.2 (scalar cross sections): eleven of twelve entry points
  bit-equal; one moves by 3.12e-15.** The eleven closed forms —
  `sigma_xx_to_s_to_{ff,gg,pi0pi0,pipi}`, `sigma_xx_to_ss`,
  `sigma_ss_to_xx`, `sigma_x{l,pi,pi0,g,s}_to_x{l,pi,pi0,g,s}` —
  reproduce the Cython **exactly** at all 12,155 values the corpus
  compares them on, on the capturing platform, at `rtol = 0`. Measured
  directly against the live Cython before deletion, not only through the
  corpus. That was not free either: one of them
  (`sigma_xx_to_s_to_ff`) compiled through `double _Complex` — which the
  phase's own handoff had said this module did not do — and a
  real-arithmetic transliteration misses bit-equality at 355 of 935
  points on its electron block alone.
  `cross_sections.scalar.thermal_cross_section` moves by at most
  **3.1240e-15** relative over its 285 pinned values (104 bit-equal),
  worst at `open_resonance`, `x = 0.116895`
  (`5.560975522996041e-09 → 5.560975522996024e-09`). Same cause as its
  vector twin: the Bessel prefactor and weight, not the integrator.
  Below rule 3's 1e-12 threshold, so no CHANGELOG line of its own.
  **Budget tightened, not widened:** that case goes from `QUAD_RTOL`
  (1e-8) to `PORTED_QUAD_RTOL` (1e-12), 320x headroom. It was the last
  case at the opening figure, so **`QUAD_RTOL` now has no holder**.
  - **No user-visible string change.** The scalar module's one complex
    expression can only fail on a vanishing denominator, which no corpus
    point reaches; the `TypeError` message exists but nothing pins it.
  - **`np.log(4)` → `LN_4` moves nothing.** `math.log(4.0)` is
    `1.3862943611198906`, the double NumPy returned.
  - **A tenth *known wrong* entry for the Phase 07 CHANGELOG,** filed
    before this task and carried into Rust unchanged: four elastic
    scattering kernels compute an `atan` difference that cancels away
    every significant bit near `e_cm = 2 m_x` and for a small width,
    giving the wrong sign and a fabricated pole
    ([the cancellation](../../../docs/followups/todo/scalar-elastic-cross-sections-cancel-in-atan-difference.md)).
    Reproduced under rule 1: rule 2 forbids regenerating the corpus a
    stabilised kernel would have to pass, and `PLAN.md`'s Scope excludes
    physics changes. The follow-up is updated with what closed and what
    the standalone fix now costs.

- **Task 5.3 (thermal ⟨σv⟩ validation sweep): no public value changes
  in the semi-analytic relic density; the ODE path moves at its own
  solver tolerance.** `relic_density` is the live consumer of both
  ported `thermal_cross_section`s and was measured end-to-end against
  the pre-port tree (`14f1c66`) on six model points — the same
  `open/narrow/closed_resonance` scalar and vector configurations
  `test/parity/cases.py` uses. `relic_density(semi_analytic=True)`:
  worst **4.11e-16** relative, four of six **bit-equal**. That path is a
  closed-form composition, so it carries Tasks 5.1/5.2's ≤2.06e-14
  kernel drift through undamped and no further. Below rule 3's 1e-12
  threshold — no CHANGELOG line of its own.
  `relic_density(semi_analytic=False)`: worst **3.82e-5** relative
  (`vector.narrow_resonance`), which is **not** drift. `solve_ivp` runs
  at the caller's default `rtol=1e-5`, a last-bit input change flips a
  step-acceptance decision, and the whole step sequence differs.
  Tightening only the solver collapses it — 3.82e-5 → 2.75e-7
  (`rtol=1e-8`) → 3.84e-9 (`rtol=1e-10`) — while the pre-port answer
  itself moves 1.3e-5 between those tolerances. The physics is unchanged
  to ~1e-9; what the default reports is the solver's own error.
  Pinned going forward by `test/test_relic_density.py`'s
  `TestMediatorRelicDensity` at `rtol` 1e-12 (semi-analytic) and 1e-5
  (Boltzmann). The Boltzmann pins are taken with the *solver* at
  `rtol=1e-10` rather than `relic_density`'s default, because a pin at
  the default is not platform-portable: this task's first version pinned
  it at 1e-4 and all five Linux CI jobs failed at **1.222e-4**
  (`vector.open_resonance`) while macOS passed — a different libm
  perturbs the step sequence differently. At `rtol=1e-10` the same
  comparison is 1.93e-8.
  - **`thermal_cross_section` itself**, on this sweep's 13-point
    `x = mx/T` grid per model (78 values spanning 0.1–500, both sides of
    the scalar kernel's `x = 300` cutoff): worst **1.2799e-15**, 37
    bit-equal. Consistent with the corpus-grid figures; this grid does
    not land on their worst point.
  - **Debug and release `hazma._core` are bit-identical** over all 90
    values, so the cargo profile is a speed choice only.
  - **The eleventh *known wrong* entry is now quantified at the
    consumer.** The unconverged thermal quadrature is 0.5%–5% wrong on
    ⟨σv⟩ across freeze-out, and relic abundance goes as 1/⟨σv⟩, so every
    relic density Hazma has shipped inherits that error roughly
    linearly
    ([the unconverged quadrature](../../../docs/followups/todo/thermal-cross-section-quadrature-never-converges.md)).
    Reproduced under rule 1, not fixed here.

(Per-function drift lines land here as Phase 04–06 swaps merge; the
Phase 07 CHANGELOG is assembled from this section — do not reconstruct
it from memory.)
