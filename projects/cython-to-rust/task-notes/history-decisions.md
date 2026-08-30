# Archived working memory: Decisions and Implementation Notes (Phases 00–04)

**Project:** cython-to-rust
**Moved:** 2026-08-21, from [`README.md`](README.md)
**Source lines:** 1492–1658 of that file at commit `c57ce4f`

This file is a verbatim archive. Nothing below the rule was edited,
summarised or reordered when it moved, and it sits in the same
directory as the README so every relative link in the moved text
still resolves. Reproduce the move with

```sh
git show c57ce4f:projects/cython-to-rust/task-notes/README.md | sed -n '1492,1658p'
```

The phase learnings under [`../learnings/`](../learnings/)
condense this material and are what a new task reads first — see
[ADR-0002](../../../docs/adrs/ADR-0002-read-phase-learnings-not-closed-task-notes.md).
Come here when a learnings entry, a task note or a citation sends
you to the original entry. Later phase-close sweeps append the
closed phase's entries below, verbatim, under a
`### Swept YYYY-MM-DD (Phase XX)` heading.

---

## Decisions and Implementation Notes

- **The corpus may decline to pin a value, and says so in a file rather
  than in a widened tolerance** (parity-corpus follow-up, 2026-08-18).
  `test/parity/stability.py` names 494 stored positions whose values are
  cancellation residue; `test_parity` deletes them from the comparison.
  The alternative — a per-point tolerance, which is what the follow-up
  proposed — would have needed `rtol = 1e+24` at the worst point, which
  asserts nothing while looking like a budget. Membership is established
  against `test/parity/reference.py` at a threshold (`1e-9`) placed at
  the minimum of a measured bimodal histogram, and the total is pinned by
  a test so that changing it has to show up in a diff. No ADR: it is a
  testing contract inside `test/parity/`, and `../rules.md` rules 1–3
  already govern it.
- **The `EXACT` class distinguishes a changed host from a changed
  implementation** (same follow-up). `PLATFORM_EXACT_RTOL = 1e-6` applies
  only when `Provenance.same_platform` is false; a port on the capturing
  platform is still held to bit-equality, which Tasks 4.1–4.5 all
  achieved. `same_platform` is keyed on OS **family** and CPU
  architecture, not `platform.platform()` — the capturing machine's macOS
  point release had already moved, and the fine-grained comparison
  silently dropped the gate on its own host.
- **The per-kernel swap recipe is canonical, and lives in the phase
  file** (Task 4.1): map the FMAs out of the shipped `.so` first, port
  into `rust/src/kernels/<pyx name>.rs`, register through
  `dispatch::map_unary` with the twin's quantity wording, repoint the
  wrapper, **repoint the corpus case and add a `PORTED_ENTRY_POINTS`
  row**, delete the twin (the whole `.pyx`, or just its `def` for a capi
  survivor), add a `test/test_core_<kernel>.py`, record the drift. Eight
  steps in
  [`../phases/phase-04-spectra-kernels.md`](../phases/phase-04-spectra-kernels.md)'s
  Goal rather than an ADR, because it is procedure rather than
  architecture.
- **A capi survivor loses its Python `def`, not its file** (Task 4.1).
  The phase Goal already said the four survivors would be
  "Python-unreferenced"; deleting the `def` while keeping the `cdef`s
  makes that literally true — the `__pyx_capi__` capsules the mediator
  modules cimport are built from the `cdef`s — and is as close to rule 1
  as the phase's own declared exception allows.
- **Per-kernel test modules no longer copy `test/test_core_dispatch.py`,
  reversing Task 2.3's instruction** (Task 4.1). That instruction was
  written when the plan was one dispatch implementation per kernel;
  Task 3.5 replaced it with three shared helpers, so those 118 tests now
  cover code every kernel routes through unchanged and copying them
  sixteen times would re-test one function sixteen times.
  `test/test_core_positron_muon.py` is the shape to copy: one assertion
  per contract branch with *this* kernel's wording, the twin as a
  two-mode oracle (bit-for-bit on the capturing platform, a peak-scaled
  budget elsewhere), and physics that outlives the Cython. 47 tests
  rather than ~160. **The two-mode half of that is not universal**: a
  quadrature-backed kernel has no bit-equality mode on any platform,
  because the port replaces scipy's QUADPACK with the in-tree one, so
  `test/test_core_{photon,positron}_pion.py` carry one measured budget
  and no platform branch. The twin's *fate* picks the module shape; the
  kernel's *numerics* pick the comparison.
- Rust + PyO3 + maturin over pybind11; single abi3 `hazma._core`;
  setuptools-rust coexistence during migration → ADR-0001 (Accepted).
- No GSL-derived (GPL-3) code in tree or dependency graph; cephes
  lineage (`spec_math`) for specfun; netlib-QUADPACK translation for
  the integrator; cyphus crates as out-of-repo oracles only →
  ADR-0002 (Accepted 2026-08-04 — Hazma stays MIT).
  **Implemented in Task 3.2** as `rust/src/special.rs` over
  `spec_math` 0.1.6 (MIT OR Apache-2.0), with one deliberate departure
  the ADR does not disturb: `bessel_kn` is an upward recurrence on
  cephes `k0`/`k1` seeds rather than cephes' own `kn`, because scipy's
  `kn` is `kv` and the faithful routine misses it by 5.1e-9. That is
  original work over cephes seeds, so nothing GSL-derived enters the
  graph — the ADR's fallback (vendor the cephes routine) would have
  reproduced the miss instead of fixing it.
- Capi-survivor exception: four spectra Cython extensions outlive their
  Phase 04 swap until Phase 06 Task 6.4, because the mediator spectrum
  `.pyx` cimport their `__pyx_capi__` symbols — recorded in the
  Phase 04 file's Goal block.
- Constants bit-parity before consolidation → `../rules.md` rule 4.
  **Implemented in Task 3.1** as three namespaces mapped to sources:
  `constants::pdg` ← `_utils/constants.pxd` (151), `constants::legacy` ←
  `_utils/legacy_parameters.pxd` (48), `constants::derived::<source_pyx>`
  ← the module-local `DEF`s of the five `.pyx` that declare any (25).
  Every module-local `DEF` is carried, aliases included, so the coverage
  check can rescan the tree rather than trust a transcribed list; the
  module is `pub` in `lib.rs` where its neighbours are private, because
  224 unread `const`s in a private module is a wall of `dead_code`.
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
- **Task 2.1 (2026-08-08)** patched two canonical documents in the same
  PR as the code: the phase-02 file's Task 2.1 exit criteria gained a
  "the parity gate still runs in bit-equality mode" bullet (the task was
  widened past the plan's wording, so the plan now says so), and
  `../references/numerics-replacements.md`'s dispatch-contract section
  gained the measured live Cython behavior it was silent about. No ADR:
  nothing revises ADR-0001, and this task is its first executable form.
- **Task 3.2 (2026-08-09)** patched two canonical documents in the same
  PR as the code, for the same reason. The phase-03 file's Task 3.2
  block gained three "criteria added during execution" bullets: the
  `kn` deviation and its measurement (the first criterion's fallback
  clause prescribed a cephes translation that would not have worked),
  the bound on where the underflow criterion can hold for `kn` (above
  `x ≈ 698` scipy returns `0`, so the stated rtol is unachievable there
  and always will be), and keeping the corpus's served-kernel predicate
  sound. `../references/numerics-replacements.md` gained the measured
  block, because its own sentence "scipy's `spence`, `k1`, `kn` are
  themselves cephes wrappers" is what made the wrong choice look safe.
  No ADR: nothing revises ADR-0002, no interface or ordering moves, and
  the decision is a per-function implementation choice carried by the
  code, the phase file and the task note.
- **Task 3.3 (2026-08-10)** patched two canonical documents in the same
  PR as the code, for the same reason as Task 3.2. The phase-03 file's
  Task 3.3 block gained four "criteria added during execution" bullets:
  the break-point contract is scipy's rather than QUADPACK's and both
  live degeneracies are discards; only `qk21` is on the live path, so
  `qk15` is a cross-check rather than production code; the agreement
  criterion is met with two orders of headroom and its boundary is
  `limit`; and two adaptive-loop heuristics needed purpose-built inputs.
  `../references/numerics-replacements.md` gained the measured contract,
  because its own sentence — "must be pinned *empirically*" — is the
  instruction, and leaving the answer in a task note would make the next
  reader re-derive it. No ADR: nothing revises ADR-0002 (the provenance
  is exactly what it prescribes), and no interface or ordering moves.
- **Task 3.4 (2026-08-10)** patched the same two canonical documents, for
  the third time running and for the third distinct reason. The phase-03
  file's Task 3.4 block gained five "criteria added during execution"
  bullets: the oracle is the live Cython through `__pyx_capi__` rather
  than the Phase 01 micro-fixtures the criterion named (they do not exist
  and could not — the corpus sees only top-level `def`s); the port must
  reproduce the compiler's fused multiply-adds where they occur and not
  where they do not, held to bit-equality rather than a tolerance; the
  interior sum's dropped cell is reproduced and filed; `interp` carries
  NumPy's undocumented quirks as well as its contract; and the corpus's
  served-kernel predicate stays sound with two more test-only submodules.
  `../references/numerics-replacements.md` gained the measured block,
  because its own three-bullet `np.interp` contract and its boost
  paragraph are exactly what a next reader would port from, and both are
  incomplete in ways that change the numbers. No ADR: the provenance is
  original work plus NumPy's BSD-3-Clause behavior, which ADR-0002
  permits (its rule is that nothing GSL-derived enters the tree), and no
  interface or ordering outside Task 3.4 moves.
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

## Phase 07 (moved 2026-08-29 at project close)

- **The version's source of truth is `pyproject.toml`'s
  `[project] version`** (Task 7.1); `hazma.VERSION` and `__version__`
  survive as public API by reading it back from `importlib.metadata`. A
  build backend cannot import the package it has not built, and maturin
  stamps the distribution from that field. `preflight.sh --closing`,
  `docs/versioning.md`, `docs/workflow.md`,
  `docs/agents/{preflight,doc-consistency}.md` and all three project
  `PLAN.md` closing paragraphs were repointed in the same pass. No ADR:
  ADR-0001 already names maturin as the packaging decision.
