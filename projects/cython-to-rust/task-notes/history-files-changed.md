# Archived working memory: Files Changed (Phases 00–04)

**Project:** cython-to-rust
**Moved:** 2026-08-21, from [`README.md`](README.md)
**Source lines:** 1659–1953 of that file at commit `c57ce4f`

This file is a verbatim archive. Nothing below the rule was edited,
summarised or reordered when it moved, and it sits in the same
directory as the README so every relative link in the moved text
still resolves. Reproduce the move with

```sh
git show c57ce4f:projects/cython-to-rust/task-notes/README.md | sed -n '1659,1953p'
```

The phase learnings under [`../learnings/`](../learnings/)
condense this material and are what a new task reads first — see
[ADR-0002](../../../docs/adrs/ADR-0002-read-phase-learnings-not-closed-task-notes.md).
Come here when a learnings entry, a task note or a citation sends
you to the original entry. Later phase-close sweeps append the
closed phase's entries below, verbatim, under a
`### Swept YYYY-MM-DD (Phase XX)` heading.

---

## Files Changed

### Parity-corpus stability follow-up (2026-08-18)

- `test/parity/reference.py`, `test/parity/stability.py`,
  `test/parity/data/unpinnable.json` — new: the 60-digit oracle, the
  unpinnable-point registry, and the mask it produces.
- `test/parity/tolerances.py` — `PLATFORM_EXACT_RTOL`,
  `ZERO_FLOOR_FRACTION`, `zero_floor()`, `_libm_identity()`,
  `Provenance.same_platform`, the `effective_budget` platform branch.
- `test/parity/test_parity.py` — `_drop_unpinnable`, the split
  zero/non-zero comparison, six new guards.
- `test/parity/README.md`, `.github/workflows/ci.yml` (the `PARITY` env
  removed), `pyproject.toml` (`mpmath` in `dev`; the stale `addopts`
  comment), `test/test_core_{positron_muon,interp,boost}.py` (six
  sentences that claimed CI skips the corpus off-platform).
- `projects/cython-to-rust/phases/phase-01-parity-corpus.md`,
  `task-notes/phase-01/README.md`,
  `task-notes/phase-01/followup-parity-corpus-stability.md`.
- `docs/followups/` — the item moved to `done/`, a new `todo/` item for
  the underlying kernel defect, the index, and 20-odd inbound links
  repointed.

### Phase 04 (Task 4.6 — closes the phase)

- New: `rust/src/kernels/{positron_pion,neutrino_flavors,neutrino_muon,
  neutrino_pion}.rs`, `test/test_core_positron_pion.py`,
  `test/test_core_neutrino.py`,
  `docs/followups/todo/neutrino-pion-electron-line-counted-twice.md`,
  `projects/cython-to-rust/learnings/phase-04-spectra-kernels.md`.
- Deleted: `hazma/spectra/_neutrino/_{muon,pion,neutrino}.{pyx,pxd,pyi}`
  (8 files — nothing outside the package cimported them) and
  `hazma/spectra/_positron/_pion.pyi`. `_positron/_pion.pyx` keeps its
  `cdef`s: both mediator positron-spectrum modules cimport them.
- Changed: `rust/src/{kernels,positron,neutrino,quad,constants}.rs`,
  `hazma/spectra/_positron/{__init__.py,_pion.pyx}`,
  `hazma/spectra/_neutrino/__init__.py`, `setup.py`,
  `test/parity/{cases,tolerances}.py`,
  `test/parity/oracles/entry_points.py`,
  `test/test_core_{dispatch,constants}.py`, `docs/followups/README.md`,
  `../phases/phase-04-spectra-kernels.md`, `../PLAN.md`, and — correcting
  a claim about this task that was always wrong —
  `../../parity-pinned-defect-repair/{PLAN.md,references/defect-blast-radius.md}`.

### Phase 04 (Task 4.5)

- New: `rust/src/kernels/photon_rho.rs`, `test/test_core_photon_rho.py`,
  `docs/followups/todo/rho-rest-frame-branch-returns-the-integrand.md`.
- Deleted: `hazma/spectra/_photon/_rho.{pyx,pxd,pyi}` (whole module —
  nothing cimported it).
- Changed: `rust/src/{kernels,photon,constants,quad}.rs`,
  `rust/src/kernels/photon_pion.rs`, `hazma/spectra/_photon/__init__.py`,
  `setup.py`, `test/parity/{cases,tolerances}.py`,
  `test/test_core_{dispatch,constants,photon_pion,quad}.py`,
  `docs/followups/README.md`,
  `docs/followups/todo/charged-pion-photon-spectrum-misses-the-forward-cone.md`.

### Phase 04 (Task 4.4)

- New: `rust/src/kernels/photon_pion.rs`, `test/test_core_photon_pion.py`,
  and one follow-up
  (`charged-pion-photon-spectrum-misses-the-forward-cone.md`).
- Changed: `rust/src/{kernels,photon}.rs`,
  `hazma/spectra/_photon/{__init__.py,_pion.pyx}` (the two `def`s only),
  `hazma/_core.pyi`, `setup.py`, `test/parity/{cases,tolerances}.py`,
  `test/test_core_dispatch.py`, `docs/followups/README.md`.
- Deleted: `hazma/spectra/_photon/_pion.pyi`.
- Full list in [phase-04/README.md](phase-04/README.md).

### Phase 04 (Task 4.2)

- New: `rust/src/kernels/photon_tables.rs`,
  `test/test_core_photon_tables.py`, and two follow-ups
  (`eta-prime-two-photon-line-missing-factor-two.md`,
  `phi-photon-lines-use-the-daughter-meson-energy.md`).
- Changed: `rust/src/{kernels,photon,boost,interp}.rs`,
  `hazma/spectra/_photon/__init__.py`, `hazma/_core.pyi`, `setup.py`,
  `test/parity/{cases,tolerances,test_parity}.py`,
  `test/test_core_{boost,interp}.py`, `docs/followups/README.md` and two
  sibling follow-ups.
- Deleted: `hazma/spectra/_photon/{_eta,_eta_prime,_kaon,_omega,_phi}.{pyx,pxd,pyi}`
  and `hazma/spectra/_photon/path.py` — 16 files, 1,020 lines, of which
  204 were commented-out `quad`-based dead code.
- Full per-task list: [phase-04/README.md](phase-04/README.md).

### Phase 04 (Task 4.1)

- New: `rust/src/kernels/positron_muon.rs`,
  `test/test_core_positron_muon.py`,
  `docs/followups/todo/positron-muon-spectrum-normalization-inverted.md`.
- Changed: `rust/src/{kernels,positron}.rs`,
  `hazma/spectra/_positron/{__init__.py,_muon.pyx}`, `hazma/_core.pyi`,
  `test/parity/{cases,test_parity}.py`, `docs/followups/README.md`,
  `../phases/phase-04-spectra-kernels.md` (the swap recipe).
- Deleted: `hazma/spectra/_positron/_muon.pyi`.
- Full per-task list: [phase-04/README.md](phase-04/README.md).

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

### Phase 02

- **Task 2.1** — the `rust/` crate (`Cargo.toml` with an explicit empty
  `[workspace]`, `Cargo.lock`, `build.rs`, and
  `src/{lib,dispatch,kernels}.rs` plus the five per-domain registration
  modules); build wiring in `setup.py` (`RustExtension("hazma._core",
  …, py_limited_api=True)`), `pyproject.toml` (`setuptools-rust` in
  `[build-system] requires`), `MANIFEST.in` (the crate, which no
  `global-include` pattern reaches) and `.gitignore`; the new
  `hazma/_core.pyi` stub; the served-kernel predicate across
  `test/parity/{cases.py,tolerances.py,test_parity.py,README.md}`; and
  the two canonical doc patches above. 25 files: 12 modified, 13 added.
  Full list in [phase-02/README.md](phase-02/README.md).
- **Task 2.2** — the Rust toolchain pinned in both workflows
  (`.github/workflows/ci.yml` gains a `rust` job and a per-entry
  toolchain step; `.github/workflows/release.yml` gains the host
  toolchain for macOS, rustup-in-container for Linux, `hazma._core` in
  the wheel test command, and a step asserting every wheel carries
  `hazma/_core.abi3.so`); the three cargo gates in
  `scripts/agents/preflight.sh`; the `.rs` dev loop in `AGENTS.md`,
  `docs/agents/environment.md` and `docs/agents/preflight.md`; the
  rebuild-awareness sweep across `docs/agents/{review-lenses,README}.md`
  and seven skill files under `.claude/` and `.codex/`; two canonical
  patches (the phase file's Task 2.2 exit criteria, `../rules.md` Rust
  rule 1); and one struck-through risk bullet in Task 2.1's note.
  21 files: 20 modified, 1 added. Nothing under `hazma/` or `rust/`.
  Full list in [phase-02/README.md](phase-02/README.md).
- **Task 2.3** — phase closure. New `test/test_core_dispatch.py`
  (54 tests in six classes, the template every Phase 04–06 kernel swap
  copies); one non-executable hunk in `rust/src/lib.rs` (`roundtrip`'s
  `text_signature` `"(x, /)"` → `"(x)"` plus the doc comment recording
  why); one canonical patch (`../phases/phase-02-rust-scaffold.md`'s
  Task 2.3 exit criteria gained the signature bullet). Phase closed:
  learnings written, phase frontmatter `status: Complete`, `PLAN.md`
  Phases row updated. Nothing under `hazma/`. Full list in
  [phase-02/README.md](phase-02/README.md).

### Phase 03

- **Task 3.1** — new `rust/src/constants.rs` (224 `pub const`s in
  `pdg` / `legacy` / `derived::*`, a `# Sources` provenance header, and
  five unit tests); `pub mod constants;` plus its rationale paragraph in
  `rust/src/lib.rs`; new `test/test_core_constants.py` (25 tests). No
  canonical patch — the phase file's three Task 3.1 criteria are
  satisfied as written. Nothing under `hazma/`. Full list in
  [phase-03/README.md](phase-03/README.md).
- **Task 3.2** — new `rust/src/special.rs` (`spence` / `bessel_k1` /
  `bessel_kn` over `spec_math`, a `# Sources and licensing` provenance
  header, 9 unit tests) and `rust/src/special_probe.rs`
  (registration-only `hazma._core.special`); `rust/src/lib.rs` admits
  both; `rust/Cargo.toml` / `Cargo.lock` gain `spec_math = "0.1.6"`. New
  `test/test_core_special.py` (65 tests).
  `test/parity/{cases.py,test_parity.py,README.md}` gain
  `_CORE_TEST_ONLY_MODULES` and its importer guard test.
  `hazma/_core.pyi` gains a comment — the only change under `hazma/`,
  and non-executable. **Two canonical patches:** the phase file's Task
  3.2 block gained three "criteria added during execution" bullets, and
  [`../references/numerics-replacements.md`](../references/numerics-replacements.md)
  gained the measured block correcting its claim that scipy's `kn` is a
  cephes wrapper. Full list in
  [phase-03/README.md](phase-03/README.md).

- **Task 3.3** — new `rust/src/quad.rs` (`qk15` / `qk21` / `qelg` /
  `qpsrt` / `qagse` / `qagpe` plus the scipy-shaped `quad` driver and
  `filter_points`, a `# Sources and licensing` provenance header, and 24
  unit tests) and `rust/src/quad_probe.rs` (registration-only
  `hazma._core.quad`, taking a Python callable so scipy and the port see
  the same integrand); `rust/src/lib.rs` admits both. New
  `test/test_core_quad.py` (58 tests in 8 classes).
  `test/parity/{cases.py,test_parity.py,README.md}` gain the
  `hazma._core.quad` exemption. `hazma/_core.pyi` gains a comment — the
  only change under `hazma/`, and non-executable. **Two canonical
  patches:** the phase file's Task 3.3 block gained four "criteria added
  during execution" bullets, and
  [`../references/numerics-replacements.md`](../references/numerics-replacements.md)
  gained the measured break-point contract. Full list in
  [phase-03/README.md](phase-03/README.md).
- **Task 3.4** — new `rust/src/interp.rs` (`np.interp` with NumPy's full
  contract, a `# Sources and licensing` header and 11 unit tests) and
  `rust/src/boost.rs` (`boost_beta` / `boost_gamma` /
  `boost_delta_function` / `boost_integrate_linear_interp` plus
  `trapezoid` / `pairwise_sum` and `BoostError`, with the contracted-site
  rationale, the `# Faithfulness notes` on the four preserved defects, and
  13 unit tests); `rust/src/{interp_probe,boost_probe}.rs` register the two
  test-only submodules and `rust/src/lib.rs` admits all four. New
  `test/test_core_interp.py` and `test/test_core_boost.py` (the latter
  carries the `__pyx_capi__` ctypes oracle).
  `test/parity/{cases.py,test_parity.py,README.md}` gain the
  `hazma._core.{interp,boost}` exemptions. `hazma/_core.pyi` gains a
  comment — the only change under `hazma/`, and non-executable. One
  follow-up filed
  ([the boost integral's window coverage](../../../docs/followups/todo/boost-integral-drops-last-interior-cell.md)).
  **Two canonical patches:** the phase file's Task 3.4 block gained five
  "criteria added during execution" bullets, and
  [`../references/numerics-replacements.md`](../references/numerics-replacements.md)
  gained the measured fused-arithmetic block. Full list in
  [phase-03/README.md](phase-03/README.md).
- **Task 3.5** — `rust/src/dispatch.rs` rewritten around a shared
  `classify` and grown by `map_flavors` (the neutrino 3-tuple / `(3, N)`
  shape) and `require_vector` (`partial_widths`); new
  `rust/src/dispatch_probe.rs` (registration-only `hazma._core.dispatch`,
  three probes taking the quantity wording); `roundtrip_flavors` and two
  units in `rust/src/kernels.rs`; `rust/src/lib.rs` admits the probe.
  `test/test_core_dispatch.py` grew 54 → 118 tests; the parity corpus's
  `_CORE_TEST_ONLY_MODULES` and its three prose sites gained
  `hazma._core.dispatch`;
  [`../../../docs/followups/todo/model-spectra-reject-scalar-energies.md`](../../../docs/followups/todo/model-spectra-reject-scalar-energies.md)
  records its compiled half as decided. **Two canonical patches:** the
  phase file's Task 3.5 block gained five "criteria added during
  execution" bullets, and
  [`../references/numerics-replacements.md`](../references/numerics-replacements.md)
  gained a "settled contract" section superseding its design sketch.
  **Phase closed:** learnings written, phase frontmatter
  `status: Complete`, `PLAN.md` Phases row updated. Across all five tasks
  Phase 03 changed exactly one file under `hazma/` — the non-executable
  `hazma/_core.pyi`, comments only. Full list in
  [phase-03/README.md](phase-03/README.md).
