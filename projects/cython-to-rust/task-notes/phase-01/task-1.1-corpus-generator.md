# Task 1.1: Corpus specification and generator

**Date:** 2026-08-07
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-01-parity-corpus.md` (Task 1.1);
`../../PLAN.md` (Scope, Numerical impact); `../../rules.md` rules 1–3
**Related ADRs:** none
**Depends On:** Phase 00 complete

## Objective

Stand up `test/parity/generate.py` and the reference data it produces:
pinned arrays for all 41 consumed compiled entry points, captured from
the pre-port Cython, plus a manifest that records the provenance
(git SHA, environment, per-array hashes) and a `--check` mode that
re-verifies the stored data against it.

## Exit Criteria

Copied from `../../phases/phase-01-parity-corpus.md`, Task 1.1:

- `test/parity/generate.py` produces `test/parity/data/*.npz` covering
  every **consumed** entry point in
  `../../references/cython-inventory.md` ("Entry points by module" —
  41 functions; the two unimported `sigma_xx_to_all` are excluded, with
  the exclusion asserted by an import re-check in the generator).
- Grids are log-spaced and bracket thresholds and kinematic endpoints
  (`E → m/2`, table edges).
- ≥4 parent energies per spectrum (rest frame + ε, mildly and strongly
  boosted).
- For the mediators, ≥3 model-parameter points including a
  near-resonance configuration.
- Thermal ⟨σv⟩ over an x grid spanning freeze-out.
- A manifest (JSON) records generator git SHA, package versions, and
  per-array hashes.
- Total data size ≤ ~10 MB.
- Grids deliberately include the known NaN/negative-prone kinematic
  edges; captured values are stored as-is (edge behavior is part of the
  contract).

Plus the project rule this task is the first to be bound by:

- rules.md rule 2 — the corpus is generated only from pre-port Cython;
  the generator must refuse to run against a tree in which any kernel
  already runs on Rust.

## Inputs Reviewed

- `../../PLAN.md` (Scope, Numerical impact, Phases);
  `../README.md` (all sections — the `two_body_momentum` out-of-band
  repair is what the corpus must now pin);
  `../../phases/phase-01-parity-corpus.md`; `phase-01/README.md`.
- `../../rules.md` — parity discipline (rules 1–3), constants bit-parity.
- `../../references/cython-inventory.md` — entry-point table, live
  surface, data files, bug list.
- `../../references/numerics-replacements.md` — quad call-site table
  (tolerances and QAGP breakpoint degeneracies), `np.interp` semantics,
  boost-integral spec, entry-point dispatch contract.
- `docs/agents/lessons.md`, `docs/agents/environment.md`,
  `docs/agents/doc-consistency.md`.
- Source read directly: every surviving `.pyx` public `def` signature,
  the nine Python wrapper import sites, and
  `hazma/spectra/_photon/data/*.csv`.

## Findings

- **Two entry points *raise* at a kinematic edge, and nothing recorded
  it.** `sigma_xx_to_v_to_pipi` and `sigma_xx_to_v_to_pi0v` raise
  `TypeError` — Cython's "cannot convert complex with non-zero imaginary
  component to double", from a `**0.5` of a quantity that rounds
  negative — at **exactly** `e_cm = 2·mx` (observed at `e_cm = 400.0`
  with `mx = 200`, and `e_cm = 600.0` with `mx = 300`). It is not in the
  inventory's bug list. The corpus pins it: the stored value is `nan`
  and the manifest's per-block `raises` entry records the index, the
  argument and the exception type. The scalar-mediator siblings do
  **not** raise there.
- **The two `thermal_cross_section` implementations disagree with each
  other above `x = 300`.** The scalar returns `0.0` outright
  (`hazma/scalar_mediator/_c_scalar_mediator_cross_sections.pyx:1401-1402`);
  the vector clips `xnew = min(x, 300)` and keeps returning the value
  there
  (`hazma/vector_mediator/_c_vector_mediator_cross_sections.pyx:649`).
  Verified in the captured data: over the 10 grid points above 300, the
  scalar block is identically zero while the vector's first point above
  the cutoff is `1.230610e-09`. A Phase 05 port that unifies the two
  would move published numbers; `x = 300` is now an explicit grid anchor
  in both.
- **A batched call cannot be trusted to degrade gracefully.** One
  raising grid point takes the whole array down, so the generator falls
  back to a per-point sweep (still through the *batched* entry point, one
  length-1 array at a time) to recover the surviving values. Any later
  tool that evaluates the corpus in bulk needs the same fallback.
- **The array path and the `*_pt` path agree bit-for-bit** for both
  mediator positron modules and the vector photon module — checked with
  `np.array_equal(..., equal_nan=True)` across every stored array. They
  are still pinned separately: they are separate entry points and a port
  can break one alone.
- **Negative and infinite values are widespread at the edges, and are
  now contract.** `spectra.photon.muon` carries 26 negative values,
  `cross_sections.scalar.sigma_xl_to_xl` 123 negatives and 5 infinities,
  `spectra.photon.neutral_pion` one infinity. All pre-existing; the
  corpus stores them as returned rather than filtering.
- **`generate.py --check` needs no built tree.** Verified by inspecting
  `sys.modules` after a `--check` run: only `hazma` and
  `hazma.parameters` are imported, neither compiled. This is load-bearing
  and fragile — it is why `cases.py` imports `HiggsPortal` /
  `KineticMixing` inside the factory functions rather than at module
  scope (marked with `# noqa: PLC0415`). Moving those imports up would
  make the integrity check require a full build.
- **The corpus does not enter the sdist.** `MANIFEST.in`'s
  `global-include` lists `*.txt *.rst *.pyx *.pxd *.c *.md`, and neither
  `.npz` nor `.json` is on it. Measured on the same tree: 400 files
  without this change, 402 with — the two additions are the
  `test/parity/` directory entry and `README.md`, 2.5 KiB. The 2.9 MiB
  of data stays out.
- **Some grid anchors legitimately fall outside the base log range**,
  and clipping them would have dropped the point worth sampling. The
  tabulated photon spectra start their energy tables around `M / 1e6`
  (the charged-kaon table's first energy is `4.936770e-04` MeV), five
  decades below the base grid's lower end of `1e-5 · M`, and that table
  edge is exactly where the kernel switches between its `1/E` tail and
  the interpolant. `log_grid` therefore keeps every anchor and lets the
  grid extend past the base range; 59 anchors across the tabulated and
  strongly-boosted blocks do so.
- **A worktree can inherit `.so` files whose source package no longer
  exists.** This tree carried `hazma/_gamma_ray/*.so` and
  `hazma/_phase_space/*.so` — 25 extensions where `setup.py` declares 20
  — left over from before Task 0.2 deleted those packages. They import
  fine and would have made a "20 extensions" claim look wrong. Extends
  the existing `environment.md` entry on stale generated `.c`; the same
  clean-first recipe fixes it.

## Decisions and Implementation Notes

- **Specification and generator are separate modules.** `cases.py`
  declares what is pinned and how each entry point is invoked; it
  contains no numbers and asserts nothing. `generate.py` evaluates it and
  owns I/O. Task 1.2's runner imports `cases` to re-evaluate the same
  calls against whatever is live, so the call convention lives in exactly
  one place.
- **No `__init__.py` under `test/parity/`.** Matches the existing
  sibling-import idiom (`test/spectra/test_dnde_photon_fsr.py:40` does a
  bare `import msqrd_corpus`); adding one would change how pytest imports
  the directory.
- **`kernel_digest`, not the git SHA, is the real provenance record.**
  rules.md rule 2 wants the SHA, and the manifest carries it — but the
  generating commit does not exist yet when the data is written, so
  `dirty` is `true` by construction and the SHA alone identifies nothing.
  The manifest therefore also hashes every `.pyx`, `.pxd` and photon CSV
  in the tree (44 files). Here `git diff origin/master -- hazma` is
  empty, so digest `f5e6e269be47` corresponds exactly to the kernels at
  `origin/master` (`f025448`) — and still does after the round-1
  regeneration, whose manifest records SHA `010747c`. A changed SHA with
  an unchanged digest is exactly the signal the two-field design is meant
  to give.
- **Rule 2 is enforced in code, not only in prose.** `assert_no_rust_core`
  refuses to generate once `hazma._core` is importable.
- **Coverage is derived, not asserted.** `assert_full_coverage` walks the
  surviving `.pyx` for top-level `def`s and fails both ways — a `def`
  with no case, and a case naming a `def` that no longer exists. It is
  what keeps the corpus honest as Phases 04–06 delete Cython modules.
  Likewise `assert_unconsumed_exports_are_unimported` re-derives at
  generation time that nothing imports either `sigma_xx_to_all`, rather
  than trusting the inventory snapshot.
- **Five parent energies per spectrum, not the four required.** The extra
  one is `M·(1 + 1e-12)`, which straddles the
  `E − M < DBL_EPSILON` rest-frame short-circuit — a branch every ported
  kernel has to reproduce (rules.md, Rust convention 4).
- **Anchors are offset by ±1e-9 and ±1e-6, not just sampled.** A port
  that moves a branch boundary by a representable amount is caught rather
  than stepped over.
- **Table edges are read from the CSVs at generation time**, not
  transcribed, so they cannot drift from the shipped data. The two
  hand-transcribed literals (`ENG_GAM_MAX_MURF`, `ENG_GAM_MAX_PIRG`) are
  cited to their source lines.
- **Near-resonance is a `stheta`/`eps` of 1e-4**, which narrows
  `width_s`/`width_v` so the s-channel pole is a spike, with grid anchors
  at `m_med` and `m_med ± width`. The other two model points put the pole
  well above threshold (broad) and below threshold (unreachable).
- **Mediator *spectrum* points use `mx = m_med`.** With `m_med < 2·mx`
  the `→ x x` channel is closed, so the visible partial widths carry the
  whole total and the pinned spectra are not identically zero. Resonance
  is meaningless for these entry points, so the three points vary the
  mass across the `2m_μ` / `2m_π` / `2m_π0` thresholds instead.
- **The 4 duplicated `*_pt` cases are kept.** They cost ~1.2 MiB of the
  2.9 MiB total and duplicate their array-path siblings bit-for-bit, but
  they are distinct entry points in the 41 and a port can break one
  alone.
- **Round-1 review: imports are now pinned to the repository tree.**
  `kernel_digest` walks `REPO_ROOT`, but `importlib.import_module` follows
  `sys.path`, so a site-packages install could have supplied the values
  while the manifest described the checkout. `Case.resolve` now calls
  `assert_module_is_repo_tree` on every module it loads, `generate()`
  calls `hazma_package_path()` before anything else, and the manifest
  records `hazma_package`. Putting the check inside `resolve()` rather
  than only in `generate()` means Task 1.2's runner inherits it. The
  regeneration that followed left **every per-case hash unchanged**, so
  the guard is a provenance fix, not a numerical one.
- **Exception messages are deliberately not recorded** in the `raises`
  entries — Cython rewords its errors between releases. The type and the
  argument are what a port must reproduce.

## Files Changed

All new; no existing file under `hazma/` or `test/` was modified.

- `test/parity/cases.py` — corpus specification: 41 `Case`s / 623
  `Block`s, grid construction, mediator model points, and the three
  guards (`assert_no_rust_core`,
  `assert_unconsumed_exports_are_unimported`, `assert_full_coverage`).
- `test/parity/generate.py` — CLI. Default regenerates `data/`;
  `--check` re-hashes it against the manifest without importing a kernel.
- `test/parity/README.md` — what the corpus is, the two commands, and
  when *not* to regenerate.
- `test/parity/data/*.npz` (41 files) + `data/manifest.json` — the
  reference arrays and their provenance.
- `projects/cython-to-rust/phases/phase-01-parity-corpus.md` — Task 1.2
  exit criteria gained the raise-replay bullet (see Plan Impact).
- `projects/cython-to-rust/task-notes/phase-01/README.md`,
  `projects/cython-to-rust/task-notes/README.md` — status bookkeeping.
- `docs/agents/lessons.md` — one new class
  ([measured-tree-vs-imported-module]) and PR #50 added to
  [sibling-copies-of-a-fixed-claim]'s citations (round-1 review).

## Verification

Regenerated from the command outputs below on the final tree.

### Corpus generation and integrity

```text
$ .venv/bin/python test/parity/generate.py
wrote 41 cases / 623 blocks / 2937.3 KiB to test/parity/data

$ .venv/bin/python test/parity/generate.py --check
corpus OK: 41 cases / 1580 arrays match the manifest (generated at
010747c6125d, kernel digest f5e6e269be47)
```

The recorded SHA is `010747c` (this task's own commit) with
`dirty: true`, because the round-1 review fixes regenerated the corpus
after that commit landed. The *kernel* digest is unchanged at
`f5e6e269be47`: neither commit touches `hazma/`, so the captured values
come from exactly the same kernels as before. That is precisely the
property `kernel_digest` exists to make checkable, and why the SHA alone
was never the provenance record.

**Determinism** — two independent full runs on the same tree produced a
byte-identical manifest (`diff manifest-run1.json
test/parity/data/manifest.json` → no output). A third run after the lint
refactor reproduced every hash (`diff` of both manifests with the `git`
and `environment` blocks removed → no output). A fourth, after the
round-1 repo-tree guard, reproduced **every per-case hash**: `diff` of
the manifests' `cases` blocks before and after → no output, i.e. the
guard moved no captured value.

**Guards fire when they should** (each provoked deliberately, then
reverted):

| Guard | Provocation | Result |
| --- | --- | --- |
| `--check` array hash | perturbed one stored value by 1e-7 relative | `FAILED … sha256 'c33c282d…' != manifest '8e1dbdf2…'`, exit 1 |
| `--check` file presence | removed `spectra.photon.phi.npz` | `FAILED … missing spectra.photon.phi.npz`, exit 1 |
| `assert_full_coverage` | dropped a case | `public defs with no corpus case: [('hazma.spectra._photon._phi', 'dnde_photon_phi')]` |
| `assert_full_coverage` | added a case naming a nonexistent `def` | `corpus cases with no public def: [('hazma.spectra._photon._phi', 'no_such_def')]` |
| `assert_unconsumed_exports_are_unimported` | marked a *consumed* export as unconsumed | raised, naming the importers |
| `assert_no_rust_core` | stubbed `find_spec("hazma._core")` | `hazma._core is importable: this tree runs Rust kernels…` |
| `assert_module_is_repo_tree` | module `__file__` under `/usr/lib/.../site-packages` | `hazma.fake resolves to …, outside the repository at …` |
| `assert_module_is_repo_tree` | module with `__file__ = None` (namespace pkg) | `hazma has no __file__, so it cannot be shown to come from …` |
| `Case.resolve` | shadowed one entry point's module with an out-of-tree `.so` | `hazma.spectra._photon._muon resolves to /opt/site-packages/…, outside the repository at …` |

**Specification and data round-trip.** Rebuilding the specification and
comparing against the stored arrays: *623 blocks checked, 0 mismatches*
on both `block.grid` (`np.array_equal`) and `block.label` vs the
manifest. This is the property Task 1.2's runner depends on — it
re-evaluates `cases.py` and compares against `data/`, so a spec that
drifted from the data would produce false failures.

**`--check` imports no compiled kernel** — after a `--check` run,
`sorted(m for m in sys.modules if m.startswith('hazma'))` is
`['hazma', 'hazma.parameters']` and the list of hazma modules whose
`__file__` ends in `.so` is empty.

**Existing suites unchanged** (both match the Phase 00 closing baseline
recorded in `../README.md`):

```text
$ .venv/bin/python -m pytest -q
57 passed, 10 skipped in 0.32s

$ .venv/bin/python -m pytest -q test
244 passed, 20 skipped in 244.26s (0:04:04)
```

No new pytest tests: `test/parity/test_parity.py` is Task 1.2's
deliverable and wiring `--check` into the gate is Task 1.3's. Until
those land, the corpus is verified by the commands above and nothing
runs it in CI — that is the plan's sequencing, not an oversight.

### Lint

```text
$ .venv/bin/black --check test/parity/ && .venv/bin/isort --check test/parity/
All done! ✨ 🍰 ✨
2 files would be left unchanged.
Skipped 1 files

$ .venv/bin/ruff check test/parity/
All checks passed!

$ .venv/bin/ruff check --isolated --select E9,F63,F7,F82 test/parity/
All checks passed!
```

Both the configured ruff and the `--isolated` form CI runs are clean on
the new files — a zero delta against the trunk, which carries thousands
of configured-ruff findings.

### Coverage arithmetic

| Group | Entry points | Cases |
| --- | --- | --- |
| Spectra (photon 12, positron 2, neutrino 2) | 16 | 16 |
| Scalar-mediator cross sections (13 public, 12 consumed) | 12 | 12 |
| Vector-mediator cross sections (7 public, 6 consumed) | 6 | 6 |
| Mediator spectra | 7 | 7 |
| **Total** | **41** | **41** |

`assert_full_coverage` re-derives this from the tree on every
generation, so the table is checked rather than asserted.

## Open Questions

- **Task 1.4 inherits a scope question the corpus now answers in part.**
  The 90 `.npy` reference arrays the two skipped mediator classes read
  overlap the cross-section and mediator-spectrum cases pinned here.
  Whether that makes them redundant (delete) or complementary
  (regenerate) is Task 1.4's call; the corpus gives it a concrete
  comparison target.
- **The `x > 300` divergence between the two `thermal_cross_section`
  implementations is pinned, not resolved.** Phase 05 must either
  reproduce both behaviors or declare the unification as a numerical
  change. No follow-up filed: it is inside this project's scope and
  named in the Phase 05 handoff below rather than deferred out of it.
- The two raising entry points (`sigma_xx_to_v_to_pipi`,
  `sigma_xx_to_v_to_pi0v`) are the same shape: a `TypeError` where a
  domain guard belongs. Fixing them is a **behavior change** and out of
  scope here — Phase 05 ports them as-is per rules.md rule 1, and any
  repair is a separate declared change.

## Plan Impact

**Impact Level:** Update phase file.

`../../phases/phase-01-parity-corpus.md`, Task 1.2 exit criteria, gained
one bullet: the runner must **replay** the manifest's `raises` records —
asserting the live implementation raises the same exception type at the
same argument — rather than comparing the stored `nan`. Without it a
runner would pass against an implementation that silently returned a
number at `e_cm = 2·mx`, which is precisely the regression the corpus
exists to catch. This is a canonical-contract patch made in the same
task that discovered the need, not deferred.

Round-1 review added three more edits to the same file, all
corrections of claims that were already stale when this task inherited
them and that became this PR's responsibility the moment it touched the
file: the frontmatter `status:` moved `Not started` -> `In Progress`;
Task 1.3's parenthetical test count was re-derived (`51 passed / 20
skipped` -> `244 passed / 20 skipped` as of 2026-08-07) and dated; the
stale `collect_ignored` clause was dropped (`test/conftest.py` has
listed only `setup.py` since Task 0.2); and Task 1.4's "159 reference
arrays" was re-derived to **90** -- 159 was a collision with Task 0.2's
unrelated 159-array impact check.

Nothing else moved: no ADR, no `rules.md` change, no `PLAN.md` change
(the phase table's Phase 01 row describes the phase, not its tasks). The
corpus's coverage count (41) matches what `PLAN.md` and the inventory
already state, verified by `assert_full_coverage` rather than by
re-reading them.

## Stale-state sweep

Run against `claude/cython-to-rust/task-1.1-corpus-generator`.

### Identifier sweep

New public names introduced by this task, swept with
`rg -n '<identifier>' projects/cython-to-rust/ docs/ README.md hazma/ test/`
(folded to one row per file; the citing docs match their own commands):

| Identifier | Occurrences | Disposition |
| --- | --- | --- |
| `assert_full_coverage` | `test/parity/cases.py`, `test/parity/generate.py`, `test/parity/README.md`, this note | KEPT — all describe the same function |
| `assert_unconsumed_exports_are_unimported` | `test/parity/cases.py`, `test/parity/generate.py`, `test/parity/README.md`, this note | KEPT |
| `assert_no_rust_core` | `test/parity/cases.py`, `test/parity/generate.py`, this note | KEPT |
| `kernel_digest` | `test/parity/generate.py`, `test/parity/README.md`, this note | KEPT |
| `build_cases` | `test/parity/cases.py`, `test/parity/generate.py` | KEPT |
| `test/parity` | phase file, `phase-01/README.md`, `../README.md`, this note, `test/parity/*` | KEPT |
| `hazma._core` | `test/parity/cases.py`, `test/parity/README.md`, `../../PLAN.md`, `../README.md`, phase files 02–07, this note | KEPT — pre-existing project vocabulary, unchanged |

No identifier was removed or renamed by this task, so there is no stale
side to sweep.

### Line-number citation sweep

`--changed-vs` diffs *committed* history, so on this uncommitted tree it
scans zero files and prints a success-shaped line
(`docs/agents/lessons.md`, `[changed-vs-sees-only-commits]`). Run with
explicit paths instead:

```text
$ scripts/agents/check_doc_citations.py \
    projects/cython-to-rust/task-notes/phase-01/task-1.1-corpus-generator.md \
    projects/cython-to-rust/phases/phase-01-parity-corpus.md \
    test/parity/README.md \
    projects/cython-to-rust/task-notes/phase-01/README.md \
    projects/cython-to-rust/task-notes/README.md
docs scanned: 5
in-repo citations checked: 6
  resolved by exact: 6
external citations skipped: 0
out-of-range or ambiguous: NONE
```

The checker bounds-checks citations in *markdown*; every
`file:line` citation embedded in `test/parity/cases.py` was additionally
re-read against the tree with `sed -n '<line>p'` and corrected — four
were wrong on first writing (`_pion.pyx` 18-19 → 16-18; `_eta.pyx` 37-43
→ 38-44; the scalar thermal bound cited as "the 1370 region" → the real
`:1412`; `_scalar_mediator_cross_sections.py` 41-42 → 42-43;
`_vector_mediator_cross_sections.py` 55-57 → 54). Because Phases 04–06
delete every file these point into, the module docstring now pins them:
"read against the pre-port tree at commit `f025448`".

### Forward-looking phrase sweep

```text
$ rg -n '(Task [0-9]+ will|will be added|still pending|today: ?stub|In Progress)' \
    projects/cython-to-rust/ hazma/ | sort
```

17 hits, zero under `hazma/`. Folded by kind, all KEPT:

| Kind | Where | Why kept |
| --- | --- | --- |
| Live project status | `PLAN.md:2`, `task-notes/README.md:5`, `:25`, `:506`, `phase-01/README.md:6` | The project *is* in progress and Phase 01 *is* in progress; four of these this task wrote |
| Template placeholders | `phases/_template.md:4`, `task-notes/_template.md:5` | Enumerations of allowed values, not claims |
| Dated Phase 00 records | `task-0.1:414,429`, `task-0.4:228`, `task-0.5:277,279,375` | Historical notes describing sweeps already done |
| This note's own sweep | `task-1.1:382,384,385,389` | The command matches the doc that cites it |

No `Task N will…`, `will be added`, `still pending` or `today: stub`
claim was introduced anywhere.

### Count sweep

| Claim location | Command | Actual | Status |
| --- | --- | --- | --- |
| "41 cases / 623 blocks" (this note, Verification) | `python test/parity/generate.py` | `41 cases / 623 blocks` | OK |
| "1580 arrays" | `python test/parity/generate.py --check` | `1580 arrays` | OK |
| "2.9 MiB", "≤ ~10 MB budget" | `du -sh test/parity/data`; byte sum over `data/` | `2.9M` (3,007,843 B) | OK |
| "5 parent energies per spectrum" | `len(manifest[…]['spectra.photon.muon']['blocks'])` | `5` | OK |
| "3 model points per cross section" | `len(manifest[…]['cross_sections.scalar.sigma_xx_to_ss']['blocks'])` | `3` | OK |
| "16 + 12 + 6 + 7 = 41 entry points" | `assert_full_coverage` on the live tree | passes | OK |
| "44 files in the kernel digest" | `manifest['kernel_digest']['n_files']` | `44` | OK |
| "3 blocks carry `raises` records" | `sum(1 for c in cases for b in c['blocks'] if 'raises' in b)` | `3` | OK |
| "Cython 3.2.9 built the extensions" | `head -1 hazma/spectra/_photon/_muon.c` | `/* Generated by Cython 3.2.9 */` | OK |
| "sdist 400 → 402 files" | `uv build --sdist; tar tzf dist/*.tar.gz \| wc -l`, both halves on this tree | `400` stashed / `402` applied, re-measured on the final tree | OK |
| "179,695 pinned values" | sum of every block's `values` array size | `179695` | OK |
| "623 blocks / 0 mismatches" round-trip | rebuild `cases.build_cases()`, compare grids and labels to `data/` | `623 blocks checked, 0 mismatches` | OK |
| "59 anchors outside the base range" | replay `boosted_edges` against each block's `[lo, hi]` | `59` | OK |
| "25 extensions where setup.py declares 20" | `find hazma -name '*.so' \| wc -l` before cleaning | `25`, of which 5 under deleted packages | OK |
| "244 passed, 20 skipped" matches the Phase 00 baseline | `pytest -q test` | `244 passed, 20 skipped` | OK |
| "57 passed, 10 skipped" matches the Phase 00 baseline | `pytest -q` | `57 passed, 10 skipped` | OK |

### Numerical-impact statement

**No public value changes (verified: `git diff origin/master -- hazma`
is empty).** This task adds only new files under `test/parity/` and
`projects/`, plus one bullet in a phase file. No library module,
signature, constant, or build input is touched, so no grid evaluation
applies — the diff cannot reach a public code path. Confirmed
independently by both suites reproducing the Phase 00 closing counts
exactly.

The corpus itself *records* two pre-existing behaviors that later phases
must not silently change, logged in `../README.md` under "Numerical
impact so far" as observations rather than drifts: the `TypeError` at
`e_cm = 2·mx` in two vector cross sections, and the `x > 300` divergence
between the two `thermal_cross_section` implementations.

### Exit Criteria → artifact mapping

| Exit criterion | Artifact / evidence |
| --- | --- |
| `generate.py` produces `data/*.npz` covering all 41 consumed entry points | 41 `.npz`; `assert_full_coverage` derives coverage from the tree |
| `sigma_xx_to_all` exclusion asserted by an import re-check | `assert_unconsumed_exports_are_unimported`; negative-tested |
| Log-spaced grids bracketing thresholds and endpoints (`E → m/2`, table edges) | `log_grid` + `boosted_edges`; `M/2` anchored for every parent; table ends read from the shipped CSVs |
| ≥4 parent energies per spectrum | 5 per spectrum (`parent_energies`), incl. the `DBL_EPSILON` short-circuit boundary |
| ≥3 mediator parameter points incl. near-resonance | `_scalar_model_points` / `_vector_model_points`; `narrow_resonance` uses `stheta`/`eps` = 1e-4 with anchors at `m_med ± width` |
| Thermal ⟨σv⟩ over an x grid spanning freeze-out | `_thermal_blocks`: `x ∈ [0.1, 1000]`, 95 points, anchored at 20, 0.5, 1/3, 1, 300, `m_med/mx`, `2m_med/mx` |
| Manifest records git SHA, package versions, per-array hashes | `manifest.json` `git` / `environment` / per-block `arrays[*].sha256`; plus `kernel_digest` |
| Total ≤ ~10 MB | 3,007,843 B = 2.9 MiB; `MAX_TOTAL_BYTES` fails generation above 10 MiB |
| NaN/negative-prone edges included, values stored as-is | 26 negatives in `spectra.photon.muon`, 123 negatives + 5 infinities in `sigma_xl_to_xl`, `raises` records for the two `TypeError` points |
| rules.md rule 2 enforced | `assert_no_rust_core`; negative-tested |

### Task-note self-consistency

`**Status:** Complete` matches every Exit Criterion having an artifact
row above. Every file named in §Files Changed appears in
`git status --short` / `git diff --stat origin/master --`; every
function named in §Decisions and §Findings
(`assert_no_rust_core`, `assert_full_coverage`,
`assert_unconsumed_exports_are_unimported`, `build_cases`,
`log_grid`, `boosted_edges`, `parent_energies`, `_thermal_blocks`,
`_scalar_model_points`, `_vector_model_points`) exists in
`test/parity/cases.py`.

## Handoff to Next Task

**Read first:** `test/parity/README.md`, then `test/parity/cases.py`'s
module docstring (grid design), then this note's Findings.

**Currently safe to assume:**

- The corpus is complete and reproducible: `python
  test/parity/generate.py --check` verifies it in under a second without
  a built tree, and two full regenerations produced byte-identical
  manifests.
- Coverage is self-checking. Task 1.2 does not need to re-derive the
  41; `assert_full_coverage` fails generation if the set ever drifts.
- The corpus pins the **post-fix** `two_body_momentum` values (the
  out-of-band repair landed before Phase 01 — see `../README.md`), so
  the Rust port must reproduce those, not the pre-fix ones.
- `cases.py` is the single source of the call convention. Task 1.2's
  runner should import it and re-evaluate `block.array_call` /
  `block.scalar_call` against the live implementation rather than
  re-deriving argument tuples.
- The manifest's per-block `params` is enough to reconstruct any call
  without reading `cases.py` — useful for a debugging one-liner.

**Currently risky / unknown:**

- **Task 1.2 must replay the `raises` records**, now an exit criterion
  in the phase file. Three blocks carry them (`sigma_xx_to_v_to_pipi`
  at `narrow_resonance` and `closed_resonance`, `sigma_xx_to_v_to_pi0v`
  at `closed_resonance`); a runner that only compares the stored `nan`
  passes vacuously there.
- **Do not move `cases.py`'s deferred model imports to module scope.**
  They are what keeps `--check` free of a build; ruff is silenced there
  with a reason.
- Regeneration takes ~4.5 minutes, almost all of it the nested
  quadrature in the rho and mediator-spectrum kernels. It is a
  developer command, not a test — Task 1.3 wires `--check` (fast) into
  the gate, not `generate`.
- The corpus was captured on macOS/arm64 with numpy 2.5.1 / scipy
  1.18.0 / Cython 3.2.9 (all recorded in the manifest; the Cython version
  is read off the generated `.c` header, because the build backend's
  isolated environment means `importlib.metadata` cannot see it).
  Whether every stored value is bit-reproducible on the Linux CI matrix
  is **unverified** —
  Task 1.2 will find out when it sets tolerances, and that is the right
  place for the answer, since it is exactly what the budget per function
  has to absorb.
  _(Task 1.2, 2026-08-07: it did not — no Linux runner was available.
  What it did instead is make the question harmless: the runner demands
  bit-equality only when the manifest's platform, toolchain and kernel
  digest all match, and enforces the declared budgets otherwise, so a
  Linux runner cannot fail an exactness claim nobody has evidence for.
  The measurement itself moves to Task 1.3, which wires CI.)_
