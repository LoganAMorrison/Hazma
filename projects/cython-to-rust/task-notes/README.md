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
| 02 | Rust scaffold | [phase-02-rust-scaffold.md](../phases/phase-02-rust-scaffold.md) | [phase-02/README.md](phase-02/README.md) | **Complete (2026-08-09)** — all three tasks done; [learnings](../learnings/phase-02-rust-scaffold.md) |
| 03 | Numerics foundation | [phase-03-numerics-foundation.md](../phases/phase-03-numerics-foundation.md) | [phase-03/README.md](phase-03/README.md) | **Complete (2026-08-11)** — all five tasks done; [learnings](../learnings/phase-03-numerics-foundation.md) |
| 04 | Spectra kernels | [phase-04-spectra-kernels.md](../phases/phase-04-spectra-kernels.md) | [phase-04/README.md](phase-04/README.md) | In Progress — Tasks 4.1 (2026-08-11), 4.2 (2026-08-12), 4.3 (2026-08-16) and 4.4 (2026-08-17) done; 4.5–4.6 open |
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
  `.py`. Scope `--paths` to changed files. **The advice that used to
  close this bullet — "omit it entirely when the diff has no Python" —
  was wrong, and Task 2.1 tripped over it.**
  `scripts/agents/preflight.sh:90` is
  `[[ -n "${PATHS}" ]] || PATHS="hazma test"`, so omitting `--paths`
  selects the maximal *directory* form, which is the second trap in the
  same sentence: a three-file markdown diff came back `FAIL isort` /
  `FAIL ruff` over 98 files and 6,187 findings, none of them in the diff
  ([`../../../docs/followups/todo/preflight-isort-ruff-red-on-trunk.md`](../../../docs/followups/todo/preflight-isort-ruff-red-on-trunk.md)).
  On a docs-only diff, scope `--paths` to the branch's Python rather than
  leaving it off.
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
- **A predicate that means "the port has started" must ask whether a
  kernel is *served*, not whether the extension exists** (Task 2.1). From
  Phase 02 on, `hazma._core` is in every build while every value still
  comes from Cython. `tolerances.provenance` keyed on
  `find_spec("hazma._core") is not None`, so the scaffold alone would
  have taken the corpus out of bit-equality mode into the 1e-8 budgets
  for the whole of Phases 02–03, with nothing turning red
  (`docs/agents/lessons.md` `[gate-disabled-stays-green]`). It now keys
  on `cases.rust_core_kernels()`, and so does `assert_no_rust_core` —
  which is also what keeps the corpus *repairable* until the first swap.
  **Both flip permanently at the first Phase 04 kernel**, so the
  ill-conditioned-points repair has to land before that swap.
- **abi3 is verifiable on a laptop, and the check has a side benefit**
  (Task 2.1). The exact `hazma/_core.abi3.so` built under CPython 3.12.12
  loads and runs under 3.13.7. Doing it in an interpreter with no NumPy
  installed is what exposed that the `numpy` crate **panics** rather than
  raising when NumPy is missing — `cast::<PyUntypedArray>` reaches for
  the array-API capsule. The dispatch helper takes a `PyFloat` fast path
  first so a scalar never touches NumPy; every Phase 03–06 kernel
  inherits that ordering.
- **The live Cython dispatch is not the contract
  `../references/numerics-replacements.md` described** (Task 2.1, now
  written into that reference): a 0-d array raises rather than taking the
  scalar path, a Python list is *accepted*, shape errors are
  `AssertionError` not `ValueError`, and
  `hazma/spectra/_neutrino/_muon.pyx:205` says "Photon energies". Two of
  the four become silent public-API narrowings if the port transcribes
  the design instead of the code. **Settled by Task 3.5 (2026-08-11)**,
  which found the deeper problem: there are four dispatch *shapes*, not
  one shape with four divergences — see the next Task 3.5 bullet below.
- **The Rust half of the build is now pinned rather than lucky** (Task
  2.2). Until this task, CI had no toolchain step and passed anyway
  because the GitHub-hosted images ship cargo — a dependency nothing in
  the repo required and an image refresh could have removed from every
  matrix entry at once. Every entry now installs one, CI grew a `rust`
  job running the same three cargo gates as `preflight.sh`, and
  `release.yml` grew a rustup step inside the manylinux container
  (cibuildwheel builds Linux wheels in a container that cannot see a
  host toolchain; macOS builds on the runner and uses the host one —
  two artifacts, two mechanisms, exactly like `MANIFEST.in` vs the
  wheel in Task 0.4).
- **`cargo build` publishes nothing to Python** (Task 2.2). The crate
  reaches an importable `hazma/_core.abi3.so` only through
  `pip install -e .`; cargo works out of `rust/target/`. This is the
  `.pyx`-rebuild trap with an extra step, since the fast iteration
  command and the publishing command are now different commands. It is
  written into `AGENTS.md`, `docs/agents/environment.md`,
  `docs/agents/preflight.md`, and the rebuild-awareness bullet of all
  seven review/commit skills, so Phases 03–06 inherit a reviewer who is
  expected to challenge a cargo-only run quoted as a Python result.
- **The dispatch contract is now pinned from Python, and three of its
  behaviors were not in the prose** (Task 2.3; **(b) superseded by Task
  3.5, 2026-08-11**). `test/test_core_dispatch.py` holds every branch of
  the contract: (a) **rank is checked before dtype**, so a 2-D int64
  array reports the dimension message; (b) ~~a 0-d array still enforces
  dtype where a Python `int` does not~~ — **a 0-d array of any numeric
  dtype is now the scalar it holds**, and only a non-numeric dtype
  (`<U4`, `object`) is rejected there, because a 0-d array *is* a scalar
  and `np.int64(4)` was already accepted; (c) **non-`float` NumPy scalars
  are accepted** (`np.float32`, `np.int64`, `np.uint8`, `np.bool_`) via
  the `extract::<f64>` arm. **That module is the template Phases 04–06
  copy** — swap the probe and the quantity wording, keep every test, add
  the numerical tests beside rather than merged in. Two side facts worth
  carrying: bit patterns survive intact (NaN payload included), so the
  module argues about no tolerances; and a "fresh" array from the `numpy`
  crate has `owndata == False` with a `PySliceContainer` base, so
  **non-aliasing is the assertable property, never `owndata`**. The
  settled contract — including the `TypeError` arm, sequence acceptance,
  and the `map_flavors` / `require_vector` siblings — is the Task 3.5
  bullet below and
  [`../references/numerics-replacements.md`](../references/numerics-replacements.md)'s
  "settled contract" section.
- **`text_signature` is a claim PyO3 does not enforce** (Task 2.3).
  `roundtrip` advertised `(x, /)` while `roundtrip(x=1.5)` worked;
  enforcing positional-only takes `#[pyo3(signature = (x, /))]`. The
  Cython entry points are `def` functions that accept keywords
  (measured), so a `/` in a template's `text_signature` is a latent
  public-API narrowing waiting to be copied into every Phase 04–06
  wrapper. Fixed to `"(x)"` — the same thing `hazma/_core.pyi` already
  described.
- **A fresh env drops the parity corpus out of bit-equality mode, and
  the cause is a NumPy patch release** (Task 3.1). `uv pip install -e .`
  on a new venv resolved NumPy **2.5.2**; the corpus manifest records
  **2.5.1**, and `tolerances.provenance` compares the whole numerics
  environment, not just the kernel digest and the served-kernel
  predicate. Result: `exact: False`, detail `numpy '2.5.1' -> '2.5.2'`,
  the budgets in force, and a bare suite reporting **14 skipped instead
  of 13** — the corpus's own signal working exactly as Task 2.3
  documented, but it costs a re-run if you notice it only at the end.
  **Build the env with `numpy==2.5.1` pinned**, and rebuild
  `--no-build-isolation` so the extensions compile against the same
  headers:

  ```sh
  uv pip install --python .venv/bin/python "numpy==2.5.1" setuptools \
      wheel "cython==3.2.9" setuptools-rust
  uv pip install --python .venv/bin/python -e . --no-build-isolation
  ```

  Check before trusting any parity claim, in one second and without
  running the suite:

  ```sh
  python -c "import json,sys; sys.path.insert(0,'test/parity'); \
  import tolerances; \
  print(tolerances.provenance(json.load(open('test/parity/data/manifest.json'))))"
  ```

  The digest and the served-kernel predicate were both clean here — it
  was purely the dependency. This will recur for every Phase 03–06 task
  that builds a fresh env, and it will recur harder as NumPy and SciPy
  move on.
- **Hazma holds three fine-structure constants** (Task 3.1):
  `_utils/constants.pxd` `1/137.035999084` (a pre-CODATA-2022 value —
  CODATA 2022 is 137.035999177(21), arXiv:2409.03787),
  `_utils/legacy_parameters.pxd` `1/137`, and `hazma/parameters.py:205`
  `1/137.04`. The masses, by contrast, agree: all **fourteen** in
  `constants.pxd` are bit-equal to their `parameters.py` counterparts
  (checked; `MASS_K0`/`MASS_KL`/`MASS_KS` share one value, so the
  correspondence is 14 names onto 12 distinct numbers — see
  [phase-03/README.md](phase-03/README.md) for why that count is a
  trap). The third α is pure Python and outside this project's scope,
  but any future table-merge follow-up has to account for it.
- **One `.pyx` reads from *both* constant tables, and the port had to
  find that out rather than be told** (Task 3.1).
  `hazma/spectra/_photon/_pion.pyx` `include`s `constants.pxd`, so its
  `MPI` / `ME` / `MMU` aliases are PDG values — but its five hard-coded
  kinematic literals (`ENG_MU_PIRF`, `GAMMA_MU_PIRF`, `BETA_MU_PIRF`,
  `ENG_GAM_MAX_MURF`, `ENG_GAM_MAX_PIRG`) reproduce **bit-exactly** from
  `legacy_parameters.pxd`'s masses and from no other table — someone
  evaluated the formulas once, against the older header, and pasted the
  digits. Recomputing them from the header the file actually includes
  moves `ENG_MU_PIRF` by 4.7e-5 MeV and every charged-pion photon
  spectrum with it. The divergence recorded in
  `../references/cython-inventory.md` §Bugs 3 was between two *files*;
  this one is inside a single module, so "which header does this
  extension include" is not enough to answer "which masses does it use".
  **Phase 04 must not consolidate it**; `constants::derived::photon_pion`
  carries both halves with the reasoning, and two tests in each language
  fail if either half moves.
- **A per-file bit-equality check cannot catch a consolidation**
  (Task 3.1). Adopting one table's masses in the other passes every
  file-by-file comparison — each side still matches *some* source. What
  catches it is a literal roster of the 19 names the two `.pxd` share
  and the 12 they disagree on, asserted as a partition
  (`test_the_two_tables_diverge_on_exactly_the_recorded_names`). A
  computed partition would have accepted any partition, which is the
  general shape: rule 4's content is that a specific split does not move,
  so the test has to name it.
- **`R_FACTOR`'s Cython comment has an exponent typo** (Task 3.1). Both
  muon kernels annotate the literal `1.0001870858234163` with a
  `12 r^2 ln(r^2)` log term; only `r^4` reproduces the digits. The number
  is right, the comment is wrong, and the `.pyx` is left untouched — but
  a Phase 04 port that recomputes from the comment instead of copying the
  number lands 0.3% away — `0.9972020119096803` against
  `1.0001870858234163`. Pinned in `test_core_constants.py`.
- **The plan's "scipy is cephes, so this is algorithm-for-algorithm
  parity" is true for `spence` and `k1` and false for `kn`** (Task 3.2).
  `scipy.special.kn` dispatches integer orders to `kv`; only `k0`/`k1`
  are still cephes there. So the *faithful* cephes `kn` — `spec_math`'s,
  and equally the plan's "vendor the cephes routine" fallback — misses
  scipy by up to **5.1e-9** relative over `x ∈ [1e-8, 300]`, worst at
  `x = 9.531`. That is not academic: the mediator prefactor is
  `x/(2·kn(2,x))²`, so it enters `thermal_cross_section` **squared**,
  right at the parity corpus's 1e-8 budget for that function — a Phase
  05 swap could have shipped it inside budget and moved published
  numbers. `crate::special::bessel_kn` builds `Kₙ` from the upward
  recurrence `K_{m+1} = K_{m-1} + (2m/x)·K_m` on cephes `k0`/`k1` seeds
  instead: ≤ 3.4e-15 vs scipy for orders 0–5. `../references/numerics-replacements.md`
  was patched, since its own prose is what made cephes `kn` look safe.
  **The general shape, and the one Task 3.3 should carry: the plan's
  model of a third-party library is a hypothesis, and the sweep that
  "confirms" it is the only thing that can refute it.**
- **A Python-visible test surface on `hazma._core` reads as a started
  port** (Task 3.2) — the second instance of
  `docs/agents/lessons.md` `[gate-disabled-stays-green]` in this
  project. `cases.rust_core_kernels()` counts every public callable
  except the literal name `roundtrip`, so registering
  `hazma._core.special` (needed because the oracle, scipy, lives in
  Python) flipped the corpus to `exact=False, detail='hazma._core serves
  3 kernel(s)'` for the rest of Phases 03–06 with nothing turning red.
  Fixed with a **submodule**-level exemption
  (`cases._CORE_TEST_ONLY_MODULES`) rather than a name-level one — a
  name exemption would also cover a future real kernel of the same name
  — and made conditional on a checkable property of the tree by
  `test_test_only_core_submodules_have_no_importer`, which fails the
  moment anything under `hazma/` imports an exempted module. **Any later
  task putting a non-kernel on the extension inherits this mechanism and
  must not widen the exemption to quiet a red mode check.**
- **The quadrature break-point contract is scipy's, not QUADPACK's**
  (Task 3.3), and the plan's own instruction to pin it empirically is
  what surfaced that. `scipy.integrate.quad` filters `points` in Python
  before `qagpe` ever sees them — `np.unique`, then strictly interior —
  so the rule a documentation-driven port would implement (sort, and
  `ier = 6` unless the extremes equal `a` and `b`) is unreachable, and
  the five `points=[-1, 1]` call sites would have **errored** under it.
  Both live degeneracies turn out to be discards, and the consequence is
  a dispatch one: `points is None` selects `qagse`, "no break point
  survived" does not, so five of the twelve live call sites run `qagpe`
  with an *empty* list. This is the same shape as Task 3.2's `kn`
  finding — the plan's model of a third-party library is a hypothesis,
  and the sweep that "confirms" it is the only thing that can refute it.
- **The QUADPACK port tracks scipy wherever QUADPACK converges, and only
  there** (Task 3.3). Over 11,274 random (integrand, tolerance, limit,
  points) combinations the 4,461 converged runs reproduced scipy's
  `neval` and `last` on all but **5** (0.11%) and landed within 3.6e-2 of
  the requested tolerance (8.2e-11 relative worst case); the 6,813 that
  exhausted `limit` can separate without bound (4.5e-5 in that sweep, 11%
  on a hand-picked case), because Wynn's ε-algorithm is chaotic on a
  non-converging sequence. Termination flags agreed on all 11,274.
  **Phases 04–06 inherit the obligation**: no live shape reaches the
  second regime today, every one returns `ier = 0`, and
  `test/test_core_quad.py` asserts it — a future kernel that does reach it
  would be a silent change, since QUADPACK returns a number either way.
  **The sweep's parameter space is part of its result:** an earlier
  6,000-combination design capped at two break points reported *zero*
  subdivision mismatches among converged runs, and the mismatches only
  appeared once 9- and 39-point grids entered the draw.
- **A mutation harness can poison its own baseline** (Task 3.3). Two
  copies of the campaign ran concurrently, because the first was read as
  having failed to start when it had not, so the second's "pristine"
  source already carried the first's mutation and every result was
  measured against a wrong Gauss–Kronrod table. The tell was easy to
  rationalise — mutating a `qk15` weight reported `qk21` tests failing —
  and what settled it was a check owing nothing to the crate: re-parsing
  the Fortran `data` statements and comparing f64 bit patterns against the
  Rust literals. **Assert a green baseline before a campaign and again
  after it, and hold a lock.** Two smaller siblings worth carrying:
  `cargo test`'s default parallelism interleaves `test NAME ... FAILED`
  lines, so a scraped failure list names the wrong tests
  (`-- --test-threads=1`); and a background job reported as failed may
  still be running.
- **The compiled Cython's arithmetic is fused, and the port had to match
  it to pass the corpus** (Task 3.4). Clang defaults to
  `-ffp-contract=on`, so `a*b + c` becomes a fused multiply-add; the
  corpus's capturing platform (macOS/arm64) does it at eight distinct
  expressions across `boost.pyx`, and NumPy's `arr_interp` does it too.
  Written the obvious unfused way, the Rust misses the corpus by up to
  **3.6e-12** relative *on the corpus's own grids* for the seven
  tabulated photon spectra — past the 1e-12 `TABULATED` budget, so the
  Phase 04 swap would have failed its own gate and the only alternatives
  would have been widening a budget by three decades or shipping a
  declared drift. With `f64::mul_add` at those sites the port is
  bit-equal at every one of those points. The sites were established
  twice over: `fmsub`/`fmadd` in the disassembled `.so`, and a
  16-combination bisection against the live kernel in which only all-on
  reaches zero mismatches (the next best leaves 115 of 2,462).
  **The converse is the trap Phases 04–06 inherit:** `boost_beta` spells
  its square as `(mass/energy) ** 2`, whose rounded product completes
  before the subtraction, and *none* of its ten inlining call sites
  contract it — fusing it would move every boosted spectrum. Contraction
  is a per-expression fact, so measure each kernel rather than adopting a
  house style. Same shape as Tasks 3.2 and 3.3: the model of the
  third-party artifact is a hypothesis, and only the sweep can refute it.
- **`boost_integrate_linear_interp` mis-covers its window at both ends,
  and near threshold the public spectra are wrong by four orders of
  magnitude** (Task 3.4). The interior sum's slice `yy[ilow:ihigh]` is
  exclusive at the top while the upper partial-cell term starts at
  `x[ihigh]`, so one cell is covered by nothing; and with both bounds
  inside a single cell the two partial-cell terms **overlap**,
  over-counting by (cell width)/(window width), which diverges as
  `β → 0`. All seven tabulated photon spectra therefore blow up instead
  of converging to their own rest-frame spectrum as the parent slows —
  6,500× to 33,000× at one part in 1e12 above rest, measured through the
  public API. This is a live defect in hazma 2.1.0, not something the
  port introduced. Reproduced per rule 1, pinned in both languages, and
  filed as
  [`../../../docs/followups/todo/boost-integral-drops-last-interior-cell.md`](../../../docs/followups/todo/boost-integral-drops-last-interior-cell.md)
  — blocked until after Phase 06 Task 6.4, because the repair needs a
  declared corpus regeneration. **The corpus pins the wrong values by
  design**, so a Phase 04 swap that "fixes" this fails the gate.
  `../references/cython-inventory.md` §Bugs lists the same class in the
  *dead* `boost_integrate_linear_interp_massive`; the live twin was never
  flagged, which is worth remembering before trusting that audit's
  coverage of the surviving code.
- **A test whose oracle is something *you compiled* is scoped to that
  build** (Task 3.4, PR #61 review round 1). Nineteen bit-equality
  assertions against the Cython and NumPy passed on macOS/arm64 and failed
  on Linux/x86-64, because `-ffp-contract=on` is the C default and only a
  target with hardware FMA actually contracts — so the assertion was a
  statement about the platform, not about the port. The fix is to measure
  the property at import and skip where it does not hold, which is what
  `test/parity` already does (CI: `pytest --ignore=test/parity` off
  macOS); loosening to a tolerance is wrong, because the worst relative
  gap between two roundings lands at whatever cancellation point the
  domain contains. **Every Phase 04–06 kernel test that uses the Cython
  twin as its oracle inherits this** —
  `docs/agents/lessons.md` `[platform-scoped-oracle-asserted-globally]`.
- **A `cdef` declared in a `.pxd` is callable from Python** (Task 3.4).
  Cython exports it through the module's `__pyx_capi__` as a `PyCapsule`,
  so `ctypes` can call the *live* kernel at arbitrary arguments — which
  is what stood in for the "micro-fixtures captured in Phase 01" the
  phase file's Task 3.4 criterion named and that Phase 01 could never
  have captured (the corpus enumerates top-level `def`s only). Two
  constraints: use `ctypes.PYFUNCTYPE`, never `CFUNCTYPE`, because the
  latter releases the GIL and anything calling back into Python
  segfaults with no Python-level error; and the capsule's *name* is its C
  signature string, so assert on it rather than trusting the argument
  list. **Any later task needing an oracle for `cdef` code should reach
  for this rather than adding a temporary shim to a `.pyx`.**
- **The dispatch contract the reference described is not the dispatch the
  Cython implements, and there are four shapes rather than one**
  (Task 3.5, now written into that reference). Classified from source over
  all 43 surviving top-level `def`s and then measured on the built tree:
  15 entry points branch on `hasattr(x, '__len__')` and raise
  `AssertionError` on a 0-d array while accepting a list (12 photon, 2
  positron, and `scalar_mediator_decay_spectrum`); the 18
  cross-section entry points branch on `hasattr(...) and x.ndim > 0`,
  *accept* a 0-d array via `.item()`, carry no rank guard at all, and
  reject a list with `AttributeError`; the 2 neutrino ones are the first
  shape with a 3-tuple / `(3, N)` return; and `partial_widths` is a
  required-1-D argument with its own two messages. The port is three
  helpers over one classification — `dispatch::{map_unary, map_flavors,
  require_vector}` — and **Phase 05's cross sections use the same one as
  Phase 04's spectra**, which is what makes the two shapes agree for the
  first time.
- **The rule that settles every dispatch divergence, and the one Phases
  04–06 should quote: each exception the Cython raises *explicitly* keeps
  its type; only its `assert`s change type** (Task 3.5; rules.md rule 9).
  That is what makes the answer checkable instead of a taste question —
  rank errors become `ValueError` carrying the assert's message verbatim,
  dtype errors stay `ValueError`, a non-number stays `TypeError`, and
  `partial_widths`' explicit `raise ValueError` keeps type *and* wording.
  The five resulting behavior changes are listed under "Numerical impact
  so far" because the Phase 07 CHANGELOG is assembled from there.
- **A 0-d array's `__float__` forwards to its element, and `np.str_`
  subclasses `str`** (Task 3.5), so `float(np.array("15.0"))` is `15.0`.
  A first draft that accepted a 0-d array by attempting the conversion
  therefore returned a *number* for `dnde_photon("15.0", 200.0)` where the
  Cython raises. The fix is to ask the dtype's `kind`. **Any Phase 04–06
  check that means "is this numeric?" and answers it by trying a float
  conversion has the same hole**, and PyO3's `extract::<f64>` goes through
  `PyNumber_Float`, which is the conversion in question.
- **A mutation campaign can refute the implementation's own comment**
  (Task 3.5). Thirteen of fourteen mutations were caught; the survivor
  swapped two arms of the classification that a code comment called
  load-bearing against the string bug above. It is not — the only objects
  with both `__len__` and a working `__float__` are 0-d ndarrays, taken by
  an earlier arm — so the ordering is fidelity to the Cython and the dtype
  check is the actual guard. **A survivor is a result to read, not a hole
  to paper over**: Task 3.4's three survivors named the bisection tests it
  needed, and this one named a false sentence.
- **A worktree can inherit `.so` files whose source package is gone**
  (Task 1.1): this tree carried `_gamma_ray/` and `_phase_space/`
  extensions deleted in Task 0.2, giving 25 `.so` against `setup.py`'s
  20. Same class as the stale generated `.c` already in
  `docs/agents/environment.md`; the same clean-then-rebuild recipe
  fixes it, and any "N extensions" claim must be taken after it.
- **The port has now surfaced two live 2.1.0 numerical defects, both by
  writing an analytic test the original never had** (Tasks 3.4 and 4.1).
  The second: `hazma/spectra/_positron/_muon.pyx` **divides** by the
  Michel normalization where it should multiply, so every positron
  spectrum is low by `1/R_FACTOR² = 0.9996259` — **0.0374%, uniformly**,
  at every parent energy. The raw polynomial integrates to exactly
  `1/R_FACTOR` (`scipy.integrate.quad` 0.999812949171142, closed form
  0.9998129491711419, `1/R_FACTOR` 0.999812949171142 — all three to the
  last digit), so normalizing means multiplying. **The sibling proves it
  is an inversion and not a convention:**
  `hazma/spectra/_neutrino/_muon.pyx:23` declares the identical constant
  and `:58`/`:114` multiply by it. It propagates to
  `dnde_positron_charged_pion` (∫ = 0.999623 at `E_π = 500 MeV`), both
  mediator positron modules, and every positron-based limit. Reproduced
  per rule 1, pinned in both languages, filed as
  [`../../../docs/followups/todo/positron-muon-spectrum-normalization-inverted.md`](../../../docs/followups/todo/positron-muon-spectrum-normalization-inverted.md)
  and blocked behind Phase 06 Task 6.4 for the same corpus reason as its
  Task 3.4 sibling. **Worth telling the maintainer separately from this
  project's schedule; it affects published numbers today.**
- **Disassemble the shipped `.so` before writing the Rust, and the port
  lands bit-equal first try** (Task 4.1). Nine `fmadd`/`fmsub` in
  `_positron/_muon`, all inlined into the point kernel; three expressions
  that look fusable are not — `x² − 4r²`, `1 − β²`, and any sum whose
  operand went through a division, which breaks the contraction. No
  bisection round was needed, unlike Task 3.4's. Reading the folded
  constants out of the `movk` immediates is worth doing too (the
  halfwords are little-endian, assembled high-last; transposing two is
  exactly what the resulting test catches).
- **Scope a bit-equality-against-Cython test to the corpus's capturing
  platform, not to a probe for the mechanism you think differs** (Task
  4.1, and only CI could see it — two failures after two green macOS
  runs). The first version of `test/test_core_positron_muon.py` skipped
  its against-the-Cython class by comparing the compiled kernel against
  an unfused Python transcription: agree ⇒ this build does not contract ⇒
  skip. The transcription first got an *association* wrong
  (`pre * (num/den)` is not `pre * num / den`), and once fixed, Linux
  still disagreed — on a build with no `-march` flag and so no hardware
  FMA for the probe's own mechanism to explain. **The cause was not
  localized and does not need to be:** a compiler contracting a different
  set of expressions, or a libm rounding one call differently, breaks the
  comparison just as thoroughly, and a probe over one mechanism cannot
  see the others. The scope is now read out of
  `test/parity/data/manifest.json`, the same mechanism `test/parity` and
  `.github/workflows/ci.yml` already use. **Mirror of Task 3.4's
  `[platform-scoped-oracle-asserted-globally]`: there the claim was
  asserted too widely; here the scoping mechanism itself was wrong, and
  the capturing platform answers True either way so it could never tell.**
  The mode is now *declared* from the platform and the off-platform
  divergence measured, so the comparison degrades to a peak-scaled
  budget rather than skipping: 47 tests, nothing skipped on any
  platform.
- **A fused Python reference reproduces the shipped macOS Cython
  bit-for-bit, and is a disassembly-independent check on an FMA map**
  (Task 4.1). Built with a correctly-rounded `fma` (`Fraction`-based;
  `math.fma` needs 3.13 and the suite supports 3.10) at exactly the sites
  the port fuses: **0 mismatches in 21,000 points**, against 11,713 for
  the unfused form on the same draw. Cheap, and worth doing per kernel —
  but note it confirms the map on *one* platform, which is why the scope
  above is a platform rather than an arithmetic property.
- **Repointing the corpus case is part of a swap, not bookkeeping**
  (Task 4.1). `test/parity/cases.py` names the `.pyx` module for every
  entry point, so a swap that repoints only the wrapper leaves the gate
  calling the twin — green, and measuring the implementation the swap
  replaced. The runner keys everything by *case name*, never by
  `module:function`, so repointing disturbs no stored data.
  `cases.PORTED_ENTRY_POINTS` records the `.pyx` origin, which keeps
  `assert_full_coverage` balanced and now also makes it an error for a
  ported entry point's `.pyx` to still export its `def` — rule 1's
  no-drift-window, in code.
- **A `NaN` energy does not propagate through a kernel that clips with
  `fmax`/`fmin`** (Task 4.1), in either language: `fmaxnm`/`fminnm` and
  Rust's `f64::max`/`min` both return the non-`NaN` operand, so the two
  kinematic limits collapse onto the rest-frame support and a finite
  number comes back. The corpus samples no `NaN`, so only a hand-written
  test catches a port that differs here. Expect the same shape in every
  boosted kernel.
- **A parameterised port is a diff between siblings, and that is how it
  finds bugs** (Task 4.2). The five tabulated photon `.pyx` differed only
  in table, parent mass and line terms. Written as one `dnde` over seven
  `Spectrum` values, the five line-weight expressions sit in one column
  and two of them are visibly wrong — neither is findable one file at a
  time. **Phases 05–06 have the same shape** (sixteen near-identical
  cross sections, four near-identical mediator spectrum modules) and
  should expect the same dividend.
- **The port has now surfaced three live 2.1.0 numerical defects, and
  Task 4.2 contributed two of them.** Both are line terms, both are
  reproduced per rule 1, both are filed and blocked behind Phase 06
  Task 6.4:
  - `hazma/spectra/_photon/_eta_prime.pyx:107` weights its `η′ → γγ` line
    with `BR` where its four two-photon siblings use `2·BR`, so the mode
    contributes **0.02307 photons per decay instead of 0.04614** — 0.63%
    of the η′ photon yield, missing, all of it at `M_η′/2 = 478.89` MeV.
    The ω's and φ's weights are correctly un-doubled (their modes yield
    one photon), which is what makes the η′ the odd one out rather than
    the family a mixed convention. Filed as
    [`../../../docs/followups/todo/eta-prime-two-photon-line-missing-factor-two.md`](../../../docs/followups/todo/eta-prime-two-photon-line-missing-factor-two.md).
  - `hazma/spectra/_photon/_phi.pyx:111,113` place both photon lines at
    `(M² + m²)/(2M)`, the **daughter meson's** energy rather than the
    photon's: 656.94 MeV where 362.52 is right for `φ → ηγ`, and **959.65
    where 59.82 is right** for `φ → η′γ` — a factor of 16, and 94% of the
    φ's rest mass in one photon. The ω uses `(M² − m²)/(2M)` and is
    right, which is the control; the φ's local is even named `eng_eta`.
    Filed as
    [`../../../docs/followups/todo/phi-photon-lines-use-the-daughter-meson-energy.md`](../../../docs/followups/todo/phi-photon-lines-use-the-daughter-meson-energy.md).
  **Four blocked defects now share one eventual corpus regeneration** — a
  fifth joined in Task 4.3, below.
- **The port has now surfaced five live 2.1.0 defects, and the fifth was
  found by an identity rather than a comparison** (Task 4.3).
  `hazma/spectra/_photon/_muon.pyx:41` cuts the muon-**rest-frame** photon
  spectrum at `y = 1 − √r` while the same file's in-flight branch (`:88`)
  and `_photon/_pion.pyx:16`'s `ENG_GAM_MAX_MURF` both use `1 − r`, which
  is the kinematic endpoint `(m_μ² − m_e²)/(2m_μ)`. So the spectrum is a
  hard zero over the top **0.2543 MeV** (0.48%) of its support, where it
  is still `5.34e-7 MeV⁻¹`, and it is **discontinuous in `E_μ` at rest**:
  a muon one part in `10¹²` above rest returns `5.336e-7` where a muon
  exactly at rest returns `0.0`. `5.45e-8` photons per decay are lost —
  `1.1e-6` of the yield above 1 MeV. What found it was writing the
  statement the original never made: *the in-flight closed form is the
  boost integral of the rest-frame distribution*, which holds to machine
  precision — but only when integrated to `1 − r`. Filed as
  [`../../../docs/followups/todo/photon-muon-rest-frame-endpoint-uses-the-wrong-power-of-r.md`](../../../docs/followups/todo/photon-muon-rest-frame-endpoint-uses-the-wrong-power-of-r.md).
  **Five blocked defects now share one eventual corpus regeneration.**
- **`numpy.sum(axis=0)` is pairwise above eight terms, and exactly one
  shipped table is wide enough to notice** (Task 4.2). The Cython built
  every tabulated rest-frame spectrum as
  `np.sum(np.loadtxt(csv).T[1:], axis=0)`. Six of the seven tables reduce
  over 2–7 mode columns, where NumPy's pairwise routine degenerates to a
  sequential fold; `phi_photon.csv` has **ten**, and there a sequential
  sum is not bit-equal. `boost::pairwise_sum` — written in Task 3.4 for
  `np.trapezoid` — is reused rather than re-derived, and a mutation to
  the sequential form fails six tests, all φ. **The same trap waits
  wherever a port re-derives a NumPy reduction on a wide array.**
- **Deleting an extension strands whatever read its module *globals*, not
  only whatever imported it** (Task 4.2). `test/test_core_interp.py` and
  `test/test_core_boost.py` built their seven-table fixtures from
  `_eta.eta_data_energies` and friends, and both failed at **collection**
  — so a bare `pytest` reported two errors and ran nothing else, which
  reads like a broken build rather than a stranded dependent. Repaired by
  loading the CSVs the way the deleted modules did, which incidentally
  makes those oracles independent of the Rust that now consumes them.
  Task 0.2's rule generalises: **a stranded dependent belongs to the task
  that strands it, and "dependent" includes readers of state, not just
  importers.**
- **A monkeypatch that shadows a real submodule stops measuring a delta**
  (Task 4.2). `test/parity/test_parity.py`'s two served-kernel meta-tests
  patched `hazma._core.photon` and `hazma._core.positron` with fakes and
  asserted `baseline + 1` and `[]`; filling `photon` with seven real
  kernels made the fake *replace* seven and add one. Both now patch
  `hazma._core.not_a_real_domain`, a name no domain will take. **Any
  later task filling `neutrino`, `scalar_mediator` or `vector_mediator`
  would have hit this.**
- **Some Cython behavior has no faithful port, and the honest move is to
  declare the change rather than approximate it** (Task 4.2). A `NaN`
  photon energy made the Cython raise `IndexError` out of
  `np.flatnonzero(lb <= x)[0]` on an empty match, and made the Rust
  *panic* at an `.expect`. `dispatch::map_unary` evaluates element by
  element and has no per-element error channel, so neither exception
  survives; the port returns `NaN`, which is what the same kernels'
  rest-frame branch already returned. Declared in `rust/src/boost.rs`'s
  faithfulness notes, in "Numerical impact so far", and by test. **Expect
  this wherever a kernel's error path is per-element rather than
  per-call.**
- **Two clippy lints fire on faithful guards and must be allowed, not
  taken** (Task 4.1). `beta < 0.0 || beta > 1.0` is not
  `!(0.0..=1.0).contains(&beta)` — `contains` is false for a `NaN`, so
  the "simplification" would return `0.0` where the Cython falls through
  to the arithmetic. `emu - MASS_MU < f64::EPSILON` trips
  `float_equality_without_abs` while being a genuine one-sided threshold.
  A third, `xm > xp || xp < xm`, really was redundant. Read each one.
- **The port has now surfaced six live 2.1.0 defects, and the sixth is a
  quadrature that loses its own support** (Task 4.4).
  `hazma/spectra/_photon/_pion.pyx:123` integrates the charged pion's
  photon spectrum over the whole of `cos θ`, but the integrand is nonzero
  only where the pion-rest-frame photon energy stays under
  `ENG_GAM_MAX_PIRG = 69.783` MeV. As the lab photon approaches the
  boosted endpoint that window narrows past QUADPACK's largest first-rule
  abscissa (~0.9956), so **every node returns zero, the error estimate is
  zero, and `qagp` terminates successfully with `0.0`** where the spectrum
  is not zero. At `E_π = 10 m_π = 1396` MeV — the corpus's own most
  boosted block — `dnde_photon_charged_pion(900, 1396)` is `0.0` against a
  reference of `3.586e-07` MeV⁻¹, and the spectrum is a hard zero over
  roughly the top quarter of its support. Integrated the loss is 0.0054%
  at `E_π = 1` GeV, 0.041% at 1396 MeV and 2.96% at 5 GeV, so it is a
  **shape** defect rather than a yield defect at hazma's own scales — what
  it breaks is a line search or a tail-dominated limit, not a total. The
  port reproduces the zeros in exactly the same places. Filed as
  [`../../../docs/followups/todo/charged-pion-photon-spectrum-misses-the-forward-cone.md`](../../../docs/followups/todo/charged-pion-photon-spectrum-misses-the-forward-cone.md).
  **Six blocked defects now share one eventual corpus regeneration.**
- **Read the `cdef`s, not only the expressions** (Task 4.4).
  `_photon/_pion.pyx`'s neutral-pion kernel declares `cdef float beta` and
  `cdef float ret_val` in a file where every other local is a `double`,
  confirmed by two `fcvt` round trips in the shipped object. A port that
  transcribes the formula and not the declaration lands **8.5e-9**
  relative away, at an entry point the corpus holds to `rtol = 0`. Third
  instance in this project of a declaration carrying numerics the
  expression does not — Task 3.1's mixed constant tables and Task 4.3's
  `y**3`-is-a-libm-`pow`-call are the others.
- **An FMA site inside a quadrature integrand cannot be gated, and neither
  can a one-ulp constant that multiplies it** (Task 4.4, measured by
  mutation). Unfusing `F_A² + F_V²` in `dnde_pi_to_lnug` leaves the worst
  corpus difference at **2.618e-15 — the identical figure the correct port
  produces** — with 120 `cargo` tests and 73 per-kernel tests green; a
  one-ulp `f_π` spelling is caught only by a `cargo` test on the constant
  itself. The integral does not carry the integrand's last bit out, and
  unlike Task 4.3 there is no bit-equality mode to fall back on, because
  the port replaces *scipy's* QUADPACK rather than a closed form. **Every
  quadrature-backed kernel from here on inherits this** — Tasks 4.5, 4.6
  and the Phase 06 mediator spectra — so the FMA map is defended by the
  disassembly and by review, and the source says so rather than leaving a
  reader to assume a gate exists.
- **Phase 03 Task 3.3's divergent-regime obligation is discharged for the
  first `qagp` consumer, and the answer is reassuring** (Task 4.4). The
  charged pion's quadrature does leave `ier = 0` — first at `E_π = 4e4`
  MeV (`γ_π ≈ 290`), 40 GeV against a sub-GeV library and two decades
  above the corpus ceiling. But over an 11 × 8 grid reaching `E_π = 1e5`
  the port's `ier` **equals the flag scipy raises on the Cython twin at
  all 88 points**, including both `ier = 4` entries and the non-monotonic
  pattern between them, with the values still agreeing to 2.8e-11.
  Task 3.3's warning was that the two *may* separate without bound where
  QUADPACK does not converge; on the only live shape that reaches that
  regime, they do not.
- **A mutation harness that reverts with `git checkout --` cannot revert a
  file git has never seen** (Task 4.4). The new kernel module was
  untracked, the restore step errored, the driver did not check, and five
  mutations accumulated while being read as five independent
  measurements — Task 3.3's `[mutation-harness-poisons-its-own-baseline]`
  in a new disguise, with the same tell (implausibly uniform failure
  counts). Snapshot outside the tree, `cmp` before each mutation, verify
  the restore.

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
    "Neutrino energies".

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

(Per-function drift lines land here as Phase 04–06 swaps merge; the
Phase 07 CHANGELOG is assembled from this section — do not reconstruct
it from memory.)

## Decisions and Implementation Notes

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
  rather than ~160.
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

## Files Changed

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

## Verification

- Scaffolding PR: `scripts/agents/preflight.sh` (repo gate; no code
  changes).
- **Phase 04 Task 4.4 state (2026-08-17) — the pion pair:** bare
  `pytest -q` → **`1755 passed, 15 skipped, 8 warnings in 605.61s`** on
  the capturing environment. Collection goes 1697 → 1770 against
  `origin/master`, **+73 and every one of them
  `test/test_core_photon_pion.py`**. `pytest -q test/parity` →
  `629 passed, 1 skipped`, with `spectra.photon.charged_pion` at 2.618e-15
  worst relative against the 1e-12 budget this task tightened it to and
  `spectra.photon.neutral_pion` bit-equal at all 1,305 values.
  `cargo test --no-default-features` → `120 passed` (11 new, all
  `kernels::photon_pion`); fmt and clippy clean.
  `pytest -q test/test_theory_aggregation.py` → `69 passed` either side of
  the swap. An **eleven-mutation** validity campaign is in the task note,
  with two survivors — an FMA site inside the quadrature integrand and a
  one-ulp constant — both unobservable through the entry point for the
  same reason and both recorded in the source. The campaign's first run
  had to be discarded and rebuilt: see Findings.
- **Phase 04 Task 4.2 state (2026-08-12) — the tabulated photon family:**
  bare `pytest -q` → **`1628 passed, 15 skipped in 587.90s`** on the
  capturing environment. Collection goes 1458 → 1643 against
  `origin/master` (`pytest --collect-only -q` on both trees), **+185 and
  every one of them `test/test_core_photon_tables.py`** — no other module
  gains or loses a test, which is the check that a swap this large moved
  no existing coverage. `pytest -q test/parity` →
  `629 passed, 1 skipped`; `python test/parity/generate.py --check` →
  `corpus OK: 41 cases / 1580 arrays`;
  `cargo test --no-default-features` → `96 passed` (16 new: 15 for
  `photon_tables` and one `NaN`-window test in `boost`); clippy and fmt
  clean; `scripts/agents/preflight.sh` **RESULT: PASS**.
  Two earlier preflight runs failed on lint alone — 29 ruff findings in
  the new test module, then the wrapper's 11 pre-existing ones — with no
  gate other than black and ruff ever red. The skip count goes 14 → 15,
  and the new one is the charged kaon in the per-line photon-count test
  — it has no monochromatic line.

- **Phase 04 Task 4.1 state (2026-08-11) — first kernel swap:** bare
  `pytest -q` → **`1424 passed, 14 skipped in 555.96s`** on the
  capturing environment (from 1378/13 at Task 3.5: +47 passes for
  `test/test_core_positron_muon.py`, and −1 pass / +1 skip for
  `test_running_on_the_capturing_tree`, which now skips because the
  corpus is in budget mode — **that is the designed signal, and the skip
  count does not go back down**). `pytest test/parity -q` →
  `629 passed, 1 skipped`; `pytest test/test_core_positron_muon.py -q` →
  `47 passed`; `pytest test/test_theory_aggregation.py -q` → `69 passed`;
  `cargo test --no-default-features` → `80 passed` (11 new); clippy, fmt
  and `markdownlint` clean; `scripts/agents/preflight.sh` RESULT: PASS.
  **Eighteen mutations against `rust/src/kernels/positron_muon.rs`**, run
  sequentially from a green baseline with the baseline re-asserted after,
  each gated by `cargo test` *and* a rebuild plus the Python module —
  **13 caught, 16 after three tests were added, and the two that remain
  are provably equivalent mutants** (`x.mul_add(2.0, C)` is bit-identical
  to `x * 2.0 + C` because doubling is exact, which also means one of the
  nine `fmadd` sites in the shipped object code is unobservable; and
  `beta + beta` vs `2.0 * beta`, included as a control). The three caught
  late all moved a **branch boundary** without moving a value — Task
  3.4's shape exactly — and one of them exposed that `dndx`'s
  `beta < DBL_EPSILON` short circuit is **unreachable from
  `dnde_positron_muon`**, because the outer `E − m_μ < DBL_EPSILON` guard
  already routes everything that could reach it.
- **Phase 03 Task 3.5 state (2026-08-11) — phase closed:** bare
  `pytest -q` → `1378 passed, 13 skipped in 564.55s` on the capturing
  environment, parity suite included and in bit-equality mode (skip count
  unchanged at 13, and `tolerances.provenance` → `exact=True` checked
  directly); `pytest test/test_core_dispatch.py -q` →
  `118 passed in 4.19s`;
  `cargo test --manifest-path rust/Cargo.toml --no-default-features` →
  `69 passed` (2 new); clippy, fmt and `markdownlint --dot` clean;
  `scripts/agents/preflight.sh` RESULT: PASS. Fourteen mutations against
  `rust/src/{dispatch,kernels}.rs`, sequential from a green baseline with
  the baseline re-asserted after — **13 caught**. The survivor is the
  interesting one: it swapped two arms of the classification that the
  implementation's own comment called load-bearing, and left all 118
  tests green — so the *comment* was wrong (the real guard against a
  string parsing as a number is the 0-d dtype check), and it was
  corrected rather than the mutation dropped.
- **Phase 03 Task 3.4 state (2026-08-10):** bare `pytest -q` →
  `1314 passed, 13 skipped` on the capturing environment, parity suite
  included and in bit-equality mode (skip count unchanged at 13, and
  `tolerances.provenance` → `exact=True` checked directly);
  `pytest test/test_core_interp.py -q` → `33 passed in 0.46s`;
  `pytest test/test_core_boost.py -q` → `69 passed in 0.91s`;
  `cargo test --no-default-features` → `67 passed` (24 new); clippy,
  fmt and `markdownlint --dot` clean; `scripts/agents/preflight.sh`
  RESULT: PASS. Twenty-one mutations against `rust/src/{interp,boost}.rs`,
  run sequentially behind a lock with a green baseline asserted before
  and after — 17 of the first 20 caught, and all 21 after two tests were
  added. **The three survivors shared one shape**: each moved a *branch
  boundary* by a single double without touching any value the function
  returns, so no grid sample could see it. What catches that is
  bisecting on the bit pattern (`test_the_window_edges_sit_on_the_same
  _double_as_the_cython`) — and the parameter space matters as much as
  the sampling, since with `m = 0` the fused and unfused momenta are
  bit-identical and only massive-product draws can distinguish them.
- **Phase 03 Task 3.3 state (2026-08-10):** bare `pytest -q` →
  `1212 passed, 13 skipped` on the capturing environment, parity suite
  included and in bit-equality mode (skip count unchanged at 13);
  `pytest test/test_core_quad.py -q` → `58 passed in 5.10s`;
  `cargo test --no-default-features` → `43 passed` (27 new); clippy and
  fmt clean; `scripts/agents/preflight.sh` RESULT: PASS. Seventeen
  mutations against `rust/src/quad.rs`, 15 caught on the first pass and
  the two survivors (`qagpe`'s `ndin`, `qagse`'s roundoff threshold)
  covered by tests written afterwards against inputs found by searching
  with each mutation in place. The Gauss–Kronrod literals are bit-equal
  to the netlib Fortran (47 values, checked by a script that parses both
  sides independently of the crate).
- **Phase 03 Task 3.2 state (2026-08-09; PR #59 review round 1,
  2026-08-10):** bare `pytest -q` →
  `1154 passed, 13 skipped` on the capturing environment, parity suite
  included and in bit-equality mode (skip count unchanged at 13);
  `pytest test/test_core_special.py -q` → `65 passed in 0.50s`;
  `cargo test --no-default-features` → `16 passed` (9 new), clippy and
  fmt clean; `scripts/agents/preflight.sh` RESULT: PASS. Eleven
  mutations — nine against `rust/src/special.rs`, two against the
  corpus's served-kernel guard — each caught by the test whose name
  claimed it (tables in the task note). One of them, dropping the
  recurrence's order factor, passed `cargo test` on the first pass and
  is why the
  Wronskian unit test now runs at ν = 2 as well as ν = 1.
- **Phase 03 Task 3.1 state (2026-08-09):** bare `pytest -q` →
  `1088 passed, 13 skipped` on the capturing environment, parity suite
  included and in bit-equality mode;
  `pytest test/test_core_constants.py -q` → `25 passed in 0.03s`;
  `cargo test --no-default-features` → `7 passed` (5 new), clippy and
  fmt clean; `scripts/agents/preflight.sh` RESULT: PASS. Thirteen
  mutations — nine Python, four Rust — each caught by the test whose
  name claimed it (table in the task note).
- Per-phase Verification sections live in `phase-XX/README.md`.
- **Phase 02 closing state (2026-08-09):** bare `pytest -q` →
  `1063 passed, 13 skipped` on the capturing environment (1076
  collected), parity suite included and in **bit-equality mode**;
  `scripts/agents/preflight.sh` RESULT: PASS across all eleven rows, the
  three cargo gates green. `git diff origin/master -- hazma` is empty, so
  the public compiled surface is exactly where Phase 00 left it — the
  whole phase's only change under `hazma/` is the non-executable
  `hazma/_core.pyi` stub. Task 2.3's 54 new tests were validated by a
  six-mutation campaign against `rust/src/{dispatch,lib}.rs`, each
  mutation rebuilt and caught by the test whose name claimed it.
- **Phase 02 Task 2.2 state (2026-08-08):** `scripts/agents/preflight.sh`
  RESULT: PASS across all eleven rows — the three cargo gates green,
  `pytest` at `1009 passed, 13 skipped` (byte-identical to Task 2.1's, so
  no test outcome moved) with the parity suite still in bit-equality
  mode, markdownlint green over 16 changed docs. `git diff origin/master
  -- hazma rust` and `-- setup.py pyproject.toml MANIFEST.in` are both
  empty, so the compiled artifacts are the trunk's. PR #56's eight
  checks are green (including the new `rust` job, 30s), and the
  dispatched `release.yml` run 31297673951 is `success` on both
  platforms with `publish` skipped.
- **Phase 02 Task 2.1 state (2026-08-08):** bare `pytest -q` →
  `1009 passed, 13 skipped` in 569.45s on the capturing environment
  (1022 collected), parity suite in **bit-equality mode**;
  `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings` and
  `cargo test --no-default-features` (2 unit tests) all green from
  `rust/`; `python test/parity/generate.py --check` → `corpus OK: 41
  cases / 1580 arrays`; wheel and sdist both build, the sdist installs
  from source into a fresh venv on a *different* interpreter and runs
  both toolchains from outside the repo. `scripts/agents/preflight.sh`
  RESULT: PASS.
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
  strictly serial? Decide based on who's driving.
  **Note as of Task 4.2:** the corpus's mode flip has already happened,
  so Phase 05 no longer pays that cost by starting — but it *does* meet
  five of the six ill-conditioned blocks (all scalar cross sections), and
  now that Task 4.2 has cleared the sixth, **Phase 05 is the only work
  the follow-up below still gates.**
- **Does the φ spectrum omit a `φ → π⁰γ` line entirely?** (Task 4.2.)
  `hazma/_utils/constants.pxd:283` defines `BR_PHI_TO_PI0_A = 1.32e-3`
  and nothing reads it (checked against `origin/master` too), while the ω
  adds exactly the analogous line for its own `π⁰γ` mode. Suggestive but
  not settled — recorded as the first open question on
  [`../../../docs/followups/todo/phi-photon-lines-use-the-daughter-meson-energy.md`](../../../docs/followups/todo/phi-photon-lines-use-the-daughter-meson-energy.md)
  so whoever repairs the line energies settles it in the same PR. The
  port carries exactly what the Cython carried either way.
- **The ill-conditioned-points repair did not land before the first
  swap**, which the project handoff had asked for (Task 4.1). It was not
  blocking for `spectra.positron.muon` — none of the six affected blocks
  is that entry point — but **corpus regeneration is closed from now
  on**, so options 1 and 2 in
  [`../../../docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md`](../../../docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md)
  have to work from a pre-Phase-04 checkout. **Task 4.2 met the sixth
  block (`spectra.photon.eta[boosted_strong]`) and explicitly waived it,
  on evidence**: the port is bit-equal at 336,000 sampled points, so on
  the capturing platform there is nothing for a conditioning budget to
  absorb, and off it the parity suite does not run at all. That refutes
  the follow-up's prediction that "every affected block will produce a
  false failure the moment a Rust implementation lands" — *for that
  block*. **The remaining five are all scalar cross sections, so it is
  Phase 05, not the rest of Phase 04, that the follow-up still gates.**
- **The `TABULATED` budget class is kept rather than tightened to
  `EXACT`** (Task 4.2), even though the seven ported entry points would
  pass `EXACT` today. Unlike `spectra.positron.muon`, bit-equality here
  rests on reproducing *NumPy's summation order* — an implementation
  detail a future NumPy may change — so `EXACT` would be the wrong
  contract rather than a tighter one. The general rule for a later
  tightening: promote a class only when the bit-equality rests on an
  arithmetic identity, not on a third party's implementation.
- **One Rust module may serve several `.pyx`** (Task 4.2), stated as the
  exception to `kernels.rs`'s one-submodule-per-`.pyx` convention rather
  than left as a silent violation. `kernels::photon_tables` serves five,
  because the five differed only in data.
- **Should the corpus mode switch become per-case?** Task 4.1 measured
  what the global verdict costs: 22 of 41 cases now run at their declared
  budget rather than `rtol = 0` (the 19 `EXACT`-class ones lose nothing).
  Scoping the served-kernel half of `provenance` to
  `cases.PORTED_ENTRY_POINTS` would keep unported kernels bit-exact for
  the rest of Phases 04–06. Not done beside a kernel swap, and the kernel
  digest half of the verdict would still fire; belongs with the
  follow-up above.
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
- **Nothing gates the FMA map of a quadrature integrand** (Task 4.4),
  and every remaining quadrature-backed kernel inherits it — Task 4.5's
  nested ρ, Task 4.6's positron and neutrino pions, and the Phase 06
  mediator spectra. A `hazma._core` test-surface probe over a kernel
  module would fix it, but it would widen
  `cases._CORE_TEST_ONLY_MODULES`, which Task 3.2 warned against doing to
  quiet a check. **Task 4.5 is the right place to decide** whether the
  machinery is worth building once for the rest of the project.
- **Does the charged pion's forward-cone defect reach `_photon/_rho` and
  the mediator spectra?** (Task 4.4.) The ρ quadratures over the charged
  pion, so almost certainly — but whether the outer integral compounds or
  smears the loss is unmeasured. Recorded on
  [`../../../docs/followups/todo/charged-pion-photon-spectrum-misses-the-forward-cone.md`](../../../docs/followups/todo/charged-pion-photon-spectrum-misses-the-forward-cone.md);
  Task 4.5 is positioned to answer it.

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).

## Handoff to Next Task

**Phases 00–03 are closed (2026-08-06, 08-08, 08-09, 08-11), and Phase 04
is in progress — Task 4.1 landed 2026-08-11, Task 4.2 on 2026-08-12,
Task 4.3 on 2026-08-16 and Task 4.4 on 2026-08-17.** The next work is
Task 4.5 onward and/or Phase 05; they share no files and may run in
parallel.

**The parity corpus has now left bit-equality mode, permanently.**
Task 4.1's swap did it, and so did its `.pyx` edit — the kernel digest
moved too, so any task touching a surviving `.pyx` would have. What that
costs is measured rather than feared: 19 of the 41 cases are `EXACT` class
and still run at `rtol = 0`; the other 22 run at their declared budget.
The ill-conditioned-points repair **did not** land first, and the
consequence is that corpus *regeneration* is closed — see Open Questions.
**Task 4.2 met the one affected block in Phase 04 (`spectra.photon.eta`)
and waived it on evidence** (bit-equal at 336,000 points); the other five
are scalar cross sections, so read that follow-up before Phase 05 rather
than before Tasks 4.5–4.6. **Task 4.3 then hit the same class from the
other side** — a `β = 1.4e-6` probe where a two-ulp special-function
difference arrives amplified by `5/β`, fixed at the source rather than by
widening a budget. **Task 4.4 did not hit it at all**, and the reason is
worth carrying: the pion's `rest_plus_eps` boost enters as a Jacobian and
a Doppler factor rather than a `1/β` prefactor, so the prediction that
"the `rest_plus_eps` blocks will be the loud ones again" was wrong there.
Re-derive it for the ρ rather than inheriting it.

**For the next agent starting any task in this project:**

1. Read `../PLAN.md` end-to-end, then this file, then the closed phases'
   learnings (below), then the active phase's `phase-XX/README.md`.
2. Load the reference file(s) the phase's Prerequisites name — the
   references replace re-reading the Cython audit.
3. Check Open Questions above. No ADR sign-off is outstanding — all
   three project ADRs are Accepted, so no phase carries a decision gate.

**Currently safe to assume:**

- The dead-code map and entry-point inventory in
  [`../references/cython-inventory.md`](../references/cython-inventory.md)
  were verified against 2.1.0 (Aug 2026) and the file declares itself a
  snapshot. **Every row of its dead-code table is now done.** Read that
  file for the **live surface** and the cimport DAG, which Phases 04–06
  still need; read its headline counts as history.
- **15 `.pyx` and 12 `.pxd` after Task 4.2**, unchanged by Tasks 4.3 and
  4.4 — both were capi survivors, so each lost only its `def`s and its
  `.pyi`. Zero C++. Re-derive with the clean-then-rebuild recipe rather
  than quoting this; a stale `.so` makes a wrong list look right.
- **`hazma._core` now serves eleven kernels**:
  `positron.dnde_positron_muon` (Task 4.1), the seven
  `photon.dnde_photon_*` tabulated meson spectra (4.2),
  `photon.dnde_photon_muon` (4.3) and
  `photon.dnde_photon_{charged,neutral}_pion` (4.4), each called by its
  wrapper in `hazma/spectra/_{positron,photon}/__init__.py`. Three are
  bit-equal to `cdef`s the still-Cython modules cimport — the positron
  muon at 126,182 points, the photon muon at 144,000 and the neutral pion
  at 9,000, all 0 mismatches — and the charged pion agrees to 6.5e-15 over
  8,000 points, which is as close as two independent adaptive quadratures
  get. So Tasks 4.5 and 4.6 each have verified Rust dependencies to call
  natively.
- **Three test-module shapes, and the choice is not stylistic.**
  `test/test_core_positron_muon.py` / `test_core_photon_muon.py` for a
  kernel whose twin survives *and* admits bit-equality;
  `test/test_core_photon_tables.py` for one whose twin does **not**
  survive the PR (no Cython oracle — an independent reference built from
  the shipped data instead); and, new in Task 4.4,
  `test/test_core_photon_pion.py` for a kernel with **no bit-equality mode
  on any platform**, because the port replaces scipy's QUADPACK. That
  module carries two oracle classes at two standards in one file, with the
  reasoning in its docstring. **Not** `test/test_core_dispatch.py` — see
  Decisions.
- **`crate::quad` is proven on a live shape** (Task 4.4). Copy the call
  site's `epsabs`/`epsrel`/`points` from the `.pyx` verbatim into a
  `const QuadOpts`; the twelve sites use five different combinations and
  two reach scipy's defaults by passing no keyword. `points=[-1, 1]` on
  `[-1, 1]` survives scipy's filter as *no* break point, so `qagpe` runs
  over an empty list — Task 3.3 predicted it and Task 4.4 confirmed it.
  Its `Err` arm depends only on the options, never on the integrand, so it
  is unreachable for a `const` opts value; return `NaN` there rather than
  panicking, and assert the unreachability with a `cargo` test.
- **Two things no gate can see, for any quadrature-backed kernel**
  (Task 4.4, measured): a single unfused FMA inside the integrand, and a
  one-ulp constant that multiplies it. Neither survives the integration,
  so neither the corpus nor a per-kernel oracle catches them. Read the
  disassembly; do not read a green suite as confirmation of an FMA map.
- **`hazma_core::constants` exists and is bit-equal to the Cython**
  (Task 3.1). Name the table the `.pyx` `include`s — `pdg` for everything
  under `hazma/spectra/**`, `legacy` for the four mediator spectrum
  extensions — **except** `derived::photon_pion`, which legitimately reads
  both. Task 4.4 was the first kernel to consume that mixed module and it
  held: swapping the legacy `ENG_MU_PIRF` for the PDG-consistent value
  fails 10 per-kernel tests and 5 corpus blocks.
- **Task 3.5 is done, so the dispatch and error contract is settled** —
  three helpers over one classification, and a Phase 04–06 wrapper writes
  `dispatch::map_unary(x, "Photon energies", kernel)` and inherits every
  message, return type and edge case. The rule that decided every
  divergence: **each exception the Cython raises explicitly keeps its
  type; only its `assert`s change type** (`../rules.md` rule 9).
- **`test/test_core_dispatch.py`'s spectra oracle is now `_photon/_rho`**
  (moved by Task 4.4 from `_photon/_pion`, which Task 4.3 had moved it to
  from `_photon/_muon`). **Task 4.5 deletes that one and there is no
  fourth photon candidate** — the class docstring names the two remaining
  options. Its `TestCythonMessageParity` reads error strings out of the
  surviving `.pyx`, so a deletion that removes the last site carrying a
  wording turns its roster assertion red; that is the test asking for a
  roster update, not a defect.
- **Task 3.4 is done, so `hazma_core::{interp, boost}` exist**, both
  bit-equal to what they replace on the capturing platform. Do not touch
  the `mul_add`s and do not repair
  `boost_integrate_linear_interp`'s window coverage, however obviously
  wrong it looks — the corpus pins the wrong values and the repair is
  [its own follow-up](../../../docs/followups/todo/boost-integral-drops-last-interior-cell.md).
- **The parity corpus is the gate from here on.** `python
  test/parity/generate.py --check` verifies it; `test/parity/cases.py` is
  the single source of every entry point's call convention. Do not
  regenerate it from a tree in which any kernel runs on Rust —
  `rules.md` rule 2, enforced in code by `assert_no_rust_core`.
  `test/parity/tolerances.py` now has a `PORTED_QUAD_RTOL = 1e-12` taken
  by `spectra.photon.charged_pion` alone; the two unported `QUAD` cases
  keep the 1e-8 opening figure until Task 4.6 measures them, and `NESTED`
  is untouched with Task 4.5 owning the same decision for it.
- **The suites are merged and green on the capturing platform**: bare
  `pytest -q` → **1755 passed / 15 skipped** as of Task 4.4 (2026-08-17),
  from 1682/15 at Task 4.3, 1628/15 at 4.2, 1378/13 at 3.5, 1063/13 at
  Phase 02 close and 1006/13 at Phase 01 close. Re-derive rather than
  quoting; the historical series is in [phase-01/README.md](phase-01/README.md),
  [phase-02/README.md](phase-02/README.md) and
  [phase-04/README.md](phase-04/README.md).
- **`test/test_theory_aggregation.py` is the model-layer gate the corpus
  cannot be** (Task 1.4): identities over `hazma/theory/`'s pure-Python
  aggregation, no golden data, and the only numerical gate in the repo
  that is not scoped to the capturing platform. **Phases 04–06 run it
  either side of every kernel swap** — `69 passed` as of Task 4.4.
- **Off macOS the corpus does not reproduce**, so CI runs
  `pytest --ignore=test/parity` on every entry except macOS. That is a
  corpus defect and it is tracked in
  [`../../../docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md`](../../../docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md)
  — **read it before Phase 05**.
- **A `.rs` edit needs `pip install -e .`, not `cargo build`** (Task 2.2).
  Iterate with
  `cargo test --manifest-path rust/Cargo.toml --no-default-features`,
  reinstall before quoting any pytest or parity number, and confirm with
  `python -c "import hazma._core; print(hazma._core.__file__)"`.
- **The build entry point is `setup.py`.** `_build.py` was deleted in
  `7a817f9` (2026-08-02) and Task 0.4 swept the durable docs that named it.
- **The sdist and wheel both build, and the sdist installs and runs** in a
  fresh venv from outside the repo (recipe in Task 0.4's note; reuse it in
  Phase 07).
- **`hazma.gamma_ray` is gone, docs and all** (Task 0.5 swept, Task 0.2
  deleted). Surviving mentions are dated records. The settled replacement
  wording for the Phase 07 aggregate: `gamma_ray_decay` →
  `hazma.spectra.dnde_photon`, `gamma_ray_fsr` →
  `hazma.spectra.dnde_photon_fsr`, **neither a drop-in**.
- The legacy constants table lives at
  `hazma/_utils/legacy_parameters.pxd` and is now its **only** copy.
  `hazma.utils` is the only home for `cross_section_prefactor` and
  `minkowski_dot`.

**Currently risky / unknown:**

- **Six blocked defects now share one eventual corpus regeneration** —
  the positron normalization (4.1), the boost integral (3.4), the η′ line
  weight and the φ line energies (both 4.2), the muon photon spectrum's
  rest-frame endpoint (4.3), and the charged pion's lost forward cone
  (4.4). Do not "fix" any of them in passing; each fails the gate that
  governs the remaining swaps. **Worth telling the maintainer separately
  from this project's schedule** — several affect published numbers today,
  and the sixth affects the *shape* of a spectrum rather than a total,
  which is the kind a limit calculation notices.
- **Nested-ρ drift (Task 4.5) is the project's numerical stress test.**
  Measure before adjusting any budget: attributing a difference to a
  single term turned a proposed 300x widening into a 60-line fix in
  Task 4.3, and turned "the quadrature diverges" into "the quadrature
  diverges exactly where scipy does, and agrees with it there" in 4.4.
- **Phase 05 has to name the cross sections' `quantity` wording.** They
  carry no dispatch message at all today, so the port invents it and it is
  user-visible from the first swap. `"Center-of-mass energies"` is the
  placeholder `test/test_core_dispatch.py` uses.
- **Two Task 1.4 follow-ups ripen inside this project.** The
  [`MASS_E` `nan`](../../../docs/followups/todo/positron-spectrum-nan-at-legacy-electron-mass.md)
  before Phases 05/06, and the
  [scalar-energy contract](../../../docs/followups/todo/model-spectra-reject-scalar-energies.md)
  during 04–06 — Task 3.5 settled the compiled half; what is left is pure
  Python.
- **`release.yml` has no pull-request trigger**, so any future change to
  it needs its own dispatch to be measured at all
  (`../../../docs/agents/lessons.md`
  `[unrun-workflow-cannot-close-a-criterion]`). Phase 07 Task 7.1 rewrites
  it for maturin and inherits that.
