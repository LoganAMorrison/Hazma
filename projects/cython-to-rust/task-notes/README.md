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
| 03 | Numerics foundation | [phase-03-numerics-foundation.md](../phases/phase-03-numerics-foundation.md) | [phase-03/README.md](phase-03/README.md) | In Progress — 3.1 and 3.2 done (2026-08-09), 3.3–3.5 open |
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
  the design instead of the code. Task 3.5 decides each one.
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
  behaviors were not in the prose** (Task 2.3).
  `test/test_core_dispatch.py` holds every branch of
  `dispatch::map_unary` through `hazma._core.roundtrip`: (a) **rank is
  checked before dtype**, so a 2-D int64 array reports the dimension
  message; (b) **a 0-d array still enforces dtype** where a Python `int`
  does not, because the 0-d path lives inside the array branch behind the
  typed view; (c) **non-`float` NumPy scalars are accepted**
  (`np.float32`, `np.int64`, `np.uint8`, `np.bool_`) via the
  `extract::<f64>` arm. **That module is the template Phases 04–06
  copy** — swap the kernel and the `QUANTITY` wording, keep every test,
  add the numerical tests beside rather than merged in. Two side facts
  worth carrying: bit patterns survive intact (NaN payload included), so
  the module argues about no tolerances; and a "fresh" array from the
  `numpy` crate has `owndata == False` with a `PySliceContainer` base, so
  **non-aliasing is the assertable property, never `owndata`**.
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
  passed, at `1142 passed, 13 skipped` (+54 on Task 3.1's 1088, all of
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
  header, 7 unit tests) and `rust/src/special_probe.rs`
  (registration-only `hazma._core.special`); `rust/src/lib.rs` admits
  both; `rust/Cargo.toml` / `Cargo.lock` gain `spec_math = "0.1.6"`. New
  `test/test_core_special.py` (53 tests).
  `test/parity/{cases.py,test_parity.py,README.md}` gain
  `_CORE_TEST_ONLY_MODULES` and its importer guard test.
  `hazma/_core.pyi` gains a comment — the only change under `hazma/`,
  and non-executable. **Two canonical patches:** the phase file's Task
  3.2 block gained three "criteria added during execution" bullets, and
  [`../references/numerics-replacements.md`](../references/numerics-replacements.md)
  gained the measured block correcting its claim that scipy's `kn` is a
  cephes wrapper. Full list in
  [phase-03/README.md](phase-03/README.md).

## Verification

- Scaffolding PR: `scripts/agents/preflight.sh` (repo gate; no code
  changes).
- **Phase 03 Task 3.2 state (2026-08-09):** bare `pytest -q` →
  `1142 passed, 13 skipped` on the capturing environment, parity suite
  included and in bit-equality mode (skip count unchanged at 13);
  `pytest test/test_core_special.py -q` → `53 passed in 0.26s`;
  `cargo test --no-default-features` → `15 passed` (7 new), clippy and
  fmt clean; `scripts/agents/preflight.sh` RESULT: PASS. Ten mutations —
  eight against `rust/src/special.rs`, two against the corpus's
  served-kernel guard — each caught by the test whose name claimed it
  (tables in the task note). One of them, dropping the recurrence's
  order factor, passed `cargo test` on the first pass and is why the
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
- **`hazma._core` exists, and nothing under `hazma/` calls it** (Task
  2.1; still true after Task 3.2). The crate lives in `rust/`,
  `uv pip install -e .` builds Cython and Rust in one pass, and the tree
  carries 21 `.so`: the 20 Cython extensions plus `hazma/_core.abi3.so`.
  Its public surface is `roundtrip` — a plumbing probe — plus the
  `special` submodule Task 3.2 added, which is a test surface reachable
  only from `test/test_core_special.py`. Neither is a ported kernel, and
  `cases.rust_core_kernels()` is the predicate that says so.
  `dispatch::map_unary` is the single
  implementation of the entry-point dispatch contract — Phases 03–06 call
  it rather than touching PyO3 inside a kernel (`../rules.md` Rust rule
  3). The three cargo gates are `cargo fmt --check`,
  `cargo clippy --all-targets -- -D warnings` and
  `cargo test --no-default-features`; the last one's flag is load-bearing,
  since `extension-module` leaves the test harness unlinkable. **As of
  Task 2.2 you no longer have to remember to run them**: they are gates
  4–6 of `scripts/agents/preflight.sh` (ahead of pytest, since they cost
  seconds and it costs minutes) and CI's `rust` job.
- **A `.rs` edit needs `pip install -e .`, not `cargo build`** (Task
  2.2). Cargo works out of `rust/target/`, which nothing Python imports;
  the editable install is what re-links the crate as
  `hazma/_core.abi3.so`. Iterate with
  `cargo test --manifest-path rust/Cargo.toml --no-default-features`,
  reinstall before quoting any pytest or parity number, and confirm with
  `python -c "import hazma._core; print(hazma._core.__file__)"`.
- **Phases 00, 01 and 02 are all Complete** (2026-08-06, 2026-08-08 and
  2026-08-09). Read
  [`../learnings/phase-00-dead-code-purge.md`](../learnings/phase-00-dead-code-purge.md),
  [`../learnings/phase-01-parity-corpus.md`](../learnings/phase-01-parity-corpus.md)
  and
  [`../learnings/phase-02-rust-scaffold.md`](../learnings/phase-02-rust-scaffold.md)
  rather than their twelve task notes — the learnings are the
  distillation, the notes are history. **Phase 03 is in progress:
  Tasks 3.1 and 3.2 both landed 2026-08-09, so the next task is 3.3 or
  3.5 (dependency-free) or 3.4 (unblocked by 3.1).** No phase carries a
  decision gate.
- **`hazma_core::constants` exists and is bit-equal to the Cython**
  (Task 3.1). `constants::pdg` is `hazma/_utils/constants.pxd` (151
  values), `constants::legacy` is `hazma/_utils/legacy_parameters.pxd`
  (48), and `constants::derived::<source_pyx>` holds the module-local
  `DEF`s of the five `.pyx` that declare any (25). When porting a
  kernel, name the table its `.pyx` `include`s — `pdg` for everything
  under `hazma/spectra/**`, `legacy` for the four mediator spectrum
  extensions — **except** `derived::photon_pion`, which legitimately
  reads both (see Findings). Two gates hold this:
  `test/test_core_constants.py` (25 tests, 0.03s, needs no build, runs
  on every platform) compares the Rust and Cython *sources* bit-for-bit,
  and five `cargo test` units check the compiled side. Both die with the
  Cython, and each says in its own text what to delete when a `.pyx`
  goes.
- **`test/test_core_dispatch.py` is the template every Phase 04–06 kernel
  swap copies** (Task 2.3; 54 tests, 0.27s, platform-independent). It
  pins every branch of `dispatch::map_unary` through
  `hazma._core.roundtrip`. Copying it means: swap `roundtrip` for the
  kernel and `QUANTITY` for the wording that kernel passes to
  `map_unary`, keep every test, and add the kernel's numerical tests
  *beside* them rather than merged in. The instructions are in the module
  docstring so they travel with the file. It deliberately re-pins no
  value against Cython — the corpus does that, at bit-equality, across
  all 41 entry points.
- **The parity corpus is the gate from here on.** `python
  test/parity/generate.py --check` verifies it (still, with `_core`
  present); `test/parity/cases.py` is the single source of every entry
  point's call convention. Do not regenerate it from a tree in which any
  kernel runs on Rust — rules.md rule 2, enforced in code by
  `assert_no_rust_core`, which since Task 2.1 tests whether `hazma._core`
  *serves* a kernel rather than whether it exists. Until the first
  Phase 04 swap the corpus is therefore still repairable and the runner
  is still in bit-equality mode; after it, neither.
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
  `pytest -q` → **1142 passed / 13 skipped** as of Task 3.2
  (2026-08-09), from 1088/13 at Task 3.1, 1063/13 at Phase 02 close,
  1006/13 at Phase 01 close and 935/30 at Task 1.3. The Phase 02 deltas:
  +3 (Task 2.1's scaffold checks), 0 (Task 2.2, byte-identical, which is
  what showed no test outcome moved), +54 (Task 2.3's plumbing module);
  Phase 03 so far: +25 (Task 3.1's constants), +54 (Task 3.2's specfun
  module plus one corpus guard). **The skip count has not moved since
  Phase 01 closed**, and that is the tell: forcing budget mode drops one
  test to a skip rather than failing, so an unchanged 13 is how the
  parity suite reports it is still in bit-equality mode. Re-derive rather
  than quoting; the historical series is in
  [`phase-01/README.md`](phase-01/README.md) and
  [`phase-02/README.md`](phase-02/README.md).
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

- ~~**CI has no Rust toolchain step**~~ — **added by Task 2.2
  (2026-08-08)**, in both workflows, along with the `rust` job that runs
  the cargo gates. The measurement that made it urgent still stands as
  history: all seven checks passed first try on PR #55 with no toolchain
  step at all, on ubuntu py3.10–3.14 plus macos py3.14, because the
  runner images happen to ship cargo. **`release.yml` was dispatched and
  is green too** (PR #56 review round 1, run 31297673951): both
  `build-wheels` jobs plus `build-sdist` succeeded, `publish` skipped on
  its `github.event_name == 'release'` gate, and the new assertion step
  reported `5 wheel(s) carry hazma/_core.abi3.so` per platform — 10
  CPython-tagged wheels, `cp310`–`cp314` × {macOS arm64,
  manylinux_2_28 x86_64}. That job has **no pull-request trigger**, so
  any future change to it needs its own dispatch to be measured at all
  (`../../../docs/agents/lessons.md`
  `[unrun-workflow-cannot-close-a-criterion]`). Phase 07 Task 7.1
  rewrites it for maturin and inherits that.
- ~~`spec_math`'s `li2` argument convention vs scipy's `spence` is
  unverified — Task 3.2 pins it before anything depends on it.~~ —
  **pinned in Task 3.2 (2026-08-09): the same convention, `Li₂(1−z)`,
  because `li2`'s body is `cephes64::spence`.** What the same check
  found instead was a divergence nobody had flagged: **`scipy.special.kn`
  is not cephes**, and the faithful cephes `kn` misses it by 5.1e-9 in
  the live domain. See Findings; the live obligation is that Phase 05
  must not swap `crate::special::bessel_kn`'s recurrence back to
  `spec_math`'s routine.
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
