# Archived working memory: Findings (Phases 00–05)

**Project:** cython-to-rust
**Moved:** 2026-08-21, from [`README.md`](README.md), in two passes —
Phases 00–04 at Phase 04 close, Phase 05's two bullets at Phase 05 close
**Source lines:** 58–892 of that file at commit `c57ce4f` (Phases 00–04);
lines 72–91 at commit `cbe5555` (Phase 05)

This file is a verbatim archive. Nothing below the rule was edited,
summarised or reordered when it moved, and it sits in the same
directory as the README so every relative link in the moved text
still resolves. Reproduce the move with

```sh
git show c57ce4f:projects/cython-to-rust/task-notes/README.md | sed -n '58,892p'
```

The phase learnings under [`../learnings/`](../learnings/)
condense this material and are what a new task reads first — see
[ADR-0002](../../../docs/adrs/ADR-0002-read-phase-learnings-not-closed-task-notes.md).
Come here when a learnings entry, a task note or a citation sends
you to the original entry. Later phase-close sweeps append the
closed phase's entries below, verbatim, under a
`### Swept YYYY-MM-DD (Phase XX)` heading.

---

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
  at exactly `e_cm = 2·mx`. The scalar-mediator siblings do not. Pinned
  as `nan` plus a manifest `raises` record, which `test_parity.py`
  **replays rather than skips**. **Task 5.1 ported them as-is per
  rules.md rule 1 and measured the mechanism, which is not quite what
  this bullet said before it:** the exponent is `** 1.5`, and Cython 3's
  default `cpow` semantics compile the *whole enclosing expression* in
  `double _Complex` — not just the power. At `e_cm = 2·mx` the
  denominator is exactly zero, compiler-rt's `__divdc3` takes C99 Annex
  G's zero-denominator recovery and returns `(±inf, nan)`, and
  `__Pyx_SoftComplexToDouble` rejects the non-zero imaginary part. The
  port reproduces the type and not the wording, and needed a new
  dispatch shape (`map_unary_try`) to do it. Repair is filed as a
  separate declared change:
  [the `2 m_x` raise](../../../docs/followups/todo/vector-cross-sections-raise-at-the-two-mx-threshold.md).
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
  `hazma/spectra/_neutrino/_muon.pyx:205` (at `ed1fa20`; Task 4.6 deletes
  the file) says "Photon energies". Two of
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
  `test/parity` already does, through
  `tolerances.Provenance.same_platform`); loosening to a tolerance is
  wrong, because the worst relative
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
  and `:58`/`:114` multiply by it — line numbers at `ed1fa20`, since
  Task 4.6 deletes that file; `rust/src/kernels/neutrino_muon.rs` carries
  the same constant and the same multiplication now. It propagates to
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
  reader to assume a gate exists. **Task 4.5 narrowed this**: arithmetic
  in the integration *limits* can be lifted into its own `fn` and pinned,
  and only arithmetic genuinely *inside* an integrand stays ungated. See
  Open Questions.
- **Phase 03 Task 3.3's divergent-regime obligation is discharged for the
  first `qagp` consumer, and the answer is reassuring** (Task 4.4). The
  charged pion's quadrature does leave `ier = 0` — first at `E_π = 4e4`
  MeV (`γ_π ≈ 290`), 40 GeV against a sub-GeV library and two decades
  above the corpus ceiling. But over an 11 × 8 grid reaching `E_π = 1e5`
  the port's `ier` **equals the flag scipy raises on the Cython twin at
  all 88 points**, including both `ier = 4` entries and the non-monotonic
  pattern between them. **The flags agree; the values in that regime do
  not have to, and two CI rounds on PR #68 proved they do not.** The
  separation there is 2.8e-11 on macOS/arm64, **6.2998e-10** on
  Linux/glibc at one point, and **3.0552e-08** at another — each time
  bit-identical across py3.10–3.14, so a toolchain property rather than
  noise, and each time revealed only by asserting a bound the numerics do
  not support. Asserting 1e-10 failed; raising it to 1e-8 moved the
  failure to the next-worst point. **The lesson is the one
  [`parity-corpus-pins-ill-conditioned-points.md`](../../../docs/followups/done/parity-corpus-pins-ill-conditioned-points.md)
  already warns about, met from a new direction: widening a budget until
  it passes is how a gate becomes vacuous — the fix was to stop asserting
  a tolerance the regime cannot support.** The test now partitions its
  grid by *scipy's own convergence verdict* and holds each half to what is
  true of it: 1e-12 where QUADPACK converged (measured worst 2.22e-16 —
  one ulp, because the two subdivide identically there), and sign plus a
  factor of two where it did not. **Every quadrature-backed kernel from
  here on should partition rather than pick one tolerance.**
- **A mutation harness that reverts with `git checkout --` cannot revert a
  file git has never seen** (Task 4.4). The new kernel module was
  untracked, the restore step errored, the driver did not check, and five
  mutations accumulated while being read as five independent
  measurements — Task 3.3's `[mutation-harness-poisons-its-own-baseline]`
  in a new disguise, with the same tell (implausibly uniform failure
  counts). Snapshot outside the tree, `cmp` before each mutation, verify
  the restore.
- **Part of the corpus was pinning numbers that are simply wrong, and
  only a higher-precision oracle could say which** (parity-corpus
  follow-up, 2026-08-18). Four scalar elastic cross sections —
  `sigma_xl_to_xl`, `sigma_xpi_to_xpi`, `sigma_xpi0_to_xpi0`,
  `sigma_xg_to_xg` — evaluate `P·atan(u) − P·atan(v)`, which cancels
  completely near `e_cm = 2 mx` and throughout `closed_resonance`. At
  `mx = 300`, muon target, the library returns `-1.504081e-02` where the
  formula is worth `+6.198557e-07`, and the `-inf` at `e_cm = 2 mx` is a
  removable 0/0 rather than a pole. **The follow-up's own proposed
  detector — nudge the inputs by an ulp — cannot find these**: it
  measures conditioning, and the points are well conditioned; what is
  broken is the stability of the algorithm. Neither can "which platforms
  disagree": the visible set is not stable even between x86_64 libm code
  paths. `test/parity/reference.py` (the same closed forms at 60 digits)
  is what settled it. **Reusable pattern**: when a pinned number comes
  under suspicion, a verbatim `mpmath` copy of the `.pyx` body answers it
  in an afternoon and needs no build.
- **`atol = 0` everywhere assumed below-threshold regions return exactly
  `0.0`, which is false of the quadrature-backed kernels** (same
  follow-up). `spectra.positron.charged_pion` at `E = m_e` lands on
  exactly zero on macOS/arm64 and x86_64 and on 2.6e-13 on
  Linux/aarch64 — an *infinite* relative error against a stored zero.
  Any future budget design should assume "returns exactly zero" is a
  property of one libm, not of the mathematics.

- **Phase 04 is closed (2026-08-20)**, and its distillation is
  [phase-04-spectra-kernels.md](../learnings/phase-04-spectra-kernels.md)
  — read that rather than the six task notes. The three findings that
  outlive the phase and reach Phases 05–07:
  - **The phase found seven live 2.1.0 numerical defects, none of them by
    porting.** Every one came from writing a statement the original never
    made — an analytic normalization check, a sibling-to-sibling diff, a
    rest-frame limit, a forward-cone argument, a continuum subtraction.
    Phases 05 and 06 should budget for the same.
  - **Every task's numerical prediction was wrong, in a different
    direction each time.** Nothing in six tasks' history supports
    predicting a kernel's drift from its shape. Re-derive.
  - **Interrogate a mutation survivor; do not accept one.** It is either
    unobservable *by construction* — say so in the test, with the
    argument — or it is a seam that needs lifting out. Task 4.6's
    surviving γ spelling was a real error sitting **29x outside the
    corpus's own budget** at energies the corpus does not sample, and only
    a lifted, bit-pinned `fn` caught it.

---

## Phase 05 (moved 2026-08-21 at Phase 05 close)

- **The scalar mediator's threshold behavior is explained, not just
  observed** (Task 5.2, updating
  [`history-findings.md`](history-findings.md)'s "Two live entry points
  raise" entry, which said only that the scalar siblings do not raise).
  The scalar module *does* compile one expression through
  `double _Complex` — `__sigma_xx_to_s_to_ff` — but its vanishing root
  sits in the **numerator**, so `__divdc3`'s zero-denominator recovery
  is never reached. Two consequences for Phase 06: the archived claim
  that this module has no `** 1.5` is **wrong**, and
  `grep -c SoftComplexToDouble` on the generated C — not a read of the
  `.pyx` — is what settles the question for the four modules left.
- **Both `thermal_cross_section` divergences above `x = 300` were
  reproduced, not unified** (Tasks 5.1 and 5.2, closing
  [`history-findings.md`](history-findings.md)'s "Phase 05 must
  reproduce both or declare the unification"). The scalar returns `0.0`,
  the vector clips to `xnew = 300`; they live in separate Rust modules
  with the divergence documented on each, so no published number moved.
  A shared Rust helper is the obvious design and is the one that would
  have moved them — do not revisit it in Phase 06 without declaring the
  change.
