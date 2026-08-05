# Task 0.3: Delete superseded per-particle kernels and helpers

**Date:** 2026-08-04
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-00-dead-code-purge.md` (Task 0.3);
`../../rules.md` (Process rule 1 — verify-before-delete; Constants rule 1);
`../../PLAN.md` (Scope, Numerical impact)
**Related ADRs:** none (ADR-0003 gates Tasks 0.2/0.5, not this task)
**Depends On:** Task 0.1 (constants header relocated out of `_decay/`)

## Objective

Delete the superseded per-particle Cython packages (`_decay/`, `_positron/`,
`_neutrino/`), their legacy double-underscore Python shims, the two
zero-importer `field_theory_helper_functions` modules, the uncompilable
`spectra/_positron/_kaon.pyx`, the buggy dead half of `_utils/boost.pyx`,
and the already-ignored `test/decay/` — after re-verifying at execution
time that nothing live imports them, and after giving the two live
`common_functions` symbols a pure-Python home.

## Exit Criteria

Copied from the phase file's Task 0.3 `**Exit criteria:**` block, quoted
in its **patched** form — this task added the declared-drift note and the
forced-config-edit criterion (see Plan Impact); the criteria below are
what was actually worked to.

- [x] Deleted: `hazma/_positron/`, `hazma/_neutrino/`, `hazma/_decay/`
      (incl. `interpolation_data/` and backups), `hazma/__decay.py`,
      `hazma/__positron_spectra.py`, `hazma/__neutrino_spectra.py`,
      `hazma/spectra/_positron/_kaon.pyx`,
      `hazma/field_theory_helper_functions/` (both modules), the dead
      ~165-line half of `hazma/_utils/boost.pyx`, and `test/decay/`
      (already `collect_ignore`d and importing a nonexistent module).
- [x] `cross_section_prefactor` callers use `hazma.utils`;
      `minkowski_dot` given a pure-Python home and
      `hazma/experimental/axial_vector_mediator/avm_msqrd.py` repointed.
      The `cross_section_prefactor` swap is a **declared** numerical
      change near threshold — recorded in `../README.md` "Numerical
      impact so far", not absorbed silently.
- [x] The three config files that name the deleted sources
      (`setup.py`'s three extension groups, `test/conftest.py`'s
      collection list, the `_decay` packaging entries in
      `pyproject.toml` / `MANIFEST.in`) updated here, because the build
      and the test collection break the moment the sources go.
- [x] Import smoke (`hazma.theory`, `hazma.limits`, `hazma.cmb`,
      `hazma.pbh`, both mediators, `hazma.spectra._photon._muon`) passes.

## Inputs Reviewed

- `../../PLAN.md` (Scope, Numerical impact, Phases table).
- `../../phases/phase-00-dead-code-purge.md` (Goal, Task 0.3, phase Exit
  Criteria).
- `../README.md` (project working memory) and `README.md` (phase working
  memory) — including Task 0.1's handoff.
- `../../rules.md` — "Process" rule 1 (verify-before-delete), "Constants"
  rule 1 (bit-parity), "Parity discipline" rule 3 (declare every shift).
- `../../references/cython-inventory.md` — the dead-code map rows for each
  deletion target, plus the "Bugs" section.
- `../../../../docs/agents/lessons.md` — checked the diff against each
  class; `[derived-count-not-rederived]` and `[stale-ci-capability-claim]`
  both applied and are honored in the sweep block below.
- `setup.py`, `pyproject.toml`, `MANIFEST.in`, `test/conftest.py` — the
  build/packaging/collection config that names the deleted paths.
- `hazma/utils.py`, `hazma/field_theory_helper_functions/common_functions.pyx`
  — to establish that the pure-Python twins are algebraically identical.

## Findings

- **Three config files break the moment the sources are deleted, so this
  task had to touch them.** `setup.py` declares live `Extension`s for
  `_positron` (3), `_neutrino` (2), and `field_theory_helper_functions`
  (2) — `pip install -e .` fails immediately if their `.pyx` disappear.
  `test/conftest.py` calls `THIS_DIR.joinpath("decay").iterdir()`
  unconditionally, which raises `FileNotFoundError` at *collection* once
  `test/decay/` is gone — the whole suite would error out, not just skip.
  Both are forced, not scope creep. Task 0.4 still owns the remaining
  reconciliation (the `_gamma_ray` / `_phase_space` groups Task 0.2
  deletes, and the final survivor count).
- **`cross_section_prefactor` is algebraically identical but *not*
  numerically identical.** The Cython form used the factored
  `sqrt((m1-m2-cme)(m1+m2-cme)(m1-m2+cme)(m1+m2+cme))`; `hazma.utils`
  builds the same quantity from `kallen_lambda`, which cancels to zero at
  threshold. Away from threshold they agree to roundoff (≤4.8e-15 at
  `cme ≥ 1.1 ×` threshold); within 1e-7 of threshold the difference
  reaches 2.1e-7 relative. Measured, declared, and filed as
  [`docs/followups/todo/cross-section-prefactor-threshold-cancellation.md`](../../../../docs/followups/todo/cross-section-prefactor-threshold-cancellation.md)
  — full table in Verification.
- **`minkowski_dot` is also not bit-identical**, for a different reason:
  the C compiler contracts `a*b - c*d` into an FMA, the Python version
  cannot. ≤2.7e-14 relative over 1998 random four-vector pairs. Its only
  in-library consumer is `hazma/experimental/`, which
  `docs/versioning.md` explicitly excludes from the public surface.
- **`hazma/experimental/axial_vector_mediator/__init__.py` is broken on
  `origin/master`** — it does `from hazma.theory import Theory`, and
  `hazma/theory/__init__.py` exports `TheoryAnn` / `TheoryDec`. Verified
  against the trunk (`git show origin/master:...`). `avm_msqrd.py` itself
  imports and evaluates fine with the repointed `minkowski_dot`; the
  package `__init__` is pre-existing breakage, out of scope here.
- **`git rm -r` leaves the directory behind when it holds untracked
  `__pycache__`,** and an empty directory on `sys.path` is an importable
  *namespace package*. Right after `git rm`, `import
  hazma.field_theory_helper_functions` still succeeded. Any
  verify-after-delete check that only greps the index will miss this —
  `rm -rf` the directories and re-run the negative import check.
- **`boost_jac` and `boost_eng` are unused but stay.** They have zero
  cimporters repo-wide, but unlike the three functions this task deletes
  they *are* declared in `_utils/boost.pxd`, i.e. part of the module's
  published C-level API. The inventory names only the three; deleting the
  other two is a Phase 06 Task 6.4 question, recorded there rather than
  taken here.
- **The `hazma.decay` / `hazma.positron_spectra` imports scattered
  through `notebooks/` were already dead before this task.** Those names
  only ever existed via the alias block in `hazma/__init__.py`, which is
  commented out on `origin/master`; `hazma/decay.py` does not exist there
  either (`git show origin/master:hazma/decay.py` → `fatal: path ... does
  not exist`). They are pre-existing rot, not fallout from this task —
  see the boundary in Decisions.

## Decisions and Implementation Notes

- **`minkowski_dot`'s pure-Python home is `hazma/utils.py`**, beside
  `cross_section_prefactor` (which the same deleted module also provided)
  and `ldot` (its array-oriented generalization). It is deliberately
  **not** added to `docs/source/utils.rst`: the function is a relocation
  of an existing public name, not a new documented API, and `PLAN.md`
  Scope forbids API additions. Consumers that had it keep it; the
  documented surface is unchanged.
- **Repointed `hazma/gamma_ray.py` too**, not just `deprecated/rambo.py`.
  The exit criterion says "`cross_section_prefactor` callers use
  `hazma.utils`" and `gamma_ray.py:236` is one, even though the module is
  broken on import and ADR-0003 deletes it. Leaving it would put a
  dangling import into a file Task 0.2 has not yet been cleared to touch.
- **Deleted `docs/source/positron.rst` and `docs/source/decay.rst`.**
  Every `autofunction` in them targets `hazma.positron_spectra.*` /
  `hazma.decay.*`, i.e. exactly the two shims this task deletes; both
  files are already orphaned from `docs/source/index.rst`'s toctree, so
  nothing that ships changes. A page whose every target is gone is rot,
  not documentation. `gamma_ray.rst` and `rambo.rst` are left for Tasks
  0.2/0.5, which own those modules.
- **Notebook boundary: repoint what *this task* made dangle, leave what
  was already dangling.** Fixed the three references that this task
  broke — `hazma.__positron_spectra` in `notebooks/hazma_paper/snippets.py`
  and `field_theory_helper_functions` in
  `notebooks/dev/gamma_ray_fsr/partial_integration.py` and
  `notebooks/dev/K0_radiative_decay_4_24_18.ipynb`. Left the ~20
  pre-existing-broken `hazma.decay` / `hazma.positron_spectra` imports in
  other notebooks: they were dead before this branch (see Findings),
  `notebooks/` is excluded from every lint gate, and repointing them all
  is its own change with its own re-derived output values.
- **Recomputed, not guessed, the pinned outputs in `snippets.py`.** The
  two repointed cells carry `# array([...])` result comments. The live
  `hazma.spectra` kernels give different values than the legacy ones, so
  the comments were regenerated from the built tree rather than carried
  over — carrying them over would have shipped four wrong numbers.
- **`positron_decay`'s docstring rewritten as a removal stub.** Its
  Examples block imported `hazma.positron_spectra`; the function has
  raised `NotImplementedError` unconditionally for some time. Repointing
  the example without saying that would have documented a call that
  always fails. The new example is `hazma.spectra.dnde_positron` and was
  run before being written down (see Verification).
  **Review round 1** then caught that adding a `Raises: Always` section
  left the docstring self-contradictory — its `Returns` and `Notes` still
  described a spectrum it never produces. Both sections are now gone,
  along with two pre-existing defects they carried: the summary and
  `Returns` said "gamma ray spectrum" in the *positron* module (a
  copy-paste from `gamma_ray.py`), and `Notes` cited
  ``hazma.phase_space_generator.rambo``, a module path that has never
  existed. Swept the class: of the 43 `raise NotImplementedError` sites
  in `hazma/`, every other one is an abstract-base stub, so this was the
  only concrete public function documenting a return it cannot deliver.
  The one surviving `phase_space_generator` reference is
  `hazma/gamma_ray.py:103`, in the module ADR-0003 deletes wholesale.
- **Dropped `fmin`, `fmax`, and the whole `libc.float` cimport from
  `boost.pyx`.** They were used only by the deleted `integration_bounds`
  and `boost_integrate_linear_interp_massive`. `fmin`/`fmax` still appear
  inside the commented-out `boost_dnde` stub, which is inert.
- **Removed `hazma/__init__.py`'s commented-out legacy-shim import block
  and its three `__all__` entries** (`decay`, `neutrino_spectra`,
  `positron_spectra`). They name modules this task deletes. The rest of
  the commented block — which references `gamma_ray` and `rambo` — is
  left for Tasks 0.2/0.5.
- **Did not touch `hazma/utils.py`'s `cross_section_prefactor`
  implementation.** Switching it to the numerically better factored form
  would move `hazma.phase_space.PhaseSpace.cross_section`, a live public
  API — a declared numerical change that has nothing to do with deleting
  dead code. Filed as a follow-up with the measured magnitudes instead.

## Files Changed

### Deleted

- `hazma/_decay/` — 54 files: 29 at the top level (`.pyx`, `.pxd`,
  `.pyi`, `get_path.py`, the `_decay_muon_bak.pyx` /
  `_decay_charged_pion.pyx.bak` backups) plus 25 under
  `interpolation_data/` (21 `.dat` in `ckaon/`+`skaon/`+`lkaon/`, 3 stale
  top-level `.dat`, and `gen_ckaon_interp.py`).
- `hazma/_positron/` — 11 files (3 built modules + `parameters.pxd` +
  stubs).
- `hazma/_neutrino/` — 8 files (2 built modules + the never-built
  `neutrino.pyx`/`.pxd` they cimport + stubs).
- `hazma/field_theory_helper_functions/` — 4 files:
  `common_functions.{pyx,pxd}`, `three_body_phase_space.pyx`,
  `__init__.py`.
- `hazma/__decay.py`, `hazma/__positron_spectra.py`,
  `hazma/__neutrino_spectra.py` — the double-underscore legacy shims.
- `hazma/spectra/_positron/_kaon.pyx` — not in `setup.py`, references
  undefined names, would not compile.
- `test/decay/` — 23 files: `test_decay.py`, `generate_test_data.py`,
  and the `mu_data/` / `pi_data/` / `pi0_data/` `.npy` fixtures.
- `docs/source/positron.rst`, `docs/source/decay.rst` — see Decisions.

### Modified — library

- `hazma/_utils/boost.pyx` — deleted `integrate_linear_interp_edge`,
  `integration_bounds`, and `boost_integrate_linear_interp_massive`;
  dropped the now-unused `fmin` / `fmax` / `DBL_EPSILON` cimports.
  461 → 241 lines (`git diff --numstat` → `1 221`).
- `hazma/utils.py` — added `minkowski_dot` (pure-Python home).
- `hazma/deprecated/rambo.py:25-31` — `cross_section_prefactor` now from
  `hazma.utils`.
- `hazma/gamma_ray.py:16` — same repointing.
- `hazma/experimental/axial_vector_mediator/avm_msqrd.py:10` —
  `minkowski_dot` now from `hazma.utils`.
- `hazma/spectra/_positron/__init__.py` — dropped the commented-out
  `hazma._positron.positron_decay` import; `positron_decay`'s docstring
  gained a `Raises` section and a runnable Example.
- `hazma/__init__.py` — dropped the commented legacy-shim import block
  and its three `__all__` entries.
- Lint-only, no behavior: `isort` applied to `hazma/gamma_ray.py`,
  `hazma/deprecated/rambo.py`, `hazma/spectra/_positron/__init__.py`,
  `test/conftest.py`; `black` to `hazma/gamma_ray.py` and
  `hazma/deprecated/rambo.py` (16 lines: a module docstring and a
  `warnings.warn` call); `test/conftest.py`'s unused
  `from importlib.resources import path` removed. Every one of these is
  a file this task already edits — see the preflight table in
  Verification for why they were taken rather than deferred.

### Modified — build, packaging, collection

- `setup.py` — removed the `field_theory_helper_functions`, `_positron`,
  and `_neutrino` extension groups (forced; see Findings).
- `test/conftest.py` — removed the `test/decay/` `iterdir()` (forced).
- `pyproject.toml` — removed the three
  `hazma._decay.interpolation_data.*` package-data entries.
- `MANIFEST.in` — removed the three matching `include` lines.

### Added

- `test/test_utils.py` — 16 pinned tests for `minkowski_dot` and
  `cross_section_prefactor`.
- `docs/followups/todo/cross-section-prefactor-threshold-cancellation.md`
  (+ index row in `docs/followups/README.md`).

### Modified — durable docs

- `docs/source/usage.rst:476-480` — `hazma.positron_spectra` /
  `hazma.decay` imports repointed to `hazma.spectra` (this page **is** in
  the toctree).
- `docs/source/gamma_ray.rst:85` — `minkowski_dot` import repointed.
- `docs/PR_GUIDELINES.md` — dropped the `decay` scope row (its area,
  `hazma/_decay/`, no longer exists).
- `docs/agents/preflight.md:32-35` — the "rebuild after a change under …"
  directory list re-derived from the surviving tree.
- `docs/agents/environment.md:65-77` — the `conftest.py` exclusion note
  and the `test/` tree listing re-derived (`rambo/` was already gone and
  `agents/` was missing — pre-existing rot, corrected here).
- `docs/agents/review-lenses.md:132` — same `conftest.py` claim.
- `docs/versioning.md:43` — the underscore-package example no longer
  names `hazma/_decay/`.
- `docs/followups/todo/legacy-parameters-width-exponent-bug.md` — its two
  `_decay` / `_positron` entry points are now struck through as deleted.

### Modified — notebooks

- `notebooks/hazma_paper/snippets.py:183-196` — repointed to
  `hazma.spectra`, pinned outputs regenerated.
- `notebooks/dev/gamma_ray_fsr/partial_integration.py:10`,
  `notebooks/dev/K0_radiative_decay_4_24_18.ipynb` — `minkowski_dot`
  repointed.

<!-- markdownlint-disable MD013 -- pasted command output and evidence tables; wrapping them would falsify the record -->

## Verification

Environment: `uv venv --python 3.12` + `uv pip install -e .` in the
worktree; CPython 3.12.12, Cython 3.2.9, NumPy 2.5.1, SciPy 1.18.0.
Stale artifacts cleared before every build, per the phase README's build
hygiene note. Import path confirmed to be the worktree:

```text
$ .venv/bin/python -c "import hazma; print(hazma.__file__)"
/Users/logan.morrison/dev/Hazma/.claude/worktrees/cython-to-rust/task-0.3-delete-superseded/hazma/__init__.py
```

**Verify-before-delete** (`rules.md` Process rule 1). Re-run at execution
time, not trusted from the inventory snapshot. Every importer of a
deletion target, repo-wide, with its disposition:

| Target | Importers found | Disposition |
| --- | --- | --- |
| `hazma._decay` | `hazma/__decay.py` (also deleted); `pyproject.toml` package-data | shim deleted; config pruned |
| `hazma._positron` | `hazma/__positron_spectra.py` (also deleted); one commented line in `hazma/spectra/_positron/__init__.py:8` | shim deleted; comment removed |
| `hazma._neutrino` | two commented lines in `hazma/__neutrino_spectra.py` (also deleted) | none live |
| `hazma.__decay` / `__neutrino_spectra` | none | — |
| `hazma.__positron_spectra` | `notebooks/hazma_paper/snippets.py:183,191` | repointed to `hazma.spectra` |
| `field_theory_helper_functions.cross_section_prefactor` | `hazma/deprecated/rambo.py:24`, `hazma/gamma_ray.py:18` | both repointed to `hazma.utils` |
| `field_theory_helper_functions.minkowski_dot` | `hazma/experimental/.../avm_msqrd.py:8`, `hazma/_decay/interpolation_data/gen_ckaon_interp.py` (deleted with `_decay/`), 2 notebooks, `docs/source/gamma_ray.rst` | all repointed to `hazma.utils` |
| `three_body_phase_space` | none repo-wide | — |
| `spectra/_positron/_kaon.pyx` | not in `setup.py`; nothing cimports it | — |
| `boost_integrate_linear_interp_massive`, `integrate_linear_interp_edge`, `integration_bounds` | none (not in `boost.pxd`, no `def` wrapper) | — |
| `test/decay/` | `test/conftest.py:11` (a `collect_ignore` entry) | entry removed |

**Post-delete negative check.** Note the namespace-package trap in
Findings — this was run *after* `rm -rf`ing the emptied directories:

```text
$ .venv/bin/python -c "for m in [...]: try import"
gone: hazma._decay
gone: hazma._positron
gone: hazma._neutrino
gone: hazma.field_theory_helper_functions
gone: hazma.__decay
gone: hazma.__positron_spectra
gone: hazma.__neutrino_spectra
```

**Build.** Clean rebuild; the extension count drops exactly as the phase
predicts:

```text
$ find hazma -name '*.c' -o -name '*.cpp' -o -name '*.so' | xargs -r rm -f
$ uv pip install -e .
      Built hazma @ file:///.../task-0.3-delete-superseded
$ find hazma -name '*.so' | wc -l
      25
```

32 → 25. Task 0.2 removes `_gamma_ray` (2) + `_phase_space` (3) → the 20
survivors the phase Exit Criteria name. Consistent with the source count:

```text
$ find hazma -name '*.pyx' | wc -l   # 26 = 20 survivors + 6 Task 0.2 deletes
      26
$ find hazma -name '*.pxd' | wc -l
      19
```

The 6 remaining Task-0.2 sources are `_gamma_ray/gamma_ray_{fsr,generator}.pyx`,
`_phase_space/{generator,histogram,modifiers}.pyx`, and
`rh_neutrino/_rh_neutrino_fsr_four_body.pyx`.

**Import smoke** — the phase-file set plus everything this task touched:
`hazma`, `hazma.theory`, `hazma.limits`, `hazma.cmb`, `hazma.pbh`,
`hazma.utils`, `ScalarMediator`, `VectorMediator`, `hazma.spectra`,
`spectra._photon.{_muon,_kaon,_phi}`, `spectra._positron.{_muon,_pion}`,
`spectra._neutrino.{_muon,_pion}`, `hazma._utils.boost`,
`hazma.phase_space`, `hazma.relic_density`, `hazma.form_factors`,
`hazma.single_channel`, `hazma.rh_neutrino`, `hazma.deprecated.rambo`,
and `from hazma.utils import minkowski_dot, cross_section_prefactor`:
`import smoke OK`.

`hazma.experimental.axial_vector_mediator` is excluded — its `__init__.py`
is broken on `origin/master` (see Findings). `avm_msqrd.py` itself was
loaded directly and evaluated:

```text
$ .venv/bin/python -c "<load avm_msqrd.py by path>; msqrd_xx_to_a_to_ff(...)"
msqrd_xx_to_a_to_ff = 4.491085714285714
```

**Tests.** Both suites, which stay disjoint until Task 1.3 merges them:

```text
$ .venv/bin/python -m pytest -q test
68 passed, 20 skipped in 245.53s (0:04:05)

$ .venv/bin/python -m pytest test --collect-only -q | tail -n 1
88 tests collected in 0.65s

$ .venv/bin/python -m pytest -q          # setup.cfg testpaths -> hazma/**
57 passed, 10 skipped in 0.31s
```

`test` was 52 passed / 20 skipped at Task 0.1; the 16 new
`test/test_utils.py` cases account for the whole delta (52 + 16 = 68), and
no test was lost — `test/decay/` was already in `collect_ignore`, so it
contributed 0 to that count before deletion.

Coverage added by `test/test_utils.py`, by category:

- **Metric convention** (2): an exact small-integer value pinning
  (+,-,-,-), and an explicit not-Euclidean guard.
- **Analytic invariant** (3, parametrized over e/μ/π masses): `p.p = m²`
  for an on-shell four-momentum, `rel=1e-12` justified in the docstring
  from the (E/m)² cancellation.
- **Cross-implementation agreement** (1): bit-for-bit against `ldot` over
  100 seeded random pairs.
- **Input shape** (1): plain lists, not just arrays.
- **`cross_section_prefactor` closed forms** (3 + 3): the massless limit
  `1/(2 cme²)`, the equal-mass form `1/(4 p cme)` with `p` computed by a
  route that shares no floating-point path with `kallen_lambda`, and the
  `1/cme²` scaling far above threshold.
- **Threshold behavior** (2): monotone growth toward threshold, and the
  known cancellation at exactly threshold, pinned so the follow-up cannot
  land silently.

**Test validity (stash-proof).** Both mutations run and reverted:

```text
# 1. remove minkowski_dot from hazma/utils.py
ImportError: cannot import name 'minkowski_dot' from 'hazma.utils'
1 error in 0.25s

# 2. flip the metric to Euclidean (+,+,+,+)
6 failed, 10 passed in 0.23s
   test_minkowski_dot_sign_convention
   test_minkowski_dot_is_not_euclidean
   test_minkowski_dot_on_shell_invariant[0.5109989461]
   test_minkowski_dot_on_shell_invariant[105.6583745]
   test_minkowski_dot_on_shell_invariant[139.57039]
   test_minkowski_dot_matches_ldot

# restored
16 passed in 0.21s
```

**Docstring / doc-example execution.** Both edited examples were run
before being written down:

```text
$ <hazma/spectra/_positron/__init__.py positron_decay Examples block>
docstring example OK: (200,) 0.004423744528751423

$ <docs/source/usage.rst import block, exercised at its three call sites>
neutral_pion : [0.         0.00079174 0.00079174 0.00079174]
charged_pion : [0.01818893 0.00174195 0.00013356 0.        ]
pspec_cpion  : [0.00045571 0.00087032 0.00086591 0.00052472]
```

The first draft of the `positron_decay` example used the legacy
final-state names (`'muon'`, `'charged_pion'`) and raised
`ValueError: Encountered unknown particle muon` — `dnde_positron` uses
`'mu'` / `'pi'`. Caught only because the example was executed.

**Preflight.** Run as

```sh
scripts/agents/preflight.sh --paths "<the 9 changed .py>" --tests test \
                            --md "<the 12 changed docs>"
```

with `.venv/bin` on `PATH`. Result:

```text
PASS   black --check           <the 9 changed .py>
PASS   isort --check-only      <the 9 changed .py>
FAIL   ruff check              Found 263 errors.
PASS   pytest                  68 passed, 20 skipped in 251.07s (0:04:11)
PASS   import hazma            version 2.1.0
PASS   markdownlint            <the 10 docs this task authored or fully swept>
SKIP   version bump            not a closing PR (pass --closing)
PASS   forbidden tokens        none added
-------------------------------------------------------------------
RESULT: FAIL — blocked commit. Fix the red gates and re-run.
```

`--md` is scoped to the ten docs this task authored or swept end-to-end.
Adding `docs/PR_GUIDELINES.md` and `docs/versioning.md` — where the diff
is a one-row and a one-word edit inside otherwise untouched files — turns
that row red on their pre-existing findings; see the table below.

**The gate exits non-zero on one row — `ruff check` — and that is standing
repo debt this branch strictly reduces.** Flagged rather than reported as green:
whoever commits this makes that call knowingly. Measured against a
checkout of `origin/master` at the same paths with the same tool
versions:

| Gate | At `origin/master` | On this branch | Delta |
| --- | --- | --- | --- |
| `black --check hazma test` (CI's black 24.10.0) | clean, 249 files | **clean**, 224 files | unchanged |
| `isort --check-only` (the 8 shared files) | 5 errors | **0** | fixed |
| `ruff check` (same 8, configured) | 269 errors | 263 (+0 from the new `test/test_utils.py`) | −6 |
| `ruff check hazma test` (configured, repo-wide) | 6844 | 6619 | −225 |
| `ruff check --isolated --select E9,F63,F7,F82 .` (CI's) | clean | **clean** | unchanged |
| `markdownlint docs/PR_GUIDELINES.md` | 10 | 10 | unchanged |
| `markdownlint docs/versioning.md` | 15 | 15 | unchanged |
| `markdownlint` (all 11 other touched/new docs) | 0 | **0** | clean |

**Black must be run at CI's pinned version.** The first push of this
branch turned CI Lint red: `pyproject.toml`'s dev extra allows
`black<27.0` while `.github/workflows/ci.yml` pins `black<25.0`, and
black 26.5.1 had reformatted `rambo.py`'s `warnings.warn` into a
"hugged" style that black 24.10.0 rejects. Reverted to CI's form, and the
local venv repinned to `black>=23.3,<25.0`. This also **corrects a Task
0.1 finding** that had entered `../README.md` as fact — "black wants to
reformat 34 files on `origin/master`" was that same version skew; CI's
black reports the trunk clean. Root cause filed as
[`docs/followups/todo/black-pin-divergence-pyproject-vs-ci.md`](../../../../docs/followups/todo/black-pin-divergence-pyproject-vs-ci.md),
class added to `docs/agents/lessons.md` as
`[unpinned-formatter-version]`, trap documented in
`docs/agents/environment.md`.

isort was brought to green because every remaining complaint sat in a
file this task already edits. Per-file ruff, branch vs trunk:
`hazma/utils.py` 51 → 51, `spectra/_positron/__init__.py` 25 → 24,
`test/conftest.py` 2 → 0, everything else unchanged; the new
`test/test_utils.py` contributes **0**. Note that CI's ruff step is
`--isolated --select E9,F63,F7,F82`, not the configured one, and it
passes; the 263/6619 figures are the stricter `pyproject.toml` config,
which is red repo-wide on the trunk and does not gate CI. The two
markdownlint-red files fail on lines this task never touched (`PR_GUIDELINES.md` lines 14–19,
the commit-format block; `versioning.md` lines 117–127, a table) —
fixing them means reflowing prose and realigning a table that has
nothing to do with a dead-code purge.

**Deferred:** the remaining repo-wide ruff debt (263 findings, all
pre-existing, concentrated in `deprecated/rambo.py` (126) and
`gamma_ray.py` (39) — both Task 0.2 deletion targets); the two
markdownlint-red docs above; and the ~20 pre-existing-broken
`hazma.decay` / `hazma.positron_spectra` imports in `notebooks/` (see
Decisions). None was surfaced by this task, so none is filed as a new
follow-up — they are standing conditions of the tree.

Two invocation notes carried forward from Task 0.1 and re-confirmed:
passing `--paths` a `.pxd` makes black and ruff parse Cython as Python
and fail, and passing it a *directory* drags in that directory's
pre-existing unformatted `.py`. Scope `--paths` to the files you changed.

## Open Questions

- `cross_section_prefactor`'s threshold cancellation — filed as
  [`docs/followups/todo/cross-section-prefactor-threshold-cancellation.md`](../../../../docs/followups/todo/cross-section-prefactor-threshold-cancellation.md).
  It is a declared numerical change to a live public API, so it is
  deliberately *not* bundled into a dead-code deletion.
- `boost_jac` / `boost_eng` in `_utils/boost.pxd` have zero cimporters but
  are part of the declared C API; whether they die with the rest of the
  header is a Phase 06 Task 6.4 call (noted in `../README.md` Findings).
- `hazma/experimental/axial_vector_mediator/__init__.py`'s
  `from hazma.theory import Theory` is broken on the trunk. Not filed as a
  follow-up: `experimental/` is explicitly outside the public surface
  (`docs/versioning.md`) and excluded from the lint gate (`AGENTS.md`).

## Plan Impact

**Impact Level:** Update phase file.

Two of Task 0.4's exit-criterion bullets are now partly satisfied by this
task, and the phase file's Task 0.3 criteria did not mention build config
at all. The phase file is patched so the criteria match what the build
actually forces:

- Task 0.3 gains a criterion naming the three forced config edits
  (`setup.py` extension groups, `test/conftest.py` collection, the
  `_decay` packaging entries) — exactly the shape of the correction Task
  0.1 made when it found a fifth include site.
- Task 0.4's criteria are re-scoped to what is left for it: the
  `_gamma_ray` / `_phase_space` groups Task 0.2 deletes, plus the final
  survivor-count and sdist reconciliation.

No change to task ordering, dependencies, or any interface. `PLAN.md`
holds only the phase table and needs no edit — the phase's deliverable is
unchanged.

## Stale-state sweep

Run against branch `claude/cython-to-rust/task-0.3-delete-superseded`
with the working tree staged. Blocks are folded to per-file counts
(`rg -c`) where the unfolded output would include this note quoting
itself; that is labelled where it applies.

**Deleted-path sweep** — nothing outside `projects/` and the stale
top-level `searchindex.js` (a checked-in Sphinx build artifact predating
this work) may still name a deleted module:

```text
$ rg -c 'hazma\._decay|hazma\._positron|hazma\._neutrino|field_theory_helper_functions|hazma\.__decay|hazma\.__positron_spectra|hazma\.__neutrino_spectra|hazma\.positron_spectra|from hazma\.decay' \
    hazma/ test/ docs/ setup.py pyproject.toml MANIFEST.in | sort
docs/followups/todo/cross-section-prefactor-threshold-cancellation.md:2
hazma/spectra/_positron/__init__.py:1
hazma/utils.py:1
test/test_utils.py:1
```

Disposition, one row per file — all four are deliberate historical
references, none is a live path:

| File | Disposition |
| --- | --- |
| `docs/followups/todo/cross-section-prefactor-threshold-cancellation.md` | KEPT — cites the deleted `common_functions.pyx` as prior art for the factored form, with a `git show` recipe |
| `hazma/spectra/_positron/__init__.py` | KEPT — the new `Raises` section names `hazma._positron.positron_decay` as the removed backend |
| `hazma/utils.py:143` | KEPT — provenance note on the relocated `minkowski_dot` |
| `test/test_utils.py:4` | KEPT — module docstring records which deleted module these tests took over from |

`setup.py`, `pyproject.toml`, `MANIFEST.in`, and all of `test/` except
that one docstring produced no hits. `notebooks/` is excluded here and
handled by the documented boundary in Decisions.

**Dead-boost-symbol sweep** — the three deleted `cdef`s must have no
remaining mention:

```text
$ rg -n 'boost_integrate_linear_interp_massive|integrate_linear_interp_edge' hazma/
(no matches)

$ rg -n '\bintegration_bounds\b' hazma/
(no matches)
```

(`hazma/phase_space/_three_body.py` defines `_integration_bounds_x/y/s/t`
— different, pure-Python, underscore-prefixed names, matched only by a
substring search. The word-boundary form above is clean.)

**Surviving-Cython count sweep** — re-derived from source per
`lessons.md` `[derived-count-not-rederived]`, never carried over:

| Claim | Command | Actual | Status |
| --- | --- | --- | --- |
| 25 extensions built (was 32) | `find hazma -name '*.so' \| wc -l` | `25` | OK |
| 26 `.pyx` remain (20 survivors + 6 for Task 0.2) | `find hazma -name '*.pyx' \| wc -l` | `26` | OK |
| 19 `.pxd` remain (was 33) | `find hazma -name '*.pxd' \| wc -l` | `19` | OK |
| `test/` suite | `pytest -q test` | `68 passed, 20 skipped` | OK |
| `test/` collection | `pytest test --collect-only -q \| tail -1` | `88 tests collected` | OK |
| in-package suite | `pytest -q` | `57 passed, 10 skipped` | OK |
| new tests | `pytest -q test/test_utils.py` | `16 passed` | OK |
| diff size | `git diff --cached --stat \| tail -1` | `135 files changed, 1261 insertions(+), 29337 deletions(-)` | OK |

**Line-number citation sweep.** `--changed-vs origin/master` reports "no
docs to check" pre-commit (it diffs commits; this branch has none yet), so
the touched/created docs are passed explicitly:

```text
$ ./scripts/agents/check_doc_citations.py <the 12 touched/created docs>
docs scanned: 12
in-repo citations checked: 27
  resolved by exact: 22
  resolved by suffix: 5
external citations skipped: 4
  hazma/_decay/common.pxd (1)
  hazma/_positron/parameters.pxd (1)
  hazma/experimental/.../avm_msqrd.py (1)
  hazma/field_theory_helper_functions/common_functions.pyx (1)
out-of-range or ambiguous: NONE
```

The four "external" (i.e. unresolvable) citations are exactly the
deliberate references to files this task deleted — the two struck-through
entry points in the `WIDTH_K` follow-up, the prior-art citation in the new
one, and an elided `.../` path. Each is intended; none is a live path.

**Numerical-impact statement.** Three independent measurements; the
scripts are in the session scratchpad, the grids are stated below.

*(1) Public compiled surface — no change.* Every compiled-backed public
entry point over `np.logspace(-2, 3, 200)` MeV: the 12 `dnde_photon_*`,
2 `dnde_positron_*`, and 2 `dnde_neutrino_*` (× 3 flavors) spectra at
three parent energies; plus `ScalarMediator` / `VectorMediator`
`spectra()`, `positron_spectra()`, `annihilation_cross_sections()`, and
`thermal_cross_section()` at three mediator masses — **171 arrays**.
Captured at the pre-change build, then after the deletions and a full
clean rebuild:

```text
arrays compared: 171
arrays NOT bit-identical: 0
max relative deviation: 0.000e+00
```

*(2) `cross_section_prefactor`: Cython → `hazma.utils` — declared drift.*
216 points: all 36 ordered mass pairs from
{e, μ, π⁰, π±, K±, p} at ten multiples of the `m1 + m2` threshold.
Both implementations still existed pre-deletion, so this is a direct
comparison, not a before/after:

| `cme / (m1+m2)` | bit-identical | max relative deviation |
| --- | --- | --- |
| 1 + 1e-7 | 0/36 | 2.060e-07 |
| 1 + 1e-6 | 0/36 | 1.187e-08 |
| 1 + 1e-5 | 0/36 | 3.701e-09 |
| 1 + 1e-4 | 0/36 | 1.621e-10 |
| 1 + 1e-3 | 0/36 | 1.107e-11 |
| 1 + 1e-2 | 0/36 | 1.985e-13 |
| 1.1 | 7/36 | 4.759e-15 |
| 2 | 20/36 | 3.229e-16 |
| 10 | 16/36 | 3.374e-16 |
| 100 | 17/36 | 2.395e-16 |

Cause: `kallen_lambda` cancels at threshold; the deleted factored form did
not. Affected public paths: `hazma.deprecated.rambo`
(`compute_annihilation_cross_section`, `PhaseSpace.cross_section`) and
`hazma.gamma_ray` (broken on import; ADR-0003 removes it).
`hazma.phase_space._rambo` already used `hazma.utils` and is untouched.

End-to-end confirmation on the one live path, seeded so it is
reproducible — `PhaseSpace.cross_section(m1, m2, n=20000, seed=1234)`:

```text
case0  e e -> mu mu,   cme=1000        rel=0.000e+00
case1  e e -> pi pi,   cme=1000        rel=0.000e+00
case2  mu mu -> 4e,    cme=500         rel=0.000e+00
case3  pi pi -> e e,   cme=2mpi(1+1e-7) rel=1.761e-10
case4  pi pi -> e e,   cme=2mpi(1+1e-3) rel=1.723e-14
```

Bit-identical at ordinary kinematics; the shift appears only within
~1e-7 of threshold, and is orders of magnitude below these estimators'
own Monte-Carlo error (`case3` reports ±2.1e-19 on 5.7e-04 only because
the integrand is flat there; the physical MC error at `n = 20000` is
~1/√n).

*(3) `minkowski_dot`: Cython → `hazma.utils` — declared drift.*

```text
1998 random four-vector pairs (scale 100):
  bit-identical:          1254/1998
  max relative deviation: 2.696e-14
1999 on-shell charged-pion four-vector pairs:
  bit-identical:          1335/1999
  max relative deviation: 3.203e-15
```

Cause: the C compiler contracts `a*b - c*d` into an FMA; Python cannot.
Not a public-surface change — the only in-library consumer is
`hazma/experimental/`, which `docs/versioning.md` excludes.

**Versioning re-check.** Both drifts land in `hazma/deprecated/` (public
per `docs/versioning.md` §6) and in `hazma/experimental/` (explicitly
not public). A moved published number is at least `minor`; the project's
`version_bump: major` already covers it, driven by the API removals. No
change to `PLAN.md` frontmatter.

**Exit Criteria → verification mapping:**

| Exit criterion | Satisfied by |
| --- | --- |
| All named paths deleted | `git diff --cached --stat` (135 files, 29,337 deletions) + the post-delete negative import check |
| Dead `boost.pyx` half deleted | dead-boost-symbol sweep (no matches); `boost.pyx` 461 → 241 lines |
| `cross_section_prefactor` callers use `hazma.utils` | deleted-path sweep (no live hits) + `rg cross_section_prefactor hazma/` showing both callers on `hazma.utils` |
| `minkowski_dot` has a pure-Python home; `avm_msqrd.py` repointed | `hazma/utils.py:124`; `avm_msqrd.py:10`; direct-load evaluation above |
| Import smoke passes | `import smoke OK` over 21 modules |

**Task-note self-consistency:** `**Status:** Complete` matches all five
mapping rows satisfied; every path named in §Files Changed appears in
`git diff --cached --name-status` or is created by this task; the phase
README row and this note's status agree.

<!-- markdownlint-enable MD013 -->

## Handoff to Next Task

- **Read first:** `../README.md` (project working memory) → this phase's
  `README.md` → the phase file → this note.
- **Now safe to assume:** the tree carries 25 extensions and 26 `.pyx`;
  everything the purge can delete without ADR-0003 is gone. Nothing
  outside `_gamma_ray/`, `_phase_space/`, and
  `rh_neutrino/_rh_neutrino_fsr_four_body.pyx` is dead Cython.
  `hazma.utils` is the single home for `cross_section_prefactor` and
  `minkowski_dot`. `test/conftest.py` now ignores only
  `test_gamma_ray.py`.
- **Build hygiene, still load-bearing:** clear stale `.c`/`.cpp`/`.so`
  before every rebuild. **New:** after `git rm -r` on a package, also
  `rm -rf` the directory — a leftover `__pycache__` keeps it importable
  as a namespace package and silently passes a negative import check.
- **Still blocked:** Task 0.2 needs ADR-0003 accepted (Task 0.5), which no
  implementation step can unblock. Task 0.4 is now unblocked only once
  0.2 lands, and its remaining scope is narrower — see Plan Impact.
- **Carry forward:** the `cross_section_prefactor` follow-up must not be
  folded into a Rust port silently; if Phase 01 captures the corpus first,
  the Rust side inherits the current (cancelling) values by construction.
