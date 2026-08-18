# Task 4.5: `_photon/_rho` (nested quadrature)

**Date:** 2026-08-18
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-04-spectra-kernels.md` (Goal, "The
per-kernel swap recipe", Task 4.5); `../../PLAN.md` (Numerical impact)
**Related ADRs:** ADR-0001, ADR-0002
**Depends On:** Task 4.4

## Objective

Port `hazma/spectra/_photon/_rho.pyx` — both ρ photon spectra — to
`hazma._core`, swap the two wrapper entry points, delete the twin, and give
the nested integral (ρ `quad` over `_pion`, which `quad`s over `_muon`) the
dedicated drift analysis the phase file asks for. This is the project's
declared numerical stress test.

## Exit Criteria

Copied from `../../phases/phase-04-spectra-kernels.md` "Task 4.5":

- Both ρ entry points corpus-green; the nested integral gets a dedicated
  drift analysis in the task note.
- The Cython version's untyped `cdef` locals (Python-boxed) are ported as
  plain f64 — confirm no value shift beyond budget.

Plus the phase's standing per-kernel recipe (steps 1–8):

- FMA map read off the shipped `.so` before any Rust is written.
- Kernel in `rust/src/kernels/photon_rho.rs`, PyO3-free.
- Registered through `dispatch::map_unary` with the twin's `assert`
  wording.
- `hazma/spectra/_photon/__init__.py` repointed.
- `test/parity/cases.py` repointed to the wrapper + `PORTED_ENTRY_POINTS`
  rows; corpus green.
- Twin deleted — `_rho.pyx`, `_rho.pxd`, `_rho.pyi`, and its `setup.py`
  entry (it is not a capi survivor).
- `test/test_core_photon_rho.py` added.
- Drift recorded here and in `../README.md`'s "Numerical impact so far".

## Inputs Reviewed

- `../../PLAN.md`; `../README.md`; `../../rules.md`.
- `../../phases/phase-04-spectra-kernels.md`.
- `task-4.4-photon-pion.md` (Handoff, Findings), `task-4.3-photon-muon.md`.
- `hazma/spectra/_photon/_rho.{pyx,pxd,pyi}`, `_pion.pxd`,
  `hazma/_utils/{boost,kinematics,constants}.pxd`.
- `rust/src/kernels/photon_pion.rs`, `rust/src/{photon,quad,boost,constants}.rs`.
- `test/parity/{cases,tolerances}.py`, `test/test_core_{photon_pion,photon_tables,dispatch,constants,quad}.py`.
- `../../../docs/agents/{environment,lessons,doc-consistency}.md`.

## Findings

### The FMA map is empty, and that is the whole exit criterion

```console
$ objdump -d hazma/spectra/_photon/_rho.cpython-312-darwin.so \
    | grep -c 'fmadd\|fmsub\|fnmadd\|fnmsub'
0
```

Zero contracted operations in the entire object — the first kernel in
Phase 04 where that is true. Tasks 4.1/4.3/4.4 found 3, 22 and 19.

The cause is the phase file's second exit criterion, seen from the other
side. Both `*_point` `cdef`s declare `beta`, `gamma`, `emin`, `emax` and
`pre` **untyped**, so Cython boxes each into a `PyFloatObject` and
evaluates `gamma * e * (1 - beta)` through `PyNumber_Multiply` and
`__Pyx_PyFloat_SubtractCObj`. There is no C expression for clang to
contract. The disassembly of `dnde_photon_neutral_rho_point` shows it
directly: `fdiv`/`fmul`/`fsub`/`fsqrt` for the inlined `boost_beta`, then
`PyFloat_FromDouble`, then a chain of `ldr d, [x, #0x10]` unboxing loads
around scalar arithmetic and one `__Pyx_PyFloat_TrueDivideCObj` for
`0.5 / (beta * gamma)`.

Python float arithmetic *is* IEEE `f64` arithmetic, correctly rounded one
operation at a time, so "ported as plain f64" is not an approximation
here — it is exact. **Adding a `mul_add` would have been the error**, and
the port contains none.

The two integrands are the only C-typed arithmetic in the file, and both
are trivial: `integrand_neutral_rho` emits `fadd d0, d9, d9` (the `2 *`
as `x + x`, the same double) then `fdiv`; `integrand_charged_rho` emits
`fadd` then `fdiv`.

### `two_body_energy` is folded to two immediates, and they are pinned

`integrand_charged_rho` calls `two_body_energy` twice on `DEF` constants.
clang folds both and materialises them with a `mov`/`movk` quartet:

| Immediate | Value (MeV) | Expression |
| --- | --- | --- |
| `0x4078_4718_126d_6814` | 388.4433769487089 | `two_body_energy(m_ρ, m_π±, m_π⁰)` |
| `0x4078_2d10_e355_2748` | 386.8166230512911 | `two_body_energy(m_ρ, m_π⁰, m_π±)` |

Both reproduce bit-for-bit from `(q² + m1² − m2²)/(2q)` evaluated left to
right without contraction, in Rust `const` context and in Python. Pinned
by `the_two_body_energies_match_the_shipped_immediates`.

### The drift: the project's stress test came in five decades inside its budget

Measured against the live Cython twin **before** deleting it, with the
Rust already registered — the configuration the phase recipe's step 5
requires.

On the exact points the parity corpus pins (`test/parity/cases.py`
blocks, 1,395 values per entry point):

| block | `E_ρ` | n | bit-equal | worst relative |
| --- | --- | --- | --- | --- |
| `charged_rho` / `rest` | 775.26 | 255 | 226 | 4.04e-16 |
| `charged_rho` / `rest_plus_eps` | 775.26 | 285 | 236 | 5.08e-16 |
| `charged_rho` / `near_rest` | 814.023 | 285 | 227 | 2.67e-15 |
| `charged_rho` / `boosted_mild` | 1550.52 | 285 | 182 | 1.03e-15 |
| `charged_rho` / `boosted_strong` | 7752.6 | 285 | 199 | **1.51e-13** |
| `neutral_rho` / `rest` | 775.26 | 255 | 201 | 2.94e-15 |
| `neutral_rho` / `rest_plus_eps` | 775.26 | 285 | 219 | 1.73e-15 |
| `neutral_rho` / `near_rest` | 814.023 | 285 | 213 | **3.21e-15** |
| `neutral_rho` / `boosted_mild` | 1550.52 | 285 | 204 | 1.49e-15 |
| `neutral_rho` / `boosted_strong` | 7752.6 | 285 | 215 | 8.12e-16 |

Totals: **1,070 / 1,395 bit-equal (76.7%)** for the charged ρ and
**1,052 / 1,395 (75.4%)** for the neutral one.

A denser off-corpus sweep — 400 log-spaced photon energies at each of
eight parent energies including four the corpus does not sample — found
the worst case anywhere:

| entry point | n | bit-equal | worst relative | where |
| --- | --- | --- | --- | --- |
| `charged_rho` | 3,200 | 2,133 (66.7%) | **2.50e-11** | `E_ρ = 5 m_ρ`, `E_γ = 431.99` MeV |
| `neutral_rho` | 3,200 | 2,104 (65.8%) | 4.92e-13 | `E_ρ = 2 m_ρ`, `E_γ = 904.08` MeV |

The 2.5e-11 outlier is explained, not absorbed. At that point the boost
window is `[43.64, 4276.24]` MeV and the π⁰ box's **upper** edge sits at
374.66 MeV — a jump discontinuity strictly inside the interval, which is
exactly the shape where a last-ulp change moves a bisection decision.
scipy's own report for that call is `abserr = 8.82e-08` on a value of
`1.183e-02`, i.e. a claimed relative accuracy of 7.5e-06. **The two
implementations differ five decades below the error either of them
admits to.**

Task 4.4's prediction that the ρ would be loud was wrong in the same way
Task 4.3's `1/β` prediction was wrong for the pion: re-derived rather
than inherited, the nesting turns out to *damp* rather than amplify,
because the outer integral averages the inner one over a window.

### The mutation that no ordinary gate can see, and the seam that fixes it

Six mutations, each applied to `rust/src/kernels/photon_rho.rs` alone and
gated on `cargo test`, `test/test_core_photon_rho.py` (49 tests) and
`pytest test/parity -k rho` (10 blocks):

| # | Mutation | `cargo` | rho module | parity(rho) |
| --- | --- | --- | --- | --- |
| M1 | swap the charged-ρ daughter energies | pass | **6 failed** | **5 failed** |
| M2 | drop the neutral-ρ factor of two | **fail** | **6 failed** | **5 failed** |
| M3 | fuse `emin`'s multiply-subtract | pass | pass | pass |
| M4 | drop the rest-frame short circuit | **fail** | **3 failed** | **2 failed** |
| M5 | halve the boost prefactor | pass | **9 failed** | **8 failed** |
| M6 | widen the quadrature `epsrel` by a decade | pass | **3 failed** | **5 failed** |

**M3 survived everything.** This is Task 4.4's warning arriving from a
new direction: there it was an unfused FMA *inside* an integrand, here it
is one in the integration **limits**, and the reason is the same — the
outer call's `epsrel` is 1e-5, and one ulp of an endpoint does not reach
it.

Unlike Task 4.4's case, this one had a fix. `boost_window(e, erho)` is
now a separate `fn` returning `(emin, emax, pre)`, and
`the_boost_window_is_computed_without_contraction` pins all three bit-for-bit
at four live arguments. Re-running M3 against it: **`test result: FAILED.
132 passed; 1 failed`**, the boost-window test. So the campaign closes
6 / 6, and the module's one previously-untestable FMA site is now tested.

The general lesson, for Task 4.6 and Phase 06: when a mutation survives a
quadrature-backed kernel's gates, check whether the arithmetic can be
lifted *out* of the integral before concluding it is untestable.

### The ρ compounds the charged pion's forward-cone defect

Task 4.4's handoff asked whether the lost forward cone
([`../../../../docs/followups/todo/charged-pion-photon-spectrum-misses-the-forward-cone.md`](../../../../docs/followups/todo/charged-pion-photon-spectrum-misses-the-forward-cone.md))
reaches the ρ. It does, and it gets **worse**, which was not the expected
answer.

A pure inheritance would preserve the *fraction* of the endpoint at which
the cliff sits: the boost maps both the onset and the endpoint by the
same `γ(1+β)`, because `γ(1−β)·γ(1+β) = γ²(1−β²) = 1`. The inner kernel
at the ρ's daughter energy (`E_π = 388.44` MeV, `γ_π = 2.78`) first
returns a spurious zero at 0.945 of its own endpoint, so 0.945 is what
every ρ energy should show. Measured:

| `E_ρ` (MeV) | `γ_ρ` | charged-ρ onset / endpoint | neutral-ρ onset / endpoint |
| --- | --- | --- | --- |
| 814 | 1.05 | 0.9963 | 0.9420 |
| 1163 | 1.5 | 0.9866 | 0.9315 |
| 1551 | 2 | 0.9707 | 0.9185 |
| 2326 | 3 | 0.9326 | 0.8806 |
| 3876 | 5 | 0.8249 | 0.7803 |
| 7753 | 10 | 0.5366 | 0.5073 |

At `γ_ρ = 10` the ρ loses the top **46%** of its spectrum where pure
inheritance predicts 5.5%. The mechanism is the followup's own, one level
out: the outer window spans decades while the integrand is nonzero only
near its lower end, so once that sub-window is narrower than the
21-point Gauss–Kronrod spacing on the full interval, every node returns
zero. **Repairing the charged-pion kernel is necessary but not sufficient
for the ρ** — the followup has been updated with this table and with the
consequence for its repair plan.

Blocked on the same corpus regeneration as the other six defects: the ρ's
own corpus blocks pin the zeros.

### A seventh blocked defect: the rest-frame branch returns the integrand

The `E_ρ − m_ρ < DBL_EPSILON` short circuit returns
`integrand_*_rho(e)`, which carries the boost kernel's `1/E'`. The
`β → 0` limit of the quadrature branch is the rest-frame spectrum `f(E)`;
the branch returns `f(E)/E`. That is MeV⁻² where the other branch is
MeV⁻¹.

Measured by stepping from `E_ρ = m_ρ` to the next representable double —
the ratio is **exactly `E_γ`**:

| `E_γ` (MeV) | at `E_ρ = m_ρ` | one ulp above | ratio |
| --- | --- | --- | --- |
| 13 | 5.040024e-04 | 6.552032e-03 | 13.000000 |
| 50 | 1.124352e-04 | 5.621762e-03 | 50.000000 |
| 200 | 2.728379e-05 | 5.456758e-03 | 200.000000 |
| 300 | 1.817474e-05 | 5.452422e-03 | 300.000000 |

The blast radius is narrow and sharp: the guard is **absolute** and one
ulp at 775.26 MeV is 1.14e-13, about 500x `DBL_EPSILON`, so the branch
fires at `E_ρ == m_ρ` and at no other double. Reproduced under rule 1,
pinned by `test_the_rest_frame_branch_returns_the_bare_integrand` in both
languages, and filed as
[`../../../../docs/followups/todo/rho-rest-frame-branch-returns-the-integrand.md`](../../../../docs/followups/todo/rho-rest-frame-branch-returns-the-integrand.md).

The other boosted kernels (`photon_muon`, `photon_tables`,
`positron_muon`) return a genuine rest-frame spectrum from the same
branch shape, so this is a ρ-specific defect and not a library
convention.

### Two environment traps, both recorded in `docs/agents/environment.md`

Hit while gating this task, and both cost a red preflight to find:

1. **Deleting a `.pyx` does not make its module unimportable.** The built
   `_rho.cpython-312-darwin.so` and generated `_rho.c` sit beside the
   source, are gitignored, and survive the deletion, a `git checkout` and
   a `git stash` cycle. The first version of
   `test_the_cython_twin_is_gone_from_the_tree` asserted
   `pytest.raises(ImportError)` and duly failed — correctly, on a stale
   artifact my own stash cycle had regenerated. It now asserts on the
   **source files and the `setup.py` entry**, which is what rule 1
   actually claims; both halves are stash-proofed below.
2. **`git checkout <path>` restores from the *index*, not from HEAD.**
   Restoring `setup.py` after a mutation probe silently reinstated a
   version staged by an earlier `git add -A`, putting `_rho` back in the
   extension list. Preflight then rebuilt the extension and the test
   above went red for a third distinct reason. It prints only
   `Updated 1 path from the index`, so nothing announces the revert.

### `derived::photon_rho` went with its source

`rust/src/constants.rs` carried a `derived::photon_rho` submodule from
Task 3.1 — three bare aliases of `pdg` masses, "no arithmetic" by its own
docs — and `test/test_core_constants.py`'s `DERIVED_SOURCES` mapped it to
`hazma/spectra/_photon/_rho.pyx`. That file is now gone, and
`test_no_pyx_declares_constants_this_module_ignores` scans the tree
fresh, so the row could not stay. Both went; the kernel reads
`constants::pdg` directly. The precedent is Task 4.2, whose five deleted
`.pyx` left no `derived::` submodule behind either.

## Decisions and Implementation Notes

- **One `boosted` helper, two entry points.** The two `*_point` `cdef`s
  are character-for-character identical apart from which integrand they
  name, so the port takes an `fn(f64) -> f64`. This is a structural
  choice, not a numerical one — the arithmetic is unchanged.
- **`points: None`, so `qagse` not `qagpe`.** Unlike `photon_pion`'s call
  site the `.pyx` passes no `points` keyword. Recorded in `RHO_QUAD`'s
  docs because the two neighbouring kernels differ here.
- **The `Err` arm returns `NaN`, not a panic** — inherited verbatim from
  Task 4.4's reasoning, and asserted unreachable for a `const` opts value
  by `rho_quad_options_are_always_accepted`.
- **`test/parity/tolerances.py` gains `PORTED_NESTED_RTOL = 1e-9`**,
  taken by the two ρ cases; the seven unported mediator-spectrum cases
  keep `NESTED_RTOL = 1e-6` until Phase 06 measures them. This is
  Task 4.4's `PORTED_QUAD_RTOL` pattern, per-case rather than class-wide.
  1e-9 is 6,600x over the worst pinned drift (1.5e-13) and 40x over the
  worst found anywhere (2.5e-11); the class docstring carries the
  derivation. Rule 2's one-line justification is in both `Budget.why`
  strings.
- **The test module is shaped after `test_core_photon_tables.py`, not
  `test_core_photon_pion.py`,** and the choice is forced: nothing
  cimported `_rho.pyx`, so there is no surviving `cdef` to use as an
  oracle. The substitute is
  `TestAgainstAnIndependentBoostIntegral` — the deleted `.pyx`'s three
  branches transcribed into Python over `scipy.integrate.quad`, at the
  same `epsabs`/`epsrel`, over the *ported* pion kernels. It is a second
  opinion on the layer this task added and says nothing about the pion
  kernels, which both sides share.
- **`test/test_core_dispatch.py`'s spectra oracle moved to
  `_positron/_pion`.** Task 4.4's handoff flagged this: the class needs a
  `.pyx` still exporting a unary `def`, and this task exhausted the
  photon candidates. `_positron/_pion.pyx` has the identical
  `hasattr(__len__)` / `assert` / array-return shape and differs only in
  saying `"Positron energies"`, which those tests read from source rather
  than hard-code. `TestCythonMessageParity`'s roster is **unaffected** —
  `"Photon energies must be 0 or 1-dimensional."` survives in
  `hazma/spectra/_neutrino/_muon.pyx:205`, the copy-paste defect Task 3.5
  recorded. Task 4.6 exhausts the unary candidates entirely; the class
  docstring now says so and names the choice it leaves.

## Files Changed

| Path | Purpose |
| --- | --- |
| `rust/src/kernels/photon_rho.rs` | **new** — the port, PyO3-free, 12 `cargo` tests |
| `rust/src/kernels.rs` | register `photon_rho`; module-doc roster |
| `rust/src/photon.rs` | two `#[pyfunction]`s through `map_unary` |
| `rust/src/constants.rs` | retire `derived::photon_rho` with its source |
| `rust/src/quad.rs` | mark the ρ rows of the live-call-site table as ported |
| `rust/src/kernels/photon_pion.rs` | Task 4.5 is landed; past tense, `super::photon_rho` link |
| `hazma/spectra/_photon/__init__.py` | both wrappers repointed to `_core_photon` |
| `hazma/spectra/_photon/_rho.pyx` | **deleted** (not a capi survivor) |
| `hazma/spectra/_photon/_rho.pxd` | **deleted** |
| `hazma/spectra/_photon/_rho.pyi` | **deleted** |
| `setup.py` | drop `_rho` from the `_photon` extension list; comment re-derived |
| `test/parity/cases.py` | both cases repointed to the wrapper; two `PORTED_ENTRY_POINTS` rows; charged-pion cimporter comment corrected |
| `test/parity/tolerances.py` | `PORTED_NESTED_RTOL`; both ρ budgets; `NESTED` class docs |
| `test/test_core_photon_rho.py` | **new** — 49 tests |
| `test/test_core_dispatch.py` | spectra oracle `_photon/_rho` → `_positron/_pion` |
| `test/test_core_constants.py` | drop the `derived::photon_rho` row, with the reason |
| `test/test_core_photon_pion.py` | capsule-roster comments lost a cimporter |
| `test/test_core_quad.py` | two `_rho.pyx` citations marked ported |
| `docs/followups/todo/rho-rest-frame-branch-returns-the-integrand.md` | **new** |
| `docs/followups/todo/charged-pion-photon-spectrum-misses-the-forward-cone.md` | the ρ compounds it — measured table + repair consequence |
| `docs/followups/README.md` | index row for the new follow-up |
| `docs/agents/environment.md` | two new traps: a deleted `.pyx` stays importable, and `git checkout` restores from the index |
| `projects/cython-to-rust/task-notes/phase-04/task-4.5-photon-rho.md` | **new** — this note |
| `projects/cython-to-rust/task-notes/phase-04/README.md` | Task 4.5 status, findings, handoff |
| `projects/cython-to-rust/task-notes/README.md` | numerical impact, findings, handoff |

## Verification

### Gates

```console
$ cargo fmt --manifest-path rust/Cargo.toml --check
$ cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.50s
$ cargo test --manifest-path rust/Cargo.toml --no-default-features
test result: ok. 133 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
$ .venv/bin/python -m pytest -q
1802 passed, 15 skipped, 6 warnings in 47.15s
$ .venv/bin/python -m pytest test/test_core_photon_rho.py -q -p no:randomly
49 passed in 4.41s
$ .venv/bin/python -m pytest test/parity -q -p no:randomly
629 passed, 1 skipped, 9 warnings in 31.84s
$ PY=$(git diff origin/master --name-only --diff-filter=d | grep '\.py$' | tr '\n' ' ')
$ MD=$(git diff origin/master --name-only --diff-filter=d | grep '\.md$' | tr '\n' ' ')
$ env PATH="$PWD/.venv/bin:$PATH" scripts/agents/preflight.sh --paths "$PY" --md "$MD"
RESULT: PASS          # all eleven rows; only "version bump" SKIPs (not a closing PR)
```

**Scope `--paths` to `.py` files.** A first run passed the whole changed-
file list and the black / isort / ruff gates went red on the `.md` and
`.rs` paths in it — a harness artifact, not a finding. They are also red
on the trunk, which is the standing
[`preflight-isort-ruff-red-on-trunk.md`](../../../../docs/followups/todo/preflight-isort-ruff-red-on-trunk.md)
follow-up; measured side by side with the same tool versions, this branch
is strictly better on each:

| gate | `origin/master` | this branch |
| --- | --- | --- |
| `black --check hazma test` | 214 files unchanged | 214 files unchanged |
| `ruff check hazma test` | 6141 errors | **6135** |
| `isort --check-only hazma test` | 99 ERROR lines | **97** |

`1802 passed / 15 skipped`, from `1755 / 15` at Task 4.4 — **+47**, and
the arithmetic is 49 − 2. Both halves measured by stashing this branch's
diff and re-running on the pre-task tree:

| Module | before | after | why |
| --- | --- | --- | --- |
| `test/test_core_photon_rho.py` | — | 49 | new |
| `test/test_core_constants.py` | 25 | 23 | two parameterized rows retired with `derived::photon_rho` |
| `test/test_core_dispatch.py` | 118 | 118 | the oracle module was swapped, not the assertions |
| whole suite | 1755 | 1802 | +47 |

`cargo test --no-default-features` goes **121 → 133**, and the +12 is
exactly `photon_rho`'s own tests (`cargo test | grep -c
'^test kernels::photon_rho'` → 12).

The parity run above is the **baseline**, taken before any edit on this
branch, and it is quoted to show the tree was green to start with; the
post-swap parity result is inside the `1802 passed` line.

### What the 49 per-kernel tests cover

- **`TestDispatchWiring` (20)** — one assertion per contract branch for
  each entry point: scalar → `float`, NumPy scalar and 0-d array on the
  scalar path, array → fresh `float64` array, array path bit-equal to the
  scalar path, sequence accepted, empty grid, the rank message verbatim,
  non-`float64` dtype → `ValueError`, non-number → `TypeError`, plus both
  arguments by keyword.
- **`TestWrapperAndPublicApi` (4)** — the private wrappers return the
  kernel's bytes; `hazma.spectra` names and `__all__`; the three source
  files and the `setup.py` entry are gone (**not** an `ImportError`
  assertion — see Findings); `_nbody._dnde_photon_dict`'s `rho`/`rho0`
  rows reach the ported wrappers.
- **`TestAgainstAnIndependentBoostIntegral` (12)** — 10 swept-grid
  comparisons (2 entry points x 5 parent energies) against the Python +
  scipy transcription, plus two non-vacuity checks: a doubled prefactor
  and swapped daughter energies both break the comparison.
- **`TestPhysics` (13)** — below-threshold exact zero at four energies;
  the rest-frame branch and the exact factor-`E` discontinuity; the
  spectrum vanishes above its kinematic endpoint; the π⁰ box edge
  reverses which ρ is brighter (8 probes); the boost pushes flux past the
  rest-frame endpoint; the charged-ρ plateau falls and the neutral-ρ
  spectrum rises with boost; non-negative and finite everywhere; a `NaN`
  parent propagates.

### The 12 `cargo` tests

Constants (2): the shipped immediates, and the daughter energies summing
exactly to `m_ρ`. Arithmetic (3): no contraction at a probe where it
matters, the boost window bit-for-bit at four arguments, the window
brackets the lab energy with `√(emin·emax) = E`. Branches (3):
below-threshold zero, the rest-frame integrand, `NaN` propagation.
Physics (3): both integrands positive, the π⁰ box edge reversal, boosting
lowers the charged spectrum. Contract (1): `RHO_QUAD` is accepted on
every interval the kernel can produce, so `boosted`'s `Err` arm is
unreachable.

### Test validity: a six-mutation campaign

The table under Findings. Five mutations were caught by the gates as
written; M3 was not, and the `boost_window` seam was added specifically
so that it is. Re-verified after the refactor:

```console
$ # M3 re-applied to boost_window
$ cargo test --manifest-path rust/Cargo.toml --no-default-features
test kernels::photon_rho::tests::the_boost_window_is_computed_without_contraction ... FAILED
test result: FAILED. 132 passed; 1 failed
$ # restored
test result: ok. 133 passed; 0 failed
```

### Test validity: the deletion test, both halves

`test_the_cython_twin_is_gone_from_the_tree` makes two claims; each was
reverted independently and confirmed to turn it red.

```console
$ git show origin/master:hazma/spectra/_photon/_rho.pyx > hazma/spectra/_photon/_rho.pyx
$ pytest test/test_core_photon_rho.py -q -k cython_twin
E  +  where exists = (PosixPath('.../hazma/spectra/_photon') / '_rho.pyx').exists
1 failed in 1.36s
$ rm hazma/spectra/_photon/_rho.pyx     # and put "_rho" back in setup.py
$ pytest test/test_core_photon_rho.py -q -k cython_twin
test/test_core_photon_rho.py:377: AssertionError
1 failed in 1.35s
```

A `git stash` of the whole branch is **not** a validity check here: it
removes the test module too, and `pytest` then reports `no tests ran`
with exit 5, which reads as green in a pipeline.

### Numerical impact (rule 3, phase-file recipe step 8)

**Both ρ entry points moved, and both are declared.**

| Entry point | Grid | Worst relative | Verdict |
| --- | --- | --- | --- |
| `hazma.spectra.dnde_photon_charged_rho` | 1,395 corpus-pinned values | **1.5e-13** | intended; five decades inside `NESTED`, absorbed by the tightened 1e-9 |
| `hazma.spectra.dnde_photon_neutral_rho` | 1,395 corpus-pinned values | **3.2e-15** | intended |
| `hazma.spectra.dnde_photon_charged_rho` | 3,200 off-corpus points, 8 parent energies | **2.5e-11** | intended; explained above (π⁰ box edge inside the window) |
| `hazma.spectra.dnde_photon_neutral_rho` | 3,200 off-corpus points, 8 parent energies | 4.9e-13 | intended |

The charged ρ's 2.5e-11 is beyond rule 3's 1e-12 declaration threshold,
so it is recorded here, in `../README.md`'s "Numerical impact so far",
and belongs in the PR body. Nothing else in the tree moved: the diff
touches no other kernel, and `pytest -q` is green at the same counts
modulo this task's own tests.

Command:
`.venv/bin/python <scratchpad>/corpus_drift.py` and `<scratchpad>/drift.py`,
both run with `hazma._core` registered and `_rho.pyx` still on disk.

## Open Questions

- **Does the ρ's own forward-cone loss need a separate repair, or does a
  restricted outer interval fall out of the inner fix?** Measured that it
  needs its own (Findings); the *shape* of the repair is left to the
  followup, which now carries the table. Blocked on the same corpus
  regeneration as the other six defects
  ([`parity-corpus-pins-ill-conditioned-points.md`](../../../../docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md)).
- **Seven blocked defects now share one regeneration**, up from six.
  Worth telling the maintainer separately from this project's schedule —
  three of the seven affect the *shape* of a spectrum rather than a
  total, which is what a limit calculation notices.
- **Whether to build a `hazma._core` probe for the kernel layer** —
  Task 4.4 left this decision here. **Answered: not needed.** The
  `boost_window` seam gave the untestable arithmetic a `cargo`-level test
  without widening `_CORE_TEST_ONLY_MODULES`, which is cheaper than a
  probe and does not enlarge the public surface Task 3.2 warned about.

## Plan Impact

**Impact Level:** None (task note only).

Canonical-contract diff, run against the phase file and all three ADRs:

- `phases/phase-04-spectra-kernels.md` Task 4.5's two exit-criterion
  bullets are both discharged as written and neither is now factually
  wrong — the nested integral got its drift analysis, and the untyped
  `cdef` locals are plain `f64` with no value shift beyond budget (the
  shift is 1.5e-13 against a 1e-9 budget).
- The phase Goal's capi-survivor list does not name `_rho`, and the
  sentence "All other twins (rho, kaon, eta family, neutrino pair) delete
  in their swap PR as usual" is exactly what happened.
- The per-kernel swap recipe's eight steps were followed in order and
  none needs amending. Step 1's warning ("read it, do not pattern-match")
  paid for itself here: the expected answer was "some FMAs" and the
  actual answer was none.
- ADR-0001, ADR-0002, ADR-0003 are untouched by this task.
- `rules.md` rules 1–4 and 9 all held without exception.

`PLAN.md`'s phase table row for 04 still reads "16 entry points swapped;
twins deleted (4 capi survivors defer to 06)", which remains true — 13 of
the 16 are now served.

## Stale-state sweep

### Identifier sweep

```console
$ rg -n '_rho\.pyx|_rho\.pxd|_rho\.pyi|_photon\._rho|_photon/_rho' \
    --glob '!projects/**' --glob '!.venv/**' hazma rust test setup.py docs
docs/followups/todo/charged-pion-photon-spectrum-misses-the-forward-cone.md:115  (historical + the new measured table)
docs/followups/todo/rho-rest-frame-branch-returns-the-integrand.md:33,38          (the deleted source, cited as origin)
rust/src/kernels/photon_rho.rs:2,47,48,74,109,262                                 (provenance citations)
rust/src/kernels/photon_pion.rs:8                                                 (past tense: "before Task 4.5 deleted it")
rust/src/quad.rs:38                                                               (call-site table, marked "(ported, Task 4.5 …)")
rust/src/kernels.rs:13                                                            (module roster)
test/parity/cases.py:1412,1425,1429,1433                                          (PORTED_ENTRY_POINTS origins + corrected comment)
test/parity/tolerances.py:265,267,275,277                                         (Budget.why citations; the file's "Citations" section already declares these are evidence, not live refs)
test/test_core_photon_pion.py:14,453                                              (cimporter roster, corrected)
test/test_core_quad.py:492,523                                                    (marked ported)
test/test_core_constants.py:85                                                    (the removal, with its reason)
```

No live import, no build entry, no dispatch-table row still names the
module. `rg 'import.*_rho|from.*_rho'` over `hazma/` and `test/` returns
nothing. `docs/agents/environment.md` gained one more mention when the
stale-`.so` trap was recorded; it is a worked example, not a reference.

### Deletion proof (rules.md process rule 1, verify-before-delete)

```console
$ rg -n 'cimport.*_rho|from hazma.spectra._photon._rho' hazma/
(no matches)
$ .venv/bin/python -c "import hazma.spectra._photon._rho"
ModuleNotFoundError: No module named 'hazma.spectra._photon._rho'
$ .venv/bin/python -c "import hazma.scalar_mediator, hazma.vector_mediator; print('cimporters import')"
cimporters import
```

`_rho.pyx` was the only cimporter of `_pion`'s `*_point` `cdef`s outside
the mediator modules; both mediator modules still import and run
(covered by `test_the_still_cython_dependents_import_and_run` in
`test/test_core_photon_pion.py`, green in the 1802).

### Count sweep

```console
$ find hazma -name '*.pyx' | wc -l   # 15 -> 14
14
$ find hazma -name '*.pxd' | wc -l   # 12 -> 11
11
$ .venv/bin/python -c "import sys; sys.path.insert(0,'test/parity'); import cases; print(len(cases.rust_core_kernels()))"
13
$ .venv/bin/python -m pytest test/test_core_photon_rho.py --collect-only -q | tail -1
49 tests collected in 0.31s
$ cargo test --manifest-path rust/Cargo.toml --no-default-features | grep 'test result: ok' | head -1
test result: ok. 133 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

Every count quoted in this note and in the two working-memory READMEs is
one of these. `hazma._core` now serves **13** kernels, from 11.

### Forward-looking phrase sweep

```console
$ rg -n 'Task 4\.5' --glob '!projects/**' --glob '!.venv/**' .
rust/src/kernels/photon_rho.rs   (this task's own provenance, past tense)
rust/src/kernels/photon_pion.rs:8
rust/src/photon.rs:16
rust/src/quad.rs:38
test/parity/cases.py:1424
test/parity/tolerances.py       (NESTED class docs, Budget.why x2)
test/test_core_photon_rho.py    (module docstring)
docs/followups/todo/*.md x2
```

Every one is past tense or a provenance citation. The three sites that
said "Task 4.5 will …" before this task — `rust/src/photon.rs:16`,
`rust/src/kernels/photon_pion.rs:6`, `test/test_core_dispatch.py:713` —
were all rewritten.

### Lint and gate sweep

```console
$ scripts/agents/preflight.sh --paths "<the 9 .py files>" --md "<the 6 .md files>"
RESULT: PASS
```

Black, isort and ruff are clean on all nine changed `.py` files; the
tree-wide ruff/isort debt is the trunk's and this branch reduces it (table
under Verification). `markdownlint` is green on all six changed `.md`.

### Numerical-impact statement

`spectra.photon.charged_rho` moved by up to **1.5e-13** relative on the
1,395 values the parity corpus pins and up to **2.5e-11** on a denser
off-corpus sweep; `spectra.photon.neutral_rho` by up to **3.2e-15** and
**4.9e-13** respectively. Both budgets were **tightened**, not widened,
from `NESTED_RTOL` (1e-6) to `PORTED_NESTED_RTOL` (1e-9). Recorded in
`../README.md`'s "Numerical impact so far".

### Exit Criteria → test mapping

| Criterion | Evidence |
| --- | --- |
| Both ρ entry points corpus-green | `pytest test/parity -q` inside the 1802; 10 ρ blocks at the tightened 1e-9 |
| Dedicated drift analysis for the nested integral | Findings, "The drift" — two tables, the outlier explained against scipy's own `abserr` |
| Untyped `cdef` locals ported as plain f64, no shift beyond budget | Findings, "The FMA map is empty"; `the_kernel_contracts_nothing_because_the_pyx_boxes_its_locals`; `the_boost_window_is_computed_without_contraction`; measured shift 1.5e-13 vs a 1e-9 budget |
| Recipe step 1 (FMA map before writing Rust) | `objdump` output quoted, run against the shipped `.so` before the port |
| Recipe step 2 (PyO3-free kernel module) | `rust/src/kernels/photon_rho.rs`; `cargo test --no-default-features` links |
| Recipe step 3 (dispatch + wording) | `rust/src/photon.rs`; `TestDispatchWiring` (20 tests) |
| Recipe step 4 (wrapper repointed) | `hazma/spectra/_photon/__init__.py`; `TestWrapperAndPublicApi` |
| Recipe step 5 (corpus repointed + `PORTED_ENTRY_POINTS`) | `test/parity/cases.py`; `assert_full_coverage` green |
| Recipe step 6 (twin deleted) | Deletion proof above |
| Recipe step 7 (per-kernel test module) | `test/test_core_photon_rho.py`, 49 tests |
| Recipe step 8 (drift recorded) | This note + `../README.md` |

### Task-note self-consistency

`**Status:** Complete` here matches the `Complete (2026-08-18)` cell in
`phase-04/README.md`'s Tasks table. Phase 04 is **not** complete —
Task 4.6 remains — so the phase file frontmatter stays `In Progress`, its
`PLAN.md` row is untouched, and no phase learnings are written.

## Handoff to Next Task

**Task 4.6 (`_positron/_pion` + the neutrino pair) is next**, and it
closes Phase 04. Six things to carry across:

1. **`hazma._core` serves 13 kernels**, and with the two ρ entry points
   the **photon domain is finished**: all 12 public
   `hazma.spectra.dnde_photon_*` decay spectra now come from Rust. What
   remains in Phase 04 is one positron entry point and three neutrino
   ones (`test/parity/cases.py`'s `rust_core_kernels()` → 13, one of them
   `positron.dnde_positron_muon` from Task 4.1).
2. **The `NESTED` class is measured now.** `PORTED_NESTED_RTOL = 1e-9`
   exists and two of the nine `NESTED` cases take it; the seven mediator
   spectrum cases keep 1e-6 for Phase 06. `dnde_neutrino_charged_pion`
   also nests (it boosts the muon spectrum) — the ρ's result says expect
   ~1e-13, not ~1e-6, but **re-derive rather than inherit**: that is the
   third time in a row a phase prediction was wrong in a different
   direction.
3. **`test/test_core_dispatch.py`'s spectra oracle is
   `_positron/_pion`,** which Task 4.6 deletes the `def` of. There is no
   fourth unary candidate — the neutrino entry points return a 3-tuple or
   a `(3, N)` array. The class docstring names the two options
   (rewrite around the neutrino shape, or retire). `TestCythonMessageParity`
   is **not** affected: its `"Photon energies"` roster entry lives in
   `hazma/spectra/_neutrino/_muon.pyx:205`, which survives Phase 04.
4. **When a mutation survives a quadrature-backed kernel's gates, look
   for a seam before declaring it untestable.** Task 4.4 concluded its 15
   in-integrand FMA sites were unobservable and it was right; Task 4.5
   found its one survivor was arithmetic in the integration *limits*,
   lifted it into `boost_window`, and pinned it bit-for-bit. Run the
   mutation campaign, then ask that question about each survivor.
5. **The blocked-defect count is seven**, not six — the ρ rest-frame
   branch joined the list, and the charged-pion forward-cone entry now
   records that the ρ **compounds** it (0.945 of endpoint predicted,
   0.537 measured at `γ_ρ = 10`) and needs its own repair. Read both
   before Phase 06.
6. **`_positron/_pion.pyx` and `_photon/_pion.pyx` are the two capi
   survivors Task 4.6 touches**, so its `def`-only deletion is the same
   shape Tasks 4.3/4.4 used — not the whole-file deletion this task did.
   `assert_full_coverage` enforces the difference.

**Currently safe to assume:** the `_photon` package has **no** Python
entry point in Cython; `_muon.pyx` and `_pion.pyx` survive there for
their `cdef` capsules alone, which only the two mediator decay-spectrum
modules now read. **14 `.pyx` and 11 `.pxd`** in the tree — re-derive
with a clean rebuild rather than quoting.
