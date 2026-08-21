# Task 4.6: `_positron/_pion` + neutrino pair (`_muon`, `_pion`, struct)

**Date:** 2026-08-20
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-04-spectra-kernels.md` (Goal, the
per-kernel swap recipe, Task 4.6); `../../rules.md` rules 1–3, 9;
`../../PLAN.md` (Numerical impact)
**Related ADRs:** none (ADR-0001/0002/0003 all Accepted, none gating)
**Depends On:** Task 4.1 (`positron_muon`, the pion's integrand),
Task 4.3 (the `.pyx` cimport DAG's ordering)

## Objective

Port the last three `hazma/spectra/**` entry points —
`dnde_positron_charged_pion`, `dnde_neutrino_muon` and
`dnde_neutrino_charged_pion` — to `hazma._core`, turn
`NeutrinoSpectrumPoint` into a plain Rust struct, and close Phase 04.

## Exit Criteria

From the phase file, verbatim:

- [x] `dnde_positron_charged_pion`, `dnde_neutrino_muon`,
      `dnde_neutrino_charged_pion` corpus-green.
- [x] The `NeutrinoSpectrumPoint` struct becomes a plain Rust struct.
- [x] Tuple / `(3, N)` return contract verified against existing wrapper
      tests.
- [x] Neutrino/rho/kaon/eta-family twins deleted; only the four capi
      survivors plus `_utils` headers remain as `.pyx`/`.pxd` under
      `spectra/` + `_utils/`.

## Inputs Reviewed

- `../../PLAN.md` — Goal, Scope, Numerical impact, Phases table
- `../../phases/phase-04-spectra-kernels.md` — Goal (the capi-survivor
  exception), the eight-step swap recipe, Task 4.6
- `../README.md` (phase working memory) and `../../task-notes/README.md`
- `../../rules.md` rules 1–3 (parity discipline), 4 (constants), 9 (edge
  guards)
- `../../references/cython-inventory.md` — the cimport DAG
- Task notes 4.1, 4.4, 4.5 (the two-oracle-shape decision, the
  `boost_window` lift, the mutation-campaign discipline)
- `docs/agents/lessons.md`, `docs/agents/environment.md`
- The three `.pyx` and their `.pxd`, plus
  `hazma/_utils/{boost,kinematics}.pxd`
- `test/parity/{cases,tolerances}.py`, `test/parity/oracles/`
- `test/test_core_{positron_muon,photon_pion,photon_rho,dispatch,constants}.py`

## Findings

### The FMA map (recipe step 1)

Disassembled the shipped `cpython-312-darwin.so` for each of the three,
`objdump -d … | grep -c 'fmadd\|fmsub\|fnmadd\|fnmsub'`:

| extension | FMAs | where |
| --- | --- | --- |
| `_positron/_pion` | **2** | `dnde_positron_charged_pion_point`: `fmsub d0, d9, d12, d8` and `fmadd d0, d9, d12, d8`, i.e. `E ∓ β·k` |
| `_neutrino/_muon` | **14** | all in `c_muon_decay_spectrum_point` (the rest-frame helper is inlined): 3 rest-frame, 4 boosted electron, 7 boosted muon |
| `_neutrino/_pion` | **0** | nothing to contract — see below |
| `_neutrino/_neutrino` | **0** | the struct module is three stores |

`_neutrino/_pion`'s zero is **not** the `_rho.pyx` story. Task 4.5's rho
contracted nothing because its locals were untyped and Cython boxed them;
every local here is a `cdef double`, and the file simply contains no
`a*b + c`: `1 − β²`, `γE(1∓β)`, `0.5/(γβ)` and the `two_body_energy` folds
are all plain `fmul`/`fdiv`/`fsub` chains. **The two files reach the same
answer for different reasons**, and the phase's standing advice — read the
disassembly, do not pattern-match — held again.

Four expressions that look fusable and are not, each read off the same
disassembly: `e**2 - me**2` (both files), `1 - (m/E)**2` inside every
inlined `boost_beta`, `xm**2 + 3r⁴` and `xp**2 + …` in the boosted
electron polynomial, and `gam**2 * x * (1 ∓ beta)`.

### `emax_pi_rf` is folded **with** contraction

`_positron/_pion.pyx`'s three module-level `cdef double`s (`beta_mu`,
`emax_mu_rf`, `emax_pi_rf`) are computed at module init from `DEF`
constants, and clang folds all three: `__pyx_pymod_exec__pion` stores one
immediate, `0x4051_724f_f60e_5ca3`, and the other two never materialise.

The stored double is **one ulp above** what the unfused expression gives.
Brute-forcing the eight fused/unfused combinations of
`gamma_mu * emax_mu_rf * (1.0 + beta_mu * sqrt(1.0 - (me/emax_mu_rf)**2))`
against the immediate isolates it: only `1.0 + beta_mu * root` fused as an
FMA reproduces `…5ca3`; the `sqrt` arguments' fusion state does not
matter. `kernels::positron_pion::EMAX_PI_RF` is therefore a literal, and
`the_endpoint_constant_matches_the_shipped_object_code` re-derives it with
`mul_add` **and** asserts the unfused spelling lands one ulp low.

A physical fact fell out: `emax_pi_rf` and the `DEF` `eng_e_pi_rf` are
**adjacent doubles** — the muon channel's positron endpoint and the
two-body `π → e ν` line are the same energy, because the most energetic
positron from `π → μ ν → e ν ν ν` is emitted forward at every step. The
port keeps both spellings distinct; collapsing them is a defensible
simplification and a corpus failure.

### `scipy.integrate.quad` short-circuits an empty interval; `crate::quad` did not

`dnde_neutrino_charged_pion(0.0, epi)` returned `(0, 0, 0)` from the
Cython and `(nan, nan, 0)` from the first version of the port. The
integrand is `(dN/dE)_μ(E)/E`, which is `0/0` at the origin, and at
`E_ν = 0` the boost window collapses onto it.
`scipy/integrate/_quadpack_py.py:436` returns `(0., 0.)` for `a == b`
**before** the limits are ordered and before QUADPACK is reached — the
integrand is never called (verified: a counting integrand records 0
calls). `crate::quad::quad`
handed `[0, 0]` to `qagse`, where every Gauss-Kronrod node collapses onto
the point and `f(x)·0` is `NaN`.

Fixed in `crate::quad::quad` rather than worked around in the kernel: it
is a fidelity gap in the Task 3.3 port, this is simply its first live call
site, and the fix can only ever turn a `NaN` into scipy's `0`. `a == b` is
false for a `NaN`, so `NaN` limits still fall through, as in scipy.
`quad.rs`'s `a_zero_width_interval_integrates_to_zero` had passed the
whole time because its integrand was `exp` — smooth, so `f(x)·0` is `0`
either way. It is now
`an_empty_interval_returns_zero_without_evaluating_the_integrand`, with a
call counter and a singular integrand.

### The neutrino muon spectrum normalises **correctly**, unlike its positron sibling

`_neutrino/_muon.pyx` writes `common = R_FACTOR * x² …` — the Michel
normalization as a **factor**. `_positron/_muon.pyx:28` **divides** by the
same literal, which is the inverted-normalization defect Task 4.1 filed.
So both neutrino rows integrate to exactly 1 (measured: 1.0000000000054
by Simpson on 100,001 panels) while the positron spectrum is low by
`1/N²` = 0.0374%.

This is a trap for anyone who meets the positron defect first, so it is
asserted from both sides:
`kernels::neutrino_muon::both_flavors_integrate_to_one_neutrino_each`
pins the 1 *and* checks that the positron deficit is two decades outside
its own budget, and `TestPhysics` in `test/test_core_neutrino.py` says the
same in prose. The two files really do disagree and only one is wrong.

### A new pinned defect: the `π → e ν` neutrino line is counted twice

`hazma/spectra/_neutrino/_pion.pyx:196-200` sums `c_dnde_mu_numu_point`
and `c_dnde_e_nue_point`, and **both** add the boosted `π → e ν_e` line
(same file, `:112-114` and `:167`). Every line number here is at
`ed1fa20`, this branch's merge-base — the file is deleted by this task, so
read it with `git show ed1fa20:hazma/spectra/_neutrino/_pion.pyx`. The
electron-neutrino row therefore carries `2 BR(π → e ν)`. The muon row is
unaffected — `c_dnde_e_nue_point` writes nothing there — which is what
makes it a transcription slip rather than a convention.

Measured at `E_π = 400` MeV by subtracting the muon-decay continuum
recomputed with `scipy.integrate.quad` over the ported muon kernel: the
excess is **exactly 2.0000** copies of one line's plateau at `E_ν` = 20,
30 and 50 MeV, against **1.0000** for the muon line at the same points.
Integrated, the electron-neutrino yield is overweight by `1.23e-4` per
pion (0.0123%); locally, the plateau the line sits on is 0.062% high at
`E_π = 200` MeV and 0.036% at 1000 MeV.

Reproduced under rule 1 and filed as
[`docs/followups/todo/neutrino-pion-electron-line-counted-twice.md`](../../../../docs/followups/todo/neutrino-pion-electron-line-counted-twice.md).
Its repair needs no Cython oracle — the excess is a closed-form plateau
over a computable window — so deleting the twin in this PR costs the
`parity-pinned-defect-repair` project nothing.

A second, softer asymmetry is recorded in the same follow-up's Risks
section rather than filed separately: a pion **at rest** loses *both*
prompt lines, because a delta function has no rest-frame representation in
this API. Deciding what it should return is a design question, not a
transcription fix.

### x86-64's baseline has no FMA, so the shipped Cython is unfused there

**Found by CI, not locally** (PR #74 round 1: green on macOS/arm64, red on
all five Linux jobs). `test_the_kinematic_edges_match` swept up to
`E_π = 1e6` MeV and failed at 7.5e-9 relative plus a delta-function
*branch flip*, against a 1e-12 budget measured on macOS.

The port is not wrong. The kernel is ill-conditioned there and the two
builds differ in their FMA:

- the boost integral runs from `emin = γ(E − βk)`, and as `β → 1` that
  difference falls like `E/(2γ²)` while both terms stay `O(E)` — so
  `emin`'s *relative* error grows like `2γ²ε`, which is **2.3e-8** at
  `E_π = 1e6` (γ = 7165);
- clang contracts `E − β·k` into an `fmsub` on arm64, and **cannot** on
  x86-64 without `-march`, because SSE2 has no FMA. So the shipped Cython
  is fused on the capturing platform and unfused on Linux, while the
  port's `mul_add` is fused on both. One ulp, amplified by `2γ²`.

Locally, macOS/arm64 agrees to 7.9e-16 at that same point — which is why
no local run could have caught it. `test_core_positron_muon.py`'s
docstring records the same asymmetry for its own kernel; this task's
mistake was writing a module docstring that said "one budget, no platform
branch" and then deriving that budget from one platform's measurement.

**Fixed by bounding the claim, not by widening the budget.** Every grid in
`test/test_core_positron_pion.py` now stops at `E_π = 1e4` MeV — 71x the
pion mass, seven times what the corpus samples, and far past hazma's
sub-GeV domain — and a new
`TestPhysics.test_the_boost_window_is_ill_conditioned_at_extreme_boosts`
asserts the *mechanism* (the `E/(2γ²)` cancellation, its quadratic
envelope, and that the envelope at 1e6 is four decades outside the budget)
rather than a value. The envelope is a bound on what can propagate, not
what does: it is already 2.3e-12 at `E_π = 1e4`, where every sweep passes
on Linux at 1e-12, because the integrand vanishes at its own threshold and
damps a wobble in the lower limit. What made 1e6 different is that the
wobble also crossed `eminus < e0 < eplus`, turning a rounding difference
into a *support* difference — which no tolerance should absorb.

### `_positron/_pion` returns zero at rest, unlike every sibling

`if fabs(epi - mpi) < DBL_EPSILON: return 0.0` — where the photon, rho and
positron-muon kernels return a rest-frame *value*. And the two-sided
`fabs` is inoperative: `epi < mpi` has already cut the lower half off, and
`DBL_EPSILON` is an absolute MeV threshold against a 139.57 MeV mass whose
ulp is 2.8e-14 — 128x larger — so the only double the guard can see is
`m_π` itself. Recorded in the kernel docs and pinned by
`the_near_rest_guard_admits_exactly_the_pion_rest_mass`.

### `derived::neutrino_muon` retires with its `.pyx`

`test/test_core_constants.py` scores `constants::derived::` against the
surviving `.pyx`, so a submodule whose source file is gone cannot stay.
Unlike Task 4.5's `derived::photon_rho` — whose three `DEF`s were bare
`pdg` aliases and simply vanished — this one's five (`R`, `R2`, `R4`,
`R6`, `R_FACTOR`) are arithmetic the kernel needs, so they **moved** into
`rust/src/kernels/neutrino_muon.rs` beside the constants clang folds out
of them. `R_FACTOR` now appears as a literal in two Rust files, and
`test_r_factor_is_the_michel_normalization_over_the_pdg_ratio` reads both
so they cannot drift apart.

### `test_core_dispatch.py`'s Cython oracle had to move, and where to was forced

`TestDeclaredDivergencesFromCython` needs a live `.pyx` `def` with the
unary `hasattr(__len__)` / `assert` / array-return shape. That was
`_positron/_pion` until this task. The class docstring offered "rewrite
them around the neutrino shape" as the alternative — but this task swaps
the neutrino entry points too, so `hazma/spectra/` has no top-level `def`
left at all.

The oracle is `scalar_mediator_decay_spectrum` now: the *identical*
dispatch shape on its first argument
(`scalar_mediator_decay_spectrum.pyx:268-271`), differing only in three
extra arguments and in the quantity its message names. It survives until
Phase 06. The message roster shrank with the tree, from four `assert`
wordings to two; the port still emits `"Positron energies"` and
`"Neutrino energies"`, and each is now pinned in its own kernel's test
module rather than in the roster.

## Decisions and Implementation Notes

- **`kernels::neutrino_flavors`, not `kernels::neutrino_neutrino`.** The
  convention is one submodule per ported `.pyx` named for it; applied
  literally to `_neutrino/_neutrino.pyx` it doubles, and shortening to
  `neutrino` would collide in the reader's eye with `crate::neutrino`, the
  PyO3 registration module. Second documented exception after
  `photon_tables`, and it says so in its own docs. The struct keeps the
  Cython's name, `NeutrinoSpectrumPoint`, and its field order, because
  that order is the row order of the published `(3, N)` array.
- **`new_neutrino_spectrum_point()` becomes a `const`**,
  `NeutrinoSpectrumPoint::ZERO`, so the "start from zero and fill what
  applies" idiom costs nothing and cannot be forgotten. Pinned as three
  *positive* zeros — a `-0.0` constructor would pass any `==` while
  changing what a below-threshold spectrum stores, and the corpus treats a
  stored zero as exact.
- **`Flavor` is an enum, not the `.pyx`'s `int gen`.** The two `quad` call
  sites differ only in which row the integrand returns, and transposing
  them swaps two rows of the published array — invisible to any tolerance.
  `the_integrand_rows_are_distinguishable` and the transposition mutation
  both depend on the rows being nameable.
- **`crate::quad` gained the empty-interval short circuit** rather than
  the kernel gaining a guard. See Findings.
- **`neutrino_pion::boost_window` was lifted out** for the reason Task
  4.5 lifted the rho's: it was the module's only directly observable
  arithmetic, and the mutation campaign found it unobservable otherwise.
  See "Mutation campaign" below.
- **The two new test modules take different shapes, and both are forced.**
  `test/test_core_positron_pion.py` follows
  `test/test_core_photon_pion.py`'s charged pion — one budget on every
  platform, no bit-equality mode — because the twin survives as a capi
  provider but the port replaces *scipy's* QUADPACK with the in-tree one,
  and two adaptive integrators are not bit-equal anywhere.
  `test/test_core_neutrino.py` follows `test/test_core_photon_rho.py` —
  independent Python references — because its twins are gone.

- **Review round 1 (blocking): every citation into a file this task
  deletes now carries the full path and the revision.** The note cited
  `hazma/spectra/_neutrino/_pion.pyx` by bare basename plus a line range,
  and `scripts/agents/check_doc_citations.py --changed-vs origin/master`
  fails that — two files of that basename survive in the tree
  (`_photon/` and `_positron/`) and the one meant, `_neutrino/`, is gone.
  Fixed as a class rather than at the cited line: all seven citations
  into `hazma/spectra/_neutrino/{_muon,_pion}.pyx` across four docs now
  read `<full path>:<lines>` plus `at ed1fa20` (this branch's
  merge-base), including three that pre-dated this task and that *this
  task's deletion* is what made unresolvable. A line-wrapped
  `scipy/integrate/_quadpack_py.py:436` was un-wrapped for the same
  reason — the wrap left a bare basename on its own line.
- **The same sweep found a claim the deletion falsified.**
  `task-notes/phase-04/README.md` said `TestCythonMessageParity`'s
  `"Photon energies"` roster entry "survives Phase 04". It does not: the
  entry lives in `hazma/spectra/_neutrino/_muon.pyx:205` (at `ed1fa20`),
  which this task deletes. The
  bullet now says what actually happens — the roster shrinks with the
  tree, to two wordings, and the two the port still emits are pinned in
  each kernel's own test module.

## Files Changed

### Rust

- `rust/src/kernels/positron_pion.rs` — **new.** The charged-pion positron
  spectrum: one `quad` over the positron's rest-frame energy with
  Task 4.1's Michel kernel as the integrand, plus the boosted `π → e ν`
  line.
- `rust/src/kernels/neutrino_flavors.rs` — **new.**
  `NeutrinoSpectrumPoint`, ported from `_neutrino/_neutrino.pyx`.
- `rust/src/kernels/neutrino_muon.rs` — **new.** The muon's two neutrino
  spectra, closed form, plus the five `DEF` constants
  `derived::neutrino_muon` used to hold.
- `rust/src/kernels/neutrino_pion.rs` — **new.** The charged pion's
  neutrino spectra: two prompt lines plus two `quad`s over the muon
  spectrum, with `boost_window` lifted out and pinned.
- `rust/src/kernels.rs` — register the four; document the second naming
  exception.
- `rust/src/positron.rs` — register `dnde_positron_charged_pion`.
- `rust/src/neutrino.rs` — register both neutrino entry points through
  `map_flavors`; record why the wording diverges from the twin's.
- `rust/src/quad.rs` — the `a == b` short circuit, and the strengthened
  test that pins it.
- `rust/src/constants.rs` — retire `derived::neutrino_muon`; repoint the
  two unit tests that read it.

### Python surface

- `hazma/spectra/_positron/__init__.py` — wrapper on
  `hazma._core.positron`.
- `hazma/spectra/_neutrino/__init__.py` — wrapper on
  `hazma._core.neutrino`.
- `hazma/spectra/_positron/_pion.pyx` — `def` deleted, `cdef`s kept (capi
  survivor).
- `hazma/spectra/_positron/_pion.pyi` — deleted.
- `hazma/spectra/_neutrino/_{muon,pion,neutrino}.{pyx,pxd,pyi}` — deleted
  (8 files).
- `setup.py` — drop the `_neutrino` extension block.

### Tests

- `test/test_core_positron_pion.py` — **new**, 47 tests.
- `test/test_core_neutrino.py` — **new**, 58 tests.
- `test/parity/cases.py` — repoint the three cases at their wrappers; add
  three `PORTED_ENTRY_POINTS` rows.
- `test/parity/tolerances.py` — tighten both `QUAD` spectra cases to
  `PORTED_QUAD_RTOL`; record the measurements.
- `test/test_core_quad.py` — mark the two now-ported call-site
  citations as provenance rather than live sites.
- `hazma/_core.pyi` — refresh the per-submodule comment, which
  enumerated kernels and had gone stale two tasks ago.
- `test/test_core_dispatch.py` — move the Cython oracle to
  `scalar_mediator_decay_spectrum`; shrink the message roster.
- `test/test_core_constants.py` — retire the `derived::neutrino_muon`
  row, lower `FLOOR_DERIVED`, pin `R_FACTOR` in both Rust files.

### Docs / cross-project

- `docs/followups/todo/neutrino-pion-electron-line-counted-twice.md` —
  **new**, plus its index row in `docs/followups/README.md`.
- `test/parity/oracles/entry_points.py`,
  `projects/parity-pinned-defect-repair/PLAN.md` and
  `.../references/defect-blast-radius.md` — all three said Task 4.6
  deletes `hazma/spectra/_positron/_pion.pyx`. It does not: the file is a
  capi survivor and dies at Phase 06 Task 6.4. Corrected in place; the
  correction *relaxes* that project's earliest deadline, and its Task 2
  capture had already landed anyway.

## Verification

### Gates

```console
$ cargo fmt --manifest-path rust/Cargo.toml --check
$ cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 3.38s
$ cargo test --manifest-path rust/Cargo.toml --no-default-features
test result: ok. 169 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
$ .venv/bin/python -m pytest -q
1935 passed, 15 skipped, 7 warnings in 151.28s (0:02:31)
$ .venv/bin/python -m pytest test/parity -q -p no:randomly
658 passed, 1 skipped in 77.75s
$ .venv/bin/python -m pytest test/test_core_positron_pion.py test/test_core_neutrino.py -q -p no:randomly
106 passed
$ .venv/bin/python -m pytest test/test_theory_aggregation.py -q -p no:randomly
69 passed
$ env PATH="$PWD/.venv/bin:$PATH" scripts/agents/preflight.sh --paths "<9 .py>" --md "<10 .md>"
RESULT: PASS          # all eleven rows; only "version bump" SKIPs (not a closing PR)
```

**`--paths` excludes the two wrapper `__init__.py` files, and the reason
is the trunk's, not this change's.** `hazma/spectra/_{positron,neutrino}/
__init__.py` carry long-standing ruff debt — `Optional`/`Union`
annotations, docstring rules, an unused local — and the gate asserts
absolute cleanliness rather than comparing against the merge base, which
is the standing
[`preflight-isort-ruff-red-on-trunk.md`](../../../../docs/followups/todo/preflight-isort-ruff-red-on-trunk.md)
follow-up. That file's own instruction is to prove the red rows are
pre-existing; measured side by side with the same tool versions:

| gate | `origin/master` | this branch |
| --- | --- | --- |
| `ruff check hazma/spectra/_neutrino/__init__.py` | 50 | **49** |
| `ruff check hazma/spectra/_positron/__init__.py` | 24 | 24 |
| `ruff check hazma test` (whole tree) | 6135 | **6127** |
| `isort --check-only hazma test` (ERROR lines) | 97 | **92** |
| `black --check hazma test` | clean | clean |

Every remaining finding in those two files pre-dates this task, the
branch is strictly better on three of the five rows and equal on the
other two, and the nine `.py` files this task actually authored or
substantially rewrote are clean on all three linters. `ruff --fix` would
close the two wrapper files' debt in one pass, but it rewrites 64
annotations in two public modules for no reason this task supplies — that
is the follow-up's option 1, not this task's business.

`1935 passed / 15 skipped`, from `1831 / 15` on `origin/master` — **+104**,
and the arithmetic is `48 + 58 − 2`:

| Module | before | after | why |
| --- | --- | --- | --- |
| `test/test_core_positron_pion.py` | — | 48 | new |
| `test/test_core_neutrino.py` | — | 58 | new |
| `test/test_core_constants.py` | 23 | 21 | two parameterized rows retired with `derived::neutrino_muon` |
| `test/test_core_dispatch.py` | 118 | 118 | the oracle module was swapped, not the assertions |
| `test/parity` | 659 | 659 | cases repointed, none added or dropped |
| whole suite | 1831 | 1935 | **+104** |

`cargo test --no-default-features` goes **133 → 169**, and the +36 is
exactly the four new kernel modules' own tests (`cargo test | grep -c
'^test kernels::neutrino_flavors::'` → 2, `neutrino_muon` → 10,
`neutrino_pion` → 15, `positron_pion` → 9).

### What the 106 per-kernel tests cover

**`test/test_core_positron_pion.py` (48)**

- **`TestDispatchWiring` (11)** — one assertion per contract branch:
  scalar → `float`, NumPy scalar and 0-d array on the scalar path, array →
  fresh `float64` array, array path bit-equal to the scalar path, a
  sequence, an empty grid, the rank message verbatim, a non-`float64`
  dtype, a non-number, and both arguments by keyword.
- **`TestWrapperAndPublicApi` (7)** — the private wrapper's bytes, the
  public `hazma.spectra` name, the `def` gone, the two capsules intact and
  under the expected C signature, the `_nbody` dispatch table, and both
  mediator positron modules still importing.
- **`TestAgainstTheCythonTwin` (18)** — the surviving `cdef` through
  `__pyx_capi__`, at 7 pion energies × (swept grid + random arguments),
  plus the kinematic edges at 4 more, the support comparison, the budget's
  own non-vacuity, the at-rest zero, and a `NaN` in either argument.
- **`TestPhysics` (12)** — the three thresholds, the boosted endpoint
  against `γ(1+β) E_rf`, finiteness and non-negativity at 4 pion energies,
  positron-number conservation at 3 (including that the total is the
  *shipped* `BR_μ/N² + BR_e` and not the un-defected one), the electron
  line's plateau located by its window edge, the peak falling with the
  boost, and — added after PR #74's first CI round — the boost window's
  ill-conditioning at extreme boosts, which is what bounds every grid in
  the module.

**`test/test_core_neutrino.py` (58)**

- **`TestDispatchWiring` (20)** — the same roster as above for **both**
  entry points, with the two shapes only these kernels have: a 3-tuple of
  `float` for a scalar and a `(3, N)` `float64` array for a grid.
- **`TestFlavorSelection` (5)** — each `flavor=` string selects its own
  row, the rows are pairwise distinguishable, and an unknown flavor
  raises. The only place the row order is user-visible.
- **`TestWrapperAndPublicApi` (6)** — the wrappers' bytes, the public
  names, the three twins gone from the source tree *and* from `setup.py`,
  the `_nbody` table, and the package's CSV-driven siblings still working.
- **`TestAgainstAnIndependentReference` (16)** — the muon against a
  Python transcription at 6 parent energies, the pion against a scipy
  boost integral over the ported muon kernel at 6 more, the support
  comparison, both budgets' non-vacuity, and that a permuted row order
  would be caught.
- **`TestPhysics` (11)** — thresholds, the never-written tau row, the
  muon's one-neutrino-per-flavor integral at 4 parent energies (with the
  positron sibling's defect shown to be outside the bound), the pion's
  `2 BR_μ` muon yield, the doubled electron line and the single muon line
  by continuum subtraction, the at-rest branch's missing lines, the
  zero-energy zero, finiteness at 3 energies, and the peak falling with
  the boost.

### CI

PR #74 round 1: `Lint`, `Rust (fmt, clippy, test)` and
`Test (macos-latest, py3.14)` green; all five `Test (ubuntu-latest, …)`
jobs red on one assertion,
`test_core_positron_pion.py::TestAgainstTheCythonTwin::test_the_kinematic_edges_match`
at `E_π = 1e6` — 3 of 10 edges, worst 7.5e-9 relative and two of them a
zero-vs-1.23e-10 support flip. Diagnosed as the kernel's own
ill-conditioning plus x86-64's missing baseline FMA (see Findings), and
resolved by bounding the module's grids to `E_π = 1e4` and asserting the
mechanism. Nothing in `test/parity`, `test/test_core_neutrino.py` or
`cargo test` was red in that round.

### Test validity (stash-proof)

- **The `crate::quad` short circuit.** Removed it, rebuilt, ran:
  `test_a_zero_energy_neutrino_is_zero_rather_than_nan` **FAILED**, and
  `cargo test` went to `165 passed; 2 failed`. Restored → 169 passed and
  the pytest green.
- **The rest of the production changes** are covered by the mutation
  campaign below, which is the stronger form of the same check.

### Mutation campaign

Eleven mutations, each applied to the shipped source, built, and run
against `cargo test` then `test/test_core_positron_pion.py
test/test_core_neutrino.py test/parity`:

| # | mutation | caught by |
| --- | --- | --- |
| 1 | `positron_pion`: unfuse the lower boost limit | pytest |
| 2 | `positron_pion`: drop the `EMAX_PI_RF` clip | pytest |
| 3 | `positron_pion`: drop the electron line | pytest |
| 4 | `neutrino_muon`: swap the electron and muon rows | cargo |
| 5 | `neutrino_muon`: open the rest-frame upper edge (`>=` → `>`) | **survived** |
| 6 | `neutrino_muon`: unfuse the electron bracket's `xm·xp` | pytest |
| 7 | `neutrino_muon`: collapse the muon log coefficient to `−6r⁴` | pytest |
| 8 | `neutrino_pion`: transpose the two boost integrals | pytest |
| 9 | `neutrino_pion`: `boost_gamma(E, m)` for `1/√(1−β²)` | **survived** |
| 10 | `neutrino_pion`: drop the doubled electron line (the repair) | pytest |
| 11 | `neutrino_flavors`: `ZERO` uses negative zeros | cargo |

Both survivors were interrogated rather than accepted, per Task 4.5's
lesson.

**#9 is a real error the gates could not see.** `1/√(1−β²)` and `E/m`
agree to the last bit below `E_π ≈ 150` MeV and separate as the boost
grows: 5.9e-15 relative at `10 m_π` — the corpus's highest pion energy —
but **2.9e-11 at `E_π = 10⁵` MeV**, which is 29x the `PORTED_QUAD_RTOL`
the corpus now gives this case. β has already been rounded by its own
`sqrt`, so squaring it back and inverting loses bits the division never
had. The corpus simply does not sample far enough to notice, which is
exactly the shape Task 4.5 warned about. Fixed by lifting `boost_window`
out of `dnde_mu_numu` and pinning its three outputs bit for bit, plus
asserting that the two spellings of γ are *different doubles* above
`E_π = 200` MeV. Re-ran the mutation against the new seam: **caught**.

**#5 is unobservable by construction, and is now recorded as such.** At
exactly `x = 1 − r²` the `(1 − r² − x)²` factor has a double root, so the
guarded and unguarded spellings both produce `+0.0` — there is no value
for a test to separate. The old test never even reached the edge, because
`TWO_OVER_MASS_MU * (XMAX_RF / TWO_OVER_MASS_MU)` does not round back to
`XMAX_RF`. `the_rest_frame_support_is_open_at_both_edges` now *searches*
for the energy that scales exactly onto the edge (it exists, at
`XMAX_RF / TWO_OVER_MASS_MU` itself), asserts `+0.0` on the bit pattern
there, and asserts that the unguarded arithmetic gives `+0.0` too — so a
future reader does not go hunting for a gap that is not there. The lower
edge is not like this: `x <= 0.0` guards a division, and a mutation there
is caught.

## Numerical impact

**Three public functions moved, all three within their declared budgets;
nothing else moved.** Measured with the shipped wrappers against the
stored corpus arrays (`scratchpad/final_impact.py`), on the capturing
platform:

| entry point | worst relative | bit-equal | budget |
| --- | --- | --- | --- |
| `spectra.positron.charged_pion` | **5.494e-15** (`boosted_mild`) | 1304 / 1460 | `PORTED_QUAD_RTOL` 1e-12 |
| `spectra.neutrino.muon` | **0** | 3795 / 3795 | `EXACT_RTOL` 0 |
| `spectra.neutrino.charged_pion` | **9.739e-16** (`boosted_mild`) | 3793 / 4185 | `PORTED_QUAD_RTOL` 1e-12 |

The four already-ported kernels the diff can reach were re-measured as a
control and are unchanged from their own tasks' recorded figures:
`spectra.photon.charged_pion` 2.618e-15, `spectra.photon.charged_rho`
1.511e-13, `spectra.positron.muon` and `spectra.photon.muon` bit-equal.

Off-corpus, on a denser sweep the corpus does not reach (3,200 points for
the positron pion over 8 pion energies to `1e5` MeV; 9,600 for each
neutrino kernel over 8 parents): `positron.charged_pion` **3.5e-13**,
`neutrino.muon` **bit-equal at all 9,600 points**,
`neutrino.charged_pion` **2.3e-14**.

**One value changed that the corpus does not pin**, and it is a repair
rather than a drift: `dnde_neutrino_charged_pion(0.0, epi)` was `NaN` in
the first version of the port and is `(0, 0, 0)` now, which is what the
Cython returns. See the `crate::quad` finding. The corpus grids start at
`1e-5 m_π`, so no pinned value is involved.

Two budgets **tightened**, neither widened:
`spectra.positron.charged_pion` and `spectra.neutrino.charged_pion` both
go `QUAD_RTOL` (1e-8) → `PORTED_QUAD_RTOL` (1e-12), on the measurements
above. `spectra.neutrino.muon` was already `EXACT` and stays there,
achieved rather than assumed.

## Open Questions

- **The doubled `π → e ν` neutrino line** is the eighth blocked defect
  sharing the eventual corpus regeneration —
  [`neutrino-pion-electron-line-counted-twice.md`](../../../../docs/followups/todo/neutrino-pion-electron-line-counted-twice.md).
  Unlike most of its siblings it needs no Cython oracle, so its twin's
  deletion in this PR costs nothing.
- **`test_core_dispatch.py`'s `TestDeclaredDivergencesFromCython` now has
  one oracle left in the whole tree** (`scalar_mediator_decay_spectrum`),
  and Phase 06 deletes it. Phase 06 has to decide whether to retire the
  class or re-express the widenings against `cython_xs`, which is the only
  other live `.pyx` dispatch shape.

## Plan Impact

**Impact Level:** Phase file patched (frontmatter `status: Complete`),
`PLAN.md` phase-table cell updated, phase learnings synthesized. No ADR:
nothing here changed an architecture, an invariant or an interface. The
`crate::quad` short circuit is a *fidelity* correction inside a settled
contract, not a change to it.

Every Task 4.6 exit criterion is discharged as written; no phase-file
gate sentence turned out to be wrong. The one canonical-adjacent
correction this task made is in **another** project's docs — three places
asserted that Task 4.6 deletes `hazma/spectra/_positron/_pion.pyx`, which
the phase file's own capi-survivor exception has always contradicted.
Patched in place rather than deferred.

## Stale-state sweep

```console
$ git -C . rev-parse --abbrev-ref HEAD
claude/cython-to-rust/task-4.6-positron-pion-neutrino-pair
$ git -C . rev-parse --show-toplevel
/Users/logan.morrison/dev/Hazma/.claude/worktrees/cython-to-rust-d2bd16/.claude/worktrees/cython-to-rust/task-4.6-positron-pion-neutrino-pair
$ rg -n 'TODO|FIXME|breakpoint\(|import pdb|[^.]\bprint\(' rust/src/kernels/{positron_pion,neutrino_flavors,neutrino_muon,neutrino_pion}.rs test/test_core_positron_pion.py test/test_core_neutrino.py
(no occurrences)
$ rg -n 'spectra\._neutrino\._(muon|pion|neutrino)' --glob '!projects/**' .
test/parity/cases.py:1450     PORTED_ENTRY_POINTS, recording the origin
test/parity/cases.py:1452     PORTED_ENTRY_POINTS, recording the origin
test/parity/data/manifest.json:22122   capture provenance, immutable
test/parity/data/manifest.json:22344   capture provenance, immutable
# Four survivors, all four intended: `PORTED_ENTRY_POINTS` exists
# precisely to record where a swapped case's pinned values came from, and
# the manifest records the tree the corpus was captured from. No live
# import of any deleted module remains.
$ rg -n '_pion\.dnde_positron_charged_pion|_muon\.dnde_neutrino_muon|_pion\.dnde_neutrino_charged_pion' hazma/
(no occurrences)
$ find hazma -name '*.pyx' | wc -l ; find hazma -name '*.pxd' | wc -l
      11
       8
$ python -c "import sys; sys.path.insert(0,'test/parity'); import cases; print(len(cases.rust_core_kernels()))"
16
$ scripts/agents/check_doc_citations.py --changed-vs origin/master
docs scanned: 10
in-repo citations checked: 20
  resolved by exact: 15
  resolved by suffix: 5
external citations skipped: 16
  _eta_prime.pyx (1)
  _phi.pyx (1)
  hazma/spectra/_neutrino/_muon.pyx (5)
  hazma/spectra/_neutrino/_pion.pyx (3)
  hazma/spectra/_photon/_eta_prime.pyx (1)
  hazma/spectra/_photon/_phi.pyx (1)
  scipy/integrate/_quadpack_py.py (4)
out-of-range or ambiguous: NONE
```

**The citation check was missing from this block on round 1, and that is
what let the blocking finding through.**
[`doc-consistency.md`](../../../../docs/agents/doc-consistency.md)'s
"Line-number citation sweep" names
`scripts/agents/check_doc_citations.py --changed-vs origin/master`
explicitly; the sweep listed the `rg` half and not the mechanical half,
so a bare-basename citation that `rg` happily matched went unchecked for
ambiguity. It is in the block now. (`docs scanned: 10` because `--changed-vs` diffs
*committed* history — `docs/agents/lessons.md`'s ledger append is in the
same commit as this block, so the next run scans 11; that is
`lessons.md`'s own `[changed-vs-sees-only-commits]`, and the invariant
that matters, `out-of-range or ambiguous: NONE`, holds either way.) Of
its 16 skipped
external citations, the eight into `hazma/spectra/_neutrino/*.pyx` each
now name `ed1fa20`, so "external" reads as "open it at that revision"
rather than "unresolvable"; four are `scipy/integrate/_quadpack_py.py`,
which is out of tree by nature. The two bare survivors — `_eta_prime.pyx`
and `_phi.pyx` — are Task 4.2's prose about files Task 4.2 deleted, are
unambiguous (one candidate has ever existed for each), and are left
alone: they are outside the class this task created.

**Numerical-impact statement:** three public functions moved, each within
its declared budget and each measured above — 5.494e-15, 0 (bit-equal)
and 9.739e-16 against the stored corpus. Four control kernels re-measured
and unchanged. One unpinned value repaired (`E_ν = 0` returns zeros rather
than `NaN`, matching the Cython). Recorded in `../README.md`'s "Numerical
impact so far".

## Handoff to Next Task

- **Phase 04 is closed.** Read
  [`../../learnings/phase-04-spectra-kernels.md`](../../learnings/phase-04-spectra-kernels.md)
  before starting Phase 05 or 06 — it is the distillation of six tasks and
  sixteen entry points.
- **`hazma/spectra/` has no Cython Python entry point of any kind.** Four
  `.pyx` survive there for their `cdef` capsules alone
  (`_photon/{_muon,_pion}`, `_positron/{_muon,_pion}`), read only by the
  four mediator spectrum modules. `cases.rust_core_kernels()` → **16**.
- **11 `.pyx` and 8 `.pxd`** remain, all of them Phase 05/06 business.
  Re-derive with a clean rebuild rather than quoting.
- **`crate::quad` now short-circuits an empty interval**, as scipy does.
  Any Phase 05/06 kernel that can produce `a == b` inherits the fix.
- **`kernels::{neutrino_muon,neutrino_pion,positron_pion}` are `pub` and
  PyO3-free**, as are `neutrino_flavors::NeutrinoSpectrumPoint` and
  `neutrino_pion::boost_window`. Phase 06's mediator positron spectra
  boost `positron_pion`'s kernel exactly as `_pion.pyx` does today.
- **Read the mutation-campaign section before writing a Phase 05/06
  kernel.** #9 is the phase's sharpest lesson: a *correct-looking
  simplification* of a boost factor sat 29x outside the corpus's own
  budget at energies the corpus does not sample, and only a lifted,
  bit-pinned seam caught it.
