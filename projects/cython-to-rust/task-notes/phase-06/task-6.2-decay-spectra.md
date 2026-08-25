# Task 6.2: Decay spectrum pair (`scalar`, `vector`)

**Date:** 2026-08-23
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-06-mediator-spectra.md` Task 6.2;
`../../rules.md` rules 1–3 (parity), 4 (constants), 6–9 (Rust
conventions), 12 (benchmark)
**Related ADRs:** ADR-0001, ADR-0002
**Depends On:** Task 6.1

## Objective

Move `scalar_mediator_decay_spectrum` and
`dnde_decay_v`/`dnde_decay_v_pt` onto `hazma._core`, on top of Task 6.1's
table/cache/mode foundation, and delete both Cython twins in the same PR.

## Exit Criteria

From the phase file:

- `scalar_mediator_decay_spectrum`, `dnde_decay_v`/`dnde_decay_v_pt` on
  Rust; corpus green (quad budget); wrappers swapped; both Cython twins
  deleted.
- Benchmark vs pre-swap Cython recorded (expected large win — the old
  path rebuilt two quad-backed tables per call).

## Inputs Reviewed

- `../../PLAN.md`; `../README.md`; `README.md`;
  `../../phases/phase-06-mediator-spectra.md`; `../../rules.md`.
- `task-6.1-table-struct.md` — `## Findings` and `## Handoff`.
- `../../learnings/phase-04-spectra-kernels.md` §§2, 4, 5 and
  `../../learnings/phase-05-mediator-cross-sections.md` (the one
  syntactic FMA rule; the `SoftComplexToDouble` grep; the
  detached-worktree baseline recipe).
- `rust/src/kernels/mediator_tables.rs`, `rust/src/dispatch.rs`,
  `rust/src/quad.rs` (`QuadOpts`), `rust/src/kernels/soft_complex.rs`,
  `rust/src/kernels/vector_xs.rs` (the fallible-integrand pattern).
- The two `.pyx` and their generated `.c`;
  `hazma/{scalar,vector}_mediator/_*_mediator_spectra.py`;
  `test/parity/{cases,tolerances}.py`; `test/test_core_dispatch.py`;
  `test/test_core_photon_tables.py` (the deleted-twin test shape).
- `docs/agents/lessons.md`; `docs/agents/environment.md`;
  `docs/agents/doc-consistency.md`.

## Findings

- **The residual is the integrator's, not the transliteration's.** Over
  5,325 points the port is 71.6% bit-equal to the Cython with a worst
  relative difference of 2.2e-12. Setting `eng_s == ms` makes the boost
  integrand a *constant* — `β = 0`, so `jac = 1/2` and
  `E_γ,rf = E_γ` exactly — and every channel then agrees to within one
  ulp (worst 4.2e-16). The tables are bit-equal (Task 6.1) and so are
  their columns against the Phase 04 entry points, so what is left is
  `crate::quad` versus scipy's QUADPACK, which was already known not to
  be bit-equal — `PORTED_QUAD_RTOL` exists for that.
- **A constant integrand is *not* automatically reproduced exactly
  either.** `∫₋₁¹ c dcl` comes back one ulp off `2c` on both sides at
  about a third of the `c` values tried, in *different* places: at
  `E_γ = 0.01` MeV the port matches the exact `2c` and the Cython does
  not; at `0.1` MeV the reverse. That is a Gauss–Kronrod weight sum
  rounding differently, and it is the floor for every case here.
- **The `.pyx` returns `0.0` where a length check would have raised.**
  `pws[4]` (scalar) and `pws[2]` (vector) are read *only* inside the
  boosted line window, so at `ms = 550`, `eng_s = 600` a four-element
  `partial_widths` succeeds for a 30 MeV photon and raises `IndexError`
  for a 300 MeV one. Verified against the shipped 2.1.0 extension. A
  port that validated the buffer up front would have broken the working
  half, so the reads stay lazy and `PartialWidths::get` carries
  `boundscheck(True)`.
- **The two `.pyx` differ in *laziness*, and it is observable.** The
  scalar integrand guards each channel with a bitflag `if`, so a mode
  that excludes the lepton FSR never evaluates it. The vector integrand
  computes all six components and *then* selects one, so a mode that
  names none of the charged-pion FSR still raises where that
  coefficient's complex division fails. Confirmed:
  `dnde_decay_v_pt(0.0, 2mπ, 2mπ, pws, "e e g")` raises `TypeError` in
  the shipped extension. A lazy vector port would have returned a
  number.
- **`sqrt(4πα)² == 4πα` at the legacy `α = 1/137`** — measured, and
  the reason `QE_SQUARED` is the folded product rather than a
  sqrt-then-square. The round trip is *not* an identity in general (it
  loses a bit at `α = 1`), so the equality is asserted rather than
  assumed.
- **Cython builds a Python wrapper for a `cdef` integrand handed to
  `scipy.integrate.quad`,** and an exception raised inside it propagates
  out of `quad` rather than being absorbed — which is why the port
  remembers the first failure and raises after the integrator returns
  (`crate::kernels::vector_xs::thermal_cross_section`'s shape).
- **The port stops emitting `IntegrationWarning`.** Both entry points
  raise one from scipy today at every call tried, because the boost
  integrand's `1/|1 − β cos θ|` is hard for QAGP. The `.pyx` subscripts
  `quad(...)[0]`, so the warning never affected a *value*; the same
  silence was already accepted for every quad-backed kernel Phase 04
  ported.

### The FMA mutation campaign

Every `f64::mul_add` in the two new kernel modules was rewritten to its
unfused spelling, one at a time, rebuilt, and re-measured against the
live Cython on the 5,325-point grid. Baseline **3815/5325 bit-equal**.

| Outcome | Sites | Which |
| --- | --- | --- |
| **Killed** — bit-equality drops | 21 | both `log`-argument fusions in each of the four FSR functions; every `numerator = term1 + term2`; the `-1 - 4μ²(x−1)` and `2 - 4μ²(x−1)` polynomial heads; `8μ²·x`; `(-2+x)·x`; `-4μ²·x`; four of the five channel accumulations; **and both `1 − β cos θ` Doppler factors**, which are the two largest single effects (71.6% → 64.5% and → 64.5%) |
| **Killed on a grid that reaches them** | 2 | `-12 μ² + 2` (site 7) and the muon-FSR accumulation (site 17) |
| **Survive by construction** | 14 | thirteen whose coefficient is a **power of two**, plus the electron-FSR accumulation |

**The fourteen survivors are not a gap in the campaign.** Where the
coefficient is a power of two — `1 − 4μ²`, `−1 + 2μ²`, `−1 + 4μ²`,
`1 + 2μ²`, `16μ⁴ + w`, `−2x + w`, `2 − 8μ⁴` — the product only shifts an
exponent, so it is exact and the fused and unfused spellings round
exactly once either way. Checked with exact rational arithmetic over
40,002 mediator masses per shape (both lepton masses × 20,001 masses from
211 MeV to 2 GeV): **zero disagreements**, for every shape. Pinned by
`a_power_of_two_coefficient_makes_fusion_unobservable`. The fifteenth,
`pwee.mul_add(fsr, result)`, is the *first* channel in the scalar
integrand's `if`-chain, so its addend is exactly `0.0`.

**The two that needed a better grid are the interesting result.** `12` is
not a power of two, so `2 − 12μ²` genuinely can differ — and does at
**8.5%** of mediator masses above `2 m_μ`, the threshold below which the
muon FSR channel is closed and the site unreachable at all. The base
grid's lightest mass was 550 MeV and none of its ten `(lepton, mass)`
pairs happened to land on a disagreeing value; at `ms = 212 MeV` the
mutation drops bit-equality from 2,593/3,195 to 2,485/3,195 and pushes
the worst relative difference from 1.2e-14 to 3.0e-12. `ms = 200 MeV`
kills site 17 the same way. Both are load-bearing;
`the_twelve_coefficient_is_not_in_that_class` pins the first, including
the fact that the *electron* never reaches it (`12 (m_e/m_s)²` is far
below the ulp of 2).

**A harness note worth carrying forward.** The campaign was run three
times. The first two used `uv pip install -e .` without forcing a
rebuild, and the second run's measurements turned out to lag the
mutations by **two** iterations — a stale `hazma/_core.abi3.so` being
measured against the current source. It was caught because
`(-4.0).mul_add(m2, 1.0)` was reported killed in the vector module and
surviving in the scalar module, which the exactness argument above says
is impossible. The fix is one line: `rm -f hazma/_core.abi3.so` before
each install and `test -f` after it.

This is the **third** instance of the class nine places in the tree call
`[mutation-harness-poisons-its-own-baseline]` — and that class turns out
not to be in `docs/agents/lessons.md` at all, nor in
`lessons-examples.md`, so every one of those citations dangles
(`test/parity/oracles/capture.py:373` names a guard after it;
`phase-04/README.md:299` attributes it to Task 3.3). Filed as
[`../../../../docs/followups/todo/lessons-ledger-missing-the-mutation-harness-class.md`](../../../../docs/followups/todo/lessons-ledger-missing-the-mutation-harness-class.md)
rather than fixed here: the ledger's own format requires a real PR
citation, the PR that learned it is Task 3.3's and needs recovering from
history, and the ledger is already past its working-set cap.

## Decisions and Implementation Notes

- **Two kernel modules, not one.** `rust/src/kernels/scalar_decay_photon.rs`
  and `vector_decay_photon.rs`, one per `.pyx`, named the way
  `scalar_xs`/`vector_xs` were: the literal transcription's
  `<model>_mediator` half is already the PyO3 submodule's name. What the
  two share is already in `mediator_tables`; what is left — the FSR
  formulae, the channel list, the laziness — differs in every line.
- **One test module for both**, `test/test_core_mediator_decay_photon.py`,
  because the independent reference is one function parameterised by
  clone-pair. That is `test_core_photon_tables.py`'s shape, and the
  alternative was the same 300-line reference twice.
- **`SpectrumError` rather than two error types.** Two things can fail —
  a short `partial_widths` (`IndexError`) and a complex FSR coefficient
  (`TypeError`) — and both are needed by all four mediator spectrum
  modules, so the enum lives in `mediator_tables` beside `PartialWidths`
  and each PyO3 submodule maps it to the exception its `.pyx` raised.
- **`modes` membership is Python's `in`, not a list comparison.** The
  `.pyx` wrote `if "pi pi" in modes:` seven times, so `modes="pi pi g"`
  sets two bits by substring today and a set or tuple works as well as a
  list. `scalar_photon_modes` asks the object, in the `.pyx`'s own name
  order, so a `__contains__` that raises reports the first name it was
  asked about. Folding into `Vec<String>` instead would have narrowed
  the accepted types.
- **`dnde_decay_v` takes `require_vector`.** Task 3.5 built that helper
  for exactly this shape — an argument that must be a 1-D `float64`
  array and is never a scalar — and it is what the same function already
  uses for `partial_widths`. Two declared divergences come with it, both
  on paths no working call takes: a scalar `eng_gam` raised `TypeError`
  and now raises `ValueError`, and a `list` was refused and is now
  accepted (the widening `crate::dispatch` already declares for every
  entry point). The alternative, declaring the parameter
  `PyReadonlyArray1<f64>`, would have kept the scalar case's type and
  turned the *rank* and *dtype* cases from `ValueError` into `TypeError`
  — two regressions against the contract's own text instead of one
  against nothing.
- **`modes=None` explicitly passed now takes the default** where the
  `.pyx` raised `TypeError` on `"pi pi" in None`. PyO3 cannot
  distinguish an omitted argument from an explicit `None`, and the
  `text_signature` still advertises the `.pyx`'s seven-element default.
- **The shared photon table set stays shared.** The scalar module never
  reads the muon column, so building it is 500 wasted evaluations of a
  closed-form kernel per *distinct mass*. Measured below; it is far
  inside the noise of the win from fixing the dead cache, and splitting
  the cache would mean two `LazyLock`s differing by a field.
- **`test_core_dispatch.py`'s two Cython-oracle classes are retired
  here**, because this task deletes the last `.pyx` that spells a
  dispatch message. `cython_dispatch_messages()` survives as a *guard*
  — it now asserts the tree is silent — and the roster the port emits is
  frozen in that module with per-message provenance back to the deleted
  sources.

## Files Changed

- `rust/src/kernels/mediator_tables.rs` — `SpectrumError`,
  `PartialWidths`, `ScalarPhotonModes::{NAMES, from_bits, contains}`.
- `rust/src/kernels/scalar_decay_photon.rs` — **new.** Both FSR
  functions, the integrand, `spectrum_point`, `tables_for`.
- `rust/src/kernels/vector_decay_photon.rs` — **new.** The same four.
- `rust/src/kernels.rs` — register both; document the naming.
- `rust/src/scalar_mediator.rs` — `scalar_mediator_decay_spectrum`, the
  mode fold, the four message constants.
- `rust/src/vector_mediator.rs` — `dnde_decay_v`, `dnde_decay_v_pt`.
- `hazma/scalar_mediator/_scalar_mediator_spectra.py`,
  `hazma/vector_mediator/_vector_mediator_spectra.py` — imports
  repointed at `hazma._core`.
- `hazma/scalar_mediator/scalar_mediator_decay_spectrum.pyx`,
  `hazma/vector_mediator/vector_mediator_decay_spectrum.pyx` —
  **deleted**.
- `setup.py` — both extensions dropped; the capi-survivor comment
  re-derived.
- `test/test_core_mediator_decay_photon.py` — **new.**
- `test/test_core_dispatch.py` — the two Cython-oracle classes retired
  and re-expressed against a frozen roster.
- `test/parity/cases.py` — the `mediator_tables` rationale re-derived
  (two live `.pyx`, not four).
- `test/parity/tolerances.py` — the three photon cases tightened from
  `NESTED_RTOL` to `PORTED_NESTED_RTOL`.
- `test/parity/oracles/{entry_points,defects}.py` — the three photon
  cases' Cython origin re-declared `"restored"`; the missing
  `RESTORED_SOURCES` rows explained rather than guessed.
- `test/test_core_scalar_xs.py` — the served-roster assertion no longer
  claims cross sections are all `hazma._core.scalar_mediator` holds.
- `test/test_core_mediator_tables.py` — Task 6.1's two live-twin mode
  oracles repointed at the port, docstring tense corrected.
- `pyproject.toml` — the `cython<3.3` cap lifted (see `## Verification`).
- `hazma/vector_mediator/vector_mediator_decay_spectrum.pyi` — deleted
  with its module; a stub with no module makes the deleted import path
  look resolvable to a type checker.
- `projects/cython-to-rust/task-notes/phase-06/{README.md,task-6.2-decay-spectra.md}`,
  `projects/cython-to-rust/task-notes/{README.md,numerical-impact.md}`.

## Verification

Every command below was run from the worktree root against an editable
install built by cython 3.2.9 and numpy 2.5.1 (the parity manifest's
pins), with `python -c "import hazma._core; print(hazma._core.__file__)"`
confirming the worktree's own extension.

- `cargo fmt --manifest-path rust/Cargo.toml --check` — clean.
- `cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings`
  — clean.
- `cargo test --manifest-path rust/Cargo.toml --no-default-features` →
  **249 passed** (222 at Task 6.1's close). The 27 added cover: both FSR
  functions' kinematic windows and monotonicity; the two reachable
  `SpectrumError` arms; the lazy-vs-eager difference between the two
  integrands; the line term's analytic step; channel additivity; the
  `pws[4]`-inside-the-window behaviour; `QE_SQUARED` and `PI_SQUARED`
  against libm; the quadrature options' unreachable error arm; and the
  two mutation-campaign facts above.
- `pytest -q` → **2262 passed, 15 skipped, 12 subtests passed**
  (2163/15/12 at Task 6.1's close).
- `pytest test/parity -q` → **658 passed, 1 skipped** (657/1 before the
  three cases were repointed; the extra is the served-roster guard now
  that `PORTED_ENTRY_POINTS` carries three more rows).
- `pytest test/test_theory_aggregation.py -q` → **69 passed**, the
  model-layer gate the corpus cannot be, run either side of the swap.
- `pytest test/test_core_mediator_decay_photon.py -q` → **98 passed**.
- `find hazma -name "*.pyx"` → **7** (was 9): `_utils/boost`, the four
  capi survivors under `hazma/spectra/`, and the two `*_positron_spec`
  that Task 6.3 takes.
- Model-layer smoke test after the swap: `HiggsPortal.spectra` and
  `KineticMixing.spectra` both return finite spectra through the
  repointed wrappers.
- **The cython cap experiment**, which is why `pyproject.toml` changed:
  built the tree's extensions with cython **3.3.0** (`uv pip install -e .
  --no-build-isolation` after deleting every `.c` and `.so`) and ran the
  full suite → **2262 passed, 15 skipped, 12 subtests passed**, the same
  counts as 3.2.9, parity corpus and every Cython-twin bit-equality
  comparison included. The tree was then rebuilt with 3.2.9 and the suite
  re-run to the same counts before any other number in this note was
  taken.

Intentionally deferred: nothing. The two `RESTORED_SOURCES` rows the
oracle roster now wants are filed rather than guessed, because they need
this commit's own SHA (see `## Open Questions`).

## Numerical impact

**All three entry points move, none beyond 5.4e-12; budgets tightened.**
The full record is in [`../numerical-impact.md`](../numerical-impact.md);
the summary is:

| Entry point | Bit-equal | Worst relative | Worst at |
| --- | --- | --- | --- |
| `scalar_mediator_decay_spectrum` | 6,379 / 8,610 | **5.3327e-12** | `ms_550.boosted_strong.default` |
| `dnde_decay_v` | 22,918 / 29,295 | **1.1935e-12** | `mv_900.boosted_strong.mu_mu` |
| `dnde_decay_v_pt` | 22,918 / 29,295 | **1.1935e-12** | identical to `dnde_decay_v` |

All three above rule 3's 1e-12 threshold, so each owes a Phase 07
CHANGELOG line. Budgets go `NESTED_RTOL` (1e-6) → `PORTED_NESTED_RTOL`
(1e-9): 188x headroom on the scalar, 838x on the vector pair. Ten
tightened and none widened across the project.

Off-corpus, over 5,325 points (five `(mass, energy)` configurations ×
every mode of both entry points × 71 energies spanning six decades):
**3,815 bit-equal (71.6%), worst 2.2046e-12**, at
`modes=["pi0 pi0"]` where the integrand is the neutral pion's
discontinuous box.

Benchmark (release builds of both sides, macOS/arm64, same machine):

| Shape | Cython | Rust | Speedup |
| --- | --- | --- | --- |
| cold — fresh mass every call | 7.457 ms | 1.793 ms | **4.2x** |
| `dnde_decay_v_pt`, one energy | 9.413 ms | 0.0017 ms | 5,540x |
| `scalar_mediator_decay_spectrum`, 100 energies | 20.768 ms | 1.605 ms | 12.9x |
| `dnde_decay_v`, 100 energies | 27.322 ms | 0.832 ms | 32.8x |
| 20-point partial-width sweep at fixed mass | 186.269 ms | 0.045 ms | **4,180x** |

The 4.2x is the honest figure for the *work*; everything above it is the
dead cache being repaired, which `rules.md` rules 3 and 12 declare as
performance-only and which the drift table above shows moved no value
beyond the integrator's own noise.

## Open Questions

- ~~**Does the charged pion's forward-cone defect reach the mediator
  spectra?**~~ — **closed: yes, and the measurement already existed.**
  Carried in unanswered from Phase 04 and owed by this task. It needed no
  new experiment: `test/parity/oracles/data/manifest.json` already holds
  the corrected-value capture for defect A3 over exactly these three
  corpus cases, and its `diff_against_corpus` block records repairing the
  forward cone moving **1,032 of 8,610** scalar values by up to
  **1.63e-06** relative and **2,013 of 29,295** vector values by up to
  **7.77** relative (a factor of 8.8, at an absolute 7.3e-10). So the
  defect reaches both spectra, and in the vector case it changes the
  *shape* of the low-energy tail rather than a total — the kind of change
  a limit calculation notices. Reproduced under rule 1, not repaired
  here. Task 6.3 owes the same question for the positron pair, where the
  relevant defect is A4 and the same manifest already answers it.
- **The oracle roster has no restore revision for the two `.pyx` this
  task deleted**, because `RESTORED_SOURCES` records literal SHAs and this
  task cannot know its own commit's. `entry_points.py`'s three `Source`
  rows are updated to `"restored"` and say how to resolve it; the rows
  themselves are filed as
  [`../../../../docs/followups/todo/oracle-restore-revisions-for-the-mediator-decay-pyx.md`](../../../../docs/followups/todo/oracle-restore-revisions-for-the-mediator-decay-pyx.md),
  which ripens the moment this PR merges and which Task 6.3 should
  discharge for both pairs at once. Not blocking: `capture.py --check`,
  the gate that runs in `pytest`, does not read `RESTORED_SOURCES`.
- **`test_core_dispatch.py` has no `.pyx` oracle left at all.** This task
  spent the last one. `cython_dispatch_messages()` survives as a guard
  that the tree stays silent, and the roster it used to read is frozen
  with per-message provenance — but from Task 6.3 on, "the port's
  messages are the Cython's" is a claim backed by transcription rather
  than by execution. Nothing to do now; worth stating so a later reader
  does not mistake the frozen roster for a live comparison.

## Plan Impact

**Impact Level:** None. The phase file's Task 6.2 exit criteria are met
as written; nothing in `PLAN.md`, the phase file or `rules.md` is now
factually wrong.

## Stale-state sweep

Run against `claude/cython-to-rust/task-6.2-decay-spectrum-pair` at the
end of the task, with the numbers pasted rather than described.

| Check | Command | Result |
| --- | --- | --- |
| The deleted modules have no live importer | `grep -rn "scalar_mediator\.scalar_mediator_decay_spectrum\|vector_mediator\.vector_mediator_decay_spectrum" --include='*.py' --include='*.pyx' --include='*.toml' --include='*.yml' --include='*.rst' .` | 6 hits, **all intended**: 3 in `cases.py`'s `PORTED_ENTRY_POINTS` (which records the `.pyx` *origin* by contract) and 3 in `oracles/entry_points.py` (the `"restored"` rows, which name a module a re-capture resurrects). No importer. |
| `.pyx` / `.pxd` inventory | `find hazma -name "*.pyx" \| wc -l` / `*.pxd` | **7** and **8** (was 9 and 8). Re-derived after a clean rebuild, not quoted. |
| No stray type stub outlives its module | `find hazma -name "*.pyx" -o -name "*.pyi"` cross-check | `vector_mediator_decay_spectrum.pyi` found and deleted; `vector_mediator_positron_spec.pyi` is Task 6.3's. |
| Present-tense claims about four live mediator `.pyx` | `grep -rn "all four mediator\|four mediator-spectrum" test/ hazma/ docs/` | 3 hits, all corrected to past tense or given a "two of the four are gone" clause. The 10 in `oracles/data/manifest.json` are **captured provenance** and correctly left alone. |
| The cap's stated cause still exists | `grep -n "cython<3.3" pyproject.toml` | The cause was `scalar_mediator_decay_spectrum.pyx`, deleted here. Cap removed on measurement (see `## Verification`); `grep` now returns nothing. |
| Cited lesson class resolves | `grep -rn "mutation-harness" docs/agents/` | **Nothing** — the class 9 places cite is in neither `lessons.md` nor `lessons-examples.md`. Filed rather than cited (see `## Open Questions`). |
| Forbidden tokens | `preflight.sh` gate 11 | `PASS none added`. `grep` for `TODO\|FIXME\|breakpoint()\|pdb` over the three new files: none. |
| Formatters, changed files only | `black --check` / `isort --check-only` over `git diff origin/master --name-only -- '*.py'` plus untracked | **both PASS**, 12 files. |
| ruff, changed files vs `origin/master` | per-file counts, both trees | **80 → 79.** Every remaining error is in the two wrappers, which are pre-existing trunk debt ([its follow-up](../../../../docs/followups/todo/preflight-isort-ruff-red-on-trunk.md)); the new module and all nine touched test/parity files are at **0**, and the scalar wrapper's isort debt was fixed in passing. No regression, one improvement. |
| markdownlint | `preflight.sh --md <7 files>` | `PASS`. |
| Suite, corpus, cargo | see `## Verification` | `2262 passed, 15 skipped, 12 subtests`; `658 passed, 1 skipped`; `249 passed`. |
| **Numerical-impact statement** | `corpusdrift.py` over the three swapped cases | **All three entry points moved**, worst **5.3327e-12** relative; recorded in [`../numerical-impact.md`](../numerical-impact.md) with grids, worst points and the benchmark; three budgets tightened, none widened. Not "no public value changes". |

## Handoff to Next Task

**For Task 6.3 (the positron pair) — read this note's `## Findings` and
`## Decisions` first; it is the direct template.** The positron modules
are the *other* clone-pair, and four of the five things that cost this
task time are already answered for them.

**Now safe to assume:**

- **`crate::kernels::mediator_tables` is complete for 6.3's needs.**
  `SpectrumError`, `PartialWidths` (with `boundscheck(True)` semantics),
  `PositronMode`, and `positron_tables(mass)` returning memoized
  `Arc<PositronTables>` with `BelowGrid::Clamp` — all shipped and
  exercised. 6.3 adds no foundation.
- **The module layout and naming are settled**: one kernel module per
  `.pyx`, named `<model>_decay_<product>` (`scalar_decay_positron`,
  `vector_decay_positron`), documented in `kernels.rs` as the same
  naming exception `scalar_xs`/`vector_xs` took. One shared test module
  for the pair, `test/test_core_mediator_decay_positron.py`, shaped like
  `test_core_mediator_decay_photon.py`.
- **The fallible-integrand shape**: capture the first `SpectrumError` in
  an `Option`, return `NaN` from the closure, raise after `quad` returns.
  Both PyO3 submodules already own `spectrum_error`, `PARTIAL_WIDTHS` and
  `OUT_OF_BOUNDS_MESSAGE` — 6.3 reuses them rather than re-deriving.
- **`require_vector` is the right helper for `dnde_decay_s`/`dnde_decay_v`**
  (the array-only entry points), with the two declared divergences this
  note records. `dnde_decay_s_pt`/`dnde_decay_v_pt` take a plain `f64`.
- **Task 6.1's finding that neither positron module needs
  `soft_complex`** is confirmed by this task's experience of the two that
  do: the `grep -c SoftComplexToDouble` answer of `0 / 0` means no
  `SpectrumError::NonReal` arm is reachable from the positron kernels, so
  their `spectrum_point` may return a plain `Result` carrying only
  `OutOfBounds`.

**Still risky / unknown for Task 6.3:**

- **The positron pair's cache key is the mass, but its *below-grid*
  behaviour differs from this task's.** `BelowGrid::Clamp`, not the `1/E`
  tail, and the grid starts at the **legacy** `m_e = 0.510998928` — which
  is exactly where
  [`../../../../docs/followups/todo/positron-spectrum-nan-at-legacy-electron-mass.md`](../../../../docs/followups/todo/positron-spectrum-nan-at-legacy-electron-mass.md)
  says the positron spectra return `nan`. The project handoff has been
  asking for that follow-up to be met "before Phases 05/06"; 6.3 is where
  it actually lands on the grid's first abscissa.
- **`dnde_decay_s` short-circuits `fs == "e e"` before the integral**
  (`scalar_mediator_positron_spec.pyx:207`), returning `lines_contrib`
  alone — a fifth structural difference from the decay pair, and one that
  changes which `pws` indices are read. Read the `.pyx` for the read
  order before writing the integrand.
- **Take the benchmark before deleting the twins**, and take it from a
  **release** build of both sides in one interpreter. The editable
  install is ~20x pessimistic and inverts the comparison; the recipe that
  worked here is a detached `git worktree add --detach <dir> origin/master`
  plus a non-editable `uv pip install .` of each side into one scratch
  venv, run from outside the repo so `sys.path` cannot prefer the source
  tree. That last clause is load-bearing — a suite run from the repo root
  imports `hazma` from the worktree, not from site-packages, which
  silently invalidated the first cython-3.3 measurement this task took.
- **`test/parity/cases.py` needs three edits per swap, not one**: the
  `Case.module` moves to the wrapper, a `PORTED_ENTRY_POINTS` row is
  added, and the `_CORE_TEST_ONLY_MODULES` comment about how many
  mediator `.pyx` are alive needs re-deriving. Missing the second turns
  `test_the_served_roster_is_exactly_the_ported_entry_points` red with a
  set-difference message that does not name the cause.
- **Task 6.4's ground is now visible.** With both decay modules gone,
  `hazma/spectra/_photon/{_muon,_pion}.pyx` have no consumer outside
  their own pair, and `hazma/spectra/_positron/{_muon,_pion}.pyx` are read
  only by the two `.pyx` Task 6.3 deletes. So 6.4's `rg` sweep should come
  back empty for all four the moment 6.3 lands — and `pyproject.toml`'s
  Cython requirement, not just its cap, goes with them.
