# Working Memory: Phase 04 — Spectra kernels

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Phase:** 04
**Status:** In Progress
**Plan References:** `../../phases/phase-04-spectra-kernels.md`
**Related ADRs:** ADR-0001, ADR-0002
**Depends On:** Phase 03 complete

## Objective

Track live per-task status and phase-scoped findings for the spectra
kernel ports.

## Tasks

| # | Task | Depends on | Status | Task Note |
| --- | ------ | ------------ | -------- | ----------- |
| 4.1 | `_positron/_muon` (template swap) | — | **Complete (2026-08-11)** | [task-4.1-positron-muon.md](task-4.1-positron-muon.md) |
| 4.2 | Photon table family (kaon + eta/omega/eta′/phi) | 4.1 | **Complete (2026-08-12)** | [task-4.2-photon-table-family.md](task-4.2-photon-table-family.md) |
| 4.3 | `_photon/_muon` (spence) | 4.1 | **Complete (2026-08-16)** | [task-4.3-photon-muon.md](task-4.3-photon-muon.md) |
| 4.4 | `_photon/_pion` | 4.3 | **Complete (2026-08-17)** | [task-4.4-photon-pion.md](task-4.4-photon-pion.md) |
| 4.5 | `_photon/_rho` (nested quad) | 4.4 | Not started | [task-4.5-photon-rho.md](task-4.5-photon-rho.md) |
| 4.6 | `_positron/_pion` + neutrino pair | 4.1, 4.3 | Not started | [task-4.6-positron-pion-neutrino.md](task-4.6-positron-pion-neutrino.md) |

## Exit Criteria

- All rows Complete; phase file frontmatter `status: Complete`.
- Phase learnings at `../../learnings/phase-04-spectra-kernels.md`.

## Inputs Reviewed

- `../../phases/phase-04-spectra-kernels.md` (incl. the capi-survivor
  exception in its Goal); `../README.md`;
  `../../references/cython-inventory.md` (cimport DAG).

## Findings

- **The port surfaced a second live 2.1.0 numerical defect** (Task 4.1).
  `hazma/spectra/_positron/_muon.pyx` **divides** by the Michel
  normalization `R_FACTOR` where it should multiply, so every positron
  spectrum is low by `1/R_FACTOR²` — **0.0374%**, uniformly, propagating
  through `dnde_positron_charged_pion` and both mediator positron
  modules. The sibling `hazma/spectra/_neutrino/_muon.pyx` declares the
  same constant and multiplies by it, which is what makes this an
  inversion rather than a convention. Reproduced per rule 1 and filed as
  [`positron-muon-spectrum-normalization-inverted.md`](../../../../docs/followups/todo/positron-muon-spectrum-normalization-inverted.md),
  blocked behind Phase 06 Task 6.4. **Found the same way Task 3.4 found
  the boost-integral defect:** by writing an analytic test the original
  never had. Every task in this phase should write one.
- **Disassemble before porting** (Task 4.1). `objdump -d` the shipped
  `.so` and read the `fmadd`/`fmsub` sites; `_positron/_muon` has nine,
  and three expressions that look fusable are not (`x² − 4r²`,
  `1 − β²`, and any sum whose operand went through a division). Written
  from the map, the port was bit-equal on the first build — no
  bisection round, unlike Task 3.4.
- **Scope a bit-equality-against-Cython class to the corpus's capturing
  platform, never to a "does this compiler contract" probe** (Task 4.1,
  learned from two CI failures after two green macOS runs). The probe
  asks the wrong question: a compiler contracting a *different* set of
  expressions, or a libm rounding one call differently, breaks the
  comparison just as thoroughly, and no probe over one mechanism sees the
  others. `test/test_core_positron_muon.py` now reads the platform out of
  `test/parity/data/manifest.json`, which is the mechanism `test/parity`
  and `ci.yml` already use. **Copy that, not a probe.**
- **The capturing platform cannot see a bug in its own skip logic.** On
  macOS the probe answered True whether or not it was right, so every
  local run was green and no test in the module could tell a working
  guard from a broken one. Expect to learn this class from CI, and read
  a Linux failure in a bit-equality test as "the scope is wrong" before
  "the port is wrong".
- **A fused Python reference (correctly-rounded `fma` via `Fraction`)
  reproduces the shipped macOS Cython bit-for-bit** — 0 mismatches in
  21,000 points for `_positron/_muon`, against 11,713 for the unfused
  form. A cheap second confirmation of an FMA map, independent of the
  disassembly. It says nothing about other platforms, which is exactly
  why the scope above is a platform.
- **Repointing the corpus case is part of the swap, not bookkeeping**
  (Task 4.1). `cases.py` names the `.pyx` module; leave it and the gate
  keeps calling the twin while the wrapper calls Rust — green and
  vacuous. `PORTED_ENTRY_POINTS` records the origin so
  `assert_full_coverage` still balances, and now also fails if a ported
  entry point's `.pyx` still exports its `def`.
- **A `NaN` energy does not propagate through a kernel that clips with
  `fmax`/`fmin`** (Task 4.1), in either language: both limits collapse
  onto the rest-frame support and a finite number comes back. The corpus
  samples no `NaN`, so only a hand-written test catches a port that
  differs. Expect the same shape in every boosted kernel.

- **Five near-copies port to one implementation, and that is what
  surfaces the defects** (Task 4.2). The five tabulated photon `.pyx`
  differed only in table, parent mass and line terms; written as one
  `dnde` over seven `Spectrum` values, the five line-weight expressions
  sit in one column and two of them are visibly wrong. Neither is
  findable one file at a time, which is the general shape: **a
  parameterised port is a diff between siblings.**
- **The port has now surfaced three live 2.1.0 defects, all by writing a
  statement the original never made.** Task 4.2's two are both in the
  line terms: `_eta_prime.pyx:107` weights its `η′ → γγ` line with `BR`
  where four siblings use `2·BR` (0.02307 photons per decay instead of
  0.04614 — 0.63% of the η′ yield), and `_phi.pyx:111,113` place both
  photon lines at the **daughter meson's** energy (656.94 MeV where
  362.52 is right; 959.65 where 59.82 is right, a factor of 16). Both
  reproduced per rule 1, filed as
  [`eta-prime-two-photon-line-missing-factor-two.md`](../../../../docs/followups/todo/eta-prime-two-photon-line-missing-factor-two.md)
  and
  [`phi-photon-lines-use-the-daughter-meson-energy.md`](../../../../docs/followups/todo/phi-photon-lines-use-the-daughter-meson-energy.md),
  both blocked behind Phase 06 Task 6.4. **Four blocked defects now share
  one eventual corpus regeneration** — a fifth joined in Task 4.3.
- **`numpy.sum(axis=0)` is pairwise above eight terms, and exactly one
  live table is wide enough to care** (Task 4.2). The φ CSV has ten decay-mode
  columns; the other six have 2–7, where NumPy's reduction degenerates to
  a sequential fold. Reusing `boost::pairwise_sum` is what makes the
  embedded parse bit-equal — a mutation to a sequential fold fails six
  tests, all φ, and none on the other six tables.
- **Deleting an extension strands whatever read its module *globals*, not
  just whatever imported it** (Task 4.2). `test/test_core_interp.py` and
  `test/test_core_boost.py` built their seven-table fixtures from
  `_eta.eta_data_energies` and friends, and both failed at *collection* —
  so the whole suite reported two errors and ran nothing, which reads
  like a broken build rather than a stranded dependent. Repaired by
  loading the CSVs the way the deleted modules did, which also makes
  those oracles independent of the Rust that now consumes them.
- **A monkeypatch that shadows a real submodule stops measuring a delta**
  (Task 4.2). `test/parity/test_parity.py`'s served-kernel meta-tests
  patched `hazma._core.photon` with a one-kernel fake and asserted
  `baseline + 1`; once `photon` held seven real kernels the fake replaced
  seven and added one. Repointed at `hazma._core.not_a_real_domain`.
  **Any later task filling `neutrino`, `scalar_mediator` or
  `vector_mediator` would have hit the same thing.**
- **A `NaN` energy had no faithful answer, and the honest move was to
  declare a change** (Task 4.2). The Cython raised `IndexError` out of
  `np.flatnonzero(lb <= x)[0]`; the Rust panicked at an `.expect`.
  `dispatch::map_unary` has no per-element error channel, so neither type
  survives an element-wise map — the port returns `NaN`, which is what
  the same kernels' rest-frame branch already did. Declared in
  `rust/src/boost.rs`, in the numerical-impact log, and by test.
  **Expect this shape wherever a kernel's error path is per-element.**
- **The Rust and Python halves of a kernel port do not accept the same
  physics notation** (Task 4.2). `rust/src/kernels/photon_tables.rs`
  writes `η′ → γγ` and `(M² − m²)/(2M)` freely; the same strings in a
  Python docstring produce 22 `RUF002` "ambiguous unicode" findings,
  because ruff reads `γ` as a Latin `y`, `′` as a backtick, `−` as a
  hyphen and `×` as an `x`. Every other `test/test_core_*.py` is clean, so
  this is a rule the suite already follows silently and a new module has
  to learn: **spell final states the way hazma's own CSV headers do**
  (`a` for a photon — `a_a`, `pi0_a`, `eta_a`), and use ASCII `-`, `x` and
  `'`. `η`, `φ`, `ω`, `β`, `δ`, `→`, `·` and superscripts are *not*
  flagged, so the notation stays readable. Three `PLR2004` magic-value
  comparisons and one missing return annotation came with it — all four
  worth fixing rather than silencing.

- **A third-party special function is a hypothesis until the kernel that
  amplifies it says otherwise** (Task 4.3). `spec_math`'s cephes `spence`
  agrees with `scipy.special.spence` to 2.0e-15 — fine everywhere except
  in the one kernel that forms `(5/β)·(spence(x₋) − spence(x₊))` at the
  corpus's `β = 1.4e-6` probe, where `1/β ≈ 3.5e6` turns two ulps into a
  **3.15e-11** relative shift, 320x the `SPECFUN` budget. The difference
  is pure **FP contraction**: scipy ships cephes compiled by clang with
  `-ffp-contract=on`, so `polevl`'s Horner and the reflection's
  `π²/6 − ln(x)·ln(1−x)` are fused; `spec_math` writes them unfused and
  Rust does not contract. Transcribing cephes in-tree with that
  contraction map (~60 lines) makes `spence` **bit-identical** to scipy at
  13,000 points over all four branches, and the kernel bit-identical to
  the Cython at 144,000. `special.rs` now bypasses `spec_math` for two of
  its three functions, each for its own measured reason.
- **Attribute the difference to a term before touching a budget**
  (Task 4.3). All 24 failing corpus points were reproduced to a ratio of
  **1.000** by `(5/β)·Δspence·α/(3π E_μ)` computed from the kernel's own
  `x∓`, which said in one measurement both that `spence` was the entire
  cause and that the other 22 FMA sites, two `pow` calls and four folded
  constants were already exact. The alternative reading — "the port is
  1e-11 off, widen `SPECFUN` 300x" — was available and wrong.
- **The corpus cannot see a single wrong FMA site** (Task 4.3, measured by
  mutation). Unfusing one of the boosted polynomial's 22 multiply-adds
  leaves **all 630 corpus assertions and all 109 `cargo` tests green**;
  only `test/test_core_photon_muon.py`'s bit-equality sweep catches it,
  and only at 4 of 10,000 *random* arguments — the swept grid misses it
  too. The per-kernel oracle is not redundant with the corpus; it is the
  half that reaches arbitrary arguments.
- **A ported `.pyx`'s `def` need not be named after its public entry
  point** (Task 4.3). `_photon/_muon.pyx` spelled it `dnde_photon`, not
  `dnde_photon_muon`, which broke
  `test_the_served_roster_is_exactly_the_ported_entry_points` — it derived
  the expected roster from `PORTED_ENTRY_POINTS`'s *values*, which
  `assert_full_coverage` needs to be the `.pyx` origin. Now derived from
  the live cases' `function`. Tasks 4.4–4.6 inherit the fix.
- **The port has now surfaced five live 2.1.0 defects**, the fifth found
  the same way as the rest: by asserting a property the original never
  did. Task 4.3's is an endpoint.
  `hazma/spectra/_photon/_muon.pyx:41` cuts the *rest-frame* spectrum at
  `y = 1 − √r` while the in-flight branch of the same file (`:88`) and
  `hazma/spectra/_photon/_pion.pyx:16`'s `ENG_GAM_MAX_MURF` both use
  `1 − r`, which is the
  kinematic endpoint `(m_μ² − m_e²)/(2m_μ)`. So `dnde_photon_muon(E, m_μ)`
  is a hard zero over the top **0.2543 MeV** (0.48%) of its support where
  the spectrum is still `5.34e-7 MeV⁻¹`, and the published spectrum is
  **discontinuous in `E_μ` at rest**. The statement that found it: the
  in-flight closed form is *exactly* the boost integral of the rest-frame
  distribution — to machine precision — but only when integrated to
  `1 − r`. Filed as
  [`photon-muon-rest-frame-endpoint-uses-the-wrong-power-of-r.md`](../../../../docs/followups/todo/photon-muon-rest-frame-endpoint-uses-the-wrong-power-of-r.md),
  blocked behind Phase 06 Task 6.4.
- **Read `fmla` as well as `fmadd`** (Task 4.3). `grep -c fmadd` finds 17
  sites in this kernel and misses five: one scalar `fmla.d` and four
  **`fmla.2d`**, because clang vectorises the `x₋` and `x₊`
  log-coefficient Horner chains into one 2-wide chain. Also `y**3` and
  `y**4` are libm **`pow` calls** (`otool -Iv` names the stub) while
  `y**2` is an `fmul` — clang folds `pow(x, 2.0)` and refuses the cubic.
  `y*y*y` would have been a different double.

- **Read the `cdef`s, not only the expressions** (Task 4.4).
  `hazma/spectra/_photon/_pion.pyx`'s neutral-pion kernel declares
  `cdef float beta` and `cdef float ret_val` in a file where every other
  local is a `double`, and the shipped object confirms it with two
  `fcvt s, d` / `fcvt d, s` round trips. A port that transcribes the
  formula and not the declaration lands **8.5e-9** relative away, which
  the corpus holds this entry point to `rtol = 0`. With `as f32 as f64`
  at both sites it is bit-equal at 9,000 sampled points and all 1,305
  pinned values. Third instance in this phase of a *declaration* carrying
  numerics the expression does not — Task 3.1's mixed constant tables and
  Task 4.3's `y**3`-is-a-`pow`-call are the other two.
- **An FMA site inside a quadrature integrand cannot be gated** (Task
  4.4, measured by mutation). Unfusing `F_A² + F_V²` in `dnde_pi_to_lnug`
  leaves the worst corpus difference at **2.618e-15 — the same figure the
  correct port produces** — with 120 `cargo` tests and 73 per-kernel tests
  green. The integral does not carry the integrand's last bit out, and
  unlike Task 4.3 there is no bit-equality mode to fall back on, because
  the port replaces *scipy's* QUADPACK. The 4 sites in
  `charged_pion_integrand`, outside the integrand's own arithmetic, *are*
  caught. `rust/src/kernels/photon_pion.rs` states the limitation at the
  top so the next reader does not assume a gate exists.
  **Task 4.5 inherits it one integration deeper.**
- **A module-level `cdef double` is the one constant `movk`-pinning
  cannot reach** (Task 4.4). `FPI = DECAY_CONST_PI * M_SQRT1_2` is loaded
  from memory, not folded into an immediate — and the two obvious Rust
  spellings differ: `130.41 * FRAC_1_SQRT_2` is right,
  `130.41 * (1.0/2.0_f64.sqrt())` is **one ulp low**, because `1/√2` by
  division is not the correctly-rounded `√0.5`. Only a `cargo` test
  catches it; like the FMA sites, one ulp in the integrand does not
  survive the quadrature.
- **The port has now surfaced six live 2.1.0 defects; the sixth is a
  quadrature that loses its own support** (Task 4.4).
  `hazma/spectra/_photon/_pion.pyx:123` integrates over the whole of
  `cos θ`, but the integrand is nonzero only where the pion-rest-frame
  photon energy stays under `ENG_GAM_MAX_PIRG = 69.783` MeV. As the lab
  photon approaches the boosted endpoint that window narrows past
  QUADPACK's largest first-rule abscissa (~0.9956), so **every node
  returns zero, the error estimate is zero, and `qagp` terminates
  successfully with `0.0`** where the spectrum is not zero. At
  `E_π = 10 m_π` — the corpus's own most boosted block —
  `dnde_photon_charged_pion(900, 1396)` is `0.0` against `3.586e-07`
  MeV⁻¹, and the differential spectrum is a hard zero over roughly the
  top quarter of its support. Integrated, the loss is 0.0054% at
  `E_π = 1` GeV, 0.041% at 1396 MeV and 2.96% at 5 GeV, so it is a
  **shape** defect rather than a yield defect at hazma's scales. The port
  reproduces the zeros in exactly the same places. Filed as
  [`charged-pion-photon-spectrum-misses-the-forward-cone.md`](../../../../docs/followups/todo/charged-pion-photon-spectrum-misses-the-forward-cone.md),
  blocked behind Phase 06 Task 6.4. **Six blocked defects now share one
  eventual corpus regeneration.**
- **The divergent regime is reachable and the port agrees with scipy
  inside it** (Task 4.4 — Phase 03 Task 3.3's obligation, discharged for
  the project's first `qagp` consumer). The first non-`Ok` flag appears at
  `E_π = 4e4` MeV (`γ_π ≈ 290`), 40 GeV against a sub-GeV library and two
  decades above the corpus ceiling. Over an 11 × 8 grid reaching
  `E_π = 1e5` the port's `ier` **equals scipy's flag on the Cython twin at
  all 88 points**, including both `ier = 4` entries and the non-monotonic
  pattern between them, with values agreeing to **2.8e-11** on
  macOS/arm64. Task 3.3's warning was that the two *may* separate without
  bound there; on the only live shape that reaches it, they do not.
  **This is also the one number in the kernel that the platform reaches**
  (PR #68's first CI run): the converged regime holds at 1e-12 on Linux as
  well, while the non-converged separation is **6.2998e-10** there — 23x,
  and the same double across py3.10–3.14. Its budget is a flat 1e-8 rather
  than a platform branch, because a tight bound on a chaotic quantity
  certifies nothing.
- **The `1/β` amplification Task 4.3 warned about does not reach `_pion`**
  (Task 4.4). `rest_plus_eps` is the *pion's* `β = 1.4e-6`, which enters as
  a Jacobian and a Doppler factor rather than a `1/β` prefactor — the muon
  kernel is always evaluated at the fixed `ENG_MU_PIRF = 109.78` MeV,
  where `β = 0.27`. So this kernel never evaluates the muon spectrum near
  its own rest frame, `rest_plus_eps` is no louder than any other block
  (60/308 values not bit-equal, worst 2.98e-16), and the **Task 4.3 open
  question "does the rest-frame endpoint defect reach `_pion`?" is
  answered: no.** `_rho` boosts *this* kernel, not the muon kernel, so the
  same argument covers Task 4.5 — but the ρ does sweep the pion's own `β`,
  so re-derive rather than inherit.
- **A mutation harness that reverts with `git checkout --` cannot revert a
  file git has never seen** (Task 4.4). The new kernel module was
  untracked, so the restore step errored, the driver did not check, and
  five mutations accumulated while being read as five independent
  measurements. Task 3.3's `[mutation-harness-poisons-its-own-baseline]`
  in a new disguise, with the same tell — implausibly uniform failure
  counts (`26 | 7` three times running). Snapshot to a file outside the
  tree, `cmp` before each mutation, and verify the restore.

## Decisions and Implementation Notes

- **The per-kernel swap recipe now lives in the phase file's Goal**
  (Task 4.1), so it is canonical rather than inferred from one task
  note. Eight steps, of which "map the FMAs first" and "repoint the
  corpus case" are the two that a reader would otherwise skip.
- **A capi survivor loses its `def`, not its file** (Task 4.1) — the
  `cdef`s and their `__pyx_capi__` capsules stay, so the mediator
  modules keep importing while no Python caller can reach the replaced
  implementation. Tasks 4.3 and 4.4 (the other two survivors) do the
  same.
- **Per-kernel test modules do not copy `test/test_core_dispatch.py`**
  (Task 4.1), reversing Task 2.3's instruction: since Task 3.5 the
  dispatch layer is three shared helpers, so those 118 tests cover code
  every kernel routes through unchanged. `test/test_core_positron_muon.py`
  is the shape to copy — 47 tests, one per contract branch plus the twin
  as a two-mode oracle (bit-for-bit on the capturing platform, a
  peak-scaled budget elsewhere, nothing skipped) plus physics.
- **The ill-conditioned-corpus follow-up is waived for the tabulated
  photon family, not resolved** (Task 4.2). The waiver rests on a
  measurement: the port is bit-equal to the Cython at all 336,000 sampled
  points, so on the capturing platform there is nothing for a
  conditioning budget to absorb, and off it the parity suite does not run
  (`ci.yml` passes `--ignore=test/parity`). The follow-up's prediction
  that "every affected block will produce a false failure the moment a
  Rust implementation lands" is **refuted for `spectra.photon.eta`**; it
  stays open for the five cross-section blocks.
- **The `TABULATED` budget class is kept rather than tightened to
  `EXACT`** (Task 4.2), even though the port would pass `EXACT` today.
  Unlike `spectra.positron.muon`, bit-equality here rests on reproducing
  *NumPy's summation order* — an implementation detail a future NumPy may
  change — so `EXACT` would be the wrong contract rather than a tighter
  one.
- **One module may serve several `.pyx`** (Task 4.2), which is the stated
  exception to `kernels.rs`'s one-submodule-per-`.pyx` convention rather
  than a silent violation of it. `photon_tables` serves five.
- **Fix the special function rather than widen the budget** (Task 4.3).
  `SPECFUN` would have had to go 1e-13 → 1e-10 to admit a two-ulp
  `spence`; that is the "widening until it passes makes the gate vacuous
  exactly where the numerics are most fragile" that
  [`parity-corpus-pins-ill-conditioned-points.md`](../../../../docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md)
  warns against. `special.rs`'s contract was already *match scipy* rather
  than *use `spec_math`* — Task 3.2 set that precedent when it dropped
  `spec_math`'s `bessel_kn`.
- **The `SPECFUN` budget is kept rather than tightened to `EXACT`**
  (Task 4.3), for the same reason Task 4.2 kept `TABULATED`: bit-equality
  rests on scipy's C being compiled with FP contraction, a property of
  that build, so `EXACT` would be the wrong contract rather than a tighter
  one. Recorded in the class docstring in `test/parity/tolerances.py`.
- **`test/test_core_dispatch.py`'s spectra oracle moved to
  `_photon/_pion`** (Task 4.3), because its three declared-divergence
  tests called the `def` this task deleted. `_pion`'s entry point has the
  identical shape and wording. **Task 4.4 deletes that one too** — move it
  again or retire the class; by the end of this phase no spectra `.pyx`
  exports a Python entry point at all.

- **Two oracle standards in one test module** (Task 4.4), and two classes
  to make the asymmetry a decision rather than an inconsistency. The
  neutral pion is closed form, so it gets the template's two-mode
  comparison (bit-for-bit on the capturing platform, a peak-scaled budget
  elsewhere). The charged pion replaces *scipy's* QUADPACK with the
  in-tree one, and **two independent adaptive integrators are not
  bit-equal on any platform** — so there is no capturing-platform branch
  to take and it gets one measured budget everywhere.
- **The `QUAD` budget is tightened per case, not class-wide** (Task 4.4).
  `test/parity/tolerances.py` gains `PORTED_QUAD_RTOL = 1e-12`, which
  `spectra.photon.charged_pion` takes; the two unported `QUAD` cases keep
  `QUAD_RTOL = 1e-8` until Task 4.6 measures them, because 1e-8 was the
  envelope over *arbitrary* integrands and each live shape lands far
  inside it. What the tightening buys was measured rather than asserted:
  most mutations are caught at either budget, and the one that
  distinguishes them models Task 4.3's near-miss — perturbing
  `dnde_photon_muon` (this kernel's integrand) by the 3.2e-11 that a
  two-ulp `spence` produced moves the corpus by **3.199e-11**, silently
  inside 1e-8 and loudly outside 1e-12.
- **`spectra.photon.neutral_pion` keeps `EXACT`** (Task 4.4) — the mirror
  image of Tasks 4.2/4.3 keeping a loose class. There bit-equality rested
  on a build property, so the loose class was the right contract; here it
  rests on IEEE arithmetic and an `f32` cast, which are portable, so there
  is nothing to loosen *to*.
- **`quad`'s `Err` arm returns `NaN` rather than panicking** (Task 4.4),
  with a `cargo` test asserting the arm is unreachable: `QuadError`
  depends only on the `const` options, never on the integrand. `NaN` is
  the shape Task 4.2 settled for a per-element error channel that does not
  exist.
- **`test/test_core_dispatch.py`'s spectra oracle moved again**, to
  `_photon/_rho` (Task 4.4). **Task 4.5 swaps that one too and there is no
  fourth photon candidate** — the class docstring now names the two
  remaining options (move to a `_positron`/`_neutrino` entry point, which
  Task 4.6 then deletes, or retire the three spectra tests).

## Files Changed

### Task 4.1

- New: `rust/src/kernels/positron_muon.rs`,
  `test/test_core_positron_muon.py`,
  `docs/followups/todo/positron-muon-spectrum-normalization-inverted.md`.
- Changed: `rust/src/{kernels,positron}.rs`,
  `hazma/spectra/_positron/{__init__.py,_muon.pyx}`, `hazma/_core.pyi`,
  `test/parity/{cases,test_parity}.py`, `docs/followups/README.md`,
  `../../phases/phase-04-spectra-kernels.md`.
- Deleted: `hazma/spectra/_positron/_muon.pyi`.

### Task 4.2

- New: `rust/src/kernels/photon_tables.rs`,
  `test/test_core_photon_tables.py`,
  `docs/followups/todo/eta-prime-two-photon-line-missing-factor-two.md`,
  `docs/followups/todo/phi-photon-lines-use-the-daughter-meson-energy.md`.
- Changed: `rust/src/{kernels,photon,boost,interp}.rs`,
  `hazma/spectra/_photon/__init__.py`, `hazma/_core.pyi`, `setup.py`,
  `test/parity/{cases,tolerances,test_parity}.py`,
  `test/test_core_{boost,interp}.py`, `docs/followups/README.md`,
  `docs/followups/todo/{boost-integral-drops-last-interior-cell,positron-muon-spectrum-normalization-inverted}.md`.
- Deleted: `hazma/spectra/_photon/{_eta,_eta_prime,_kaon,_omega,_phi}.{pyx,pxd,pyi}`
  and `hazma/spectra/_photon/path.py` — 16 files, 1,020 lines.

### Task 4.3

- New: `rust/src/kernels/photon_muon.rs`, `test/test_core_photon_muon.py`,
  `docs/followups/todo/photon-muon-rest-frame-endpoint-uses-the-wrong-power-of-r.md`.
- Changed: `rust/src/{kernels,photon,special}.rs`,
  `hazma/spectra/_photon/{__init__.py,_muon.pyx}`, `hazma/_core.pyi`,
  `setup.py`, `test/parity/{cases,tolerances,test_parity}.py`,
  `test/test_core_{dispatch,special}.py`, `docs/followups/README.md`.
- Deleted: `hazma/spectra/_photon/_muon.pyi`.

### Task 4.4

- New: `rust/src/kernels/photon_pion.rs`, `test/test_core_photon_pion.py`,
  `docs/followups/todo/charged-pion-photon-spectrum-misses-the-forward-cone.md`.
- Changed: `rust/src/{kernels,photon}.rs`,
  `hazma/spectra/_photon/{__init__.py,_pion.pyx}`, `hazma/_core.pyi`,
  `setup.py`, `test/parity/{cases,tolerances}.py`,
  `test/test_core_dispatch.py`, `docs/followups/README.md`.
- Deleted: `hazma/spectra/_photon/_pion.pyi`, and the two `def`s in
  `_pion.pyx` (42 lines — the file stays, capi survivor).

## Verification

- Per task: corpus suite for the swapped entry points + full pytest +
  import smoke (mediator modules must stay importable — capi survivors
  intact).
- **After Task 4.2 (2026-08-12):** bare `pytest -q` →
  `1628 passed, 15 skipped in 587.90s`; collection 1458 → 1643 against
  `origin/master`, all +185 in `test/test_core_photon_tables.py`.
  `pytest -q test/parity` → `629 passed, 1 skipped`, all seven tabulated
  photon cases green at `TABULATED`. `cargo test --no-default-features`
  → `96 passed` (15 new). `python test/parity/generate.py --check` →
  `corpus OK: 41 cases / 1580 arrays`. `scripts/agents/preflight.sh` **RESULT: PASS**.
- **After Task 4.3 (2026-08-16):** bare `pytest -q` →
  `1682 passed, 15 skipped, 5 warnings in 559.08s`; collection 1643 → 1697
  against `origin/master` (+53 in `test/test_core_photon_muon.py`, +1 in
  `test/test_core_special.py`). `pytest -q test/parity` →
  `629 passed, 1 skipped`, `spectra.photon.muon` green at `SPECFUN` in all
  five blocks with a measured difference of exactly zero.
  `cargo test --no-default-features` → `109 passed` (13 new: 11 in
  `kernels::photon_muon`, 2 in `special`). A five-mutation validity
  campaign is in the task note — **one of the five is invisible to the
  corpus and to `cargo`**, and only the per-kernel bit-equality sweep
  catches it.
- **After Task 4.4 (2026-08-17):** bare `pytest -q` →
  `1755 passed, 15 skipped, 8 warnings in 605.61s`; collection 1697 → 1770
  against `origin/master`, **+73 and all of them
  `test/test_core_photon_pion.py`**. `pytest -q test/parity` →
  `629 passed, 1 skipped`, both pion cases green in all five blocks —
  `spectra.photon.charged_pion` at 2.618e-15 worst relative against the
  1e-12 budget this task tightened it to, `spectra.photon.neutral_pion`
  **bit-equal** at all 1,305 values. `cargo test --no-default-features` →
  `120 passed` (11 new, all `kernels::photon_pion`).
  `pytest -q test/test_theory_aggregation.py` → `69 passed` either side of
  the swap. An **eleven-mutation** validity campaign is in the task note,
  with **two survivors** — an FMA site inside the quadrature integrand and
  a one-ulp constant, both unobservable through the entry point for the
  same reason, both recorded in the source.
- **CI round 1 on PR #68** surfaced one Linux failure across all five
  ubuntu jobs and none on macOS: the divergent-regime assertion at
  `E_γ = 1.0, E_π = 4e4`, measuring **6.2998e-10** against a 1e-10 budget
  (2.8e-11 on macOS). Budget raised to a flat 1e-8 with both measurements
  recorded. The converged-regime comparisons — the ones the phase's
  history would have predicted as the casualty — passed on Linux at 1e-12
  across py3.10–3.14.

## Open Questions

- Run Phase 05 in parallel? (Project-level question; nothing in Phase 04
  has blocked on it so far.)
- **Should the corpus's mode switch become per-case?** Task 4.1 measured
  the cost of the global one: 22 of 41 cases now run at their declared
  budget rather than `rtol = 0`, though the 19 `EXACT`-class cases lose
  nothing. Belongs with
  [`parity-corpus-pins-ill-conditioned-points.md`](../../../../docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md),
  not beside a kernel swap.
- ~~**Task 4.2 is the first task that meets one of the six
  ill-conditioned corpus blocks** (`spectra.photon.eta`). Resolve or
  explicitly waive that follow-up before starting it.~~ **Waived by Task
  4.2 on evidence**: the port is bit-equal at 336,000 sampled points, so
  the block held. The follow-up stays open for its five cross-section
  blocks; nothing in Phase 04 is expected to meet those.
- **Does the φ omit a `φ → π⁰γ` line entirely?** `BR_PHI_TO_PI0_A` is
  defined and read by nothing, and the ω adds the analogous line. Not
  settled by Task 4.2 — recorded on
  [`phi-photon-lines-use-the-daughter-meson-energy.md`](../../../../docs/followups/todo/phi-photon-lines-use-the-daughter-meson-energy.md)
  for whoever repairs the line energies.
- ~~**Does the rest-frame endpoint defect reach `_pion` and `_rho`?**
  (Task 4.3.)~~ **Answered by Task 4.4: no.** `_pion` evaluates the muon
  kernel at the fixed `ENG_MU_PIRF = 109.78` MeV — 4.1 MeV above `m_μ`,
  never within one `DBL_EPSILON` of rest — at every pion energy, so the
  truncated branch is unreachable from there. `_rho` boosts `_pion`, not
  the muon kernel, so the same argument covers Task 4.5.
- **Nothing gates the FMA map of a quadrature integrand** (Task 4.4). A
  `hazma._core` test-surface probe over `kernels::photon_pion` would fix
  it, but it would widen `cases._CORE_TEST_ONLY_MODULES`, which Task 3.2
  warned against doing to quiet a check. Left as a known limitation;
  **Task 4.5 has the same shape one level deeper and is the right place to
  decide whether the machinery is worth it.**
- **Does the forward-cone defect reach `_photon/_rho`?** (Task 4.4.)
  Almost certainly — the ρ quadratures over the charged pion — but
  unmeasured, and whether the ρ's own quadrature compounds the loss is
  open. Recorded on the follow-up; Task 4.5 is positioned to answer it.
- **Should the corpus's *budget granularity* become per-block?**
  (Task 4.3.) The `spectra.photon.muon` failure was a 1.15e-14 absolute
  difference on a block whose peak is 17.2 — a per-block `atol` would have
  absorbed it with no loss of gate strength, where the only available
  lever was a 300x `rtol` widening. Belongs with
  [`parity-corpus-pins-ill-conditioned-points.md`](../../../../docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md)
  alongside the per-case mode-switch question above.

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).

## Handoff to Next Task

**For the next agent working in Phase 04:** read `../../PLAN.md`,
`../README.md`, this file, then the phase file — its Goal carries the
eight-step swap recipe *and* the capi-survivor exception. **Task 4.5
(`_photon/_rho`) is next**, and it is the phase's declared numerical
stress test: a quadrature whose integrand is `_pion`'s quadrature, which
is itself a quadrature over `_muon`. Its corpus class is `NESTED`
(`rtol = 1e-6`) — the loosest budget in the file — so measure before
touching it, and expect the two Task 4.4 findings below to apply one level
deeper. `_rho.pyx` is **not** a capi survivor: no `.pyx` cimports it, so
the whole file goes in the swap PR (check `rg` at execution time rather
than trusting this sentence). Deleting the `_photon/_muon`,
`_photon/_pion`, `_positron/_muon` or `_positron/_pion` extensions still
breaks the mediator imports.

**Currently safe to assume:**

- The foundation (interp, boost, quad, dispatch, constants) is unit-tested
  against scipy and NumPy, and Tasks 4.1–4.4 have exercised
  `constants::{pdg,derived}`, all four `boost::*`, `interp::interp`,
  `special::spence`, `quad::quad` and `dispatch::map_unary` through eleven
  real kernels end to end.
- **`kernels::photon_pion::{dnde_photon_charged_pion,
  dnde_photon_neutral_pion, charged_pion_integrand, dnde_pi_to_lnug}` are
  `pub` and PyO3-free.** The neutral one is **bit-equal** to the `cdef`
  `_rho.pyx` cimports (9,000 points, 0 mismatches); the charged one agrees
  to **6.5e-15** worst relative over 8,000 points, which is as close as two
  independent adaptive quadratures get. Task 4.5 should call both Rust
  `fn`s directly as its integrands rather than routing through Python.
- **`crate::quad` is proven on a live shape.** Task 4.4 is its first real
  consumer: `CHARGED_PION_QUAD` copies the `.pyx`'s
  `epsabs=1e-10, epsrel=1e-5, points=[-1, 1]` verbatim, the break points
  are both discarded by scipy's filter (so `qagpe` runs over an empty
  list, exactly as Task 3.3 predicted), and the termination flag matches
  scipy's at all 88 points of an 11 × 8 grid reaching `E_π = 1e5` MeV.
- **`hazma._core.photon` serves ten kernels**: the seven tabulated meson
  spectra (4.2), the radiative muon spectrum (4.3) and both pion spectra
  (4.4). `rust/src/photon.rs` now shows three registration shapes — a
  guard resolved once before any element is mapped (the tabulated seven),
  a kernel that guards per element (the muon), and a kernel that runs a
  quadrature per element (the charged pion).
- **`test/test_core_photon_pion.py` is the module to copy when a kernel
  has no bit-equality mode.** 73 tests, two oracle classes at two
  standards, and the reasoning for the split in the module docstring.
  `test/test_core_photon_muon.py` remains the copy for a kernel that
  *does*, and `test/test_core_photon_tables.py` for one whose twin does
  not survive the PR.
- **`test/parity/tolerances.py` has a `PORTED_QUAD_RTOL = 1e-12`**, taken
  by `spectra.photon.charged_pion` only. The other two `QUAD` cases keep
  the 1e-8 opening figure until Task 4.6 measures them. `NESTED` is
  untouched and Task 4.5 owns the same decision for it.
- The corpus is in budget mode from Task 4.1 and **cannot be
  regenerated**. `EXACT`-class cases are still `rtol = 0` — and
  `spectra.photon.neutral_pion` now *passes* at `rtol = 0` on Rust.
- `test/test_core_dispatch.py`'s spectra oracle is `_photon/_rho`, which
  Task 4.5 deletes. There is no fourth photon candidate; see Decisions.

**Currently risky / unknown:**

- **Six blocked defects now share one eventual corpus regeneration** —
  the positron normalization (4.1), the boost integral (3.4), the η′ line
  weight and the φ line energies (both 4.2), the muon photon spectrum's
  rest-frame endpoint (4.3), and the charged pion's lost forward cone
  (4.4). Do not "fix" any of them in passing; each fails the gate that
  governs the remaining swaps.
- **The forward-cone defect probably reaches the ρ** — the ρ quadratures
  over the charged pion — and Task 4.5 is the task positioned to measure
  it. See Open Questions.
- **Nothing gates the FMA map of a quadrature integrand**, and the ρ's is
  two integrations from anything observable. Read the disassembly, and do
  not read a green suite as confirmation of the map.
- **Do not inherit Task 4.3's `1/β` prediction unexamined.** It did not
  apply to `_pion` (see Findings) — but the ρ *does* sweep the pion's own
  `β`, so it may apply there. Re-derive.
- Nested-ρ drift is the project's numerical stress test: attribute a
  difference to a single term before touching any budget. Doing that in
  4.3 turned a proposed 300x widening into a 60-line fix; doing it in 4.4
  turned "the quadrature diverges" into "the quadrature diverges where
  scipy does, and agrees with it there".
