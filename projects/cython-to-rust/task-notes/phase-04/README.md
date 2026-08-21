# Working Memory: Phase 04 — Spectra kernels

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Phase:** 04
**Status:** **Complete (2026-08-20)**
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
| 4.5 | `_photon/_rho` (nested quad) | 4.4 | **Complete (2026-08-18)** | [task-4.5-photon-rho.md](task-4.5-photon-rho.md) |
| 4.6 | `_positron/_pion` + neutrino pair | 4.1, 4.3 | **Complete (2026-08-20)** | [task-4.6-positron-pion-neutrino.md](task-4.6-positron-pion-neutrino.md) |

## Exit Criteria

- [x] All rows Complete; phase file frontmatter `status: Complete`.
- [x] Phase learnings at
      [`../../learnings/phase-04-spectra-kernels.md`](../../learnings/phase-04-spectra-kernels.md).

**Both discharged 2026-08-20 with Task 4.6.** Read the learnings rather
than this file: it is the distillation of six tasks and sixteen entry
points, and everything below is history.

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
  pattern between them. The **flags** agree; the **values** there do not
  have to. Two CI rounds on PR #68 measured 2.8e-11 (macOS/arm64),
  6.2998e-10 and 3.0552e-08 (Linux/glibc, two different grid points, each
  bit-identical across py3.10–3.14). Asserting 1e-10 failed; 1e-8 moved
  the failure to the next point. **Do not chase it** — the test now
  partitions its grid by scipy's own convergence verdict, holding the
  converged half to 1e-12 (measured worst 2.22e-16, one ulp, because the
  two subdivide identically there) and the non-converged half to sign plus
  a factor of two. Tasks 4.5–4.6 and Phase 06 should partition the same
  way rather than look for one tolerance.
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

- **`scipy.integrate.quad` short-circuits `a == b` before QUADPACK is
  reached, and `crate::quad` did not** (Task 4.6). The Python wrapper
  returns `(0., 0.)` without calling the integrand at all
  (`scipy/integrate/_quadpack_py.py:436`, verified with a counting
  integrand); the port handed `[x, x]` to `qagse`, where every
  Gauss-Kronrod node collapses onto the point and a singular integrand
  gives `f(x)·0 = NaN`. Live at `dnde_neutrino_charged_pion(0.0, epi)`,
  whose integrand is `(dN/dE)/E`. Fixed in `crate::quad::quad`, which is
  where the gap is, not in the kernel. **The existing test could not see
  it** because its integrand was `exp` — smooth, so the two paths agree —
  which is the general shape: *a degenerate-input test needs a
  degenerate integrand.*
- **A folded compile-time constant can be folded with contraction**
  (Task 4.6). `_positron/_pion.pyx`'s `emax_pi_rf` is a module-level
  `cdef double` that clang reduces to one stored immediate, and the
  immediate is **one ulp above** the unfused expression because
  `1.0 + β·√…` fused. Reproducing it takes a literal plus a `mul_add`
  re-derivation; a `const` expression lands one ulp low. Brute-forcing
  the eight fused/unfused combinations against the immediate is how the
  responsible term was isolated.
- **Two files can contract nothing for opposite reasons** (Task 4.6).
  `_photon/_rho.pyx` boxes its untyped locals, so clang sees no
  expression; `_neutrino/_pion.pyx` types every local and simply contains
  no multiply-add. Same `grep -c` answer, different cause — one more
  reason step 1 of the recipe reads the disassembly rather than the
  source.
- **The two muon files disagree about the Michel normalization and only
  one is wrong** (Task 4.6). `_neutrino/_muon.pyx` **multiplies** by
  `R_FACTOR`, which is correct — both its rows integrate to exactly one
  neutrino — while `_positron/_muon.pyx` divides and is 0.0374% low.
  A reader who meets Task 4.1's defect first will be tempted to "fix"
  the neutrino kernel; both sides are now asserted so that attempt fails.
- **The phase's seventh live defect: the charged pion's `π → e ν`
  neutrino line is added twice** (Task 4.6). `_pion.pyx` sums two `cdef`s
  that were meant to partition the decay modes, and both add the boosted
  electron-neutrino line; the muon row has no second copy. Measured by
  continuum subtraction at **exactly 2.0000** copies against the muon
  line's 1.0000. Filed as
  [`neutrino-pion-electron-line-counted-twice.md`](../../../../docs/followups/todo/neutrino-pion-electron-line-counted-twice.md);
  unlike its six siblings it needs no Cython oracle, so its twin's
  deletion costs nothing.
- **An absolute `DBL_EPSILON` guard on a MeV quantity is not a tolerance
  band** (Task 4.6). `fabs(epi - mpi) < DBL_EPSILON` at
  `m_π = 139.57` MeV admits exactly one double — `m_π` itself — because
  one ulp there is 2.8e-14, 128x `DBL_EPSILON`. The two-sided `fabs` is
  inoperative, and `_positron/_pion` is also the one kernel that returns
  **zero** at rest rather than a rest-frame value.

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
  conditioning budget to absorb. (Off it the parity suite did not run at
  the time; `ci.yml`'s `--ignore=test/parity` came out on 2026-08-18.)
  The follow-up's prediction that "every affected block will produce a
  false failure the moment a Rust implementation lands" is **refuted for
  `spectra.photon.eta`** — and the closing measurement confirms it: the
  ported eta is bit-identical on macOS/arm64, Linux/aarch64 and
  Linux/x86_64, and carries no mask. The four *cross sections* were the
  real population, and they are masked rather than budgeted.
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
  [`parity-corpus-pins-ill-conditioned-points.md`](../../../../docs/followups/done/parity-corpus-pins-ill-conditioned-points.md)
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
- **`test/test_core_dispatch.py`'s spectra oracle is
  `_positron/_pion`** (Task 4.5), having been `_photon/_muon` →
  `_photon/_pion` → `_photon/_rho` across Tasks 4.3–4.4. The photon
  candidates are exhausted; the positron one has the identical
  `hasattr(__len__)` / `assert` / array-return shape and differs only in
  saying `"Positron energies"`, which those tests read from source.
  **Task 4.6 deletes that `def` too and there is no unary candidate after
  it** — the neutrino entry points return a 3-tuple or a `(3, N)` array.
  `TestCythonMessageParity` is unaffected either way — **but not for the
  reason this bullet gave before Task 4.6 landed.** Its `"Photon
  energies"` roster entry lived in
  `hazma/spectra/_neutrino/_muon.pyx:205` (at `ed1fa20`), the copy-paste
  defect Task 3.5 recorded, and that file is *deleted* by Task 4.6, so
  the entry does **not** survive Phase 04. What the roster does instead is
  shrink with the tree: it now reads two `assert` wordings, both in the
  mediator decay-spectrum modules, and the two wordings the port still
  emits are pinned in each kernel's own test module.
- **The `.pyx` that boxes its locals contracts nothing** (Task 4.5).
  `_rho.pyx`'s untyped `cdef beta/gamma/emin/emax/pre` make every
  arithmetic operation a `PyNumber_*` call, so `objdump` finds **zero**
  FMA instructions in the whole object — the only Phase 04 kernel where
  that is true. Porting such a file means writing the arithmetic out
  plainly; a `mul_add` would be the error. Check for untyped locals before
  assuming a `.pyx` contracts anything.
- **When a mutation survives, look for a seam before calling it
  untestable** (Task 4.5). Its six-mutation campaign had one survivor —
  fusing the boost window's `γ·E·(1−β)` — invisible to `cargo`, to 49
  per-kernel tests and to 10 parity blocks, because the outer `epsrel` is
  1e-5. Lifting the three lines into a `boost_window(e, erho)` `fn` and
  pinning `(emin, emax, pre)` bit-for-bit closed it, at no cost to the
  public surface. This is the answer to Task 4.4's open question about a
  `hazma._core` kernel probe: not needed.

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

### Task 4.5

- New: `rust/src/kernels/photon_rho.rs`, `test/test_core_photon_rho.py`,
  `docs/followups/todo/rho-rest-frame-branch-returns-the-integrand.md`.
- Changed: `rust/src/{kernels,photon,constants,quad}.rs`,
  `rust/src/kernels/photon_pion.rs`,
  `hazma/spectra/_photon/__init__.py` (both wrappers repointed, both
  gained return units), `setup.py`, `test/parity/{cases,tolerances}.py`,
  `test/test_core_{dispatch,constants,photon_pion,quad}.py`,
  `docs/followups/README.md`,
  `docs/followups/todo/charged-pion-photon-spectrum-misses-the-forward-cone.md`.
- Deleted: `hazma/spectra/_photon/_rho.{pyx,pxd,pyi}` — the whole module,
  not just its `def`s: nothing cimported it, so the capi exception in the
  phase Goal does not apply.

### Task 4.6

- New: `rust/src/kernels/{positron_pion,neutrino_flavors,neutrino_muon,
  neutrino_pion}.rs`, `test/test_core_positron_pion.py`,
  `test/test_core_neutrino.py`,
  `docs/followups/todo/neutrino-pion-electron-line-counted-twice.md`.
- Changed: `rust/src/{kernels,positron,neutrino,quad,constants}.rs`,
  `hazma/spectra/_positron/{__init__.py,_pion.pyx}`,
  `hazma/spectra/_neutrino/__init__.py`, `setup.py`,
  `test/parity/{cases,tolerances}.py`,
  `test/parity/oracles/entry_points.py`,
  `test/test_core_{dispatch,constants}.py`, `docs/followups/README.md`,
  `../../phases/phase-04-spectra-kernels.md`, `../../PLAN.md`,
  and — correcting a claim about this task that was always wrong —
  `../../../parity-pinned-defect-repair/{PLAN.md,references/defect-blast-radius.md}`.
- Deleted: nine files — `hazma/spectra/_positron/_pion.pyi`, and the
  eight under `hazma/spectra/_neutrino/`
  (`_{muon,pion}.{pyx,pxd,pyi}` plus `_neutrino.{pyx,pxd}`).

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
- **After Task 4.5 (2026-08-18):** bare `pytest -q` →
  `1802 passed, 15 skipped, 6 warnings in 47.15s`, from `1755 / 15`
  measured on the stashed pre-task tree — **+47**, being 49 new tests in
  `test/test_core_photon_rho.py` less 2 parameterized rows retired from
  `test/test_core_constants.py` with `derived::photon_rho`
  (`test/test_core_dispatch.py` is unchanged at 118: the oracle module was
  swapped, not the assertions). `pytest -q test/parity` →
  `629 passed, 1 skipped`, both ρ cases green in all five blocks at the
  **tightened** `PORTED_NESTED_RTOL` (1e-9) against a measured worst of
  1.5e-13 (charged) and 3.2e-15 (neutral).
  `cargo test --no-default-features` → `133 passed`, from 121 (+12, all
  `kernels::photon_rho`). A **six-mutation** validity campaign is in the
  task note with **one survivor**, and unlike Task 4.4's the survivor was
  *fixed*: the arithmetic was lifted out of the integral into
  `boost_window` and pinned bit-for-bit, so the campaign closes 6 / 6.
- **CI rounds 1 and 2 on PR #68** both surfaced one Linux failure across
  all five ubuntu jobs and none on macOS, both in the same
  divergent-regime assertion and at *different* grid points: **6.2998e-10**
  at `E_γ = 1.0, E_π = 4e4`, then **3.0552e-08** at `E_γ = 0.01,
  E_π = 6e4` once the budget was raised to 1e-8. Resolved by restructuring
  the test rather than widening again — see Findings. The converged-regime
  comparisons, which the phase's history would have predicted as the
  casualty, passed on Linux at 1e-12 in both rounds.
- **After Task 4.6 (2026-08-20), which closes the phase:** bare
  `pytest -q` → `1935 passed, 15 skipped, 7 warnings in 151.28s`, from
  `1831 / 15` on `origin/master` — **+104**, being 48 new tests in
  `test/test_core_positron_pion.py` plus 58 in `test/test_core_neutrino.py`
  less 2 parameterized rows retired from `test/test_core_constants.py`
  with `derived::neutrino_muon` (`test/test_core_dispatch.py` unchanged at
  118: the oracle module was swapped, not the assertions).
  `pytest -q test/parity` → `658 passed, 1 skipped`, all three swapped
  cases green — `spectra.neutrino.muon` **bit-equal at all 3,795 pinned
  values** at `EXACT_RTOL`, `spectra.positron.charged_pion` at 5.494e-15
  and `spectra.neutrino.charged_pion` at 9.739e-16 against the 1e-12 both
  were **tightened** to. `cargo test --no-default-features` →
  `169 passed`, from 133 (+36 across the four new kernel modules).
  `pytest -q test/test_theory_aggregation.py` → `69 passed` either side of
  the swap. An **eleven-mutation** validity campaign is in the task note,
  with two survivors — one lifted out into `neutrino_pion::boost_window`
  and killed, the other shown unobservable *by construction* and recorded
  as such, so the campaign closes 11 / 11.
- **PR #74's first CI round is the phase's fourth Linux-only failure, and
  the same class as the first three.** One assertion —
  `test_core_positron_pion.py`'s kinematic-edge sweep at `E_π = 1e6` MeV —
  green on macOS/arm64 and red on all five ubuntu jobs at 7.5e-9 relative
  plus a delta-function support flip. Cause: `emin = γ(E − βk)` is a
  catastrophic cancellation whose relative error grows like `2γ²ε`
  (2.3e-8 at γ = 7165), **and** clang cannot contract `E − β·k` on x86-64
  without `-march` because SSE2 has no FMA, so the shipped Cython is fused
  on the capturing platform and unfused on Linux while the port's
  `mul_add` is fused on both. Resolved by bounding the module's grids to
  `E_π = 1e4` and asserting the mechanism, not by widening the budget. The
  lesson is Task 4.1's, one level out: **a budget derived from one
  platform's measurement must state the range it holds over**, or the
  first CI run finds the range for you.

## Open Questions

- Run Phase 05 in parallel? (Project-level question; nothing in Phase 04
  has blocked on it so far.)
- **Should the corpus's mode switch become per-case?** Task 4.1 measured
  the cost of the global one: 22 of 41 cases now run at their declared
  budget rather than `rtol = 0`, though the 19 `EXACT`-class cases lose
  nothing. Belongs with
  [`parity-corpus-pins-ill-conditioned-points.md`](../../../../docs/followups/done/parity-corpus-pins-ill-conditioned-points.md),
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
- ~~**Nothing gates the FMA map of a quadrature integrand** (Task 4.4).~~
  **Decided by Task 4.5: no probe, and none needed.** Its mutation
  campaign found the one survivor was arithmetic in the integration
  *limits* rather than inside the integrand, so lifting it into a
  `boost_window(e, erho) -> (emin, emax, pre)` `fn` and pinning the three
  bits made it a plain `cargo` test — no `_CORE_TEST_ONLY_MODULES`
  widening, no new public surface. The 15 sites *inside*
  `dnde_pi_to_lnug` remain ungated and remain defended by disassembly and
  review; that limitation stands, narrowed.
- ~~**Does the forward-cone defect reach `_photon/_rho`?**~~ **Answered by
  Task 4.5: yes, and it compounds.** A pure inheritance would preserve the
  fraction of the endpoint at which the cliff sits, because
  `γ(1−β)·γ(1+β) = 1`; the inner kernel's 0.945 should therefore appear at
  every ρ energy. Measured, the charged ρ runs 0.9963 at `γ_ρ = 1.05` down
  to **0.5366** at `γ_ρ = 10` (neutral: 0.9420 → 0.5073), because the
  outer window spans decades while its integrand is nonzero only near the
  bottom. **Repairing the charged-pion kernel is necessary but not
  sufficient for the ρ.** Table and consequence recorded on the follow-up.
- **Should the corpus's *budget granularity* become per-block?**
  (Task 4.3.) The `spectra.photon.muon` failure was a 1.15e-14 absolute
  difference on a block whose peak is 17.2 — a per-block `atol` would have
  absorbed it with no loss of gate strength, where the only available
  lever was a 300x `rtol` widening. Belongs with
  [`parity-corpus-pins-ill-conditioned-points.md`](../../../../docs/followups/done/parity-corpus-pins-ill-conditioned-points.md)
  alongside the per-case mode-switch question above.
- ~~**`dnde_photon_{charged,neutral}_rho` state no return units**~~
  **Done in Task 4.5' swap.** All **12** entry points in
  `hazma/spectra/_photon/__init__.py` now carry
  `Units are MeV^-1; … are both in MeV.`
  (`grep -c 'Units are MeV' hazma/spectra/_photon/__init__.py` → 12).

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).

## Handoff to Next Task

**Phase 04 is closed (2026-08-20).** Read
[`../../learnings/phase-04-spectra-kernels.md`](../../learnings/phase-04-spectra-kernels.md)
rather than this file — it is the distillation of six tasks and sixteen
entry points, and everything above is history kept for provenance.

**For the next agent starting Phase 05 or 06:**

1. `../../PLAN.md`, then `../README.md`, then the phase-04 learnings,
   then the target phase file.
2. Phase 05 must read
   [`../phase-01/followup-parity-corpus-stability.md`](../phase-01/followup-parity-corpus-stability.md)
   before porting the scalar cross sections — 494 pinned positions in the
   four it ports assert nothing.
3. Phase 06's Task 6.4 is the only place the four capi survivors
   (`_photon/{_muon,_pion}`, `_positron/{_muon,_pion}`) and the `_utils`
   headers go.

**Currently safe to assume:**

- **`hazma/spectra/` holds no Cython Python entry point of any kind.**
  `cases.rust_core_kernels()` → **16**; **11 `.pyx` and 8 `.pxd`** remain
  in the tree, all of them Phase 05/06 business. Re-derive with a clean
  rebuild rather than quoting.
- **Every kernel module under `rust/src/kernels/` is `pub` and
  PyO3-free**, so Phase 06's mediator spectra call them natively the way
  the `.pyx` cimport the Cython today.
- **`crate::quad` short-circuits an empty interval**, as scipy does
  (Task 4.6). Any Phase 05/06 kernel whose limits can coincide inherits
  the fix.

**Currently risky / unknown:**

- **Eight blocked defects now share one eventual corpus regeneration** —
  the seven this phase and Task 3.4's found, plus the boost integral.
  Do not "fix" any in passing.
- **`test_core_dispatch.py`'s `TestDeclaredDivergencesFromCython` has one
  Cython oracle left in the tree** (`scalar_mediator_decay_spectrum`), and
  Phase 06 deletes it. Retire the class or re-express its widenings
  against `cython_xs`.
- **Phase 05 still has to name the cross sections' `quantity` wording.**
  They carry no dispatch message today, so the port invents it and it is
  user-visible from the first swap. `"Center-of-mass energies"` is the
  placeholder `test/test_core_dispatch.py` uses.
