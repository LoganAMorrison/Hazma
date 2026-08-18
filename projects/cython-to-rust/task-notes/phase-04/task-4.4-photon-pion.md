# Task 4.4: `_photon/_pion`

**Date:** 2026-08-17
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-04-spectra-kernels.md` (Goal — the
eight-step swap recipe and the capi-survivor exception; Task 4.4),
`../../rules.md` rules 1–3 (parity discipline), 4 (constants), 6–9 (Rust
conventions)
**Related ADRs:** ADR-0001 (framework), ADR-0002 (license-clean numerics —
the QUADPACK port this task is the first real consumer of)
**Depends On:** Task 4.3 (`kernels::photon_muon`, which this kernel calls
natively), Task 3.3 (`crate::quad`), Task 3.1
(`constants::derived::photon_pion`)

## Objective

Port `hazma/spectra/_photon/_pion.pyx` — the charged pion's radiative
photon spectrum (a `qagp` over `cos θ` whose integrand boosts the muon
spectrum) and the neutral pion's `π⁰ → γγ` box — to
`hazma._core.photon.dnde_photon_{charged,neutral}_pion`, keeping the Rust
`fn`s in the PyO3-free layer so Task 4.5 and Phase 06 can call them
natively.

## Exit Criteria

From the phase file's Task 4.4 block:

- Both entry points (charged: π→ℓνγ radiative + boosted-μ `qagp` over
  cosθ; neutral: π⁰→γγ box) corpus-green within the quad budget.
  **Met, and the neutral one better than asked:** `spectra.photon.charged_pion`
  is green at 2.618e-15 worst relative against a budget this task
  *tightened* to 1e-12, and `spectra.photon.neutral_pion` is **bit-equal**
  at all 1,305 pinned values at its `EXACT` (`rtol = 0`) budget.
- First real `qagp` consumer — record measured drift vs the corpus in the
  task note and tighten the budget if warranted. **Met:** drift table
  below; `test/parity/tolerances.py` gains `PORTED_QUAD_RTOL = 1e-12` and
  `spectra.photon.charged_pion` moves onto it from `QUAD_RTOL = 1e-8`. The
  tightening is justified by measurement, not by taste — see
  "The tightening buys exactly one thing, and it is the one that nearly
  bit" below.

Plus the phase Goal's recipe, steps 1–8: FMA map read from the shipped
`.so` before writing Rust; kernel in `kernels/<pyx name>.rs`; both entry
points registered through `dispatch::map_unary` with the twins' wording;
wrappers repointed; both corpus cases repointed with `PORTED_ENTRY_POINTS`
rows; the twins' `def`s deleted (capi survivor — the file stays);
`test/test_core_photon_pion.py` added; drift recorded here and in the
working-memory README.

## Inputs Reviewed

- `../../PLAN.md` (Numerical impact), `../../phases/phase-04-spectra-kernels.md`,
  `../../rules.md`, `../README.md` (phase working memory), `../../../README.md`
  — read the Handoff and Findings sections before touching anything.
- `task-4.1-positron-muon.md` (the template), `task-4.3-photon-muon.md`
  (the immediate dependency, and the source of the `1/β` amplification
  warning this task was told to expect).
- `../../references/cython-inventory.md` (cimport DAG — who still needs
  `_pion`'s capsules), `../../references/numerics-replacements.md`
  (the quad call-site table).
- `docs/agents/lessons.md`, `docs/agents/environment.md`,
  `docs/agents/doc-consistency.md`, `docs/agents/preflight.md`.
- Sources: `hazma/spectra/_photon/_pion.{pyx,pxd,pyi}`, its shipped
  `_pion.cpython-312-darwin.so` (disassembled),
  `rust/src/{quad,boost,photon,kernels,constants}.rs`,
  `rust/src/kernels/{photon_muon,photon_tables}.rs`,
  `test/parity/{cases,tolerances,test_parity}.py`,
  `test/test_core_{photon_muon,positron_muon,dispatch,quad}.py`,
  `hazma/spectra/_photon/__init__.py`, `setup.py`, `hazma/_core.pyi`.

## Findings

### `cdef float` in a file of `cdef double`s, and the corpus depends on it

`dnde_photon_neutral_pion_point` declares

```cython
cdef float beta = sqrt(1.0 - (MASS_PI0 / eng_pi)**2)
cdef float ret_val = 0.0
```

— **single** precision, in a file where every other local is a `double`,
and in the one entry point the corpus holds to `EXACT` (`rtol = 0`). The
shipped object confirms it with two `fcvt s, d` / `fcvt d, s` round trips,
one after the `fsqrt` and one on the returned quotient. A port that reads
the formula and not the declaration lands **8.5e-9** relative away —
bit-equal is impossible and `EXACT` is `rtol = 0`, so it fails at every
one of the 1,305 pinned values.

Written as `as f32 as f64` at both sites, the port is bit-equal to the
Cython at **9,000** sampled points across nine parent energies, 0
mismatches, and at all 1,305 corpus values. Mutation m1 (drop the `beta`
truncation) fails 6 per-kernel tests and 4 corpus blocks; m2 (drop the
return truncation) fails 14 and 4.

This is the third instance in this phase of a **declaration** carrying
numerics the formula does not: Task 3.1's mixed constant tables, Task 4.3's
`y**3` being a libm `pow` call, and now a `float` local. Read the `cdef`s,
not only the expressions.

### The FMA map: 19 sites, and 15 of them are untestable

`objdump -d` the shipped `.so`: 15 fused instructions in `dnde_pi_to_lnug`
(14 `fmadd`, 1 `fmsub`, 1 `fnmsub` — no vectorised `fmla` this time,
unlike Task 4.3), 4 in `charged_pion_integrand` (1 `fmsub` for
`1 − β cosθ` and 3 `fmadd` for the three-term accumulation onto a `0.0`
seed), and **zero** in either point function. Two neighbours look fusable
and are not, both for reasons worth carrying:

- `1 − β²` inside `boost_beta`, which both point functions inline —
  already recorded by `crate::boost`, and confirmed again here.
- `2r² − 2rx` and `r² − rx`: the *first* operand is an `fnmul`
  (`−(x·2r)`, `−(x·r)`) so the second can be the fused one. The negation
  is exact, so the value is the same either way — but which product gets
  the extra precision is not, and writing `(2.0*r*r).mul_add(...)` would
  fuse the wrong half.

**The 15 sites inside `dnde_pi_to_lnug` cannot be gated, and this task
measured that rather than assuming it.** They sit inside a quadrature:
unfusing one moves the integrand in its last bit, and the integral does
not carry that bit out. Mutation m4 (unfuse `F_A² + F_V²`) leaves the
worst corpus difference at **2.618e-15 — the identical figure the correct
port produces** — with 120 `cargo` tests and 73 per-kernel tests green.
Task 4.3's muon kernel had the same class of mutation caught by its
bit-equality sweep at 4 of 10,000 random arguments; this kernel has no
bit-equality mode to catch it with, because the port replaces *scipy's*
QUADPACK. So the map here is defended by the disassembly reading and by
review, and `rust/src/kernels/photon_pion.rs` says so at the top rather
than leaving the next reader to assume a gate exists. The 4 sites in
`charged_pion_integrand` *are* covered — they are outside the integrand's
own arithmetic, and m5 (unfuse the Doppler factor) turns a corpus block
red.

**Task 4.5 inherits this doubly**: `_photon/_rho` quadratures over this
kernel, so its own integrand's arithmetic is two integrations away from
anything observable.

### `f_π` has a one-ulp trap, and only the disassembly settles it

`FPI = DECAY_CONST_PI * M_SQRT1_2` is a module-level `cdef double`, so
clang loads it from memory rather than folding it into an immediate — the
one constant in this file the `movk` trick cannot pin. C's `M_SQRT1_2` is
the decimal literal `0.70710678118654752440`, which rounds to
`0x3FE6A09E667F3BCD`; Rust's `FRAC_1_SQRT_2` is the same double. But
`130.41 * (1.0 / 2.0_f64.sqrt())` is `0x40570DAED2A0781A` — **one ulp
low** — because `1/√2` computed as a division is not the correctly-rounded
`√0.5`. Mutation m6 (the division spelling) fails a `cargo` test and
nothing else: the difference is 1 ulp in a constant that multiplies the
integrand, so like the FMA sites it does not survive the quadrature.
`the_folded_constants_match_the_shipped_immediates` is what holds it.

### A sixth live 2.1.0 defect: the quadrature loses the forward cone

Found by asking what the port *should* return where it returns zero.

`charged_pion_integrand` is nonzero only where the pion-rest-frame photon
energy stays below `ENG_GAM_MAX_PIRG = 69.783` MeV, i.e. where
`cos θ > (1 − 69.783/(E_γ γ_π))/β_π`. As the photon energy approaches the
boosted endpoint that window narrows; once it is narrower than QUADPACK's
largest first-rule abscissa (~0.9956), **every node returns zero, the
error estimate is zero, and `qagp` terminates successfully with `0.0`**.
The spectrum is not zero there.

| `E_π` (MeV) | `γ_π` | boosted endpoint | first spurious zero, as a fraction of it |
| --- | --- | --- | --- |
| 200 | 1.4 | 171.6 | 0.99 |
| 500 | 3.6 | 490.1 | 0.99 |
| 1000 | 7.2 | 995.1 | 0.77 |
| 2000 | 14.3 | 1997.5 | 0.37 |
| 5000 | 35.8 | 4998.9 | 0.095 |
| 10000 | 71.6 | 9999.3 | 0.025 |

At `E_π = 10 m_π = 1396` MeV — the parity corpus's own most boosted block
— `dnde_photon_charged_pion(900, 1396)` is `0.0` against a reference of
`3.586e-07` MeV⁻¹, and `(1200, 1396)` is `0.0` against `1.135e-08`. The
reference integrates only the surviving window with
`scipy.integrate.quad(..., epsrel=1e-10)`.

The **integrated** effect stays small inside hazma's own domain and grows
fast outside it — photons per decay, shipped vs reference: 0.307432 /
0.307432 at `E_π = 200`, 0.333544 / 0.333562 (**0.0054%** missing) at
1000, 0.339368 / 0.339508 (**0.041%**) at 1396, 0.351920 / 0.362662
(**2.96%**) at 5000. So this is a **shape** defect rather than a yield
defect at hazma's scales: the differential spectrum is a hard zero over
roughly the top quarter of its support at `γ_π = 10`, which is what a
line search or a tail-dominated limit reads.

Reproduced per rule 1 — the port's zeros are in the *same places* as the
Cython's at every parent energy sampled — pinned by
`test_the_forward_cone_is_a_hard_zero_the_quadrature_invented`, and filed
as
[`charged-pion-photon-spectrum-misses-the-forward-cone.md`](../../../../docs/followups/todo/charged-pion-photon-spectrum-misses-the-forward-cone.md),
blocked behind Phase 06 Task 6.4 for the same corpus reason as its five
siblings. **Six blocked defects now share one eventual corpus
regeneration.**

### The divergent regime is reachable, and the port agrees with scipy inside it

Phase 03 Task 3.3 measured that the QUADPACK port tracks scipy *only*
where QUADPACK converges — past that Wynn's ε-algorithm is chaotic on a
non-converging sequence and the two can separate without bound — and made
"no live shape reaches the other regime" an obligation each consumer
re-checks. This is the project's first `qagp` consumer, so this is that
check.

It **is** reachable: over an 11 × 8 grid the first non-`Ok` flag appears at
`E_π = 4e4` MeV (`γ_π ≈ 290`), with the whole photon grid still `ier = 0`
at `3e4`. That is 40 GeV against a sub-GeV library, and two orders of
magnitude above the corpus's `10 m_π` ceiling.

And it is not a cliff. Over the same grid the port's `ier` **equals the
flag scipy raises on the Cython twin at all 88 points**, including both
`ier = 4` (`NoConvergence`) entries and the non-monotonic pattern in
between — the map is `ier = 5` at (1e-2, 6e4) but `0` at (1.0, 6e4), and
both implementations agree on that too. Values in the divergent regime
still agree to **2.8e-11** worst case on the capturing platform
(6.2998e-10 on Linux/glibc — see below).

The result is worth stating positively: Task 3.3's warning was that the
two implementations *may* separate without bound where QUADPACK does not
converge, and on the only live shape that reaches that regime, they do
not.

**And this is the one place in the kernel the platform reaches** — which
is the opposite of where the phase's own history says to look. Tasks 4.1
and 4.3 both learned `[platform-scoped-oracle-asserted-globally]` from a
*bit-equality-against-a-compiled-twin* assertion, so going into CI the
expected casualty here was `CHARGED_PION_BUDGET`, the flat 1e-12 this
module holds the port to on every platform. It survived: all three
converged-regime classes — swept grid, random arguments, kinematic edges,
at seven parent energies each — pass on Linux/glibc across py3.10–3.14.
The single Linux failure was the **divergent-regime** assertion, at
`E_γ = 1.0, E_π = 4e4`, where the separation is **6.2998e-10** against
2.8e-11 on macOS. Two things make it a clean reading rather than noise:
all five Linux jobs reported the same double, so it is a toolchain
property; and it lands exactly where Task 3.3 said the two are entitled to
diverge, and nowhere else.

**The first fix was wrong, and CI round 2 said so.** Raising the budget
to a flat 1e-8 with both measurements recorded looked principled and was
not: the assertion simply moved to the next-worst point, **3.0552e-08** at
`E_γ = 0.01, E_π = 6e4`. Two rounds of that is the shape
[`parity-corpus-pins-ill-conditioned-points.md`](../../../../docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md)
warns about — widening until it passes, in the one place the numerics are
least able to support a bound. **There is no honest tolerance to assert
in a chaotic regime, and the right move was to stop asserting one.**

The test now partitions its 64-point grid by *scipy's own convergence
verdict* — the `IntegrationWarning` the Cython raises — and holds each
half to what is true of it:

| Half | Points | Assertion | Measured worst (capturing platform) |
| --- | --- | --- | --- |
| QUADPACK converged | 51 | `CHARGED_PION_BUDGET`, 1e-12 | **2.22e-16** — one ulp |
| did not converge | 13 | same sign, within a factor of 2 | 2.75e-11 |

The converged half's headroom is not luck: where QUADPACK converges the
two implementations subdivide *identically*, so the only difference left
is the integrand's own last bit accumulated over a few hundred
evaluations. That is why it holds at 1e-12 on Linux too, in both CI
rounds, while the chaotic half moved by two orders of magnitude between
grid points. A test that asserts one tolerance across both halves is
asserting the wrong thing about one of them.

Two guards keep the partition honest: neither half may be empty
(`converged > 0 and diverged > 0`), and the two must sum to the grid — so
a future change that pushes the whole grid into one regime fails loudly
rather than silently reducing the test to its easy half. **Tasks 4.5–4.6
and Phase 06 should partition the same way** rather than hunt for a single
number; the ρ's nested quadrature will reach the chaotic regime sooner
than this one does.

### The tightening buys exactly one thing, and it is the one that nearly bit

`test/parity/tolerances.py`'s `QUAD` class docstring said the 1e-8 opening
budget would be tightened "once Phase 03 has measured the port". It has,
and this task is the first case to measure a live shape, so
`spectra.photon.charged_pion` moves to a new `PORTED_QUAD_RTOL = 1e-12`.
The other two `QUAD` cases keep 1e-8 until Task 4.6 measures them —
class-wide would pre-judge kernels nobody has run.

Whether that is worth doing is a measurement, not an opinion. Of the
mutations in the campaign below, **most are caught at either budget**:
m3 (`epsrel` 1e-5 → 1e-4) lands at 2.1e-05 and m10 (`epsabs` 1e-10 →
1e-8) at 0.9986, both far outside 1e-8. The two that 1e-8 would miss are
the ones 1e-12 also misses (m4, m6 — see above).

So the case for the tightening rests on **m11**, which was built to model
a real near-miss rather than to justify a number. Task 4.3 nearly shipped
a `spence` two ulps off scipy, which arrived amplified as a **3.2e-11**
relative shift in `dnde_photon_muon` — and `dnde_photon_muon` is this
kernel's integrand. Perturbing it by exactly that much moves
`spectra.photon.charged_pion` by **3.199e-11**: silently inside 1e-8,
loudly outside 1e-12. The scenario is not hypothetical; it is last week's
task, one convolution downstream.

### The `1/β` amplification the phase README predicted is not here

Task 4.3's handoff predicted the `rest_plus_eps` block would be the loud
one again in Tasks 4.4–4.5, since `_pion` boosts the muon kernel. It is
not: `rest_plus_eps` is the *pion's* `β = 1.4e-6`, and the pion's boost
enters as a Jacobian and a Doppler factor rather than as a `1/β`
prefactor — the `1/β` in the muon kernel is evaluated at
`ENG_MU_PIRF = 109.78` MeV, a fixed argument with `β = 0.27`, at every
pion energy. So this kernel never evaluates the muon spectrum near its own
rest frame at all, and the `rest_plus_eps` block is no worse than the
others (60 of 308 values not bit-equal, worst 2.98e-16, against 76 of 308
and 2.62e-15 for `boosted_mild`).

**Task 4.5 should not inherit the prediction unexamined either** — the ρ
boosts the pion, and the pion's own `β` *is* swept there.

### The `.pyx` carries a dead `cdef` that is not ported

`eng_gam_max` computes the lab-frame photon endpoint from
`ENG_GAM_MAX_MURF`, `GAMMA_MU_PIRF` and `BETA_MU_PIRF`. Nothing calls it,
it is not in `_pion.pxd`, and `rg` finds no other reader — so it is dead
inside a live module, and the port leaves it out. Its three constants stay
in `constants::derived::photon_pion` because Task 3.1 put them there and
`test_core_constants.py` pins them against the `.pyx` source; the corpus
also uses two of them as grid anchors. The formula is worth keeping in
mind anyway: it is exactly the endpoint the forward-cone defect above
needs, already written down in the file that has the defect.

## Decisions and Implementation Notes

- **Two entry points, two oracle standards, two test classes.** The
  neutral pion is closed form, so `test/test_core_photon_pion.py` gives it
  the template's two-mode comparison (bit-for-bit on the corpus's
  capturing platform, a peak-scaled budget elsewhere). The charged pion
  replaces scipy's QUADPACK with the in-tree one, and **two independent
  adaptive integrators are not bit-equal on any platform** — there is no
  capturing-platform branch to take, so it gets one measured budget
  everywhere. Splitting them into two classes is what makes that
  asymmetry a stated decision rather than an inconsistency.
- **`PORTED_QUAD_RTOL` is per-case, not class-wide.** See above. The
  constant is named for what it means (a `QUAD` case that has been ported
  and measured) so Tasks 4.5–4.6 can adopt it after their own measurement
  rather than inheriting a figure derived from this integrand.
- **The `Err` arm of `quad` returns `NaN` rather than panicking**, and a
  `cargo` test (`charged_pion_quad_options_are_always_accepted`) asserts
  the arm is unreachable: `QuadError` is a statement about the *options*
  (`epsabs > 0`, `limit` above the surviving break-point count) and these
  are `const`. `NaN` is the shape Task 4.2 settled for a per-element error
  channel that does not exist; a panic would take down a whole array where
  the Cython raises once.
- **`test/test_core_dispatch.py`'s spectra oracle moved again**, from
  `_photon/_pion` to `_photon/_rho` — the last photon `.pyx` with a Python
  entry point, and one whose `hasattr(__len__)`/`assert` shape is
  identical. **Task 4.5 swaps that one too and there is no fourth photon
  candidate**: the class docstring now says so and names the two remaining
  options (move to a `_positron`/`_neutrino` entry point, which Task 4.6
  then deletes, or retire the three spectra tests).
- **`spectra.photon.neutral_pion` keeps `EXACT`** — it is genuinely exact
  here, and unlike Tasks 4.2/4.3's kept-not-tightened classes there is
  nothing to tighten *to*. Worth saying because the reasoning is the
  mirror image: those two kept a loose class because bit-equality rested
  on a build property; this one's bit-equality rests on IEEE arithmetic
  and an `f32` cast, which are portable.
- **A mutation harness needs a guarded baseline, and this task proved it
  the hard way** — see Verification.

## Files Changed

*(This section describes **this task's** diff, on the branch
`claude/cython-to-rust/task-4.4-photon-pion`.)*

- New: `rust/src/kernels/photon_pion.rs`, `test/test_core_photon_pion.py`,
  `docs/followups/todo/charged-pion-photon-spectrum-misses-the-forward-cone.md`,
  this task note.
- Changed: `rust/src/{kernels,photon}.rs`,
  `hazma/spectra/_photon/{__init__.py,_pion.pyx}`, `hazma/_core.pyi`,
  `setup.py`, `test/parity/{cases,tolerances}.py`,
  `test/test_core_dispatch.py`, `docs/followups/README.md`,
  `../README.md`, `../../task-notes/README.md`.
- Deleted: `hazma/spectra/_photon/_pion.pyi`, and the two `def`s in
  `_pion.pyx` (42 lines — the file itself stays, capi survivor).

## Verification

### Gates

- `cargo test --manifest-path rust/Cargo.toml --no-default-features` →
  `120 passed; 0 failed` (109 → 120: **11 new**, all in
  `kernels::photon_pion`; `grep -c '#\[test\]'` on that file is 11).
- `cargo fmt --manifest-path rust/Cargo.toml --check` → clean.
- `cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings`
  → clean. One lint taken rather than allowed: `assign_op_pattern` on
  `poly_r = r + poly_r`, which is the same double under `+=`.
- `pytest -q test/test_core_photon_pion.py` → `73 passed`.
- `pytest -q test/parity` → `629 passed, 1 skipped`; both pion cases green
  in all five blocks.
- `pytest -q test/parity -k pion` → `20 passed, 610 deselected` (the fast
  loop used by the mutation campaign).
- Bare `pytest -q` → `1755 passed, 15 skipped, 8 warnings in 588.92s`
  (1682/15 at Task 4.3; collection 1697 → 1770, all +73 in the new test
  module).
- `scripts/agents/preflight.sh --paths … --md …` → **RESULT: PASS**.
  Two earlier runs failed, and on nothing but formatting and prose shape:
  the first because the venv's tools were not on `PATH` at all (`black not
  installed` / `pytest not installed` — the gate invokes whatever is on
  `PATH`, so a `uv` venv has to be exported first, and a `WARN` row there
  is a hole rather than a pass); the second on `black` wanting to reformat
  `test/test_core_dispatch.py`, six `PLR2004` magic-value comparisons in
  the new test module, and three markdownlint errors in this note and the
  project README. **No numerical or structural gate was ever red.** The
  six magic values were fixed with named constants rather than silenced,
  which is what Task 4.2 decided the last time this module class tripped
  the same rule.

### CI round 1 (PR #68, run 32084888951)

`Lint` and `Rust (fmt, clippy, test)` green; `Test (macos-latest, py3.14)`
green; **all five `Test (ubuntu-latest, py3.10–3.14)` red on exactly one
test**, with the identical figure in each:

```text
FAILED test/test_core_photon_pion.py::TestChargedPionAgainstTheCythonTwin
       ::test_the_termination_flag_agrees_with_scipy_across_the_divergent_regime
AssertionError: port and scipy separated at egam=1.0, epi=40000.0:
  -4.194057051835751e-06 vs -4.194057049193561e-06
assert 6.299842019142071e-10 < 1e-10
====== 1 failed, 1124 passed, 15 skipped, 9 warnings in 563.32s ======
```

### CI round 2 (PR #68, run 32087037574)

The round-1 fix — raising `DIVERGENT_REGIME_BUDGET` to 1e-8 — failed the
same way at a different point:

```text
AssertionError: port and scipy separated at egam=0.01, epi=60000.0:
  -6.903734330326644e-06 vs -6.903734541248649e-06
assert 3.055187072594416e-08 < 1e-08
====== 1 failed, 1124 passed, 15 skipped, 9 warnings in 793.74s ======
```

Resolved by restructuring the test to partition on scipy's convergence
verdict rather than by widening a third time — see "The divergent regime
is reachable" above.

**Across both rounds no other test moved**, which is the substantive
result: the flat `CHARGED_PION_BUDGET = 1e-12` holds on Linux across five
interpreters, in the converged-regime sweep classes *and* in the converged
half of this test.

### What the 73 per-kernel tests cover

- **Dispatch wiring (22 — 11 assertions × both entry points):** scalar →
  `float`, NumPy scalar and 0-d array on the scalar path, 1-D array →
  fresh non-aliasing `float64`, array path bit-equal to the scalar path,
  sequence accepted, empty grid, the rank message verbatim from the
  `.pyx`'s `assert`, non-`float64` dtype, non-number `TypeError`.
- **Keyword acceptance (1):** both entry points by keyword — the twins
  were `def`s.
- **Wrapper and public API (6):** private wrapper and `hazma.spectra` name
  both resolve to the core kernel; neither `def` survives on the `.pyx`;
  the four capsules the cimporters need are intact with the right
  signature names; `_rho` and a `VectorMediator.total_spectrum` still run.
- **Neutral pion vs the Cython (26):** swept grid and 4,000 random
  arguments at each of eight parent energies, the box edges and their
  `nextafter` neighbours, the support as a boolean mask, the `f32`
  truncation being load-bearing, the off-platform budget rejecting a real
  error, and `NaN` → `0.0` in both.
- **Charged pion vs the Cython (17):** swept grid and 300 random
  arguments at each of seven parent energies, the kinematic edges, the
  budget rejecting a real error, below-threshold exact zero, `NaN`
  propagation, and the 64-point termination-flag comparison across the
  divergent regime.
- **Physics (7):** the π⁰ box carries `2·BR` photons at four boosts, is
  flat, is symmetric about `E_π/2`; the charged pion at rest equals its
  own integrand assembled from parts; the spectrum vanishes above the
  boosted endpoint; positivity across the bulk; the radiative channels are
  a percent-level correction; and the forward-cone defect, pinned as
  deliberate.

Not covered, deliberately: the 15 FMA sites inside `dnde_pi_to_lnug`
(unobservable — see Findings), and `eng_gam_max` (dead, not ported).

### Test validity: an eleven-mutation campaign

Every mutation applied to a **pristine copy** and reverted from it, with
`cmp` against that copy before each run — the guard exists because the
first attempt at this campaign did not have one. `git checkout --` cannot
revert a file git has never seen, so on a **new** file the restore step
failed silently and five mutations accumulated; the results were read as
five independent measurements when only the first was. This is Task 3.3's
`[mutation harness poisons its own baseline]` in a new disguise, and the
tell was the same: implausibly uniform failure counts (`26 | 7` three
times in a row). The rebuilt harness aborts on a poisoned baseline and
verifies the restore.

| # | Mutation | `cargo` | per-kernel | corpus (pion) |
| --- | --- | --- | --- | --- |
| m1 | drop the `f32` on the neutral `beta` | 0 | 6 | 4 |
| m2 | drop the `f32` on the returned height | 0 | 14 | 4 |
| m3 | `epsrel` 1e-5 → 1e-4 | 0 | 11 | 3 |
| m4 | unfuse `F_A² + F_V²` in `dnde_pi_to_lnug` | 0 | **0** | **0** |
| m5 | unfuse `1 − β cosθ` in the integrand | 0 | 1 | 1 |
| m6 | `f_π` via `1.0/√2` instead of `M_SQRT1_2` | 1 | 0 | 0 |
| m7 | PDG `ENG_MU_PIRF` instead of the legacy literal | 0 | 10 | 5 |
| m8 | `poly_g -= 2.0` → `+= 2.0` | 2 | 18 | 5 |
| m9 | drop the `π → eνγ` channel | 0 | 17 | 5 |
| m10 | `epsabs` 1e-10 → 1e-8 | — | — | worst rel 0.9986 |
| m11 | muon integrand × (1 + 3.2e-11) | — | — | worst rel 3.199e-11 |

**Two survivors, both explained and both recorded in the source:** m4 (an
FMA site inside the quadrature — it leaves the worst corpus difference at
2.618e-15, the identical figure the correct port produces) and m6 (a
one-ulp constant, same mechanism). m6 is caught by `cargo`, m4 by nothing.
m10 and m11 were run as drift measurements rather than pass/fail, since
their point is *which budget* would see them.

### Numerical impact (rule 3, phase-file recipe step 8)

**Functions checked:** `hazma.spectra.dnde_photon_charged_pion` and
`hazma.spectra.dnde_photon_neutral_pion`, i.e. both swapped entry points,
plus the whole 41-case parity corpus for everything else.

**Grids:** (a) the corpus's own — five parent energies each
(`rest`, `rest_plus_eps`, `near_rest`, `boosted_mild`, `boosted_strong`),
1,500 pinned values for the charged pion and 1,305 for the neutral; (b) an
independent sweep of 1,000 energies (700 log-spaced over `[1e-4, 100 E_π]`
plus 300 uniform draws) at eight parent energies for the charged pion and
nine for the neutral, against the surviving Cython `cdef`s through
`__pyx_capi__`.

| Entry point | Corpus: values / not bit-equal / worst rel | Sweep: points / mismatches / worst rel |
| --- | --- | --- |
| `dnde_photon_charged_pion` | 1,500 / 317 / **2.618e-15** | 8,000 / 1,374 / **6.499e-15** |
| `dnde_photon_neutral_pion` | 1,305 / **0** / **0.000e+00** | 9,000 / **0** / **0.000e+00** |

Per block, charged pion: `rest` 54/268 at 3.540e-16, `rest_plus_eps`
60/308 at 2.981e-16, `near_rest` 59/308 at 6.735e-16, `boosted_mild`
76/308 at 2.618e-15, `boosted_strong` 68/308 at 3.434e-16.

**Is it intended?** Yes, and it is the whole reason the `QUAD` class
exists: `spectra.photon.charged_pion` moves from scipy's QUADPACK to the
in-tree port, which is a different implementation of the same algorithm,
so bit-equality was never available. The worst shift is **2.6e-15
relative — below rule 3's 1e-12 declaration threshold**, so this is
recorded rather than declared as a drift, and no `version_bump` level
changes. `spectra.photon.neutral_pion` moves nothing at all.

No other public value moved: the remaining 39 corpus cases are green at
their own budgets, and `test/test_theory_aggregation.py` (the model-layer
gate the corpus cannot be) is `69 passed` either side of the swap.

**Verified with:** `pytest -q test/parity`, the block-by-block script in
this task's Verification, and `pytest -q test/test_theory_aggregation.py`.

## Open Questions

- **Nothing gates the FMA map of a quadrature integrand.** Recorded above
  and in `rust/src/kernels/photon_pion.rs`. A `hazma._core` test-surface
  probe over `kernels::photon_pion` would fix it, but it would also widen
  `cases._CORE_TEST_ONLY_MODULES`, which Task 3.2 explicitly warned
  against doing to quiet a check. Left as a known limitation rather than
  built; Task 4.5 has the same shape one level deeper and is the right
  place to decide whether it is worth the machinery.
- **Does the forward-cone defect reach `_photon/_rho`?** Almost certainly
  — the ρ quadratures over this kernel — but it is unmeasured. Recorded on
  the follow-up; **Task 4.5 is positioned to answer it** and should, since
  the ρ's own quadrature may or may not compound the loss.
- **The Task 4.3 open question "does the rest-frame endpoint defect reach
  `_pion`?" is answered: no.** `_pion` evaluates `dnde_photon_muon` at the
  fixed `ENG_MU_PIRF = 109.78` MeV, which is 4.1 MeV above `m_μ` — never
  within one `DBL_EPSILON` of rest, so the truncated rest-frame branch is
  unreachable from here at any pion energy. `_rho` boosts *this* kernel,
  not the muon kernel directly, so the same argument covers Task 4.5.

## Plan Impact

**Impact Level:** Task note only.

No canonical contract moved. The phase file's Task 4.4 exit criteria are
met as written — including its "tighten the budget if warranted", which is
why `tolerances.py` changed rather than the phase file. `rules.md` needed
no amendment: rule 1 (reproduce, do not repair) governed the forward-cone
defect and rule 4 (bit-parity of the divergent constant tables) governed
`ENG_MU_PIRF`, both as written. No ADR: the two decisions this task made
(a per-case rather than class-wide budget tightening, and two oracle
standards in one test module) are implementation choices scoped to files
this task owns, and both are recorded in the files themselves.

Checked against the canonical text before writing this: the phase file's
Goal (the eight-step recipe and the capi-survivor exception), its Task 4.4
and Task 4.5 blocks, `PLAN.md`'s Numerical impact section, and all three
project ADRs. Nothing in them is now factually wrong.

## Stale-state sweep

Run against `claude/cython-to-rust/task-4.4-photon-pion` after the last
content edit. Every command's real output; every hit marked.

### Identifier sweep

```sh
rg -c 'dnde_photon_(charged|neutral)_pion' \
   projects/ docs/ hazma/ test/ rust/ setup.py -g '!*.c' -g '!*.so'
```

27 files. The ones this task had to reach:

| File | Hits | Status |
| --- | --- | --- |
| `hazma/spectra/_photon/__init__.py` | 12 | **EDITED** — both `return`s now call `_core_photon` |
| `hazma/spectra/_photon/_pion.pyx` | 6 | **EDITED** — only the `cdef`s remain (both `def`s gone) |
| `hazma/spectra/_photon/_pion.pxd` | 4 | KEPT — the four `cdef`s the cimporters need |
| `rust/src/photon.rs` | 6 | **EDITED** — two `#[pyfunction]`s + two registrations |
| `rust/src/kernels/photon_pion.rs` | 22 | **NEW** |
| `test/parity/cases.py` | 4 | **EDITED** — two `_SPECTRA` rows repointed, two `PORTED_ENTRY_POINTS` rows added |
| `test/test_core_photon_pion.py` | 35 | **NEW** |
| `docs/followups/todo/charged-…-forward-cone.md` | 1 | **NEW** |
| `hazma/spectra/__init__.py` | 6 | KEPT — public re-export, unchanged by design |
| `docs/source/{spectra,usage}.rst` | 6, 2 | KEPT — public names unchanged, so the docs are still right |
| `hazma/{scalar,vector}_mediator/*_decay_spectrum.pyx` | 6, 5 | KEPT — cimport the `cdef`s; see below |
| `hazma/spectra/_photon/_rho.pyx` | 5 | KEPT — Task 4.5 |
| `hazma/vector_mediator/_gev/spectra.py` etc. | 24, 2, 3, 2 | KEPT — pure-Python callers of the public name |
| `test/parity/data/manifest.json` | 2 | KEPT — the corpus keys by *case name*, not module |
| `references/cython-inventory.md` | 5 | KEPT — a declared 2.1.0 snapshot, as Tasks 4.1–4.3 left it |

**One prose correction the sweep forced.** `rg -n '_photon\._pion cimport' hazma/`
returns **three** dependants, not two:

```text
hazma/scalar_mediator/scalar_mediator_decay_spectrum.pyx:2-5  (all four cdefs)
hazma/vector_mediator/vector_mediator_decay_spectrum.pyx:10-12
hazma/spectra/_photon/_rho.pyx:10-11
```

The first drafts of `test/parity/cases.py`'s `PORTED_ENTRY_POINTS`
comment, `test/test_core_photon_pion.py`'s docstring and capsule test, and
the follow-up's §What all named only `_rho` and the *vector* mediator.
**EDITED** in all four places, and the capsule test now also runs a
`HiggsPortal.total_spectrum` beside the `VectorMediator` one.

### Line-number citation sweep

`scripts/agents/check_doc_citations.py <the five markdown files this task
touched>` — paths passed explicitly, because `--changed-vs` takes its file
list from committed history and would silently skip an uncommitted edit
(`docs/agents/lessons.md` `[gate-green-is-not-citations-green]`):

```text
docs scanned: 5
in-repo citations checked: 28
  resolved by exact: 24
  resolved by suffix: 4
external citations skipped: 4
out-of-range or ambiguous: NONE
```

An earlier run of it reported **three AMBIGUOUS** hits, all in this
block's own table below: three files in the tree are named `_pion.pyx`, so
a bare basename does not resolve. Rewritten to name the line numbers in
prose. Re-run after that edit, the output above is a fixed point.

`rg -on '_pion\.pyx:[0-9-]+' projects/ docs/ test/ rust/` — 52 hits. The
42 deleted lines were all at `127+`, so every citation at or below `:123`
is unmoved. Two were not:

| Citation | Status |
| --- | --- |
| `test/parity/tolerances.py:239`, citing lines 168-196 of the file | **EDITED** → `hazma/spectra/_photon/_pion.pyx:147-171` (the neutral-pion closed form, shifted up by the charged `def`'s 22 lines) |
| the new follow-up's three citations, at lines 86-107, 98 and 123 | **NEW**, all verified against the post-deletion file |
| `cases.py:{50,323,324,325}` at lines 16-18, `tolerances.py:229` and `test_core_quad.py:{237,448,458,475}` at 94-99 / 123 | KEPT — above the deletions, re-checked by hand |

(Written with the line numbers in prose rather than as `_pion.pyx:N`
citations of their own: three files in this tree are named `_pion.pyx`,
so `check_doc_citations.py` reports a bare basename as AMBIGUOUS — a
sweep block that cites other people's citations has to qualify them or
stop looking like one.)

### Forward-looking phrase sweep

`rg -n '(Task [0-9.]+ will|will be added|still pending|today: ?stub|Not started)'`
over the six files this task touched under `projects/`, `docs/`, `hazma/`,
`rust/` and `test/`: **5 hits, all correct live status** — Tasks 4.5/4.6
`Not started` in the phase table, Phases 05/06/07 `Not started` in the
project table. No stale prediction survives; the two that this task
falsified were **EDITED** (`test_core_dispatch.py`'s "Task 4.4 swaps that
one in turn", and the phase README's open question about the rest-frame
endpoint reaching `_pion`).

### Count sweep

| Claim location | Command | Actual | Status |
| --- | --- | --- | --- |
| note + phase README: 11 new `cargo` tests | `grep -c '#\[test\]' rust/src/kernels/photon_pion.rs` | `11` | OK |
| note + both READMEs: `cargo test` 120 | `cargo test --manifest-path rust/Cargo.toml --no-default-features` | `120 passed; 0 failed` | OK |
| note + both READMEs: 73 per-kernel tests | `pytest --collect-only -q test/test_core_photon_pion.py` | `73 tests collected` | OK |
| note: 19 FMA sites | `objdump -d hazma/spectra/_photon/_pion.cpython-312-darwin.so \| grep -cE 'fmadd\|fmsub\|fmla\|fnmadd\|fnmsub'` | `19` | OK |
| note + phase README: 42 lines deleted from `_pion.pyx` | `git diff origin/master --numstat -- hazma/spectra/_photon/_pion.pyx` | `0  42` | OK |
| note + both READMEs: bare suite | `pytest -q` (preflight's own run) | `1755 passed, 15 skipped, 8 warnings in 588.92s` | OK |
| note + both READMEs: collection +73 | `pytest --collect-only -q` on both trees | `1697 → 1770` | OK |
| note + both READMEs: parity 629/1 | `pytest -q test/parity` | `629 passed, 1 skipped` | OK |
| note: model-layer gate | `pytest -q test/test_theory_aggregation.py` | `69 passed` | OK |
| note: charged-pion corpus drift 2.618e-15 over 1,500 values, 317 not bit-equal | the block-by-block script in §Verification | `2.618e-15`, `317/1500` | OK |
| note: neutral-pion corpus drift 0 over 1,305 values | same | `0.000e+00`, `0/1305` | OK |
| note: sweep 6.499e-15 / 8,000 and 0 / 9,000 | the `__pyx_capi__` sweep script | as claimed | OK |
| note: forbidden tokens | `rg -c 'TODO\|FIXME\|breakpoint()\|import pdb'` over both new files | none | OK |
| sweep §Identifier: 27 files | the `rg -c` above, piped to `wc -l` | `27` | OK |
| sweep §Line-number: 52 `_pion.pyx` citations | `rg -on '_pion\.pyx:[0-9-]+' … \| wc -l` | `52` | OK |

### Numerical-impact statement

**Two public functions moved, one of them by exactly zero.**
`hazma.spectra.dnde_photon_neutral_pion` is bit-equal to its Cython twin
at all **1,305** corpus values and **9,000** independently sampled points
(0 mismatches). `hazma.spectra.dnde_photon_charged_pion` differs at
**317 of 1,500** corpus values by at most **2.618e-15** relative, and at
1,374 of 8,000 swept points by at most **6.499e-15** — intended, since the
entry point moves from scipy's QUADPACK to the in-tree port, and **below
rule 3's 1e-12 declaration threshold**, so recorded rather than declared.
No other public value moved: the remaining 39 corpus cases are green at
their own budgets and `test/test_theory_aggregation.py` is `69 passed`
either side of the swap. Verified with `pytest -q test/parity`, the
block-by-block script in §Verification, and
`pytest -q test/test_theory_aggregation.py`.

### Exit Criteria → test mapping

| Exit criterion | Satisfied by |
| --- | --- |
| Both entry points corpus-green within the quad budget | `pytest -q test/parity` → `629 passed, 1 skipped`; `spectra.photon.charged_pion` all five blocks at `PORTED_QUAD_RTOL`, `spectra.photon.neutral_pion` all five at `EXACT` |
| First real `qagp` consumer: record measured drift | §Verification "Numerical impact", the per-block table, and `test/test_core_photon_pion.py::TestChargedPionAgainstTheCythonTwin` |
| …and tighten the budget if warranted | `test/parity/tolerances.py`'s new `PORTED_QUAD_RTOL = 1e-12`, justified by mutation m11 |
| Recipe step 1 — FMA map from the shipped `.so` | `the_folded_constants_match_the_shipped_immediates` + the module docs' site-by-site list |
| Recipe step 2 — kernel in `kernels/<pyx name>.rs`, PyO3-free | `rust/src/kernels/photon_pion.rs` (imports no PyO3) |
| Recipe step 3 — registered via `map_unary` with the twin's wording | `TestDispatchWiring::test_the_rank_message_is_the_cython_assert_verbatim` |
| Recipe step 4 — wrapper repointed | `TestWrapperAndPublicApi::test_the_private_wrappers_return_the_core_kernels_values` |
| Recipe step 5 — corpus case repointed + `PORTED_ENTRY_POINTS` row | `test/parity/cases.py`; enforced by `assert_full_coverage` and `test_the_served_roster_is_exactly_the_ported_entry_points` |
| Recipe step 6 — twin's `def`s deleted, `.pyi` gone | `TestWrapperAndPublicApi::test_the_cython_module_no_longer_exports_a_python_entry_point`; `git diff --numstat` shows `0 42` |
| Recipe step 7 — `test/test_core_<kernel>.py` | `test/test_core_photon_pion.py`, 73 tests |
| Recipe step 8 — drift recorded | this note, `../README.md`, `../../task-notes/README.md` |

### Task-note self-consistency

`**Status:** Complete` in this note's header; the phase README's Tasks
table cell reads **Complete (2026-08-17)**; the project README's Phases
row names Task 4.4 among the done ones; the phase file's frontmatter
stays `status: In Progress` (Tasks 4.5–4.6 remain), which is correct.
Every file, function and identifier cited in §Files Changed, §Decisions
and §Findings appears in `git diff origin/master --stat` (17 files) or is
one of the four created files. `projects/README.md` is untouched — this is
not a closing PR.

## Handoff to Next Task

**Task 4.5 (`_photon/_rho`) is next**, and it is the phase's declared
numerical stress test: a quadrature whose integrand is this kernel's
quadrature, which is itself a quadrature over `_photon/_muon`. Five things
to carry across:

1. **Call the Rust directly.**
   `kernels::photon_pion::{dnde_photon_charged_pion,
   dnde_photon_neutral_pion}` are `pub` and PyO3-free. The neutral one is
   bit-equal to the `cdef` `_rho.pyx` cimports (9,000 points, 0
   mismatches); the charged one agrees to 6.5e-15 over 8,000 points, which
   is as close as two independent adaptive quadratures get.
2. **`_rho.pyx` is *not* a capi survivor** — nothing cimports it, so the
   whole file goes in the swap PR. Re-check with `rg` at execution time
   rather than trusting this sentence.
3. **Two things no gate can see** for a quadrature-backed kernel: a single
   unfused FMA inside the integrand, and a one-ulp constant that
   multiplies it. Both measured here. The ρ's integrand arithmetic is two
   integrations from anything observable, so read the disassembly and do
   not read a green suite as confirmation of the map. Whether to build a
   `hazma._core` probe for the kernel layer — and pay the
   `_CORE_TEST_ONLY_MODULES` widening Task 3.2 warned about — is a
   decision Task 4.5 is better placed to make than this task was.
4. **Do not inherit Task 4.3's `1/β` prediction unexamined.** It did not
   apply here (see Findings) — but the ρ *does* sweep the pion's own `β`,
   so it may apply there. Re-derive.
5. **`test/test_core_dispatch.py`'s spectra oracle is `_photon/_rho`**,
   which Task 4.5 deletes, and there is no fourth photon candidate. The
   class docstring names the two remaining options.

**And measure whether the forward-cone defect reaches the ρ.** The ρ
quadratures over the charged pion, so it almost certainly does; whether
the outer integral compounds or smears the loss is unmeasured, and the
follow-up records the question. `NESTED` (`rtol = 1e-6`) is the loosest
budget in `test/parity/tolerances.py` and Task 4.5 owns the same
tighten-once-measured decision this task made for `QUAD`.
