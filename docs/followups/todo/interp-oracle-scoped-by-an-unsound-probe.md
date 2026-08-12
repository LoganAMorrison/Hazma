# `test_core_interp.py` still scopes its NumPy oracle with a probe

- **Added:** 2026-08-12
- **Source:** cython-to-rust — carved out of the `test/test_core_boost.py`
  probe removal (the same mechanism, the other module)
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** none. Independent of the Phase 04–06 kernel
  swaps; `hazma._core.interp` is a foundation module those swaps consume
  but do not change.

## Why

`test/test_core_interp.py` decides whether to compare `hazma._core.interp`
against `np.interp` bit-for-bit by *measuring* whether the installed NumPy
fuses its interpolation step — `numpy_contracts()` at
`test/test_core_interp.py:81`, cached in `NUMPY_CONTRACTS` and applied as
`requires_a_contracting_numpy` to `TestAgainstNumpy` and
`TestFusedArithmetic`.

That is the mechanism cython-to-rust Task 4.1 retired in
`test/test_core_positron_muon.py` and the 2026-08-12 rewrite retired in
`test/test_core_boost.py`, for a reason that applies unchanged here: a
probe tests **one** contraction mechanism, so it is blind to every other
way two builds can disagree. It fails in both directions — claiming
bit-equality on a build that diverges for a reason it cannot see (PR #63,
runs 31562223329 and 31564747071), and voiding the comparison outright
when its own mechanism is absent.

The interp module is resolving the second way, and it is measurable:
building this worktree for linux/amd64 (Debian, gcc, glibc, CPython
3.12.13, NumPy 2.5.1) and running the module gives **24 passed, 9
skipped** — every comparison against `np.interp` is skipped. Those nine
are the whole of the module's cross-implementation gate: seven
parametrised `test_matches_numpy_bit_for_bit` cases over the live photon
tables, `test_matches_numpy_on_a_random_grid`, and
`test_the_rust_sides_with_numpy_where_the_forms_differ`. On every CI entry
but macOS, `hazma._core.interp` is checked against nothing but its own
clamping contract, quirks, and error paths.

This is latent rather than breaking — the module is green — which is
exactly why it needs tracking: nothing will surface it.

## What

Apply the shape `test/test_core_boost.py` and
`test/test_core_positron_muon.py` now share:

1. Replace `numpy_contracts()` / `NUMPY_CONTRACTS` /
   `requires_a_contracting_numpy` with `ON_THE_CAPTURING_PLATFORM`,
   derived from `test/parity/data/manifest.json`'s
   `environment.machine`, so this module's scope cannot drift from
   `test/parity`'s.
2. Give the comparison two declared modes — bit-equality on the capturing
   platform, a budget elsewhere — behind an
   `assert_matches_numpy(got, want, context)` helper.
3. **Measure the off-platform divergence before choosing the budget.**
   Build the tree for linux/amd64 and compare directly; do not infer the
   figure from a local fused-versus-unfused proxy, which is the reasoning
   PR #63 refuted.
4. Scale the budget to the **peak** of the compared array
   (`atol = BUDGET * peak`, `rtol = BUDGET`), not pointwise. This module
   is the sharpest case for that: Task 3.4 rejected a tolerance precisely
   because the worst relative gap sits at the eta tail, where the
   interpolant is 2.4e-26 against a table of scale 0.2 — an absolute gap
   of 1.4e-30, invisible against the peak and catastrophic as an `rtol`.
   Peak scaling is what makes a budget viable where a pointwise one is
   not.
5. Add a guard that the budget still rejects a real error, since on the
   capturing platform nothing else exercises the tolerance branch.

`TestFusedArithmetic` deserves the treatment its boost counterpart got
rather than a budget: rewrite it to discriminate against a **fused Python
reference** (`fma` at the interpolation step, via the `Fraction`-based
helper — `math.fma` needs 3.13 and the suite supports 3.10) instead of
against NumPy. `f64::mul_add` is correctly rounded on every target Rust
supports, so "the port fuses here" becomes a platform-independent claim
and the class stops needing a scope at all.

## Entry points

- `test/test_core_interp.py:81` — `numpy_contracts()`
- `test/test_core_interp.py:100` — `NUMPY_CONTRACTS`
- `test/test_core_interp.py:102` — `requires_a_contracting_numpy`
- `test/test_core_interp.py:134`, `:177` — the two scoped classes
- `test/test_core_boost.py` — the pattern to copy, including the
  measured budget, the ulp budget for support edges, and
  `TestOffPlatformBudgets`
- `test/test_core_positron_muon.py` — the per-kernel template it came from
- `projects/cython-to-rust/task-notes/phase-04/task-4.1-positron-muon.md`
  §Findings — why the probe is unsound
- `docs/agents/lessons.md` `[platform-scoped-oracle-asserted-globally]`

## Risks / open questions

- **`np.interp` is a weaker oracle than a Cython twin in one respect:** it
  is not the implementation being replaced, it is an independent one, so
  the budget is a statement about two independent implementations rather
  than about one compiler's contraction choices. The figure may need to be
  looser than boost's `1e-10`; measure it rather than reusing that number.
- **The quirk tests must stay exact.** NaN propagation, the one-point
  grid's asymmetry, the duplicate-node tie-break and both infinite-cell
  rescues are structural, not numerical, and already run everywhere; a
  budget must not be extended over them.
