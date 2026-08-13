# `test_core_interp.py` still scopes its NumPy oracle with a probe

- **Added:** 2026-08-12
- **Source:** cython-to-rust — carved out of the `test/test_core_boost.py`
  probe removal (the same mechanism, the other module)
- **Scope:** cross-cutting
- **Status:** done — resolved 2026-08-12 by the rewrite of
  `test/test_core_interp.py` (see [Resolution](#resolution)).
- **Triggers / blockers:** none. Independent of the Phase 04–06 kernel
  swaps; `hazma._core.interp` is a foundation module those swaps consume
  but do not change.

## Why

*Written against `707b07c`; every line number below is that revision's.
The §Resolution at the end records what replaced them.*

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

The four `test/test_core_interp.py` line numbers below are **as of
`707b07c`**, the revision this follow-up was written against; the symbols
were removed by the resolution, so at any later revision those lines
point into unrelated code. Read them with
`git show 707b07c:test/test_core_interp.py`.

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

## Resolution

Done 2026-08-12. `test/test_core_interp.py` now runs **42 passed, 0
skipped on both platforms** — measured, not inferred: `pytest
test/test_core_interp.py -q` gives `42 passed` on macOS/arm64 and the
same `42 passed` inside a linux/amd64 container built from this tree
(Debian bookworm, glibc 2.36, CPython 3.12.13, NumPy 2.5.1). Before, that
second run was `24 passed, 9 skipped`.

All five steps of §What landed, plus one the measurement forced:

1. `numpy_contracts()` / `NUMPY_CONTRACTS` / `requires_a_contracting_numpy`
   are gone, replaced by `ON_THE_CAPTURING_PLATFORM` derived from
   `test/parity/data/manifest.json`'s `environment.machine`.
2. `assert_matches_numpy(got, want, context)` carries the two declared
   modes; `assert_within_the_off_platform_budget` is split out so the
   budget arm is reachable on the capturing platform too.
3. The divergence was **measured on linux/amd64**, not inferred. Over
   1,154,010 abscissae (the seven live photon tables plus 50 random
   grids), the port and that build's `np.interp` differ at 311,501
   points, by up to **4.0e-02 relative** and **2.2e-16 of the peak**, with
   no non-finite value on either side (0 of 2,308,020) and no
   disagreement anywhere about which abscissae return exactly zero. The
   per-table table is in the module docstring.
4. `OFF_PLATFORM_BUDGET = 1e-12`, scaled to the peak
   (`atol = BUDGET * peak`, `rtol = BUDGET`) — 4.6e3x the worst measured
   peak-relative disagreement. The risk note above guessed the figure
   might need to be *looser* than boost's `1e-10`; measured, it is two
   orders **tighter**, because peak scaling collapses the very
   cancellation population the pointwise reading blows up.
5. `TestOffPlatformBudget` guards both the budget (a `1e-8`-of-peak
   perturbation must be rejected) and the mode dispatch (one ulp:
   rejected on the capturing platform, tolerated off it), with the
   expected mode re-derived from `sys`/`platform` rather than read back
   out of the module.

`TestFusedArithmetic` got the treatment its boost counterpart got, and
more: the reference is not the `fma` site in isolation but
`interp_reference`, a full Python transcription of `rust/src/interp.rs`
(bisection, clamps, exact-node short circuit, NumPy's two-step NaN
rescue) parameterised by its multiply-add. The port is asserted
**bit-for-bit against `mul_add=fma` on every platform**, which makes it
the module's strongest off-platform gate — stronger than the budget,
which peak scaling makes blind to a defect confined to a small value.
The discrimination is direct rather than assumed: `np.interp` is
bit-equal to `mul_add=fma` on all seven tables on macOS/arm64 and
bit-equal to `mul_add=unfused` on all seven on linux/amd64.

**One extra fix the measurement forced.** `sweep_abscissae` drew its
20,000 interior points with `rng.uniform(lo, hi, n)`, which NumPy
computes as `lo + (hi - lo) * u` in C — and macOS/arm64 contracts *that*,
so the same seed drew a different sweep on each platform (eta: 1,567
fused/unfused disagreements on macOS against 1,571 on Linux). The new
`spread()` helper writes it as separate ufunc calls, where nothing can
contract; the underlying `rng.random` doubles were verified identical on
both platforms by hash. Without this every count recorded here would have
been a per-platform anecdote. The per-table seed also moved from
`hash(name)` — randomised per process unless `PYTHONHASHSEED` is set — to
`zlib.crc32`, so the sweep is reproducible run to run as well.

`rust/src/interp.rs`'s module docs carried the stale claim that an
unfused NumPy "would differ from this function by ≤1.1e-13 — an order
inside the 1e-12 budget `test/parity/tolerances.py` sets"; the pointwise
figure is 5.9e-05 on the live tables, and the corpus is skipped
off-platform anyway. Corrected in the same change (comments only — no
Rust statement moved).

### Verification

Commands and their real output, all from the worktree at this change:

| check | result |
| --- | --- |
| `pytest test/test_core_interp.py -q` (macOS/arm64) | `42 passed in 0.94s` |
| the same inside the linux/amd64 container | `42 passed in 2.61s` |
| `scripts/agents/preflight.sh --paths test/test_core_interp.py --md …` | `RESULT: PASS`, all eleven rows; `pytest 1444 passed, 14 skipped in 571.95s` |
| `cargo test --manifest-path rust/Cargo.toml --no-default-features` | `80 passed` |
| `black`/`isort`/`ruff` on `test/test_core_interp.py` | clean under the repo's strict config |

The 14th skip is `test/parity/test_parity.py` in budget rather than
bit-equality mode (`hazma._core serves 1 kernel(s)`) — Task 4.1's
positron-muon swap, present on master before this change and untouched
by it.

**Test validity, stash-proof.** A test that cannot fail proves nothing,
so `rust/src/interp.rs`'s `slope.mul_add(x - xp[j], fp[j])` was replaced
with `slope * (x - xp[j]) + fp[j]`, rebuilt with `pip install -e .`, and
the module re-run on both platforms:

- macOS/arm64: **15 failed, 27 passed** — all seven `TestAgainstNumpy`
  table cases, the random grid, and all seven `TestFusedArithmetic` cases.
- linux/amd64: **7 failed, 35 passed** — the seven
  `TestFusedArithmetic` cases *only*.

That second line is the whole point of this change, in both directions.
`TestAgainstNumpy` **passes the mutant** off-platform, because there
`np.interp` *is* the unfused form — so the budget genuinely cannot see
this defect, exactly as §Resolution claims, and the fused reference is
what catches it. And under the old module the off-platform count would
have been **0 failed** — every test that could have seen the mutant was
behind `requires_a_contracting_numpy`, which off macOS was always false.
The mutation was reverted and the tree rebuilt before the gate was run.

### Stale-state sweep

| command | result |
| --- | --- |
| `rg -n 'numpy_contracts\|NUMPY_CONTRACTS\|requires_a_contracting_numpy'` | no live occurrences; the survivors are this file's own §Why/§What, the module docstring's history paragraph, and two dated Task 3.4 records — all past-tense by construction |
| `rg -n 'test_matches_numpy_bit_for_bit\|test_the_rust_sides_with_numpy'` | only this file's §Why, describing what was replaced |
| `rg -n 'followups/todo/interp-oracle'` | no occurrences — both inbound links in `task-3.4-interp-boost.md` repointed to `done/` |
| `rg -n '1\.1e-13\|1,549'` | no live claim about this module; `test/parity/tolerances.py:43`'s 1.1e-13 is the unrelated `501 · 2^-52` derivation |
| `markdownlint` over the four touched `.md` | clean |
| **Numerical impact** | **none.** The diff is one test module, comment-only edits to `rust/src/interp.rs` (`git diff origin/master -- rust/src/interp.rs` filtered to non-`//!` lines is empty), and four docs. No public function changed; verified by `pytest` → `1444 passed, 14 skipped`, the same as master. |

**A sweep-size slip corrected in the Task 3.4 records.** Those notes gave
the sweep as "20,204 abscissae per table"; the sweep is
`20,000 + 3n + 4`, so it is **20,304** for the 100-row eta table and
**21,504** for the six 500-row tables. The number was wrong when written
rather than changed since — `task-3.4-interp-boost.md`'s own enumeration
("20,000 random interior points, every node, every node nudged ±1e-13,
and four out-of-range") does not sum to 20,204 for any table. Re-derived
from the live tree:

```text
eta            rows=100  sweep=20304
eta_prime      rows=500  sweep=21504
charged_kaon   rows=500  sweep=21504
long_kaon      rows=500  sweep=21504
short_kaon     rows=500  sweep=21504
omega          rows=500  sweep=21504
phi            rows=500  sweep=21504
```

Corrected in place at all three sites, each marked with the date and the
old value, following the precedent of `c316afc`
("docs(projects): re-derive the Task 4.1 verification counts"): an
arithmetic slip is an error to fix, not a superseded decision to preserve.
The companion `1,549` at `task-3.4-interp-boost.md:95` is annotated
rather than replaced — it was drawn with a `hash()`-seeded sweep and so
was never reproducible; the deterministic seed gives 1,571.
