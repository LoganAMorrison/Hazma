# Follow-up: the parity corpus pins ill-conditioned points

**Date:** 2026-08-18
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-01-parity-corpus.md` (Task 1.3
exit criteria and the phase Exit Criteria), `../../rules.md` rules 1–3
(parity discipline)
**Related ADRs:** none — the decision is a testing contract inside
`test/parity/`, not a project invariant; see Plan Impact
**Depends On:** Phase 01 Task 1.3 (which measured the problem and filed
the follow-up)

## Objective

Close
[`docs/followups/done/parity-corpus-pins-ill-conditioned-points.md`](../../../../docs/followups/done/parity-corpus-pins-ill-conditioned-points.md):
decide what the corpus should store where the pre-port Cython's answer is
dominated by cancellation, implement it, and take the CI scoping that
Task 1.3 put in as a workaround back out.

Not a numbered plan task. Phase 01 closed on 2026-08-08 and this fixes an
output of it rather than completing it, so the note sits with the phase
whose artifact it repairs and the phase stays `Complete` — the same shape
the `interp-oracle-scoped-by-an-unsound-probe` follow-up took.

## Exit Criteria

Taken from the follow-up's "What" section:

- The corpus stops asserting values at cancellation-dominated points, by
  a rule that is stated and defensible rather than platform archaeology.
- The mechanism also fixes the **port** gate, not only the cross-platform
  symptom — Phases 05 swaps these four kernels and must not be gated by
  numbers no implementation reproduces.
- `EXACT_RTOL = 0.0` stops applying unchanged in budget mode: the class
  distinguishes "a different platform" (a fact) from "a different
  implementation" (a drift to declare under `../../rules.md` rule 3).
- `.github/workflows/ci.yml`'s `PARITY` env comes out, restoring the
  phase file's original intent that the gate be green on every matrix
  entry.
- `../../rules.md` rule 2 is respected: no reference array is
  regenerated, and nothing is derived from a tree where a kernel runs on
  Rust.

## Inputs Reviewed

- The follow-up file itself, and
  [`task-1.3-test-wiring.md`](task-1.3-test-wiring.md) §"Round 2: the
  grid fix exposed the real problem" (the magnitude table and the
  exemplar point).
- `../../phases/phase-01-parity-corpus.md` — Task 1.3 exit criteria and
  the phase Exit Criteria, both of which carry the scoping amendment.
- `../../rules.md` rules 1–3, and `../PLAN.md` "Numerical impact".
- `test/parity/{cases,generate,tolerances,test_parity}.py` and
  `test/parity/README.md`.
- `hazma/scalar_mediator/_c_scalar_mediator_cross_sections.pyx`, the four
  affected `cdef double __sigma_*` bodies.
- `docs/agents/lessons.md`, `docs/agents/doc-consistency.md`,
  `docs/agents/environment.md`.

## Findings

### 1. The follow-up's proposed detector does not work, and could not

The follow-up's option 1 says to "evaluate each in a perturbed arithmetic
(e.g. recompute with the inputs nudged by an ulp and compare) and drop,
or refuse to pin, any whose output is not stable". Measured at the
exemplar point — `sigma_xl_to_xl[closed_resonance.mu]`, grid index 217,
`e_cm = 599.99940000000004`:

```text
value at e_cm            -1.50408082e-02
value at nextafter(-)    -1.50408082e-02
value at nextafter(+)    -1.50408082e-02
max relative spread       1.617e-10
```

1.6e-10, against a stored value that is wrong by a factor of 2.4e4. The
reason is that the two things are not the same: an input perturbation
measures **conditioning**, and this point is perfectly well conditioned.
What is broken is the **stability of the algorithm**, which no
perturbation of the inputs can see. The same measurement kills the
cheaper variant of thresholding on the `atan` difference itself
(Finding 3).

### 2. The true function has no pole, and the corpus pins a fabricated one

`test/parity/reference.py` (new) evaluates the same closed forms at 60
decimal digits. Across the `e_cm = 2 mx` cluster of
`sigma_xl_to_xl[closed_resonance.mu]`:

```text
e_cm              corpus (macOS/arm64)   exact
595.380695738     -1.381932e-06          +6.775521e-07
599.9994          -1.504081e-02          +6.198557e-07
599.9999994       -1.504133e+01          +6.198489e-07
600.0             -inf                   +6.198489e-07  (removable 0/0)
600.0000006       +1.504133e+01          +6.198489e-07
600.0006          +1.504186e-02          +6.198421e-07
629.407008563     +6.524279e-07          +4.076274e-07
```

The function is smooth and positive through `e_cm = 2 mx`. The apparent
`1/(4 mx**2 - s)` pole, the sign change and the `-inf` are **all**
artifacts: the numerator's `atan` difference cancels to nothing, the
denominator's `4 mx**2 - s` is genuinely small, and dividing rounding
residue by a small number manufactures a pole where the physics has
none. Both terms vanish together at `s = 4 mx**2` — the whole log tail
collapses to `log(4) * 0` there — so the singularity is removable and
the limit is the 6.198489e-07 above.

Task 1.3's Linux value at index 217 (`+5.624212846110624e-07`) is not
right either; it is off by 10%. Both platforms were wrong, one much more
than the other.

### 3. The blast radius is four entry points, not six blocks

All four affected kernels have the identical construction — the same
prefactor `P` on both `atan`s:

| Kernel | `.pyx` lines | `atan` pair |
| --- | --- | --- |
| `__sigma_xl_to_xl` | 265–292 | 278, 281 |
| `__sigma_xpi_to_xpi` | 293–391 | 320, 341 |
| `__sigma_xpi0_to_xpi0` | 392–489 | 420, 441 |
| `__sigma_xg_to_xg` | 490–515 | 502, 505 |

`__sigma_xs_to_xs` is in the same family of elastic cross sections and
does **not** have the construction; it is unaffected.

Two regimes reach the cancellation, and between them they touch **every
one of the 15 blocks** those four cases have — not the six Task 1.3
counted:

- `e_cm -> 2 mx`: the two `atan` arguments become equal. Four grid points
  wide, at the `2 mx` anchors of every model point.
- `width_s -> 0`: both arguments exceed ~9e15 and each `atan` rounds to
  the double nearest `pi/2`. This is `closed_resonance`
  (`width_s = 3.7e-15`), where it spoils everything above `e_cm ~ 595`.
  Below `e_cm = sqrt(4 mx**2 - ms**2)` the second argument is negative
  and large, the two `atan`s sit at opposite ends, and their difference
  is ~`pi` — that half of the block is fine and stays pinned.

Task 1.3 counted six because `pytest` stops at the first failing array in
a block and because which points visibly blow up depends on the libm
code path. Measured directly (Verification): Linux/x86_64 under SSE2
disagrees on a different subset than CI's AVX2 x86_64 did, and
Linux/aarch64 reproduces macOS/arm64 bit-for-bit at the exemplar point.
The set of *visibly* broken points is not even stable across x86_64
variants, which is the sharpest possible statement of why platform
archaeology cannot define the mask.

### 4. The follow-up's sixth block fixed itself

`spectra.photon.eta[boosted_strong]` was one of Task 1.3's six. It is now
bit-identical on all three
platforms, because Task 4.2 replaced the Cython boost integral with a
Rust one that reproduces NumPy's summation order deterministically. It
needs no mask and gets none.

### 5. A second, unrelated portability defect: `atol = 0` at stored zeros

Not in the follow-up; found by re-measuring the whole corpus.
`spectra.positron.charged_pion` integrates the positron-muon spectrum
over `cos(theta)`, and at `E = m_e` the integrand vanishes only at the
endpoint. macOS's QUADPACK weighted sum lands on exactly `0.0`; Linux's
lands on 2.6e-13. `tolerances`'s "`atol` is 0.0 everywhere" section
argues that below-threshold regions return exactly zero so no floor is
needed — true of the closed-form kernels, false of the quadrature-backed
ones. With `atol = 0` a 2.6e-13 against a stored `0.0` is an *infinite*
relative error and the case's 1e-8 budget never gets to speak. Four
blocks, one point each:

```text
block                                     grid           corpus  Linux
spectra.positron.charged_pion[rest_plus_eps]  0.510998946  0.0  2.686e-14
spectra.positron.charged_pion[near_rest]      0.510998946  0.0  2.908e-14
spectra.positron.charged_pion[boosted_mild]   0.510998946  0.0  5.510e-14
spectra.positron.charged_pion[boosted_strong] 0.510998946  0.0  2.605e-13
```

### 6. The `EXACT` class's off-platform floor is set by the corpus's own grid

`sigma_xx_to_ss[closed_resonance]` moves by 5.6e-8 on Linux/x86_64 — an
`EXACT`-class case, at exactly the `2 mx` anchor. That is not a
cancellation defect; it is the corpus sampling `1 + 1e-9` times a
threshold, where a `sqrt(1 - 4 mx**2 / s)`-shaped factor amplifies a
last-ulp difference by `eps / (2 * 1e-9) = 1.1e-7`. The measurement lands
where the derivation says it should, which is what makes 1e-6 a derived
budget rather than a fitted one. `spectra.neutrino.muon[rest_plus_eps]`
(2.2e-11) is the same mechanism at the `parent_energy = mass * (1+1e-12)`
anchor.

## Decisions and Implementation Notes

### D1. Establish the mask against ground truth, not against a proxy

Findings 1 and 3 rule out both cheap detectors, and Finding 3 rules out
defining the mask by which platforms disagree. What is left is to know
the right answer, so `test/parity/reference.py` provides it: the four
`cdef double` bodies copied **verbatim** from the `.pyx` and evaluated in
`mpmath` at 60 digits. A `cdef double` body in that file is pure
Python-syntax arithmetic, so the copy needed no transcription — only the
`cdef` declarations dropped and the arguments promoted. That matters:
a re-derived expression would answer whether the *published formula* is
right, and the question here is whether the *evaluation* of it is.

Verbatim-copy risk is covered two ways: the first hand-written
transcription of `sigma_xl_to_xl`, written independently while
diagnosing, agrees with the extracted one to all 12 printed digits; and
2,522 of the 4,675 stored grid values reproduce the reference to zero
relative difference, which no mistranscribed expression does.

Phase 05 deletes the `.pyx`. `reference.py` is a standing copy with its
provenance (file, line ranges, kernel digest `f5e6e269be47`) in its
header, so it survives that deletion as evidence.

### D2. Threshold at 1e-9, because that is where the histogram's valley is

Binning all 4,675 stored grid values of the four cases by their
disagreement with the reference is bimodal:

```text
decade   1e-16  1e-15  1e-14  1e-13  1e-12  1e-11  1e-10  1e-9
points     663    149    191    245     80     19     11      4
decade    1e-8   1e-7   1e-6   1e-5   1e-4   1e-3   1e-2   1e-1  ...
points      23     35     30     42     66     75     73     52  ...
```

The left mode is accumulated rounding through a long expression; the
right mode is the cancellation, running out to 1e+24. 1e-9 is the
minimum between them and four points sit in that decade — the only ones
the choice could move either way. It is also seven decades looser than
`EXACT_RTOL` and three tighter than `NESTED_RTOL`, so nothing lands in
the mask that any declared budget could have carried.

### D3. Drop the masked positions rather than widen a tolerance over them

The follow-up's option 2 ("pin with a per-point tolerance") would need a
tolerance of ~1e+24 at the worst point, which asserts nothing while
looking like a budget. Deleting the positions from both arrays before
`assert_allclose` keeps the printed "max relative difference" line
meaningful, and `test_every_masked_index_addresses_a_real_stored_value`
refuses a block masked in full — a block that asserts nothing must leave
`cases.py`, not be emptied in place.

The mask is 494 positions of the corpus's ~180k values, all inside the
15 blocks of the four declared cases. That total is pinned by
`test_the_mask_is_a_small_fraction_of_the_corpus`, so regenerating it has
to show up in a diff and be argued for, which is what rule 2 asks of
anything that loosens the gate.

### D4. Relax `EXACT` on the platform axis only, keyed coarsely

`Provenance` gains `same_platform`. `effective_budget` returns
`PLATFORM_EXACT_RTOL` for an `EXACT`-class case only when that is false.
A port **on** the capturing platform is still held to bit-equality:
Tasks 4.1–4.5 each achieved it, so relaxing there would give up a gate
that is demonstrably reachable. Every other class is already ≥1e-13 and
needs no branch — `test_the_platform_branch_only_moves_the_exact_class`
holds that.

The first implementation compared the manifest's `platform` and
`machine` keys directly, and the stash-proof pass caught what that costs:
the capturing machine has moved from macOS 26.5.2 to 26.6.1 since
capture, so the whole-string comparison read an OS point release as a
platform change and took the corpus off bit-equality **on the host it
was captured on** — passing, six decades weaker, with nothing in the
output saying so. `tolerances._libm_identity` compares the OS family and
the CPU architecture instead, which is what actually selects an
implementation of `atan`. If a point release does move one, the corpus
fails loudly at `rtol = 0` and somebody looks, which is what the `EXACT`
class docstring already asks for. Pinned by
`test_an_os_point_release_is_not_a_platform_change`.

### D5. Floor the stored zeros per array, and twice as narrowly as first written

`tolerances.zero_floor` returns `ZERO_FLOOR_FRACTION` (1e-9) times a
scale drawn from the array itself, and the runner applies it *only*
where the stored value is exactly `0.0` — as a separate
`assert_array_less`, not as an `atol` on the main call. Passing it as
`atol` would also loosen every small non-zero value in the same array,
which is exactly the objection `tolerances`'s "`atol` is 0.0 everywhere"
section raises and which still stands.

The first version used **1e-9 of the array maximum, for every case**, and
self-review measured what that actually permits. Two things were wrong
with it and both are now closed:

- **It gave `EXACT`-class blocks a floor.** The worst was **10.42
  absolute**, on `cross_sections.scalar.sigma_xl_to_xl[closed_resonance.mu]`,
  whose non-zero values all sit near 4e9 — a port could have returned
  `10.0` in each of that block's 192 sub-threshold positions and passed.
  An `EXACT` kernel's zeros come from an explicit branch
  (`if e_cm < mx + ml: return 0.0`) that a port reproduces exactly, so
  the class gets no floor at all. Measured: no `EXACT` block needs one on
  any of the three platforms.
- **The maximum is the wrong scale for a block spanning nine decades.**
  `spectra.photon.long_kaon[rest_plus_eps]` peaks at 6.6e5 near its
  endpoint, so a fraction of the max is a floor set by the spike. The
  scale is the **median** non-zero magnitude instead.

Third trap, caught in the same pass: the class test must read the
**declared** budget, not `effective_budget`'s answer. Off the capturing
platform that answer is `PLATFORM_EXACT_RTOL`, so keying on it handed the
whole `EXACT` class a floor precisely where the class is already at its
most permissive — `sigma_xl_to_xl[closed_resonance.mu]` came back at
4.399 on the first attempt at the fix.

After all three: the four blocks that need a floor get 8.8e-13 to 4.4e-12
against intrusions of 2.7e-14 to 2.6e-13 (6x–80x headroom), and the
loosest floor anywhere in the corpus is 1.6e-6, on a
`mediator_spectra.vector.positron` block whose values there have a median
of 1.6e3.

### D6. Rule 2 is not engaged

Nothing here regenerates a reference array. The mask is derived from the
*stored* corpus and from `reference.py`, neither of which touches a live
kernel, so a tree with 13 Rust kernels served cannot skew it. The mask
records the manifest's kernel digest and
`test_the_mask_was_built_from_this_corpus` fails if the two drift.

## Files Changed

- `test/parity/reference.py` — **new.** Arbitrary-precision copies of the
  four cancellation-prone kernels, with provenance.
- `test/parity/stability.py` — **new.** `UNPINNABLE_RTOL`,
  `AFFECTED_CASES`, the mask reader, and `--regenerate`.
- `test/parity/data/unpinnable.json` — **new.** The mask: 494 positions
  in 12 blocks.
- `test/parity/tolerances.py` — `PLATFORM_EXACT_RTOL`,
  `ZERO_FLOOR_FRACTION`, `zero_floor()`, `_libm_identity()`,
  `Provenance.same_platform`, and the `effective_budget` platform
  branch.
- `AGENTS.md`, `docs/agents/preflight.md` — both enumerate what
  `pip install --group dev` pulls; `mpmath` added to each (review
  round 1).
- `test/parity/test_parity.py` — `_drop_unpinnable`, the split
  zero/non-zero comparison, and six new guards (four on the mask, two on
  the platform branch).
- `test/parity/README.md` — the two new files, the new commands, the two
  carve-outs, and the corrected CI paragraph.
- `.github/workflows/ci.yml` — the `PARITY` env removed.
- `pyproject.toml` — `mpmath` added to the `dev` group (regeneration
  only); the `addopts` comment's claim that only the macOS entry pays for
  the corpus corrected.
- `test/test_core_positron_muon.py`, `test/test_core_interp.py`,
  `test/test_core_boost.py` — six sentences that asserted CI passes
  `--ignore=test/parity` off the capturing platform, repointed to what
  actually scopes those modules now.
- `docs/agents/preflight.md`, `docs/agents/environment.md`,
  `docs/agents/lessons.md` — the same claim, in the three places an agent
  is most likely to read it as a live instruction.
- `projects/cython-to-rust/task-notes/README.md`,
  `task-notes/phase-01/README.md`, `task-notes/phase-04/README.md`,
  `learnings/phase-01-parity-corpus.md` — working memory and the phase
  distillation, including the Follow-on seed whose guess at the fix
  (re-siting the abscissae) was wrong.
- `projects/cython-to-rust/phases/phase-01-parity-corpus.md` — Task 1.3's
  exit criteria and the phase Exit Criteria, both of which carried the
  scoping amendment and its "restore this bullet" instruction.
- `docs/followups/done/parity-corpus-pins-ill-conditioned-points.md` →
  `docs/followups/done/`, plus the index row.

## Verification

### The gate, on three platforms

The corpus was captured on macOS/arm64. Two Linux platforms were built
from this worktree in containers (`python:3.12-slim-bookworm`, pinned to
the capture's `numpy==2.5.1`, `scipy==1.18.0`, `cython==3.2.9`, so the
libm is the only axis that moves) and run against the committed data.

```text
macOS 26.6.1 / arm64   pytest test/parity -q   637 passed, 1 skipped in 29.42s
Linux glibc / aarch64  pytest test/parity -q   637 passed, 1 skipped in 31.86s
Linux glibc / x86_64   pytest test/parity -q   637 passed, 1 skipped in 67.66s
```

x86_64 is the platform Task 1.3 measured at 70–75 failing blocks. The
macOS entry is in **budget** mode but with `same_platform` true, so its
19 `EXACT`-class cases still run at `rtol = 0` — the gate Phase 05 will
swap against is unchanged.

Full suite, same trees:

```text
macOS 26.6.1 / arm64   pytest -q   1806 passed, 15 skipped in 53.92s
Linux glibc / x86_64   pytest -q   1794 passed, 16 skipped, 16 errors in 126.10s
```

The 16 x86_64 errors are all `test/agents/test_resolve_phase.py`, which
runs `git rev-parse --show-toplevel` at import. The container's copy of
the tree is a `tar` extract with `.git` excluded, so that fails there and
only there; confirmed by `git rev-parse` returning
`fatal: not a git repository` inside the container. Unrelated to this
change, and CI checkouts are git repositories.

### The drift the fix had to cover

Every corpus block re-evaluated on each platform and diffed against the
stored arrays, before any of these changes:

| Platform | blocks > 1e-13 | worst | worst after the fix's mask |
| --- | --- | --- | --- |
| Linux/aarch64 | 12 of 623 | 5.9e-09 | unchanged (no masked point was in the worst set) |
| Linux/x86_64 | 16 of 143 non-mediator | ~1.0e+02 | 5.6e-08 |

The 5.6e-08 residue is `sigma_xx_to_ss[closed_resonance]` at the `2 mx`
anchor — Finding 6, and what sets `PLATFORM_EXACT_RTOL`. The
mediator-spectrum blocks were measured separately on x86_64 (480 blocks,
worst 9.4e-09) and sit four decades inside `NESTED_RTOL`.

### What the tests cover

- **623 corpus blocks** — grid, replayed raises, and values, on every
  platform, with the mask and the zero floor in force.
- **The mask's shape** — `test_only_the_declared_cases_are_masked` (no
  case masked without an `AFFECTED_CASES` row),
  `test_the_mask_was_built_from_this_corpus` (kernel digest agrees with
  the manifest), `test_every_masked_index_addresses_a_real_stored_value`
  (in range, no duplicates, never a whole block),
  `test_the_mask_is_a_small_fraction_of_the_corpus` (the 494 total
  pinned).
- **The platform branch** — `test_an_os_point_release_is_not_a_platform_change`
  (a macOS point release does not flip the mode; a different
  architecture or OS family does) and
  `test_the_platform_branch_only_moves_the_exact_class` (the set of
  cases whose budget changes off-platform is exactly the `EXACT` class,
  and it lands on `PLATFORM_EXACT_RTOL`).
- **The pre-existing guards** — budget coverage, budget justification,
  the served-kernel roster, and the capturing-tree mode report — all
  unchanged and still passing.

### Stash-proof

Each of the three changes was reverted in place and the corpus re-run, on
both Linux platforms (`pytest test/parity -q`; baseline
`637 passed, 1 skipped` on each):

| reverted | Linux/x86_64 | Linux/aarch64 |
| --- | --- | --- |
| `data/unpinnable.json` emptied | **13 failed** | **1 failed** |
| the `PLATFORM_EXACT_RTOL` branch in `effective_budget` | **55 failed** | **34 failed** |
| `stability.PORTABILITY_ZEROS` emptied | 635 passed | **4 failed** |

(The `PORTABILITY_ZEROS` row runs with the two registry-integrity tests
deselected, since emptying the registry is exactly what they forbid —
hence 635 rather than 637 on the platform where nothing fails.)

Two things worth keeping from this:

- **The zero floor is not exercised on x86_64 at all.** The QUADPACK sum
  at `E = m_e` lands on exactly `0.0` there as it does on macOS, and only
  aarch64 produces the 2.6e-13. Had the stash-proof pass run on one
  platform it would have reported the floor as dead code. CI's matrix is
  x86_64 and macOS, so nothing in CI covers it either — the containers
  are the only place this assertion is currently proven.
- **Each revert fails on both platforms except that one**, and the counts
  differ between them, which is the same "which points visibly break
  depends on the libm path" from Finding 3 showing up again.

### Preflight

```text
$ scripts/agents/preflight.sh --paths "hazma test" --md "<the 23 changed .md>"
PASS   black --check           hazma test
FAIL   isort --check-only      run `isort hazma test` and re-check
FAIL   ruff check              see output below
PASS   cargo fmt --check       rust/
PASS   cargo clippy            rust/
PASS   cargo test              rust/
PASS   pytest                  1808 passed, 15 skipped, 6 warnings
PASS   import hazma            version 2.1.0
PASS   markdownlint            <the 23 changed .md>
SKIP   version bump            not a closing PR (pass --closing)
PASS   forbidden tokens        none added
RESULT: FAIL — blocked commit.
```

The two red rows are the trunk condition
[`preflight-isort-ruff-red-on-trunk.md`](../../../../docs/followups/todo/preflight-isort-ruff-red-on-trunk.md)
tracks, and this task's delta against it is **zero**:

```text
$ git stash -u && ruff check hazma test | grep ^Found ; git stash pop
trunk:  Found 6135 errors.
branch: Found 6135 errors.

$ isort --check-only hazma test | grep -c ERROR
trunk: 97      branch: 97

$ ruff check <every .py this task touched>
All checks passed!
$ isort --check-only <the same set>
(clean)
```

Both red rows come entirely from files this task does not touch
(`hazma/experimental/`, `test/vector_mediator/herwig4dm/`, and the
untyped legacy suites). `test/parity/` itself is clean on trunk and clean
here, which is the only part this task could have regressed.

### Review round 1 (PR #71)

Four blocking findings; two accepted, one partially, one rejected. Detail
in the response tables on the PR. The two that changed behaviour:

- **The zero floor was applied far too broadly.** It floored every exact
  zero in every non-`EXACT` array — **66,840 positions across 605
  arrays** — to cover four measured ones, so
  `spectra.photon.long_kaon[rest_plus_eps]` accepted 1.69e-07 where the
  Cython returns exactly zero and a small below-threshold regression
  would have passed. D5 below records the two narrowings I had already
  made in self-review; both were about the *size* of the floor, and I
  never questioned its *scope*. `stability.PORTABILITY_ZEROS` now names
  the four positions, all at `E = m_e` in
  `spectra.positron.charged_pion`, and the other 66,836 are back to the
  exact-zero contract.

  Re-measured to build the allowlist rather than assuming the four I had
  already seen were all of them: the sweep over every array on
  Linux/aarch64 returns exactly those four, each immediately followed by
  the first non-zero stored value.
  `test_every_portability_zero_is_a_boundary_zero` pins that shape, so
  the registry cannot drift back into interior positions — deeper below
  threshold the integrand is identically zero at every quadrature node
  and the sum is exact on every platform, which is *why* the broad
  version bought nothing.

- **`pyproject.toml`'s `addopts` comment still said only macOS runs the
  corpus.** I had spotted this mid-task, said I would fix it, moved on to
  the `test/test_core_*.py` sweep, and never came back — and then the
  Stale-state sweep block below listed it among the sites I had EDITED.
  The miss is minor; **claiming a sweep hit I had not made is not**, and
  it is the reason the block exists. Fixed, along with two siblings the
  reviewer did not cite: `test/parity/README.md` still described the
  floor as a fraction of the array's *maximum* (stale since I changed it
  to the median in self-review), and seven other places described the
  floor's scope without the "four declared" qualifier.

### Deferred

- **`sigma_xg_to_xg[narrow_resonance]` etc. are masked, not fixed.** The
  Cython is wrong at those points, and the Rust port will be too if it
  transcribes the same expression. Stabilising the closed form — writing
  `atan(u) - atan(v)` as `atan((4 mx**2 - s) / (ms * width_s * (1 + u v)))`,
  which is exact and well conditioned — is a *physics* change to a
  published number, out of scope for a testing repair and out of scope
  for this project (`../PLAN.md`, "Out of scope: any physics change").
  Filed as
  [`docs/followups/todo/scalar-elastic-cross-sections-cancel-in-atan-difference.md`](../../../../docs/followups/todo/scalar-elastic-cross-sections-cancel-in-atan-difference.md).
- **Native x86_64 with AVX2.** The containers run under Rosetta, so
  glibc's ifunc dispatch picks the SSE2 `atan` rather than CI's AVX2 one.
  Finding 3 is the reason that does not undermine the result — the mask
  is defined against ground truth, not against a platform — but the CI
  run on this PR is the first native-AVX2 measurement and is the last
  outstanding check.

## Open Questions

- **Is the corpus wrong anywhere else?** `reference.py` covers four
  entry points. The other 37 have no high-precision twin, so their
  pinned values are trusted rather than verified. Nothing in the
  three-platform sweep suggests another case with this defect (the next
  worst is 5.6e-08, and it is explained), but "no platform disagreed" is
  weaker evidence than "the reference agrees" — that is Finding 3's own
  lesson turned on the rest of the corpus.
- **Should `closed_resonance` be respecified?** With `width_s = 3.7e-15`
  the model point is degenerate for these four kernels: 29% of the block
  is unpinnable and the rest is a formula being evaluated far outside
  where it is numerically meaningful. Kept as-is here because changing
  `cases.py` moves abscissae, which rule 2 would then make expensive.
  Recorded in the follow-up above.

## Plan Impact

**Impact Level:** Update phase file

`../../phases/phase-01-parity-corpus.md` carried the scoping in two
places, both written by Task 1.3 as explicit amendments with a
"restore this bullet when it does" instruction:

- Task 1.3's third exit criterion said CI and preflight run the same
  collection **on the capturing platform**, and that the Linux entries
  run `pytest --ignore=test/parity`.
- The phase Exit Criteria said "green on all matrix entries" is
  unreachable until the follow-up lands.

Both are patched to what is now true. The phase stays `status: Complete`
— it was complete; this repairs one of its artifacts.

No ADR. The decision is a testing contract inside `test/parity/`, and
`../../rules.md` rules 1–3 already govern it: rule 2's "widening a
tolerance budget requires a one-line justification in the tolerance file
and a note in the task note" is what `PLATFORM_EXACT_RTOL`,
`ZERO_FLOOR_FRACTION` and `UNPINNABLE_RTOL` each carry. `../PLAN.md`'s
"Numerical impact" section is unaffected: no public value moves.

## Stale-state sweep

Run against this branch, after every prose edit was frozen.

### Identifier sweep

Every name this change introduces, and where it is referenced. All KEPT —
each site is either the definition, a test of it, or prose written in this
task.

```text
PLATFORM_EXACT_RTOL       test/parity/{tolerances,test_parity,README}.py|md,
                          test/test_core_{boost,interp,positron_muon}.py,
                          .github/workflows/ci.yml, docs/agents/environment.md,
                          docs/followups/{README,done/parity-corpus-...}.md,
                          projects/cython-to-rust/{phases/phase-01-...,
                          task-notes/README,task-notes/phase-01/followup-...}.md
ZERO_FLOOR_FRACTION       test/parity/{tolerances,test_parity,README}, project docs
UNPINNABLE_RTOL           test/parity/stability.py, this note
zero_floor                same set as PLATFORM_EXACT_RTOL
_libm_identity            test/parity/{tolerances,test_parity}.py, project docs
same_platform             test/parity/{tolerances,test_parity}.py, project docs
unpinnable_indices        test/parity/{stability,test_parity}.py
AFFECTED_CASES            test/parity/{stability,test_parity}.py, both follow-ups
EXPECTED_MASKED_POSITIONS test/parity/test_parity.py
```

The reverse sweep — the identifier this change *removes* — is the `PARITY`
env, and it is the one that mattered:

```text
$ rg -n 'ignore=test/parity' --glob '!.venv' .
.github/workflows/ci.yml:132     # (historical note in the Run tests comment)  KEPT
docs/agents/lessons.md:348       # the PR #52 example, marked as history      KEPT
test/parity/README.md:36         # the same, marked as history                KEPT
+ 14 sites in per-task notes under projects/cython-to-rust/task-notes/       KEPT
  (records of what was true when written)
```

Everything that read as a **live** claim was EDITED: `docs/agents/preflight.md`
(2 sites), `docs/agents/environment.md` (3), `projects/cython-to-rust/`
working memory (`task-notes/README.md` 3, `phase-01/README.md` 2,
`phase-04/README.md` 2), `learnings/phase-01-parity-corpus.md` (2),
`phases/phase-01-parity-corpus.md` (2), `pyproject.toml`'s `addopts`
comment, and the three `test/test_core_*.py` module docstrings.

### Line-number citation sweep

```text
$ python scripts/agents/check_doc_citations.py <the 23 changed .md files>
docs scanned: 23
in-repo citations checked: 75
  resolved by exact: 60
  resolved by suffix: 14
  resolved by context: 1
external citations skipped: 8
out-of-range or ambiguous: NONE
```

`.pyx` line citations added by this task (`_c_scalar_mediator_cross_sections.pyx`
265-292 / 293-391 / 392-489 / 490-515, and the `atan` sites 278, 281, 320,
341, 420, 441, 502, 505) were checked by hand against the file, since the
checker skips `.pyx`:

```text
$ grep -n "atan(" hazma/scalar_mediator/_c_scalar_mediator_cross_sections.pyx \
    | awk -F: '$1>=265 && $1<=515'
278: 281: 320: 341: 420: 441: 502: 505:      all eight confirmed
```

### Forward-looking phrase sweep

```text
$ rg -n '(Task [0-9]+ will|will be added|still pending|today: ?stub)' \
    test/parity/ <this note> <the new follow-up>
no occurrences
```

### Count sweep

| Claim location | Command | Actual | Status |
| --- | --- | --- | --- |
| "494 positions masked" (this note, `test_parity`, `test/parity/README.md`, both follow-ups, working memory) | `python test/parity/stability.py` | `494 positions masked` | OK |
| "12 blocks carry a mask" | `python -c` over `unpinnable.json` | 12 | OK |
| "15 blocks / 4,675 grid values in `AFFECTED_CASES`" | `cases.build_cases()` | 15 blocks, 4,675 | OK |
| "623 blocks plus 15 guards" (`test/parity/README.md`) | `pytest test/parity --collect-only -q` | 638 collected, 623 of them `test_entry_point_matches_corpus` | OK |
| "637 passed, 1 skipped" (three platforms) | `pytest test/parity -q` | 637/1 on each | OK |
| "494 of the corpus's ~180k stored values" | value entries in `data/*.npz` | 181,191 value entries (0.27%) | OK |
| "four declared portability zeros" (`stability`, `tolerances`, `test/parity/README.md`, ci.yml, docs) | `sum(len(v) for v in PORTABILITY_ZEROS.values())` | 4 | OK |
| "66,840 stored zeros the first version floored" (review-round note, `stability`, `tolerances`) | pre-fix `zero_floor` sweep over every array | 66,840 across 605 arrays | OK |
| "41 cases / 1580 arrays" (unchanged) | `python test/parity/generate.py --check` | `corpus OK: 41 cases / 1580 arrays` | OK |
| "1,808 passed, 15 skipped" (bare suite) | `scripts/agents/preflight.sh` gate 7 | `1808 passed, 15 skipped` | OK |
| "worst kept 9.4e-10, best masked 1.9e-9" | `python test/parity/stability.py --regenerate` | same | OK |
| histogram decade counts (`stability.py`, this note) | the binning script in the scratchpad | same | OK |

### Numerical-impact statement

**No public value changes (verified: `python test/parity/generate.py --check`
→ `corpus OK: 41 cases / 1580 arrays match the manifest`, and
`pytest test/parity -q` → `637 passed, 1 skipped` on the capturing
platform with the `EXACT` class still at `rtol = 0` there).** Nothing
under `hazma/` is touched by this diff — `git diff origin/master --stat --
hazma/` is empty — so no function the public API reaches can have moved.
What changed is which stored values the corpus *compares*: 494 of 181,191
value entries (0.27%) are no longer asserted, all inside the four scalar
elastic cross sections, because they are rounding residue rather than
numbers any implementation reproduces.

### Exit Criteria → evidence

| Criterion | Evidence |
| --- | --- |
| corpus stops asserting cancellation-dominated points, by a stated rule | `test/parity/stability.py` + `data/unpinnable.json`; `test_only_the_declared_cases_are_masked`, `test_the_mask_was_built_from_this_corpus`, `test_every_masked_index_addresses_a_real_stored_value`, `test_the_mask_is_a_small_fraction_of_the_corpus` |
| the mechanism fixes the **port** gate, not just the platform symptom | the mask is defined against `reference.py`, not against platform disagreement (Finding 1, Finding 3); Phase 05 handoff says what it means for the swap |
| `EXACT_RTOL = 0.0` no longer applies unchanged in budget mode | `tolerances.PLATFORM_EXACT_RTOL` + `_libm_identity`; `test_an_os_point_release_is_not_a_platform_change`, `test_the_platform_branch_only_moves_the_exact_class`; stash-proof B (55 / 34 failures without it) |
| CI's `PARITY` env removed | `.github/workflows/ci.yml`; `637 passed, 1 skipped` on macOS/arm64, Linux/aarch64 and Linux/x86_64 |
| rules.md rule 2 respected | no `data/*.npz` in `git status`; `generate.py --check` passes against the unchanged manifest; the mask records the kernel digest and `test_the_mask_was_built_from_this_corpus` enforces it |

### Task-note self-consistency

```text
$ git status --short | awk '{print $1}' | sort | uniq -c
   5 ??     new files
   1 AM     the follow-up staged at its done/ path
   1 D      its todo/ path
  28 M      modified
```

`**Status:** Complete` matches the Exit-Criteria table above (every row
has evidence). Every file named in §Files Changed appears in
`git status --short`, and every identifier named in §Decisions and
§Findings appears in the identifier sweep. The one deliberate asymmetry:
§Deferred names
`docs/followups/todo/scalar-elastic-cross-sections-cancel-in-atan-difference.md`,
which is a **new** file rather than a modified one, and is listed as such.

## Handoff to Next Task

- **Phase 05 is the consumer.** It ports these four cross sections. Read
  `test/parity/stability.py`'s module docstring first: 494 pinned
  positions in those kernels assert nothing, and a Rust rewrite that
  reproduces the *formula* will disagree with the corpus there no matter
  how faithful it is. The mask is what stops that from reading as a
  regression.
- **A Phase 05 port that wants those points back** has one honest route:
  stabilise the closed form (the `atan` addition identity in Deferred
  above) and declare the moved numbers under rule 3. That is a physics
  change and needs its own decision — the follow-up is filed.
- **The corpus is now platform-portable and CI runs it everywhere.** If a
  future matrix entry goes red on `test/parity`, the question to ask
  first is which of the three carve-outs it lands in — the mask, the
  `EXACT` platform branch, or the zero floor — because each names what it
  covers and none of them is a catch-all.
- **`reference.py` is reusable.** It is the pattern for answering "is
  this pinned number real?" for any other case that comes under
  suspicion, and it needs no build — only the committed corpus.
