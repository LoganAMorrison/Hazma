# The parity corpus pins ill-conditioned points, so six blocks gate nothing

- **Added:** 2026-08-07
- **Source:** `projects/cython-to-rust/` Phase 01 Task 1.3 — enabling the
  corpus in CI produced its first non-macOS run and measured this
  ([PR #52](https://github.com/LoganAMorrison/Hazma/pull/52), run
  31238785136)
- **Scope:** project (`cython-to-rust`, Phase 01; blocks the port gate
  from Phase 04 on)
- **Status:** open
- **Triggers / blockers:** ripens **before Phase 04**. Nothing blocks it.
  Phases 04-06 swap kernels against this corpus, so every affected block
  will produce a false failure the moment a Rust implementation lands,
  whatever platform it runs on.

## Why

The corpus stores what the pre-port Cython returned at each sampled
point. At most points that is a well-conditioned number and any faithful
reimplementation reproduces it to within the declared budget. At a
handful it is not: the kernel computes a difference of nearly-equal
quantities, the result is dominated by rounding, and what got pinned is
one platform's particular cancellation residue rather than a property of
the physics.

Task 1.3 measured this by accident — it wired the suite into CI, which
ran the corpus off macOS/arm64 for the first time. On Linux/glibc, 70-75
of the 626 blocks fail, consistently across Python 3.10-3.14, while
macOS passes. The failures are two unrelated populations:

| Max relative difference | Count (py3.11) | What it is |
| --- | --- | --- |
| ≤ 4.5 ulp | 35 | `libc.math` differs between glibc and macOS libm in the last bits. Real, benign, and absorbable by a derived budget. |
| 1e-15 – 1e-12 | 20 | the same, accumulated through longer expressions |
| 1e-12 – 1e-6 | 14 | the same, amplified by conditioning |
| **≈ 1.0** | **6** | **not absorbable — see below** |

The last six are the reason this file exists. The clearest is
`cross_sections.scalar.sigma_xl_to_xl[closed_resonance.mu]`, scalar
probe index 5:

```text
macOS/arm64 (what the corpus pinned): -1.504080817723100e-02
Linux/glibc  (same source, same commit): +5.624212846110624e-07
```

A sign flip and seven orders of magnitude, from identical Cython. No
tolerance absorbs that, and widening one until it does would make the
gate vacuous exactly where the numerics are most fragile. Five of the
six are `closed_resonance` blocks of scalar cross sections
(`sigma_xl_to_xl`, `sigma_xpi_to_xpi`, `sigma_xg_to_xg`,
`sigma_xpi0_to_xpi0`); the sixth is `spectra.photon.eta[boosted_strong]`.
This is the same region Phase 01's working memory already flags as
storing "123 negatives + 5 infinities" in `sigma_xl_to_xl` — recorded
there as "branch behavior", which understated it.

**The cross-platform failure is the symptom, not the problem.** These
points cannot gate the Rust port either. A faithful Rust reimplementation
with a different instruction order, different FMA contraction, or a
different `exp` will land somewhere else in the same cancellation region
and the corpus will call it a regression — or, worse, absorb a genuine
one under a budget widened to accommodate the noise. Six blocks of the
gate the whole port swaps against currently assert nothing meaningful.

Task 1.3 worked around the CI symptom by scoping the parity suite to the
capturing platform (`.github/workflows/ci.yml`, the `PARITY` env on the
`Run tests` step). That unblocked the wiring and is explicitly *not* a
fix for this.

## What

Decide what the corpus should store at cancellation-dominated points,
then implement it. Three shapes, roughly in increasing cost:

1. **Identify and exclude.** Detect the affected points at generation
   time — evaluate each in a perturbed arithmetic (e.g. recompute with
   the inputs nudged by an ulp and compare) and drop, or refuse to pin,
   any whose output is not stable. Smallest change; loses coverage at
   exactly the kinematic corners the phase file says to sample.
2. **Pin with a per-point tolerance.** Keep the points but store a
   stability estimate alongside each value, and have
   `test/parity/tolerances.py` consult it instead of applying one
   per-function budget across a whole block. Keeps coverage, makes the
   contract honest, and is the only option that also fixes the port gate
   for Phases 04-06.
3. **Per-platform corpora.** Capture on each supported platform. Rejected
   on sight here — it multiplies the 2.9 MiB payload, does nothing for
   the Rust-port case (which is a third "platform"), and
   `projects/cython-to-rust/rules.md` rule 2 would need every one of them
   regenerated from pre-port Cython.

Option 2 is the one that pays for itself; option 1 is the cheap
stopgap if Phase 04 is imminent.

Whichever lands, also revisit two things Task 1.3 left standing:

- **`EXACT_RTOL = 0.0` applies in budget mode too**
  (`tolerances.effective_budget` returns the declared budget off the
  capturing tree, and the declared budget for the EXACT class is zero).
  That is why 35 last-bit differences became failures rather than
  passes. `tolerances.provenance` already records `platform` and
  `machine` separately from the kernel digest, so the class can
  distinguish "a different platform" (a fact) from "a different
  implementation" (a drift to declare under rules.md rule 3).
- **The CI scoping should be removed** once the corpus is
  platform-robust, restoring the phase file's original intent that the
  gate be green on every matrix entry.

## Entry points

- `test/parity/cases.py` — the specification; grids and probe points.
- `test/parity/generate.py` — where a stability check would run.
- `test/parity/tolerances.py` — `EXACT_RTOL`, `effective_budget`,
  `abscissa_budget`, and the budget-class docstring.
- `.github/workflows/ci.yml` — the `PARITY` env on `Run tests`, to be
  reverted when this is fixed.
- `projects/cython-to-rust/phases/phase-01-parity-corpus.md` — Task 1.3's
  exit criteria and the phase Exit Criteria, both reworded for the
  scoping.
- `projects/cython-to-rust/task-notes/phase-01/task-1.3-test-wiring.md` —
  the full measurement, log greps, and per-magnitude breakdown.
- `projects/cython-to-rust/rules.md` — rules 1-3 (parity discipline)
  govern any budget change here.

## Risks / open questions

- **Does the instability indicate a bug in the kernels themselves?**
  A cross section that returns `-1.5e-02` where the physical answer is a
  small positive number is not merely ill-conditioned, it is wrong at
  that point. Phase 01 recorded the negatives as contract; that decision
  deserves re-examination alongside this, and it may belong with the
  `two_body_momentum` / Källén work in
  [`kallen-under-sqrt-remaining-call-sites.md`](kallen-under-sqrt-remaining-call-sites.md),
  which is the same class of catastrophic-cancellation defect.
- If the affected points are dropped rather than fixed, the port loses
  its only pinned evidence at precisely the kinematic edges
  `phase-01-parity-corpus.md` singles out as must-sample.
- Regenerating any part of the corpus is governed by rules.md rule 2 —
  pre-port Cython only, never from a tree where a kernel runs on Rust.
