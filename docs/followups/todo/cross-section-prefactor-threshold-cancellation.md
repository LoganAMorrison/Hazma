# `cross_section_prefactor` loses accuracy near the 2-body threshold

- **Added:** 2026-08-04
- **Source:** cython-to-rust Task 0.3 (measured while repointing callers off
  the deleted Cython twin)
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** none — but see "Risks" on why this must be a
  *declared* numerical change, not a silent fix.

## Why

`hazma.utils.cross_section_prefactor` computes the incoming momentum as

```python
p = np.sqrt(kallen_lambda(cme**2, m1**2, m2**2)) / (2 * cme)
```

and `kallen_lambda(a, b, c) = a² + b² + c² − 2ab − 2ac − 2bc` cancels to
zero at `cme = m1 + m2`. Near threshold the terms are each of size `cme⁴`
while their sum tends to zero, so the result is dominated by roundoff.

The Cython twin deleted in Task 0.3
(`hazma/field_theory_helper_functions/common_functions.pyx:52-53`) used the
factored form instead, which has no such cancellation:

```text
p = sqrt((m1 - m2 - cme)(m1 + m2 - cme)(m1 - m2 + cme)(m1 + m2 + cme)) / (2 cme)
```

The two are algebraically identical; they differ only in conditioning.
Measured over 36 mass pairs drawn from {e, μ, π⁰, π±, K±, p}:

| `cme / (m1+m2)` | max relative difference (factored vs Källén) |
| --- | --- |
| 1 + 1e-7 | 2.060e-07 |
| 1 + 1e-6 | 1.187e-08 |
| 1 + 1e-5 | 3.701e-09 |
| 1 + 1e-4 | 1.621e-10 |
| 1 + 1e-3 | 1.107e-11 |
| 1 + 1e-2 | 1.985e-13 |
| ≥ 1.1 | ≤ 4.759e-15 |

At *exactly* threshold the residue is roundoff rather than zero, so the
function returns a large finite number instead of diverging — pinned by
`test/test_utils.py::test_cross_section_prefactor_threshold_cancellation`.

Away from threshold (`cme ≥ 1.1 ×` threshold) the two forms agree to
≤5e-15, i.e. roundoff, so this only matters for near-threshold work.

## What

Switch `cross_section_prefactor` (and, if the same pattern appears
elsewhere, other `kallen_lambda`-under-a-square-root call sites) to the
factored form, or give `kallen_lambda` a numerically stable variant for the
`λ(s, m1², m2²)` case.

Then update:

- `test/test_utils.py::test_cross_section_prefactor_threshold_cancellation`
  — it pins the *current* behavior deliberately and must be rewritten to
  assert the divergence once the fix lands.
- `test/test_utils.py::test_cross_section_prefactor_grows_toward_threshold`
  — its `1e-4` floor exists only because of this limitation and can be
  tightened.

## Entry points

- `hazma/utils.py` — `cross_section_prefactor` and `kallen_lambda`.
- `hazma/phase_space/_rambo.py:537` — live public caller.
- `hazma/deprecated/rambo.py:413-414,~1001` — deprecated public callers,
  repointed here in cython-to-rust Task 0.3.
- `hazma/gamma_ray.py:241` — caller in the broken-on-import module that
  cython-to-rust ADR-0003 removes.
- Prior art (deleted): `hazma/field_theory_helper_functions/common_functions.pyx`
  at `origin/master`, `git show <pre-Task-0.3-sha>:...`.

## Risks / open questions

- **This is a declared numerical change, not a cleanup.** It moves the
  values returned by `hazma.phase_space.PhaseSpace.cross_section`, a live
  public API. It needs a CHANGELOG line with a magnitude, per
  `projects/cython-to-rust/rules.md` parity rule 3, and — if it lands while
  cython-to-rust is still open — a line in that project's
  "Numerical impact so far".
- Sequencing: doing this *before* the Rust port means the parity corpus
  (Phase 01) captures the fixed values and the Rust side must reproduce
  them; doing it *after* means it is one clean, isolated change. After is
  simpler.
- The shift is far below the Monte-Carlo statistical error of every current
  caller (~1/√N, percent-level), so there is no urgency.
