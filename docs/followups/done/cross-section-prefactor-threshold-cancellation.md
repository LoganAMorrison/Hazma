# `cross_section_prefactor` loses accuracy near the 2-body threshold

- **Added:** 2026-08-04
- **Source:** cython-to-rust Task 0.3 (measured while repointing callers off
  the deleted Cython twin)
- **Scope:** cross-cutting
- **Status:** done — fixed by `hazma.utils.two_body_momentum`, the
  factored form with the heavier mass subtracted first; see
  "Resolution" below
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

## Resolution

Landed as `hazma.utils.two_body_momentum(cme, m1, m2)`, now the single
definition of the two-body momentum in the library. Callers repointed:
`cross_section_prefactor`, `hazma/phase_space/_two_body.py` (both
integration branches), `hazma/phase_space/_rambo.py`, and
`hazma/deprecated/rambo.py`.

Two things came out differently than the plan above anticipated.

**The plain factored form was not enough.** Written symmetrically as
`(cme - m1 - m2)(cme - m1 + m2)(cme + m1 - m2)(cme + m1 + m2)`, it still
loses digits when the masses are very unequal: `cme - m1` for the
*lighter* `m1` carries an `ulp(cme)` absolute error into a difference
that tends to zero. Subtracting the **heavier** mass first fixes it —
near threshold `cme <= 2 * max(m1, m2)`, which puts `cme - max(m1, m2)`
inside the Sterbenz range where the subtraction is exact, and its result
is then of order `min(m1, m2)`, so the second subtraction is exact too.
Relative error against an exact-rational (`fractions.Fraction`)
reference over 21 mass pairs from {e, μ, π⁰, π±, K±, p}:

| `cme / (m1+m2)` | Källén | factored | factored, heavier first |
| --- | --- | --- | --- |
| 1 (threshold) | 4.0e-02 | 4.3e-05 | 2.2e-16 |
| 1 + 1e-10 | 1.3e-04 | 4.3e-07 | 2.6e-16 |
| 1 + 1e-7 | 1.3e-07 | 4.3e-10 | 3.4e-16 |
| 1 + 1e-3 | 3.6e-12 | 3.8e-14 | 2.2e-16 |
| ≥ 1.1 | ≤2.0e-15 | ≤6.4e-16 | ≤2.6e-16 |

**The threshold is now resolved to the last bit**, which is a second
behavior change at the edge. For unequal masses `m1 + m2` rounds, so a
`cme` handed in as `m1 + m2` sits an ulp above or below the *true*
threshold. The new form returns a real momentum in the first case and
NaN in the second — correct in both, but the old form could not tell the
two apart because its own roundoff residue was larger than the distance
being resolved. Equal masses are unaffected: `m + m = 2m` is exact, so
`p` is exactly `0.0` and `cross_section_prefactor` returns `+inf`.

The `test/test_utils.py` tests named in "What" were rewritten as
promised: the cancellation test became
`test_cross_section_prefactor_diverges_at_threshold`, and the monotone
test's floor moved from `1e-4` to `1e-12` above threshold.

### Deliberately left alone

- **`hazma/_utils/kinematics.pxd::two_body_three_momentum`** carries the
  same Källén expression, but nothing `cimport`s it (only
  `two_body_energy` from that header is used). Dead code that
  cython-to-rust Phase 00 deletes; editing it would force a rebuild for
  no behavior change.
- **`hazma/_gamma_ray/gamma_ray_fsr.pyx::c_cross_section_prefactor`** —
  same expression, in the module cython-to-rust ADR-0003 removes.
- **The other ~25 `sqrt(kallen_lambda(...))` call sites** across
  `form_factors/`, `vector_mediator/`, `rh_neutrino/`,
  `scalar_mediator/`, `phase_space/_three_body.py`, and
  `phase_space/_utils.py`. These have the same conditioning defect at
  their own thresholds, but each is a distinct physics function whose
  published numbers would move, which is a much larger declared change
  than this follow-up scoped. Carried forward as
  [`kallen-under-sqrt-remaining-call-sites`](../todo/kallen-under-sqrt-remaining-call-sites.md).
