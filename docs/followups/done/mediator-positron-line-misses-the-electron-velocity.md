# The mediator positron line is low by the electron's rest-frame velocity

- **Added:** 2026-08-27
- **Source:** cython-to-rust Task 6.3 (`projects/cython-to-rust/task-notes/phase-06/task-6.3-positron-spectra.md`)
- **Scope:** cross-cutting
- **Status:** done. Repaired as parity roster entry `C1`, the first
  label issued under
  [ADR-0003](../../adrs/ADR-0003-corpus-repairs-are-declared-deltas.md).
- **Triggers / blockers:** none. It was sequenced beside two other
  spectrum-normalization items
  (`eta-prime-two-photon-line-missing-factor-two.md`,
  `neutrino-pion-electron-line-counted-twice.md`), both of which
  `projects/parity-pinned-defect-repair` repaired first.

> **Resolved.** `spectrum_point` in
> `rust/src/kernels/mediator_decay_positron.rs` divides the line by `r`,
> so the `e⁺e⁻` box integrates to `pw_ee` at every mass. All four
> mediator positron entry points move in every recognised mode string,
> because each of them adds the line; an unrecognised mode still returns
> `0.0`. The sibling `boost::boost_delta_function`
> already carried the factor: its height is `1 / (2 γ β k₀)`, and `k₀ =
> e₀ r` is the daughter's momentum. The two now agree. The measurement is
> under "Resolution (measured)" below.

## Why

`S/V → e⁺e⁻` is a two-body decay, so in the mediator rest frame the
positron is monochromatic at `m/2` with momentum `p* = (m/2)·r`, where

```text
r = sqrt(1 - 4 m_e² / m²)
```

is its velocity. Boosting with `β` spreads it into a flat box between
`E∓ = E(1 ∓ rβ)/2`, whose width is `E·r·β`. A box carrying one positron
per decay therefore has height `1/(E·r·β)`.

Both mediator positron modules used `1/(E·β)` — the `r` is in the box's
*edges* and missing from its *height*:

```text
hazma/scalar_mediator/scalar_mediator_positron_spec.pyx:197-203
hazma/vector_mediator/vector_mediator_positron_spec.pyx:198-204

    r       = sqrt(1.0 - 4.0 * me * me / (ms * ms))
    eplus   = eng_s * (1. + r * beta) / 2.0
    eminus  = eng_s * (1. - r * beta) / 2.0
    if eminus <= eng_p <= eplus:
        lines_contrib = pws[0] * 1. / (eng_s * beta)
```

So the line integrates to `pw_ee · r` rather than `pw_ee`. Measured by
trapezoid over the box on a 4,001-point grid at `m = 125` MeV,
`E = 200` MeV, `pw_ee = 0.31`:

```text
integral = 0.3099896385890334
pw_ee    = 0.31
pw_ee·r  = 0.3099896385890333
```

The deficit is `1 − r ≈ 2 (m_e/m)²`: **3.3e-5** at `m = 125` MeV and
**1.4e-6** at `m = 600` MeV, so it is invisible against the corpus
budgets and against any realistic measurement. It is recorded because it
is a *normalization*, and normalizations are the thing users compose:
`Theory.positron_spectra` weights this by a branching fraction and sums
it with continua that do carry their full count, so the error does not
cancel. It also diverges as `m → 2 m_e`, where `r → 0` and the true box
height is unbounded while the shipped one stays finite.

The port reproduces this exactly — `rules.md` rule 1 forbids a physics
change inside a swap — and pins it in
`test/test_core_mediator_positron.py::TestPhysics::test_the_electron_line_carries_its_own_positron_count`,
which asserts `pw_ee · r` and names this file.

## What

Divide the line term by `r`, in the one place it now lives:

```text
rust/src/kernels/mediator_decay_positron.rs — spectrum_point's
`lines_contrib = pws.get(0)? / (eng_m * beta)`
```

`r` is already computed two lines above it for the window edges, so the
change is one factor. Then:

- Flip the assertion in
  `test/test_core_mediator_positron.py::TestPhysics::test_the_electron_line_carries_its_own_positron_count`
  from `pw_ee · r` to `pw_ee`, and delete the pointer to this file.
- Re-measure the four `mediator_spectra.*.positron.*` corpus cases. Every
  pinned value inside a line window moves by `1/r − 1`, which is **above**
  the `PORTED_NESTED_RTOL = 1e-9` those cases now hold, so this needs a
  corpus re-capture or a declared exception — it is a deliberate
  behavior change, not drift.
- Record it in `CHANGELOG.md` as a numerical change with its magnitude,
  per `docs/versioning.md` (a moved published number is `minor`).

Check the sibling boost helpers in the same pass: `boost_delta_function`
in `hazma/_utils/boost.pyx` — which the positron pion kernel uses for its
own `π → e ν` line — takes the daughter mass and may or may not already
carry the factor. Whichever it does, the two should agree.

## Entry points

- `rust/src/kernels/mediator_decay_positron.rs` — `spectrum_point`, the
  `lines_contrib` assignment
- `test/test_core_mediator_positron.py` — `TestPhysics`
- `test/parity/tolerances.py` — the four
  `mediator_spectra.*.positron.*` budgets
- `hazma/_utils/boost.pyx` — `boost_delta_function`, the sibling to check
- Related follow-up: `docs/followups/done/eta-prime-two-photon-line-missing-factor-two.md`
- Related follow-up: `docs/followups/done/neutrino-pion-electron-line-counted-twice.md`
- Related project: `projects/cython-to-rust/` (Task 6.3)

## Resolution (measured)

**Kernel.** The line term is `pws[0] / (E β) / r`. Dividing by `r` last
makes the repaired line exactly the shipped one over `r`. That is what
lets the parity declaration reproduce it bit for bit. The term is skipped
when `pws[0] == 0`. Without that guard, a closed channel reads `0 / 0`
wherever the box has no width: at `r = 0`, exactly at `m = 2 m_e`, which
the division by `r` introduced, and at `β = 0`, at rest, which the shipped
kernel already had. Both returned `NaN` at `E_e = E/2` and now return
`0.0`. PR #103 review caught the first, and
`TestPhysics::test_a_closed_electron_channel_adds_no_line` pins both.
No corpus value moves, because every corpus block has an open `e⁺e⁻`
channel.

**Physics invariant.**
`test/test_core_mediator_positron.py::TestPhysics::test_the_electron_line_carries_its_own_positron_count`
now asserts that the box integrates to `pw_ee` to 1e-9, at `m = 125` MeV
and `E = 200` MeV. The shipped value was `pw_ee · r`, 3.3e-5 low. The
module's independent reference, `reference`, divides by `r` as well, and
its docstring names that as its one departure from the `.pyx`. The
kernel's own `a_dark_continuum_leaves_the_line_alone` pins the closed
form `pw_ee / (E r β)`.

**Corpus.** The committed arrays are untouched, and the moved positions
are declared in `test/parity/deltas.py` as `C1`. The model is an
`Additive` closed-form term, `shipped / r − shipped`, inside the kernel's
own window. The window edges use the kernel's fused `r β + 1`, reproduced
exactly through `Fraction` arithmetic, because the corpus grids anchor
points on those edges. Measured with
`deltas.DELTA_MODELS["C1"].relation.term` over every block of the four
cases:

| Mass | Line shift, `1/r − 1` |
| --- | --- |
| 250 MeV | 8.356e-06 |
| 550 MeV | 1.726e-06 |
| 900 MeV | 6.447e-07 |

- **Reach.** 8,912 positions in 192 arrays, 2,228 per case. They cover
  all four modes of the `rest_plus_eps`, `near_rest`, `boosted_mild` and
  `boosted_strong` blocks at every mass. 6,596 of them move by more than
  the cases' `PORTED_NESTED_RTOL = 1e-9`, and the rest sit under a
  continuum large enough to hide the shift. Nothing moves at `rest`,
  where the box is one point whose height is infinite before and after.
- **`C1` alone** covers 64 arrays: every `e_e` block, and `pi_pi` at
  250 MeV, where the pion channel is closed and the line is all the
  array holds. The prediction matches the repaired kernel bit for bit at
  all 3,000 positions it moves.
- **`A4+C1`** covers the other 128 arrays, which A4 already declared:
  `total` and `mu_mu` at every mass, and `pi_pi` at 550 and 900 MeV. The
  composition's worst residual is 3.55e-12 (scalar) and 6.40e-12
  (vector) relative, which are A4's own figures on those arrays.
- **Independent oracle.**
  `test/parity/test_delta_models.py::test_the_electron_line_loses_exactly_its_velocity`
  multiplies the model's corrected `e_e` box by `r` and requires the
  stored array. The support must match position for position, and the
  values match bit for bit at all 557 in-window positions per case. No
  kernel is evaluated.
- **Revert check.** Restoring the shipped height fails exactly the 192
  declared arrays in `pytest test/parity -k positron`.

`EXPECTED_DECLARED_ARRAYS` goes from 343 to 407, because 64 arrays are
newly declared and 128 A4 keys became composites.
