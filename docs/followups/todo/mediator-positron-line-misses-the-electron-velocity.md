# The mediator positron line is low by the electron's rest-frame velocity

- **Added:** 2026-08-27
- **Source:** cython-to-rust Task 6.3 (`projects/cython-to-rust/task-notes/phase-06/task-6.3-positron-spectra.md`)
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** none. Independent of the port — the port
  reproduces the shipped value bit-for-bit and declares nothing here.
  Best sequenced with the other Group A spectrum-normalization items
  (`eta-prime-two-photon-line-missing-factor-two.md`,
  `neutrino-pion-electron-line-counted-twice.md`), which move published
  numbers the same way.

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
- Related follow-up: `docs/followups/todo/eta-prime-two-photon-line-missing-factor-two.md`
- Related follow-up: `docs/followups/todo/neutrino-pion-electron-line-counted-twice.md`
- Related project: `projects/cython-to-rust/` (Task 6.3)
