# Two vector cross sections raise `TypeError` at `e_cm = 2 m_x`

- **Added:** 2026-08-20
- **Source:** cython-to-rust Task 5.1
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** none. Independent of the rest of the
  cython-to-rust port — the port reproduces the behavior under
  `rules.md` rule 1 and does not fix it.

## Why

`hazma.vector_mediator.KineticMixing(...).sigma_xx_to_v_to_pipi(e_cm)`
raises

```text
TypeError: Cannot convert 'complex' with non-zero imaginary component to
'double' ...
```

when `e_cm` is exactly `2 * mx`, and so does `sigma_xx_to_v_to_pi0v`.
The other four channels return a non-finite number there instead, and
they do not agree with each other about which: `sigma_xx_to_v_to_ff` and
`sigma_xx_to_v_to_pi0g` give `inf`, `sigma_xx_to_vv` gives `nan`. One
kinematic point, six entry points, three different answers, one of which
is an exception — and `2 m_x` is not an exotic argument. It is the
annihilation threshold, the first energy at which the process exists at
all, and a caller sweeping a grid that includes it gets an exception
rather than a spectrum.

The mechanism is a Cython artifact rather than physics. All six
expressions divide by `sqrt(e_cm**2 - 4*mx**2)`, which is exactly zero
there. Two of them also raise a kinematic factor to the power `1.5`, and
Cython 3's default `cpow` semantics make `double ** double` a *possibly
complex* operation — so it compiles the whole enclosing expression in
`double _Complex`, and a division by zero there goes through C99 Annex
G's recovery clause, which returns `(±inf, nan)`. Cython's
`__Pyx_SoftComplexToDouble` sees the non-zero imaginary part and raises.
The exponent, in other words, decides whether you get `inf` or an
exception.

The parity corpus pins the behavior in three blocks
(`test/parity/data/manifest.json`, `raises`), so it is reproduced rather
than fixed — see `test/parity/generate.py`'s `evaluate_block`. That
makes this a known, deliberate wart, not an unnoticed one; what it is
not yet is *decided*.

## What

Decide what the six channels should return at their common threshold and
make all six do it. The physics answer is not `inf`, `nan` or a raise:
the cross section diverges as `1/beta` there because the flux factor
does, and every sensible consumer either avoids the point or wants a
finite sentinel.

Options, cheapest first:

1. **Return `inf` uniformly.** Smallest change: guard the two complex
   channels so they never reach the complex division at a vanishing
   denominator. Matches four of the six already, and matches what a
   `1/beta` divergence means.
2. **Return `0.0` at `e_cm <= 2 m_x`**, i.e. move the threshold guard
   from `<` to `<=`. Defensible — the process has no phase space at
   threshold — and it is what makes a swept grid usable. It moves a
   published number at exactly one point per grid.
3. Leave the values and remove only the raise, so the two complex
   channels return `nan` like `sigma_xx_to_vv` does.

Any of the three is a behavior change and needs a `CHANGELOG.md` entry.
Option 2 also touches the scalar mediator's twelve cross sections, which
carry the same `<` guard, and should be scoped together with them.

The port has no `**` operator and no complex arithmetic, so whatever is
chosen is a two-line change per channel in
`rust/src/kernels/vector_xs.rs` plus the same in the scalar module once
Task 5.2 lands it. What is *not* cheap is the corpus: the pinned raises
and the pinned values at those indices would both have to move, and
`projects/cython-to-rust/rules.md` rule 2 forbids regenerating the
corpus from a ported tree. Expect to hand-edit three manifest entries
with the change recorded, or to take this after Phase 07 closes.

## Entry points

- `rust/src/kernels/vector_xs.rs` — `complex_quotient_real_denominator`
  and its two callers; `tests::the_complex_kernels_raise_at_the_dark_matter_threshold`
  is where the current behavior is pinned.
- `test/test_core_vector_xs.py` — `TestTheThresholdRaise`.
- `test/parity/data/manifest.json` — the three `raises` records, under
  `cross_sections.vector.sigma_xx_to_v_to_pipi` blocks 1 and 2 and
  `cross_sections.vector.sigma_xx_to_v_to_pi0v` block 2.
- `rust/src/kernels/scalar_xs.rs` — the twelve scalar guards option 2
  would also touch. Task 5.2 ported them on 2026-08-21; the scalar
  module raises nowhere, because its one complex expression puts the
  vanishing root in the numerator.
- Related project: `projects/cython-to-rust/`

## Risks / open questions

- Which option the maintainer wants is a physics-API question, not a
  numerics one. Option 2 is the friendliest and the most invasive.
- The scalar mediator has the same `<` guards and the same vanishing
  `sqrt`, and it *does* have a `** 1.5` — one, in
  `__sigma_xx_to_s_to_ff`, found by Task 5.2 on 2026-08-21. It still
  raises nowhere, but for a different reason than this entry first
  recorded: that expression's vanishing root sits in the **numerator**,
  so `__divdc3` never sees a zero denominator and the imaginary part
  stays zero. What its twelve channels return at `2 m_x` is a
  parameter-dependent mixture of `0.0` and `±inf` (occasionally `nan`),
  not the uniform `inf`/`nan` this entry first claimed — measured over
  four mediator points, eight-to-eleven finite, one-to-three `±inf`,
  never a raise. Fixing only the vector side would make the two models
  disagree at threshold, which is worse than the current state.
