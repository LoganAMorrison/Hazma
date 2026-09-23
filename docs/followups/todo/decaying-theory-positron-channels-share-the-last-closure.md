# Every decaying-theory positron channel evaluates the last channel

- **Added:** 2026-09-20
- **Source:** PR #99 review (parity-pinned-defect-repair Task 10, Reviewer E)
- **Scope:** commit
- **Status:** open
- **Triggers / blockers:** none. The parity corpus does not pin any
  `TheoryDec` positron spectrum, so the repair moves no corpus value.

## Why

`TheoryDec.positron_spectrum_funcs` (`hazma/theory/__init__.py`, near
line 600) defines `dnde_wrapped` inside a `for fs, dnde in ...` loop, and
the closure reads `fs` and `dnde` when it is called rather than when it
is defined. Every wrapped function therefore evaluates the dictionary's
last channel. For `RHNeutrino` that channel is `ve vt vt`, which has no
positron, so every channel and the total return zeros.

Measured on trunk and on the Task 10 build alike, for
`RHNeutrino(500., 1e-3, "e")` at `E = np.geomspace(1, 250, 6)` MeV:
`total_positron_spectrum(E)` returns six zeros, while
`sum(bf[fs] * _positron_spectrum_funcs()[fs](E))` over open channels
returns `[4.75e-05, 4.64e-04, 3.41e-03, 7.64e-03, 6.37e-03, 0]` MeV⁻¹
(`e pi` alone has branching fraction 0.41). A single-channel model such
as `SingleChannelDec` is unaffected, because its only channel is also
its last.

## What

Bind the loop variables at definition time the way the sibling
annihilation wrappers in the same file already do, with a
`make_dnde_wrapped(fs)` factory, and read the spectrum function from the
dictionary inside it. Add a test in `test/test_theory_aggregation.py`
that a multi-channel `TheoryDec` (e.g. `RHNeutrino`) returns a positron
total equal to the branching-fraction-weighted channel sum, and that
each channel matches its unwrapped `_positron_spectrum_funcs` entry.

This changes published numbers (`RHNeutrino` positron spectra go from
zero to their physical value) and needs a `CHANGELOG.md` entry.

## Entry points

- `hazma/theory/__init__.py`: `TheoryDec.positron_spectrum_funcs`
- `hazma/theory/__init__.py`: the `make_dnde_wrapped` factories in the
  `TheoryAnn` spectrum wrappers, which show the correct pattern
- `hazma/rh_neutrino/_model.py`, `hazma/pbh.py`: the multi-channel
  `TheoryDec` subclasses
- `test/test_theory_aggregation.py`

## Risks / open questions

Anything downstream that consumed `RHNeutrino` positron spectra
(positron-derived CMB or limit calculations) has silently seen zero and
will change. Check `hazma/pbh.py` for whether its channels hit the same
path before stating the blast radius.
