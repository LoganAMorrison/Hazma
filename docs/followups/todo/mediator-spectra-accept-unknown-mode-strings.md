# Mediator spectra return 0.0 for an unrecognised mode string

- **Added:** 2026-08-23
- **Source:** `projects/cython-to-rust/task-notes/phase-06/task-6.1-table-struct.md`
- **Scope:** cross-cutting (four public entry points, both mediator models)
- **Status:** open
- **Triggers / blockers:** none technically, but the repair is a
  behaviour change on a public API, so it wants the same
  major-version window as the cython-to-rust port itself. Reproduced
  under `projects/cython-to-rust/rules.md` rule 1 by Phase 06.

## Why

All four mediator-spectrum entry points take a mode selector — a `str`
for six of the seven and a `list[str]` for the seventh — and none of
them validates it. A caller who writes `"pi0g"` for `"pi0 g"`, or
`"e e g"` where the positron modules accept only `"e e"`, gets `0.0`
back rather than an exception. A spectrum that is identically zero is
easy to mistake for a closed channel, so the failure is silent at the
call site *and* plausible downstream.

The mechanism is uniform. Every integrand is a `cdef double` ending in
a chain of `if mode == ...: return ...` with no `else`
(`hazma/vector_mediator/vector_mediator_decay_spectrum.pyx:166-178`,
`hazma/scalar_mediator/scalar_mediator_positron_spec.pyx:150-161`, and
the two clones). A C function that falls off its end returns zero, so
an unknown mode integrates a zero integrand, and the enclosing
`__dnde_decay_*` adds a line term only for modes it recognises — so
`0.0` comes all the way back out. The list-valued entry point reaches
the same place by a different route: `scalar_mediator_decay_spectrum`
folds its list with `if "pi pi" in modes: bitflag += BITFLAG_PP` and an
unrecognised name simply sets no bit
(`hazma/scalar_mediator/scalar_mediator_decay_spectrum.pyx:253-266`).

## What

Raise `ValueError` naming the offending selector and listing the
accepted ones, at each entry point, once per call. The parsing already
happens in one place after cython-to-rust Task 6.1 —
`rust/src/kernels/mediator_tables.rs`'s `PhotonMode::parse`,
`PositronMode::parse` and `ScalarPhotonModes::from_names` all return
`Option`/skip precisely so the current `0.0` is reproducible — so the
change is at their call sites in `rust/src/{scalar,vector}_mediator.rs`
plus the accompanying tests, not in the parsers.

Two decisions the repair has to make explicitly:

- **the empty list.** `scalar_mediator_decay_spectrum(..., modes=[])`
  is a legitimate way to ask for nothing and today returns `0.0`. It
  should probably keep doing so, which means the list-valued entry
  point rejects *unrecognised names*, not an empty result.
- **whether the two positron modules and the two photon modules share
  one vocabulary.** They do not today: `"e e g"` is a photon mode and
  `"e e"` a positron one, and each set rejects the other's spelling. A
  single error message listing the wrong vocabulary would be worse
  than none.

## Entry points

- `hazma/scalar_mediator/scalar_mediator_decay_spectrum.pyx:253-266`
- `hazma/vector_mediator/vector_mediator_decay_spectrum.pyx:166-178`
- `hazma/scalar_mediator/scalar_mediator_positron_spec.pyx:150-161`
- `hazma/vector_mediator/vector_mediator_positron_spec.pyx:151-162`
- `rust/src/kernels/mediator_tables.rs` — the parsers, and the module
  docs' "Modes are parsed once" section
- `test/test_core_mediator_tables.py` —
  `test_the_rejected_set_is_what_the_cython_answers_with_zero` pins the
  current behaviour against the live Cython twins in two classes, and
  is what a repair has to rewrite
- Related project: `projects/cython-to-rust/`

## Risks / open questions

The parity corpus samples only valid modes
(`test/parity/cases.py:1147-1219`), so it neither pins the `0.0` nor
gates a change to it — but the two Cython twins that could serve as an
oracle are deleted by Phase 06 Tasks 6.2–6.4, so the pins in
`test/test_core_mediator_tables.py` become the only record of the
pre-repair behaviour. Do not delete them without replacing them.
