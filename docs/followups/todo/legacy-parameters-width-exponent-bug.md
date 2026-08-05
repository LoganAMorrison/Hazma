# `WIDTH_K` / `WIDTH_PI` exponent bug in the legacy constants tables

- **Added:** 2026-08-04
- **Source:** `projects/cython-to-rust/references/cython-inventory.md`
  "Bugs" §3 (original observation); surfaced again and given a durable
  home by
  `projects/cython-to-rust/task-notes/phase-00/task-0.1-relocate-constants.md`
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** ripens when the constants tables are
  consolidated — the consolidation that `projects/cython-to-rust/rules.md`
  ("Constants" rule 1) defers as a separate, declared numerical change.
  Nothing blocks it technically; it is deferred because it must not be
  smuggled into a mechanical file move.

## Why

Three copies of the legacy Standard-Model constants table define the
charged-kaon and charged-pion widths with `**` where a decimal exponent
was meant:

```cython
cdef double WIDTH_K = 3.3406**-13.
cdef double WIDTH_PI = 2.528511206475808**-14.
```

`3.3406**-13.` is exponentiation, not scientific notation: it evaluates
to ≈ 1.8e-7 rather than a width of order 1e-13 MeV. Likewise
`2.528511206475808**-14.` ≈ 4.4e-6 instead of ≈ 2.53e-14 MeV. That
`WIDTH_PI` was meant to read `2.528511206475808e-14` is confirmed by
`hazma/_utils/constants.pxd:321`, which carries the same quantity
correctly as `WIDTH_PI = 2.5284e-14` (Γ[π+] = 2.5284e-14 ± 5e-18 MeV).

**No value is wrong in any published output today, because nothing reads
these two names.** A repo-wide search finds only the six definition
lines and no consumer (see Entry points), independently confirming the
"~10⁶ off; both currently unused" note the cython-to-rust audit already
recorded in `references/cython-inventory.md` "Bugs" §3.

This gets a follow-up of its own — rather than living only in that
audit — because the audit reference is a project-scoped snapshot that
retires when cython-to-rust closes, whereas
`hazma/_utils/legacy_parameters.pxd` survives the whole migration and
carries the defect forward.

Note the two names are not merely mistyped but also inconsistent between
tables: the legacy `WIDTH_K` mantissa is `3.3406`, whereas
`constants.pxd:324` records Γ[K+] = 5.317e-14 MeV. Repairing the
exponent alone would still leave a wrong kaon width; the correct value
has to be sourced from the PDG, not inferred from the typo.

## What

As part of (or immediately after) constants consolidation:

1. Decide the canonical source for both widths — `constants.pxd`'s
   PDG-cited values are the obvious candidate.
2. Delete `WIDTH_K` / `WIDTH_PI` from the legacy tables rather than
   repairing them in place, if consolidation makes the legacy tables
   redundant. If any legacy table survives, fix the literals and cite
   the PDG value inline, matching `constants.pxd`'s comment style.
3. Confirm the "no consumer" claim still holds at that time; if a
   consumer has appeared, the fix becomes a declared numerical change
   and needs a CHANGELOG line and a magnitude, per
   `projects/cython-to-rust/rules.md` parity rule 3.

## Entry points

- `hazma/_utils/legacy_parameters.pxd:63-64` — relocated in
  cython-to-rust Task 0.1; values kept verbatim for bit-parity.
- ~~`hazma/_decay/common.pxd:75-76`~~ and
  ~~`hazma/_positron/parameters.pxd:56-57`~~ — the second and third copies,
  **deleted** in cython-to-rust Task 0.3. `legacy_parameters.pxd` is now
  the only surviving copy of the bad literals.
- `hazma/_utils/constants.pxd:321,324` — the correct, PDG-cited values.
- Related project: `projects/cython-to-rust/` — see `rules.md`
  ("Constants" rule 1) and `references/cython-inventory.md` ("Bugs" §3)
  for the wider constants-divergence picture.

## Risks / open questions

- Task 0.3 has landed, so only `legacy_parameters.pxd` and
  `constants.pxd` carry these constants now; Task 6.4 retires
  `legacy_parameters.pxd` itself. The consolidation surface is already
  down to two files.
- The correct Γ[K+] must come from a PDG citation; do not back it out of
  the `3.3406` mantissa, whose provenance is unknown.
