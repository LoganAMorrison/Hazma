# The φ photon spectrum omits its direct `φ → π⁰γ` line

- **Added:** 2026-09-07
- **Source:** `projects/parity-pinned-defect-repair` Task 6 — the risk
  bullet on
  [`phi-photon-lines-use-the-daughter-meson-energy.md`](phi-photon-lines-use-the-daughter-meson-energy.md)
  asked for this check before that repair landed, and the measurement
  settled it
  (`projects/parity-pinned-defect-repair/task-notes/task-6-phi-lines.md`)
- **Scope:** cross-cutting (a published spectrum is missing a feature and
  a yield; the repair moves parity-pinned values)
- **Status:** open
- **Triggers / blockers:** none. It is a declared numerical change like
  its siblings, and the declared-delta mechanism
  (`projects/parity-pinned-defect-repair/adrs/ADR-0001-corpus-repairs-are-declared-deltas.md`)
  makes it independent of every other repair. It is **not** part of that
  project's roster of ten, which `test/parity/deltas.py`'s `REPAIRS` holds
  as a closed set, so adding it means adding a roster entry — either as a
  new task in that project before it closes, or as its own change after.

## Why

Five tabulated photon spectra add one or more monochromatic lines on top
of a CSV continuum, for decay modes whose photon is *direct* rather than
a daughter's decay product. The ω does this for both of its modes:

```rust
rust/src/kernels/photon_tables.rs  OMEGA: lines = [
    (OMEGA_TO_PI0_A_ENERGY, BR_OMEGA_TO_PI0_A),
    (OMEGA_TO_ETA_A_ENERGY, BR_OMEGA_TO_ETA_A),
]
```

The φ has three such modes — `φ → ηγ`, `φ → η′γ` and `φ → π⁰γ` — and
carries lines for only the first two. `BR_PHI_TO_PI0_A = 1.32e-3` is
defined at `rust/src/constants.rs:359` and read by nothing; the pre-port
`constants.pxd` defined it and no `.pyx` read it either.

**The CSV columns show the direct photon is genuinely absent rather than
folded into the continuum.** Each `X → π⁰γ` mode contributes a `pi0_a`
column holding the π⁰'s own decay photons. Normalized by the mode's own
branching ratio, that column should integrate to `2 BR(π⁰ → γγ) = 1.976`
— two photons per π⁰ — if it holds the daughter's photons and nothing
else:

| Parent | `pi0_a` integral | `BR(X → π⁰γ)` | ratio | has a line? |
| --- | --- | --- | --- | --- |
| φ | 0.0026115 | 1.32e-3 | 1.978 | no |
| ω | 0.16275 | 8.34e-2 | 1.951 | yes |

Both land on `2 BR(π⁰ → γγ)`; the ω's 1.3% shortfall is its table's
low-energy truncation, and the φ's grid starts proportionally higher so
it loses less. The ω's direct photon is therefore not in its column
either — it comes from the line — and the φ, which has no line, is simply
missing that photon. Measured with
`numpy.trapezoid` over `hazma/spectra/_photon/data/{phi,omega}_photon.csv`.

## What

1. Add a third entry to `PHI`'s line list in
   `rust/src/kernels/photon_tables.rs`:
   `(photon_line_energy(pdg::MASS_PHI, pdg::MASS_PI0), pdg::BR_PHI_TO_PI0_A)`,
   which puts it at 500.795 MeV in the φ rest frame.
2. Declare the resulting corpus shift. Unlike B2 this **raises the
   yield**, by `BR(φ → π⁰γ) = 1.32e-3` photons per decay at every boost,
   so a magnitude check is meaningful here where B2 needed a position
   check. It reaches the same `spectra.photon.phi` arrays as A1 and B2,
   so the declaration extends the existing `A1+B2` composite rather than
   standing beside it (`projects/parity-pinned-defect-repair/rules.md`
   rule 7).
3. `CHANGELOG.md` entry with the magnitude, and a `minor` bump at least
   (`docs/versioning.md`).

Before implementing, re-check whether the same omission reaches any other
tabulated spectrum: the sweep behind the table above covered only the two
parents with a `pi0_a` column, and the general question is whether every
`X → Y γ` mode with a branching ratio in `rust/src/constants.rs` has a
line in `photon_tables.rs`.

## Entry points

- `rust/src/kernels/photon_tables.rs` — `PHI`'s line list, and `OMEGA`'s
  beside it as the worked example.
- `rust/src/constants.rs:359` — `BR_PHI_TO_PI0_A`, currently unread.
- `hazma/spectra/_photon/data/phi_photon.csv` — the `pi0_a` column that
  establishes what the line would add rather than duplicate.
- `test/test_core_photon_tables.py` — `SPECTRA["phi"]`'s line list, which
  is the independent statement of which lines the kernel carries.
- `test/parity/deltas.py` — `REPAIRS`, `DELTA_MODELS` and the `A1+B2`
  declaration this would extend.

## Risks / open questions

- **The evidence is an integral, and an integral can hide a shape.** The
  table above compares each `pi0_a` column's total against the π⁰'s two
  decay photons and finds no room for a direct photon, but a monochromatic
  line smeared into a tabulated column would move that total by exactly
  the amount attributed to truncation. Comparing the two columns' *shapes*
  — the φ's against the ω's, each normalized by its own branching ratio
  and rescaled to the parent's mass — would settle it independently, and
  is worth doing before the line is added rather than after.
- **The same omission may reach other modes.** `rust/src/constants.rs`
  defines branching ratios for several `X → Y γ` modes; only the ones with
  a line in `photon_tables.rs` contribute their direct photon. That sweep
  is item 3's "before implementing" note above and is the reason this is
  filed as cross-cutting rather than as a one-line fix.
