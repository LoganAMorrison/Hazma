# The φ photon spectrum omits its direct `φ → π⁰γ` line

- **Added:** 2026-09-07
- **Source:** `projects/parity-pinned-defect-repair` Task 6 — the risk
  bullet on
  [`phi-photon-lines-use-the-daughter-meson-energy.md`](../done/phi-photon-lines-use-the-daughter-meson-energy.md)
  asked for this check before that repair landed, and the measurement
  settled it
  (`projects/parity-pinned-defect-repair/task-notes/task-6-phi-lines.md`)
- **Scope:** cross-cutting (a published spectrum is missing a feature and
  a yield; the repair moves parity-pinned values)
- **Status:** done. Repaired as parity roster entries `C4` (the φ) and
  `C5` (the η′, which the sweep below found carries the same omission),
  under
  [ADR-0003](../../adrs/ADR-0003-corpus-repairs-are-declared-deltas.md).
- **Triggers / blockers:** none. It is a declared numerical change like
  its siblings, and the declared-delta mechanism
  (`projects/parity-pinned-defect-repair/adrs/ADR-0001-corpus-repairs-are-declared-deltas.md`)
  makes it independent of every other repair. It is **not** part of that
  project's roster of ten, which `test/parity/deltas.py`'s `REPAIRS` holds
  as a closed set, so adding it means adding a roster entry — either as a
  new task in that project before it closes, or as its own change after.

> **Resolved.** `rust/src/kernels/photon_tables.rs` gives `PHI` a
> `φ → π⁰γ` line at 500.795 MeV and `ETA_PRIME` an `η′ → ρ⁰γ` line at
> 165.129 MeV and an `η′ → ωγ` line at 159.111 MeV. In flight, the φ
> gains `1.32e-3` photons per decay and the η′ gains `0.3202`; at rest
> both return the table alone and do not move. See
> "Resolution (measured)" below.

The sections from "Why" through "Risks / open questions" describe the
defect as it was filed, before the repair. In particular,
`BR_PHI_TO_PI0_A` is now read by `PHI`'s third line.

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

## Resolution (measured)

**The sweep.** The generator, `notebooks/decay_spectra/utils.py`, gives a
final-state photon no decay spectrum and no FSR, so no table column holds
a mode's direct photon. Every two-body `X → Y γ` mode therefore needs a
line. Over the seven tabulated parents, three such modes had a column
and no line:

| Mode | BR | column integral / BR | daughter's own yield on that grid |
| --- | --- | --- | --- |
| `φ → π⁰γ` | 1.32e-3 | 1.978 | 1.978 |
| `η′ → ρ⁰γ` | 29.5e-2 | 0.2155 | 0.2155 |
| `η′ → ωγ` | 2.52e-2 | 2.351 | 2.230 |

The daughter's yield is today's `dnde_photon_neutral_pion`,
`dnde_photon_neutral_rho` or `dnde_photon_omega` at the daughter's
two-body energy, integrated by trapezoid on the parent table's own grid.
The π⁰ and ρ⁰ agree to 2e-9 and 2e-7, so each column is exactly `BR`
times the daughter's boosted spectrum. That settles the risk bullet
above: the φ's column is not hiding a smeared line, because it is the π⁰
spectrum and nothing else. The ω column sits 0.12 photons above today's
ω kernel, so the table was generated from a different ω spectrum, but a
direct photon would add 1.0. `test/test_core_photon_tables.py`
(`test_no_table_column_carries_its_modes_direct_photon`) pins all three.

The omission was absent elsewhere. The ω's two modes and the φ's other
two have lines. The φ's `f₀(980)γ` and `a₀(980)γ` modes are commented
out of the generator and have neither a column nor a line. Three-body
radiative modes such as `η → π⁺π⁻γ` also lack their direct photon, but
that is a continuum rather than a line, and it is out of this scope.

**The ρ⁰ line is a modeling choice.** The ρ⁰ is 149 MeV wide, and
`BR(η′ → ρ⁰γ)` includes the non-resonant `π⁺π⁻γ` continuum, so its
photon is not truly monochromatic. The line sits at the ρ⁰ pole mass,
which is how the table's own `rho0_a` column treats the ρ⁰.

**Kernel.** Three `Line`s, with energies through `photon_line_energy`,
appended after the existing lines, so the kernel's fused multiply-adds
for the shipped lines are untouched. `every_line_energy_is_the_photons_not_the_daughters`
and `every_folded_constant_is_the_shipped_immediate_or_its_declared_repair`
pin the new energies: `0x407f_4cb8_6c25_0057`, `0x4064_a420_91c3_fbee`
and `0x4063_e389_d44f_de3c`.

**Yield.** The rest-frame table plus lines, integrated over each table's
grid, goes from 2.17466 to 2.17598 photons per φ decay (+0.061%) and
from 3.64032 to 3.96052 per η′ decay (+8.8%). Pointwise at twice the
parent mass, `dnde_photon_phi(500, 2038.922)` goes from 1.033088e-3 to
1.033849e-3 MeV⁻¹ (+0.074%) and `dnde_photon_eta_prime(160, 1915.56)`
from 8.451172e-3 to 9.012606e-3 MeV⁻¹ (+6.6%).

**Corpus.** The committed arrays are untouched. Both terms are closed
form, and `test/parity/deltas.py` declares them as `Additive` relations.

- **`C4`** raises 179 positions over five `spectra.photon.phi` arrays by
  4.4e-5 to 0.13 relative to the stored value. All five are already
  `A1+B2`, so they become `A1+B2+C4`. That keeps A1+B2's 1e-12 budget
  and its 1e-20 floor, and measures 4.3e-16 worst at the positions C4
  moves.
- **`C5`** raises 164 positions over six `spectra.photon.eta_prime`
  arrays by 1.7% to 56%. Five are already `A1+B1` and become
  `A1+B1+C5`, measured 3.3e-16 worst. The sixth, `near_rest.scalar_values`,
  was `A1` alone and becomes `A1+C5`, measured 0.0.
- Neither moves `rest`, where the kernel returns the table alone and
  adds no line, or `rest_plus_eps`, where the window is 2.8e-6 wide and
  no grid point falls inside it. No other corpus case moves.

**Release.** The change is recorded under `[Unreleased]` in
`CHANGELOG.md` as a numerical change, which `docs/versioning.md` classes
as `minor`. The version in `pyproject.toml` moves with the release that
ships it, as it did for the other fixes already under `[Unreleased]`.
