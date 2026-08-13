# The φ spectrum places its two photon lines at the daughter meson's energy

- **Added:** 2026-08-12
- **Source:** `projects/cython-to-rust/` Phase 04 Task 4.2 — porting the
  five tabulated photon `.pyx` files to Rust; writing the ω's and the φ's
  line energies as one parameterised list is what put `(M² − m²)` and
  `(M² + m²)` side by side
  (`projects/cython-to-rust/task-notes/phase-04/task-4.2-photon-table-family.md`)
- **Scope:** cross-cutting (a published spectrum has a feature in the
  wrong place; the repair is gated by the `cython-to-rust` corpus)
- **Status:** open
- **Triggers / blockers:** **blocked until after `cython-to-rust` Phase 06
  Task 6.4**, for the same reason as
  [`eta-prime-two-photon-line-missing-factor-two.md`](eta-prime-two-photon-line-missing-factor-two.md):
  the parity corpus pins the shipped values by construction
  (`projects/cython-to-rust/rules.md` rule 2), so a repair landed during
  the port would fail the gate that governs every remaining swap.
  Cheapest to fix in one declared regeneration together with its
  siblings.

## Why

In a two-body decay `X → Y γ` the photon carries `(M² − m²)/(2M)` and
the meson `(M² + m²)/(2M)`; the two sum to `M`. The ω kernel uses the
first, correctly:

```text
hazma/spectra/_photon/_omega.pyx:111  eng_a_w_to_pi0_a = (MASS_OMEGA**2 - MASS_PI0**2) / (2 * MASS_OMEGA)
hazma/spectra/_photon/_omega.pyx:112  eng_a_w_to_eta_a = (MASS_OMEGA**2 - MASS_ETA**2) / (2 * MASS_OMEGA)
```

The φ kernel uses the second, for both of its lines, and then feeds it to
`boost_delta_function(..., m = 0.0, ...)` — i.e. as a *photon* energy:

```text
hazma/spectra/_photon/_phi.pyx:111  eng_eta = (MASS_PHI**2 + MASS_ETA**2)  / (2 * MASS_PHI)
hazma/spectra/_photon/_phi.pyx:112  res += BR_PHI_TO_ETA_A  * boost_delta_function(eng_eta, photon_energy, 0.0, beta)
hazma/spectra/_photon/_phi.pyx:113  eng_eta = (MASS_PHI**2 + MASS_ETAP**2) / (2 * MASS_PHI)
hazma/spectra/_photon/_phi.pyx:114  res += BR_PHI_TO_ETAP_A * boost_delta_function(eng_eta, photon_energy, 0.0, beta)
```

(Quoted from the pre-port sources, which Task 4.2 deleted in the same
PR as the swap — `git show 665aed5:<path>` recovers them. The
expressions themselves live on unchanged in
`rust/src/kernels/photon_tables.rs`.)

The local's name — `eng_eta` — says what the expression computes. It is
the η's energy, reused for the η′ without being renamed, and used for
neither's photon:

| Line | Shipped energy / MeV | Correct photon energy / MeV | Ratio |
| --- | --- | --- | --- |
| `φ → η γ` | 656.942002472385 | 362.5189975276151 | 1.81 |
| `φ → η′ γ` | 959.6459594437648 | 59.815040556235125 | 16.04 |

The second is the serious one: it puts 94% of the φ's entire rest mass
into a single photon from a decay whose real photon carries 5.9%. Both
misplaced lines still land below `M_φ`, so nothing raises and no
kinematic guard fires — the spectrum simply has a spike in the wrong
place. Together the two lines carry `BR(φ → ηγ) + BR(φ → η′γ)` =
0.013092 photons per decay, **0.60% of the φ's photon yield** (continuum
2.1616 from `phi_photon.csv`), all of it relocated by +294.4 MeV and
+899.8 MeV respectively in the φ rest frame, and smeared over a
correspondingly wrong window once boosted.

**The CSV columns are what make this a misplaced line rather than a
double count.** `phi_photon.csv` carries an `eta_a` column (non-zero) and
an `etap_a` column (identically zero); the first is the continuum from
the η's *own* decay products, not the direct photon, which is why the
kernel adds a line at all. The same split holds for the ω, whose line is
placed correctly.

## What

1. Change both expressions to `(M_φ² − m²)/(2 M_φ)` in the Rust port,
   which is where the kernel lives from Task 4.2 on:
   `rust/src/kernels/photon_tables.rs`, `PHI_TO_ETA_A_ENERGY` and
   `PHI_TO_ETAP_A_ENERGY`.
2. Regenerate the affected parity corpus case (`spectra.photon.phi`), or
   re-pin it, under whatever mechanism the port has by then. This is a
   **declared numerical change**: `projects/cython-to-rust/rules.md`
   rule 3 and `docs/versioning.md` make a moved published spectrum a
   `minor` bump at least, and it needs a `CHANGELOG.md` entry.
3. Note that this moves a *feature*, not a scale: a band that contained
   the old line and not the new one loses 0.6% of the yield outright,
   while a band containing neither does not move at all. Unlike the
   sibling normalization defect, no downstream result can be corrected
   by a constant factor.

The port's tests already encode the correct statement alongside the
shipped one, so the repair mostly means flipping which is asserted:
`the_phi_line_energies_are_the_daughter_mesons` in
`rust/src/kernels/photon_tables.rs`, and
`TestPhysics::test_the_phi_lines_sit_at_the_daughter_mesons_energy` in
`test/test_core_photon_tables.py`.

## Entry points

- `rust/src/kernels/photon_tables.rs` — `PHI_TO_ETA_A_ENERGY`,
  `PHI_TO_ETAP_A_ENERGY`, and `OMEGA_TO_PI0_A_ENERGY` /
  `OMEGA_TO_ETA_A_ENERGY` beside them, which are the control.
- `test/test_core_photon_tables.py` —
  `TestPhysics::test_the_phi_lines_sit_at_the_daughter_mesons_energy`.
- `test/parity/tolerances.py` — `spectra.photon.phi` is `TABULATED`
  (`rtol = 1e-12`), so nothing absorbs this quietly.
- `hazma/spectra/_photon/data/phi_photon.csv` — the `eta_a` / `etap_a`
  columns that establish what the line is for.
- Sibling defects, same class and same blocker:
  [`eta-prime-two-photon-line-missing-factor-two.md`](eta-prime-two-photon-line-missing-factor-two.md),
  [`positron-muon-spectrum-normalization-inverted.md`](positron-muon-spectrum-normalization-inverted.md),
  [`boost-integral-drops-last-interior-cell.md`](boost-integral-drops-last-interior-cell.md).

## Risks / open questions

- **The φ may also be missing a `φ → π⁰γ` line entirely, and this is not
  established.** `constants.pxd` defines `BR_PHI_TO_PI0_A = 1.32e-3`, no
  `.pyx` reads it, and the ω adds exactly the analogous line for its own
  `π⁰γ` mode. The φ's `pi0_a` column integrates to 0.002612, against
  `2 × BR(π⁰ → γγ) × BR(φ → π⁰γ) = 0.002609` for the π⁰'s decay photons
  alone — suggestive of the direct photon being absent, but the two
  agreeing to 0.1% is weaker evidence than it looks, since the tables are
  truncated at low energy. **Check this properly before repairing the two
  lines above**; if it holds, the same PR should add the missing line
  rather than leaving a third declared change for later.
- Sequencing against the corpus is the real cost, as with its siblings:
  one declared regeneration after Phase 06 Task 6.4 covering all four
  defects is cheaper than four.
