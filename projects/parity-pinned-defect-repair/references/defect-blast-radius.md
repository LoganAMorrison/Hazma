# Which corpus cases each defect reaches

**Audience:** Task 2 (what to capture) and Tasks 4–10 (what to declare).
**Nature:** Grounded facts, derived 2026-08-19 at `3e01590`.

**This table is a prediction, not a measurement.** It is derived from the
composition graph below plus the committed manifest, and it exists so a
repair task knows what to *look* at — not so it can skip looking. Every
repair task re-derives its own row by running the repaired kernel over
the whole corpus and seeing what moved. A case this table omits that
turns out to move is a finding about the graph, not a tolerance to widen.

## The composition graph

Cython, from `grep -rn cimport hazma/` on this tree:

```text
hazma/_utils/boost.pyx
  ├── hazma/spectra/_photon/_pion.pyx        (boost_beta, boost_gamma)
  ├── hazma/spectra/_positron/_muon.pyx      (boost_beta, boost_gamma)
  ├── hazma/spectra/_positron/_pion.pyx      (+ boost_delta_function)
  └── hazma/spectra/_neutrino/_pion.pyx      (+ boost_delta_function)

hazma/spectra/_photon/_muon.pyx
  ├── hazma/spectra/_photon/_pion.pyx
  ├── hazma/scalar_mediator/scalar_mediator_decay_spectrum.pyx
  └── hazma/vector_mediator/vector_mediator_decay_spectrum.pyx

hazma/spectra/_photon/_pion.pyx
  ├── hazma/scalar_mediator/scalar_mediator_decay_spectrum.pyx
  └── hazma/vector_mediator/vector_mediator_decay_spectrum.pyx

hazma/spectra/_positron/_muon.pyx
  ├── hazma/spectra/_positron/_pion.pyx
  ├── hazma/scalar_mediator/scalar_mediator_positron_spec.pyx
  └── hazma/vector_mediator/vector_mediator_positron_spec.pyx
```

Rust, from each kernel module's own call-site table:

```text
rust/src/boost.rs::boost_integrate_linear_interp
  └── rust/src/kernels/photon_tables.rs   (the only consumer)

rust/src/kernels/photon_muon.rs
  └── rust/src/kernels/photon_pion.rs
        └── rust/src/kernels/photon_rho.rs   (nested quadrature)
```

Two facts the graph makes easy to get wrong:

- `boost_integrate_linear_interp` is reached **only** by the seven
  tabulated photon spectra. It is not on the muon, pion, rho, positron
  or neutrino paths — those use `boost_beta` / `boost_gamma` /
  `boost_delta_function`, none of which A1 changes. So the
  boost-window repair does *not* move the mediator spectra. B5 does touch
  `boost_delta_function`, but by removing one of its call sites rather
  than by changing the function, so the two repairs stay disjoint.
- The rho spectra reach the muon kernel *through* the charged pion — and
  that is **not** enough to put them in A2's radius. This bullet used to
  conclude that it was, and Task 2 measured otherwise: A2's defect sits
  behind a guard that fires only for a muon exactly at rest, and every
  caller on that path boosts the muon first, so the edge exists and the
  defect never travels it. A3 has no such guard and does move both rho
  cases, which is why the overlap to manage is A3 against B3 (`rules.md`
  rule 7), not A2 against A3.

  The general form, since this file is a graph-derived prediction and
  will mislead the same way again: **an edge in the graph is a path for a
  *call*, not necessarily for a *defect*.** When the defect is inside a
  branch, ask what argument each caller passes before putting its case in
  the row.

## Per-defect blast radius

Case names are `test/parity/data/manifest.json` keys, written out in
full rather than brace-elided. **This is the canonical enumeration** —
`PLAN.md`'s per-task gates quote it, and any disagreement between the
two is resolved here, then swept into the plan. The brace shorthand this
table used to carry is what let `PLAN.md` say "both mediator photon
cases" against a population of three (PR #72 review).

Counts are derived, not typed:

```sh
python3 -c "import json; m=json.load(open('test/parity/data/manifest.json')); \
  print(sum(1 for n in m['cases'] if n.startswith('mediator_spectra') and '.photon.' in n))"
```

### A1 — boost integral window (7 cases)

`spectra.photon.eta`, `spectra.photon.eta_prime`,
`spectra.photon.omega`, `spectra.photon.phi`,
`spectra.photon.charged_kaon`, `spectra.photon.long_kaon`,
`spectra.photon.short_kaon`.

Blocks: the three boosted blocks and `rest_plus_eps`. **Measured by
Task 2**, which is what the instruction below asked for: `rest` moves 0
of its 1750 positions, because all seven callers short-circuit to the
rest-frame spectrum before the integral and so it never runs at β = 0.
The sign splits by block rather than being uniform — `rest_plus_eps`
moves down at all 1156 positions, the boosted blocks up at 2997 of 2998.
`../task-notes/task-2-cython-oracles.md` has the table.

### A2 — muon photon rest-frame endpoint (1 case, predicted 7)

**Measured by Task 2: `spectra.photon.muon` only, its `rest` block, four
positions.** The other six this row predicted —
`spectra.photon.charged_pion`, `spectra.photon.charged_rho`,
`spectra.photon.neutral_rho`,
`mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum`,
`mediator_spectra.vector.photon.dnde_decay_v` and
`mediator_spectra.vector.photon.dnde_decay_v_pt` — move **0 values
each**.

The prediction came from the composition graph, which is right about who
calls the muon kernel and silent about *how*. The defective branch is
guarded by `emu - MASS_MU < DBL_EPSILON`: it fires only for a muon
exactly at rest. Every composed caller boosts the muon first — the
charged pion evaluates it at `ENG_MU_PIRF = 109.778` MeV, both mediators
at `m/2 ≥ 125` MeV — so no chain reaches the branch at all. "Smeared by
each boost" was the wrong picture: there is nothing to smear.

The general form, worth carrying to the other rows: **a composition edge
in the graph above is not by itself a path to a defect.** If the defect
sits behind a guard, ask what argument the caller passes.

This one also cannot be read off the corpus in the obvious direction.
All four moved positions go from a shipped `0.0` to a small *negative*
value, because the corpus grid samples the top 0.0198 MeV of the
regained window (where the O(α) formula is below zero) and none of the
0.234 MeV of positive spectrum beneath it. See
`../task-notes/task-2-cython-oracles.md`.

### A3 — charged-pion forward cone (6 cases)

`spectra.photon.charged_pion`, `spectra.photon.charged_rho`,
`spectra.photon.neutral_rho`, and the same three mediator photon cases
A2 names —
`mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum`,
`mediator_spectra.vector.photon.dnde_decay_v`,
`mediator_spectra.vector.photon.dnde_decay_v_pt`.

Blocks: all, concentrated in the boosted ones, where the window narrows
past QUADPACK's largest first-rule abscissa.

### A4 — positron-muon normalization (6 cases)

`spectra.positron.muon`, `spectra.positron.charged_pion`,
`mediator_spectra.scalar.positron.dnde_decay_s`,
`mediator_spectra.scalar.positron.dnde_decay_s_pt`,
`mediator_spectra.vector.positron.dnde_decay_v`,
`mediator_spectra.vector.positron.dnde_decay_v_pt`.

Blocks: all, every non-zero position — it is an overall factor.

### B1 — η′ line weight (1 case)

`spectra.photon.eta_prime`. All blocks, at the line's image only.

### B2 — φ line energies (1 case)

`spectra.photon.phi`. All blocks, at both lines' images only.

### B3 — rho rest-frame branch (2 cases)

`spectra.photon.charged_rho`, `spectra.photon.neutral_rho` — the `rest`
block **only**. The guard `E_ρ − m_ρ < DBL_EPSILON` is absolute and one
ulp at 775.26 MeV is 1.14e-13, ~500× `DBL_EPSILON`, so no other double
reaches it.

### B4 — scalar decay FSR normalization (1 case)

`mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum` — the 15
`.default` blocks, both value arrays, at the 3,065 of their 4,305
positions where the FSR term is non-zero (`test/parity/deltas.py`,
`MOVED`); the 15 `mu_mu_only` blocks open no FSR channel and do not
move. The case is inside A3's set (above). Found and repaired after this
roster was drawn
([`scalar-decay-fsr-half-normalized.md`](../../../docs/followups/done/scalar-decay-fsr-half-normalized.md)):
both rest-frame FSR coefficients of the scalar kernel were half the
pair-summed spectrum, so `repaired = stored + fsr_old`, with `fsr_old`
half the repaired kernel's FSR-only spectrum. Declared in
`test/parity/deltas.py`, which landed with the repair rather than under
Task 1.

### B5 — the charged pion's doubled neutrino line (1 case)

`spectra.neutrino.charged_pion` — the `near_rest`, `boosted_mild` and
`boosted_strong` blocks, both value arrays, at the 215 of the case's
4,305 positions where the boosted `π → e ν_e` line's window straddles
`ENU_E_PI_RF` (`test/parity/deltas.py`, `MOVED`). **Measured by Task
10a**, by capturing every block twice — once from a build carrying the
defect, once from the repaired build — so the pre-existing platform drift
between the stored corpus and this tree does not enter. Only the electron
row moves, and only downward; the muon and tau rows are bit-identical.

`rest` and `rest_plus_eps` do not move, for two different reasons: at
`E_π = m_π` the kernel takes a branch that drops both prompt lines, and
one epsilon above it `β` is small enough that no grid point's boost
window straddles the line at all. Found and filed after this roster was
drawn ([`neutrino-pion-electron-line-counted-twice.md`](../../../docs/followups/todo/neutrino-pion-electron-line-counted-twice.md)),
which is also why it is disjoint from everything else here: the pion's
neutrino path reaches `boost_delta_function` and `super::neutrino_muon`,
neither of which any other roster entry touches.

### B6 — thermal quadrature never converged (2 cases)

`cross_sections.scalar.thermal_cross_section` and
`cross_sections.vector.thermal_cross_section` — all three blocks of
each, at 539 of their 570 positions. The first two `cross_sections.*`
cases any defect reaches, and the only entry in this roster whose defect
is in the *integrator* rather than in a closed form: both kernels
inherited `scipy.integrate.quad`'s default `epsabs = 1.49e-8` against an
integral of order 1e-27, so QUADPACK met the absolute criterion on its
first Gauss–Kronrod pass and returned the initial partition unrefined.
Found and repaired after this roster was drawn
([`thermal-cross-section-quadrature-never-converges.md`](../../../docs/followups/done/thermal-cross-section-quadrature-never-converges.md)).

Of the 31 positions that do not move, 30 are the ten points per scalar
block above `x = 300`, where that kernel returns `0.0` before it
integrates; the last is vector `narrow_resonance` at `x = 0.1367`, small
enough that the relative criterion already bound.

Declared with a `Reference` relation rather than an `Additive` one:
there is no term the physics names, because the stored value is not the
true one shifted but the true one unresolved. `test/parity/thermal_reference.py`
supplies what it should hold, integrating the same integrand with
scipy's QUADPACK at `epsrel = 1e-12`.

### The defects, and which group each is in

Group A still has a live Cython twin and is on the clock for its oracle
capture; Group B does not, and has no ordering constraint at all.

| # | Defect | Follow-up | Twin | Serving kernel |
| --- | --- | --- | --- | --- |
| A1 | Boost integral mis-covers its window at both ends | [`boost-integral-drops-last-interior-cell.md`](../../../docs/followups/todo/boost-integral-drops-last-interior-cell.md) | `hazma/_utils/boost.pyx` (live) | `rust/src/boost.rs` |
| A2 | Muon photon rest-frame branch stops short of the endpoint | [`photon-muon-rest-frame-endpoint-uses-the-wrong-power-of-r.md`](../../../docs/followups/todo/photon-muon-rest-frame-endpoint-uses-the-wrong-power-of-r.md) | `hazma/spectra/_photon/_muon.pyx` (live) | `rust/src/kernels/photon_muon.rs` |
| A3 | Charged-pion photon spectrum returns zero in the forward cone | [`charged-pion-photon-spectrum-misses-the-forward-cone.md`](../../../docs/followups/todo/charged-pion-photon-spectrum-misses-the-forward-cone.md) | `hazma/spectra/_photon/_pion.pyx` (live) | `rust/src/kernels/photon_pion.rs` |
| A4 | Muon positron spectrum divides by its normalization | [`positron-muon-spectrum-normalization-inverted.md`](../../../docs/followups/todo/positron-muon-spectrum-normalization-inverted.md) | `hazma/spectra/_positron/_muon.pyx` (live) | `rust/src/kernels/positron_muon.rs` |
| B1 | η′ two-photon line missing its factor of two | [`eta-prime-two-photon-line-missing-factor-two.md`](../../../docs/followups/todo/eta-prime-two-photon-line-missing-factor-two.md) | deleted, Task 4.2 | `rust/src/kernels/photon_tables.rs` |
| B2 | φ photon lines use the daughter meson's energy | [`phi-photon-lines-use-the-daughter-meson-energy.md`](../../../docs/followups/todo/phi-photon-lines-use-the-daughter-meson-energy.md) | deleted, Task 4.2 | `rust/src/kernels/photon_tables.rs` |
| B3 | Both rho spectra return the boost integrand at rest | [`rho-rest-frame-branch-returns-the-integrand.md`](../../../docs/followups/todo/rho-rest-frame-branch-returns-the-integrand.md) | deleted, Task 4.5 | `rust/src/kernels/photon_rho.rs` |
| B4 | Scalar decay spectrum's FSR coefficients are half size | [`scalar-decay-fsr-half-normalized.md`](../../../docs/followups/done/scalar-decay-fsr-half-normalized.md) | deleted, Task 6.2 | `rust/src/kernels/scalar_decay_photon.rs` — **repaired** |
| B5 | Charged pion's prompt `π → e ν` neutrino line is added twice | [`neutrino-pion-electron-line-counted-twice.md`](../../../docs/followups/todo/neutrino-pion-electron-line-counted-twice.md) | deleted, Task 4.6 | `rust/src/kernels/neutrino_pion.rs` — **repaired** |
| B6 | Both thermal averages return the integrator's initial estimate | [`thermal-cross-section-quadrature-never-converges.md`](../../../docs/followups/done/thermal-cross-section-quadrature-never-converges.md) | deleted, Tasks 5.1/5.2 | `rust/src/kernels/vector_xs.rs`, `rust/src/kernels/scalar_xs.rs` — **repaired** |

## Coverage arithmetic

The corpus has 41 cases. The rows above name
7 + 1 + 6 + 6 + 1 + 1 + 2 + 1 + 1 + 2 = **28 case slots** across ten
defects, but four of the ten sets are wholly contained in another —
derived, not eyeballed:

```text
B1 ⊆ A1   B2 ⊆ A1   B3 ⊆ A3   B4 ⊆ A3
A1, A2, A3, A4, B5, B6 are pairwise disjoint
```

So the union is `|A1| + |A2| + |A3| + |A4| + |B5| + |B6|` =
7 + 1 + 6 + 6 + 1 + 2 = **23**. Two consequences worth carrying into the
tasks. A2 and A3 are
now *disjoint* — A2 reaches only `spectra.photon.muon`, A3 reaches
exactly the six cases A2 was predicted to share with it — so Task 8
opens six cases of its own rather than adding positions to ones Task 7
already declared, and `rules.md` rule 7's no-overlap requirement binds
between A3 and B3, and between A3 and B4, rather than between A2 and
A3. B4 is the one repair that has already landed: its declaration on
`scalar_mediator_decay_spectrum` covers the positions the FSR term
moves, and A3 will reach the same arrays through the `pi pi` decay
channel, so Task 8 either proves the two position sets disjoint or
folds B4's declaration into a composite. And A4 is disjoint from
everything else, which is what makes Task 10 safe to run in parallel.

Untouched: **18** — the 16 `cross_sections.*` that are not the two
thermal averages, `spectra.neutrino.muon` (the muon's own neutrino
spectrum has no defect on its path), and `spectra.photon.neutral_pion`
(the π⁰ → γγ box reaches neither the muon kernel nor the boost
integral). 23 + 18 = 41. Counts re-derived with the command above at
Task 10a: 41 cases, 18 of them `cross_sections.*`.

That arithmetic is the cheapest check on this file, and it has now done
its job three times. Task 2's measurement cut A2 from 7 cases to 1, the
slot count fell from 30 to 24 and one containment (`A3 ⊆ A2`) inverted
into a disjointness — but the union stayed at 20, because the six cases A2
lost are exactly the six A3 keeps. Then B4 joined the roster with one case
that was already inside A3's set, so the slot count rose from 24 to 25
and the union stayed at 20 again. Then B5 joined with one case that was
in nobody's set, and this time the union did move — 20 to 21 — and the
untouched count fell from 21 to 20, taking with it this section's own
claim that *neither* `spectra.neutrino.*` case had a defect on its path.
B6 then added the first two `cross_sections.*` cases any defect reaches,
which no spectra defect can touch, so slots went 26 → 28, the union
21 → 23 and the untouched count 20 → 18. Redo the sum after any measured
change and make it come out to 41 again rather than patching one cell.

## The deletion schedule this radius has to beat

From `projects/cython-to-rust/phases/phase-04-spectra-kernels.md` Task
4.6 and `phase-06-mediator-spectra.md` Tasks 6.2–6.4:

| Task | Deletes | Group A capture it strands |
| --- | --- | --- |
| 4.6 | the neutrino pair and their struct module. **Not** `hazma/spectra/_positron/_pion.pyx`: both mediator positron spectrum modules cimport it, so 4.6 removed only its `def` and the file dies with the other capi survivors at 6.4 (corrected when 4.6 landed, 2026-08-20) | A4's `spectra.positron.charged_pion` |
| 6.2 | the two mediator decay spectrum `.pyx` | A2's and A3's three `mediator_spectra.*.photon` cases |
| 6.3 | the two mediator positron spectrum `.pyx` | A4's four `mediator_spectra.*.positron` cases |
| 6.4 | `hazma/spectra/_photon/{_muon,_pion}.pyx`, `hazma/spectra/_positron/{_muon,_pion}.pyx`, `hazma/_utils/boost.{pyx,pxd}` | everything remaining in A1–A4 |

Task 4.6 landed 2026-08-20 and closed Phase 04. It deleted less than
this table expected: `hazma/spectra/_positron/_pion.pyx` is a capi
survivor, so only its `def` went and the `cdef` A4's capture reads is
now in the 6.4 row above. Nothing in Group B appears in this
table — that is what "corpus re-pinning only" means for B1–B3.

### Two windows that had already closed

Added 2026-08-19 by Task 2, which found them by looking rather than by
reading this table. The rows above are the waves still ahead; these two
were behind, and the table as first written implied the whole schedule
was in the future.

| Task | Deleted | Group A chain it stranded |
| --- | --- | --- |
| 4.2 (`0954e5a`) | `hazma/spectra/_photon/{_eta,_eta_prime,_kaon,_omega,_phi}.pyx` | **all seven of A1's cases** |
| 4.5 (`b5f7f90`) | `hazma/spectra/_photon/_rho.pyx` | A2's and A3's two rho cases |

Neither contradicts the A1 row of the roster table: `hazma/_utils/boost.pyx`
*is* live, and `boost_integrate_linear_interp` still evaluates. What died
is every Cython caller of it — the composition graph above says
`rust/src/kernels/photon_tables.rs` is now the only consumer — so no
corpus case could be reached from Cython through the primitive alone. The
same split holds for the rho: A3's pion kernel is live, the outer
quadrature that makes it `spectra.photon.charged_rho` is not.

Task 2 recovered both by rebuilding the deleted sources from git rather
than capturing at the primitive boundary; `test/parity/oracles/defects.py`
carries the file list and the revision each comes from, and
`test/parity/oracles/README.md` carries the loop. The lesson for anything
reading this file later: a twin listed as *live* is a statement about one
`.pyx`, not about the chain from it to a corpus case. Check the chain.
