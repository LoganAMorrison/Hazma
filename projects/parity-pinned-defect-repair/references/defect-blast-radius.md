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
  `boost_delta_function`, which this project does not touch. So the
  boost-window repair does *not* move the mediator spectra.
- The rho spectra reach the muon kernel *through* the charged pion, so
  A2 and A3 both move both rho cases. Their declared positions will
  overlap and must be handled as one composite declaration or as two
  provably disjoint ones.

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

Blocks: all boosted blocks. Whether `rest` moves depends on whether the
integral runs at β = 0 — **measure it**, do not assume.

### A2 — muon photon rest-frame endpoint (7 cases)

`spectra.photon.muon`, `spectra.photon.charged_pion`,
`spectra.photon.charged_rho`, `spectra.photon.neutral_rho`,
`mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum`,
`mediator_spectra.vector.photon.dnde_decay_v`,
`mediator_spectra.vector.photon.dnde_decay_v_pt`.

Blocks: all. The moved region is the last 0.25 MeV below the endpoint in
the muon rest frame, smeared by each boost.

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

## Coverage arithmetic

The corpus has 41 cases. The rows above name 7 + 7 + 6 + 6 + 1 + 1 + 2
= **30 case slots** across seven defects, but four of the seven sets are
wholly contained in another — derived, not eyeballed:

```text
A3 ⊆ A2   B3 ⊆ A2   B1 ⊆ A1   B2 ⊆ A1     and A1, A2, A4 are pairwise disjoint
```

So the union is exactly `|A1| + |A2| + |A4|` = 7 + 7 + 6 = **20**.
Two consequences worth carrying into the tasks. Every case A3 touches,
A2 touches first — so Task 8 declares nothing on a case Task 7 has not
already opened, and `rules.md` rule 7's no-overlap requirement binds on
*positions* there rather than on cases. And A4 is disjoint from
everything else, which is what makes Task 10 safe to run in parallel.

Predicted untouched: **21** — the 18 `cross_sections.*`, the 2 `spectra.neutrino.*`
(no defect on their path; they use `boost_delta_function`, not the
interpolating integral), and `spectra.photon.neutral_pion` (the π⁰ → γγ
box reaches neither the muon kernel nor the boost integral). 20 + 21 = 41.

That arithmetic is the cheapest check on this file: if a repair task's
measured radius changes any row, redo the sum and make it come out to 41
again rather than patching one cell.

## The deletion schedule this radius has to beat

From `projects/cython-to-rust/phases/phase-04-spectra-kernels.md` Task
4.6 and `phase-06-mediator-spectra.md` Tasks 6.2–6.4:

| Task | Deletes | Group A capture it strands |
| --- | --- | --- |
| 4.6 | `hazma/spectra/_positron/_pion.pyx`, the neutrino pair | A4's `spectra.positron.charged_pion` |
| 6.2 | the two mediator decay spectrum `.pyx` | A2's and A3's three `mediator_spectra.*.photon` cases |
| 6.3 | the two mediator positron spectrum `.pyx` | A4's four `mediator_spectra.*.positron` cases |
| 6.4 | `hazma/spectra/_photon/{_muon,_pion}.pyx`, `hazma/spectra/_positron/_muon.pyx`, `hazma/_utils/boost.{pyx,pxd}` | everything remaining in A1–A4 |

Task 4.6 is the only task left in Phase 04, so the first of these
windows is the one closing soonest. Nothing in Group B appears in this
table — that is what "corpus re-pinning only" means for B1–B3.
