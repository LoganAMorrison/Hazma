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

Case names are `test/parity/data/manifest.json` keys.

| Defect | Corpus cases predicted to move | Blocks |
| --- | --- | --- |
| A1 boost window | `spectra.photon.{eta, eta_prime, omega, phi, charged_kaon, long_kaon, short_kaon}` | all boosted blocks; whether `rest` moves depends on whether the integral runs at β = 0 — **measure it** |
| A2 photon-muon endpoint | `spectra.photon.{muon, charged_pion, charged_rho, neutral_rho}`, `mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum`, `mediator_spectra.vector.photon.{dnde_decay_v, dnde_decay_v_pt}` | all; the moved region is the last 0.25 MeV below the endpoint in the muon rest frame, smeared by each boost |
| A3 charged-pion forward cone | `spectra.photon.{charged_pion, charged_rho, neutral_rho}`, the same three `mediator_spectra.*.photon` cases | all; concentrated in the boosted blocks, where the window narrows past QUADPACK's largest first-rule abscissa |
| A4 positron-muon normalization | `spectra.positron.{muon, charged_pion}`, `mediator_spectra.scalar.positron.{dnde_decay_s, dnde_decay_s_pt}`, `mediator_spectra.vector.positron.{dnde_decay_v, dnde_decay_v_pt}` | all blocks, every non-zero position — it is an overall factor |
| B1 η′ line weight | `spectra.photon.eta_prime` | all blocks, at the line's image only |
| B2 φ line energies | `spectra.photon.phi` | all blocks, at both lines' images only |
| B3 rho rest-frame branch | `spectra.photon.{charged_rho, neutral_rho}` | `rest` **only** — the guard `E_ρ − m_ρ < DBL_EPSILON` is absolute and one ulp at 775.26 MeV is 1.14e-13, ~500× `DBL_EPSILON` |

## Coverage arithmetic

The corpus has 41 cases. Union of the rows above: **20**. Predicted
untouched: **21** — the 18 `cross_sections.*`, the 2 `spectra.neutrino.*`
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
