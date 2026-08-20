# Charged-pion photon spectrum returns exactly zero in the forward cone

- **Added:** 2026-08-17
- **Source:** `projects/cython-to-rust/task-notes/phase-04/task-4.4-photon-pion.md`
- **Scope:** cross-cutting (public spectrum values)
- **Status:** open
- **Triggers / blockers:** **capture the corrected values BEFORE the
  deletion wave that strands them** — the deadline is on the oracle, not
  on the fix. The parity corpus does pin the zeros, and
  `projects/cython-to-rust/rules.md` rule 2 does forbid regenerating them
  from a tree with ported kernels, so the repair needs corrected
  reference values from somewhere else. `hazma/spectra/_photon/_pion.pyx`
  is that somewhere: fix the `.pyx` in a scratch build, drive it through
  its `__pyx_capi__` capsules, and the corrected values come from a
  compiler and a source tree that both predate the Rust port. The
  mediator-spectra composition of this kernel is stranded earlier still,
  at Tasks 6.2/6.3; Task 6.4 then deletes the twin, and after it the only
  remaining source is the fixed Rust itself, which pins the port against
  its own answer.
  The repair itself has no deadline. Under the plan's mechanism it lands
  as a *declared delta* against the committed corpus arrays rather than
  as a regeneration, so it is legal on a tree with ported kernels and can
  follow the capture by any interval. What cannot follow the deletion is
  the capture. Sequenced in
  [`projects/parity-pinned-defect-repair/PLAN.md`](../../../projects/parity-pinned-defect-repair/PLAN.md)
  — Task 2 (capture) and
  Task 8 (repair); where a later section of this file still reads "after
  Task 6.4", that wording is superseded and the plan is authoritative.

## Why

`hazma.spectra.dnde_photon_charged_pion` integrates over the photon's
angle to the pion, `cos θ ∈ [-1, 1]`, with a fixed adaptive quadrature:

```python
# hazma/spectra/_photon/_pion.pyx:123
quad(charged_pion_integrand, -1.0, 1.0, points=[-1.0, 1.0],
     args=(eng_gam, eng_pi), epsabs=1e-10, epsrel=1e-5)[0]
```

The integrand is nonzero only where the *pion-rest-frame* photon energy
`E' = E_γ γ_π (1 − β_π cos θ)` is below `ENG_GAM_MAX_PIRG = 69.783` MeV.
For a lab photon well above that, the surviving window in `cos θ` is
`cos θ > 1 − 69.783/(E_γ γ_π)` divided by `β_π` — which narrows as the
photon energy approaches the boosted endpoint. QUADPACK starts from a
single 21-point Gauss–Kronrod rule on `[-1, 1]` whose largest abscissa is
about `0.9956`; once the window is narrower than that, **every node
returns zero, the error estimate is zero, and the routine terminates
successfully with the answer `0.0`**. The spectrum is not zero there.

Measured on this tree (`hazma` 2.1.0 + the Task 4.4 port, which reproduces
the Cython's zeros *exactly* — same energies, both implementations, at
every parent energy sampled):

| `E_π` (MeV) | `γ_π` | boosted endpoint (MeV) | first spurious zero, as a fraction of the endpoint |
| --- | --- | --- | --- |
| 200 | 1.4 | 171.6 | 0.99 |
| 500 | 3.6 | 490.1 | 0.99 |
| 1000 | 7.2 | 995.1 | 0.77 |
| 2000 | 14.3 | 1997.5 | 0.37 |
| 5000 | 35.8 | 4998.9 | 0.095 |
| 10000 | 71.6 | 9999.3 | 0.025 |

Spot values against a reference that integrates only the surviving window
(`scipy.integrate.quad` over `[1 − 69.783/(E_γ γ_π))/β_π, 1]`,
`epsrel = 1e-10`):

| `E_π` | `E_γ` | shipped | reference |
| --- | --- | --- | --- |
| 1396 (`10 m_π`) | 900 | `0.0` | `3.586e-07` MeV⁻¹ |
| 1396 | 1200 | `0.0` | `1.135e-08` MeV⁻¹ |
| 1000 | 800 | `0.0` | `5.427e-08` MeV⁻¹ |
| 10000 | 1000 | `0.0` | `9.628e-06` MeV⁻¹ |

The **integrated** effect is small inside hazma's sub-GeV domain and grows
quickly outside it — photons per decay, shipped vs reference:

| `E_π` (MeV) | `γ_π` | shipped | reference | missing |
| --- | --- | --- | --- | --- |
| 200 | 1.4 | 0.307432 | 0.307432 | < 1e-6 |
| 500 | 3.6 | 0.321591 | 0.321591 | < 1e-6 |
| 1000 | 7.2 | 0.333544 | 0.333562 | 0.0054% |
| 1396 | 10.0 | 0.339368 | 0.339508 | 0.041% |
| 5000 | 35.8 | 0.351920 | 0.362662 | 2.96% |

So this is not a yield problem at hazma's own scales; it is a **shape**
problem. At `E_π = 10 m_π` — the parity corpus's most boosted block, and
an ordinary configuration for a ~1 GeV mediator — the differential
spectrum is a hard zero over roughly the top quarter of its support, where
the true spectrum is `O(1e-7)` MeV⁻¹. Anything that reads the high-energy
tail rather than the integral (a spectral-line search, a tail-dominated
limit, a fit to the endpoint region) sees a cliff that is an artefact of
the quadrature.

This is a live defect in hazma 2.1.0, not something the Rust port
introduced. `projects/cython-to-rust/rules.md` rule 1 required the port to
reproduce it, and it does — `test/test_core_photon_pion.py` pins the
agreement, and `rust/src/kernels/photon_pion.rs` records the reasoning.

## What

Repair the quadrature so it samples the physical window rather than the
whole angular range. The cheapest faithful fix is to pass the window edge
as an integration bound (or as a break point), which the call site is
already shaped for — it passes a `points` list today, whose entries scipy
discards because both are endpoints:

```python
cos_min = max((1.0 - ENG_GAM_MAX_PIRG / (eng_gam * gamma_pi)) / beta_pi, -1.0)
quad(charged_pion_integrand, cos_min, 1.0, ...)
```

Points to settle before writing it:

1. **Where the edge belongs.** `ENG_GAM_MAX_PIRG` is one of the five
   *legacy*-table literals in a file that `include`s the PDG table
   (`projects/cython-to-rust/references/cython-inventory.md` §Bugs 3, and
   `rust/src/constants.rs`'s `derived::photon_pion`). Deriving the window
   from a PDG-consistent edge instead moves the answer a second time; do
   one or the other deliberately, not both by accident.
2. **The muon and radiative channels have different edges.** The boosted
   muon spectrum runs to `ENG_GAM_MAX_MURF` boosted out of the muon frame;
   `π → μνγ` closes at `(m_π² − m_μ²)/(2 m_π) = 29.8` MeV and `π → eνγ` at
   `69.783`. Using the widest as a single bound is correct and simplest;
   using three bounds as break points is tighter and needs an argument
   about `qagp` behaviour with interior points.
3. **Everything downstream inherits it, and the ρ compounds it.** Both ρ
   spectra quadrature over this kernel — in
   `rust/src/kernels/photon_rho.rs` since cython-to-rust Task 4.5, in
   `hazma/spectra/_photon/_rho.pyx` before it — and both mediator
   decay-spectrum modules
   (`hazma/{scalar,vector}_mediator/*_decay_spectrum.pyx`) call the
   charged-pion `cdef`s directly.

   **Task 4.5 measured that repairing this kernel is necessary but not
   sufficient for the ρ**, because the outer boost integral hits the same
   QUADPACK failure a second time. A pure inheritance would preserve the
   *fraction* of the endpoint at which the cliff sits — the boost maps
   `E_onset` and the endpoint by the same `γ(1+β)`, since
   `γ(1−β)·γ(1+β) = 1` — so the inner kernel's onset at 0.945 of its own
   endpoint (at the ρ's daughter energy `E_π = 388.44` MeV, `γ_π = 2.78`)
   should appear at 0.945 of the ρ's endpoint at *every* ρ energy. It does
   not:

   | `E_ρ` (MeV) | `γ_ρ` | charged-ρ onset / endpoint | neutral-ρ onset / endpoint |
   | --- | --- | --- | --- |
   | 814 | 1.05 | 0.9963 | 0.9420 |
   | 1163 | 1.5 | 0.9866 | 0.9315 |
   | 1551 | 2 | 0.9707 | 0.9185 |
   | 2326 | 3 | 0.9326 | 0.8806 |
   | 3876 | 5 | 0.8249 | 0.7803 |
   | 7753 | 10 | 0.5366 | 0.5073 |

   The mechanism is the one this file already describes, one level out:
   the outer window `[γE(1−β), γE(1+β)]` spans decades while the
   integrand is nonzero only near its lower end, so once that sub-window
   is narrower than the 21-point Gauss–Kronrod spacing on the full
   interval, every node returns zero. The repair therefore needs a
   restricted outer interval (or break points) in
   `rust/src/kernels/photon_rho.rs`'s `boosted` as well as the inner fix
   here. Measured with
   `projects/cython-to-rust/task-notes/phase-04/task-4.5-photon-rho.md`'s
   probe; the ρ's own corpus cases (`spectra.photon.charged_rho`,
   `spectra.photon.neutral_rho`) pin the zeros, so the same regeneration
   block applies.
4. **The neutral pion is unaffected** — it is closed form, no quadrature.
5. **The repair moves published numbers**, so it is `minor` at least under
   `docs/versioning.md`, needs a `CHANGELOG.md` entry stating the
   magnitude, and needs the parity corpus regenerated for
   `spectra.photon.charged_pion`, `spectra.photon.charged_rho` and
   `spectra.photon.neutral_rho` (plus whatever the mediator spectra pick
   up in Phase 06).

## Entry points

- `hazma/spectra/_photon/_pion.pyx:123` — the `quad` call.
- `hazma/spectra/_photon/_pion.pyx:86-107` — `charged_pion_integrand`, and
  the `E'` relation the window comes from (`:98`).
- `rust/src/kernels/photon_pion.rs` — the port, `CHARGED_PION_QUAD` and
  the faithfulness note.
- `test/test_core_photon_pion.py::TestPhysics` — where the zeros are
  pinned as deliberate.
- `test/parity/tolerances.py` — `spectra.photon.charged_pion` and the two
  rho cases whose stored values would move.
- Related project: `projects/cython-to-rust/`, Phase 06 Task 6.4 is the
  unblocking task.
- Sibling defects sharing the same corpus regeneration:
  `docs/followups/todo/boost-integral-drops-last-interior-cell.md`,
  `docs/followups/todo/photon-muon-rest-frame-endpoint-uses-the-wrong-power-of-r.md`.
