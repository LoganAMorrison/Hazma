---
status: In Progress
phased: false
version_bump: minor
deliverable: The ten parity-pinned numerical defects repaired (B4, B5 and B6 landed ahead of the task sequence — PR #87, Task 10a and Task 13), each with a declared per-array delta asserted against the corpus arrays that pinned the defect — which stay committed
created: 2026-08-19
---

# Project: Repair the parity-pinned numerical defects

**Structure:** Flat task list.

## Goal

Repair the ten live numerical defects the `cython-to-rust` port
surfaced — seven rostered when this plan was drawn, then three repaired
ahead of the task sequence: B4 in
[PR #87](https://github.com/LoganAMorrison/Hazma/pull/87), B5, whose
follow-up was filed the day after this plan and named it as its home
without a row ever being added, and B6, whose follow-up was corpus-pinned
and so could not be repaired anywhere else — and do it
under a corpus mechanism that keeps the arrays which
pinned each defect rather than overwriting them. Every repair ships with
a **declared delta**: a named statement of which corpus positions move,
by how much, and why — asserted as a gate, so a fix that leaks past its
intended blast radius fails rather than being absorbed.

The project exists because the seven follow-ups' original sequencing was
backwards, and the correction is time-critical. See "The premise this
project corrects" below.

## The premise this project corrects

All seven follow-ups under [`docs/followups/todo/`](../../docs/followups/todo/)
carried some form of *"blocked until after `cython-to-rust` Phase 06
Task 6.4"*. The stated reason is right: the parity corpus pins the
shipped-but-wrong values by construction, `projects/cython-to-rust/rules.md`
rule 2 forbids regenerating it from a tree with ported kernels, and
`test/parity/generate.py` enforces that by refusing to run once
`hazma._core` serves a kernel.

The timing conclusion drawn from it is backwards. Task 6.4 is the task
that **deletes** the last Cython twins
(`projects/cython-to-rust/phases/phase-06-mediator-spectra.md`, Task 6.4
exit criteria: the `.pyx` and `.pxd` of
`hazma/spectra/_photon/_muon`, `hazma/spectra/_photon/_pion`,
`hazma/spectra/_positron/_muon` and `hazma/spectra/_positron/_pion`,
plus `hazma/_utils/boost.{pyx,pxd}` and the `_utils` headers).
Those twins are the only independent implementation a corrected corpus
case can be re-pinned from. After 6.4 the only source of corrected
reference values is the fixed Rust — which pins the port against its own
answer. Waiting for 6.4 destroys the very thing the wait was for.

Every step of that is verified against this tree at `3e01590`, with the
command beside each claim, in
[`references/the-premise.md`](references/the-premise.md) — including the
one that surprises: a whole-corpus regeneration has not been possible
since Phase 04 Task 4.1, so "one declared regeneration after Task 6.4"
was never an available move rather than a mistimed one.

**The deadline is earlier than Task 6.4 for two of the four.** The twins
die in three waves, and a corrected value has to be reachable through the
*whole* Cython composition chain, not just through the one defective
function:

| Wave | Task | Deletes | Closes the window for |
| --- | --- | --- | --- |
| 1 | 4.6 (the only Phase 04 task left) | the neutrino pair and their struct module — **not** `hazma/spectra/_positron/_pion.pyx`, which turned out to be a capi survivor and now dies in wave 3 (corrected when 4.6 landed, 2026-08-20) | `spectra.positron.charged_pion` re-derivation for the positron-muon defect |
| 2 | 6.2 / 6.3 | the four mediator spectrum `.pyx` | every `mediator_spectra.*` case for the muon and charged-pion defects |
| 3 | 6.4 | the four capi survivors + `hazma/_utils/boost.{pyx,pxd}` | everything else |

Task 2 below is what buys the deadline out: it captures the corrected
oracle arrays once, commits them, and from that point the repairs are no
longer racing the port.

## Scope

**In scope:**

- The ten defects rostered in
  [`references/defect-blast-radius.md`](references/defect-blast-radius.md),
  each repaired in the Rust kernel that now serves it (B4 already is).
- A delta-declaration layer under `test/parity/` that pins each repair's
  blast radius while leaving `test/parity/data/*.npz` untouched.
- A committed, provenance-stamped oracle capture from the four live
  Cython twins (Task 2), taken before the port deletes them.
- One `CHANGELOG.md` entry per moved published spectrum, with the
  magnitude, per `projects/cython-to-rust/rules.md` rule 3 and
  [`docs/versioning.md`](../../docs/versioning.md).

B6 also widens the scope sentence above: it is the first defect here
whose blast radius is `cross_sections.*` rather than a spectrum, and the
first whose repair is a quadrature setting rather than a closed form.

**Out of scope:**

- Any change to the `cython-to-rust` port's own task sequence. This
  project runs beside it and consumes its artifacts; it does not
  reorder its phases. Task 11 reconciles the superseded prose, and
  anything canonical there needs that project's change control.
- Regenerating `test/parity/data/*.npz`. The committed arrays are the
  historical record of what 2.1.0 shipped and are never rewritten by
  this project — that is the point of the delta layer.
- The other open follow-ups under `docs/followups/todo/` that are not
  corpus-pinned (`kallen-under-sqrt-remaining-call-sites.md`,
  `scalar-elastic-cross-sections-cancel-in-atan-difference.md`,
  `model-spectra-reject-scalar-energies.md`, and the infrastructure
  items).
- Widening any existing tolerance in `test/parity/tolerances.py`. A
  repair declares a delta; it does not loosen a budget.

## Numerical impact

**This project moves published numbers, deliberately, ten times**
(three of them, B4, B5 and B6, already landed). That
is the whole deliverable, and it is what sets `version_bump: minor` —
no public name, signature, return shape or documented unit changes, but
users' plots move. Known magnitudes, from the follow-ups' own
measurements:

- η′ photon yield rises 0.63%, all of it in a line at `M_η′/2 = 478.89`
  MeV.
- φ: 0.60% of the photon yield relocates **down** by 294.4 MeV and
  899.8 MeV in the φ rest frame — the repair moves each line from where
  it ships to where it belongs (656.942 → 362.519 MeV for `φ → ηγ`,
  959.646 → 59.815 MeV for `φ → η′γ`). The follow-up states the same two
  magnitudes with a `+` sign because it describes the *defect's*
  displacement, which runs the other way.
- Charged-pion photon spectrum: a hard zero over roughly the top quarter
  of its support disappears; integrated, 0.0054% at `E_π = 1` GeV,
  0.041% at 1396 MeV, 2.96% at 5 GeV. A shape defect, not a yield
  defect, at hazma's scales.
- Muon photon spectrum: the rest-frame branch regains the last 0.25 MeV
  to the endpoint.
- Positron muon spectrum: normalization moves by `R_FACTOR²`.
- Boost integral: all seven tabulated photon spectra move, and **the sign
  splits by regime rather than being one-signed**. Near threshold the
  shipped values are 6,500x to 33,000x too high and now converge to their
  own rest-frame spectrum; away from it they were low by the dropped cell
  and rise, by a median 3.3% at `γ = 1.05` falling to 7.1e-5 at `γ = 10`.
  4,154 of the 10,045 pinned positions move; a parent exactly at rest does
  not. Measured by Task 4; the follow-up's "systematically low" reading
  described only the second regime.
- Both rho spectra at `E_ρ = m_ρ` exactly: divided by `E_γ`, i.e. the
  value changes by a factor of `E_γ` at a single parent energy.
- Charged-pion **neutrino** spectrum: the electron-neutrino row loses one
  `BR(π → e ν_e) = 1.230e-4` per pion, 0.0123% integrated; pointwise about
  5e-5 on the plateau and exactly a factor of two above the muon-decay
  continuum's support. The muon and tau rows do not move.
- Both mediator `thermal_cross_section` implementations, and with them
  `relic_density` for every model that supplies one: ⟨σv⟩ was wrong by
  up to **100%** across the freeze-out region, and the relic densities
  `test/test_relic_density.py` pins move by −92% and −99.9% at the two
  closed-resonance points. **The largest correction in the project by
  four orders of magnitude**, and the only one that is not a spectrum.

Task 12 aggregates the measured figures. Every bullet above except the
last is a pre-repair estimate the follow-up recorded rather than one of
this project's own measurements, and is re-derived by its repair task;
the B5 bullet is the measurement, because Task 10a has run.

## Tasks

The live task table, status, and dependency diagram are tracked in
[`task-notes/README.md`](task-notes/README.md). This `PLAN.md` describes
the canonical *shape* of each task below.

## Orientation

| Reference | Nature |
| --- | --- |
| [`references/the-premise.md`](references/the-premise.md) | Grounded facts — the corrected sequencing premise, claim by claim, with the command that produced each. Read once; Task 11 re-derives it. |
| [`references/corpus-repinning.md`](references/corpus-repinning.md) | Spec — the delta-declaration mechanism, the oracle capture protocol, and the per-defect proof obligations. Tasks 1–3 and every repair task. |
| [`references/defect-blast-radius.md`](references/defect-blast-radius.md) | Grounded facts — which corpus case and block each defect reaches, derived from the cimport graph and the committed manifest. Every repair task. |
| [`rules.md`](rules.md) | This project's cross-cutting rules. All tasks. |

## The defects

Ten, labelled **A1–A4** (a live Cython twin, so an oracle capture is on
the clock) and **B1–B6** (no twin, and no ordering constraint at all).
The roster — each defect's follow-up, its twin's fate, the Rust kernel
that serves it now, and the corpus cases it reaches — is one table, in
[`references/defect-blast-radius.md`](references/defect-blast-radius.md).
It lives there rather than here so the labels, the case lists and the
counts have exactly one home; the task gates below quote it by row.

## Task Details

### Task 1: The delta-declaration layer

**Objective:** Give the parity suite a way to say "this array moved,
here, by this much, because of this repair" without rewriting the array.

**Scope / implementation notes:** A new `test/parity/deltas.py`, keyed on
the same `(case_name, block_label, array_suffix)` tuple
`test/parity/stability.py` already uses for `PORTABILITY_ZEROS`. Each
declaration names the repair, the affected positions, the expected
relation between stored and repaired value, and the measurement that
justifies it. `test/parity/test_parity.py` consults it *after* the
existing budget selection: a declared position is compared against the
declared relation, an undeclared position against the original stored
array under its existing `test/parity/tolerances.py` budget. See
[`references/corpus-repinning.md`](references/corpus-repinning.md) for
the declaration schema and why it is an allowlist rather than a rule
over positions of the same shape.

**Deliverable / gate:** `pytest test/parity` green with an empty
declaration set and a collected count that matches today's, proving the
layer is inert before any repair lands. A shape test that fails if a
declaration names a case, block, array or position the corpus does not
contain. `git diff --stat -- test/parity/data` empty.

**Status:** landed in [PR #87](https://github.com/LoganAMorrison/Hazma/pull/87)
together with the B4 repair rather than ahead of the first one, so the
inertness proof took a different shape — see
`task-notes/task-1-delta-declarations.md` and
[`adrs/ADR-0001-corpus-repairs-are-declared-deltas.md`](adrs/ADR-0001-corpus-repairs-are-declared-deltas.md).

### Task 2: Capture the corrected-value oracles from the four live twins

**Objective:** Take the one measurement that stops being possible when
the port deletes the Cython, and commit it. **This task is the deadline;
everything else in the project is schedulable afterwards.**

**Scope / implementation notes:** For each Group A defect, apply the
repair to the `.pyx` in a scratch build, drive the patched `cdef` through
its `__pyx_capi__` capsules the way `test/test_core_boost.py` already
drives `hazma._utils.boost`, and capture the corrected values on the
corpus's own capturing platform (read it from
`test/parity/data/manifest.json`, per `docs/agents/lessons.md`
`[platform-scoped-oracle-asserted-globally]` — do not probe for it).
Commit the arrays under `test/parity/oracles/` with a manifest carrying
the patched-source digest, the platform, and the `hazma` package path
actually imported (`[measured-tree-vs-imported-module]`). **No library
behavior ships in this task** — the `.pyx` patch exists only inside the
capture and is reverted, verified by `git diff -- hazma` being empty on
the final tree.

Capture the composition chain too, not only the defective function: for
A2 and A3 that means the `mediator_spectra.*` cases, which lose their
Cython at Tasks 6.2/6.3; for A4 it means `spectra.positron.charged_pion`,
which was expected to lose its Cython at Task 4.6 and in the event kept
it — 4.6 removed only the `def`, and the `cdef` this capture reads
survives to Task 6.4. Task 2 had already captured it by then, so the
correction relaxes the deadline rather than moving it. See
[`references/defect-blast-radius.md`](references/defect-blast-radius.md)
for the per-defect case list and
[`references/corpus-repinning.md`](references/corpus-repinning.md) for
the capture protocol.

**Deliverable / gate:** `test/parity/oracles/` committed and
self-checking (a `--check` mode that verifies the arrays against their
manifest hashes without a build, mirroring
`python test/parity/generate.py --check`). A test that the oracle
manifest's platform matches the corpus manifest's. A recorded diff, per
defect, between the oracle and the corresponding stored corpus array —
the first measurement of each repair's size, taken from Cython rather
than from Rust.

### Task 3: Closed-form delta models for the three twin-less defects

**Objective:** Establish, for B1–B3, a non-circular statement of the
expected delta that needs no Cython twin.

**Scope / implementation notes:** Each of the three has a closed form:

- **B3 (rho):** the corrected `rest` value is the stored value times
  `E_γ` exactly. The follow-up's own ratio table measures the ratio as
  `13.000000`, `50.000000`, `200.000000`, `300.000000` at those photon
  energies. The delta is a transform of the committed array and needs
  no kernel.
- **B1 (η′):** the fix adds a second copy of a line term the stored
  spectrum already carries once, so the delta is
  `BR_ETAP_TO_A_A · boost_delta_function(M_η′/2, …)`. This paragraph
  called that a function of `hazma/_utils/boost.pyx` and said the file
  was still live; `cython-to-rust` Task 6.4 has since deleted it, so the
  model reads `hazma._core.boost` — a kernel B1 does not repair, and one
  whose window arithmetic the model has to match bit for bit, because
  that is what decides which positions `MOVED` resolves to.
- **B2 (φ):** both line energies are closed forms,
  `(M_φ² − m²)/(2 M_φ)`; the delta is the two boosted line terms
  recomputed there minus the two the corpus stored.

Where the closed form is analytic, an `mpmath` reference in the shape of
`test/parity/reference.py` is the precedent for settling it without
trusting either implementation. Task 3 measured that none of these three
earns one: that file exists because its kernels lose about 33 decimal
digits to an `atan` cancellation, and against 60-digit `mpmath` these
forms lose under half a digit — the two-body energies are good to 15.6
to 16.2 digits, and a boosted line's `height × width` is `1.0` to within
one ulp. What checks them instead is the committed corpus, which is the
stronger oracle: an independent implementation, already pinned.

**Deliverable / gate:** For each of B1–B3, a model in
`test/parity/deltas.py` — its relation, its budget, and the measurement
that justifies the budget — and a test that the model reproduces the
**shipped** value from the corrected form plus the named defect, which
is falsifiable without either the repaired Rust or a Cython twin and is
what makes the model non-circular. The model does **not** take a
`DECLARED_DELTAS` key here: ADR-0001's staleness rule fails a
declaration whose array has not moved, so the key lands with the repair
(Tasks 5, 6 and 9) and until then the model waits in `DELTA_MODELS`,
gated by `test/parity/test_delta_models.py`. Recovering the deleted
sources for review is `git show 665aed5:<path>` (B1, B2) and
`git show b5f7f90^:hazma/spectra/_photon/_rho.pyx` (B3).

### Task 4: Repair A1 — the boost integral window

**Objective:** Cover `[x[ihigh-1], x[ihigh]]`, and stop dropping the
table's final row when the boosted window reaches past the table.

**Scope / implementation notes:** `rust/src/boost.rs`,
`boost_integrate_linear_interp`. The consumer set is narrow and entirely
Rust: `rust/src/kernels/photon_tables.rs` is the only kernel that reaches
it, per that module's own call-site table. Sequenced first because those
seven tabulated photon cases are also where B1 and B2 land, and doing the
primitive first means their deltas are measured once.

**Deliverable / gate:** Declared deltas on all **7** cases the A1 row of
[`references/defect-blast-radius.md`](references/defect-blast-radius.md)
names — `spectra.photon.eta`, `spectra.photon.eta_prime`,
`spectra.photon.omega`, `spectra.photon.phi`,
`spectra.photon.charged_kaon`, `spectra.photon.long_kaon`,
`spectra.photon.short_kaon` — and on nothing else; the repaired value
reproduces Task 2's Cython oracle within the function's existing
budget. **The sign is not one-signed** — this paragraph said it was, on
the follow-up's "systematically low" characterization, and Task 2
measured otherwise. It splits by block, and the assertion is per block:
`rest_plus_eps` moves **down** at all 1156 of its positions (the two
partial-cell terms overlap there, and the shipped value is ~9,800× too
high), `near_rest`, `boosted_mild` and `boosted_strong` move **up** at
2997 of 2998, and `rest` does not move at all — no caller reaches the
integral at β = 0. `task-notes/task-2-cython-oracles.md` has the table.
Every other corpus case unchanged against its original stored array.

### Task 5: Repair B1 — the η′ two-photon line weight

**Objective:** `2 · BR(η′ → γγ)`, matching its four `X → γγ` siblings.

**Scope / implementation notes:** `rust/src/kernels/photon_tables.rs`,
`ETAP_TO_A_A_WEIGHT`. Depends on Task 4: the same
`spectra.photon.eta_prime` arrays carry the boost-window delta.
The port's existing tests already state the correct physics alongside the
shipped defect, so the repair largely flips which is asserted —
`the_eta_prime_line_is_missing_its_factor_of_two` and
`TestPhysics::test_the_eta_prime_line_carries_half_the_photons_it_should`
must be renamed as well as re-pointed, per `docs/agents/lessons.md`
`[test-name-claims-an-unmade-assertion]`.

**Deliverable / gate:** Declared delta on `spectra.photon.eta_prime`
only. The line-term integral measures `2 · BR = 0.04614` photons per
decay against the `0.02306998 ± 1.3e-08` the follow-up measured
pre-repair; the continuum is unchanged. `spectra.photon.eta`,
`spectra.photon.long_kaon` and `spectra.photon.short_kaon` — the three
siblings that were already right — do not move.

### Task 6: Repair B2 — the φ photon line energies

**Objective:** Place both φ lines at `(M_φ² − m²)/(2 M_φ)`.

**Scope / implementation notes:** `rust/src/kernels/photon_tables.rs`.
Depends on Task 4, same reason as Task 5. The repair moves both lines
**down**: `φ → ηγ` from 656.942 to 362.519 MeV (−294.4), and `φ → η′γ`
from 959.646 to 59.815 MeV (−899.8), a factor of 16. Nothing raises and no kinematic
guard fires either before or after, so the gate has to be a *position*
assertion, not a "does it still return finite" one.

**Deliverable / gate:** Declared delta on `spectra.photon.phi` only, and
the declaration names the two energies rather than a magnitude — the
total yield is unchanged (0.013092 photons per decay, relocated), so a
yield-only check would pass on an unrepaired kernel. `spectra.photon.omega`,
whose lines were already right, does not move.

### Task 7: Repair A2 — the muon photon rest-frame endpoint

**Objective:** Guard on `y ≥ 1 − r` with `r = (m_e/m_μ)²`, not
`1 − √r`.

**Scope / implementation notes:** `rust/src/kernels/photon_muon.rs`.
This kernel is composed by three others (charged pion, both rhos, and
both mediator decay spectra), so it comes before Tasks 8 and 9. The Rust
`fn` is in the PyO3-free kernel layer and Phase 06 calls it natively, so
the repair reaches the mediator spectra whether or not Phase 06 has
landed.

**Deliverable / gate:** A declared delta on **1** case —
`spectra.photon.muon`, its `rest` block, four positions. The A2 row of
[`references/defect-blast-radius.md`](references/defect-blast-radius.md)
predicted seven; Task 2 measured the other six at zero moved values
each, because the rest-frame branch is guarded by
`emu - MASS_MU < DBL_EPSILON` and every composed caller boosts the muon
first (the charged pion at `ENG_MU_PIRF = 109.778` MeV, both mediators at
`m/2 ≥ 125` MeV), so no composition chain can reach it. Re-derive that
against the *Rust* rather than inheriting it — Task 2 measured Cython —
but expect one case, and treat six moving as a finding about
`photon_muon.rs` rather than as the prediction being vindicated.

**Settle the negative sliver before declaring anything.** All four
positions that move go from a shipped `0.0` to a *negative* value
(−2.92e-09 to −3.00e-09): the corpus grid puts no point in the 0.234 MeV
of positive spectrum the repair regains, and four in the 0.0198 MeV at
the top where the O(α) rest-frame formula evaluates below zero. So this
task's entire numerical impact is the sign question, not the endpoint
extension. `task-notes/task-2-cython-oracles.md` has the measurement and
the three options.

The endpoint invariant Task 4.3 wrote —
`the_in_flight_form_is_the_boost_integral_of_the_rest_frame_form` — must
still hold. Note it is an *in-flight* identity, so it is not what the
four moved positions test.

### Task 8: Repair A3 — the charged-pion forward cone

**Objective:** Stop `qagp` from terminating successfully at `0.0` because
every abscissa fell outside the integrand's support.

**Scope / implementation notes:** `rust/src/kernels/photon_pion.rs`,
`CHARGED_PION_QUAD`. The integrand is nonzero only where the
pion-rest-frame photon energy stays under the widest of its three
channel edges; the fix is to integrate over that window rather than over
all of `cos θ` and hope the adaptive rule finds it. Task 2's oracle used
`(m_π² − m_e²)/(2 m_π) = 69.784260` MeV rather than the
`ENG_GAM_MAX_PIRG = 69.783458` literal this paragraph used to name: the
literal is the *narrower* of the two by 8.0e-4 MeV and clips the last
sliver of `π → eνγ`. Match it or re-derive the oracle. Depends on Task 7 (this
kernel boosts the muon spectrum). Partition the verification grid by
scipy's own convergence verdict rather than picking one tolerance —
`projects/cython-to-rust/task-notes/README.md` records that lesson from
this exact kernel, and PR #68's two CI rounds are why.

**Deliverable / gate:** Declared deltas on all **6** cases the A3 row of
[`references/defect-blast-radius.md`](references/defect-blast-radius.md)
names — `spectra.photon.charged_pion`, `spectra.photon.charged_rho`,
`spectra.photon.neutral_rho`,
`mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum`,
`mediator_spectra.vector.photon.dnde_decay_v` and
`mediator_spectra.vector.photon.dnde_decay_v_pt`. Task 2 measured all six
moving, so this row stands. A3 was described here as a strict subset of
A2; A2 measured at one case (`spectra.photon.muon`) and the two are now
**disjoint**, so every case here is one Task 8 opens itself rather than
one Task 7 has already touched, and `rules.md` rule 7's no-overlap
requirement binds between this task and Task 9 (B3, the rho `rest`
blocks) rather than against Task 7 — and against the B4 declaration
already on `scalar_mediator_decay_spectrum`, which this task must prove
disjoint from its own positions or fold into one composite. The specific
figure the follow-up
pins is confirmed from Cython:
`dnde_photon_charged_pion(900, 1396)` moves from `0.0` to
`3.585860e-07` MeV⁻¹, against the `3.586e-07` predicted. A test that no
stored zero in that case survives repair *except* the ones outside the
kinematic support — the difference between the two is the whole defect,
and a declaration that covers both would hide it.

### Task 9: Repair B3 — the rho rest-frame branch

**Objective:** Return the rest-frame spectrum, not the boost integrand.

**Scope / implementation notes:** `rust/src/kernels/photon_rho.rs`,
`boosted`. One line, plus the two Rust unit tests and the Python test
that currently pin the defect — all three named in the follow-up's
"Entry points", all three needing a rename as well as a re-point. Last
of the rho-touching tasks because Tasks 7 and 8 also move those arrays;
the branch fires only at `E_ρ == m_ρ` exactly, so its declared positions
are disjoint from theirs and the two deltas must be shown not to overlap.

**Deliverable / gate:** Declared delta on the **2** cases the B3 row of
[`references/defect-blast-radius.md`](references/defect-blast-radius.md)
names — the `rest` block of `spectra.photon.charged_rho` and of
`spectra.photon.neutral_rho` — and nowhere else. The guard
`E_ρ − m_ρ < DBL_EPSILON` is absolute and one ulp at 775.26 MeV is
1.14e-13, about 500× `DBL_EPSILON`, so no other double reaches it, and a
delta declared on any other block is a bug in the declaration. The ratio
to the stored value is `E_γ` at every declared position, exactly.

### Task 10: Repair A4 — the muon positron normalization

**Objective:** Multiply by the normalization instead of dividing.

**Scope / implementation notes:** `rust/src/kernels/positron_muon.rs`.
Independent of Tasks 4–9 — a separate branch of the composition graph —
so it may run in parallel with them — A4's radius is disjoint from every
other defect's, which is what makes that safe.

**Deliverable / gate:** Declared deltas on all **6** cases the A4 row of
[`references/defect-blast-radius.md`](references/defect-blast-radius.md)
names — `spectra.positron.muon`, `spectra.positron.charged_pion`,
`mediator_spectra.scalar.positron.dnde_decay_s`,
`mediator_spectra.scalar.positron.dnde_decay_s_pt`,
`mediator_spectra.vector.positron.dnde_decay_v`, and
`mediator_spectra.vector.positron.dnde_decay_v_pt`. Four mediator cases,
not two: each mediator ships both a `dnde_decay_*` and a
`dnde_decay_*_pt` entry point. The repaired value reproduces Task 2's
oracle. The analytic normalization test that
found the defect (Task 4.1's) now passes against the corrected constant
rather than recording the inversion, and the Michel spectrum integrates
to 1 over its support to a stated tolerance.

### Task 10a: Repair B5 — the charged pion's doubled neutrino line

**Objective:** Let `dnde_e_nue` be the sole source of the `π → e ν_e`
line, so each prompt line is counted once.

**Scope / implementation notes:**
`rust/src/kernels/neutrino_pion.rs`, the `delta_e` binding in
`dnde_mu_numu` and its use in the returned `electron` field. Numbered
between Tasks 10 and 11 rather than appended after Task 12 because the
close has to come last and this repair has to precede the prose sweep
that moves its follow-up to `done/`; it depends on Task 1 alone and is
disjoint from every other defect's blast radius, so it may run in
parallel with Tasks 3–10. Group B: the Cython twin
(`hazma/spectra/_neutrino/_pion.pyx`) died in `cython-to-rust` Task 4.6,
before this roster entry existed, so there is no Task 2 oracle and the
independent check is the closed form — the boosted line is a rectangle of
height `BR_e / (2 γ β E_ν^rf)` over `E ∈ (E_rf/(γ(1+β)), E_rf/(γ(1−β)))`,
which needs neither a kernel nor a build. The port's existing tests state
the correct physics alongside the shipped defect, so the repair largely
flips which is asserted — `the_electron_line_is_counted_by_both_halves`
and `TestPhysics::test_the_electron_line_is_counted_twice_and_the_muon_line_once`
need renaming as well as re-pointing, per `docs/agents/lessons.md`
`[test-name-claims-an-unmade-assertion]` (they are
`the_electron_line_comes_from_one_half_only` and
`test_each_prompt_line_is_counted_exactly_once` on the repaired tree), and
`test/test_core_neutrino.py`'s `reference_dnde_neutrino_charged_pion` —
the independent recomputation that satisfies rule 3 — reproduces the
doubled line on purpose and has to stop.

**Deliverable / gate:** A declared delta on the **1** case the B5 row of
[`references/defect-blast-radius.md`](references/defect-blast-radius.md)
names, `spectra.neutrino.charged_pion`, and on **3** of its 5 blocks:
`near_rest`, `boosted_mild` and `boosted_strong`. `rest` and
`rest_plus_eps` must still match their stored arrays under the case's own
budget — at rest the kernel drops both prompt lines, and one epsilon above
it no grid point's boost window is wide enough to straddle the line — and
that is the "moved only what it intended" half of the proof. The
electron-neutrino row integrates to `BR_μ + BR_e` rather than
`BR_μ + 2 BR_e`, and the muon row is bit-identical: the asymmetry is what
makes the pair discriminating, since `dnde_e_nue` never wrote to the muon
row. `spectra.neutrino.muon` does not move.

### Task 11: Reconcile the superseded sequencing prose

**Objective:** Leave no live document still telling a reader to wait for
Task 6.4.

**Scope / implementation notes:** The corrected premise is stated in the
seven follow-ups' "Triggers / blockers" bullets and here. Copies survive
elsewhere, and they split by file role, per `docs/agents/lessons.md`
`[sweep-excluded-the-canonical-directory]`: task notes and learnings are
dated history and stay as written; a phase file and a `references/` file
that declares itself spec are live and do not.

Known live copy, found by
`grep -rn "Task 6\.4" --include="*.md" .` at plan time:
`projects/cython-to-rust/phases/phase-03-numerics-foundation.md` states
the boost repair is "blocked until after Phase 06 Task 6.4 because it
needs a declared corpus regeneration". That is a canonical statement in
another project's phase file, so changing it goes through that project's
change control (`projects/cython-to-rust/rules.md`, Process; an ADR if
the correction is canonical rather than clerical). Re-derive the full
population at execution time rather than trusting this paragraph, and
sweep the behavior words as well as the task id
(`[settling-a-deferral-has-two-sweeps]`).

Also in scope, and a second class rather than a second copy: the
**twin-liveness** claims. `references/defect-blast-radius.md`'s roster
table annotates each Group A twin `(live)` and its lead-in says Group A
"still has a live Cython twin and is on the clock for its oracle
capture"; `references/corpus-repinning.md`'s Task 2 protocol says
`test/test_core_boost.py` "already drives `hazma._utils.boost`". Task 2
is complete and Task 6.4 left no `.pyx` in the tree, so all of that is
now false in a live `references/` file. Sweep on the liveness claim
(`still live`, `(live)`, `still supplies`) as well as on the task id —
PR #94's review found the same sentence surviving in two follow-ups
after the plan itself had been corrected.

Also in scope: the "Risks" sections of the seven follow-ups, which still
propose "one declared regeneration after Phase 06 Task 6.4". Those were
deliberately left standing when the blocker bullets were rewritten, so
the correction and its evidence would land in one reviewable place
first; each blocker bullet says so and points here.

**Deliverable / gate:** `scripts/agents/check_doc_citations.py` over
every touched doc — it is not in `preflight.sh`, per
`[gate-green-is-not-citations-green]` — plus a paste of the sweep
commands and their output, written after the last prose edit
(`[sweep-block-written-from-intent]`).

### Task 13: Repair B6 — the thermal averages never converged

Added after the original twelve, and executed immediately rather than
queued: it depends on nothing in Tasks 3–10 and blocks Tasks 11 and 12,
which aggregate and close.

**Objective:** Make both mediator `thermal_cross_section` implementations
integrate to a criterion that binds, and declare what that moves.

**Scope / implementation notes:** Four call sites share the defect, and
a repair confined to the two Rust kernels would leave the other two: the
generic fallback in `hazma/relic_density/_thermal_functions.py` and the
GeV vector model's own in
`hazma/vector_mediator/_gev/thermal_cross_section.py`, neither of which
the port ever touched. `epsabs = 0` at all four. The two kernels also
need a subdivision limit above `quad`'s default of 50, or the criterion
they now bind to cannot be reached — see `THERMAL_LIMIT`.

The declaration needs a relation the layer did not have. `Additive` and
`Exact` both presuppose a term or transform the physics names, and there
is none here: the stored value is not the true one shifted, it is the
true one unresolved. `Reference` supersedes it instead, against
`test/parity/thermal_reference.py`.

**Deliverable / gate:** `pytest test/parity` green with the six thermal
arrays declared, and red when the repair is reverted. The twelve pinned
values in `test/test_relic_density.py` re-derived from the corrected
kernel rather than absorbed by a widened tolerance.

### Task 12: Close — aggregate the drift and ship the bump

**Objective:** One `CHANGELOG.md` entry naming this slug, carrying every
measured shift, and the `minor` bump.

**Scope / implementation notes:** The per-repair figures accumulate in
`task-notes/README.md`'s "Numerical impact so far" as each task lands;
this task aggregates rather than reconstructs. Re-check the level against
the aggregate before bumping — ten deliberate corrections to published
numbers with no API change is `minor`, and nothing in Tasks 4–10 or 13
should have raised it, but the check is the point. B6 is the one to look
at hardest: it moves `relic_density` by up to 99.9%, which is a far
larger user-visible change than any spectrum here, and `minor` still
covers it only because no public name, signature, return shape or
documented unit moves.

**Deliverable / gate:** `scripts/agents/preflight.sh --closing` green;
`PLAN.md` `status: Complete`; the follow-ups still under `todo/` — the
original seven less B5's and B6's, which are already in `done/` — moved to
`docs/followups/done/` with their inbound links repointed and the
revision pinned, per
[`docs/workflow.md`](../../docs/workflow.md)'s follow-up lifecycle and
`[touched-doc-inherits-its-citations]`.

## Dependencies

- Requires: nothing complete. Tasks 1–3 run against the tree as it
  stands.
- **Constrains:** `cython-to-rust` Task 4.6, then Tasks 6.2/6.3, then
  6.4. Task 2 must land before Task 4.6 for the
  `spectra.positron.charged_pion` capture, and before 6.2/6.3 for the
  mediator-spectra captures. If the port reaches those first, the
  corresponding oracles are unrecoverable from any source but the Rust
  itself and the affected repairs lose their independent check — say so
  in the task note rather than proceeding as if nothing was lost.

## Related

- Background: the seven follow-ups listed under "The defects", and
  `projects/cython-to-rust/task-notes/numerical-impact.md` (that
  project's "Numerical impact so far" log, moved out of its
  `task-notes/README.md` on 2026-08-21), which is where each defect was
  first measured.
- `projects/cython-to-rust/rules.md` rules 1–3 (parity discipline) are
  the constraint this project is designed around; nothing here relaxes
  them.

## Change control

See [`../../docs/workflow.md#adr-placement`](../../docs/workflow.md#adr-placement)
for when to write an ADR and where it lives (repo-wide vs project-scoped).
Patch the affected `PLAN.md` / `rules.md` when canonical behavior
changes.

## Closing this project

The PR that flips this `PLAN.md` `status:` to `Complete` must also bump
`[project] version` in `pyproject.toml` per the `version_bump:`
frontmatter and
add a `CHANGELOG.md` entry naming this project slug. Re-check the level
against the **Numerical impact** section above before bumping. Verify
with `scripts/agents/preflight.sh --closing`. See
[`../../docs/versioning.md`](../../docs/versioning.md) for the full
policy.

### Anticipated ADRs

- **The delta-declaration layer as the standing re-pinning mechanism**
  (project-scoped, Task 1). It changes what the parity corpus asserts,
  which is `cython-to-rust`'s gate as much as this project's, so if the
  layer outlives this project it is a candidate for promotion to
  `docs/adrs/`.
- **Whether the corrected oracles stay committed after `cython-to-rust`
  closes** (project-scoped, Task 2). They are the last evidence that a
  repaired value was ever checked against a non-Rust implementation;
  the alternative is that the port's own tests become self-referential
  the moment the Cython goes.
