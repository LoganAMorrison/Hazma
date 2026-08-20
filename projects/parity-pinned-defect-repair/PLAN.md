---
status: In Progress
phased: false
version_bump: minor
deliverable: The seven parity-pinned numerical defects repaired, each with a declared per-array delta asserted against the corpus arrays that pinned the defect — which stay committed
created: 2026-08-19
---

# Project: Repair the parity-pinned numerical defects

**Structure:** Flat task list.

## Goal

Repair the seven live numerical defects the `cython-to-rust` port
surfaced, and do it under a corpus mechanism that keeps the arrays which
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
| 1 | 4.6 (the only Phase 04 task left) | `hazma/spectra/_positron/_pion.pyx`, the neutrino pair | `spectra.positron.charged_pion` re-derivation for the positron-muon defect |
| 2 | 6.2 / 6.3 | the four mediator spectrum `.pyx` | every `mediator_spectra.*` case for the muon and charged-pion defects |
| 3 | 6.4 | the four capi survivors + `hazma/_utils/boost.{pyx,pxd}` | everything else |

Task 2 below is what buys the deadline out: it captures the corrected
oracle arrays once, commits them, and from that point the repairs are no
longer racing the port.

## Scope

**In scope:**

- The seven defects rostered in
  [`references/defect-blast-radius.md`](references/defect-blast-radius.md),
  each repaired in the Rust kernel that now serves it.
- A delta-declaration layer under `test/parity/` that pins each repair's
  blast radius while leaving `test/parity/data/*.npz` untouched.
- A committed, provenance-stamped oracle capture from the four live
  Cython twins (Task 2), taken before the port deletes them.
- One `CHANGELOG.md` entry per moved published spectrum, with the
  magnitude, per `projects/cython-to-rust/rules.md` rule 3 and
  [`docs/versioning.md`](../../docs/versioning.md).

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

**This project moves published numbers, deliberately, seven times.** That
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
- Boost integral: the seven tabulated photon spectra rise by the dropped
  cell; systematic and one-signed (they are currently always slightly
  low).
- Both rho spectra at `E_ρ = m_ρ` exactly: divided by `E_γ`, i.e. the
  value changes by a factor of `E_γ` at a single parent energy.

Task 12 aggregates the measured figures; the per-defect numbers above are
the pre-repair estimates the follow-ups recorded, not this project's own
measurements, and are re-derived by each repair task.

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

Seven, labelled **A1–A4** (a live Cython twin, so an oracle capture is on
the clock) and **B1–B3** (no twin, and no ordering constraint at all).
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
which loses its Cython at Task 4.6. See
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
  `BR_ETAP_TO_A_A · boost_delta_function(M_η′/2, …)` — a constant times
  a function of `hazma/_utils/boost.pyx`, which is still live and which
  Task 2 is already capturing.
- **B2 (φ):** both line energies are closed forms,
  `(M_φ² − m²)/(2 M_φ)`; the delta is the two boosted line terms
  recomputed there minus the two the corpus stored.

Where the closed form is analytic, add an `mpmath` reference in the
shape of `test/parity/reference.py` rather than trusting either
implementation — that file is the precedent, and
`projects/cython-to-rust/task-notes/README.md` records the pattern
settling a comparable question in an afternoon with no build.

**Deliverable / gate:** For each of B1–B3, a declaration in
`test/parity/deltas.py` and a test that the declared model reproduces
the **shipped** value from the corrected form plus the named defect —
which is falsifiable without either the repaired Rust or a Cython twin,
and is what makes the model non-circular. Recovering the deleted sources
for review is `git show 665aed5:<path>` (B1, B2) and
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
budget; the sign is one-signed and upward at every
declared position, which is the follow-up's own characterization and is
asserted rather than asserted-about. Every other corpus case unchanged
against its original stored array.

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

**Deliverable / gate:** Declared deltas on all **7** cases the A2 row of
[`references/defect-blast-radius.md`](references/defect-blast-radius.md)
names — `spectra.photon.muon`, `spectra.photon.charged_pion`,
`spectra.photon.charged_rho`, `spectra.photon.neutral_rho`,
`mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum`,
`mediator_spectra.vector.photon.dnde_decay_v`, and
`mediator_spectra.vector.photon.dnde_decay_v_pt`. The `_pt` variants are
separate corpus cases and are exactly what a "both mediators"
enumeration drops; the count is the check. The repaired value reproduces
Task 2's oracle. The endpoint
invariant that Task 4.3 wrote —
`the_in_flight_form_is_the_boost_integral_of_the_rest_frame_form` —
must still hold, and now holds over the extra 0.25 MeV.

### Task 8: Repair A3 — the charged-pion forward cone

**Objective:** Stop `qagp` from terminating successfully at `0.0` because
every abscissa fell outside the integrand's support.

**Scope / implementation notes:** `rust/src/kernels/photon_pion.rs`,
`CHARGED_PION_QUAD`. The integrand is nonzero only where the
pion-rest-frame photon energy stays under `ENG_GAM_MAX_PIRG = 69.783`
MeV; the fix is to integrate over that window rather than over all of
`cos θ` and hope the adaptive rule finds it. Depends on Task 7 (this
kernel boosts the muon spectrum). Partition the verification grid by
scipy's own convergence verdict rather than picking one tolerance —
`projects/cython-to-rust/task-notes/README.md` records that lesson from
this exact kernel, and PR #68's two CI rounds are why.

**Deliverable / gate:** Declared deltas on all **6** cases the A3 row of
[`references/defect-blast-radius.md`](references/defect-blast-radius.md)
names — `spectra.photon.charged_pion`, `spectra.photon.charged_rho`,
`spectra.photon.neutral_rho`, and the same three mediator photon cases
Task 7 lists. A3 is a strict subset of A2, so every case here is one
Task 7 has already opened, and `rules.md` rule 7's no-overlap
requirement binds on *positions* rather than on cases. The specific
figure the
follow-up pins: `dnde_photon_charged_pion(900, 1396)` moves from `0.0` to
`3.586e-07` MeV⁻¹. A test that no stored zero in that case survives
repair *except* the ones outside the kinematic support — the difference
between the two is the whole defect, and a declaration that covers both
would hide it.

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

### Task 12: Close — aggregate the drift and ship the bump

**Objective:** One `CHANGELOG.md` entry naming this slug, carrying every
measured shift, and the `minor` bump.

**Scope / implementation notes:** The per-repair figures accumulate in
`task-notes/README.md`'s "Numerical impact so far" as each task lands;
this task aggregates rather than reconstructs. Re-check the level against
the aggregate before bumping — seven deliberate corrections to published
spectra with no API change is `minor`, and nothing in Tasks 4–10 should
have raised it, but the check is the point.

**Deliverable / gate:** `scripts/agents/preflight.sh --closing` green;
`PLAN.md` `status: Complete`; the seven follow-ups moved to
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
  `projects/cython-to-rust/task-notes/README.md`'s "Numerical impact so
  far", which is where each defect was first measured.
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
`VERSION` in `hazma/__init__.py` per the `version_bump:` frontmatter and
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
