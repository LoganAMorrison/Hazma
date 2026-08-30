# Consolidate the divergent constants tables

- **Added:** 2026-08-29
- **Source:** `projects/cython-to-rust/learnings/project-retrospective.md` §5
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** the cython-to-rust project's `rules.md` rule 4
  forbade touching this while any kernel was being ported, because a
  ported kernel had to read the exact constant its Cython source read for
  bit-parity to mean anything. That project closed on 2026-08-29, so the
  rule no longer binds. What does bind is the parity corpus: this is a
  **declared numerical change** and needs the same treatment as the
  defects in `projects/parity-pinned-defect-repair/` — a per-array delta
  asserted against the committed corpus arrays, not a regeneration.

## Why

Hazma carries three different fine-structure constants and two mass
tables that disagree on twelve names, and which one a given spectrum sees
depends on which Cython header its `.pyx` happened to `include`. That
accident is now frozen into `rust/src/constants.rs`, which reproduces the
split faithfully (`pdg` for everything under `hazma/spectra/**`, `legacy`
for the mediator spectrum kernels, and `derived::photon_pion` reading
both) because rule 4 required it to.

The three α values are `1/137.035999084` (the `pdg` table,
pre-CODATA-2022), `1/137` (the `legacy` table) and `1/137.04`
(`hazma/parameters.py`, pure Python and reached by a third set of
callers). Nothing about the physics justifies the split; it is a
historical artifact of two headers growing separately. A user comparing a
mediator spectrum against a meson spectrum is comparing numbers computed
with different values of α, and nothing in the API says so.

## What

Pick one table, state its provenance (which PDG edition, which CODATA
release), and move every consumer onto it. The work is:

- Reconcile `hazma_core::constants::{pdg, legacy}` — twelve names
  disagree — and decide `derived::photon_pion`, which legitimately reads
  from both today.
- Reconcile `hazma/parameters.py`'s third α with whichever survives.
- Record the provenance that
  `projects/cython-to-rust/learnings/phase-03-numerics-foundation.md` §5
  notes was never written down: the `± uncertainty` annotations are the
  only surviving evidence of which edition each value came from. Do
  **not** resolve this by re-sourcing values from a current PDG — that
  would silently move numbers a second time, for a second reason.
- Declare the resulting shift per public entry point, as a `Changed`
  entry with magnitudes, and assert it as a delta against the corpus
  arrays.

Expect the shift to be small but real: the two α values differ by
2.6e-4 relative, and a spectrum carrying `α²` moves by twice that.

## Entry points

- `rust/src/constants.rs` — the `pdg` / `legacy` / `derived` split
- `hazma/parameters.py:205` — the third α
- `test/test_core_constants.py` — the bit-parity assertions that pinned
  the split
- `projects/cython-to-rust/rules.md` rule 4 — the rule that deferred this
- `projects/cython-to-rust/learnings/phase-03-numerics-foundation.md` §5
- `projects/parity-pinned-defect-repair/PLAN.md` — the declared-delta
  mechanism to reuse

## Risks / open questions

- **Sequencing against `parity-pinned-defect-repair`.** Both move
  published numbers against the same corpus arrays. Landing them in
  either order is fine; landing them concurrently means two deltas
  against one baseline and is not.
- **Which table wins is a physics call, not a mechanical one.** The
  `legacy` table's `1/137` is plainly a placeholder, but some of its
  twelve divergent names may have been chosen to match a published
  reference the mediator models were validated against. Check before
  overwriting.
