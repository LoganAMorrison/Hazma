# Visible axial spectra working memory

**Status:** In Progress
**Project:** visible-axial-spectra
**Date:** 2026-10-12

## Tasks

| # | Task | Depends on | Status | Task note |
| --- | --- | --- | --- | --- |
| 1 | Current and unit contracts | — | Complete | `task-1-contracts.md` |
| 2 | Axial tensors and LO three pion amplitudes | 1 | Complete | `task-2-lo-three-pion.md` |
| 3 | Coherent longitudinal poles | 1 | Not started | — |
| 4 | Rates and mediator model | 3 | Not started | — |
| 5 | Rest frame yields | 4 | Not started | — |
| 6 | Polarized cascades | 5 | Not started | — |
| 7 | Resolved photons | 6 | Not started | — |
| 8 | Observation adapters | 7 | Not started | — |
| 9 | Fitted input overlays | 8 | Not started | — |

## Findings and decisions

- [ADR-0001](../adrs/ADR-0001-current-and-tensor-contracts.md) defines the
  normalization, units and source-longitudinal versus helicity distinction.
- Exact isospin LO three-pion coverage is distinct from a fitted physical
  charged/neutral mass prescription or a general singlet source.

## Numerical impact so far

The first cluster adds public primitives. The recorded existing-output grid
contains 164 values and is bit-identical after integration. Reproduce it with
`.venv/bin/python projects/visible-axial-spectra/references/check_public_impact.py`.

## Verification

The focused axial suite passed 183 tests. Full preflight passed 2,739 tests,
with 16 skips and 127 subtests, and all Rust/format/import gates. Exact
commands and literal summaries are in the completed task notes and
`projects/visible-axial-spectra/task-notes/preflight.txt`.

## Open questions

Later tasks must reconcile external photon source prescriptions and identify
missing singlet input directions. These do not block LO isovector currents.

## Handoff to next task

Read PLAN and ADR-0001, then the task notes. Neutral poles require the axial
embedding and cannot use a spin-zero mediator Breit-Wigner replacement.
