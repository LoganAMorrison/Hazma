# Task 2: Axial tensors and LO three pion amplitudes

**Date:** 2026-10-12
**Project:** visible-axial-spectra
**Status:** Complete
**Plan References:** Task 2 in projects/visible-axial-spectra/PLAN.md
**Related ADRs:** ADR-0001 (project-scoped)
**Depends On:** Task 1

## Objective

Implement the bounded public contract and validate its physics and numerical
normalization before building the model and spectrum layers.

## Exit Criteria

- [x] Covariant source projections pass an independent fermion trace oracle.
- [x] Charged and neutral currents satisfy independently derived Ward identities.
- [x] Independent phase-space and pinned research reference checks agree.
- [x] Thresholds, shapes, flavor support and declared energy range are enforced.
- [x] Full preflight and documentation checks pass.

## Inputs Reviewed

- AGENTS and the execution, environment and preflight guidance.
- Project PLAN and ADR-0001.
- Maintained public phase-space and vector interfaces.

## Findings

The implementation is additive. Generic neutral axial currents and LO
isovector channel support must remain distinct.

## Decisions and Implementation Notes

ADR-0001 records units, source normalization and tensor conventions.

## Files Changed

- `hazma/form_factors/axial/__init__.py`
- `hazma/form_factors/axial/_base.py`
- `hazma/form_factors/axial/_tensors.py`
- `hazma/form_factors/axial/_three_pi.py`
- `test/form_factors/axial/test_tensors.py`
- `test/form_factors/axial/test_three_pi.py`

## Verification

The final focused command was:

```sh
.venv/bin/python -m pytest -n 0 test/form_factors/axial -q
```

```text
183 passed in 0.68s
```

The complete gate was run with the worktree `.venv/bin` on PATH:

```sh
scripts/agents/preflight.sh --paths "hazma/form_factors/axial test/form_factors/axial"
```

The Markdown list supplied to `--md` is recorded verbatim in
`projects/visible-axial-spectra/task-notes/preflight.txt`.

```text
2739 passed, 16 skipped, 10 warnings, 127 subtests passed in 53.70s
RESULT: PASS
```

Black, isort, configured Ruff, Cargo formatting, Clippy, Cargo tests,
import smoke and Markdown checks all passed. No version bump was applied:
the project remains open for later model and spectrum tasks.

## Numerical impact

New APIs are additive. Existing values on the recorded photon, positron,
neutrino and vector-width grids are bit-identical in the captured environment:

```sh
.venv/bin/python projects/visible-axial-spectra/references/check_public_impact.py
```

The command derives 164 output samples and maximum absolute difference 0.0;
its full output is in `projects/visible-axial-spectra/task-notes/public-impact.json`.
The energy grid and captured values are stored in
`projects/visible-axial-spectra/references/public-values-baseline.json`.

## Open Questions

Later source matching and pole providers are separate planned tasks.

## Plan Impact

Project-scoped ADR-0001 and the canonical task shape.

## Stale-state sweep

Commands and sorted captures are in
`projects/visible-axial-spectra/task-notes/stale-state.json` and its
referenced text files. Saved capture files are excluded from subsequent
sweeps to avoid recursively matching their own copied output. All other
identifier matches are KEPT public declarations, implementation, tests or
documentation. Forward-looking matches are KEPT open-project/later-task
or immutable-template records. No existing identifier was renamed or removed.

The count sweep derives exports and files from the tree and tests from the
executed suite. Existing values are measured on the recorded grid above.

| Exit criterion | Evidence |
| --- | --- |
| Source trace and reconstruction | test_sources generator and singlet tests |
| Units, scales and immutability | test_units and test_sources boundary tests |
| Tensor normalization and covariance | test_tensors independent gamma trace and boost tests |
| Current Ward and field coordinates | test_three_pi independent contact and pole tests |
| Phase measure and widths | test_three_pi ThreeBody and chiral-limit tests |
| Threshold, shape and support | test_three_pi scalar/array, flavor and energy guards |
| Repository validation | Literal full preflight summary above |

The task is Complete. The wider project and remaining model/spectrum
work retain their live statuses in the working-memory table.

## Handoff to Next Task

Read PLAN, ADR-0001 and the LO derivation. Next is the coherent longitudinal
meson embedding. Preserve source/current normalization; arbitrary singlet
providers and secondary yields are still separate planned tasks.
