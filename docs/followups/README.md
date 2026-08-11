# Follow-ups

The durable backlog of ideas and deferred work that hasn't been promoted
to a `projects/<slug>/` plan yet.

Open items live in [`todo/`](todo/); resolved items move to
[`done/`](done/), so `ls todo/` is the live backlog at a glance. Items
move between the two directories; they are never deleted — the historical
reasoning is worth keeping.

The full lifecycle (create → resolve → repoint inbound links), when to
add one, and why this is not GitHub issues, live in
[`../workflow.md#follow-ups`](../workflow.md#follow-ups).

## Creating one

```sh
cp docs/followups/_template.md docs/followups/todo/<slug>.md
# fill in the fields, then add a row to the Open table below
```

## Open

| Item | Added | Source | Scope |
| --- | --- | --- | --- |
| [Citation checker skips deleted in-repo files](todo/citation-checker-skips-deleted-inrepo-files.md) | 2026-08-05 | PR #42 review | cross-cutting |
| [remaining `sqrt(kallen_lambda(...))` call sites](todo/kallen-under-sqrt-remaining-call-sites.md) | 2026-08-05 | carved out of the `cross_section_prefactor` fix | cross-cutting |
| [redundant `hazma.utils` helpers kept out of the public surface](todo/utils-public-surface-redundant-helpers.md) | 2026-08-05 | docs audit of `utils.rst` | cross-cutting |
| [`preflight.sh` isort/ruff gates are red on the trunk](todo/preflight-isort-ruff-red-on-trunk.md) | 2026-08-05 | cython-to-rust Task 0.5 | cross-cutting |
| [markdownlint was never run over `.claude/skills/`](todo/markdownlint-skips-skill-file-shapes.md) | 2026-08-06 | PR #48 review round 1 | cross-cutting |
| [sdist ships cythonized `*.c`, `docs/`, `test/`](todo/sdist-ships-generated-c-and-docs.md) | 2026-08-06 | cython-to-rust Task 0.4 | cross-cutting |
| [parity corpus pins ill-conditioned points](todo/parity-corpus-pins-ill-conditioned-points.md) | 2026-08-07 | cython-to-rust Task 1.3 (PR #52) | project |
| [model spectrum dicts reject scalar energies](todo/model-spectra-reject-scalar-energies.md) | 2026-08-08 | cython-to-rust Task 1.4 | cross-cutting |
| [positron spectra return `nan` at the legacy `MASS_E`](todo/positron-spectrum-nan-at-legacy-electron-mass.md) | 2026-08-08 | cython-to-rust Task 1.4 | cross-cutting |
| [the boost integral mis-covers its window at both ends](todo/boost-integral-drops-last-interior-cell.md) | 2026-08-10 | cython-to-rust Task 3.4 | cross-cutting |

## Promoted / Done / Pruned

| Item | Status | Resolution |
| --- | --- | --- |
| [markdownlint config for templates](done/markdownlint-config-for-templates.md) | done | Committed [`.markdownlint.jsonc`](../../.markdownlint.jsonc) encoding the repo's shapes (frontmatter-title phase files, `<placeholder>` notation, wide fact tables); all 18 inline pragmas removed. `docs/` + `projects/` errors 132 → 0. |
| [`black` pin diverges between pyproject and CI](done/black-pin-divergence-pyproject-vs-ci.md) | done | [PR #40](https://github.com/LoganAMorrison/Hazma/pull/40) — pins moved to a single PEP 735 `lint` dependency group that CI installs; repo reformatted with black 26.x (33 files). |
| [`msqrd`-driven Monte-Carlo FSR generator](done/msqrd-driven-fsr-generator.md) | done | `hazma.spectra.dnde_photon_fsr` (ADR-0001, [PR #41](https://github.com/LoganAMorrison/Hazma/pull/41)) |
| [`cross_section_prefactor` threshold cancellation](done/cross-section-prefactor-threshold-cancellation.md) | done | `hazma.utils.two_body_momentum` — factored Källén, heavier mass first; ≤4.4e-16 relative to threshold. Remaining λ-under-sqrt sites carved out to [their own follow-up](todo/kallen-under-sqrt-remaining-call-sites.md). |
| [`WIDTH_K`/`WIDTH_PI` exponent bug](done/legacy-parameters-width-exponent-bug.md) | done | Both names deleted from `hazma/_utils/legacy_parameters.pxd` (no consumer; `constants.pxd` is canonical). No published value moves. |
