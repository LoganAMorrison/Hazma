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
| [model spectrum dicts reject scalar energies](todo/model-spectra-reject-scalar-energies.md) | 2026-08-08 | cython-to-rust Task 1.4 | cross-cutting |
| [positron spectra return `nan` at the legacy `MASS_E`](todo/positron-spectrum-nan-at-legacy-electron-mass.md) | 2026-08-08 | cython-to-rust Task 1.4 | cross-cutting |
| [the boost integral mis-covers its window at both ends](todo/boost-integral-drops-last-interior-cell.md) | 2026-08-10 | cython-to-rust Task 3.4 | cross-cutting |
| [the muon positron spectrum divides by its normalization](todo/positron-muon-spectrum-normalization-inverted.md) | 2026-08-11 | cython-to-rust Task 4.1 | cross-cutting |
| [the η′ two-photon line carries one photon instead of two](todo/eta-prime-two-photon-line-missing-factor-two.md) | 2026-08-12 | cython-to-rust Task 4.2 | cross-cutting |
| [the φ photon lines sit at the daughter meson's energy](todo/phi-photon-lines-use-the-daughter-meson-energy.md) | 2026-08-12 | cython-to-rust Task 4.2 | cross-cutting |
| [the muon photon spectrum's rest frame stops short of the endpoint](todo/photon-muon-rest-frame-endpoint-uses-the-wrong-power-of-r.md) | 2026-08-16 | cython-to-rust Task 4.3 | commit |
| [the charged-pion photon spectrum returns zero in the forward cone](todo/charged-pion-photon-spectrum-misses-the-forward-cone.md) | 2026-08-17 | cython-to-rust Task 4.4 | cross-cutting |
| [both rho photon spectra return the boost integrand at rest](todo/rho-rest-frame-branch-returns-the-integrand.md) | 2026-08-18 | cython-to-rust Task 4.5 | cross-cutting |
| [four scalar elastic cross sections cancel away every significant bit](todo/scalar-elastic-cross-sections-cancel-in-atan-difference.md) | 2026-08-18 | closing the parity-corpus follow-up | cross-cutting |

## Promoted / Done / Pruned

| Item | Status | Resolution |
| --- | --- | --- |
| [parity corpus pins ill-conditioned points](done/parity-corpus-pins-ill-conditioned-points.md) | done | `test/parity/stability.py` masks the 494 stored positions whose values are cancellation residue, established against `test/parity/reference.py` (the same closed forms at 60 digits) rather than against platform disagreement; `tolerances.PLATFORM_EXACT_RTOL` gives the `EXACT` class an off-libm budget and `tolerances.zero_floor` handles stored exact zeros. CI's `--ignore=test/parity` scoping removed; `pytest test/parity` is **635 passed, 1 skipped** on macOS/arm64, Linux/aarch64 and Linux/x86_64. The underlying kernel defect is carved out to [its own follow-up](todo/scalar-elastic-cross-sections-cancel-in-atan-difference.md). |
| [`test_core_interp.py` scoped its NumPy oracle with a probe](done/interp-oracle-scoped-by-an-unsound-probe.md) | done | Probe replaced by `ON_THE_CAPTURING_PLATFORM` + a measured, peak-scaled `OFF_PLATFORM_BUDGET = 1e-12`; `TestFusedArithmetic` rewritten against a full Python transcription of `rust/src/interp.rs`. Module goes from `24 passed, 9 skipped` to **`42 passed, 0 skipped`** on linux/amd64. |
| [markdownlint config for templates](done/markdownlint-config-for-templates.md) | done | Committed [`.markdownlint.jsonc`](../../.markdownlint.jsonc) encoding the repo's shapes (frontmatter-title phase files, `<placeholder>` notation, wide fact tables); all 18 inline pragmas removed. `docs/` + `projects/` errors 132 → 0. |
| [`black` pin diverges between pyproject and CI](done/black-pin-divergence-pyproject-vs-ci.md) | done | [PR #40](https://github.com/LoganAMorrison/Hazma/pull/40) — pins moved to a single PEP 735 `lint` dependency group that CI installs; repo reformatted with black 26.x (33 files). |
| [`msqrd`-driven Monte-Carlo FSR generator](done/msqrd-driven-fsr-generator.md) | done | `hazma.spectra.dnde_photon_fsr` (ADR-0001, [PR #41](https://github.com/LoganAMorrison/Hazma/pull/41)) |
| [`cross_section_prefactor` threshold cancellation](done/cross-section-prefactor-threshold-cancellation.md) | done | `hazma.utils.two_body_momentum` — factored Källén, heavier mass first; ≤4.4e-16 relative to threshold. Remaining λ-under-sqrt sites carved out to [their own follow-up](todo/kallen-under-sqrt-remaining-call-sites.md). |
| [`WIDTH_K`/`WIDTH_PI` exponent bug](done/legacy-parameters-width-exponent-bug.md) | done | Both names deleted from `hazma/_utils/legacy_parameters.pxd` (no consumer; `constants.pxd` is canonical). No published value moves. |
