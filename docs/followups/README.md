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
| [model spectrum dicts reject scalar energies](todo/model-spectra-reject-scalar-energies.md) | 2026-08-08 | cython-to-rust Task 1.4 | cross-cutting |
| [the boost integral mis-covers its window at both ends](todo/boost-integral-drops-last-interior-cell.md) | 2026-08-10 | cython-to-rust Task 3.4 | cross-cutting |
| [the muon positron spectrum divides by its normalization](todo/positron-muon-spectrum-normalization-inverted.md) | 2026-08-11 | cython-to-rust Task 4.1 | cross-cutting |
| [the η′ two-photon line carries one photon instead of two](todo/eta-prime-two-photon-line-missing-factor-two.md) | 2026-08-12 | cython-to-rust Task 4.2 | cross-cutting |
| [the φ photon lines sit at the daughter meson's energy](todo/phi-photon-lines-use-the-daughter-meson-energy.md) | 2026-08-12 | cython-to-rust Task 4.2 | cross-cutting |
| [the muon photon spectrum's rest frame stops short of the endpoint](todo/photon-muon-rest-frame-endpoint-uses-the-wrong-power-of-r.md) | 2026-08-16 | cython-to-rust Task 4.3 | commit |
| [the charged-pion photon spectrum returns zero in the forward cone](todo/charged-pion-photon-spectrum-misses-the-forward-cone.md) | 2026-08-17 | cython-to-rust Task 4.4 | cross-cutting |
| [both rho photon spectra return the boost integrand at rest](todo/rho-rest-frame-branch-returns-the-integrand.md) | 2026-08-18 | cython-to-rust Task 4.5 | cross-cutting |
| [four scalar elastic cross sections cancel away every significant bit](todo/scalar-elastic-cross-sections-cancel-in-atan-difference.md) | 2026-08-18 | closing the parity-corpus follow-up | cross-cutting |
| [the review-lessons ledger is past its working-set cap](todo/lessons-ledger-over-its-working-set-cap.md) | 2026-08-19 | PR #72 review round 1 | cross-cutting |
| [the charged pion's `pi -> e nu` neutrino line is added twice](todo/neutrino-pion-electron-line-counted-twice.md) | 2026-08-20 | cython-to-rust Task 4.6 | cross-cutting |
| [two vector cross sections raise `TypeError` at `e_cm = 2 m_x`](todo/vector-cross-sections-raise-at-the-two-mx-threshold.md) | 2026-08-20 | cython-to-rust Task 5.1 | cross-cutting |
| [`thermal_cross_section` returns its integrator's initial estimate](todo/thermal-cross-section-quadrature-never-converges.md) | 2026-08-20 | cython-to-rust Task 5.1 | cross-cutting |
| [mediator spectra return 0.0 for an unrecognised mode string](todo/mediator-spectra-accept-unknown-mode-strings.md) | 2026-08-23 | cython-to-rust Task 6.1 | cross-cutting |
| [`[mutation-harness-poisons-its-own-baseline]` is cited but not in the ledger](todo/lessons-ledger-missing-the-mutation-harness-class.md) | 2026-08-23 | cython-to-rust Task 6.2 | cross-cutting |
| [mediator positron line misses the electron velocity](todo/mediator-positron-line-misses-the-electron-velocity.md) | 2026-08-27 | cython-to-rust Task 6.3 | cross-cutting |
| [moved follow-ups leave dangling inbound paths](todo/moved-followups-leave-dangling-inbound-paths.md) | 2026-08-27 | PR #81 review | cross-cutting |

## Promoted / Done / Pruned

| Item | Status | Resolution |
| --- | --- | --- |
| [`pip install -e .` builds `hazma._core` unoptimized](done/editable-installs-build-the-rust-extension-in-debug.md) | done | cython-to-rust Task 7.1 — resolved by the backend swap rather than by a choice. The debug default was `setuptools_rust.build_rust`'s `debug = self.inplace or self.debug`; maturin's PEP 517 hooks build release unconditionally, and only the `maturin develop` CLI (which this repo never invokes) defaults to debug. Measured after the cutover: `uv pip install -e .` leaves `rust/target/release/` and no `debug/`, and `thermal_cross_section(x=0.5)` runs at 35.8 us from the editable tree against the file's 1866 us debug figure. The switch also corrected this file's "the two profiles are numerically identical" risk note, which held for the functions Task 5.3 measured and not in general: under release the mediator table grids sit one ulp from `numpy.logspace` at 5 of 1000 sampled abscissae. No published value moved -- wheels have always been release builds -- but `test/test_core_mediator_tables.py`'s grid comparison had encoded the debug values as exact, and moved to its own derived one-ulp budget. |
| [sdist ships cythonized `*.c`, `docs/`, `test/`](done/sdist-ships-generated-c-and-docs.md) | done | cython-to-rust Task 7.1 — folded into the maturin cutover, the window this file named. `MANIFEST.in` is gone; `[tool.maturin]` answers all four items. The `*.c` question is moot (no transpiler since Task 6.4), `docs/`/`test/`/`notebooks/` are dropped, the sdist is built from the git index so working-directory state cannot reach it, and the `*.pyd` typo went with the whole `[tool.setuptools.package-data]` block. Measured on a clean tree: **415 files to 264**. Verified by source-installing the sdist into a fresh CPython 3.10 venv and import-smoking from outside the repo. |
| [the oracle roster has no restore revision for the deleted mediator spectrum `.pyx`](done/oracle-restore-revisions-for-the-mediator-decay-pyx.md) | done | cython-to-rust Task 6.4 — `defects.RESTORED_SOURCES` goes from 13 entries to 29 and every case is now covered. The four mediator modules resolve to `7594761^` (Task 6.2) and `c384aff^` (Task 6.3), both merged by then. The twelve files Task 6.4 itself deleted could not use that spelling for the reason 6.2 and 6.3 could not — a task cannot know its own commit — so they are pinned to a revision that already existed, `1b022d4`, where all of them are present in final form; `git show <rev>:<path>` does not care which spelling it is given. Verified by resolving all 29 against git. The roster now also lists the headers and cimported twins a restore has to compile against, and `oracles/README.md` says that a re-capture must restore `setup.py` and `pyproject.toml` too. |
| [positron spectra return `nan` at the legacy `MASS_E`](done/positron-spectrum-nan-at-legacy-electron-mass.md) | done | cython-to-rust Task 6.3 — the `nan` was a clang FMA contraction of `sqrt(eng_p*eng_p - me*me)`, whose radicand is the upward rounding of `me*me` (1.45e-17) and so negative at `eng_p == m_e`, not a consequence of the two `MASS_E` tables diverging. Resolved by the file's second option: `mediator_decay_positron.rs`'s `momentum` keeps the fused spelling and clamps the radicand at zero, moving that one double from `nan` to `0.0` and no other value. Consolidating the tables stays open under `rules.md` rule 4. Pinned in `test/test_core_mediator_positron.py::TestTheThresholdSingularity` rather than in the corpus, which rule 2 forbids. |
| [parity corpus pins ill-conditioned points](done/parity-corpus-pins-ill-conditioned-points.md) | done | `test/parity/stability.py` masks the 494 stored positions whose values are cancellation residue, established against `test/parity/reference.py` (the same closed forms at 60 digits) rather than against platform disagreement; `tolerances.PLATFORM_EXACT_RTOL` and `PLATFORM_SPECFUN_RTOL` give those two classes an off-libm budget and `tolerances.zero_floor` handles the four declared stored zeros a change of libm moves. CI's `--ignore=test/parity` scoping removed; `pytest test/parity` is **637 passed, 1 skipped** on macOS/arm64, Linux/aarch64 and Linux/x86_64. The underlying kernel defect is carved out to [its own follow-up](todo/scalar-elastic-cross-sections-cancel-in-atan-difference.md). |
| [`test_core_interp.py` scoped its NumPy oracle with a probe](done/interp-oracle-scoped-by-an-unsound-probe.md) | done | Probe replaced by `ON_THE_CAPTURING_PLATFORM` + a measured, peak-scaled `OFF_PLATFORM_BUDGET = 1e-12`; `TestFusedArithmetic` rewritten against a full Python transcription of `rust/src/interp.rs`. Module goes from `24 passed, 9 skipped` to **`42 passed, 0 skipped`** on linux/amd64. |
| [markdownlint config for templates](done/markdownlint-config-for-templates.md) | done | Committed [`.markdownlint.jsonc`](../../.markdownlint.jsonc) encoding the repo's shapes (frontmatter-title phase files, `<placeholder>` notation, wide fact tables); all 18 inline pragmas removed. `docs/` + `projects/` errors 132 → 0. |
| [`black` pin diverges between pyproject and CI](done/black-pin-divergence-pyproject-vs-ci.md) | done | [PR #40](https://github.com/LoganAMorrison/Hazma/pull/40) — pins moved to a single PEP 735 `lint` dependency group that CI installs; repo reformatted with black 26.x (33 files). |
| [`msqrd`-driven Monte-Carlo FSR generator](done/msqrd-driven-fsr-generator.md) | done | `hazma.spectra.dnde_photon_fsr` (ADR-0001, [PR #41](https://github.com/LoganAMorrison/Hazma/pull/41)) |
| [`cross_section_prefactor` threshold cancellation](done/cross-section-prefactor-threshold-cancellation.md) | done | `hazma.utils.two_body_momentum` — factored Källén, heavier mass first; ≤4.4e-16 relative to threshold. Remaining λ-under-sqrt sites carved out to [their own follow-up](todo/kallen-under-sqrt-remaining-call-sites.md). |
| [`WIDTH_K`/`WIDTH_PI` exponent bug](done/legacy-parameters-width-exponent-bug.md) | done | Both names deleted from `hazma/_utils/legacy_parameters.pxd` (no consumer; `constants.pxd` is canonical). No published value moves. |
