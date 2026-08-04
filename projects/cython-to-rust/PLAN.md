---
status: In Progress
phased: true
version_bump: major
deliverable: Hazma's compiled layer rebuilt in Rust (PyO3, one abi3 `hazma._core` extension, maturin-built), with zero Cython remaining and a permanent parity-test corpus
created: 2026-08-03
---

# Project: cython-to-rust

**Structure:** Phased — see [`phases/`](phases/).

## Goal

Replace Hazma's Cython layer with Rust + PyO3 packaged by maturin,
because the layer's cost is toolchain churn rather than physics
development (3 kernel commits in 3 years, two of them forced
migrations). Ship the same public API and — within declared, measured
budgets — the same numbers, from a single abi3 extension that ends the
per-CPython wheel matrix, the scipy build-ABI pin, and the Cython 3
deprecation debt. Full analysis: the August 2026 assessment
(`references/` distills its durable facts).

## Scope

**In scope:**

- Deleting the ~6,500 lines of dead/broken Cython and its data
  (Phase 00) — worth doing even if the port stopped there.
- A golden parity corpus over all 41 consumed compiled entry points
  (of 43 public defs — the two unimported `sigma_xx_to_all` exports
  are dropped in Phase 05, not ported), wired into pytest + CI
  (Phase 01).
- Porting the live surface (20 extensions: 19 kernel modules +
  `_utils.boost`) to one Rust cdylib,
  `hazma._core`: spectra kernels, boost/interp/constants foundation,
  QUADPACK-subset integrator, cephes-lineage special functions,
  mediator cross sections + thermal ⟨σv⟩, mediator spectrum modules
  (Phases 02–06).
- Packaging cutover to maturin with abi3 wheels; docs/CI sweep;
  project close with aggregated numerical-drift declaration (Phase 07).

**Out of scope:**

- Any physics change, new channel, or API addition.
- Consolidating the divergent constants tables (deliberately preserved
  bit-for-bit — see `rules.md` rule 4; consolidation is a follow-up).
- Pure-Python subsystems that happen to be numeric (`hazma/phase_space/`,
  form factors, relic-density ODEs, spline utilities) — they stay on
  NumPy/SciPy.
- Windows / linux-aarch64 wheel support (recorded as a cheap follow-up;
  Phase 07 Task 7.2 makes the call explicitly).
- Performance work beyond what parity-preserving redesign yields
  (measured, not chased — `rules.md` rule 12).

## Numerical impact

Intended: none beyond tolerance-level drift. Two knowable exceptions,
both declared: (1) quadrature moves from scipy's QUADPACK binding to an
in-tree QUADPACK port — corpus budgets start at 1e-8 relative for
quad-backed functions and tighten after measurement (closed-form
kernels: ≤1e-13); (2) Cython-`assert` edge guards become unconditional
raises (behavior tightening at invalid inputs only).

`version_bump: major` is driven by API removals, not by numbers:
Phase 00 deletes `hazma/deprecated/rambo.py` (any removal from
`hazma/deprecated/` is `major` per `docs/versioning.md` and
`AGENTS.md`) and, per ADR-0003 (proposed), removes the
broken-on-import `hazma.gamma_ray` module. The running numerical
record lives in `task-notes/README.md` ("Numerical impact so far");
the closing CHANGELOG aggregates it per function.

## Orientation

| Reference | What it holds |
| --- | --- |
| [`references/cython-inventory.md`](references/cython-inventory.md) | Dead-code map with evidence, live surface + entry-point tables, C-level dependency DAG, data files, audit bug list |
| [`references/numerics-replacements.md`](references/numerics-replacements.md) | quad call-site/tolerance table, specfun facts + conventions, `np.interp`/boost-integral specs, cyphus-crate assessment, dispatch contract |
| [`adrs/ADR-0001-rust-pyo3-maturin-over-pybind11.md`](adrs/ADR-0001-rust-pyo3-maturin-over-pybind11.md) | Framework choice (Accepted) |
| [`adrs/ADR-0002-license-clean-numerics.md`](adrs/ADR-0002-license-clean-numerics.md) | GSL/GPL boundary, cephes + netlib-QUADPACK provenance (**Proposed — sign-off gates Phase 03**) |
| [`adrs/ADR-0003-remove-gamma-ray-module.md`](adrs/ADR-0003-remove-gamma-ray-module.md) | Remove broken `hazma.gamma_ray` (**Proposed — sign-off gates Task 0.2/0.5**) |
| [`rules.md`](rules.md) | Parity discipline, constants bit-parity, licensing, Rust conventions |

## Phases

Canonical per-task shape lives in each phase file; live status lives in
`task-notes/README.md` (project) and `task-notes/phase-XX/README.md`
(per phase). Estimates are focused dev-days from the analysis; the
total landed at 21–32 days across ~33 tasks.

| # | Phase | File | Days | Delivers |
| --- | ------- | ------ | ------ | ---------- |
| 00 | Dead-code purge | [`phases/phase-00-dead-code-purge.md`](phases/phase-00-dead-code-purge.md) | 1–2 | −6,500 lines, 32→20 extensions, zero C++, `gamma_ray` removal (ADR-0003) |
| 01 | Golden parity corpus | [`phases/phase-01-parity-corpus.md`](phases/phase-01-parity-corpus.md) | 2–3 | Pinned reference arrays for all 41 consumed entry points, one pytest gate |
| 02 | Rust scaffold | [`phases/phase-02-rust-scaffold.md`](phases/phase-02-rust-scaffold.md) | 1–2 | `hazma._core` (abi3) building beside Cython via setuptools-rust |
| 03 | Numerics foundation | [`phases/phase-03-numerics-foundation.md`](phases/phase-03-numerics-foundation.md) | 3–5 | constants, spence/K-Bessels, QUADPACK port, interp, boost, dispatch |
| 04 | Spectra kernels | [`phases/phase-04-spectra-kernels.md`](phases/phase-04-spectra-kernels.md) | 4–6 | 16 entry points swapped; twins deleted (4 capi survivors defer to 06) |
| 05 | Mediator cross sections | [`phases/phase-05-mediator-cross-sections.md`](phases/phase-05-mediator-cross-sections.md) | 2–3 | 16 kernels + 2 thermal ⟨σv⟩ swapped, 2 dead exports dropped; relic validation |
| 06 | Mediator spectra | [`phases/phase-06-mediator-spectra.md`](phases/phase-06-mediator-spectra.md) | 3–4 | Table-struct redesign; last Cython deleted |
| 07 | Cutover + close | [`phases/phase-07-cutover.md`](phases/phase-07-cutover.md) | 2–3 | maturin backend, 2 abi3 wheels, docs sweep, version bump + CHANGELOG |

Ordering constraints: 00 → 01 → 02 → 03 → {04, 05} → 06 → 07. Phase 05
shares no files with 04 and may run in parallel with it. Within 04 the
cimport DAG in the inventory reference governs task order. Phase 06
Task 6.4 is the only place the four capi-survivor spectra extensions
and `_utils` headers are deleted.

## Task Details

Per-task objective, dependencies, and exit criteria live in the phase
files (`phases/phase-XX-*.md`, "Tasks" sections) — 33 tasks total.
This PLAN intentionally holds only the phase table; do not duplicate
task blocks here.

## Dependencies

- Requires: nothing upstream. ADR-0002 acceptance gates Phase 03
  Tasks 3.2/3.3.
- External facts this plan leans on (re-verify if stale): `spec_math`
  0.1.6 (MIT OR Apache-2.0) provides `bessel_k1`/`bessel_kn`/`li2`;
  PyO3 abi3-py310 wheels cover CPython ≥3.10; the rust-cyphus crates
  are GPL-3 and stay out of the tree (ADR-0002).

## Related

- Background: August 2026 migration cost analysis (session artifact
  "Hazma · Cython → Rust migration analysis"); rust-cyphus crate
  assessment 2026-08-03 (`references/numerics-replacements.md`).
- GitHub Issue: none — internal modernization.

## Change control

See [`../../docs/workflow.md#adr-placement`](../../docs/workflow.md#adr-placement)
for when to write an ADR and where it lives. Patch the affected
`PLAN.md` / phase file / `rules.md` when canonical behavior changes —
known pending: the status flips of ADR-0002 (license-clean numerics)
and ADR-0003 (`hazma.gamma_ray` removal), both Proposed and awaiting
sign-off.

## Closing this project

The PR that flips this `PLAN.md` `status:` to `Complete` must also bump
`VERSION` in `hazma/__init__.py` per the `version_bump:` frontmatter and
add a `CHANGELOG.md` entry naming this project slug. Re-check the level
against the **Numerical impact** section above before bumping.
Verify with `scripts/agents/preflight.sh --closing` (note Task 7.1
relocates the version's source of truth — the closing check must run
against the post-cutover plumbing). See
[`../../docs/versioning.md`](../../docs/versioning.md).

### Anticipated ADRs

- ADR-0003 (written, Proposed): removal of the broken-on-import
  `hazma.gamma_ray` module — Task 0.5 executes it once accepted.
- Possible: QUADPACK-port deviation record, if faithful translation
  proves impractical for `qelg` and a documented algorithmic
  substitution is made instead (would revise corpus budgets).
