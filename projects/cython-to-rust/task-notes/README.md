# Working Memory: cython-to-rust

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Status:** Complete (2026-08-29, shipped as hazma 3.0.0)
**Plan References:** `../PLAN.md` (all sections)
**Related ADRs:** ADR-0001 (accepted), ADR-0002 (accepted 2026-08-04 —
Phase 03 Tasks 3.2/3.3 no longer gated), ADR-0003 (accepted
2026-08-04 with an Addendum the same day; non-deletion steps executed in
Task 0.5 on 2026-08-05, deletion executed in Task 0.2 on 2026-08-06 —
**fully discharged**)
**Depends On:** none

## Objective

Track cross-phase context and live phase status for the Cython→Rust
migration so any agent picking up work mid-project starts from facts,
not re-discovery. Per-task status lives in each `phase-XX/README.md`.

## Phases

| # | Phase | Phase file | Working memory | Status |
| --- | ------- | ----------- | ---------------- | -------- |
| 00 | Dead-code purge | [phase-00-dead-code-purge.md](../phases/phase-00-dead-code-purge.md) | [phase-00/README.md](phase-00/README.md) | **Complete (2026-08-06)** — all five tasks done; [learnings](../learnings/phase-00-dead-code-purge.md) |
| 01 | Golden parity corpus | [phase-01-parity-corpus.md](../phases/phase-01-parity-corpus.md) | [phase-01/README.md](phase-01/README.md) | **Complete (2026-08-08)** — all four tasks done; [learnings](../learnings/phase-01-parity-corpus.md) |
| 02 | Rust scaffold | [phase-02-rust-scaffold.md](../phases/phase-02-rust-scaffold.md) | [phase-02/README.md](phase-02/README.md) | **Complete (2026-08-09)** — all three tasks done; [learnings](../learnings/phase-02-rust-scaffold.md) |
| 03 | Numerics foundation | [phase-03-numerics-foundation.md](../phases/phase-03-numerics-foundation.md) | [phase-03/README.md](phase-03/README.md) | **Complete (2026-08-11)** — all five tasks done; [learnings](../learnings/phase-03-numerics-foundation.md) |
| 04 | Spectra kernels | [phase-04-spectra-kernels.md](../phases/phase-04-spectra-kernels.md) | [phase-04/README.md](phase-04/README.md) | **Complete (2026-08-20)** — all six tasks done; 16 entry points on Rust and `hazma/spectra/` holds no Cython `def`; [learnings](../learnings/phase-04-spectra-kernels.md) |
| 05 | Mediator cross sections | [phase-05-mediator-cross-sections.md](../phases/phase-05-mediator-cross-sections.md) | [phase-05/README.md](phase-05/README.md) | **Complete (2026-08-21)** — all three tasks done; [learnings](../learnings/phase-05-mediator-cross-sections.md) |
| 06 | Mediator spectra | [phase-06-mediator-spectra.md](../phases/phase-06-mediator-spectra.md) | [phase-06/README.md](phase-06/README.md) | **Complete (2026-08-27)** — all four tasks done; zero `.pyx`/`.pxd` remain; [learnings](../learnings/phase-06-mediator-spectra.md) |
| 07 | Cutover + close | [phase-07-cutover.md](../phases/phase-07-cutover.md) | [phase-07/README.md](phase-07/README.md) | **Complete (2026-08-29)** — all four tasks done; maturin backend and release pipeline, docs swept, project closed at 3.0.0; [learnings](../learnings/phase-07-cutover.md), [retrospective](../learnings/project-retrospective.md) |

```text
00 ──► 01 ──► 02 ──► 03 ──► 04 ──► 06 ──► 07
                        └──► 05 ──┘
```

## Exit Criteria

- All eight phases Complete; zero `.pyx`/`.pxd` in the tree; all 41
  consumed entry points served by `hazma._core` (the 2 unconsumed
  `sigma_xx_to_all` exports dropped in Phase 05); maturin backend live.
- ADR-0002 and ADR-0003 both accepted 2026-08-04.
- Closing PR bumps `[project] version` in `pyproject.toml` per
  `PLAN.md`'s
  `version_bump:` frontmatter and adds a `CHANGELOG.md` entry naming
  this project slug, with the aggregated drift table. See
  [`../../../docs/versioning.md`](../../../docs/versioning.md).

## Inputs Reviewed

- `../PLAN.md`, both `../references/*.md`, `../rules.md` — project
  contract and grounded facts.
- August 2026 analysis session: every `.pyx`/`.pxd` read in full by
  five parallel passes; findings distilled into the references.
- rust-cyphus crates (cloned + test-run 2026-08-03, rustc 1.96):
  results in `../references/numerics-replacements.md`.

## Findings

Cross-phase findings from Phases 00–05 moved verbatim to
[`history-findings.md`](history-findings.md) on 2026-08-21 (lines
58–892 of this file at `c57ce4f` for Phases 00–04; lines 72–91 at
`cbe5555` for Phase 05). The phase learnings under
[`../learnings/`](../learnings/) are their distillation and are what a
new task reads; open the archive only when a learnings entry, a task
note or a citation sends you to the original. Phase-scoped findings
for the open phase live in its working memory
([`phase-07/README.md`](phase-07/README.md)). A finding that outlives
its phase is appended below as one bullet and swept into the archive
when that phase closes
([ADR-0002](../../../docs/adrs/ADR-0002-read-phase-learnings-not-closed-task-notes.md)).

_Phase 05's two cross-phase findings were swept into the archive at
phase close on 2026-08-21._

_Phase 07's one cross-phase finding was swept into the archive at
project close on 2026-08-29._

## Numerical impact so far

The running record moved wholesale — unedited and in order — to
[`numerical-impact.md`](numerical-impact.md) on 2026-08-21 (lines
893–1491 of this file at `c57ce4f`). **Append there, not here:** one
entry per task that touched a public code path, giving the function,
the grid checked, and the result, exactly as before. `../PLAN.md`
§"Closing this project" and Phase 07's CHANGELOG aggregation read that
file; do not reconstruct it from memory.

## Decisions and Implementation Notes

Phases 00–04's entries moved verbatim to
[`history-decisions.md`](history-decisions.md) on 2026-08-21 (lines
1492–1658 of this file at `c57ce4f`). Canonical decisions live in the
ADRs and phase files; the learnings carry the rest. A new cross-phase
decision is appended below as one line with its rationale and ADR link,
and swept into the archive when its phase closes.

_Phase 07's one cross-phase decision was swept into the archive at
project close on 2026-08-29._

## Files Changed

The per-task roll-ups for Phases 00–04 and the parity-corpus stability
follow-up moved verbatim to
[`history-files-changed.md`](history-files-changed.md) on 2026-08-21
(lines 1659–1953 of this file at `c57ce4f`). Each task note's own
§Files Changed is authoritative and the per-phase roll-up lives in
`phase-XX/README.md`; this section holds cross-phase material only.

_No cross-phase roll-up recorded since the 2026-08-21 sweep._

## Verification

The per-task suite states for Phases 00–04 moved verbatim to
[`history-verification.md`](history-verification.md) on 2026-08-21
(lines 1954–2179 of this file at `c57ce4f`). Re-derive counts rather
than quoting them: the current state is in the open phase's
`phase-XX/README.md` and the latest task note's §Verification.

## Open Questions

- ~~**ADR-0002 sign-off** (Logan): accept the license-clean-numerics
  decision, or deliberately take the GPL route?~~ — **closed
  2026-08-04: accepted.** Hazma stays MIT and no GPL-3 crate enters the
  tree or the dependency graph, so **Phase 03 Tasks 3.2/3.3 are
  unblocked**: `spec_math` (cephes lineage) for specfun, a fresh
  netlib-QUADPACK translation for the integrator, cyphus as an
  out-of-repo oracle only.
- ~~**ADR-0003 sign-off** (Logan): confirm deletion of the
  broken-on-import `hazma.gamma_ray`~~ — **closed and fully
  discharged.** Accepted 2026-08-04; non-deletion steps executed in
  Task 0.5 on 2026-08-05 (replacement status recorded, docs repointed);
  the deletion itself executed in Task 0.2 on 2026-08-06, with its
  CHANGELOG entry. **Phase 00 closed the same day.**
  `gamma_ray_fsr`'s successor, `hazma.spectra.dnde_photon_fsr`, shipped
  via
  [`../../../docs/followups/done/msqrd-driven-fsr-generator.md`](../../../docs/followups/done/msqrd-driven-fsr-generator.md).
- ~~`cross_section_prefactor`'s threshold cancellation (found in Task
  0.3, filed as a follow-up): fix it **after** the port, or Phase 01's
  corpus pins the cancelling values?~~ — **closed: the repair landed
  out-of-band before Phase 01 started**, via `two_body_momentum`'s
  factored form (see "Out-of-band" under "Numerical impact so far").
  The question is moot; the corpus will pin the fixed values.
- **Are the other 37 corpus cases pinning correct numbers?** The
  parity-corpus follow-up built a 60-digit reference for four entry
  points and found part of what they pinned was wrong. The rest of the
  corpus has no such oracle; "no platform disagreed with it" is weaker
  evidence, and that follow-up's own Finding 3 is why. Nothing in the
  three-platform sweep points at another case (next worst 5.6e-08, and
  explained), so this is a standing doubt rather than a lead. The cheap
  answer for any case that comes under suspicion is
  `test/parity/reference.py`'s pattern: a verbatim `mpmath` copy of the
  `.pyx` body, no build needed.
- Phase 05 parallelism: run 05 alongside 04 (no shared files) or keep
  strictly serial? Decide based on who's driving.
  **Note as of Task 4.2:** the corpus's mode flip has already happened,
  so Phase 05 no longer pays that cost by starting — but it *does* meet
  five of the six ill-conditioned blocks (all scalar cross sections), and
  now that Task 4.2 has cleared the sixth, **Phase 05 is the only work
  the follow-up below still gates.**
- **Does the φ spectrum omit a `φ → π⁰γ` line entirely?** (Task 4.2.)
  `hazma/_utils/constants.pxd:283` defines `BR_PHI_TO_PI0_A = 1.32e-3`
  and nothing reads it (checked against `origin/master` too), while the ω
  adds exactly the analogous line for its own `π⁰γ` mode. Suggestive but
  not settled — recorded as the first open question on
  [`../../../docs/followups/todo/phi-photon-lines-use-the-daughter-meson-energy.md`](../../../docs/followups/todo/phi-photon-lines-use-the-daughter-meson-energy.md)
  so whoever repairs the line energies settles it in the same PR. The
  port carries exactly what the Cython carried either way.
- **The ill-conditioned-points repair did not land before the first
  swap**, which the project handoff had asked for (Task 4.1). It was not
  blocking for `spectra.positron.muon` — none of the six affected blocks
  is that entry point — but **corpus regeneration is closed from now
  on**, so options 1 and 2 in
  [`../../../docs/followups/done/parity-corpus-pins-ill-conditioned-points.md`](../../../docs/followups/done/parity-corpus-pins-ill-conditioned-points.md)
  have to work from a pre-Phase-04 checkout. **Task 4.2 met the sixth
  block (`spectra.photon.eta[boosted_strong]`) and explicitly waived it,
  on evidence**: the port is bit-equal at 336,000 sampled points, so on
  the capturing platform there is nothing for a conditioning budget to
  absorb, and off it the parity suite does not run at all. That refutes
  the follow-up's prediction that "every affected block will produce a
  false failure the moment a Rust implementation lands" — *for that
  block*. **The remaining five are all scalar cross sections, so it is
  Phase 05, not the rest of Phase 04, that the follow-up still gates.**
- **The `TABULATED` budget class is kept rather than tightened to
  `EXACT`** (Task 4.2), even though the seven ported entry points would
  pass `EXACT` today. Unlike `spectra.positron.muon`, bit-equality here
  rests on reproducing *NumPy's summation order* — an implementation
  detail a future NumPy may change — so `EXACT` would be the wrong
  contract rather than a tighter one. The general rule for a later
  tightening: promote a class only when the bit-equality rests on an
  arithmetic identity, not on a third party's implementation.
- **One Rust module may serve several `.pyx`** (Task 4.2), stated as the
  exception to `kernels.rs`'s one-submodule-per-`.pyx` convention rather
  than left as a silent violation. `kernels::photon_tables` serves five,
  because the five differed only in data.
- **Should the corpus mode switch become per-case?** Task 4.1 measured
  what the global verdict costs: 22 of 41 cases now run at their declared
  budget rather than `rtol = 0` (the 19 `EXACT`-class ones lose nothing).
  Scoping the served-kernel half of `provenance` to
  `cases.PORTED_ENTRY_POINTS` would keep unported kernels bit-exact for
  the rest of Phases 04–06. Not done beside a kernel swap, and the kernel
  digest half of the verdict would still fire; belongs with the
  follow-up above.
- **The sdist payload** (opened by Task 0.4, the first task in the
  project to build one): the tarball still ships 20 cythonized `*.c`,
  `docs/`, `test/` and `notebooks/`, and `pyproject.toml`'s package-data
  says `*.pyd` where `*.pxd` was surely meant. Deferred deliberately —
  judgment calls, not defects — but **time-boxed to before Phase 07 Task
  7.1**, because maturin does not read `MANIFEST.in` and the same
  decisions cost more to express afterwards. Filed as
  [`../../../docs/followups/done/sdist-ships-generated-c-and-docs.md`](../../../docs/followups/done/sdist-ships-generated-c-and-docs.md).
- ~~Whether the mediator cross-section `.pyx` include a constants
  header~~ — **closed by Task 0.1: they contain no `include` directive
  at all.**
- **Nothing gates the FMA map of a quadrature integrand** (Task 4.4),
  and every remaining quadrature-backed kernel inherits it — Task 4.6's
  positron and neutrino pions and the Phase 06 mediator spectra.
  **Task 4.5 decided against the probe**, and found a cheaper answer that
  generalises: run a mutation campaign, then ask of each survivor whether
  the arithmetic can be lifted *out* of the integral. Its one survivor —
  a fused `γ·E·(1−β)` in the boost window, invisible to `cargo`, to 49
  per-kernel tests and to 10 parity blocks — became a `boost_window(e,
  erho) -> (emin, emax, pre)` `fn` with its three values pinned
  bit-for-bit, closing the campaign 6/6 with no
  `_CORE_TEST_ONLY_MODULES` widening. What stays ungated is arithmetic
  genuinely *inside* an integrand (Task 4.4's 15 sites in
  `dnde_pi_to_lnug`); that narrower limitation stands.
- ~~**Does the charged pion's forward-cone defect reach `_photon/_rho`?**~~
  **Answered by Task 4.5: yes, and the ρ compounds it.** A pure boost
  preserves the fraction of the endpoint at which the cliff sits, because
  `γ(1−β)·γ(1+β) = 1`, so the inner kernel's 0.945 should appear at every
  ρ energy; measured, the charged ρ runs 0.9963 at `γ_ρ = 1.05` down to
  **0.5366** at `γ_ρ = 10` (neutral: 0.9420 → 0.5073). The outer window
  spans decades while its integrand is nonzero only near the bottom, so
  QUADPACK misses it the same way one level out. **Repairing the
  charged-pion kernel is necessary but not sufficient**; the table and the
  consequence are on
  [`../../../docs/followups/todo/charged-pion-photon-spectrum-misses-the-forward-cone.md`](../../../docs/followups/todo/charged-pion-photon-spectrum-misses-the-forward-cone.md).
  ~~**Whether it also reaches the mediator spectra is still open**~~ —
  **answered by Task 6.2: yes.** From the committed corrected-value
  oracle, repairing it moves 1,032 of 8,610 scalar values by up to
  1.63e-06 relative and 2,013 of 29,295 vector values by up to 7.77
  relative — a factor of 8.8, at an absolute 7.3e-10 — so in the vector
  case it changes the shape of the low-energy tail.
- **Do the *other* boosted kernels share the ρ's rest-frame branch
  defect?** (Task 4.5.) Spot-checked as no — `photon_muon`,
  `photon_tables` and `positron_muon` all return a genuine rest-frame
  spectrum from the same branch shape, and only the ρ returns its
  integrand — but that was a reading of the source rather than a
  measurement, and the neutrino kernels are unported and unchecked.
  Task 4.6 sees them.

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).

## Handoff to Next Task

**The project is closed.** All eight phases are Complete, all 33 tasks
landed, and it shipped as **hazma 3.0.0** on 2026-08-29. There is no next
task in this project.

**Read the retrospective, not this file.**
[`../learnings/project-retrospective.md`](../learnings/project-retrospective.md)
is the durable memory now — what the port established, what its quirks
were, what the test infrastructure looks like, and what it left behind.
The seven phase learnings beside it hold the per-phase detail; this
working memory and the per-phase READMEs are history
([ADR-0002](../../../docs/adrs/ADR-0002-read-phase-learnings-not-closed-task-notes.md)).

**What the next reader most likely wants:**

- **Repairing one of the twelve reproduced 2.1.0 defects** →
  [`../../parity-pinned-defect-repair/PLAN.md`](../../parity-pinned-defect-repair/PLAN.md)
  sequences six of them and defines the declared-delta mechanism; the
  rest are individual `docs/followups/todo/` entries. The 3.0.0
  CHANGELOG's `Known issues` section is the user-facing roster with
  magnitudes. **Do not repair one outside that mechanism** — the parity
  corpus pins the wrong values on purpose and `../rules.md` rule 2
  forbids regenerating it.
- **Touching a kernel** → the phase learnings for the phase that ported
  it, then `test/parity/cases.py` for its call convention and
  `test/parity/tolerances.py` for its budget and the rationale behind it.
- **Touching the build or the release** →
  [`../learnings/phase-07-cutover.md`](../learnings/phase-07-cutover.md)
  §2 and §3. `pyproject.toml` is the only build entry point and
  `[project] version` is the version's source of truth.
- **Picking up deferred work** → four seeds filed at close
  ([constants consolidation](../../../docs/followups/todo/consolidate-the-two-constants-tables.md),
  [free-threaded wheels](../../../docs/followups/todo/free-threaded-abi3t-wheels.md),
  [relic-density ODEs](../../../docs/followups/todo/relic-density-odes-in-rust.md),
  [aarch64/Windows wheels](../../../docs/followups/todo/wheels-for-aarch64-and-windows.md)),
  all described in the retrospective §5.

**The one thing that has not changed and still binds:** `../rules.md`
rule 4 kept the two constants tables divergent for the whole port so that
bit-parity meant something. That rule expires with the project, but the
divergence is still in `rust/src/constants.rs` and a cargo test still
asserts it, so consolidating it is a deliberate, declared numerical
change rather than a cleanup.
