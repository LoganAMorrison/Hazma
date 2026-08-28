# Working Memory: cython-to-rust

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Status:** In Progress
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
| 06 | Mediator spectra | [phase-06-mediator-spectra.md](../phases/phase-06-mediator-spectra.md) | [phase-06/README.md](phase-06/README.md) | **In Progress** — Tasks 6.1–6.3 complete (2026-08-23, -23, -27); 6.4 open |
| 07 | Cutover + close | [phase-07-cutover.md](../phases/phase-07-cutover.md) | [phase-07/README.md](phase-07/README.md) | Not started |

```text
00 ──► 01 ──► 02 ──► 03 ──► 04 ──► 06 ──► 07
                        └──► 05 ──┘
```

## Exit Criteria

- All eight phases Complete; zero `.pyx`/`.pxd` in the tree; all 41
  consumed entry points served by `hazma._core` (the 2 unconsumed
  `sigma_xx_to_all` exports dropped in Phase 05); maturin backend live.
- ADR-0002 and ADR-0003 both accepted 2026-08-04.
- Closing PR bumps `VERSION` in `hazma/__init__.py` per `PLAN.md`'s
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
([`phase-06/README.md`](phase-06/README.md)). A finding that outlives
its phase is appended below as one bullet and swept into the archive
when that phase closes
([ADR-0002](../../../docs/adrs/ADR-0002-read-phase-learnings-not-closed-task-notes.md)).

_Phase 05's two cross-phase findings were swept into the archive at
phase close on 2026-08-21. No cross-phase finding recorded since._

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

_No cross-phase decisions recorded since the 2026-08-21 sweep._

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
  [`../../../docs/followups/todo/sdist-ships-generated-c-and-docs.md`](../../../docs/followups/todo/sdist-ships-generated-c-and-docs.md).
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
  **Whether it also reaches the mediator spectra is still open** — they
  call the charged-pion `cdef`s directly and Phase 06 should measure it.
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

**Phases 00–05 are closed** (2026-08-06, 08-08, 08-09, 08-11, 08-20,
08-21) and **Phase 06 is open**: Task 6.1 landed the shared
table/cache/mode foundation on 2026-08-23, Task 6.2 swapped the decay
photon pair the same day, and Task 6.3 swapped the positron pair on
2026-08-27. **The next task is 6.4** (retire the capi survivors and the
`_utils` headers), which closes the phase, then Phase 07.

**All 41 consumed entry points are now on `hazma._core`.** Thirteen
`.pyx`/`.pxd` remain and every one is Task 6.4's: the four capi
survivors with their `.pxd`, `_utils/boost.{pyx,pxd}`, `constants.pxd`,
`kinematics.pxd` and `legacy_parameters.pxd`. Nothing is left to swap.

**Read [`phase-06/README.md`](phase-06/README.md) and then
[`phase-06/task-6.3-positron-spectra.md`](phase-06/task-6.3-positron-spectra.md)
before starting 6.4** — its `## Handoff` is the brief, and its
`## Findings` carry the two results a deletion task would otherwise
re-derive.

**Five Task 6.2 results a later task should not re-derive.**

1. **The drift these swaps carry is the integrator's, not the
   transliteration's**, and that is measured: setting `eng_s == ms` makes
   the boost integrand a *constant*, and every channel then agrees with
   the Cython to within one ulp. What the corpus sees — worst 5.3327e-12
   — is `crate::quad` against scipy's QUADPACK. A constant integrand is
   not reproduced exactly either: `∫₋₁¹ c dcl` lands one ulp off the exact
   `2c` on **both** sides, at different `c`. Expect the same floor in 6.3.
2. **All three swapped entry points moved**, so **three budgets were
   tightened** from `NESTED_RTOL` to `PORTED_NESTED_RTOL` (188x and 838x
   headroom) — **ten tightened, none widened** across the project — and
   the Phase 07 CHANGELOG owes all three a line.
3. **The forward-cone question is closed, from data that already
   existed.** `test/parity/oracles/data/manifest.json` holds defect A3's
   corrected-value capture over exactly the three photon corpus cases:
   repairing it moves 1,032/8,610 scalar values by up to 1.63e-06 and
   2,013/29,295 vector values by up to **7.77** relative. **Task 6.3
   answered the same question for the positron pair from the same
   manifest** (defect A4): 5,237 of 16,740 values in each of the four
   cases, all moving *up*, by up to **3.7421e-04** relative — exactly
   `R_FACTOR**2 - 1`, so a pure normalization rather than a change of
   shape. Both are recorded in `numerical-impact.md`; do not measure
   either again.
4. **A mutation survivor is a statement about the coefficient or about
   the grid.** Fourteen of 6.2's thirty-seven fused sites are provably
   identity-equivalent (power-of-two coefficient ⇒ exact product; zero
   disagreements over 40,002 masses per shape under exact rational
   arithmetic); two more were alive only because the grid never reached
   `2 m_μ`. And **force the rebuild between mutations** — 6.2's second
   campaign run measured a stale `.so` and lagged its own mutations by
   two iterations.
5. **`pyproject.toml`'s `cython<3.3` cap is gone**, because the only file
   it protected was the one 6.2 deleted. Measured first: the seven
   surviving `.pyx` compile under cython 3.3.0 and a tree whose
   extensions are *built* by it runs the suite at the same counts as
   3.2.9, corpus and Cython-twin bit-equality included.

**Two things Task 6.2 spent that do not come back.** It deleted the last
`.pyx` that spells a dispatch message, so `test_core_dispatch.py`'s
Cython-oracle classes are retired and the roster the port emits is now
*frozen* there with per-message provenance — from 6.3 on, "the port's
messages are the Cython's" is transcription rather than execution. And
`test/test_core_mediator_tables.py`'s two live-twin mode oracles now call
the port, so they pin the parser/entry-point *coupling* rather than the
behaviour.

**The performance story is the dead cache, and it is now measured on
both pairs** (release builds of both sides, `rules.md` rule 12). Task
6.2, isolating the table build: **4.2x** on the build itself and
12.9x–5,500x once the memo hits, with a 20-point partial-width sweep at
fixed mass going from 186.3 ms to 0.045 ms. Task 6.3, on a whole
200-point energy sweep, where the boost quadrature rather than the table
is the ceiling: **32.3x** at a fresh mass, **42.8x** at a repeated one,
**43.3x** over a 20-point width sweep. The `.pyx` rebuilt two 500-point
quadrature-backed tables on *every* call in both pairs.

**A fourth Task 6.1 result, and the one that cost a red CI round:** the
Rust grid is bit-equal to `numpy.logspace` on macOS/arm64 and **one ulp
off it at ~5% of points on Linux/x86-64**. Nothing downstream changes,
but **"bit-equal to the Cython" is a macOS/arm64 statement** for anything
that reads these tables, and any comparison against a NumPy oracle must
be scoped with `ON_THE_CAPTURING_PLATFORM` rather than left open.

**A sixth `_CORE_TEST_ONLY_MODULES` probe exists**,
`hazma._core.mediator_tables`, because every oracle for the foundation
lives in Python.

**Read [`../learnings/phase-05-mediator-cross-sections.md`](../learnings/phase-05-mediator-cross-sections.md)
and [`../learnings/phase-04-spectra-kernels.md`](../learnings/phase-04-spectra-kernels.md)
before starting any Phase 06 task.** They replace the Phase 04 and 05
notes. The load-bearing items for what is left: run
`grep -c SoftComplexToDouble` on the **generated C** before
transliterating (0/0 for both positron modules, per Task 6.1, so no
complex arithmetic there); clang's FMA contraction follows one syntactic
rule on the C tree, which Task 6.2 applied to 37 sites and confirmed
against the live twin — and which Task 6.3 found **predicts rather than
decides**, one of its eleven sites contradicting the rule under
measurement; every task's numerical prediction has been wrong
in a different direction; and a mutation survivor is either unobservable
*by construction* or a seam that needs lifting out.

**Phase 04 delivered 16 entry points and `hazma/spectra/` holds no
Cython Python entry point of any kind.** Four `.pyx` survive there for
their `cdef` capsules alone — `_photon/{_muon,_pion}` and
`_positron/{_muon,_pion}`. **As of Task 6.3 none has a consumer outside
its own pair**: each `_pion` cimports its `_muon` twin and
`hazma/_utils/boost`, and nothing else in the tree reads any of them. So
**Task 6.4's `rg` sweep is already empty**, and the stale comments that
claimed otherwise were corrected in 6.3.

**The parity corpus left bit-equality mode permanently in Task 4.1**, and
corpus *regeneration* is closed (see Open Questions). 19 of the 41 cases
are `EXACT` class and still run at `rtol = 0` **on the capturing
platform**; the rest run at their declared budget.

**The corpus is platform-portable as of 2026-08-18 and CI runs it on
every matrix entry.** Three carve-outs make that true and each names
exactly what it covers, so a `test/parity` failure on a new platform
should be triaged into one of them rather than absorbed by a wider
budget: `test/parity/stability.py`'s 494 unpinnable positions,
`tolerances.PLATFORM_EXACT_RTOL` and `PLATFORM_SPECFUN_RTOL` for those
two classes off the capturing libm, and `tolerances.zero_floor` for the
four declared stored zeros a change of libm moves. **Phase 06 should read
[`phase-01/followup-parity-corpus-stability.md`](phase-01/followup-parity-corpus-stability.md)**
before porting anything the mask covers, and should not read Task 5.2's
bit-equal result as evidence the mask is unnecessary.

**For the next agent starting any task in this project:**

1. Read `../PLAN.md` end-to-end, then this file, then the closed phases'
   learnings — **Phase 04's first**, since it is the most recent and the
   most directly applicable — then the active phase's
   `phase-XX/README.md`.
2. Load the reference file(s) the phase's Prerequisites name — the
   references replace re-reading the Cython audit.
3. Check Open Questions above. No ADR sign-off is outstanding — all three
   project ADRs are Accepted, so no phase carries a decision gate.

**Currently safe to assume:**

- The dead-code map and entry-point inventory in
  [`../references/cython-inventory.md`](../references/cython-inventory.md)
  were verified against 2.1.0 (Aug 2026) and the file declares itself a
  snapshot. **Every row of its dead-code table is now done.** Read it for
  the **live surface** and the cimport DAG, which Phase 06 still needs;
  read its headline counts as history.
- **7 `.pyx` and 8 `.pxd` after Task 6.2** (9/8 after Task 5.2, 10/8
  after 5.1, 11/8 after 4.6). Both mediator decay-spectrum modules were
  whole-file deletions — nothing cimported either and neither exported
  capsules — as both cross-section modules were before them. Zero C++.
  Re-derive with the clean-then-rebuild recipe rather than quoting this;
  a stale `.so` makes a wrong list look right.
- **`hazma._core` serves thirty-seven kernels.** The three added by
  Task 6.2 are the whole consumed surface of the two
  `*_mediator_decay_spectrum.pyx` —
  `scalar_mediator.scalar_mediator_decay_spectrum` and
  `vector_mediator.{dnde_decay_v, dnde_decay_v_pt}` — called by
  `hazma/{scalar,vector}_mediator/_*_mediator_spectra.py`. The two vector
  entry points are one kernel behind two dispatch shapes, and are
  bit-for-bit identical to each other. The twelve from Task 5.2 are the
  whole consumed surface of
  `_c_scalar_mediator_cross_sections.pyx` —
  `scalar_mediator.sigma_xx_to_s_to_{ff,gg,pi0pi0,pipi}`,
  `sigma_xx_to_ss`, `sigma_ss_to_xx`,
  `sigma_x{l,pi,pi0,g,s}_to_x{l,pi,pi0,g,s}` and
  `thermal_cross_section` — each called under a short alias, because that
  wrapper already uses every canonical name for a mixin method. Its
  thirteenth `def`, `sigma_xx_to_all`, was **dropped rather than ported**.
  The six from Task 5.1 are
  `vector_mediator.sigma_xx_to_v_to_{ff,pipi,pi0g,pi0v}`,
  `sigma_xx_to_vv` and `thermal_cross_section`; its seventh
  `sigma_xx_to_all` was likewise dropped and survives as a private helper
  of the Rust thermal integrand. The sixteen from Phase 04:
  `positron.dnde_positron_muon` (4.1), the seven
  `photon.dnde_photon_*` tabulated meson spectra (4.2),
  `photon.dnde_photon_muon` (4.3),
  `photon.dnde_photon_{charged,neutral}_pion` (4.4),
  `photon.dnde_photon_{charged,neutral}_rho` (4.5), and
  `positron.dnde_positron_charged_pion` plus
  `neutrino.dnde_neutrino_{muon,charged_pion}` (4.6). Every kernel module
  under `rust/src/kernels/` is `pub` and PyO3-free, so the mediator
  spectra call them natively the way the `.pyx` cimported the Cython.
- **A swap needs three edits in `test/parity/cases.py`, not one** (Task
  6.2): the `Case.module` moves to the wrapper, a `PORTED_ENTRY_POINTS`
  row is added, and the `_CORE_TEST_ONLY_MODULES` comment counting live
  mediator `.pyx` needs re-deriving. Missing the second turns
  `test_the_served_roster_is_exactly_the_ported_entry_points` red with a
  set-difference message that does not name the cause.
- **`crate::quad` short-circuits an empty interval**, as
  `scipy.integrate.quad` does (Task 4.6). Any later kernel whose limits
  can coincide inherits the fix.
- **Four test-module shapes, and the twin's fate forces the choice.**
  `test/test_core_{positron,photon}_muon.py` for a kernel whose twin
  survives *and* admits bit-equality; `test/test_core_photon_pion.py` for
  one carrying two oracle classes at two standards;
  `test/test_core_positron_pion.py` for a surviving twin that is
  quadrature-backed and therefore has **no** bit-equality mode on any
  platform; and `test/test_core_{photon_tables,photon_rho,neutrino,mediator_decay_photon}.py`
  for a kernel whose twin does **not** survive the PR — there the
  substitute is an independent Python reference plus the
  against-the-Cython numbers measured *before* the deletion. **Not**
  `test/test_core_dispatch.py`; see Decisions.
- **One test module per clone-pair, not per entry point**, when the
  independent reference is one function parameterised by pair (Task 6.2).
- **Run a mutation campaign on every kernel, and interrogate the
  survivors.** Task 4.4's eleven had two survivors and concluded they were
  unobservable; Task 4.5's six had one and *fixed* it; Task 4.6's eleven
  had two and resolved both; Task 6.2's thirty-seven had sixteen, of which
  fourteen are provably identity-equivalent and two needed a wider grid.
  Ask "can this be lifted out?" and "is this the coefficient or the
  grid?" before writing a limitation into the source.
- **A `.pyx` whose locals are untyped contracts nothing** (Task 4.5) —
  and so does one whose locals are all typed but which contains no
  multiply-add at all (Task 4.6). **Read the disassembly; do not infer it
  from the source.**
- **`crate::quad` is proven on both of its drivers**: `qagpe` by Task
  4.4's `points=[-1, 1]` site and by both Task 6.2 entry points (which
  pass the same discarded break points), and `qagse` by Tasks 4.5 and
  4.6. Copy the call site's `epsabs`/`epsrel`/`points` verbatim into a
  `const QuadOpts`. `quad`'s `Err` arm depends only on the options, never
  on the integrand, so it is unreachable for a `const` opts value; return
  `NaN` there rather than panicking, and assert the unreachability with a
  `cargo` test.
- **`hazma_core::constants` exists and is bit-equal to the Cython**
  (Task 3.1). Name the table the `.pyx` `include`s — `pdg` for everything
  under `hazma/spectra/**`, `legacy` for the mediator spectrum extensions
  — **except** `derived::photon_pion`, which legitimately reads both.
- **A `.pyx` with a fractional exponent is not doing real arithmetic**
  (Task 5.1). Cython 3's default `cpow` semantics compile `double **
  double` — and everything around it in the same expression — in
  `double _Complex`, reaching `cpow` and compiler-rt's `__divdc3`. Both
  are in `rust/src/kernels/soft_complex.rs`, and Task 6.2 needed exactly
  that pair for the two decay modules' FSR coefficients (one live site
  each, on *different* factors: the scalar's lepton coefficient and the
  vector's charged-pion one). **`grep -c SoftComplexToDouble` on the
  generated C before porting.**
- **Where clang fuses is one syntactic rule** (Task 5.2): `EmitFMulAdd`
  contracts `A ± B` when `A` is a syntactic multiply, else when `B` is,
  decided on the **C** tree Cython emits — where `x ** n` is a `pow`
  **call**, never a multiply. Task 6.2 reproduced 37 sites in two kernels
  from that rule alone without reading a disassembly per site, and its
  mutation campaign confirmed 23 of them observable.
- **`pip install -e .` builds `hazma._core` unoptimized** (Task 5.1) — a
  benchmark from an editable tree is ~20x pessimistic and inverts the
  comparison against Cython. Take rule 12's benchmark from a **release**
  build of both sides in one interpreter, and **run it from outside the
  repo**: a run from the repo root imports `hazma` from the worktree
  rather than site-packages, which silently invalidated a Task 6.2
  measurement. Filed rather than fixed
  ([the debug editable build](../../../docs/followups/todo/editable-installs-build-the-rust-extension-in-debug.md)).
- **Task 3.5 is done, so the dispatch and error contract is settled** —
  a wrapper writes `dispatch::map_unary(x, "<quantity>", kernel)`,
  `map_flavors` for a `(3, N)` return, `map_unary_try` for a kernel that
  raises at some arguments, or **`require_vector` for an argument that
  must be a 1-D array and is never a scalar** — which Task 6.2 used for
  both `partial_widths` and `dnde_decay_v`'s energies, declaring the two
  divergences that come with the latter. The rule that decided every
  divergence: **each exception the Cython raises explicitly keeps its
  type; only its `assert`s change type** (`../rules.md` rule 9).
- **Task 3.4 is done, so `hazma_core::{interp, boost}` exist**, both
  bit-equal to what they replace on the capturing platform. Do not touch
  the `mul_add`s and do not repair `boost_integrate_linear_interp`'s
  window coverage, however obviously wrong it looks — the corpus pins the
  wrong values and the repair is
  [its own follow-up](../../../docs/followups/todo/boost-integral-drops-last-interior-cell.md).
- **The parity corpus is the gate from here on.** `python
  test/parity/generate.py --check` verifies it; `test/parity/cases.py` is
  the single source of every entry point's call convention. Do not
  regenerate it from a tree in which any kernel runs on Rust —
  `rules.md` rule 2, enforced in code by `assert_no_rust_core`.
- **The suites are merged and green on the capturing platform**: bare
  `pytest -q` → **2262 passed / 15 skipped / 12 subtests** as of Task 6.2
  (2163/15/12 at 6.1, 1935/15 at 4.6, 1006/13 at Phase 01 close).
  `cargo test --no-default-features` → **249 passed**, from 222.
  Re-derive rather than quoting.
- **`test/test_theory_aggregation.py` is the model-layer gate the corpus
  cannot be** (Task 1.4): identities over `hazma/theory/`'s pure-Python
  aggregation, no golden data, and the only numerical gate in the repo
  that is not scoped to the capturing platform. **Run it either side of
  every kernel swap** — `69 passed` as of Task 6.2.
- **A `.rs` edit needs `pip install -e .`, not `cargo build`** (Task 2.2).
  Iterate with
  `cargo test --manifest-path rust/Cargo.toml --no-default-features`,
  reinstall before quoting any pytest or parity number, and confirm with
  `python -c "import hazma._core; print(hazma._core.__file__)"`. If a
  *harness* rebuilds in a loop, delete the artifact first and assert it
  came back — see Task 6.2's campaign.
- **The build entry point is `setup.py`**, and **`pyproject.toml` no
  longer caps `cython`** (Task 6.2 removed the `<3.3` cap with the file
  that forced it; the surviving `.pyx` are measured green under 3.3.0).
- **The sdist and wheel both build, and the sdist installs and runs** in a
  fresh venv from outside the repo (recipe in Task 0.4's note; reuse it in
  Phase 07).
- **`hazma.gamma_ray` is gone, docs and all.** The settled replacement
  wording for the Phase 07 aggregate: `gamma_ray_decay` →
  `hazma.spectra.dnde_photon`, `gamma_ray_fsr` →
  `hazma.spectra.dnde_photon_fsr`, **neither a drop-in**.
- The legacy constants table lives at
  `hazma/_utils/legacy_parameters.pxd` and is now its **only** copy.
  `hazma.utils` is the only home for `cross_section_prefactor` and
  `minkowski_dot`.

**Currently risky / unknown:**

- **Eight blocked defects now share one eventual corpus regeneration** —
  the positron normalization (4.1), the boost integral (3.4), the η′ line
  weight and the φ line energies (both 4.2), the muon photon spectrum's
  rest-frame endpoint (4.3), the charged pion's lost forward cone (4.4),
  the ρ's rest-frame branch returning its integrand (4.5), and the
  charged pion's doubled `π → e ν` **neutrino** line (4.6). Do not "fix"
  any of them in passing; each fails the gate that governs the remaining
  swaps. **Worth telling the maintainer separately from this project's
  schedule** — several affect published numbers today, and two affect the
  *shape* of a spectrum rather than a total, which is the kind a limit
  calculation notices. Task 4.5 found the forward-cone defect
  **compounds** through the ρ rather than merely propagating, and **Task
  6.2 quantified its reach into the mediator photon spectra**: up to
  **7.77** relative on `dnde_decay_v`, i.e. a factor of 8.8, though at an
  absolute 7.3e-10. Six of the eight are sequenced in
  [`../../parity-pinned-defect-repair/PLAN.md`](../../parity-pinned-defect-repair/PLAN.md).
- **Re-capturing `test/parity/oracles` closes at Task 6.4**, and the
  roster it would need is incomplete: `RESTORED_SOURCES` has no rows for
  the four `.pyx` Tasks 6.2 and 6.3 deleted, because a task cannot cite
  its own commit's SHA. Filed
  ([the restore revisions](../../../docs/followups/todo/oracle-restore-revisions-for-the-mediator-decay-pyx.md)),
  and 6.3 widened it to all four rather than discharging it — it hit the
  same wall 6.2 did. **Task 6.4 is the first task that can close it, and
  also the one that makes it moot**, so decide there. Not blocking — the
  `pytest` gate does not read that dict.
- **Nine places in the tree cite a lessons class that is not in the
  ledger** (`[mutation-harness-poisons-its-own-baseline]`), including a
  guard in `test/parity/oracles/capture.py` named after it. Found by Task
  6.2 on hitting the class a third time; filed
  ([the missing class](../../../docs/followups/todo/lessons-ledger-missing-the-mutation-harness-class.md))
  rather than fixed, because the ledger's format needs the PR that
  learned it (Task 3.3's) and the ledger is past its working-set cap.
- **Two Task 1.4 follow-ups ripen inside this project.** The
  [`MASS_E` `nan`](../../../docs/followups/done/positron-spectrum-nan-at-legacy-electron-mass.md)
  is now **on the grid's first abscissa** — the positron table starts at
  the legacy `m_e`, so Task 6.3 is where it bites — and the
  [scalar-energy contract](../../../docs/followups/todo/model-spectra-reject-scalar-energies.md)
  during 06; Task 3.5 settled the compiled half and what is left is pure
  Python.
- **`release.yml` has no pull-request trigger**, so any future change to
  it needs its own dispatch to be measured at all
  (`../../../docs/agents/lessons.md`
  `[unrun-workflow-cannot-close-a-criterion]`). Phase 07 Task 7.1
  rewrites it for maturin and inherits that.
