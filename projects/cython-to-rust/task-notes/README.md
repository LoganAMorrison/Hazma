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
| 06 | Mediator spectra | [phase-06-mediator-spectra.md](../phases/phase-06-mediator-spectra.md) | [phase-06/README.md](phase-06/README.md) | **In Progress** — Task 6.1 complete (2026-08-23); 6.2–6.4 open |
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
table/cache/mode foundation on 2026-08-23, so **the next task is 6.2**
(the decay spectrum pair), then 6.3, 6.4 and Phase 07.

**Three Task 6.1 results a 6.2/6.3 agent should not re-derive.** (1) The
`grep -c SoftComplexToDouble` the Phase 05 learnings demanded has been
run: **6 / 0 / 6 / 0** for scalar-decay / scalar-positron /
vector-decay / vector-positron, one live site each in the photon pair
(the `** 1.5` FSR coefficient at
`scalar_mediator_decay_spectrum.pyx:113` and
`vector_mediator_decay_spectrum.pyx:73`), both covered by the existing
`crate::kernels::soft_complex` pair — **neither positron module needs
it**. (2) An unrecognised mode string returns `0.0` today rather than
raising, so the parsers return `Option` and the entry points owe that
`0.0`; the tightening is
[filed](../../../docs/followups/todo/mediator-spectra-accept-unknown-mode-strings.md).
(3) The Cython's cache key names partial widths that `__set_spectra`
never reads, so the port keys on the mediator mass alone — the phase
file's Task 6.1 bullet is amended to say so.

**A fourth result, and the one that cost a red CI round:** the Rust
grid is bit-equal to `numpy.logspace` on macOS/arm64 and **one ulp off
it at ~5% of points on Linux/x86-64** — NumPy's vectorised `power` loop
and `f64::powf` are not the same code. Every measured disagreement was
exactly one ulp. Nothing downstream changes (the corpus already runs in
budget mode off its capturing platform), but **"bit-equal to the
Cython" is a macOS/arm64 statement for anything that reads these
tables**, and any 6.2/6.3 comparison against a NumPy oracle must be
scoped with `ON_THE_CAPTURING_PLATFORM` rather than left open.

**A sixth `_CORE_TEST_ONLY_MODULES` probe exists**,
`hazma._core.mediator_tables`, because every oracle for the foundation
lives in Python. Its `test/test_core_mediator_tables.py` also holds the
only record of the pre-repair unknown-mode behaviour once Tasks 6.2–6.4
delete the twins.

**Read [`../learnings/phase-05-mediator-cross-sections.md`](../learnings/phase-05-mediator-cross-sections.md)
before starting Phase 06.** It replaces the three Phase 05 notes. Its
three load-bearing items for 06: run
`grep -c SoftComplexToDouble` on the **generated C** of all four
mediator-spectrum `.pyx` before transliterating anything (that grep
changed the answer twice in Phase 05, and 06's four are still
unmeasured); clang's FMA contraction follows one syntactic rule on the C
tree, not a case list; and establishing a pre-port baseline now costs a
build from a git commit, because the twins are deleted — Task 5.3's
detached-worktree recipe is the cheapest way and 06 will need it.

**Phase 04 delivered 16 entry points and `hazma/spectra/` now holds no
Cython Python entry point of any kind.** Four `.pyx` survive there for
their `cdef` capsules alone — `_photon/{_muon,_pion}` and
`_positron/{_muon,_pion}` — read only by the four mediator spectrum
modules Phase 06 ports. `cases.rust_core_kernels()` → **16**.

**Read [`../learnings/phase-04-spectra-kernels.md`](../learnings/phase-04-spectra-kernels.md)
before starting either phase.** It replaces the six task notes, and three
of its lessons are the ones a Phase 05/06 agent will otherwise relearn:
the seven live defects all came from writing a statement the original
never made; every task's numerical prediction was wrong in a different
direction; and a mutation survivor is either unobservable *by
construction* or a seam that needs lifting out — Task 4.6's was the
latter, a γ spelling **29x outside the corpus's own budget** at energies
the corpus does not sample.

**The parity corpus left bit-equality mode permanently in Task 4.1**, and
corpus *regeneration* is closed (see Open Questions). 19 of the 41 cases
are `EXACT` class and still run at `rtol = 0` **on the capturing
platform**; the rest run at their declared budget. **Five budgets have now
been tightened and none widened** — `PORTED_QUAD_RTOL = 1e-12` for
`spectra.photon.charged_pion` (Task 4.4), `spectra.positron.charged_pion`
and `spectra.neutrino.charged_pion` (both Task 4.6), and
`PORTED_NESTED_RTOL = 1e-9` for both ρ cases (Task 4.5), and
`PORTED_QUAD_RTOL` for both thermal cross sections (Tasks 5.1 and 5.2).
That is **seven tightened, none widened**, and it leaves `QUAD_RTOL`
with no holder at all — it is now the documented opening figure for the
next quadrature-backed case, which Phase 06 supplies.

**The corpus is platform-portable as of 2026-08-18 and CI runs it on
every matrix entry.** Three carve-outs make that true and each names
exactly what it covers, so a `test/parity` failure on a new platform
should be triaged into one of them rather than absorbed by a wider
budget: `test/parity/stability.py`'s 494 unpinnable positions,
`tolerances.PLATFORM_EXACT_RTOL` and `PLATFORM_SPECFUN_RTOL` for those
two classes off the capturing libm, and `tolerances.zero_floor` for the
four declared stored zeros a change of libm moves. Task 5.2 read
[`phase-01/followup-parity-corpus-stability.md`](phase-01/followup-parity-corpus-stability.md)
before porting the four scalar kernels the mask covers, and the four
came back bit-equal anyway — the 494 masked positions are where a
*different* implementation would disagree, and a faithful
transliteration is not one. **Phase 06 should read it for the same
reason**, and should not read Task 5.2's result as evidence the mask is
unnecessary.

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
  the **live surface** and the cimport DAG, which Phases 05–06 still need;
  read its headline counts as history.
- **9 `.pyx` and 8 `.pxd` after Task 5.2** (10/8 after Task 5.1, 11/8
  after Task 4.6, 14/11 before it). Both mediator cross-section modules
  were whole-file deletions — nothing cimported either and neither
  exported capsules — unlike the `def`-only removal from
  `_positron/_pion.pyx`. Zero C++.
  Re-derive with the clean-then-rebuild recipe rather than quoting this;
  a stale `.so` makes a wrong list look right.
- **`hazma._core` serves thirty-four kernels.** The twelve added by
  Task 5.2 are the whole consumed surface of
  `_c_scalar_mediator_cross_sections.pyx` —
  `scalar_mediator.sigma_xx_to_s_to_{ff,gg,pi0pi0,pipi}`,
  `sigma_xx_to_ss`, `sigma_ss_to_xx`,
  `sigma_x{l,pi,pi0,g,s}_to_x{l,pi,pi0,g,s}` and
  `thermal_cross_section` — each called by
  `hazma/scalar_mediator/_scalar_mediator_cross_sections.py` under a
  short alias, because that wrapper already uses every canonical name
  for a mixin method. Its thirteenth `def`, `sigma_xx_to_all`, was
  **dropped rather than ported** on the same re-run importer check as
  the vector one. The six added by
  Task 5.1 are `vector_mediator.sigma_xx_to_v_to_{ff,pipi,pi0g,pi0v}`,
  `vector_mediator.sigma_xx_to_vv` and
  `vector_mediator.thermal_cross_section`, each called by
  `hazma/vector_mediator/_vector_mediator_cross_sections.py`. Its
  seventh `def`, `sigma_xx_to_all`, was **dropped rather than ported**
  (the importer check was re-run and came back empty) and survives as a
  private helper of the Rust thermal integrand. The sixteen from
  Phase 04:
  `positron.dnde_positron_muon` (4.1), the seven
  `photon.dnde_photon_*` tabulated meson spectra (4.2),
  `photon.dnde_photon_muon` (4.3),
  `photon.dnde_photon_{charged,neutral}_pion` (4.4),
  `photon.dnde_photon_{charged,neutral}_rho` (4.5), and
  `positron.dnde_positron_charged_pion` plus
  `neutrino.dnde_neutrino_{muon,charged_pion}` (4.6) — each called by its
  wrapper in `hazma/spectra/_{positron,photon,neutrino}/__init__.py`.
  Every kernel module under `rust/src/kernels/` is `pub` and PyO3-free,
  so Phase 06's mediator spectra call them natively the way the `.pyx`
  cimport the Cython today: `kernels::photon_pion`'s four `pub` fns and
  `kernels::positron_pion::dnde_positron_charged_pion` most of all.
- **`crate::quad` short-circuits an empty interval**, as
  `scipy.integrate.quad` does (Task 4.6, found at
  `dnde_neutrino_charged_pion(0.0, epi)`). Any Phase 05/06 kernel whose
  limits can coincide inherits the fix; the pre-existing test missed it
  because its integrand was smooth.
- **Four test-module shapes, and the twin's fate forces the choice.**
  `test/test_core_{positron,photon}_muon.py` for a kernel whose twin
  survives *and* admits bit-equality; `test/test_core_photon_pion.py` for
  one carrying two oracle classes at two standards;
  `test/test_core_positron_pion.py` for a surviving twin that is
  quadrature-backed and therefore has **no** bit-equality mode on any
  platform; and `test/test_core_{photon_tables,photon_rho,neutrino}.py`
  for a kernel whose twin does **not** survive the PR — there the
  substitute is an independent Python reference plus the
  against-the-Cython numbers measured *before* the deletion. **Not**
  `test/test_core_dispatch.py`; see Decisions.
- **Run a mutation campaign on every kernel, and interrogate the
  survivors.** Task 4.4's eleven had two survivors and concluded they were
  unobservable; Task 4.5's six had one and *fixed* it; Task 4.6's eleven
  had two and resolved both — one lifted into a pinned `fn`, one proved
  unobservable by construction. Ask "can this be lifted out?" before
  writing a limitation into the source.
- **A `.pyx` whose locals are untyped contracts nothing** (Task 4.5) —
  and so does one whose locals are all typed but which contains no
  multiply-add at all (Task 4.6, `_neutrino/_pion.pyx`). Same `grep -c`
  answer, different cause. **Read the disassembly; do not infer it from
  the source.** And a module-level `cdef double` folded at import can be
  folded *with* contraction: `_positron/_pion.pyx`'s `emax_pi_rf` is one
  ulp above its unfused expression.
- **`crate::quad` is proven on both of its drivers**: `qagpe` by
  Task 4.4's `points=[-1, 1]` site (where scipy's filter discards both
  break points, so it runs over an empty list) and `qagse` by Task 4.5's
  two and Task 4.6's three, which pass no `points` keyword at all. Copy
  the call site's `epsabs`/`epsrel`/`points` verbatim into a
  `const QuadOpts` — Task 4.6's neutrino pion is the one live site that
  passes **no** tolerance keywords, so its `const` is scipy's defaults.
  `quad`'s `Err` arm depends only on the options, never on the integrand,
  so it is unreachable for a `const` opts value; return `NaN` there rather
  than panicking, and assert the unreachability with a `cargo` test.
- **`hazma_core::constants` exists and is bit-equal to the Cython**
  (Task 3.1). Name the table the `.pyx` `include`s — `pdg` for everything
  under `hazma/spectra/**`, `legacy` for the four mediator spectrum
  extensions — **except** `derived::photon_pion`, which legitimately reads
  both. A `derived::` submodule is retired with its `.pyx`, and there are
  two precedents for what happens to its contents: `derived::photon_rho`
  simply vanished (Task 4.5 — its three entries were bare `pdg` aliases),
  while `derived::neutrino_muon`'s five **moved into**
  `rust/src/kernels/neutrino_muon.rs` (Task 4.6 — they are arithmetic the
  kernel needs). `test/test_core_constants.py` scans the tree for the
  sources it maps, so the row must go with the file either way.
- **A `.pyx` with a fractional exponent is not doing real
  arithmetic** (Task 5.1). Cython 3's default `cpow` semantics compile
  `double ** double` — and everything around it in the same expression —
  in `double _Complex`, reaching `cpow` and compiler-rt's `__divdc3`
  instead of `pow` and `/`. Neither agrees with its real spelling (up to
  9.0e-15 and 4.0e-16 relative), so both must be reproduced:
  `cpow(t + 0i, 1.5 + 0i)` is bit-for-bit `exp(1.5·ln t)` and `__divdc3`
  is C99 Annex G's scaled quotient, both in
  `rust/src/kernels/soft_complex.rs`, shared by both mediator kernel
  modules. **`grep -c SoftComplexToDouble` on the generated C before
  porting** — the scalar cross sections turned out to have **one**
  (`__sigma_xx_to_s_to_ff`), against a Task 5.1 note that said they had
  none, so run the grep rather than reading the `.pyx`. Phase 06's four
  modules are unchecked.
- **Where clang fuses is one syntactic rule** (Task 5.2). Its
  `EmitFMulAdd` contracts `A + B` when `A` is a multiply, else when `B`
  is, decided on the **C** tree Cython emits — where `x ** n` is a `pow`
  **call**, never a multiply, and `-x**n` is an `FNeg`. That one rule
  explains every case Phase 04 and Task 5.1 recorded separately
  (`-4*mx**2 + e_cm**2` fuses, `ms**2 - e_cm**2` does not,
  `-mpi0**2 + e_cm**2` does not), and Task 5.2 reproduced all 138 FMA
  sites in eleven kernels from it without reading a disassembly per
  site. **One Python-level call boxes everything above it**: `np.log(4)`
  inside a `cdef` function makes Cython evaluate the whole path to the
  root through `PyNumber_*`, so nothing there contracts while the pure-C
  operands still fuse internally — the same observable as Phase 04's
  `_photon/_rho.pyx` from a different cause.
- **`pip install -e .` builds `hazma._core` unoptimized** (Task 5.1) —
  `setuptools_rust` infers `debug = self.inplace or self.debug`. A
  benchmark from an editable tree is ~20x pessimistic and inverts the
  comparison against Cython. Filed rather than fixed
  ([the debug editable build](../../../docs/followups/todo/editable-installs-build-the-rust-extension-in-debug.md));
  until it is decided, take rule 12's benchmark from a release build and
  say so.
- **Task 3.5 is done, so the dispatch and error contract is settled** —
  a Phase 05–06 wrapper writes
  `dispatch::map_unary(x, "<quantity>", kernel)`, `map_flavors` for a
  `(3, N)` return (Task 4.6), or **`map_unary_try` for a kernel that
  raises at some arguments** (Task 5.1, the fourth live shape) and
  inherits every message, return type and edge case. The rule that decided every
  divergence: **each exception the Cython raises explicitly keeps its
  type; only its `assert`s change type** (`../rules.md` rule 9).
- **`test/test_core_dispatch.py`'s spectra oracle is now
  `scalar_mediator_decay_spectrum`** (moved by Task 4.6, which exhausted
  `hazma/spectra/` entirely; it was `_positron/_pion` before that,
  `_photon/_rho` before that). **Phase 06 deletes that too, and there is no
  candidate after it** — retire `TestDeclaredDivergencesFromCython` or
  re-express its widenings against `cython_xs`, which is the only other
  live `.pyx` dispatch shape. `TestCythonMessageParity` is a separate
  matter: its roster now reads from the surviving `.pyx` and is down to
  two `assert` wordings, both in the mediator decay-spectrum modules. The
  wordings the port still emits for swapped kernels are pinned in each
  kernel's own test module.
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
  `pytest -q` → **1935 passed / 15 skipped** as of Task 4.6 (2026-08-20),
  from 1802/15 at Task 4.5, 1755/15 at 4.4, 1682/15 at 4.3, 1628/15 at
  4.2, 1378/13 at 3.5, 1063/13 at Phase 02 close and 1006/13 at Phase 01
  close. `cargo test --no-default-features` → **169 passed**, from 133.
  Re-derive rather than quoting; the historical series is in
  [phase-01/README.md](phase-01/README.md),
  [phase-02/README.md](phase-02/README.md) and
  [phase-04/README.md](phase-04/README.md).
- **`test/test_theory_aggregation.py` is the model-layer gate the corpus
  cannot be** (Task 1.4): identities over `hazma/theory/`'s pure-Python
  aggregation, no golden data, and the only numerical gate in the repo
  that is not scoped to the capturing platform. **Phases 05–06 run it
  either side of every kernel swap** — `69 passed` as of Task 4.6.
- **A `.rs` edit needs `pip install -e .`, not `cargo build`** (Task 2.2).
  Iterate with
  `cargo test --manifest-path rust/Cargo.toml --no-default-features`,
  reinstall before quoting any pytest or parity number, and confirm with
  `python -c "import hazma._core; print(hazma._core.__file__)"`.
- **The build entry point is `setup.py`.** `_build.py` was deleted in
  `7a817f9` (2026-08-02) and Task 0.4 swept the durable docs that named it.
- **The sdist and wheel both build, and the sdist installs and runs** in a
  fresh venv from outside the repo (recipe in Task 0.4's note; reuse it in
  Phase 07).
- **`hazma.gamma_ray` is gone, docs and all** (Task 0.5 swept, Task 0.2
  deleted). The settled replacement wording for the Phase 07 aggregate:
  `gamma_ray_decay` → `hazma.spectra.dnde_photon`, `gamma_ray_fsr` →
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
  calculation notices. Task 4.5 also found that the forward-cone defect
  **compounds** through the ρ rather than merely propagating (0.945 of
  endpoint predicted, 0.537 measured at `γ_ρ = 10`), so that repair is
  larger than the follow-up originally scoped. Six of the eight are
  sequenced in
  [`../../parity-pinned-defect-repair/PLAN.md`](../../parity-pinned-defect-repair/PLAN.md);
  4.6's arrived after that roster was fixed and needs no Cython oracle,
  so it can be scheduled independently.
- **Phase 05 has to name the cross sections' `quantity` wording.** They
  carry no dispatch message at all today, so the port invents it and it is
  user-visible from the first swap. `"Center-of-mass energies"` is the
  placeholder `test/test_core_dispatch.py` uses.
- **Two Task 1.4 follow-ups ripen inside this project.** The
  [`MASS_E` `nan`](../../../docs/followups/todo/positron-spectrum-nan-at-legacy-electron-mass.md)
  before Phases 05/06, and the
  [scalar-energy contract](../../../docs/followups/todo/model-spectra-reject-scalar-energies.md)
  during 05–06 — Task 3.5 settled the compiled half; what is left is pure
  Python.
- **`release.yml` has no pull-request trigger**, so any future change to
  it needs its own dispatch to be measured at all
  (`../../../docs/agents/lessons.md`
  `[unrun-workflow-cannot-close-a-criterion]`). Phase 07 Task 7.1 rewrites
  it for maturin and inherits that.
