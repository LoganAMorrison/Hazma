# Task 8: Repair A3 — charged-pion forward cone

**Date:** 2026-09-18
**Project:** parity-pinned-defect-repair
**Status:** In Progress — review fixes prepared; updated CI pending
**Plan References:** `../PLAN.md`, Tasks 8 and 9; `../rules.md`
**Related ADRs:** ADR-0001; ADR-0002
**Depends On:** Task 7

## Objective

Integrate the charged-pion photon spectrum over its physical angular
support and declare the resulting corpus changes against the Task 2 oracle.

## Exit Criteria

- Clip the angular quadrature at the widest channel edge without changing
  its tolerance or the signed muon approximation.
- Compare all six affected corpus cases with the independent A3 capture;
  reconcile the scalar mediator's B4 declaration and future B3 rho repair.
- Test restored forward support, the electron-channel sliver, endpoints,
  scalar/array dispatch, and an independent boost invariant.
- Partition quadrature comparisons by scipy's convergence verdict.
- Measure before/after public values; preserve corpus/oracle arrays and
  existing tolerance budgets.
- Demonstrate repair-revert and declaration-widening failures; pass full
  preflight, theory aggregation, and documentation citation checks.

## Inputs Reviewed

- Project plan, rules, working memory, and corpus-repinning specification.
- Task 7 handoff; Task 2 A3 capture, patch, measurements and rho caveat.
- ADR-0001 and ADR-0002; defect blast radius and forward-cone follow-up.
- Agent lessons, environment, preflight, and doc-consistency checklist.
- Photon-pion implementation, tests, and public composition call sites.

## Findings

- The widest rest-frame edge is the electron radiative endpoint,
  `(m_pi^2 - m_e^2)/(2 m_pi) = 69.784260 MeV`. The legacy boosted-muon
  edge is 8.0e-4 MeV narrower and would clip a real electron tail.
- The independently compiled A3 capture agrees with the repaired pion
  within 4.12e-16 relative on macOS; declared rho/vector arrays within
  1.55e-13 there. Linux CI later measured 9.61566611e-12 in vector
  total spectra; see the review-response note.
  A2's signed muon endpoint does not change this comparison because the
  pion's daughter muon is already in flight.
- A3 moves 1,032 scalar positions already covered by B4. Composition is
  required. The sum of the captured A3 spectrum and B4's FSR half differs
  from the jointly integrated production total by up to 3.08e-4 relative,
  within B4's existing 1e-3 budget. At rest the residual is below 6e-16;
  an additional 1e-12 check makes reverting the much smaller A3 visible.
- A3 changes the rho rest blocks too: the daughter pion is boosted there.
  Task 9's earlier disjointness prediction was false. Its B3 factor must
  multiply the A3-corrected spectrum, not the original stored spectrum.
- The canonical Task 8 scope and oracle cover the inner pion integral.
  Task 2's historical note also proposed clipping the outer rho integral;
  that exceeds this oracle. The remaining work now has a separate
  follow-up, linked below, so project close cannot lose it.

## Decisions and Implementation Notes

- Derive the angular lower bound from `E' = E gamma (1 - beta cos(theta))`.
  Clamp it at -1 and return zero when no interval remains. Preserve the
  at-rest, nonpositive-energy and NaN behavior; no integration-tolerance
  changes.
- `PHOTON_ENDPOINT_PIRF` and `charged_pion_cos_min` express the support.
  Rust convergence tests use the same bound as production instead of
  checking the old full-angle integral.
- A3 uses its existing case budgets: 1e-12 for the pion and 1e-9 for
  rho/vector consumers, registered as `A3/nested` with repair label A3.
  A3+B4 retains B4's 1e-3 budget and the strict rest-frame gate.
  The literal allowlist adds 66 A3 arrays and replaces 20 B4 declarations
  with A3+B4. Ten B4 arrays at scalar mass 250 MeV remain B4 alone.
- The independent physics check changes variables to photon energy in
  the pion rest frame. Integrating `(dN/dE')/E'` over its allowed interval
  and dividing by `2 gamma beta` gives MeV^-1. Its scipy convergence
  verdict selects comparisons; all 20 selected grid points converged.
- The original forward-cone follow-up stays in `todo/` until the project's
  coordinated closeout, like earlier repaired defects. The separate
  `rho-photon-outer-boost-misses-support.md` stays open beyond that close.

## Files Changed — PR #97

- `rust/src/kernels/photon_pion.rs`: support bound and live convergence tests.
- `test/test_core_photon_pion.py`: corrected endpoint, captured spot value,
  restored tails, electron-only sliver and independent energy boost.
- `test/parity/deltas.py`, `test/parity/test_parity.py`: A3/A3+B4 allowlist
  and declared-array count.
- `test/parity/test_pion_repair.py`: corpus-zero support, scalar oracle,
  and overlap tests.
- `CHANGELOG.md`, follow-up index and inner/outer follow-ups: measured
  public change and durable remaining work.
- Project plan, blast-radius reference, working memory and this note:
  measured overlap, task status and Task 9 handoff.

## Verification — initial implementation at 278b1ee9

Worktree branch: `codex/parity-pinned-defect-repair/task-8-charged-pion-cone`,
starting at `19c054e3b9a5df9ce1c29b6c91aaae3423674c42`.
Python 3.13.7 in the worktree's own `.venv`. Every Rust change and mutation
was rebuilt with:

```sh
uv pip install --python .venv/bin/python -e . --config-setting build-args="--features test-probes"
.venv/bin/python -c 'import hazma, hazma._core; print(hazma.__file__); print(hazma._core.__file__)'
```

Both import paths resolve inside this task's worktree. No commit or push.
The final rebuild's captured arrays are bit-identical to the first fixed
build, including NaNs compared as equal.

```text
cargo test --manifest-path rust/Cargo.toml --no-default-features --features test-probes photon_pion
13 passed; 0 failed; 0 ignored; 0 measured; 249 filtered out
.venv/bin/pytest -n 0 -q test/test_core_photon_pion.py test/parity/test_pion_repair.py
38 passed in 0.65s
```

Mutation commands ran actual corpus nodes, after rebuilding for the Rust
mutation. Snapshots were restored before the final gate.

| Mutation | Command / observed result |
| --- | --- |
| Replace production lower bound with -1 | `pytest -n 0 -q 'test/parity/test_parity.py::test_entry_point_matches_corpus[spectra.photon.charged_pion[boosted_strong]]' test/parity/test_pion_repair.py::test_scalar_rest_composition_resolves_a3_beneath_b4_budget` — `3 failed in 0.58s` |
| Expand only pion boosted-strong values from its 118 moved positions to include unchanged position 0 | Same corpus node alone — `1 failed in 0.57s`: declaration names one position its relation does not move |

An initial widening attempt mutated the shared Delta and hit a scalar
array bounds error; that is not the widening evidence above. Early
preflight runs found lint issues and an accidental Rust path in the
Python path list; the corrected final gate is recorded below.

```text
python test/parity/generate.py --check
corpus OK: 41 cases / 1580 arrays match the manifest (generated at 010747c6125d, kernel digest f5e6e269be47)
python test/parity/oracles/capture.py --check
oracles OK: 4 defects / 940 arrays match the manifest (corpus manifest f476fb420caf)
```

Final preflight command (Rust gates run automatically; `--paths` names
Python lint inputs):

```sh
PATH="$PWD/.venv/bin:$PATH" scripts/agents/preflight.sh \
  --paths "test/parity/deltas.py test/parity/test_parity.py test/parity/test_pion_repair.py test/test_core_photon_pion.py" \
  --md "CHANGELOG.md docs/followups/README.md docs/followups/todo/charged-pion-photon-spectrum-misses-the-forward-cone.md docs/followups/todo/rho-photon-outer-boost-misses-support.md projects/parity-pinned-defect-repair/PLAN.md projects/parity-pinned-defect-repair/references/defect-blast-radius.md projects/parity-pinned-defect-repair/task-notes/README.md projects/parity-pinned-defect-repair/task-notes/task-8-charged-pion-cone.md"
```

```text
preflight — /Users/logan.morrison/dev/Hazma/.codex/worktrees/parity-pinned-defect-repair/task-8-charged-pion-cone (base origin/master)
-------------------------------------------------------------------
PASS   black --check           test/parity/deltas.py test/parity/test_parity.py test/parity/test_pion_repair.py test/test_core_photon_pion.py
PASS   isort --check-only      0 new, 2 fixed (0 pre-existing at 19c054e3b9a5)
PASS   ruff check              0 new, 2 fixed (0 pre-existing at 19c054e3b9a5)
PASS   cargo fmt --check       rust/
PASS   cargo clippy            rust/
PASS   cargo test              rust/
PASS   pytest                  2302 passed, 16 skipped, 1 warning, 37 subtests passed in 26.99s
PASS   import hazma            version 2.2.0
PASS   markdownlint            CHANGELOG.md docs/followups/README.md docs/followups/todo/charged-pion-photon-spectrum-misses-the-forward-cone.md docs/followups/todo/rho-photon-outer-boost-misses-support.md projects/parity-pinned-defect-repair/PLAN.md projects/parity-pinned-defect-repair/references/defect-blast-radius.md projects/parity-pinned-defect-repair/task-notes/README.md projects/parity-pinned-defect-repair/task-notes/task-8-charged-pion-cone.md
SKIP   version bump            not a closing PR (pass --closing)
PASS   forbidden tokens        none added
-------------------------------------------------------------------
RESULT: PASS
```

The full pytest gate includes theory aggregation and the parity suite.
Subsequent changes only append this evidence and wrap plan prose;
Markdown and citation checks were repeated after those edits.

## Numerical impact — PR #97

Capture the same corpus and public grids from the unrepaired build, then
rebuild with the repair and repeat. The task used the scratch script below
as `/tmp/hazma-task8-capture.py` and outputs `/tmp/hazma-task8-before.npz`
and `/tmp/hazma-task8-after.npz`. A final capture after the last Rust edit
was bit-identical to the latter. The recipe is reproduced so the scratch
files are not a dependency of this record.

```python
import sys
from pathlib import Path
import numpy as np
sys.path[:0] = [str(Path.cwd()), str(Path.cwd()/'test/parity')]
import cases, generate, oracle_reference
from hazma import spectra
from hazma.parameters import charged_pion_mass as m
out = {}
for name in sorted({name for name, block in oracle_reference._blocks('A3')}):
    case = cases.build_cases()[name]
    for block in case.blocks:
        values, raised = generate.evaluate_block(case.resolve(), block)
        for suffix, value in values.items():
            if suffix not in ('grid', 'scalar_grid'):
                out[f'{name}|{block.label}|{suffix}'] = value
for parent in (m, m*(1+1e-12), 200., 500., 1000., 1396., 5000.):
    e = np.geomspace(0.01, parent*1.01, 2001)
    out[f'public|{parent}'] = spectra.dnde_photon_charged_pion(e, parent)
for cme in (2*m, 2792.):
    e = np.geomspace(0.01, cme/2*1.01, 301)
    out[f'nbody|{cme}'] = spectra.dnde_photon(e, cme, ['pi', 'pi'])
np.savez(sys.argv[1], **out)
print('captured', len(out), 'arrays', sum(a.size for a in out.values()), 'values')
```

For each array count the complement of
`(before == after) | (isnan(before) & isnan(after))`; take the maximum
absolute difference on those positions. All spectra are in MeV^-1.

| Corpus case | Moved / evaluated | Maximum absolute change |
| --- | --- | --- |
| scalar photon decay | 1032 / 8610 | 4.8510951583882744e-9 |
| vector photon decay | 2013 / 29295 | 7.333655980956636e-10 |
| vector photon decay point entry | 2013 / 29295 | 7.333655980956636e-10 |
| charged pion | 245 / 1500 | 7.853305686672297e-7 |
| charged rho | 528 / 1435 | 3.0438408564481545e-10 |
| neutral rho | 528 / 1435 | 9.116134384736949e-8 |
| Total | 6359 / 71570 | — |

| Public pion energy, MeV | Moved / evaluated | Maximum absolute change |
| --- | --- | --- |
| 139.57039 | 0 / 2001 | 0 |
| 139.57039 times (1+1e-12) | 0 / 2001 | 0 |
| 200 | 363 / 2001 | 1.917867355460392e-9 |
| 500 | 720 / 2001 | 7.726457741979919e-9 |
| 1000 | 923 / 2001 | 2.6969285154466844e-7 |
| 1396 | 1009 / 2001 | 1.1265371496816012e-6 |
| 5000 | 1301 / 2001 | 2.5408050057561016e-5 |

The generic two-pion spectrum changes at 151/301 points for 2792 MeV
center-of-mass energy, maximum 2.2300137083607835e-6 MeV^-1, and at
0/301 points for `2 m_pi`. Additional consumer captures use this script,
run with the same before/after rebuild protocol:

```python
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
from hazma.scalar_mediator import ScalarMediator
from hazma.vector_mediator import VectorMediator
from hazma.single_channel import SingleChannelAnn
from hazma.parameters import charged_pion_mass as m
models = {'scalar': ScalarMediator(1396.,900.,1.,1e-3,1e-3,1e-3,1e4),
          'vector': VectorMediator(1396.,900.,1.,.1,-.1,.1,.1,.1)}
out={}
for cme in (2*m,2792.):
    energies=np.geomspace(.01,cme/2*1.01,301)
    for name,model in models.items():
        out[f'{name}|{cme}']=model.dnde_pipi(energies,cme,spectrum_type='decay')
    model=SingleChannelAnn(cme/2,'pi pi',1.)
    out[f'single|{cme}']=model.total_spectrum(energies,cme)
np.savez(sys.argv[1],**out)
print({k:(v.size,float(np.max(v))) for k,v in out.items()})
```

All three consumers change at 151/301 points at 2792 MeV, by at most
2.2300137083607835e-6 MeV^-1. All three threshold grids remain identical.
These are representative consumer measurements, not a claim that every
model parameter or multiparticle final state was sampled.

The independent reference comparison uses `oracle_reference._blocks('A3')`
and the corresponding captured arrays. For the 20 scalar composite arrays,
add `DELTA_MODELS['B4'].relation.term(fn, block)` first. The worst residual
is 3.075299199147904e-4 on the scalar composite, 1.548873091057888e-13
on declared vector arrays, 6.941931754564825e-14 on charged rho,
4.2363958088404564e-16 on neutral rho, and 4.114862711080301e-16 on pion.
The unchanged vector muon-only arrays retain their original case budget.

## Open Questions

PR #97's pushed head has failing Linux CI. Local review fixes and their
verification are recorded in
[`task-8-review-response.md`](task-8-review-response.md); completion awaits
publishing those changes and a green CI run. The outer rho support repair
is tracked in
[`rho-photon-outer-boost-misses-support.md`](../../../docs/followups/todo/rho-photon-outer-boost-misses-support.md).

## Plan Impact

**Impact Level:** Plan and reference corrected; no new ADR.

Task 8 explicitly scopes the captured inner repair. Task 9's disjointness
prediction is replaced by measured overlap and the existing ADR-0001
composition rule. No dependency or phase boundary changes. The original
corpus, oracle arrays and tolerance file are untouched.

## Stale-state sweep — initial implementation at 278b1ee9

These are folded outputs from actual commands, one filename per match
set. Dispositions apply per claim: edited live instructions are listed
explicitly; dated measurements and unrelated numbers are retained.
The note itself is included because its quoted commands match.

### Pre-fix occurrences

The initial targeted sweep was saved before documentation edits. The
complete numeric baseline was subsequently replayed from the unchanged
trunk revision, not represented as a contemporaneous working-tree read:

```sh
git grep -n -E 'the_forward_cone_is_a_hard_zero_the_quadrature_invented|Seven repairs|(^|[^[:alnum:]_])99([^[:alnum:]_]|$)' origin/master -- projects/ docs/ hazma/ test/ .claude/ .codex/ README.md CHANGELOG.md
```

Baseline: 45 files, 162 rows. The live count in
`test/parity/test_parity.py` and the working-memory count/repair summary
were EDITED; the old defect assertion in `test/test_core_photon_pion.py`
was REPLACED. Other matches were KEPT as dated records or unrelated
numbers (including decimal fractions and data tables). The baseline
command includes the immutable corpus manifest; its entries are KEPT.

### Post-fix occurrences

```sh
rg -n --hidden 'the_forward_cone_is_a_hard_zero_the_quadrature_invented|Seven repairs|\b99\b' projects/ docs/ hazma/ test/ .claude/ .codex/ README.md CHANGELOG.md rust/
```

Folded filenames, all remaining matching claims KEPT: historical evidence,
unrelated numeric values, or this sweep's own quoted prior state.

```text
hazma/gamma_ray_data/A_eff/grams_upgrade.dat
docs/followups/README.md
hazma/form_factors/vector/testdata/k0_k0.json
projects/cython-to-rust/task-notes/phase-03/task-3.3-quadpack.md
test/parity/test_delta_models.py
test/test_core_photon_muon.py
rust/src/kernels/photon_muon.rs
projects/cython-to-rust/references/numerics-replacements.md
projects/parity-pinned-defect-repair/PLAN.md
projects/cython-to-rust/task-notes/phase-04/task-4.5-photon-rho.md
rust/src/vector_mediator.rs
test/test_core_dispatch.py
projects/cython-to-rust/task-notes/phase-04/task-4.4-photon-pion.md
test/vector_mediator/herwig4dm/4pi/run.neutral.3.99.dat
projects/parity-pinned-defect-repair/task-notes/task-7-photon-muon-endpoint.md
hazma/pbh_data/pbh_secondary_spectra.csv
hazma/gamma_ray_data/A_eff/mast.dat
test/test_core_mediator_positron.py
CHANGELOG.md
test/parity/test_parity.py
rust/src/constants.rs
test/test_utils.py
docs/agents/lessons-examples.md
test/test_relic_density.py
projects/cython-to-rust/task-notes/phase-05/task-5.2-scalar-xs.md
test/test_core_positron_pion.py
projects/parity-pinned-defect-repair/task-notes/README.md
rust/src/quad.rs
rust/src/boost.rs
hazma/relic_density/smdof.dat
hazma/pbh_data/pbh_primary_spectra_bh.csv
hazma/gamma_ray_data/A_eff/pangu.dat
projects/cython-to-rust/task-notes/phase-01/task-1.3-test-wiring.md
test/test_core_positron_muon.py
test/test_core_quad.py
test/test_core_photon_pion.py
test/vector_mediator/herwig4dm/4pi/run.neutral.2.99.dat
hazma/gamma_ray_data/A_eff/grams.dat
docs/followups/done/thermal-cross-section-quadrature-never-converges.md
projects/cython-to-rust/phases/phase-03-numerics-foundation.md
hazma/pbh_data/pbh_secondary_spectra_bh.csv
test/vector_mediator/herwig4dm/4pi/run.neutral.0.99.dat
docs/followups/todo/eta-prime-two-photon-line-missing-factor-two.md
hazma/gamma_ray_data/energy_res/amego.dat
rust/src/kernels/photon_tables.rs
test/vector_mediator/herwig4dm/4pi/run.neutral.1.99.dat
projects/cython-to-rust/task-notes/numerical-impact.md
projects/parity-pinned-defect-repair/task-notes/task-13-thermal-quadrature.md
projects/cython-to-rust/task-notes/phase-04/task-4.3-photon-muon.md
projects/cython-to-rust/adrs/ADR-0002-license-clean-numerics.md
projects/parity-pinned-defect-repair/task-notes/task-8-charged-pion-cone.md
```

Current identifiers and remaining-work links:

```sh
rg -n --hidden 'PHOTON_ENDPOINT_PIRF|charged_pion_cos_min|_A3_B4|test_scalar_rest_composition_resolves_a3_beneath_b4_budget|rho-photon-outer-boost-misses-support' projects/ docs/ hazma/ test/ .claude/ .codex/ README.md CHANGELOG.md rust/
```

```text
docs/followups/README.md
test/parity/test_pion_repair.py
docs/followups/todo/charged-pion-photon-spectrum-misses-the-forward-cone.md
projects/parity-pinned-defect-repair/PLAN.md
rust/src/kernels/photon_pion.rs
test/test_core_photon_pion.py
test/parity/deltas.py
projects/parity-pinned-defect-repair/task-notes/README.md
projects/parity-pinned-defect-repair/task-notes/task-8-charged-pion-cone.md
```

All KEPT: active implementation, tests, measured handoff or follow-up.
The new outer-rho follow-up's own filename does not appear in its body;
its inbound links are included above and its file exists in the diff.

```sh
rg -n --hidden 'positions.*disjoint|disjoint.*position|Task 8 must|A3 will reach|Tasks 7 and 8 also' projects/ docs/ hazma/ test/ .claude/ .codex/ README.md CHANGELOG.md
```

Folded output: project PLAN, corpus-repinning specification,
`test/parity/test_pion_repair.py`, and this note. KEPT: the PLAN and test
reject the disjointness assumption; the specification permits either
proven disjointness or composition. No live instruction still assumes
A3 and B3 have disjoint positions.

### Forward-looking and line-citation sweeps

```sh
rg -n 'Task [0-9]+ will|will be added|still pending|today: ?stub|currently|In Progress' projects/parity-pinned-defect-repair/ hazma/
```

```text
hazma/theory/_theory_constrain.py
projects/parity-pinned-defect-repair/task-notes/task-10a-neutrino-pion-line.md
projects/parity-pinned-defect-repair/task-notes/task-6-phi-lines.md
projects/parity-pinned-defect-repair/task-notes/task-7-photon-muon-endpoint.md
projects/parity-pinned-defect-repair/PLAN.md
projects/parity-pinned-defect-repair/task-notes/task-8-charged-pion-cone.md
projects/parity-pinned-defect-repair/task-notes/task-3-closed-form-deltas.md
projects/parity-pinned-defect-repair/task-notes/task-4-boost-window.md
projects/parity-pinned-defect-repair/task-notes/task-13-thermal-quadrature.md
projects/parity-pinned-defect-repair/task-notes/README.md
projects/parity-pinned-defect-repair/task-notes/_template.md
```

KEPT: the project remains active, future tasks remain pending, and dated
notes retain their original status context. Task 8's status is Complete.

```sh
rg -n 'photon_pion\.rs:[0-9]+|test_core_photon_pion\.py:[0-9]+|deltas\.py:[0-9]+' projects/parity-pinned-defect-repair/ docs/
```

Folded output: `task-notes/task-3-closed-form-deltas.md`, two historical
sweep-table rows, KEPT. The regex in this note does not itself match a
numeric citation. No active affected line citation was found.
The citation checker ran with the explicit eight Markdown paths from
the preflight command, because uncommitted work is invisible to its
`--changed-vs` mode:

```text
docs scanned: 8
in-repo citations checked: 0
external citations skipped: 0
out-of-range or ambiguous: NONE
```

### Count and numerical-impact checks

| Claim | Canonical command / calculation | Actual |
| --- | --- | --- |
| Changed artifacts | `git diff --name-only origin/master` | 13 files, including new files added with intent-to-add |
| Declared arrays | `Counter(d.repair for d in deltas.DECLARED_DELTAS.values())` | 165: A3 66, A1 44, A3+B4 20, B4 10, A1+B1 6, A1+B2 6, B5 6, B6 6, A2 1 |
| A3 corpus reach | Before/after capture recipe above | 6 cases; 6359 of 71570 values move |
| Scalar overlap | Compare A3 capture to stored values where B4 term is nonzero | 1032 positions; 20 composite arrays |
| Independent energy grid | Four parent energies times five endpoint fractions in the boost test | 20 scipy-success comparisons |
| Public grid and consumers | Capture recipes and Numerical impact tables above | 7 pion grids of 2001 points; each consumer has 2 grids of 301 points |
| Preserved artifacts | `git diff --stat -- test/parity/data test/parity/oracles/data test/parity/tolerances.py` | Empty |
| Task readiness | `scripts/agents/resolve_task.py --project parity-pinned-defect-repair` | Task 9 ready |

### Exit Criteria to evidence

| Criterion | Test or artifact |
| --- | --- |
| Widest support and preserved approximation | Rust lower bound; electron-sliver and existing rest/NaN tests |
| Six independent oracle cases and overlap | Full parity gate; A3/A3+B4 declarations; strict scalar rest checks |
| Restored support, dispatch, endpoint and physics invariant | Forward-cone, electron-sliver, corpus-zero and energy-integral tests |
| Scipy convergence partition | Energy-reference success verdict; Rust production-bound convergence tests |
| Measured public impact and preserved corpus | Before/after tables, integrity checks and empty data/tolerance diff |
| Mutations and all gates | Measured three-test revert failure, one-test widening failure, preflight and citation outputs |

Task-note consistency: Complete matches the working-memory row; every
criterion maps above and every Files Changed entry belongs to the diff.
The plan remains In Progress. Sorted sweep outputs were re-run after this
block was pasted; filename sets and disposition conclusions are unchanged.

## Review follow-up

[PR #97 review response](task-8-review-response.md) records the corrected
consumer budgets, Linux replay, convergence remeasurement and fresh gates.
This note's original verification and sweep sections remain the record of
278b1ee9; their Complete/readiness statements preceded the CI failure.

## Handoff to Next Task

Read `../PLAN.md` Task 9 and this note's Findings. Both rho rest arrays
already carry A3, so multiply the A3 oracle by photon energy for the B3
prediction. Do not apply B3 solely to the old corpus or delete A3's gate.
Keep the separate outer-rho follow-up open at project close.
