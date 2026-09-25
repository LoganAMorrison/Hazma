# Task 7: Repair A2 — muon photon endpoint

**Date:** 2026-09-18
**Project:** parity-pinned-defect-repair
**Status:** Complete
**Plan References:** `../PLAN.md`, Task 7; `../rules.md`
**Related ADRs:** ADR-0001 (declared deltas); ADR-0002 (signed endpoint)
**Depends On:** Tasks 1 and 2

## Objective

Restore the muon rest-frame photon spectrum to its kinematic endpoint
and settle the treatment of its signed analytic approximation.

## Exit Criteria

- Correct the guard to `y >= 1 - r`, with `r = (m_e/m_mu)^2`.
- Decide and document treatment of the formula's negative endpoint sliver.
- Declare exactly the four moved positions in the muon `rest` block;
  compare with the independent Task 2 Cython capture.
- Re-derive the Rust composition radius and measure the six candidate
  composed corpus cases before and after the repair.
- Preserve the in-flight boost-integral identity and test the restored
  positive interval, endpoint, scalar and array dispatch.
- Demonstrate repair-revert and declaration-widening mutations fail.
- Preserve the committed corpus and tolerance budgets; run the theory
  aggregation gate, full preflight, and documentation citation checks.

## Inputs Reviewed

- Project plan, rules, working memory, Task 2 endpoint finding, ADR-0001.
- Corpus repinning specification and defect blast-radius reference.
- Agent lessons, environment, preflight, and doc-consistency checklist.
- Kuno and Okada, [hep-ph/9909265v1](https://arxiv.org/pdf/hep-ph/9909265),
  Eqs. (53)–(56); the endpoint and the photon approximation.
- Rust muon kernel, pion/rho and mediator callers; Python physics tests.

## Findings

- Eq. (53) gives `1 - r` as the endpoint. `1 - sqrt(r)` separates two
  electron-energy ranges. Eqs. (54)–(56) neglect mass-suppressed terms;
  their negative endpoint values are not a physical negative yield.
- With `t = 1 - y`, the bracket is
  the polynomial-logarithm form recorded in ADR-0002. At `t = r` the
  logarithm vanishes and the
  bracket is negative: this is not floating-point noise. ADR-0002 records
  why clipping only the rest branch would break the boost identity.
- The Rust rest guard compares `emu - MASS_MU` to an absolute epsilon.
  `photon_pion.rs` supplies `ENG_MU_PIRF = 109.77820123634007` MeV;
  both rho kernels compose that pion. `scalar_decay_photon.rs` and
  `mediator_tables.rs` supply half the mediator mass; corpus masses
  250, 550 and 900 MeV therefore never reach the rest branch. This is a
  statement about the corpus, not every conceivable mediator parameter.
- The seven-case capture changes four of 72,940 evaluated values.
  The muon case contributes 1,370; the six composed cases contribute
  71,570 unchanged values. Scalar probes in the corpus miss the restored
  region, so the independent Python test explicitly checks both dispatch
  routes there.

## Decisions and Implementation Notes

- Use the existing `ONE_MINUS_R` constant, removing the obsolete rest
  endpoint constant and its defect-pinning test. Remove the duplicated
  test-only rest formula; the boost-integral test now uses production.
- Retain and document the signed approximation (ADR-0002). Four reference
  points are independently evaluated from the published J+/J- forms at
  60 decimal digits, covering both the positive interval and negative
  tail. Their 1e-9 relative budget covers cancellation in the original
  polynomial; no absolute floor is used.
- A2 declares only positions `(161, 162, 163, 164)` of
  `spectra.photon.muon/rest/values`. The Cython oracle agrees bit for bit
  here. The relation reserves the existing off-platform SPECFUN budget
  of 1e-9 because the endpoint polynomial cancels; no existing budget or
  array is edited. Undeclared positions retain their original budgets.
- The follow-up stays in `todo/` with an explicit repaired status until
  Task 12's coordinated move, matching the project's existing lifecycle.

## Files Changed

- `rust/src/kernels/photon_muon.rs`: guard, comments, numerical tests,
  production boost-integral invariant, removal of duplicate formula.
- `hazma/spectra/_photon/__init__.py`: public approximation limitation.
- `test/test_core_photon_muon.py`: corrected endpoint, restored interval,
  independent reference values, scalar/array checks.
- `test/parity/deltas.py`, `test/parity/test_parity.py`: A2 declaration
  and total declared-array count 98 → 99.
- `CHANGELOG.md`, muon-endpoint follow-up, project plan, blast-radius
  reference, working memory, this note, and ADR-0002: impact and decision.

## Verification

The isolated worktree starts at `1824829c932e` on
`codex/parity-pinned-defect-repair/task-7-photon-muon-endpoint`.
A local `.venv` uses Python 3.13.7; each Rust mutation was followed by
`uv pip install --python .venv/bin/python -e .
--config-setting build-args="--features test-probes"`.
Both `hazma.__file__` and `hazma._core.__file__` resolved into this
worktree. Nothing was committed or pushed.

```text
pytest -n 0 -q test/test_core_photon_muon.py 'test/parity/test_parity.py::test_entry_point_matches_corpus[spectra.photon.muon[rest]]'
31 passed in 0.50s
```

Mutations use the same explicit corpus node above:

| Mutation | Observed result |
| --- | --- |
| Change only the rest guard back to `1.0 - MASS_RATIO`, rebuild | `1 failed in 0.70s`: four zeros fail against the four signed oracle values |
| Add position 160 to A2's tuple, repaired build | `1 failed in 0.68s`: declaration names one position its relation does not move |

Both mutations were restored before the final gate. An initial mistyped
pytest node selected no tests and is not counted as mutation evidence.
The first preflight passed all Python tests but caught excessive Rust
literal precision and Markdown line lengths; those were corrected.

```sh
PATH="$PWD/.venv/bin:$PATH" scripts/agents/preflight.sh \
  --paths "hazma/spectra/_photon/__init__.py test/parity/deltas.py test/parity/test_parity.py test/test_core_photon_muon.py" \
  --md "CHANGELOG.md docs/followups/done/photon-muon-rest-frame-endpoint-uses-the-wrong-power-of-r.md projects/parity-pinned-defect-repair/PLAN.md projects/parity-pinned-defect-repair/references/defect-blast-radius.md projects/parity-pinned-defect-repair/task-notes/README.md projects/parity-pinned-defect-repair/task-notes/task-7-photon-muon-endpoint.md projects/parity-pinned-defect-repair/adrs/ADR-0002-retain-the-signed-muon-endpoint-approximation.md"
```

[Task 12 of `parity-pinned-defect-repair` (2026-09-24) repointed the
`docs/followups/todo/` paths in the block above to `done/`, where the
follow-ups moved at that project's close; when the command ran they
were under `todo/`.]

```text
preflight — /Users/logan.morrison/dev/Hazma/.codex/worktrees/parity-pinned-defect-repair/task-7-photon-muon-endpoint (base origin/master)
-------------------------------------------------------------------
PASS   black --check           hazma/spectra/_photon/__init__.py test/parity/deltas.py test/parity/test_parity.py test/test_core_photon_muon.py
PASS   isort --check-only      0 new, 2 fixed (0 pre-existing at 1824829c932e)
PASS   ruff check              0 new, 2 fixed (0 pre-existing at 1824829c932e)
PASS   cargo fmt --check       rust/
PASS   cargo clippy            rust/
PASS   cargo test              rust/
PASS   pytest                  2294 passed, 16 skipped, 1 warning, 37 subtests passed in 26.87s
PASS   import hazma            version 2.2.0
PASS   markdownlint            CHANGELOG.md docs/followups/done/photon-muon-rest-frame-endpoint-uses-the-wrong-power-of-r.md projects/parity-pinned-defect-repair/PLAN.md projects/parity-pinned-defect-repair/references/defect-blast-radius.md projects/parity-pinned-defect-repair/task-notes/README.md projects/parity-pinned-defect-repair/task-notes/task-7-photon-muon-endpoint.md projects/parity-pinned-defect-repair/adrs/ADR-0002-retain-the-signed-muon-endpoint-approximation.md
SKIP   version bump            not a closing PR (pass --closing)
PASS   forbidden tokens        none added
-------------------------------------------------------------------
RESULT: PASS
corpus OK: 41 cases / 1580 arrays match the manifest (generated at 010747c6125d, kernel digest f5e6e269be47)
oracles OK: 4 defects / 940 arrays match the manifest (corpus manifest f476fb420caf)
```

[Task 12 of `parity-pinned-defect-repair` (2026-09-24) repointed the
`docs/followups/todo/` paths in the block above to `done/`, where the
follow-ups moved at that project's close; when the command ran they
were under `todo/`.]

The integrity lines are from `python test/parity/generate.py --check`
and `python test/parity/oracles/capture.py --check`. The bare pytest gate
includes `test/test_theory_aggregation.py` and the entire parity suite.

## Numerical impact

Capture the defective build before editing, and the repaired build after
reinstalling. Save the following as a scratch `capture.py` and run it
from the worktree as `python capture.py /tmp/before.npz` and
`python capture.py /tmp/after.npz`. The task used the scratch filenames
`/tmp/hazma-task7-before.npz` and `/tmp/hazma-task7-after.npz`.

```python
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / 'test/parity'))
import cases
import generate
import oracle_reference
from hazma import spectra
from hazma.parameters import muon_mass as m
names = sorted({name for name, block in oracle_reference._blocks('A2')})
out = {}
for name in names:
    case = cases.build_cases()[name]
    for block in case.blocks:
        values, raised = generate.evaluate_block(case.resolve(), block)
        for suffix, value in values.items():
            if suffix not in ('grid', 'scalar_grid'):
                out[f'{name}|{block.label}|{suffix}'] = value
energies = np.linspace(52.57368777695, 52.8279515698, 200001)
for parent in (m, m*(1+1e-12), 110., 500.):
    out[f'public|{parent}'] = spectra.dnde_photon_muon(energies, parent)
for cme in (2*m, 2.5*m):
    out[f'nbody|{cme}'] = spectra.dnde_photon(energies[::200], cme, ['mu', 'mu'])
np.savez(sys.argv[1], **out)
print('captured', len(out), 'arrays across', len(names), 'corpus cases')
```

Compare arrays with `(before == after) | (isnan(before) & isnan(after))`;
count its complement and take the maximum absolute difference there.
The public grid is 200,001 equally spaced energies from
52.57368777695 to 52.8279515698 MeV; the last point is above support and
stays zero. The two-body grid takes every 200th point, 1,001 in total.
The exact before/after report and the independent root/integral probes:

```text
spectra.photon.muon|rest|values moved 4 max abs 2.9958736220958345e-09
public|105.6583745 moved 200000 max abs 5.335612099422552e-07
nbody|211.316749 moved 1000 max abs 1.0671224198845104e-06
group totals {'mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum': [0, 8610, 0.0], 'mediator_spectra.vector.photon.dnde_decay_v': [0, 29295, 0.0], 'mediator_spectra.vector.photon.dnde_decay_v_pt': [0, 29295, 0.0], 'spectra.photon.charged_pion': [0, 1500, 0.0], 'spectra.photon.charged_rho': [0, 1435, 0.0], 'spectra.photon.muon': [4, 1370, 2.9958736220958345e-09], 'spectra.photon.neutral_rho': [0, 1435, 0.0], 'public': [200000, 800004, 5.335612099422552e-07], 'nbody': [1000, 2002, 1.0671224198845104e-06]}
cut, endpoint, restored width, zero, negative width: 52.573687776949996 52.82795156979882 0.2542637928488247 52.808176076465415 0.019775493333405336
minimum: -6.43368312079504e-09 at 52.82145686254833
restored integral photons: (5.4453837534795933e-08, 3.9485902677046345e-18)
negative integral photons: (-8.974824740291726e-11, 4.049444291860199e-24)
```

The root uses `brentq(f, 52.8, 52.82, xtol=1e-13)`; the minimum uses
bounded `minimize_scalar` over the root to `endpoint - 1e-10` with
`xatol=1e-13`. Integrals use `quad` with `epsrel=1e-10`, `epsabs=1e-20`
over `[old_cut, endpoint]` and `epsabs=1e-22` over `[root, endpoint]`.
Here `f(E) = dnde_photon_muon(E, 105.6583745)` in MeV^-1; integrating
in MeV gives photons per decay. The net restored yield differs from
Task 2's 5.454359e-8 positive-area estimate by the signed negative area.

The four pinned reference values in the Rust and Python tests can be
recomputed without the kernel using the paper's equations:

```python
import mpmath as mp
mp.mp.dps = 60
m = mp.mpf("105.6583745")
r = (mp.mpf("0.5109989461") / m)**2
alpha = 1 / mp.mpf("137.035999084")
for energy in ("52.6", "52.7", "52.81", "52.82"):
    y = 2 * mp.mpf(energy) / m
    t = 1 - y
    log = mp.log(t/r)
    jp = alpha*t/(6*mp.pi) * (
        3*log - mp.mpf(17)/2 + (-3*log + 7)*t
        + (2*log - mp.mpf(13)/3)*t*t)
    jm = alpha*t*t/(6*mp.pi) * (
        3*log - mp.mpf(93)/12 + (-4*log + mp.mpf(29)/3)*t
        + (2*log - mp.mpf(55)/12)*t*t)
    print(float(4*(jp + jm)/(m*y)))
```

The entire muon rest block equals the Task 2 capture bit for bit.
Only the four declared positions differ from the original corpus.
No change to `test/parity/data/`, `test/parity/oracles/data/`, or
`test/parity/tolerances.py`.

## Open Questions

None for Task 7. A finite-mass replacement of the approximation would be
new physics work affecting both branches, outside this guard repair.

## Plan Impact

**Impact Level:** ADR-0002 and Task 7 plan clarification.

The signed-endpoint convention is now explicit, the composition claim is
scoped to corpus grids, and the plan distinguishes the signed corpus delta
from the positive off-corpus interval. No new task or dependency.

## Stale-state sweep

Commands were run after the final prose edits. Outputs below are folded
from actual line-numbered matches to one row per file, with dispositions;
this note is included when its own command text matches. Closed project
records and prior task measurements are KEPT as history.

### Pre-fix occurrences

The old-identifier command below was first run before the code edit. Its
18 matches were in these files (folded from the original output):

```text
docs/followups/done/photon-muon-rest-frame-endpoint-uses-the-wrong-power-of-r.md — EDITED: repaired assertions or follow-up
projects/cython-to-rust/task-notes/phase-04/README.md — KEPT: dated port history
projects/cython-to-rust/task-notes/phase-04/task-4.3-photon-muon.md — KEPT: dated port history
projects/cython-to-rust/task-notes/phase-04/task-4.4-photon-pion.md — KEPT: dated port history
test/test_core_photon_muon.py — EDITED: repaired assertions or follow-up
```

[Task 12 of `parity-pinned-defect-repair` (2026-09-24) repointed the
`docs/followups/todo/` paths in the block above to `done/`, where the
follow-ups moved at that project's close; when the command ran they
were under `todo/`.]

### Post-fix occurrences

#### Old identifier / defect prose

```sh
rg -n --hidden 'Y_MAX|the_two_kinematic_edges_are_different_constants|the_two_branches_disagree_about_the_rest_frame_endpoint|test_a_muon_at_rest_stops_at_the_shipped_cut|test_the_rest_frame_cut_is_short_of_the_kinematic_endpoint|rest_frame_to_the_true_endpoint|endpoint defect' projects/ docs/ hazma/ test/ .claude/ .codex/ README.md CHANGELOG.md
```

```text
projects/cython-to-rust/task-notes/phase-04/README.md — KEPT: historical record / sweep evidence
projects/cython-to-rust/task-notes/phase-04/task-4.3-photon-muon.md — KEPT: historical record / sweep evidence
projects/cython-to-rust/task-notes/phase-04/task-4.4-photon-pion.md — KEPT: historical record / sweep evidence
projects/parity-pinned-defect-repair/task-notes/task-7-photon-muon-endpoint.md — KEPT: historical record / sweep evidence
```

#### Current identifiers

```sh
rg -n --hidden 'ONE_MINUS_R|_A2|dnde_photon_muon_rest_frame|the_rest_frame_reaches_the_common_endpoint_without_clipping|test_the_signed_endpoint_matches_the_published_approximation|the_in_flight_signed_endpoint_residual_is_bounded' projects/ docs/ hazma/ test/ rust/ .claude/ .codex/ README.md CHANGELOG.md
```

```text
docs/followups/done/photon-muon-rest-frame-endpoint-uses-the-wrong-power-of-r.md — KEPT: current implementation, decision, or historical reference
projects/cython-to-rust/task-notes/phase-04/task-4.3-photon-muon.md — KEPT: current implementation, decision, or historical reference
projects/parity-pinned-defect-repair/task-notes/task-7-photon-muon-endpoint.md — KEPT: current implementation, decision, or historical reference
rust/src/kernels/photon_muon.rs — KEPT: current implementation, decision, or historical reference
test/parity/deltas.py — KEPT: current implementation, decision, or historical reference
test/parity/oracles/patches/A2-muon-rest-frame-endpoint.patch — KEPT: current implementation, decision, or historical reference
test/test_core_photon_muon.py — KEPT: current implementation, decision, or historical reference
```

[Task 12 of `parity-pinned-defect-repair` (2026-09-24) repointed the
`docs/followups/todo/` paths in the block above to `done/`, where the
follow-ups moved at that project's close; when the command ran they
were under `todo/`.]

#### Forward-looking phrases

```sh
rg -n 'Task [0-9]+ will|will be added|still pending|today: ?stub|currently|In Progress' projects/parity-pinned-defect-repair/ hazma/
```

```text
hazma/theory/_theory_constrain.py — KEPT: project remains active, future task, or dated record
projects/parity-pinned-defect-repair/PLAN.md — KEPT: project remains active, future task, or dated record
projects/parity-pinned-defect-repair/task-notes/README.md — KEPT: project remains active, future task, or dated record
projects/parity-pinned-defect-repair/task-notes/_template.md — KEPT: project remains active, future task, or dated record
projects/parity-pinned-defect-repair/task-notes/task-10a-neutrino-pion-line.md — KEPT: project remains active, future task, or dated record
projects/parity-pinned-defect-repair/task-notes/task-13-thermal-quadrature.md — KEPT: project remains active, future task, or dated record
projects/parity-pinned-defect-repair/task-notes/task-3-closed-form-deltas.md — KEPT: project remains active, future task, or dated record
projects/parity-pinned-defect-repair/task-notes/task-4-boost-window.md — KEPT: project remains active, future task, or dated record
projects/parity-pinned-defect-repair/task-notes/task-6-phi-lines.md — KEPT: project remains active, future task, or dated record
projects/parity-pinned-defect-repair/task-notes/task-7-photon-muon-endpoint.md — KEPT: project remains active, future task, or dated record
```

#### Numeric prior count

```sh
rg -n --hidden '\b98\b' projects/parity-pinned-defect-repair/ docs/ hazma/ test/ .claude/ .codex/ README.md CHANGELOG.md
```

```text
CHANGELOG.md — KEPT: historical count or unrelated numeric value
docs/agents/lessons-examples.md — KEPT: historical count or unrelated numeric value
docs/followups/done/charged-pion-photon-spectrum-misses-the-forward-cone.md — KEPT: historical count or unrelated numeric value
docs/followups/todo/kallen-under-sqrt-remaining-call-sites.md — KEPT: historical count or unrelated numeric value
hazma/pbh_data/pbh_primary_spectra_bh.csv — KEPT: historical count or unrelated numeric value
hazma/pbh_data/pbh_secondary_spectra.csv — KEPT: historical count or unrelated numeric value
hazma/pbh_data/pbh_secondary_spectra_bh.csv — KEPT: historical count or unrelated numeric value
hazma/relic_density/smdof.dat — KEPT: historical count or unrelated numeric value
hazma/vector_mediator/_gev/spectra.py — KEPT: historical count or unrelated numeric value
projects/parity-pinned-defect-repair/task-notes/README.md — KEPT: historical count or unrelated numeric value
projects/parity-pinned-defect-repair/task-notes/task-3-closed-form-deltas.md — KEPT: historical count or unrelated numeric value
projects/parity-pinned-defect-repair/task-notes/task-4-boost-window.md — KEPT: historical count or unrelated numeric value
projects/parity-pinned-defect-repair/task-notes/task-5-eta-prime-line.md — KEPT: historical count or unrelated numeric value
projects/parity-pinned-defect-repair/task-notes/task-6-phi-lines.md — KEPT: historical count or unrelated numeric value
projects/parity-pinned-defect-repair/task-notes/task-7-photon-muon-endpoint.md — KEPT: historical count or unrelated numeric value
test/parity/deltas.py — KEPT: historical count or unrelated numeric value
test/parity/tolerances.py — KEPT: historical count or unrelated numeric value
test/test_core_mediator_decay_photon.py — KEPT: historical count or unrelated numeric value
test/test_core_photon_pion.py — KEPT: historical count or unrelated numeric value
test/test_core_quad.py — KEPT: historical count or unrelated numeric value
test/test_core_special.py — KEPT: historical count or unrelated numeric value
test/vector_mediator/herwig4dm/4pi/run.charged.0.98.dat — KEPT: historical count or unrelated numeric value
test/vector_mediator/herwig4dm/4pi/run.neutral.0.98.dat — KEPT: historical count or unrelated numeric value
```

[Task 12 of `parity-pinned-defect-repair` (2026-09-24) repointed the
`docs/followups/todo/` paths in the block above to `done/`, where the
follow-ups moved at that project's close; when the command ran they
were under `todo/`.]

### Line citations, counts, and exit mapping

```sh
rg -n 'photon_muon\.rs:[0-9]+|test_core_photon_muon\.py:[0-9]+|deltas\.py:[0-9]+' projects/parity-pinned-defect-repair/ docs/
```

returned two matches, both in the dated
`task-notes/task-3-closed-form-deltas.md` sweep table. KEPT as history;
no live affected numeric line citation needs repair.
The citation checker was run over the seven changed Markdown documents
listed in the preflight command, with explicit paths:

```text
docs scanned: 7
in-repo citations checked: 0
external citations skipped: 0
out-of-range or ambiguous: NONE
```

| Claim | Re-derivation | Result |
| --- | --- | --- |
| Declared arrays | `Counter(d.repair for d in deltas.DECLARED_DELTAS.values())` | 99: A1 44, B4 30, A1+B1 6, A1+B2 6, B5 6, B6 6, A2 1 |
| A2 positions | Compare captured before/after muon rest arrays | 4: 161, 162, 163, 164 |
| Candidate corpus radius | Before/after capture above | 7 cases, 72,940 values; 6 cases unchanged |
| Independent reference | 60-digit J+/J- command above | 2 positive and 2 negative reference points |
| Corpus untouched | `git diff --stat -- test/parity/data test/parity/oracles/data test/parity/tolerances.py` | Empty |
| Public impact | Before/after capture and integral commands above | Net +5.44538375e-8 photons/decay at rest; boosted probes unchanged |

| Exit criterion | Test or artifact |
| --- | --- |
| Correct kinematic guard | `test_a_muon_at_rest_stops_at_the_kinematic_endpoint` and Rust edge test |
| Resolve signed tail | ADR-0002, public docstring, published-reference test |
| Four independent oracle positions | A2 corpus rest node; whole rest block equals capture |
| Rust composition reach | Caller argument derivation and seven-case before/after capture |
| Boost identity and restored support | Production Rust boost-integral invariant; Python restored-interval and scalar/array tests |
| Revert and widening rejected | Two measured failing mutations in Verification |
| Preserved data and complete gates | Integrity checks, full preflight, citation checker |

Task-note self-check: Complete agrees with the working-memory Task 7 row;
all exit criteria map above. Every changed file is in the diff or is one
of the two new Markdown artifacts. The project remains In Progress;
Task 8 is next. The note's measurements and prior-task history are not
instructions to preserve the old endpoint.

## Handoff to Next Task

Read `../PLAN.md` Task 8, the A3 oracle, and the Task 2 A3 measurement.
A2 is confined to the muon rest block and does not overlap A3. Task 8
still must prove disjointness or compose with B4 on the scalar decay
case; Task 9 owns B3 on the rho rest blocks. The signed muon approximation
is intentional under ADR-0002; do not silently clamp it in a consumer.
