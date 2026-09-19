# Task 9: Repair B3 — rho rest-frame branch

**Date:** 2026-09-18
**Project:** parity-pinned-defect-repair
**Status:** Complete
**Plan References:** `../PLAN.md`, Task 9; `../rules.md`
**Related ADRs:** ADR-0001 (project-scoped)
**Depends On:** Tasks 3 and 8

## Objective

Return the rho rest-frame spectrum in MeV⁻¹ and declare the correction
against the A3-corrected capture without changing the historical corpus.

## Exit Criteria

- [x] Correct both rho rest branches and rename defect-pinning tests.
- [x] Compose A3 with B3 on only the rest blocks of charged and neutral
  rho; verify the energy factor at every nonzero position.
- [x] Verify a physics invariant and scalar/array and threshold behavior.
- [x] Measure before/after across the corpus; preserve arrays and budgets.
- [x] Revert and widened-declaration mutations fail; aggregation and full
  preflight pass; documentation and stale-state checks complete.

## Inputs Reviewed

- Project plan, working-memory README, rules, Task 8 handoff, ADR-0001,
  corpus-repinning and defect-blast-radius references.
- Repository lessons, environment, preflight and doc-consistency guidance.
- Rho follow-up, kernel, Python physics tests and delta models.

## Findings

- Task 8 already declares all rho rest arrays through A3. The B3 factor
  must multiply that captured prediction, not the original corpus.
- The original task prose counted the Rust test's species assertions as
  separate tests. One Rust test was renamed; a continuity test was added.
- The ordinary rest-frame photon spectrum has units MeV^-1. The former
  branch returned its boost integrand, with an erroneous extra 1/E.
- The outer rho support defect remains separately tracked and unchanged.

## Decisions and Implementation Notes

- Work isolated from fresh trunk `42561fe7` on
  `codex/parity-pinned-defect-repair/task-9-rho-rest-frame`.
- Extend Composed to apply an Exact transform after its base prediction.
  Passing the original grids alongside predicted values preserves A3 and
  avoids cancellation from writing the factor as an additive difference.
  No new relation protocol or relaxed budget is needed; ADR-0001 and the
  relation guidance now explicitly describe transforms in compositions.
- Keep the rho relation at its existing 1e-9 portability budget, with
  no absolute floor, despite the much smaller local residual.
- Follow-up administrative closure stays with Task 12, as planned.
  No version bump, commit, or push belongs to this implementation task.

## Files Changed

This task's uncommitted diff:

- `rust/src/kernels/photon_rho.rs`: restore the energy factor, correct
  unit documentation, and assert daughter sums and rest-limit continuity.
- `test/test_core_photon_rho.py`: corrected reference, public scalar and
  array checks, continuity, and neutral-pion photon-yield invariant.
- `test/parity/deltas.py`: sequential Exact composition and A3+B3 entries.
- `test/parity/test_parity.py`: exact B3 roster, missing-component
  mutations, immutable inputs and widened-position rejection.
- `test/parity/test_delta_models.py`, `test/parity/README.md`: current
  model status; historical corpus model checks remain intact.
- `CHANGELOG.md`, rho follow-up, project plan, ADR-0001, corpus-repinning
  and defect-blast-radius references, working-memory README, this note:
  record the measured repair and reconcile live instructions.

## Verification

Preflight was run with the isolated `.venv/bin` first on PATH and these
explicit scopes (no `--tests`, so pytest covers the full suite):

```sh
scripts/agents/preflight.sh --paths "test/parity/deltas.py test/parity/test_delta_models.py test/parity/test_parity.py test/test_core_photon_rho.py" --md "CHANGELOG.md docs/followups/todo/rho-rest-frame-branch-returns-the-integrand.md projects/parity-pinned-defect-repair/PLAN.md projects/parity-pinned-defect-repair/adrs/ADR-0001-corpus-repairs-are-declared-deltas.md projects/parity-pinned-defect-repair/references/corpus-repinning.md projects/parity-pinned-defect-repair/references/defect-blast-radius.md projects/parity-pinned-defect-repair/task-notes/README.md test/parity/README.md projects/parity-pinned-defect-repair/task-notes/task-9-rho-rest-frame.md"
```

The first run passed every code gate and found three Markdown formatting
issues, subsequently fixed. Its literal pytest summary was:

```text
2312 passed, 16 skipped, 1 warning, 37 subtests passed in 26.39s
```

Final preflight (same command, after the documentation fixes):

```text
PASS black, isort, ruff, cargo fmt, cargo clippy, cargo test
PASS pytest: 2312 passed, 16 skipped, 1 warning, 37 subtests passed in 27.94s
PASS import hazma, markdownlint, forbidden tokens
SKIP version bump: not a closing task
RESULT: PASS
```

The gate rows above are condensed; the pytest summary and final result
are verbatim. No code changed after the passing code gates. The final
note-result update was separately rechecked with markdownlint and the
citation checker. All 92 relative Markdown links in the touched docs
resolve. Corpus and capture integrity checks:

```text
python test/parity/generate.py --check
corpus OK: 41 cases / 1580 arrays match the manifest (generated at 010747c6125d, kernel digest f5e6e269be47)
python test/parity/oracles/capture.py --check
oracles OK: 4 defects / 940 arrays match the manifest (corpus manifest f476fb420caf)
```

The final editable rebuild reproduced every array and exception in all
623 measured blocks. Data and tolerance diffs are empty.

The old extension was built from fresh trunk before source edits. Running
its corpus nodes against the repaired declarations is the revert mutation:

```text
.venv/bin/pytest -n 0 test/parity/test_parity.py -k 'test_entry_point_matches_corpus and rho and rest'
2 failed, 4 passed, 645 deselected in 0.70s
```

Both failures are the rest blocks; nearby boosted blocks pass. An initial
filter selected zero tests; it was discarded and the command above
replaced it. After rebuild, the targeted rho/model/declaration suite:

```text
.venv/bin/pytest test/test_core_photon_rho.py test/parity/test_delta_models.py test/parity/test_parity.py -k 'rho or Rho or delta_model or declared'
98 passed in 3.51s
```

The component and position mutations in TestRhoRestDeclaration exercise
each value array: omitting B3, omitting A3, or adding one unmoved position
must raise AssertionError. The separate roster assertion permits B3 only
on the two species' rest blocks.

## Numerical impact

Captured from builds before and after the one-line kernel repair, using
one isolated Python 3.13 environment. Import paths were verified inside
this worktree. No corpus or oracle array was regenerated.

| Species / array | Changed | Up / down | Max absolute numeric shift | Max relative shift | Max relative oracle residual |
| --- | --- | --- | --- | --- | --- |
| charged / values | 170 | 97 / 73 | 302.181659743 | 350.844804609 | 4.1515e-16 |
| charged / scalar_values | 5 | 2 / 3 | 302.181659743 | 68.783527503 | 3.3251e-16 |
| neutral / values | 170 | 97 / 73 | 604.363204522 | 350.844804609 | 3.8807e-16 |
| neutral / scalar_values | 5 | 2 / 3 | 604.363204522 | 68.783527503 | 1.6631e-16 |

Grid: each rho rest corpus block spans 0.0077526–77526 MeV. The absolute
maximum is at the lowest energy. The corrected values are bit-identical
to the previous build's values multiplied by photon energy in MeV.
The oracle residual compares the repaired kernel to the independent A3
Cython capture multiplied by the same energy, not to the defective corpus.
The 623 blocks of 41 cases produce 181,191 evaluated values including
scalar probes: 350 move in four arrays; 786 other value arrays are
unchanged. This runtime count is distinct from the old stored-corpus
headline. Exceptions, grids, NaN patterns and all other blocks agree.

The generic two-rho API was also captured on 101 log-spaced photon energies
from 0.01 to 800 MeV at 1, 1.05 and 2 times production threshold.
At threshold, 94 points per species move: maximum absolute differences
362.397970755 (charged) and 724.795763660 (neutral) MeV^-1.
Above threshold all sampled values are bit-identical.

### Reproduction

Save the following as `/tmp/hazma-task9-capture.py`. Run from this
worktree with `.venv/bin/python`, first on trunk with its editable build,
then on the repaired tree after rebuilding. The output argument names a
scratch pickle; this never writes into corpus data.

```python
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pickle
import warnings
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / 'test/parity'))
import cases
import generate
CASES = cases.build_cases()
def capture(item):
    name, i = item
    case = CASES[name]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = generate.evaluate_block(case.resolve(), case.blocks[i])
    return (name, case.blocks[i].label), result
if __name__ == '__main__':
    import hazma, hazma._core
    assert Path(hazma.__file__).resolve().is_relative_to(Path.cwd())
    assert Path(hazma._core.__file__).resolve().is_relative_to(Path.cwd())
    jobs = [(n, i) for n, c in CASES.items() for i in range(len(c.blocks))]
    with ProcessPoolExecutor(max_workers=8) as pool:
        data = dict(pool.map(capture, jobs))
    Path(sys.argv[1]).write_bytes(pickle.dumps(data))
    print(f'Captured {len(data)} blocks from {len(CASES)} cases; {hazma._core.__file__}')
```

Compare with the following saved as `/tmp/hazma-task9-measure.py`, after
also producing the generic API captures below:

```python
from pathlib import Path
import sys, pickle
import numpy as np
from collections import Counter
sys.path.insert(0,str(Path.cwd()/'test/parity'))
import deltas, cases, generate
before=pickle.loads(Path('/tmp/hazma-task9-before.pkl').read_bytes())
after=pickle.loads(Path('/tmp/hazma-task9-after.pkl').read_bytes())
cases_=cases.build_cases();manifest=generate.load_manifest()
total=0; moved=0; moved_arrays=0; unchanged_arrays=0
for (name,label),(old,raised) in before.items():
    new,newraised=after[name,label]
    assert raised==newraised
    for suffix,a in old.items():
        if suffix not in ('values','scalar_values'):
            np.testing.assert_array_equal(a,new[suffix])
            continue
        b=new[suffix]; total+=a.size
        equal=(a==b)|(np.isnan(a)&np.isnan(b))
        n=int((~equal).sum());moved+=n
        if n:
            moved_arrays+=1
            assert name in ('spectra.photon.charged_rho','spectra.photon.neutral_rho') and label=='rest'
            grid=old['grid' if suffix=='values' else 'scalar_grid']
            np.testing.assert_array_equal(b, a*grid)
            case=cases_[name];block=next(x for x in case.blocks if x.label==label)
            meta=next(x for x in manifest['cases'][name]['blocks'] if x['label']==label)
            npz=np.load(generate.DATA_DIR/manifest['cases'][name]['file'])
            stored={s:npz[m['key']] for s,m in meta['arrays'].items()}
            pred=deltas.DECLARED_DELTAS[name,label,suffix].relation.expected(case.resolve(),block,stored)[suffix]
            nonzero=b!=0
            print(name,label,suffix,'moved',n,'up',int((b>a).sum()),'down',int((b<a).sum()),'max_abs',np.max(abs(b-a)),'max_rel',np.max(abs((b-a)[a!=0]/a[a!=0])),'oracle_rel',np.max(abs((b-pred)[nonzero]/b[nonzero])),'grid',grid.min(),grid.max())
        else: unchanged_arrays+=1
print('corpus',len(before),'blocks',len(cases_),'cases',total,'values',moved,'moved',moved_arrays,'arrays',unchanged_arrays,'unchanged arrays')
print('declarations',len(deltas.DECLARED_DELTAS),Counter(d.repair for d in deltas.DECLARED_DELTAS.values()))
a=np.load('/tmp/hazma-task9-public-before.npz');b=np.load('/tmp/hazma-task9-public-after.npz')
for k in a.files:
    if k=='grid':continue
    print('public',k,'moved',np.count_nonzero(a[k]!=b[k]),'max_abs',np.max(abs(a[k]-b[k])))
```

Save this as `/tmp/hazma-task9-public.py` and pass
`/tmp/hazma-task9-public-before.npz` or
`/tmp/hazma-task9-public-after.npz` from the corresponding built tree:

```python
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()))
from hazma import spectra
from hazma.parameters import rho_mass
x=np.geomspace(.01,800,101)
a={}
for kind in ['rho','rho0']:
    for factor in [1.,1.05,2.]:
        a[f'{kind}/{factor}']=spectra.dnde_photon(x,2*rho_mass*factor,[kind,kind],include_fsr=False)
np.savez(sys.argv[1],grid=x,**a)
print('6 generic two-rho grids, 101 points each')
```

## Open Questions

None.

## Plan Impact

**Impact Level:** ADR-0001 clarified; plan wording corrected. No new
protocol, scope, sequencing or numerical budget. The plan correction
counts the pre-existing Rust test rather than its assertions.

## Stale-state sweep

The tables below are **hand-folded by file** from sorted command output;
KEPT and EDITED are dispositions, not literal rg output. This note is
included in its own searches. Historical task notes keep their dated
measurements; current instructions and relation guidance were reconciled.

### Pre-fix occurrences

The pre-fix command was:

```sh
rg -n --hidden 'bare_integrand|returns.the.integrand|B3.*(not|undeclared)|\b165\b' projects/ docs/ hazma/ test/ rust/src/kernels/photon_rho.rs .claude/ .codex/ README.md CHANGELOG.md
```

It produced 129 lines in `/tmp/hazma-task9-prefix-sweep.txt`. EDITED:
the live kernel/test names, B3 model status and rho follow-up. KEPT:
165 is still the declaration count; numeric data matches are unrelated;
prior task measurements and the released changelog describe history.

### Post-fix identifier and citation sweep

```sh
rg -n --hidden '_A3_B3|TestRhoRestDeclaration|the_rest_frame_branch_returns_the_bare_integrand|the_rest_frame_branch_returns_the_daughter_spectra|the_rest_frame_spectrum_matches_the_next_parent_energy|Composed|photon_rho\.rs|test_core_photon_rho\.py' projects/parity-pinned-defect-repair/ docs/ README.md hazma/ test/ .claude/ .codex/
```

| Matched file | Disposition |
| --- | --- |
| `docs/followups/todo/rho-photon-outer-boost-misses-support.md` | KEPT — historical evidence or unrelated test/reference |
| `docs/followups/todo/rho-rest-frame-branch-returns-the-integrand.md` | EDITED — current repair or guidance |
| `projects/parity-pinned-defect-repair/PLAN.md` | EDITED — current repair or guidance |
| `projects/parity-pinned-defect-repair/references/corpus-repinning.md` | EDITED — current repair or guidance |
| `projects/parity-pinned-defect-repair/references/defect-blast-radius.md` | EDITED — current repair or guidance |
| `projects/parity-pinned-defect-repair/task-notes/README.md` | EDITED — current repair or guidance |
| `projects/parity-pinned-defect-repair/task-notes/task-3-closed-form-deltas.md` | KEPT — historical evidence or unrelated test/reference |
| `projects/parity-pinned-defect-repair/task-notes/task-5-eta-prime-line.md` | KEPT — historical evidence or unrelated test/reference |
| `projects/parity-pinned-defect-repair/task-notes/task-6-phi-lines.md` | KEPT — historical evidence or unrelated test/reference |
| `projects/parity-pinned-defect-repair/task-notes/task-9-rho-rest-frame.md` | EDITED — current repair or guidance |
| `test/parity/deltas.py` | EDITED — current repair or guidance |
| `test/parity/test_parity.py` | EDITED — current repair or guidance |
| `test/test_core_mediator_positron.py` | KEPT — historical evidence or unrelated test/reference |
| `test/test_core_neutrino.py` | KEPT — historical evidence or unrelated test/reference |
| `test/test_core_photon_rho.py` | EDITED — current repair or guidance |
| `test/test_core_positron_pion.py` | KEPT — historical evidence or unrelated test/reference |
| `test/test_core_quad.py` | KEPT — historical evidence or unrelated test/reference |

A separate repository-wide old-test-name sweep finds the dated
cython-to-rust Task 4.5 note (KEPT), plus this sweep's own command.
No current test or live follow-up retains the old assertion.

```sh
rg -n '(photon_rho\.rs|test_core_photon_rho\.py|deltas\.py|test_parity\.py|test_delta_models\.py):[0-9]+' projects/parity-pinned-defect-repair/ docs/
```

KEPT: the only existing matches are the Task 3 note's historical sweep
rows. No new source line-number citation was added. The historical rho
Cython excerpt was checked with `git show b5f7f90^` and corrected to
neutral lines 43–44 and charged lines 114–115.
`check_doc_citations.py` was passed all nine touched Markdown files
explicitly (the same `--md` list as preflight):

```text
docs scanned: 9
in-repo citations checked: 0
external citations skipped: 0
out-of-range or ambiguous: NONE
```

### Forward-looking phrase sweep

```sh
rg -n '(Task [0-9]+ will|will be added|still pending|today: ?stub|currently|In Progress)' projects/parity-pinned-defect-repair/ hazma/
```

| Matched population | Disposition |
| --- | --- |
| Project plan and working-memory status | KEPT — project remains open |
| Working-memory concurrency finding | KEPT — historical concurrent changes |
| Task-note template | KEPT — placeholder status |
| Task notes 3, 4, 6, 7, 8, 10a, 13 and Task 8 review response | KEPT — dated evidence and pasted sweep commands |
| `hazma/theory/_theory_constrain.py` | KEPT — unrelated unimplemented methods |
| This task note | KEPT — sweep expressions and dispositions |

### Count sweep

| Claim | Canonical command or recipe | Actual / disposition |
| --- | --- | --- |
| Runtime corpus population | Capture recipe above | 623 blocks, 41 cases, 181191 values — verified |
| Changed positions | Measurement recipe above | 350 in four arrays; 170 + 5 per species — verified |
| Unchanged value arrays | Measurement recipe above | 786 — verified |
| Declarations | `Counter(d.repair for d in deltas.DECLARED_DELTAS.values())` | 165: A3 62, A1 44, A3+B4 20, B4 10, A1+B1 6, A1+B2 6, B5 6, B6 6, A3+B3 4, A2 1 — verified |
| Physical repairs represented | Union of `deltas.repair_labels(d.repair)` over declarations | 9 — verified |
| Public generic grid | Public capture recipe above | 101 points, 94 changes at threshold per species, zero above — verified |
| Full-suite result | Final preflight pytest summary | 2312 passed, 16 skipped, 37 subtests — verified |
| Corpus/oracle integrity | `generate.py --check`; oracle `capture.py --check` | 41 cases / 1580 arrays; 4 defects / 940 arrays — verified |

Numerical-impact evidence is the before/after and independent-reference
comparison above. `git diff --name-only -- test/parity/data
 test/parity/oracles/data test/parity/tolerances.py` produces no output.

### Exit Criteria → test mapping

| Exit criterion | Evidence |
| --- | --- |
| Correct branches and assertions | Rust daughter-sum and next-parent tests; Python rest-spectrum tests |
| Exact declared reach and energy factor | TestRhoRestDeclaration; all rho corpus blocks; before/after comparison |
| Independent physics and public shapes | Neutral-pion box yield, daughter sum, scalar/array equality and continuity |
| Measure all effects; preserve arrays/budgets | Full corpus capture, generic API capture, integrity checks and empty data/tolerance diff |
| Mutation, aggregation, preflight, docs | Missing-component and widened-position tests, old-build corpus failure, bare pytest including theory aggregation, preflight and citation checks |

The task status, checked criteria and working-memory row agree. All
changed artifacts listed above appear in `git diff --stat` or the new
note. The sorted sweep captures are repeated after pasting to check a
fixed point; file grouping avoids introducing stale line-number claims.

## Handoff to Next Task

Task 10 is next; read the working-memory README and its plan entry.
