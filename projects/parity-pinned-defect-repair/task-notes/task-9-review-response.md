# Task 9 review response — PR #98

**Date:** 2026-09-20
**Status:** Complete — review fixes verified
**Reviewed head:** `2bb8f971737ba4f650a4da1dcdbf664b99481283`
**Merge base:** `42561fe7728e0735b0670ba15c93421189edffe3`

## Assessment

The supplied generalist review approves the implementation and raises
three non-blocking comments. Each is valid and in scope.

| # | Category | Resolution |
| --- | --- | --- |
| 1 | fix | Label the original Files Changed and Numerical impact sections with PR #98; replace the obsolete uncommitted description with its commit. |
| 2 | fix | Require each composition step's output suffixes to be present in the base prediction. Cover additive and exact steps with regression tests. |
| 3 | fix | Identify 0.98823 as the captured PDG branching fraction for pi0 to two photons. |

No comments are deferred or rejected. No new physics, public interface,
corpus data, tolerance, or Rust change is part of this response.

## Files Changed — PR #98 review response

- `test/parity/deltas.py`: restore rejection of missing base outputs.
- `test/parity/test_parity.py`: additive and exact malformed-composition
  cases; both fail before the guard and pass after it.
- `test/test_core_photon_rho.py`: explain the branching-fraction literal.
- `projects/parity-pinned-defect-repair/task-notes/task-9-rho-rest-frame.md`:
  historical section labels, implementation commit and response pointer.
- This note and the working-memory README: review disposition and status.
- `docs/agents/lessons.md`, `docs/agents/lessons-examples.md`: add PR #98
  to the historical-labeling lesson and record the mapping-fallback class.

## Verification

The new test was run before the implementation change:

```text
.venv/bin/pytest -n 0 test/parity/test_parity.py -k composition_rejects
2 failed, 656 deselected in 0.51s
```

Both failures were the expected missing AssertionError. After the guard:

```text
.venv/bin/pytest -n 0 test/parity/test_parity.py -k 'composition_rejects or RhoRestDeclaration'
7 passed, 651 deselected in 0.71s
```

Full preflight, with `.venv/bin` first on PATH and the changed paths:

```sh
scripts/agents/preflight.sh --paths "test/parity/deltas.py test/parity/test_parity.py test/test_core_photon_rho.py" --md "docs/agents/lessons-examples.md docs/agents/lessons.md projects/parity-pinned-defect-repair/task-notes/README.md projects/parity-pinned-defect-repair/task-notes/task-9-rho-rest-frame.md projects/parity-pinned-defect-repair/task-notes/task-9-review-response.md"
```

```text
PASS pytest  2314 passed, 16 skipped, 1 warning, 37 subtests passed in 26.48s
RESULT: PASS
```

All Rust gates, Python formatting/lint, import smoke, Markdown and
forbidden-token checks passed; the project-close version check was
correctly skipped. Final documentation-only updates were rechecked with
markdownlint and the citation checker (five Markdown files).

## Numerical impact — PR #98 review response

No further public numerical changes. The same environment measured both
sides of this review edit with the existing built extension in this
worktree. The guard is in test infrastructure; no library or build source
changed, so no Rust rebuild was needed.

```text
.venv/bin/python /tmp/hazma-pr98-review-measure.py before
Captured 36 composite declarations and 12 public arrays.
.venv/bin/python /tmp/hazma-pr98-review-measure.py after
All 36 composite declarations retain identical suffixes and values; 12 public arrays (1212 values) are bit-identical.
```

The measurement enumerates DECLARED_DELTAS, selects every Composed
relation and compares all returned suffixes with array equality. The
public grid has 101 logarithmic photon energies from 0.01 to 800 MeV,
for both rho species at parent energies of 1, 1.05 and 2 times rho mass;
it also measures the generic two-rho spectrum at corresponding
center-of-mass energies. The original PR's numerical-impact figures
remain valid. Reproduction script:

```python
import sys, pickle
from pathlib import Path
import numpy as np
sys.path[:0]=[str(Path.cwd()),str(Path.cwd()/'test/parity')]
import test_parity as parity
from hazma import spectra
from hazma.parameters import rho_mass
out={}
loader=parity.stored_arrays.__wrapped__()
for key,delta in parity.deltas.DECLARED_DELTAS.items():
    if not isinstance(delta.relation,parity.deltas.Composed):continue
    name,label,suffix=key
    case=parity.CASES[name]
    block=next(b for b in case.blocks if b.label==label)
    meta=next(b for b in parity.MANIFEST['cases'][name]['blocks'] if b['label']==label)
    arrays=loader(name)
    stored={s:arrays[e['key']] for s,e in meta['arrays'].items()}
    prediction=delta.relation.expected(case.resolve(),block,stored)
    out[key]={s:a.copy() for s,a in prediction.items()}
x=np.geomspace(.01,800.,101)
for kind,fn in [('rho',spectra.dnde_photon_charged_rho),('rho0',spectra.dnde_photon_neutral_rho)]:
    for factor in [1.,1.05,2.]:
        out[kind,factor]={'direct':fn(x,rho_mass*factor),'generic':spectra.dnde_photon(x,2*rho_mass*factor,[kind,kind],include_fsr=False)}
if sys.argv[1]=='before':
    Path('/tmp/hazma-pr98-review-before.pkl').write_bytes(pickle.dumps(out))
    print('Captured',len(out)-6,'composite declarations and 12 public arrays.')
else:
    before=pickle.loads(Path('/tmp/hazma-pr98-review-before.pkl').read_bytes())
    assert before.keys()==out.keys()
    for key,arrays in out.items():
        assert arrays.keys()==before[key].keys()
        for suffix,a in arrays.items():
            np.testing.assert_array_equal(a,before[key][suffix])
    print('All 36 composite declarations retain identical suffixes and values; 12 public arrays (1212 values) are bit-identical.')
```

## Plan impact

None. The explicit guard enforces the existing composition contract.

## Stale-state sweep

Commands run from the task worktree after edits; rows below are folded
and annotated, rather than represented as unmodified command output.
The full pre/post captures are in `/tmp/hazma-pr98-review-before-sweep.txt`
and `/tmp/hazma-pr98-review-after-sweep.txt`.

```sh
rg -n --hidden 'uncommitted diff|No version bump, commit|^## Files Changed$|^## Numerical impact$|0\.98823|predicted\[suffix\]|\{\*\*stored, \*\*predicted\}' projects/ docs/ hazma/ test/ .claude/ .codex/ README.md CHANGELOG.md
```

### Pre-fix occurrences

| Site (at reviewed head) | Observed claim | Disposition |
| --- | --- | --- |
| Task 9 note, line 54 | No commit or push belongs to implementation | EDITED — distinguish the completed shipping step |
| Task 9 note, lines 56, 137 | Unlabeled historical sections | EDITED — both carry PR #98 |
| Task 9 note, line 58 | Implementation described as uncommitted | EDITED — identify reviewed commit |
| Rho Python test, line 522 | Unexplained branching fraction | EDITED — add provenance comment |
| Delta composition, line 319 | Merged stored/predicted fallback | EDITED — validate output suffixes before updating |
| Other task notes, templates and lessons | Section headings, historical examples | KEPT — unrelated records and templates |
| Other tests, data and references | Same numeric constant or dictionary indexing | KEPT — unrelated values/uses |

### Post-fix occurrences

| Site | Observation / disposition |
| --- | --- |
| Task 9 note | No longer matches any obsolete heading or uncommitted claim |
| Rho Python test | Literal retained with branching-fraction comment — KEPT |
| Delta composition | Merge retained to supply grids; outputs checked against base keys — KEPT |
| Lessons examples | Describes the former wording historically — KEPT |
| This response | Includes the sweep expression and its dispositions — KEPT |
| All other matches | Unrelated historical headings, literals or dictionary uses — KEPT |

```sh
rg -n --hidden 'mapping-fallback-hides-missing-output|missing from the base prediction|test_a_composition_rejects_a_suffix_missing_from_its_base' projects/ docs/ hazma/ test/ .claude/ .codex/ README.md CHANGELOG.md
```

This identifies the new ledger/example pair, implementation assertion,
regression test and this response's command. All are KEPT and consistent.
The initial identifier output, before this note included the command:

```text
docs/agents/lessons.md:313: mapping-fallback-hides-missing-output
docs/agents/lessons-examples.md:1367: mapping-fallback-hides-missing-output
test/parity/deltas.py:324: missing from the base prediction
test/parity/test_parity.py:811: test_a_composition_rejects_a_suffix_missing_from_its_base
test/parity/test_parity.py:837: missing from the base prediction
```

The output above is folded to the relevant symbol text. The line numbers
are a record of this scan, not instructions to locate future code.

```sh
rg -n '(Task [0-9]+ will|will be added|still pending|today: ?stub|currently|In Progress)' projects/parity-pinned-defect-repair/ hazma/
```

KEPT: project status, unrelated unimplemented theory methods, templates,
and dated prior-task evidence. The response status records the passing
gate; its own command is a self-match.

Citation checks use explicit changed Markdown paths, including this new
note. The only skipped path in the lessons examples is the deleted boost
source cited in an explicitly historical example; no live source claim
is inferred from it. Post-edit sweeps are repeated and sorted to verify
that pasting the evidence has reached a fixed point.

| Count / contract | Command or evidence | Result |
| --- | --- | --- |
| Review dispositions | Assessment table | 3 fix, 0 deferred, 0 rejected |
| Missing-output regression | Targeted pytest command above | 2 red before, 2 green after; 7 targeted tests green |
| Existing composite declarations | Measurement script enumerates the live table | 36 unchanged |
| Public spectrum samples | Measurement script | 12 arrays, 1212 unchanged values |
| No production or pinned-data edits | `git diff --name-only HEAD -- hazma rust pyproject.toml test/parity/data test/parity/oracles/data test/parity/tolerances.py` | No output |
| Historical labels | Task note section headers | Files Changed and Numerical impact both carry PR #98 |

Each accepted comment maps to the artifact in Assessment. The existing
Task 9 Complete row remains valid; these review edits do not expand the
physics task. Outstanding questions: none.

## Handoff

All three review comments are addressed. Local verification is recorded
above; published verification is tracked by PR #98's checks for the
commit carrying this response. The original reviewed commit's CI is
historical evidence, separate from verification of these fixes.
