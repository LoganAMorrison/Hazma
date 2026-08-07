# Task 0.5: Execute ADR-0003 (`hazma.gamma_ray` removal)

**Date:** 2026-08-05
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-00-dead-code-purge.md` (Task
0.5); `../../PLAN.md` §Scope, §Anticipated ADRs
**Related ADRs:** ADR-0003 (project-scoped, Accepted 2026-08-04 with an
Addendum 2026-08-04); ADR-0001 (`docs/adrs/`, the replacement's design)
**Depends On:** none — this task precedes Task 0.2's delete

## Objective

Execute the two remaining, non-deletion steps of ADR-0003: confirm and
record the replacement status of the real public API of
`hazma/gamma_ray.py`, and repoint every durable doc that references the
module at `hazma.spectra`, so that when Task 0.2 deletes the file no doc
is left citing a module that no longer exists.

## Exit Criteria

Copied from the phase file's Task 0.5 block (the ADR-sign-off bullet is
already struck through there — closed 2026-08-04):

- Replacement status of the module's **actual public API** confirmed and
  recorded here: `gamma_ray_decay` → `hazma.spectra.dnde_photon`;
  `gamma_ray_fsr` → `hazma.spectra.dnde_photon_fsr`. (`gamma` /
  `gamma_point` are the compiled `_gamma_ray` names the module wraps,
  not its public surface.)
- Docs referencing `hazma.gamma_ray` repointed to `hazma.spectra`.

## Inputs Reviewed

- `../../PLAN.md` (§Scope, §Numerical impact, §Anticipated ADRs);
  `../README.md` (project working memory); `./README.md` (phase working
  memory); `../../phases/phase-00-dead-code-purge.md`;
  `../../rules.md` (rule "Process 1", verify-before-delete).
- `../../adrs/ADR-0003-remove-gamma-ray-module.md` — including its
  **Addendum (2026-08-04)**, which is the load-bearing input here.
- `docs/adrs/ADR-0001-fsr-generator-takes-both-matrix-elements.md`;
  `docs/followups/done/msqrd-driven-fsr-generator.md` (§Resolution).
- Source read to confirm the replacement claims rather than inherit
  them: `hazma/gamma_ray.py`, `hazma/spectra/_fsr.py`,
  `hazma/spectra/_nbody.py`, `hazma/spectra/_photon/__init__.py`,
  `hazma/spectra/__init__.py`.
- `docs/source/` (`index.rst`, `limits.rst`, `models.rst`,
  `spectra.rst`, `gamma_ray.rst`); `docs/PR_GUIDELINES.md`;
  `docs/agents/doc-consistency.md`, `lessons.md`, `environment.md`.

## Findings

- **The replacement status recorded in ADR-0003's *Decision* body is
  stale; its *Addendum* is the live statement.** The body says
  `gamma_ray_fsr` is "removed without a direct replacement". The
  Addendum (2026-08-04) corrects that: the follow-up it named as "the
  only route back" was implemented ad-hoc as
  `hazma.spectra.dnde_photon_fsr` under repo-wide ADR-0001 (PR #41,
  merged 2026-08-05, `git log -1 --format=%ci 629a8ec` →
  `2026-08-05 16:57:37 -0700`). The ADR body is left as written — an ADR
  is a dated record and the Addendum is the documented amendment
  mechanism — but the phase file's Task 0.5 **gate text** repeated the
  superseded wording and was patched (see §Plan Impact).
- **`docs/source/gamma_ray.rst` is an orphan page.** No toctree
  references it: `docs/source/index.rst` lists `installation, usage,
  spectra, phase_space, form_factors, models, limits, utils,
  parameters`; the only nested toctrees are `limits.rst`
  (`gamma_ray_limits`, `cmb`) and `models.rst` (`scalar_mediator`,
  `vector_mediator`). Both functions it documented are already covered
  by the published `docs/source/spectra.rst` — the n-body
  `dnde_photon` and `dnde_photon_fsr` (`autofunction` at
  `docs/source/spectra.rst:613`, prose from line 479).
- **`hazma/rh_neutrino/_rh_neutrino_spectra.py:24` still does
  `from hazma.gamma_ray import gamma_ray_decay`** — the one live
  in-library textual importer of the module. It is not reachable:
  `hazma/rh_neutrino/__init__.py` imports only `._model`, and the block
  that would import `._rh_neutrino_spectra` is commented out
  (`hazma/rh_neutrino/__init__.py:90`). Nothing else in `hazma/`
  imports that module. **Task 0.2 must handle it** — after the delete
  the import dangles. Left alone here: editing or deleting library code
  is Task 0.2's shape, not a docs repoint.
- **`hazma/spectra/_photon/electron` is dead.** Its docstring said it
  exists "so we can use the electron as a final state in
  `hazma.gamma_ray`". It is not re-exported by
  `hazma/spectra/__init__.py` (the `from ._photon import (...)` block at
  line 127 does not list it) and has no caller: the live n-body path
  carries its own zero entry, `"e": _dnde_zero` at
  `hazma/spectra/_nbody.py:58`. The docstring is repointed here so it
  does not cite a removed module; **deleting the function is a Task 0.2
  candidate** (recorded in this phase's README Findings).
- **Not a `hazma.gamma_ray` reference, but adjacent staleness left
  alone:** `docs/source/rambo.rst` is a second orphan page and it
  documents `hazma.rambo.PhaseSpace`, a module that no longer exists
  (the live API is `hazma.phase_space`, documented in the published
  `docs/source/phase_space.rst`). It belongs with Task 0.2, which
  deletes `hazma/deprecated/rambo.py`.

## Decisions and Implementation Notes

- **`docs/source/gamma_ray.rst` is deleted, not rewritten.** "Repointed
  to `hazma.spectra`" is satisfied by removal here: the page is in no
  toctree, its entire content is the usage narrative for the two
  removed functions, and the equivalent narrative for their
  replacements already exists in the published `spectra.rst`. A stub
  saying "see spectra" would be an unreferenced file with no reader —
  Sphinx has no redirect mechanism configured in `docs/source/conf.py`.
  Verified no reference dangles after the delete (§Verification).
- **ADR-0003's body is not edited.** The Addendum already states the
  correction and says "The decision recorded here is unchanged."
  Rewriting a dated Decision section to match later events would
  falsify the record. What *is* patched is the forward-looking gate
  text that repeated it: the phase file's Task 0.5 criterion and
  `PLAN.md`'s §Anticipated ADRs bullet.
- **`docs/PR_GUIDELINES.md`'s `limits` scope row loses `gamma_ray.py`.**
  The row listed `hazma/limits/`, then `gamma_ray.py`, then "gamma-ray
  limit machinery". `hazma/gamma_ray.py` is a spectrum module, not limit
  machinery — the row was already mis-scoped, and Task 0.2 makes the
  path dangle outright. `hazma/gamma_ray_parameters.py`, which *is*
  limit-adjacent, keeps its own listing in the `params` row.
- **Drive-by consistency fix: repo-wide ADR-0001's status flipped
  Proposed → Accepted.** Its status line said acceptance "rides on the
  review of the PR that implements it"; that PR (#41) merged
  2026-08-05, and `docs/followups/README.md:35` already lists the
  follow-up as `done` shipping `dnde_photon_fsr`. This task cites
  ADR-0001 as the design of record for `gamma_ray_fsr`'s replacement,
  so leaving a contradiction one hop from the claim was not an option.
  Index row in `docs/adrs/README.md:49` updated to match.
- **No CHANGELOG entry in this task.** Nothing user-facing moves here:
  the module still exists, no public signature or value changes, and
  the deleted Sphinx page is unlinked. The `Removed` entry naming both
  functions and their replacements belongs to Task 0.2, which performs
  the deletion. The replacement wording that entry must use is fixed by
  the phase file patch made here.

## Replacement status of `hazma.gamma_ray`'s public API

The record ADR-0003 asks this task to produce. Both rows verified
against source in this worktree, not inherited from the ADR.

| Removed | Replacement | Drop-in? | Evidence |
| --- | --- | --- | --- |
| `gamma_ray_decay(particles, cme, photon_energies, mat_elem_sqrd=None, num_ps_pts=1000, num_bins=25, verbose=False)` — `hazma/gamma_ray.py:49` | `hazma.spectra.dnde_photon` (n-body path, `hazma/spectra/_nbody.py`, over `hazma.phase_space`) | No | Both convolve final-state energy distributions with per-particle decay spectra, but `dnde_photon` takes final states by the short keys of `_spectra_dict` (`hazma/spectra/_nbody.py:136`), not the long names `gamma_ray_decay` accepted, and it does not take `num_bins`. |
| `gamma_ray_fsr(photon_energies, cme, isp_masses, fsp_masses, non_rad, msqrd, nevents=1000)` — `hazma/gamma_ray.py:241` | `hazma.spectra.dnde_photon_fsr` (`hazma/spectra/_fsr.py`) | No | Repo-wide ADR-0001: the non-radiative process enters as a **matrix element** (`msqrd_nonrad`), not a rate float, so every initial-state factor cancels in the ratio and `isp_masses` disappears along with the decay/annihilation branch. Returns an `FSRSpectrum` NamedTuple `(dnde, error)` in MeV⁻¹ rather than a bare array. |

Neither removal is replacement-free. `gamma`/`gamma_point`
(`hazma/_gamma_ray/gamma_ray_generator.pyx`) are the compiled kernels
the module wrapped, not its public API, and go with it in Task 0.2.

## Files Changed

- `docs/source/gamma_ray.rst` — **deleted** (orphan page; both
  documented functions removed by ADR-0003, replacements already in
  `docs/source/spectra.rst`).
- `hazma/spectra/_photon/__init__.py` — `electron` docstring repointed
  off `hazma.gamma_ray`; states the live n-body zero entry and that the
  function has no caller. Docstring text only, no code touched.
- `docs/PR_GUIDELINES.md` — `gamma_ray.py` dropped from the `limits`
  scope row.
- `docs/followups/done/msqrd-driven-fsr-generator.md` — §Entry points
  no longer says the cython-to-rust CHANGELOG "declares `gamma_ray_fsr`
  replacement-free"; it names `dnde_photon_fsr`.
- `docs/adrs/ADR-0001-fsr-generator-takes-both-matrix-elements.md` and
  `docs/adrs/README.md:49` — status Proposed → Accepted (PR #41 merged).
- `docs/followups/todo/utils-public-surface-redundant-helpers.md` — its
  §Triggers, §Why, §Entry points, and §Risks all cited
  `docs/source/gamma_ray.rst:85` as `minkowski_dot`'s sole public-docs
  reference. Deleting that page here made the citation dangle and
  half-ripened the follow-up's trigger, so all four were repointed. This
  file arrived on the trunk *during* this task, in PR #46.
- `projects/cython-to-rust/phases/phase-00-dead-code-purge.md` — Task
  0.5 exit criteria: replacement wording corrected per the ADR-0003
  Addendum; the `gamma_ray.rst` deletion recorded as the shape of the
  docs criterion.
- `projects/cython-to-rust/PLAN.md` — §Anticipated ADRs bullet updated
  (Task 0.5 executed; both functions have a named replacement).
- `docs/followups/todo/preflight-isort-ruff-red-on-trunk.md` (new) plus
  its index row in `docs/followups/README.md` — the trunk lint debt that
  keeps `preflight.sh` red for every touched file.
- `projects/cython-to-rust/task-notes/phase-00/task-0.5-gamma-ray-decision.md`
  — this note (new).
- `projects/cython-to-rust/task-notes/phase-00/README.md`,
  `projects/cython-to-rust/task-notes/README.md` — status, findings,
  handoff.

## Verification

Regenerated against this branch, not curated.

- **No test-suite change.** The diff touches one docstring and durable
  docs; no code path, no test, no fixture. Preflight's test gate ran
  the suite unchanged — see the preflight output in §Stale-state sweep.
- **Orphan-page claim** (`docs/source/gamma_ray.rst` in no toctree),
  before deleting:

  ```console
  $ grep -rn "gamma_ray$" docs/source/*.rst
  (no output)
  ```

  and `docs/source/index.rst` lists `installation usage spectra
  phase_space form_factors models limits utils parameters`;
  `limits.rst` nests `gamma_ray_limits, cmb`; `models.rst` nests
  `scalar_mediator, vector_mediator`.

- **No dangling reference after the delete:**

  ```console
  $ grep -rn "gamma_ray$\|:doc:\`gamma_ray\|   gamma_ray\b" docs/source/
  no dangling references
  ```

- **Replacement is documented where readers land:**

  ```console
  $ grep -n "autofunction:: dnde_photon_fsr\|autofunction:: dnde_photon$" docs/source/spectra.rst
  613:.. autofunction:: dnde_photon_fsr
  687:.. autofunction:: dnde_photon
  ```

- **`_photon.electron` has no caller** (justifies the docstring's new
  claim):

  ```console
  $ git grep -n "_photon.electron\|dnde_photon_electron" -- hazma/ test/
  (no output)
  $ grep -n '"e": _dnde_zero' hazma/spectra/_nbody.py
  58:    "e": _dnde_zero,
  ```

- **PR #41 merge date** (backs ADR-0001's new status line):

  ```console
  $ git log -1 --format="%ci %s" 629a8ec
  2026-08-05 16:57:37 -0700 Merge pull request #41 from LoganAMorrison/claude/msqrd-driven-fsr-generator-c0fb9b
  ```

- **Deferred, deliberately:** the `hazma.gamma_ray` importer in
  `hazma/rh_neutrino/_rh_neutrino_spectra.py:24`, the dead
  `_photon.electron` function itself, `docs/source/rambo.rst`, and the
  `test_gamma_ray.py` / `_gamma_ray/` mentions in
  `docs/agents/{environment,preflight,review-lenses}.md` — all still
  factually true until Task 0.2 deletes the module, and all are
  recorded for Task 0.2 in this phase's README.

## Open Questions

- None introduced by this task. ADR-0003's remaining step is the
  deletion itself (Task 0.2); nothing here is awaiting a decision.
- Surfaced, not owned by this task: `preflight.sh` returns `FAIL` for
  any file under `hazma/` because gates 2 (`isort`) and 3 (configured
  `ruff`) are red on the trunk. Filed as
  [`docs/followups/todo/preflight-isort-ruff-red-on-trunk.md`](../../../../docs/followups/todo/preflight-isort-ruff-red-on-trunk.md).
  Checked first that nothing already covered it — the two existing lint
  follow-ups are both `done/` and both about pins, not rule sets.

## Plan Impact

**Impact Level:** Update phase file *and* `PLAN.md` (no new ADR).

The canonical change is the replacement status of `gamma_ray_fsr`. The
phase file's Task 0.5 exit criterion instructed the CHANGELOG to declare
the removal "replacement-free for the general-`msqrd` case", and
`PLAN.md` §Anticipated ADRs repeated "the replacement-free
`gamma_ray_fsr` case". Both were true when written and were superseded
by ADR-0003's own Addendum once `hazma.spectra.dnde_photon_fsr` shipped
(PR #41). Both are patched in this task rather than deferred, because a
gate sentence that is now factually wrong is exactly what the
canonical-contract diff forbids leaving behind. The same stale phrase in
`docs/followups/done/msqrd-driven-fsr-generator.md` §Entry points is
patched with them. No new ADR: ADR-0003 already recorded this amendment
in its Addendum, and no decision is being made here.

The phase file's docs criterion also now records *how* it was satisfied
(page deleted, not rewritten), so Task 0.2's reviewer does not read the
missing page as an omission.

The phase file's frontmatter also moves `status: Not started` →
`In Progress`. Three of its five tasks (0.1, 0.3, 0.5) are Complete, so
"Not started" was wrong before this task and wronger after it; the
template allows exactly `Not started | In Progress | Complete`. The
`Complete` flip still belongs to whoever lands the phase's last task.

## Stale-state sweep

Every command below was run from the worktree root against
`claude/cython-to-rust/task-0.5-gamma-ray-decision` after all prose edits
were frozen. Output is pasted, not summarized; rows folded by hand say so.

### Identifier sweep — `replacement-free`

Pre-fix, against the trunk
(`git grep -n 'replacement-free' origin/master -- projects/ docs/ hazma/
test/ README.md CHANGELOG.md | sort`) — **5 hits**:

```text
origin/master:docs/followups/done/msqrd-driven-fsr-generator.md:103:  `gamma_ray_fsr` replacement-free and should link here.
origin/master:projects/cython-to-rust/PLAN.md:155:  (replacement status recorded, docs repointed); the replacement-free
origin/master:projects/cython-to-rust/phases/phase-00-dead-code-purge.md:148:  is declared as replacement-free for the general-`msqrd` case in the
origin/master:projects/cython-to-rust/task-notes/README.md:249:  unblocked.** The replacement-free `gamma_ray_fsr` case is tracked at
origin/master:projects/cython-to-rust/task-notes/phase-00/README.md:168:  `hazma.gamma_ray`). The replacement-free `gamma_ray_fsr` case now
```

Disposition — **all five EDITED**: the three canonical/durable ones
(follow-up §Entry points, `PLAN.md` §Anticipated ADRs, the phase file's
Task 0.5 criterion) now name `hazma.spectra.dnde_photon_fsr`; the two
working-memory ones were rewritten as part of this task's bookkeeping.

ADR-0003's Decision body is **not** in this sweep — it uses the phrase
"removed without a direct replacement", and it is KEPT by design: a dated
record amended by its own Addendum.

Post-fix (`rg -n 'replacement-free' projects/ docs/ hazma/ test/
README.md CHANGELOG.md | sort`) — every surviving hit either states the
corrected fact or is this note describing the correction:

```text
docs/followups/done/msqrd-driven-fsr-generator.md:105:  the removal replacement-free (ADR-0003 Addendum, 2026-08-04).
projects/cython-to-rust/phases/phase-00-dead-code-purge.md:150:  Neither removal is replacement-free, so the CHANGELOG names a
projects/cython-to-rust/task-notes/README.md:108:- **`gamma_ray_fsr` is no longer replacement-free** (Task 0.5). ADR-0003
projects/cython-to-rust/task-notes/README.md:260:  "replacement-free" wording corrected in the phase file, `PLAN.md`, and
projects/cython-to-rust/task-notes/phase-00/README.md:198:  longer calls the removal replacement-free.
projects/cython-to-rust/task-notes/phase-00/README.md:235:  `gamma_ray_fsr` is **not** replacement-free: it is superseded by
```

(Folded: the four further hits are inside this note's own sweep block and
§Plan Impact, i.e. the citing doc matching its own command.)

### Identifier sweep — `hazma.gamma_ray` / `gamma_ray.rst`

`rg -n 'hazma\.gamma_ray\b|gamma_ray\.rst' docs/ hazma/ test/ projects/
README.md CHANGELOG.md` — 30 pre-fix hits on the trunk, folded to one row
per file with disposition:

| File | Disposition |
| --- | --- |
| `docs/source/gamma_ray.rst` (3 hits) | DELETED — whole file |
| `hazma/spectra/_photon/__init__.py:354` | EDITED — docstring no longer names the module |
| `docs/PR_GUIDELINES.md:42` | EDITED — `gamma_ray.py` dropped from the `limits` scope row (matched on the bare path, not the dotted name) |
| `docs/followups/done/msqrd-driven-fsr-generator.md:5,103` | EDITED at 103; line 5 KEPT (states which ADR removes the function — still true) |
| `hazma/gamma_ray.py`, `hazma/rh_neutrino/_rh_neutrino_spectra.py:24`, `test/test_gamma_ray.py`, `test/conftest.py:10`, `test/test_utils.py:7` | KEPT — Task 0.2 deletes or repoints these; every statement is true today |
| `hazma/spectra/_fsr.py:4` | KEPT — names the removed antecedent in a module-history sentence, which is the point |
| `docs/adrs/ADR-0001-*.md:9` | KEPT — dated context |
| `projects/cython-to-rust/{PLAN.md,phases/,adrs/ADR-0003,references/,task-notes/}` | KEPT except the two lines this task rewrote; the rest are dated records or correct forward statements |
| `CHANGELOG.md:28` | KEPT — already names the removed function as `dnde_photon_fsr`'s antecedent, which is the wording this task settles |

Post-fix, no `docs/source/` reference dangles:

```text
$ grep -rn "gamma_ray$\|:doc:\`gamma_ray\|   gamma_ray\b" docs/source/
no dangling references
```

### Line-number citation sweep

`--changed-vs origin/master` reports "no docs to check" before the commit
exists (this branch's `HEAD` is still `origin/master`), so the explicit
form was run over all nine changed markdown docs:

```text
docs scanned: 12
in-repo citations checked: 26
  resolved by exact: 22
  resolved by suffix: 4
external citations skipped: 0
out-of-range or ambiguous: NONE
```

Pass the file list through `xargs -0`, not a shell variable: an
unsplit `$MD` reaches `markdownlint` as one nonexistent path, which
prints its usage banner and **exits 0** — the exact false green
`docs/agents/preflight.md` warns about for glob arguments.
`preflight.sh --md` checks the paths exist; a bare invocation does not.

### Forward-looking phrase sweep

`rg -n '(Task [0-9]+ will|will be added|still pending|today: ?stub)'
projects/cython-to-rust/ hazma/` — the only two hits are the sweep
commands themselves, quoted in this note and in Task 0.1's:

```text
projects/cython-to-rust/task-notes/phase-00/task-0.1-relocate-constants.md:414
projects/cython-to-rust/task-notes/phase-00/task-0.5-gamma-ray-decision.md:<this block>
```

### Count sweep

| Claim location | Command | Actual | Status |
| --- | --- | --- | --- |
| This note, "orphan page" | `grep -c gamma_ray docs/source/index.rst` | `0` | OK |
| This note, `_nbody.py:58` zero entry | `grep -n '"e": _dnde_zero' hazma/spectra/_nbody.py` | `58` | OK |
| This note, `spectra.rst:613` | `grep -n 'autofunction:: dnde_photon_fsr' docs/source/spectra.rst` | `613` | OK |
| This note, `rh_neutrino/__init__.py:90` | `grep -n '_rh_neutrino_spectra' hazma/rh_neutrino/__init__.py` | `90` | OK |
| Phase README, 25 extensions | `find hazma -name '*.so' \| wc -l` | `25` | OK — unchanged by this task |
| Phase README / project README, test counts | `pytest -q test` | `244 passed, 20 skipped` | **Updated** — the suite grew from Task 0.3's `68 passed, 20 skipped` when PR #41 landed `test/spectra/test_dnde_photon_fsr.py`. Not caused by this task; recorded so the next agent does not read 68 as current |

### Numerical-impact statement

**No public value changes.** The only file under `hazma/` in the diff is
`hazma/spectra/_photon/__init__.py`, and `git diff origin/master --
hazma/` is four docstring lines inside `electron`:

```text
-    The purpose of this function is so we can use the electron as a final
-    state in `hazma.gamma_ray`.
+    This exists so an electron can appear in a final-state list handed to
+    a multi-particle spectrum routine. Nothing in hazma calls it: the
+    live n-body path behind `hazma.spectra.dnde_photon` carries its own
+    ``"e"`` zero entry (``_dnde_zero`` in ``hazma/spectra/_nbody.py``).
```

No constant, expression, signature, or control-flow line moves, so no
grid evaluation applies. `pytest -q test` on the rebuilt worktree
(25 `.so`) returns `244 passed, 20 skipped`.

### Preflight disposition

`scripts/agents/preflight.sh` returns three red rows. Two are trunk
conditions this task neither caused nor is allowed to silently absorb;
one was mine and is fixed:

| Gate | Verdict | Evidence |
| --- | --- | --- |
| `black --check` | PASS | also passes in CI's whole-tree form: `black --check hazma test` → `227 files would be left unchanged` |
| `isort --check-only` | FAIL — **pre-existing** | the complaint is the unsorted `from hazma.spectra._photon import (...)` block at lines 12–21, which this task never touched. CI runs no isort step at all |
| `ruff check` (configured) | FAIL — **pre-existing** | 17 findings, unchanged. CI's actual gate, `ruff check --isolated --select E9,F63,F7,F82 --exclude hazma/experimental --exclude notebooks .`, returns `All checks passed!`. Matches the project README's standing finding that the configured form is red on the trunk (6844 findings) and does not gate CI |
| `pytest` | PASS | `244 passed, 20 skipped, 1 warning in 276.65s` |
| `import hazma` | PASS | `version 2.1.0`, resolved inside this worktree |
| `markdownlint` | FAIL → **fixed** | four `MD014` hits, all mine: this sweep block used `$ cmd` fences with no output. Each now pastes real output |
| forbidden tokens | PASS | none added |

Both were proven to be trunk conditions by re-running them on the
pristine file — `git stash -u`, then the same two commands against
`hazma/spectra/_photon/__init__.py` with this task's edit removed:

```text
### TRUNK STATE (my changes stashed) ###
-- isort --
ERROR: .../hazma/spectra/_photon/__init__.py Imports are incorrectly sorted and/or formatted.
isort_exit=1
-- ruff --
Found 17 errors.
[*] 5 fixable with the `--fix` option (1 hidden fix can be enabled with the `--unsafe-fixes` option).
```

Identical verdicts with and without the change. Neither red gate is a
regression, and neither is in this task's scope to repair: reformatting
an untouched import block, or clearing 17 trunk-wide
ruff findings inside a docs task, would be exactly the scope creep the
skill forbids. Both are the trunk's problem, and both are invisible to
CI. Flagged here rather than absorbed, and filed as
[`docs/followups/todo/preflight-isort-ruff-red-on-trunk.md`](../../../../docs/followups/todo/preflight-isort-ruff-red-on-trunk.md)
so the next agent does not repeat this analysis.

### Exit Criteria → artifact mapping

| Exit criterion | Satisfied by |
| --- | --- |
| Replacement status confirmed and recorded | the §"Replacement status of `hazma.gamma_ray`'s public API" table, each row cited to source read in this worktree |
| Docs referencing `hazma.gamma_ray` repointed to `hazma.spectra` | `docs/source/gamma_ray.rst` deleted; `hazma/spectra/_photon/__init__.py` docstring; `docs/PR_GUIDELINES.md`; `docs/followups/done/msqrd-driven-fsr-generator.md`. Dangling-reference check above |

### Task-note self-consistency

`**Status:** Complete` agrees with the phase README's Task 0.5 row and
with the Exit Criteria mapping (no unmapped bullet). Every file named in
§Files Changed appears in `git diff --stat origin/master`. Run with
this note itself excluded, so the figure does not move every time the
note is edited (`git diff --stat origin/master -- . ':!…/task-0.5-gamma-ray-decision.md'`):

```text
 docs/PR_GUIDELINES.md                              |   2 +-
 ...001-fsr-generator-takes-both-matrix-elements.md |   5 +-
 docs/adrs/README.md                                |   2 +-
 docs/followups/README.md                           |   1 +
 docs/followups/done/msqrd-driven-fsr-generator.md  |   6 +-
 .../todo/preflight-isort-ruff-red-on-trunk.md      |  92 +++++++++++++
 .../todo/utils-public-surface-redundant-helpers.md |  28 ++--
 docs/source/gamma_ray.rst                          | 150 ---------------------
 hazma/spectra/_photon/__init__.py                  |   6 +-
 projects/cython-to-rust/PLAN.md                    |  11 +-
 .../phases/phase-00-dead-code-purge.md             |  25 ++--
 projects/cython-to-rust/task-notes/README.md       |  75 +++++++++--
 .../cython-to-rust/task-notes/phase-00/README.md   | 138 ++++++++++++++++---
 13 files changed, 324 insertions(+), 217 deletions(-)
```

The fourteenth file is this note, added whole.

**Base note:** `origin/master` advanced from `d81c267` to `b04d01c`
(PR #46) partway through this task, so the branch was rebased onto
`b04d01c` before this sweep ran — every command above is against the
current trunk, and `git rev-parse HEAD origin/master` agree. PR #46 is
what added `docs/followups/todo/utils-public-surface-redundant-helpers.md`,
which is why a file this task never planned to touch appears in the
diff.

## Handoff to Next Task

- **Read first:** this note's §Findings, then
  `../../phases/phase-00-dead-code-purge.md` Task 0.2. The next task is
  **Task 0.2** — the deletion ADR-0003 authorizes. Its gate text is
  unchanged by this task; only Task 0.5's wording moved.
- **Now safe to assume:** no durable doc points a reader at
  `hazma.gamma_ray` as a live API, and the replacement wording every
  downstream artifact must use is settled — `gamma_ray_decay` →
  `hazma.spectra.dnde_photon`, `gamma_ray_fsr` →
  `hazma.spectra.dnde_photon_fsr`, neither a drop-in. Task 0.2's
  CHANGELOG `Removed` entry writes exactly that.
- **Still open / carried into Task 0.2** (all listed with evidence in
  §Findings, and mirrored in this phase's README):
  1. `hazma/rh_neutrino/_rh_neutrino_spectra.py:24` imports
     `hazma.gamma_ray` and will dangle after the delete — unreachable
     today, but it must be repointed or deleted in the same PR.
  2. `hazma/spectra/_photon/electron` is exported by nothing and called
     by nothing; its reason to exist dies with the module.
  3. `docs/source/rambo.rst` documents the long-gone `hazma.rambo`;
     it pairs with Task 0.2's `hazma/deprecated/rambo.py` deletion.
  4. `docs/agents/{environment,preflight,review-lenses}.md` and
     `test/conftest.py` describe `test_gamma_ray.py` / `_gamma_ray/`;
     all go stale the moment Task 0.2 lands.
