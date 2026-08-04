# Task 0.1: Relocate legacy constants header

**Date:** 2026-08-04
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-00-dead-code-purge.md` (Task 0.1);
`../../rules.md` (Constants rule 1); `../../PLAN.md` (Numerical impact)
**Related ADRs:** none (ADR-0003 gates Task 0.2/0.5, not this task)
**Depends On:** none

## Objective

Move the legacy Standard-Model constants header out of `hazma/_decay/`
— a directory Task 0.3 deletes wholesale — into `hazma/_utils/`, and
repoint the `.pyx` files that textually `include` it, so the four
mediator spectrum extensions keep building once `_decay/` is gone.
Values move verbatim; no numbers change.

## Exit Criteria

Copied from the phase file's Task 0.1 `**Exit criteria:**` block. The
second bullet is quoted in its **patched** form — the original said
"the four live sites", which this task corrected to five built sites
(see Plan Impact); the criteria below are what was actually worked to.

- [x] `hazma/_decay/parameters.pxd` moved verbatim (byte-identical
      values) to `hazma/_utils/legacy_parameters.pxd`.
- [x] All five `include "../_decay/parameters.pxd"` sites in **built**
      extensions repointed — the four live mediator ones plus
      `_gamma_ray/gamma_ray_generator.pyx` — and the two unbuilt
      `_decay/*.pyx` sites too, so no `.pyx`/`.pxd` carries a dangling
      include; `.pyx.bak` files left alone.
- [x] `pip install -e .` rebuilds; import smoke passes (excluding
      `_gamma_ray.gamma_ray_generator`, never importable — see ADR-0003).
- [x] Do **not** merge values into `_utils/constants.pxd`
      (`rules.md` rule 4 → "Constants" rule 1).

## Inputs Reviewed

- `../../PLAN.md` (Scope, Numerical impact, Phases table).
- `../../phases/phase-00-dead-code-purge.md` (Goal, Task 0.1, phase
  Exit Criteria).
- `../README.md` (project working memory) and `README.md` (phase
  working memory).
- `../../rules.md` — "Constants" rule 1 (bit-parity first, cleanup
  second) and "Process" rule 1 (verify-before-delete).
- `../../references/cython-inventory.md` lines 61–66 (the relocation
  instruction and the "doomed `_gamma_ray/gamma_ray_generator.pyx:24`"
  fifth include site).
- `setup.py`, `pyproject.toml`, `MANIFEST.in` — to confirm nothing in
  the build/packaging config names the header path.

## Findings

- **There are seven `include` sites, not four.** The phase file names
  the four live ones. The inventory flags a fifth
  (`_gamma_ray/gamma_ray_generator.pyx:24`), and `rg` found two more in
  unbuilt `_decay/` sources (`decay_electron.pyx:4`,
  `_decay_muon_bak.pyx:8`, both spelled `include "parameters.pxd"`).
  The fifth site is **load-bearing for the build** — `_gamma_ray` is a
  live `Extension` in `setup.py:49-51`, so leaving it unrepointed
  breaks `pip install -e .` immediately, before Task 0.2 can delete it.
- The working memory's open question — "whether the scalar/vector
  cross-section `.pyx` textually include any `_utils` constants header"
  — is now answered: **they do not.** `_c_scalar_mediator_cross_sections.pyx`
  and `_c_vector_mediator_cross_sections.pyx` contain no `include`
  directive at all (full enumeration in the sweep block below). Only
  the two `*_decay_spectrum.pyx` and two `*_positron_spec.pyx` modules
  in the mediator packages include anything.
- **The tree does not build from a dirty worktree.** The first
  `uv pip install -e .` failed compiling
  `_gamma_ray/gamma_ray_generator.cpp` with `no member named 'subarray'
  in '_PyArray_Descr'`. Root cause was a **stale generated `.cpp`**
  carried into the worktree from another environment (its embedded
  path comments cite `.venv-reformat/.../numpy/__init__.cython-30.pxd`);
  its mtime suppressed regeneration. `find hazma -name '*.c' -o -name
  '*.cpp' -o -name '*.so' | xargs rm -f` before building fixes it, and
  the tree then builds clean on Cython 3.2.9 / NumPy 2.5.1. This is an
  environment trap, not a repo defect — but it will bite every agent in
  this project, so it is promoted to the phase README.
- `hazma._gamma_ray.gamma_ray_generator` **compiles but cannot be
  imported**, on unmodified `master` as well as after this change: it
  does `from hazma import rambo` (`git show
  HEAD:hazma/_gamma_ray/gamma_ray_generator.pyx`, line 11) and
  `hazma/rambo.py` does not exist. This is the pre-existing breakage
  ADR-0003 and Task 0.2 remove; it is *not* a regression from this
  task, and it is why the import smoke set excludes that module.
- No build or packaging config names the header path. `MANIFEST.in`
  ships `.pxd` via a `global-include *.pxd` (line 1), which covers the
  new location with no edit; `pyproject.toml` package-data lists no
  `.pxd` entries; `setup.py` declares only `.pyx` sources.

## Decisions and Implementation Notes

- **Repointed seven sites, not four.** The four live ones are the
  task's stated scope; the `_gamma_ray` one is forced (the build breaks
  otherwise); the two unbuilt `_decay/` ones cost one line each and
  leave the tree with no dangling relative `include` in any file Cython
  can be pointed at. All three extras are deleted in Tasks 0.2/0.3
  regardless.
- **Left `hazma/_decay/_decay_charged_pion.pyx.bak:8` unrepointed.**
  The boundary drawn is *repoint every `.pyx`/`.pxd`; leave `.bak`*. A
  `.bak` file is a frozen artifact, not a Cython source — `cythonize()`
  in `setup.py` is called only on explicit `Extension` objects, never a
  glob, so it can never be compiled. Task 0.3 deletes the directory.
- **Added a 6-line PROVENANCE note to the relocated header's
  docstring**; the constant definitions themselves are byte-identical
  (proven in Verification). Without it, a reader finding
  `legacy_parameters.pxd` beside `constants.pxd` has no in-file signal
  that the divergent values are deliberate, and the obvious "cleanup"
  is exactly the silent numerical change `rules.md` rule 4 forbids. The
  note is inside the existing module docstring, which Cython pastes in
  as an inert string expression.
- **Did not fix the two known value bugs** in the relocated file —
  `WIDTH_K = 3.3406**-13.` and `WIDTH_PI = 2.528511206475808**-14.`
  (lines 56-57 of the original) are `**` exponentiation where `e-`
  notation was meant. Already recorded in
  `../../references/cython-inventory.md` "Bugs" §3 ("~10⁶ off; both
  currently unused"); `rg` confirms no consumer today — only the six
  definition lines across the three legacy tables. Bit-parity comes
  first (`rules.md` "Constants" rule 1): this must be fixed as a
  *declared* numerical change, not smuggled into a file move. Given a
  durable home in `docs/followups/` because the inventory reference
  retires with this project while `legacy_parameters.pxd` outlives it
  (see Open Questions).

## Files Changed

- `hazma/_decay/parameters.pxd` → `hazma/_utils/legacy_parameters.pxd`
  — relocated (git rename); PROVENANCE paragraph added to the module
  docstring, constant definitions untouched.
- `hazma/scalar_mediator/scalar_mediator_decay_spectrum.pyx:14` —
  include repointed.
- `hazma/scalar_mediator/scalar_mediator_positron_spec.pyx:11` —
  include repointed.
- `hazma/vector_mediator/vector_mediator_decay_spectrum.pyx:14` —
  include repointed.
- `hazma/vector_mediator/vector_mediator_positron_spec.pyx:12` —
  include repointed.
- `hazma/_gamma_ray/gamma_ray_generator.pyx:24` — include repointed
  (keeps the build green until Task 0.2 deletes the module).
- `hazma/_decay/decay_electron.pyx:4` — include repointed (unbuilt;
  deleted in Task 0.3).
- `hazma/_decay/_decay_muon_bak.pyx:8` — include repointed (unbuilt;
  deleted in Task 0.3).

## Verification

Environment: `uv venv --python 3.12` + `uv pip install -e .` in the
worktree; Cython 3.2.9, NumPy 2.5.1, SciPy 1.18.0. Import path
confirmed to be the worktree, not an installed copy:

```text
$ .venv/bin/python -c "import hazma; print(hazma.__file__)"
/Users/logan.morrison/dev/Hazma/.claude/worktrees/trusting-kirch-7e1b7d/hazma/__init__.py
```

**Byte-identity of the moved values.** Everything after the module
docstring compared against the pre-move file at `HEAD`:

```text
$ diff <(git show HEAD:hazma/_decay/parameters.pxd | sed -n '6,$p') \
       <(sed -n '13,$p' hazma/_utils/legacy_parameters.pxd)
IDENTICAL (diff exit 0)
```

**Build.** Clean rebuild after removing stale artifacts; 32 extensions
built, the same count as before the change (Phase 00 reduces this to 20
in Tasks 0.2–0.4):

```text
$ find hazma -name '*.c' -o -name '*.cpp' -o -name '*.so' | xargs rm -f
$ uv pip install -e .
      Built hazma @ file:///.../trusting-kirch-7e1b7d
$ find hazma -name '*.so' | wc -l
      32
```

**Import smoke** — `hazma.theory`, `hazma.limits`, `hazma.cmb`,
`hazma.pbh`, `ScalarMediator`, `VectorMediator`,
`hazma.spectra._photon._muon`, and all four repointed mediator
extension modules: `import smoke OK`. (`_gamma_ray.gamma_ray_generator`
is excluded — pre-existing `from hazma import rambo` failure, see
Findings.)

**Tests.** Both suites, which are disjoint until Task 1.3 merges them:

```text
$ .venv/bin/python -m pytest -q test
52 passed, 20 skipped in 251.55s (0:04:11)

$ .venv/bin/python -m pytest -q          # setup.cfg testpaths → hazma/**
57 passed, 10 skipped in 0.37s
```

Coverage categories in `test/`: mediator model spectra and positron
spectra (the code paths this change touches), gamma-ray limits, CMB
constraints, relic density, phase space, and the `hazma.spectra`
photon/positron/neutrino surfaces. No test was added: this task changes
no behavior and adds no function, so there is nothing a new unit test
could pin that the numerical-impact comparison below does not pin more
strongly (bit-for-bit over 64 arrays vs. a tolerance assertion). The
permanent regression gate for these kernels is Phase 01's parity corpus.

**Preflight.** Run as

```sh
scripts/agents/preflight.sh --tests test --md "<the six changed docs>"
```

with `.venv/bin` on `PATH`. Result:

```text
FAIL   black --check           run `black hazma test` and re-check
FAIL   isort --check-only      run `isort hazma test` and re-check
FAIL   ruff check              Found 6844 errors.
PASS   pytest                  52 passed, 20 skipped in 251.94s (0:04:11)
PASS   import hazma            version 2.1.0
PASS   markdownlint            <the six changed docs>
SKIP   version bump            not a closing PR (pass --closing)
PASS   forbidden tokens        none added
-------------------------------------------------------------------
RESULT: FAIL — blocked commit. Fix the red gates and re-run.
```

**The gate exits non-zero, and that is a standing property of the tree,
not of this change** — see the baseline table below. Flagged explicitly
rather than quietly reported as green: whoever commits this must make
that call knowingly. Fixing it is out of scope here (it would mean
reformatting 34 files and resolving 6844 ruff findings inside a
file-move PR).

The three Python gates are **pre-existing red on `origin/master`** and
are not touched by this task, which changes **zero `.py` files**
(`git diff origin/master --name-only -- '*.py'` → 0). Measured against a
detached worktree at `origin/master` with the same tool versions:

| Gate | At `origin/master` | On this branch |
| --- | --- | --- |
| `black --check hazma test` | 34 files reformat | 34 files reformat |
| `ruff check hazma test` | `Found 6844 errors.` | `Found 6844 errors.` |
| `isort --check-only hazma test` | ERROR (several) | ERROR (same) |

Identical on both sides, as it must be when no `.py` file changes.
Every gate this task can affect — `pytest`, `import hazma`,
`markdownlint`, `forbidden tokens` — is green.

Two invocation notes for whoever runs this next:

- Passing `--paths` a `.pxd` makes black and ruff try to parse Cython as
  Python and fail (`cdef double DECAY_CONST_K = ...`). This task's diff
  has no Python in it, so `--paths` is correctly omitted.
- Passing `--paths` a *directory* (e.g. `hazma/vector_mediator`) drags
  in that directory's pre-existing unformatted `.py` and reports it as
  your failure. Scope `--paths` to files you actually changed.

**Deferred:** the repo-wide black/isort/ruff debt. Out of this task's
touched area (no `.py` in the diff) and pre-existing; not filed as a
new follow-up because it is a standing condition of the tree rather
than something this task surfaced.

## Open Questions

- The `WIDTH_K` / `WIDTH_PI` `**`-vs-`e` exponent bug in the relocated
  header (see Decisions) is left as-is for bit-parity. Follow-up filed:
  [`docs/followups/todo/legacy-parameters-width-exponent-bug.md`](../../../../docs/followups/todo/legacy-parameters-width-exponent-bug.md).

## Plan Impact

**Impact Level:** Update phase file.

The phase file's Task 0.1 exit criterion says "the four live `include`
sites"; the build actually requires a fifth
(`_gamma_ray/gamma_ray_generator.pyx:24`, a live `Extension` in
`setup.py`). That is a factually wrong gate sentence, so it is patched
in this task rather than deferred: the criterion now names five
build-relevant sites and records the two unbuilt `_decay/` extras. No
change to task ordering, scope, or any interface — `PLAN.md` needs no
edit (it holds only the phase table, and this does not change the
phase's deliverable).

<!-- markdownlint-disable MD013 -- pasted command output and evidence tables; wrapping them would falsify the record -->

## Stale-state sweep

Run against branch `claude/cython-to-rust/task-0.1-relocate-constants`
with the working tree staged. Two blocks are **folded** to per-file
counts (`rg -c`) and labelled as such — the unfolded `rg -n` output
includes this note quoting itself, which is noise, and `rg`'s
multi-directory walk is order-nondeterministic. Everything else is
pasted verbatim from the command shown.

**Identifier sweep — `legacy_parameters` (folded to `rg -c`).** This
note is excluded from its own sweeps: it quotes every match, so
including it makes the count a moving target that can never reach a
fixed point (observed — the self-count went 6 → 10 → 12 as this block
was written). Its own hits are therefore recorded here rather than in
the output: 12 for `legacy_parameters`, 6 for the forward-looking
pattern.

```text
$ rg -c --glob='!task-0.1-relocate-constants.md' 'legacy_parameters' \
    projects/ docs/ README.md hazma/ test/ | sort
docs/followups/todo/legacy-parameters-width-exponent-bug.md:3
hazma/_decay/_decay_muon_bak.pyx:1
hazma/_decay/decay_electron.pyx:1
hazma/_gamma_ray/gamma_ray_generator.pyx:1
hazma/scalar_mediator/scalar_mediator_decay_spectrum.pyx:1
hazma/scalar_mediator/scalar_mediator_positron_spec.pyx:1
hazma/vector_mediator/vector_mediator_decay_spectrum.pyx:1
hazma/vector_mediator/vector_mediator_positron_spec.pyx:1
projects/cython-to-rust/phases/phase-00-dead-code-purge.md:1
projects/cython-to-rust/phases/phase-03-numerics-foundation.md:1
projects/cython-to-rust/phases/phase-06-mediator-spectra.md:1
projects/cython-to-rust/references/cython-inventory.md:1
projects/cython-to-rust/rules.md:1
projects/cython-to-rust/task-notes/README.md:2
projects/cython-to-rust/task-notes/phase-00/README.md:1
```

Disposition, one row per file:

| File | Disposition |
| --- | --- |
| the 7 `hazma/**` `.pyx` | EDITED — the repointed includes |
| `docs/followups/todo/legacy-parameters-width-exponent-bug.md` | CREATED |
| `phases/phase-00-dead-code-purge.md` | EDITED — Task 0.1 criterion (see Plan Impact) |
| `phases/phase-03-numerics-foundation.md:36`, `phases/phase-06-mediator-spectra.md:82` | KEPT — both already name `_utils/legacy_parameters.pxd`, i.e. were written against the post-relocation path and are now correct rather than aspirational |
| `references/cython-inventory.md:64` | KEPT — snapshot of the audit; its "repoint the four includes" wording is superseded by the phase file, which is canonical |
| `rules.md:33` | KEPT — "keeps its values verbatim" still true |
| `task-notes/README.md`, `task-notes/phase-00/README.md`, this note | EDITED/CREATED — the bookkeeping |

`README.md` (repo root) and `test/` produced no hits.

**Removed-path sweep** — nothing may still point at the old location:

```text
$ rg -n '_decay/parameters\.pxd|include "parameters\.pxd"' hazma/ test/ docs/ | sort
hazma/_decay/_decay_charged_pion.pyx.bak:8:include "parameters.pxd"
hazma/_utils/legacy_parameters.pxd:6:PROVENANCE: relocated verbatim from ``hazma/_decay/parameters.pxd`` so the
```

Disposition: the `.pyx.bak` hit is KEPT by the documented `.bak`
boundary (see Decisions) — Cython can never compile it, and Task 0.3
deletes the directory. The PROVENANCE line is a deliberate historical
reference, not a live path.

**Include-site enumeration** — the evidence behind the "cross-section
`.pyx` include nothing" finding; every `include` in the package:

```text
$ rg -c '^\s*include\s' --glob '*.pyx' --glob '*.pxd' hazma/ | sort
hazma/_decay/_decay_muon_bak.pyx:1
hazma/_decay/decay_charged_kaon.pyx:1
hazma/_decay/decay_charged_pion.pyx:1
hazma/_decay/decay_electron.pyx:1
hazma/_decay/decay_long_kaon.pyx:1
hazma/_decay/decay_muon.pyx:1
hazma/_decay/decay_neutral_pion.pyx:1
hazma/_decay/decay_rho.pyx:2
hazma/_decay/decay_short_kaon.pyx:1
hazma/_gamma_ray/gamma_ray_generator.pyx:1
hazma/_neutrino/charged_pion.pyx:1
hazma/_neutrino/muon.pyx:1
hazma/_positron/positron_charged_pion.pyx:1
hazma/_positron/positron_decay.pyx:1
hazma/_positron/positron_muon.pyx:1
hazma/scalar_mediator/scalar_mediator_decay_spectrum.pyx:1
hazma/scalar_mediator/scalar_mediator_positron_spec.pyx:1
hazma/spectra/_neutrino/_muon.pyx:1
hazma/spectra/_neutrino/_pion.pyx:1
hazma/spectra/_photon/_eta.pyx:1
hazma/spectra/_photon/_eta_prime.pyx:1
hazma/spectra/_photon/_kaon.pyx:1
hazma/spectra/_photon/_muon.pyx:1
hazma/spectra/_photon/_omega.pyx:1
hazma/spectra/_photon/_phi.pyx:1
hazma/spectra/_photon/_pion.pyx:1
hazma/spectra/_photon/_rho.pyx:1
hazma/spectra/_positron/_kaon.pyx:1
hazma/spectra/_positron/_muon.pyx:1
hazma/spectra/_positron/_pion.pyx:1
hazma/vector_mediator/vector_mediator_decay_spectrum.pyx:1
hazma/vector_mediator/vector_mediator_positron_spec.pyx:1
```

Neither `_c_scalar_mediator_cross_sections.pyx` nor
`_c_vector_mediator_cross_sections.pyx` appears — they carry no
`include` directive, which closes the project-level open question.

**Line-number citation sweep.** `--changed-vs origin/master` reports
"no docs to check" pre-commit (it diffs commits, and this branch has
none yet), so the six touched/created docs are passed explicitly:

```text
$ ./scripts/agents/check_doc_citations.py \
    projects/cython-to-rust/task-notes/phase-00/task-0.1-relocate-constants.md \
    projects/cython-to-rust/task-notes/phase-00/README.md \
    projects/cython-to-rust/task-notes/README.md \
    projects/cython-to-rust/phases/phase-00-dead-code-purge.md \
    docs/followups/todo/legacy-parameters-width-exponent-bug.md \
    docs/followups/README.md
docs scanned: 6
in-repo citations checked: 63
  resolved by exact: 54
  resolved by suffix: 9
external citations skipped: 0
out-of-range or ambiguous: NONE
```

**Forward-looking phrase sweep (folded to `rg -c`):**

```text
$ rg -c --glob='!task-0.1-relocate-constants.md' \
    '(Task [0-9]+ will|will be added|still pending|today: ?stub|currently|In Progress)' \
    projects/cython-to-rust/ hazma/ | sort
hazma/theory/_theory_constrain.py:2
projects/cython-to-rust/PLAN.md:1
projects/cython-to-rust/phases/_template.md:1
projects/cython-to-rust/phases/phase-01-parity-corpus.md:1
projects/cython-to-rust/references/cython-inventory.md:1
projects/cython-to-rust/task-notes/README.md:2
projects/cython-to-rust/task-notes/_template.md:1
projects/cython-to-rust/task-notes/phase-00/README.md:1
```

All KEPT and all true as written: two `NotImplementedError("currently
does not work")` raises in library code (untouched, pre-existing);
`PLAN.md` / `task-notes/**` status lines (the project *is* in progress,
and the phase-00 row now reads "In Progress (0.1 done)");
`phase-01`'s "the regression harness the repo currently lacks" (still
lacked); `cython-inventory.md:208`'s "both currently unused" — re-verified this
task:

```text
$ rg -c 'WIDTH_K|WIDTH_PI' hazma/ | sort
hazma/_decay/common.pxd:2
hazma/_positron/parameters.pxd:2
hazma/_utils/constants.pxd:6
hazma/_utils/legacy_parameters.pxd:2
```

i.e. six definition lines across the three legacy tables plus
`constants.pxd`'s own (correct) set, and **no consumer anywhere** —
and `_template.md` files (schema, not artifacts).

**Count sweep:**

| Claim location | Command | Actual | Status |
| --- | --- | --- | --- |
| This note / phase file: seven `.pyx` include sites repointed | `git diff origin/master --stat -- 'hazma/**/*.pyx' \| tail -1` | `7 files changed, 7 insertions(+), 7 deletions(-)` | OK |
| This note: 32 extensions built | `find hazma -name '*.so' \| wc -l` | `32` | OK |
| This note: `test/` suite | `pytest -q test` | `52 passed, 20 skipped` | OK |
| This note: in-package suite | `pytest -q` | `57 passed, 10 skipped` | OK |
| This note: 64 arrays compared | `numimpact.py` stdout | `wrote 64 arrays` | OK |
| Exit criterion: values not merged into `constants.pxd` | `git diff origin/master -- hazma/_utils/constants.pxd \| wc -l` | `0` | OK |
| `../README.md`: "`test/` is green (51 passed / 20 skipped)" | `pytest -q test` | `52 passed, 20 skipped` | EDITED — now records both, with dates |

**Numerical-impact statement.** Grid: `np.logspace(-2, 3, 200)` MeV
photon/positron energies, over every public code path the diff can
reach — `scalar_mediator_decay_spectrum` (6 modes × 3 mediator masses),
`dnde_decay_s` / `dnde_decay_s_pt` (4 final states × 3 masses), vector
`dnde_decay_v` photon (5 modes × 3 masses) and positron (4 final states
× 3 masses), plus the model-level `ScalarMediator` / `VectorMediator`
`total_spectrum` and `total_positron_spectrum` wrappers — 64 arrays.
Captured at `origin/master`, then again after the relocation and a full
clean rebuild:

```text
arrays compared: 64
arrays NOT bit-identical: 0
max relative deviation: 0.000e+00
```

**No public value changes** — bit-for-bit identical, as a verbatim
relocation requires. Re-confirmed after a second clean rebuild:
`arrays compared: 64; not bit-identical: 0`.

**Exit Criteria → verification mapping:**

| Exit criterion | Satisfied by |
| --- | --- |
| Moved verbatim, byte-identical values | `diff` of the post-docstring block vs `HEAD` → exit 0 |
| Live include sites repointed | removed-path sweep (no live hits) + identifier sweep (all 7 at the new path) |
| `pip install -e .` rebuilds | clean `uv pip install -e .` → Built; 32 `.so` |
| Import smoke passes | `import smoke OK` over 9 modules |
| Values not merged into `constants.pxd` | `git diff origin/master -- hazma/_utils/constants.pxd` → 0 lines |

**Task-note self-consistency:** `**Status:** Complete` matches all five
mapping rows satisfied; every file named in §Files Changed appears in
`git diff origin/master --stat --` (7 `.pyx` + the rename) or is a file
created by this task (this note, the follow-up); the phase README row
and this note's status agree.

**Fixed-point re-run:** every command above was re-run after the prose
was frozen. The deterministic ones — `rg -c`, `diff`, `wc -l`,
`check_doc_citations.py`, `git diff --stat`, the npz comparison —
reproduced byte-identically; the two `rg -n` blocks reproduced the same
rows under `sort`.

<!-- markdownlint-enable MD013 -->

## Handoff to Next Task

- **Read first:** `../README.md` (project working memory) → this phase's
  `README.md` → the phase file. Then Task 0.5, which is the gate.
- **Now safe to assume:** `hazma/_utils/legacy_parameters.pxd` is the
  only home of the legacy constants table, and nothing under
  `hazma/_decay/` is referenced by a *built* extension any more — the
  directory is free to delete in Task 0.3 with no include-path fallout.
  The project-level open question about the cross-section `.pyx` is
  closed: they include nothing.
- **Build hygiene, load-bearing:** delete stale `.c`/`.cpp`/`.so` before
  every rebuild in this project (see Findings) or you will debug a
  phantom NumPy-2 compile error in generated code.
- **Still risky / unresolved:** Task 0.2 remains blocked on ADR-0003
  sign-off, which no implementation step can unblock. Task 0.3 is
  unblocked and independent of it.
