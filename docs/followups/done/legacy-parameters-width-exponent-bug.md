# `WIDTH_K` / `WIDTH_PI` exponent bug in the legacy constants tables

- **Added:** 2026-08-04
- **Source:** `projects/cython-to-rust/references/cython-inventory.md`
  "Bugs" §3 (original observation); surfaced again and given a durable
  home by
  `projects/cython-to-rust/task-notes/phase-00/task-0.1-relocate-constants.md`
- **Scope:** cross-cutting
- **Status:** done — resolved 2026-08-05 by deleting both names from
  `hazma/_utils/legacy_parameters.pxd`; see [Resolution](#resolution).
- **Triggers / blockers:** ~~ripens when the constants tables are
  consolidated~~ — resolved ahead of consolidation. The deferral existed
  so the repair would not be smuggled into a mechanical file move; a
  standalone change is exactly what it asked for, and deleting two names
  that nothing reads does not touch the divergent values consolidation
  still has to reconcile.

## Why

Three copies of the legacy Standard-Model constants table define the
charged-kaon and charged-pion widths with `**` where a decimal exponent
was meant:

```cython
cdef double WIDTH_K = 3.3406**-13.
cdef double WIDTH_PI = 2.528511206475808**-14.
```

`3.3406**-13.` is exponentiation, not scientific notation: it evaluates
to 1.5498e-7 rather than a width of order 1e-13 MeV. Likewise
`2.528511206475808**-14.` = 2.2903e-6 instead of ≈ 2.53e-14 MeV. (The
figures first recorded here, ≈ 1.8e-7 and ≈ 4.4e-6, were estimates; the
evaluated values are the ones above. The conclusion — six to eight
orders of magnitude too large — is unchanged.) That
`WIDTH_PI` was meant to read `2.528511206475808e-14` is confirmed by
`hazma/_utils/constants.pxd:321`, which carries the same quantity
correctly as `WIDTH_PI = 2.5284e-14` (Γ[π+] = 2.5284e-14 ± 5e-18 MeV).

**No value is wrong in any published output today, because nothing reads
these two names.** A repo-wide search finds only the six definition
lines and no consumer (see Entry points), independently confirming the
"~10⁶ off; both currently unused" note the cython-to-rust audit already
recorded in `references/cython-inventory.md` "Bugs" §3.

This gets a follow-up of its own — rather than living only in that
audit — because the audit reference is a project-scoped snapshot that
retires when cython-to-rust closes, whereas
`hazma/_utils/legacy_parameters.pxd` survives the whole migration and
carries the defect forward.

Note the two names are not merely mistyped but also inconsistent between
tables: the legacy `WIDTH_K` mantissa is `3.3406`, whereas
`constants.pxd:324` records Γ[K+] = 5.317e-14 MeV. Repairing the
exponent alone would still leave a wrong kaon width; the correct value
has to be sourced from the PDG, not inferred from the typo.

## What

As part of (or immediately after) constants consolidation:

1. Decide the canonical source for both widths — `constants.pxd`'s
   PDG-cited values are the obvious candidate.
2. Delete `WIDTH_K` / `WIDTH_PI` from the legacy tables rather than
   repairing them in place, if consolidation makes the legacy tables
   redundant. If any legacy table survives, fix the literals and cite
   the PDG value inline, matching `constants.pxd`'s comment style.
3. Confirm the "no consumer" claim still holds at that time; if a
   consumer has appeared, the fix becomes a declared numerical change
   and needs a CHANGELOG line and a magnitude, per
   `projects/cython-to-rust/rules.md` parity rule 3.

## Entry points

- ~~`hazma/_utils/legacy_parameters.pxd:63-64`~~ — relocated in
  cython-to-rust Task 0.1, values kept verbatim for bit-parity;
  **deleted** here. The file's WIDTHS section now carries only a note
  saying why it is empty.
- ~~`hazma/_decay/common.pxd:75-76`~~ and
  ~~`hazma/_positron/parameters.pxd:56-57`~~ — the second and third copies,
  **deleted** in cython-to-rust Task 0.3, which left
  `legacy_parameters.pxd` as the only surviving copy. No copy of the bad
  literals remains in the tree.
- `hazma/_utils/constants.pxd:321,324` — the correct, PDG-cited values.
- Related project: `projects/cython-to-rust/` — see `rules.md`
  ("Constants" rule 1) and `references/cython-inventory.md` ("Bugs" §3)
  for the wider constants-divergence picture.

## Risks / open questions

- Task 0.3 has landed, so only `legacy_parameters.pxd` and
  `constants.pxd` carry these constants now; Task 6.4 retires
  `legacy_parameters.pxd` itself. The consolidation surface is already
  down to two files.
- The correct Γ[K+] must come from a PDG citation; do not back it out of
  the `3.3406` mantissa, whose provenance is unknown.

## Resolution

**Both names deleted from `hazma/_utils/legacy_parameters.pxd`.** Its
WIDTHS section is now an empty section carrying a note that says why.
`hazma/_utils/constants.pxd` is the canonical source for decay widths;
nothing else in the tree defines either name.

### Why delete rather than repair

"What" step 2 offered a repair branch — *if any legacy table survives,
fix the literals in place and cite the PDG value inline*. That branch was
written against consolidation of the table as a whole; applied to these
two names specifically it is the worse option, for three reasons:

1. **There is no parity to preserve.** The legacy table exists so the
   five `.pyx` that `include` it keep compiling against the exact values
   they always used. A name none of them reference serves no part of
   that contract.
2. **A repair recreates the divergence class the audit is trying to
   close.** Corrected literals in both tables means two definitions of
   the same PDG quantity that must be kept in sync — and consolidation
   would delete one of them anyway.
3. **It would make the file lie about itself.** The header declares a
   verbatim relocation whose values are deliberately divergent and must
   not be "fixed". Two silently-repaired entries in a forty-entry table
   of deliberate legacy values is a trap for the next reader. Deleting
   them, with a note, is not.

Consolidation of the *rest* of the table (`MASS_E`, `BR_PI_TO_ENU`, and
the other divergences) is untouched and still pending; it remains the
declared numerical change `projects/cython-to-rust/rules.md` describes.

### Step 3: the "no consumer" claim, re-confirmed

Two independent checks, both current as of this change:

- `git grep -n 'WIDTH_K\b\|WIDTH_PI\b'` over the whole tree returns hits
  only in `constants.pxd` (its own, correct definitions), the two
  `legacy_parameters.pxd` lines being removed, and prose in
  `docs/`/`projects/`. No `.pyx` reads either name.
- The tree rebuilds. `uv pip install -e .` recythonized and compiled all
  25 extensions, including the five that `include` the edited header
  (`{scalar,vector}_mediator/*_decay_spectrum.pyx`,
  `{scalar,vector}_mediator/*_positron_spec.pyx`,
  `_gamma_ray/gamma_ray_generator.pyx`). A `cdef`-context reference to a
  now-undeclared name would have been a Cython compile error, so this is
  a check the grep cannot give on its own — though a Python-level
  reference would have deferred to a runtime `NameError` instead, which
  is why both checks are recorded.

Neither check tells you the *outputs* did not move, so that was measured
rather than argued. The baseline is `f125493` (`master` at the time of
this branch's final rebase, its `legacy_parameters.pxd` still carrying
`3.3406**-13.`); it was extracted to a scratch tree and built into its
own venv, this branch was built into another, and both were driven
through every public entry point of the four affected extensions —
`scalar_mediator_decay_spectrum`, `dnde_decay_v` / `dnde_decay_v_pt`
(photon), and `dnde_decay_s` / `dnde_decay_v` (positron), point and
array forms, each mediator mass in {250, 500, 1000} MeV, each mode
string the extensions accept, over `np.logspace(-2, 3, 200)` MeV:

```text
arrays: 114  elements: 22800  nonzero elements in baseline: 14175
arrays differing bit-for-bit: 0
```

The same comparison was run twice, against `afa6e14` before the final
rebase and against `f125493` after, with identical results. The baseline
is pinned to a commit rather than to "`origin/master`" because `master`
moved twice under this branch — and PR #43 landed real numerical changes
to `hazma/utils.py` in between, so an unpinned baseline would make this
block mean something different on each re-read.

The fifth including extension, `_gamma_ray/gamma_ray_generator`, is
covered by the compile check only: it raises `ImportError: cannot import
name rambo` on `f125493` as well, the known breakage recorded in
`references/cython-inventory.md`. It is not callable to compare.

So the "declared numerical change" escape hatch in step 3 does not
apply: no CHANGELOG entry and no magnitude, because no published number
moves. This mirrors the precedent in
[`done/black-pin-divergence-pyproject-vs-ci.md`](black-pin-divergence-pyproject-vs-ci.md)
— nothing on the public Python API moved.

### The values, for the record

The canonical values in `constants.pxd` were checked against PDG
lifetimes rather than taken on trust (Γ = ħ/τ, ħ = 6.582119569e-22
MeV·s):

| Quantity | PDG τ | ħ/τ | `constants.pxd` | Deleted legacy literal |
| --- | --- | --- | --- | --- |
| Γ[π⁺] | 2.6033e-8 s | 2.52838e-14 MeV | 2.5284e-14 | `2.528511206475808**-14.` = 2.2903e-6 |
| Γ[K⁺] | 1.2380e-8 s | 5.31674e-14 MeV | 5.317e-14 | `3.3406**-13.` = 1.5498e-7 |

Both `constants.pxd` entries agree with ħ/τ to within the precision of
the quoted mantissas (1e-5 and 5e-5 relative), so the canonical source
decision in step 1 stands on a checked number, not a citation alone.
Note that `3.3406` is not Γ[K⁺]'s mantissa under any exponent — reading
the literal as `3.3406e-13` would still have been wrong by a factor of
6.3. Its provenance stays unknown, and nothing was inferred from it, as
this file's "Risks" section required.

### Gate

`scripts/agents/preflight.sh` on this branch: `black --check` PASS,
`pytest` PASS (95 passed / 20 skipped), import smoke PASS,
`markdownlint` PASS on the seven touched docs, forbidden tokens PASS.
`isort` and `ruff check` FAIL — the standing tree-wide debt described in
`docs/agents/environment.md`, not this diff: `git diff --name-only
origin/master -- '*.py'` is empty, so neither gate can be reading a line
this change wrote. `scripts/agents/check_doc_citations.py` over the
seven touched docs is clean (81 citations, 0 out of range).

### Docs repointed

`docs/followups/README.md` (row moved to the done table),
`projects/cython-to-rust/rules.md` ("Constants" rule 1 now records the
one settled exception to "verbatim"),
`references/cython-inventory.md` ("Bugs" §3 marks the widths gone and
the `MASS_E` / `BR_PI_TO_ENU` divergences still open),
`task-notes/phase-00/README.md` (Open Question closed, Files Changed
entry repointed), and the Task 0.1 and Task 0.3 notes' references to
this file.

Every live reference uses the `done/` path. A status-stripped
`docs/followups/<slug>.md` form was tried first, to avoid asserting a
path in a record of what a past task did; review (PR #44) correctly
rejected it, since that form resolves to no file at all. The two records
now give the `done/` path and say the `todo/` history in prose. The
verbatim pasted-command blocks in the Task 0.1 note still contain the
`todo/` path and are deliberately left alone — they are the output of
commands run on 2026-08-04, and editing them would make them not that.
Sweep for regressions with:

```sh
rg -oN --no-filename 'docs/followups/[A-Za-z0-9_./-]*\.md' -g '*.md' . \
    | sort -u | while read -r p; do [ -e "$p" ] || echo "MISSING $p"; done
```
