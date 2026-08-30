# Task 7.3: Documentation sweep

**Date:** 2026-08-29
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-07-cutover.md` (Task 7.3);
`../../PLAN.md` (Scope, Orientation)
**Related ADRs:** none
**Depends On:** Task 7.1 (maturin backend), Task 7.2 (release pipeline)

## Objective

Remove the last Cython facts from the repository's live instruction
surface — `AGENTS.md`, `docs/agents/`, the `.claude/` and `.codex/`
skills, the README and `docs/source/` install instructions — and dispose
of the setuptools-era artifacts (`requirements.txt`, `Dockerfile`,
`setup.cfg`'s `[aliases]`, the four tracked editor leftovers) that the
maturin cutover left behind.

## Exit Criteria

Copied from `../../phases/phase-07-cutover.md`, Task 7.3.

- `AGENTS.md` rewritten where it states Cython facts (layout tree,
  "Editing a .pyx requires a rebuild", layering §1, commands);
  `docs/agents/` env notes updated (Rust toolchain requirement, uv +
  editable-rebuild loop); `CLAUDE.md`/skills references checked by the
  doc-consistency checklist. The phase file enumerates eleven sites in
  `AGENTS.md` + `docs/agents/environment.md` and three package-data
  sites in the skills; line numbers re-derived before editing.
- README / `docs/source` install instructions updated (Rust toolchain
  only needed for source builds; wheels cover normal installs);
  Sphinx/RTD build verified against the maturin-built package.
- Stale artifacts removed: `requirements.txt`, `Dockerfile`,
  `docs/source/installation.rst`'s `python setup.py install` line, and
  `setup.cfg`'s `[aliases]` section — or updated if kept deliberately.
  Also decide whether the four tracked editor leftovers under `hazma/`
  (`{A_eff,energy_res}/gecco.dat.bak`, `_gev.py.bak`,
  `form_factors/notes.org`) should be deleted.

## Inputs Reviewed

- `../../PLAN.md` — Scope, Numerical impact, Phases table.
- `../../phases/phase-07-cutover.md` — Prerequisites and the Task 7.3
  site enumeration.
- `README.md` (phase working memory) — Task 7.1/7.2 findings and the
  open question handed here.
- `../../rules.md` — no rule binds a docs-only diff.
- `docs/agents/{doc-consistency,lessons,environment,preflight,review-lenses,README}.md`.
- `docs/{workflow,versioning,PR_GUIDELINES}.md`,
  `docs/source/installation.rst`, `README.md`, `requirements.txt`,
  `Dockerfile`, `setup.cfg`, `pyproject.toml`.
- `.claude/skills/*/SKILL.md`, `.codex/skills/*/SKILL.md`.
- `test/test_no_cython_remains.py` — the tree-wide build invariant.

## Findings

- **The phase file's enumeration named 14 sites in 5 files; the pattern
  it cites matches 72 lines in 23 files.** `git grep -c -E
  'Cython|cython|\.pyx|\.pxd|cythoniz' origin/master -- AGENTS.md
  CLAUDE.md README.md docs/agents docs/workflow.md docs/versioning.md
  docs/PR_GUIDELINES.md docs/source .claude/skills .codex/skills`
  (excluding `lessons-examples.md`) returns that; the enumeration was
  scoped to `AGENTS.md` and `docs/agents/environment.md`, which hold 27
  of the 72, plus three package-data lines Task 7.2 added by hand. The
  skills alone carry the pattern in **12 files**, none of them reachable
  from the enumeration's two-file grep. A claim written once and pasted —
  here the rebuild trigger ``.pyx` / `.pxd` / `rust/` / `setup.py`` — has
  to be swept by its *text* across the whole instruction surface, not
  from the file that first stated it.
- **`hazma/_utils/` outlived its contents.** Task 6.4 deleted the four
  Cython headers and left the package's empty `__init__.py`, so the
  directory has been an importable no-op since. Nothing imports
  `hazma._utils`: every `._utils` hit in the tree is a sibling package
  (`rh_neutrino/`, `spectra/_positron/`, `spectra/_neutrino/`,
  `phase_space/`, `form_factors/vector/`). `docs/versioning.md` used it
  as *the* example of a non-public package, which is what made a dead
  directory look load-bearing.
- **Two of the four "editor leftovers" are not backups.**
  `hazma/gamma_ray_data/A_eff/gecco.dat.bak` cites a different source
  than its neighbor — "Taken from slides from Alexander Moiseev" against
  "From Alexander Moiseev Febuary 7th, 2022" — and tabulates a different
  grid, starting at 0.1 MeV rather than 0.2. A `.bak` suffix is not
  evidence that a file is a copy of the one beside it.
- **Nothing builds the docs.** There is no `.readthedocs.yaml` in the
  repository, and neither `ci.yml` nor `preflight.sh` runs Sphinx, so a
  broken docs build turns nothing red. The build itself is healthy: exit
  0, 107 warnings, all pre-existing classes (36 duplicate object
  descriptions, 19 autodoc failures on `Theory`'s abstract methods, a
  theme option, an undefined label in `spectra.rst`).

## Decisions and Implementation Notes

- **`requirements.txt` and the `Dockerfile` are deleted, not updated.**
  The former pinned `Cython>=0.29.12` and `flake8` and was read by
  nothing but the latter; the latter builds on `jupyter/scipy-notebook`,
  installs from that file, and enables `jupyter_contrib_nbextensions`,
  which Notebook 7 dropped. Neither has worked for some time, and
  `pyproject.toml` answers both questions.
- **`setup.cfg` survives with `[aliases]` removed.** Its `[flake8]` and
  `[mypy]` sections are stale in a different way — this repo lints with
  ruff and type-checks with pyright — but that is a lint-tooling
  question rather than setuptools residue, and
  `test/test_no_cython_remains.py` reads the file. Left alone.
- **`hazma/_utils/` deleted.** `docs/versioning.md` excludes
  leading-underscore packages from the public surface explicitly, so an
  empty one nothing imports is removable — and removing it is what lets
  `AGENTS.md`'s layout tree be true without a footnote explaining a
  ghost. Both docs that used it as an example now name a package with
  contents.
- **The four tracked non-source files under `hazma/` are kept, and the
  question is tracked rather than left open.** Deleting a superseded
  detector response with its own cited provenance is a physics call, not
  a packaging one, and the four cost users nothing: `[tool.maturin]
  exclude` keeps them out of both artifacts and
  `test/test_no_cython_remains.py` asserts that. Reasoning and entry
  points: [`tracked-non-source-files-under-hazma.md`](../../../../docs/followups/todo/tracked-non-source-files-under-hazma.md).
- **Historical prose was left alone.** `docs/agents/lessons-examples.md`,
  `docs/followups/`, `docs/adrs/ADR-0002` and every `projects/` note
  cite `.pyx` files as evidence of what happened. They are correct as
  written; a sweep that rewrites them destroys the record.
- **A new `environment.md` entry, from a trap this task hit.** The Bash
  tool shell here is zsh, which that file did not say; zsh does not
  word-split an unquoted `$VAR`, and its `MULTIOS` turns an unquoted
  multi-word redirect target into one redirect per word. A comparison
  loop of the shape `> $dir/$(echo $f | tr / _)` therefore truncated the
  repo-root `README.md` while only meaning to read it, silently. The
  file was restored from `origin/master` and its edit re-applied with
  the file tools; the entry records the mechanism and the sweep that
  finds an empty tracked file.

## Files Changed

- `AGENTS.md` — "What Hazma is", the layout tree, layering §1, the
  commands block, the rebuild paragraph, the private-package example;
  the generated-C/C++ bullet deleted.
- `docs/agents/environment.md` — three Cython entries deleted, the
  stale-artifact entry rewritten for `.rs`, the cargo requirement folded
  into one paragraph, and the zsh/`MULTIOS` entry added.
- `docs/agents/{README,doc-consistency,preflight,review-lenses}.md`,
  `docs/{versioning,workflow,PR_GUIDELINES}.md` — one claim each.
- `README.md`, `docs/source/installation.rst` — install instructions:
  wheels for normal installs, a Rust toolchain for source builds, no
  C/C++ compiler.
- `.claude/skills/{commit-and-pr,execute-single-task,review-cycle,review-plan,review-pr,review-respond,task-pipeline}/SKILL.md`
  and `.codex/skills/{commit-and-pr,execute-single-task,review-plan,review-pr,review-respond}/SKILL.md`
  — rebuild triggers, the layering invariant, the package-data findings.
- Deleted: `requirements.txt`, `Dockerfile`, `hazma/_utils/__init__.py`,
  and `setup.cfg`'s `[aliases]` section.
- `pyproject.toml` — the `[tool.maturin] exclude` comment only; no key
  changed.
- `docs/followups/todo/tracked-non-source-files-under-hazma.md` (new)
  and its row in `docs/followups/README.md`.
- `../../phases/phase-07-cutover.md` — Prerequisites gain the Task 7.3
  facts; both enumeration tables gain a disposition column; two
  count-agreement fixes ("two more sites" over a three-row table,
  "Both tell" over three).
- `README.md` (phase working memory) and this note.

## Verification

- `PATH=".venv/bin:$PATH" scripts/agents/preflight.sh --paths "hazma test"
  --md "<the 15 touched markdown files>"` — gate table under
  `## Stale-state sweep`.
- `pytest` (bare, via preflight) — **2231 passed, 15 skipped, 12
  subtests passed in 26.49s**, from the final gate run after the last
  prose edit. Same counts as Task 7.1 recorded, on a tree rebuilt after
  the `hazma/_utils/` deletion.
- `.venv/bin/python -m pytest test/test_no_cython_remains.py -q` —
  **4 passed in 1.18s**. The four build invariants: no Cython source
  anywhere, no setuptools build script, `[build-system] requires ==
  {maturin}`, and the editor-leftover exclusion.
- `uv pip install --python .venv/bin/python -e . --group dev` then
  `python -c "import hazma; print(hazma.__file__)"` →
  `.../charming-mcnulty-7e66f4/hazma/__init__.py`; `hazma._core.__file__`
  → `.../charming-mcnulty-7e66f4/hazma/_core.abi3.so`. Both inside the
  worktree.
- `python -c "import hazma._utils"` → `ModuleNotFoundError: No module
  named 'hazma._utils'`, after the reinstall. The deletion took effect
  rather than being masked by a stale install.
- `python -m sphinx -b html docs/source <out>` — **exit 0**, 107
  warnings, none from `installation.rst`. Re-run with `-W --keep-going`
  to enumerate them: 129 warnings-as-errors, same classes. The rendered
  `installation.html` contains `rustup` twice and `cargo` once and
  neither `Cython` nor `cython`.
- Deferred: no attempt to reduce the 107 Sphinx warnings or the
  README's 13 remaining markdownlint findings. Both predate this task
  and belong to files it barely touches; see the sweep block for the
  before/after counts that show neither got worse.

## Open Questions

- **`setup.cfg`'s `[flake8]` and `[mypy]` sections configure tools this
  repo does not run.** Linting is ruff (`pyproject.toml`'s
  `[tool.ruff]`) and type-checking is pyright (`AGENTS.md`'s commands
  block). Neither section is setuptools residue, so neither was in this
  task's scope, and `test/test_no_cython_remains.py` reads the file for
  a different reason. Whoever settles the linter set should decide
  whether the file keeps anything but that.
- **`preflight.sh`'s markdownlint gate is red on `README.md` at
  `origin/master`** (14 findings), so passing `--md README.md` fails the
  gate on any PR that touches it. This task took the count to 13 and
  added none. It is the same shape as the open
  [`preflight-isort-ruff-red-on-trunk`](../../../../docs/followups/todo/preflight-isort-ruff-red-on-trunk.md)
  follow-up and is recorded there rather than in a second stub.

## Plan Impact

**Impact Level:** Update phase file.

No canonical behavior changed — the diff is documentation, two deleted
build-era files, and one empty private package. The phase file was
patched for accuracy rather than for scope: its Prerequisites block
gains the Task 7.3 facts that Task 7.4 needs, its two per-site
enumeration tables gain a disposition column, and two prose counts that
disagreed with their own tables were corrected. Task 7.3's exit criteria
themselves are unchanged and were met as written. No ADR: nothing here
is a decision a future reader needs the reasoning chain for beyond what
the follow-up stub already carries.

## Stale-state sweep

Run against this branch after every prose edit was frozen. Folded where
noted; all other rows are pasted command output.

### Identifier sweep

`rg -c 'Cython|cython|\.pyx|\.pxd|cythoniz' AGENTS.md CLAUDE.md README.md
docs/agents/ docs/workflow.md docs/versioning.md docs/PR_GUIDELINES.md
docs/source/ .claude/skills/ .codex/skills/ | grep -v lessons-examples.md`
— **15 matching lines in 7 files.** Folded to one row per file with a
disposition (files with zero hits do not appear in `rg -c` output and are
listed together):

| File | Hits | Disposition |
| --- | --- | --- |
| `AGENTS.md` | 1 | KEPT — `cython-to-rust Task 0.2` provenance on the `deprecated/` rule |
| `docs/versioning.md` | 2 | KEPT — project-slug citations (Tasks 7.1, 0.2) |
| `docs/workflow.md` | 1 | KEPT — a `projects/cython-to-rust/task-notes/` path |
| `docs/agents/preflight.md` | 1 | KEPT — the pre-Phase-02 branch SKIP rule |
| `docs/agents/environment.md` | 7 | KEPT — lines 71–72, 115, 188, 195, 216 are slug citations; 93 names `test_no_cython_remains.py` |
| `.claude/skills/execute-single-task/SKILL.md` | 2 | KEPT — `projects/cython-to-rust/` paths |
| `.codex/skills/execute-single-task/SKILL.md` | 1 | KEPT — same |
| `CLAUDE.md`, `README.md`, `docs/source/`, `docs/PR_GUIDELINES.md`, the other 10 skills | 0 | — |

No hit states a Cython fact as current. `docs/agents/lessons-examples.md`
(excluded above), `docs/followups/` and `projects/` keep theirs by
decision, recorded under §Decisions.

`rg -n 'package-data|package data' .claude/ .codex/ docs/` — 6 hits, none
in a skill. One was live and is EDITED: `docs/agents/environment.md:281`
described CI's import smoke as catching "a missing package-data entry",
a concept maturin does not have; it now says "a data file missing from
the distribution". The other five are in `docs/followups/README.md` and
`docs/followups/done/sdist-ships-generated-c-and-docs.md`, KEPT as the
record of the block Task 7.1 deleted.

`rg -n 'setuptools' --glob '!projects/**' --glob '!docs/followups/**'
--glob '!docs/agents/lessons-examples.md' --glob '!rust/target/**'` — 13
hits, all KEPT and all labeled history: `pyproject.toml` ×4 (comments
explaining what the maturin config replaced), `docs/agents/environment.md`
×2, `test/test_no_cython_remains.py` ×6 (the invariant's own docstrings),
`test/test_core_mediator_tables.py` ×1 (the debug/release profile note).

`rg -n 'hazma\._utils|hazma/_utils' hazma/` — 1 hit, KEPT:
`hazma/_core.pyi:40`, a comment recording which capsule the dispatch
contract replaced. No executable reference survives; `import hazma._utils`
raises `ModuleNotFoundError` (see §Verification).

### Line-number citation sweep

`check_doc_citations.py` run with the 15 touched docs as explicit
arguments, because `--changed-vs origin/master` reads committed history
and this tree is uncommitted:

```text
docs scanned: 15
in-repo citations checked: 0
external citations skipped: 0
out-of-range or ambiguous: NONE
```

The scope is real (15 docs) but the citation count is zero, so the tool
proves nothing on its own here. Checked by hand instead: `rg -n
'[A-Za-z0-9_/.-]+\.(py|rs|toml|md|cfg|rst):[0-9]+'` over the same 15
files returns exactly two `file:line` citations, both in this note and
both re-derived after the last edit — `docs/agents/environment.md:281`
is the sentence this task rewrote (`rg -n 'data file missing from the
distribution'` → 281) and `README.md:38-51` is the install block
(`rg -n 'prebuilt wheel on manylinux'` → 38; the block ends at 51).

### Forward-looking phrase sweep

`rg -n '(Task [0-9]+ will|will be added|still pending|today: ?stub)'
projects/cython-to-rust/task-notes/phase-07/
projects/cython-to-rust/phases/phase-07-cutover.md hazma/` — 3 hits, all
of them the sweep command quoted inside a task note (this one and the
Task 7.1 / 7.2 notes), which the pattern matches by construction. No
prose hit: nothing in the touched files promises future work.

### Count sweep

Every row re-derived after the last prose edit. Three claims written from
memory were wrong and are corrected in place rather than defended.

| Claim location | Command | Actual | Status |
| --- | --- | --- | --- |
| note: "14 sites in 5 files" | count the phase file's two tables | 11 + 3 = 14, over `AGENTS.md`, `environment.md`, 3 skills | OK |
| note: "72 lines in 23 files" | `git grep -c -E '<pat>' origin/master -- <surface>` less `lessons-examples.md` | 72 lines, 23 files | FIXED (read "34 stale sites") |
| note: "27 of the 72" | same command, `AGENTS.md` 10 + `environment.md` 17 | 27 | OK |
| note: "12 files" (skills) | `git grep -l -E '<pat>' origin/master -- .claude/skills .codex/skills` | 12 | FIXED (read "seven skill files") |
| note: 15 lines / 7 files remain | `rg -c` over the same surface, this branch | 15 lines, 7 files | OK |
| phase file: "three more sites" | rows in the Task 7.2 table | 3 | FIXED (read "two") |
| phase file: "Each tells a reviewer" | same table | 3 rows | FIXED (read "Both") |
| note: "107 warnings" | `python -m sphinx -b html`, `grep -c WARNING` | 107 | OK |
| note: "2231 passed, 15 skipped" | bare `pytest` via preflight | 2231 passed, 15 skipped, 12 subtests | OK |
| note: "4 passed in 1.18s" | `pytest test/test_no_cython_remains.py -q` | 4 passed in 1.18s | OK |
| note: README markdownlint 14 → 13 | `markdownlint --dot` on each revision | 14 (master), 13 (head) | OK |
| note: "30 files changed" | `git diff origin/master --stat` | see §Files Changed | re-derived below |

### Numerical-impact statement

**No public value changes.** The diff contains no `hazma/` source edit
other than deleting `hazma/_utils/__init__.py`, an empty file in a
package nothing imports. Verified: `pytest` (bare) is 2231 passed / 15
skipped, identical to the counts Task 7.1 recorded on `origin/master`,
and that run includes the golden parity corpus under `test/parity/`,
which pins all 41 entry points. `python -c "import hazma._utils"` raises
`ModuleNotFoundError` and no other import changed.

### Exit Criteria → test mapping

| Exit-criterion bullet | Satisfied by |
| --- | --- |
| `AGENTS.md` Cython facts rewritten | the six `AGENTS.md` rows of the phase-file table, each marked with its disposition; identifier sweep shows 1 surviving hit, a slug citation |
| `docs/agents/` env notes updated | the five `environment.md` rows, same table; the cargo-requirement paragraph is now the only build requirement stated |
| `CLAUDE.md`/skills checked | identifier sweep over `.claude/skills/` and `.codex/skills/`; `CLAUDE.md` has 0 hits (it is one `@AGENTS.md` include) |
| README / `docs/source` install updated | `README.md:38-51`, `docs/source/installation.rst` rewritten; rendered `installation.html` checked for `rustup`/`cargo`/`Cython` |
| Sphinx build verified | `python -m sphinx -b html docs/source <out>` exit 0 against the editable maturin build |
| `requirements.txt`, `Dockerfile` removed | `git status` shows both `D`; `rg 'requirements\.txt'` outside `docs/followups/` returns nothing |
| `installation.rst`'s `python setup.py install` gone | file rewritten; `rg 'setup\.py' docs/source/` → no occurrences |
| `setup.cfg` `[aliases]` removed | `rg -n 'aliases' setup.cfg` → no occurrences |
| four editor leftovers decided | kept, with reasoning in §Decisions and `docs/followups/todo/tracked-non-source-files-under-hazma.md`; index row added |

### Preflight gate

```text
PASS   black --check           hazma test
FAIL   isort --check-only      run `isort hazma test` and re-check
FAIL   ruff check              see output below
PASS   cargo fmt --check       rust/
PASS   cargo clippy            rust/
PASS   cargo test              rust/
PASS   pytest                  2231 passed, 15 skipped, 12 subtests passed
PASS   import hazma            version 2.1.0
FAIL   markdownlint            see output below
SKIP   version bump            not a closing PR (pass --closing)
PASS   forbidden tokens        none added
RESULT: FAIL — blocked commit. Fix the red gates and re-run.
```

All three red gates are red at `origin/master` and none is reachable by
this diff:

- **isort and ruff** run over `hazma test`, and this diff changes no
  Python content. `git diff origin/master --stat -- '*.py' '*.pyi'` is
  `hazma/_utils/__init__.py | 0` — **1 file changed, 0 insertions, 0
  deletions**, the deletion of an empty file. `git diff origin/master
  --name-only | sed 's/.*\.//' | sort | uniq -c` over the final set is
  `1 Dockerfile, 1 cfg, 27 md, 1 py, 1 rst, 1 toml, 1 txt`. Both gates
  are the open
  [`preflight-isort-ruff-red-on-trunk`](../../../../docs/followups/todo/preflight-isort-ruff-red-on-trunk.md)
  follow-up.
- **markdownlint** is red on `README.md`, which carries 14 findings at
  `origin/master` and 13 here — this task removed one long line and added
  none. Every other file passed to `--md` is 0/0. Per-file counts, HEAD
  against the `origin/master` blob of the same path:

| File | HEAD | master |
| --- | --- | --- |
| `README.md` | 13 | 14 |
| the other 12 pre-existing `--md` files | 0 | 0 |
| `docs/followups/todo/tracked-non-source-files-under-hazma.md` | 0 | (new) |
| `projects/.../task-7.3-docs-sweep.md` | 0 | (new) |

The gate is therefore not green, and this is stated rather than worked
around: no `--md` list was trimmed to hide `README.md`.

### Task-note self-consistency

`**Status:** Complete` matches the phase README's Tasks-table cell and
the disposition columns in the phase file. Every file named in §Files
Changed appears in `git diff --stat origin/master --` or is one of the
two created files. The phase file's `status:` stays `In progress` —
Task 7.4 is open, so the phase is not closed and no learnings file is
written yet.

## Handoff to Next Task

- **Read `../../PLAN.md`'s "Closing this project" section first**, then
  `../numerical-impact.md` — the CHANGELOG's per-function drift table is
  assembled from that log, not from memory — then the phase README's
  rewritten handoff, which carries the rest.
- **Safe to assume:** no live instruction doc states a Cython fact; the
  version is `2.1.0` in `pyproject.toml`'s `[project] version`; the docs
  build clean-ish (exit 0) but are gated by nothing; `requirements.txt`,
  `Dockerfile` and `hazma/_utils/` no longer exist.
- **Still risky:** `preflight.sh`'s isort, ruff and (for `README.md`)
  markdownlint gates are red at `origin/master`, so a closing PR must
  compare counts against the trunk rather than expecting green; and
  `--paths` feeds source paths to the Python formatters, so markdown
  goes in `--md`.
- **Task 7.4 owes a follow-up cross-check over all of
  `docs/followups/todo/`,** which now includes
  `tracked-non-source-files-under-hazma.md` filed by this task.
