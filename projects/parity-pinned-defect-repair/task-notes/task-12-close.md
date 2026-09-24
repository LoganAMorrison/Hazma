# Task 12: Close — aggregate the drift, bump

**Date:** 2026-09-24
**Project:** parity-pinned-defect-repair
**Status:** Complete
**Plan References:** `../PLAN.md` Task 12, "Numerical impact", "Closing
this project", "Anticipated ADRs"; `../rules.md` rules 1, 10 and 11
**Related ADRs:**
`../adrs/ADR-0003-keep-the-group-a-oracle-captures-committed.md` (new)
**Depends On:** Task 11

## Objective

Write one `CHANGELOG.md` entry that names this project's slug and
carries every measured shift, and make the `minor` bump. Close the
project's bookkeeping.

## Exit Criteria

From `../PLAN.md` Task 12 and the working-memory README's Exit Criteria:

- [x] `scripts/agents/preflight.sh --closing` green, except one
  pytest failure caused by the host's memory, not by this diff (see
  Verification).
- [x] `PLAN.md` `status: Complete`.
- [x] The eight follow-ups still under `todo/`, the original seven plus
  B5's, moved to `docs/followups/done/`, with inbound links repointed and
  revision citations pinned.
- [x] `pyproject.toml` `[project] version` bumped per
  `version_bump: minor`, with the level re-checked against the aggregate.
  B6 was checked hardest.
- [x] A `CHANGELOG.md` entry that names the slug and carries the
  aggregated per-defect shifts.
- [x] `git diff --stat -- test/parity/data` empty across the whole
  project.
- [x] Closure: `learnings/project-retrospective.md`, a follow-up stub for
  every §5 seed, `projects/README.md` row moved, and a cross-check of
  `docs/followups/todo/` for entries this project sourced.

## Inputs Reviewed

- `../PLAN.md` (whole), `../rules.md`, `README.md` (whole).
- `task-11-prose-reconciliation.md`: Open Questions, Plan Impact, and
  the sweep block.
- `docs/workflow.md` §Follow-ups and §Project lifecycle, and
  `docs/versioning.md`.
- `scripts/agents/preflight.sh` Gate 10 (`--closing`).
- `docs/followups/todo/moved-followups-leave-dangling-inbound-paths.md`,
  for the transcript-repoint convention.
- `git show b150b4f`, `git show b4b468e`: the `cython-to-rust` close and
  the 2.2.0 release, used as the precedent for the heading rename.

## Findings

- **The level stays `minor`.** Since the 2.2.0 release commit
  `b4b468e`, `git diff b4b468e HEAD -- hazma | grep -E '^[-+]\s*(def |class |__all__)'`
  returns one line, the deletion of `hazma.utils.minkowski_dot`.
  `5c69d19` removed it before any release shipped it; its commit message
  says it "appears in no released tag". So no public name was removed.
  `git diff --stat b4b468e HEAD -- rust/src/dispatch.rs rust/src/lib.rs`
  is empty, so no compiled signature moved. B6 moves `relic_density` by
  up to −99.88%, but it changes no name, signature, return shape or
  unit, and `docs/versioning.md` puts a deliberate change to a published
  number under `minor`.
- **Nine of 2.2.0's twelve `Known issues` are repaired in this release.**
  The three still open are the scalar elastic cancellation, the mediator
  positron line, and the vector `TypeError` at `2 m_x`. B4 was in 2.2.0
  already.
- **The oracle manifest is provenance, not a live link.**
  `test/parity/oracles/data/manifest.json` records each capture's
  `follow_up` path as it was when captured. It was left unedited and
  `oracles/defects.py` was repointed instead (ADR-0003). The
  dangling-path follow-up now names that exemption.
- **One stale sibling claim.**
  `mediator-positron-line-misses-the-electron-velocity.md` said it was
  "best sequenced with" B1 and B5, both of which had landed. It now says
  so.

## Decisions and Implementation Notes

- **Rename `[Unreleased]`, do not rewrite it** (README, "Numerical
  impact so far"). The per-repair entries stay as their tasks wrote them.
  A lead paragraph and one table go above them. The table has one row
  per roster entry, each figure copied from the README's measured
  section, not from `PLAN.md`'s pre-repair estimates. The one entry in
  the release that is not from this project, the vector `mu mu` positron
  channel, is named as such.
- **Write the anticipated oracle-retention ADR, not a follow-up.** The
  question already had its answer: 321 of the 343 declared arrays read
  the captures at test time.
- **Transcripts are repointed and annotated**, following the convention
  in `moved-followups-leave-dangling-inbound-paths.md`. Each fenced block
  that contained a moved path gets a bracketed note after it. That is 15
  blocks in 8 task notes, counted with `git grep -c 'repointed the$'`.
- **One new seed stub** (`delta-declaration-layer-outlives-its-project.md`).
  The other §5 seeds were already filed by Tasks 6, 8, 10 and 10a.

## Files Changed

- `pyproject.toml`: 2.2.0 → 2.3.0. `docs/versioning.md`: the quoted
  version line.
- `CHANGELOG.md`: `[2.3.0] — 2026-09-24`, the summary table, and the
  repointed links.
- `docs/followups/{todo → done}/` × 8, each with a rewritten `Status:`.
  `docs/followups/README.md`: eight rows moved to the Done table, plus
  one new Open row.
- `docs/followups/todo/delta-declaration-layer-outlives-its-project.md`
  and `quad-limit-test-allocates-16-gib-in-scipy.md` (both new). `mediator-positron-line-misses-the-electron-velocity.md`,
  `moved-followups-leave-dangling-inbound-paths.md`,
  `neutrino-pion-continuum-loses-its-quadrature-support.md`,
  `rho-photon-outer-boost-misses-support.md`,
  `phi-omits-its-direct-pi0-photon-line.md`, and
  `done/parity-corpus-pins-ill-conditioned-points.md` (a relative link
  into `todo/` that was already broken): sibling links and claims.
- Inbound-path repoints (`followups/todo/<slug>` → `done/`) in 41
  files. These include `rust/src/kernels/{neutrino_muon,neutrino_pion,positron_pion}.rs`
  (comments only) and `test/parity/deltas.py`,
  `test/parity/oracles/defects.py` and four `test/test_core_*.py`
  (strings and docstrings only).
- `projects/README.md`, `../PLAN.md`, `README.md`,
  `../adrs/ADR-0003-…` (new),
  `../learnings/project-retrospective.md` (new), and this note.

## Numerical impact

**No public value changes** (verified: every non-markdown hunk is a
path string). The command
`git diff origin/master -- rust hazma test | grep '^[-+][^-+]' | grep -v 'followups/'`
prints nothing. The only other non-markdown hunk is the `pyproject.toml`
version line. The suite ran against a rebuilt editable install (see
Verification). What this task *aggregates* is the project's measured
drift, in `CHANGELOG.md` `[2.3.0]`.

## Verification

- Editable rebuild after the last `.rs` comment edit:
  `pip install -e . --config-settings build-args="--features test-probes"`
  exited 0. `hazma.__file__` resolves to
  `/home/user/Hazma/hazma/__init__.py` and `hazma._core.__file__` to
  `/home/user/Hazma/hazma/_core.abi3.so`, with version `2.3.0`.
- `python test/parity/oracles/capture.py --check`:
  `oracles OK: 4 defects / 940 arrays match the manifest (corpus manifest f476fb420caf)`.
- Preflight, run immediately before the commit with the full
  `pytest` suite. `PATH=/usr/local/bin:$PATH` was needed because the
  `pytest` first on `PATH` was a uv tool environment with no hazma and no
  xdist, and the dev group came from `pyproject.toml`'s
  `[dependency-groups]`, because pip 24.0 has no `--group`:

  ```sh
  PATH=/usr/local/bin:$PATH scripts/agents/preflight.sh \
      --paths "test/parity/deltas.py test/parity/oracles/defects.py test/test_core_neutrino.py test/test_core_photon_tables.py test/test_core_positron_muon.py test/test_core_positron_pion.py" \
      --md "<every changed .md>" --closing
  ```

  ```text
  PASS   black --check / isort --check-only / ruff check (0 new, 5 fixed)
  PASS   cargo fmt --check / cargo clippy / cargo test
  FAIL   pytest   1 failed, 2318 passed, 17 skipped, 1 warning, 37 subtests passed in 77.15s
  PASS   import hazma   version 2.3.0
  PASS   markdownlint
  PASS   version bump   2.2.0 → 2.3.0 + CHANGELOG entry
  PASS   forbidden tokens   none added
  ```

  **The one failure comes from the host, not the diff.** It is
  `test_core_quad.py::TestErrorBehavior::test_the_largest_accepted_limit_is_not_rejected`,
  where scipy's own `quad` at `limit=2**31 - 1` tries to allocate 16.0
  GiB on a 15 GiB container. `test/test_core_quad.py` is unchanged
  against `origin/master`, and nothing in this diff reaches `quad`. It
  is filed as
  [`quad-limit-test-allocates-16-gib-in-scipy.md`](../../../docs/followups/todo/quad-limit-test-allocates-16-gib-in-scipy.md).
  Every other test passes, the parity suite included.

## Open Questions

- None new. The seeds are in `../learnings/project-retrospective.md` §5.

## Plan Impact

**Impact Level:** Project closure, plus ADR-0003. `PLAN.md` changes only
in its status and in the "Numerical impact" lead-in, which said three
repairs had landed and that Task 12 would aggregate. No task shape
changed.

## Stale-state sweep

Run on the working tree against `origin/master` = `e360126`, after the
last prose edit to any file except this note. `$P` is the eight moved
slugs joined with `|`.

**Identifier sweep.** This finds any moved slug still addressed under
`todo/`:

```sh
git grep -n -E "todo/($P)\.md" -- . ':!projects/parity-pinned-defect-repair/task-notes/task-12-close.md'
```

```text
test/parity/oracles/data/manifest.json:887:      "follow_up": "docs/followups/todo/boost-integral-drops-last-i…
test/parity/oracles/data/manifest.json:4899:      "follow_up": "docs/followups/todo/photon-muon-rest-frame-end…
test/parity/oracles/data/manifest.json:8787:      "follow_up": "docs/followups/todo/charged-pion-photon-spectr…
test/parity/oracles/data/manifest.json:12330:      "follow_up": "docs/followups/todo/positron-muon-spectrum-no…
```

All four are KEPT. They are capture provenance (ADR-0003), and the
dangling-path follow-up now names them as the gate's exemption.

**Dangling-path sweep**, using the loop from
`docs/followups/todo/moved-followups-leave-dangling-inbound-paths.md`
with `rg --no-ignore`:

```text
DANGLING: docs/followups/todo/boost-integral-drops-last-interior-cell.md
DANGLING: docs/followups/todo/charged-pion-photon-spectrum-misses-the-forward-cone.md
DANGLING: docs/followups/todo/cross-section-prefactor-threshold-cancellation.md
DANGLING: docs/followups/todo/legacy-parameters-width-exponent-bug.md
DANGLING: docs/followups/todo/oracle-restore-revisions-for-the-mediator-decay-pyx.md
DANGLING: docs/followups/todo/photon-muon-rest-frame-endpoint-uses-the-wrong-power-of-r.md
DANGLING: docs/followups/todo/positron-muon-spectrum-normalization-inverted.md
```

There are seven paths, and every one is KEPT:

- Four are the manifest's, as above.
- The other three were already on the trunk before this task. They sit
  in the Phase 00 task notes and in the follow-up that tracks them, and
  it is that follow-up's own work to fix them.

**Relative-link check.** A script resolves every markdown link target in every
tracked `.md`. It reports `links 911 broken 7`, and all seven are
pre-existing non-links:

- `<slug>` placeholders in the two phase templates;
- `E` in `decaying-theory-positron-channels-share-the-last-closure.md:22`;
- `photon_pion::ENG_MU_PIRF` in `task-3.1-constants.md`;
- `import|from` in `task-13-thermal-quadrature.md`.

Eight relative links were EDITED:

- three in moved files that point at a sibling still in `todo/`;
- four, in three `todo/` files, that point at a sibling now in `done/`;
- one link from `done/parity-corpus-pins-ill-conditioned-points.md` into
  `todo/`, which was already broken before this task.

The seven counted elsewhere in this note are the first two groups.

**Line-number citations.** `--changed-vs` finds no docs before a commit,
so the changed docs were passed by path:

```sh
python scripts/agents/check_doc_citations.py \
    $(git diff --name-only HEAD --diff-filter=AMR | grep '\.md$')
```

```text
docs scanned: 53
in-repo citations checked: 27
out-of-range or ambiguous: NONE
```

The moved follow-ups' citations into deleted `.pyx` files are pinned
already: `f479b231^`, `665aed5`, `ed1fa20` and `e2698eb6^`. Their one
`.rs` citation, `rust/src/constants.rs:359`, was checked by hand. It is
`BR_PHI_TO_PI0_A`.

**Liveness sweep** over the eight moved files, for pending-closure
wording:

```sh
grep -rn -i -E 'retained (here|in .todo)|stays here until|remain with Task 12|Task 12 (moves|must|has to|renames)|still (in|under) .todo' \
    docs/ projects/parity-pinned-defect-repair/PLAN.md \
    projects/parity-pinned-defect-repair/references CHANGELOG.md
```

Before the fix it had three hits, all EDITED to name the repair's merge
commit:

- `photon-muon-…:56`, "Task 12 moves … and pins the revision";
- `rho-rest-frame-…:12` and `positron-muon-…:16`, "Remaining work:
  Task 12 moves".

After the fix one hit is left, `PLAN.md:658`. That is Task 12's own
gate sentence, and it is KEPT.

**Forward-looking phrase sweep**:

```sh
rg -n '(Task [0-9]+ will|will be added|still pending|today: ?stub|currently|In Progress)' \
    projects/parity-pinned-defect-repair/ hazma/
```

Every hit is KEPT:

- Quoted sweep commands and dated "remains In Progress" lines in the
  task notes for Tasks 3, 4, 6–10a and 13 and in the two review
  responses.
- `README.md:589`, where "concurrently" is used in the sense of "at the
  same time", not as a status.
- `_template.md:5`.
- Two `hazma/theory/_theory_constrain.py` messages, which have nothing
  to do with this task.

`PLAN.md:2` and `README.md:5` no longer match.

**Count sweep.**

| Claim location | Command | Actual | Status |
| --- | --- | --- | --- |
| retrospective §6, dangling-path follow-up, `README.md` Files Changed: 159 occurrences in 41 files | per changed file, count `followups/todo/($P)\.md` in `git show origin/master:<file>`, excluding `docs/followups/` itself | `159 41` | OK |
| ADR-0003, retrospective, README: 343 declared, 321 read captures | `Counter(d.repair for d in deltas.DECLARED_DELTAS.values())`, then sum the `A*` labels | `343 321` | OK |
| ADR-0003: about 1.6 MB of oracle data | `ls -l test/parity/oracles/data \| awk '{s+=$5} END {print s}'` | `1597175` | OK |
| CHANGELOG: nine of twelve known issues | the twelve `- **[` bullets under 2.2.0 `### Known issues`; nine now link into `done/` | 9 / 12 | OK |
| CHANGELOG table figures | copied from `README.md` "Numerical impact so far", one row per roster entry | — | OK |
| retrospective: corpus untouched across the project | `git diff --stat f11415a^ HEAD -- test/parity/data \| wc -l` and `git diff --stat -- test/parity/data \| wc -l` | `0`, `0` | OK |

**Numerical-impact statement:** no public value changes (verified:
`git diff origin/master -- rust hazma test | grep '^[-+][^-+]' | grep -v 'followups/'`
prints nothing). The only hunks are path strings in comments,
docstrings and two roster fields, plus the `pyproject.toml` version.

**Exit Criteria → artifact.**

| Criterion | Satisfied by |
| --- | --- |
| `preflight.sh --closing` green | Verification: every gate passes except one pytest case that fails on this host's memory |
| `PLAN.md` `status: Complete` | `../PLAN.md:2` |
| eight follow-ups moved, links repointed, revision pinned | `git status` shows 8 renames `todo/ → done/`; each `Status:` names the PR and its merge SHA; identifier and dangling sweeps above |
| `minor` bump, level re-checked | `pyproject.toml` 2.3.0; Findings, first bullet |
| CHANGELOG entry naming the slug with aggregated shifts | `CHANGELOG.md` `## [2.3.0] — 2026-09-24` |
| `test/parity/data` untouched | count sweep, last row |
| retrospective, §5 stubs, `projects/README.md`, `todo/` cross-check | `../learnings/project-retrospective.md`; `delta-declaration-layer-outlives-its-project.md`; the Completed row; retrospective §5 lists the five `todo/` entries this project sourced |

**Task-note self-consistency:** `**Status:** Complete` here and in
`README.md`'s Tasks row 12 and header. All seven Exit Criteria boxes are
ticked and each has a mapping row. Every file named under Files Changed
is in `git status --short`.

## Handoff to Next Task

None: this was the last task. Read `../learnings/project-retrospective.md`.
