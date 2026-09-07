# Moved follow-ups leave dangling inbound paths, and nothing checks

- **Added:** 2026-08-27
- **Source:** PR #81 review (cython-to-rust Task 6.3) — four references to
  a follow-up that had just moved to `done/`
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** none. Cheap now (two slugs); it grows by a
  handful of references every time a follow-up resolves.

## Why

`docs/workflow.md:291` is explicit: the path encodes status, so resolving
an item changes it, and before committing you must
`rg -l '<slug>\.md'` and update **every** reference. Nothing enforces
that. The rule is prose in a workflow doc, and the one automated link
checker the repo has — `scripts/agents/check_doc_citations.py` — only
bounds-checks `file:line` citations into *tracked source files*. A prose
path into `docs/followups/` is invisible to it.

The result is that the rule is followed unevenly. Two of the four
follow-ups moved so far have zero stale references; two do not. Swept on
2026-08-27, after PR #81 repaired its own:

```text
$ for p in $(rg -oN --no-filename 'docs/followups/(todo|done)/[a-z0-9-]+\.md' \
      projects/ docs/ hazma/ test/ rust/ README.md CHANGELOG.md | sort -u); do
    [ -f "$p" ] || echo "DANGLING: $p"
  done
DANGLING: docs/followups/todo/cross-section-prefactor-threshold-cancellation.md
DANGLING: docs/followups/todo/legacy-parameters-width-exponent-bug.md
```

Re-run on 2026-08-27 during cython-to-rust Task 7.1, which moved two more
follow-ups and repointed all 23 of their inbound references by hand: the
same two, plus a third that had accrued in the meantime —
`docs/followups/todo/oracle-restore-revisions-for-the-mediator-decay-pyx.md`,
moved to `done/` by Task 6.4. That is the growth rate this file predicted,
observed.

All three slugs live under `done/` today. The five references to the first
two are in Phase 00 task notes:

- `projects/cython-to-rust/task-notes/phase-00/task-0.1-relocate-constants.md`
  — three, one a prose record and two inside pasted command output.
- `projects/cython-to-rust/task-notes/phase-00/task-0.3-delete-superseded.md`
  — the remainder.

This matters more than a broken link usually would, because these
documents are the project's memory. An agent picking up later work
greps a slug, gets nothing, and concludes the follow-up was deleted —
when `docs/followups/README.md` says items "are never deleted", only
moved. PR #81's own first pass made exactly that error in reverse: it
read the surviving `legacy-parameters` references as an intentional
convention ("transcripts are frozen evidence") and skipped four repoints
on that basis. A rule with visible counter-examples and no enforcement
teaches the wrong thing.

## What

Two pieces, in either order.

**1. Sweep the two remaining slugs.** Repoint the five references in the
two Phase 00 task notes to `done/`. Where the reference sits inside
pasted command output, keep the transcript honest the way PR #81 did:
update the path and add a bracketed note recording what the command saw
when it ran. Verify with the loop above returning nothing.

**2. Make the gate real**, so this does not recur. The cheapest form is a
`preflight.sh` gate — it already has a markdownlint step and a
forbidden-token step, so a third doc gate fits the existing shape:

```bash
rg -oN --no-filename 'docs/followups/(todo|done)/[a-z0-9-]+\.md' \
   projects/ docs/ hazma/ test/ rust/ README.md CHANGELOG.md \
  | sort -u | while read -r p; do [ -f "$p" ] || echo "DANGLING: $p"; done
```

Non-empty output is a failure. Scope it to the whole repo rather than to
`--paths`: the reference that goes stale is in a file the moving PR does
not touch, which is exactly why `--paths`-scoped checking missed it in
PR #81.

The ordering constraint this file used to carry — do step 1 first,
because a gate that is red on trunk compounds the standing redness of
gate 3 — no longer applies. `ruff check` stopped being red on trunk when
[`preflight-isort-ruff-red-on-trunk`](../done/preflight-isort-ruff-red-on-trunk.md)
was resolved by scoping gates 2 and 3 to the diff. A new gate is still
better introduced green, though, and step 1 is what makes this one green:
the sweep above reports three dangling paths on an untouched trunk, the
same three that section names.

**Validate the command before trusting it as a gate.** Its result set
proved sensitive to which paths it is given. Measured on 2026-09-06 with
ripgrep 15.1.0, searching for the same pattern:

```text
docs/                    11 unique paths
docs/ scripts/            1
docs/ scripts/ (after `rm -rf scripts/agents/__pycache__`)   11
docs/ scripts/ --no-ignore                                   11
```

A gitignored directory under one searched path suppressed matches from
another, and the full pytest suite creates exactly such a directory —
`test/agents/test_resolve_phase.py` loads its helper by path, leaving
`scripts/agents/__pycache__/` behind. A sweep that silently returns fewer
hits is the false green this gate exists to prevent, so pin the count
with a fixture containing a known dangling path rather than asserting
that empty output means clean. The mechanism was not established; only
the behavior above was measured.

An alternative is to fold it into `check_doc_citations.py`, which already
walks docs and resolves paths. That is a better home if the checker also
grows the deleted-in-repo-file handling
`citation-checker-skips-deleted-inrepo-files.md` asks for — the two share
the "this path used to exist" problem.

## Entry points

- `docs/workflow.md:291` — the rule that is not enforced
- `docs/followups/_template.md:8-10` — the same rule, in the template
- `docs/followups/README.md:11` — the lifecycle pointer
- `scripts/agents/preflight.sh` — where a gate would go
- `scripts/agents/check_doc_citations.py` — the alternative home
- `projects/cython-to-rust/task-notes/phase-00/task-0.1-relocate-constants.md`,
  `.../task-0.3-delete-superseded.md` — the five stale references
- Related follow-up: `docs/followups/todo/citation-checker-skips-deleted-inrepo-files.md`
- Related: `projects/cython-to-rust/task-notes/phase-06/task-6.3-positron-spectra.md`
  — PR #81's own sweep, and the wrong turn that motivated this
