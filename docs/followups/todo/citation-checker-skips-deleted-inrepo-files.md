# Citation checker skips citations to deleted in-repo files

- **Added:** 2026-08-05
- **Source:** PR #42 review (stale `boost.pyx` citation)
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** none — but it ripens as the cython-to-rust
  purge deletes more code, since every deletion grows the skipped list.

## Why

`scripts/agents/check_doc_citations.py` bounds-checks a citation only
when it can resolve the cited path to a tracked file. When no candidate
exists at all it reports EXTERNAL and **skips** — correct for a
`numpy/...` or `scipy/...` citation, but it means a citation into a
file the repo *used to have* is indistinguishable from one into a
third-party library.

That is the gap the PR #42 review exposed from the other side. The
citation to ``boost.pyx line 427 as of `e94fb21^` `` failed loudly
because `hazma/_utils/boost.pyx` still exists and merely got shorter.
Had the purge deleted the file outright, the same rotten citation would
have been skipped in silence.

The dead-code purge makes this concrete: as of PR #42 the checker skips
**25** citations, and the majority are not third-party at all — they
are `hazma/_decay/*.pyx`, `hazma/_positron/*.pyx`,
`hazma/_neutrino/*.pyx`, and `hazma/spectra/_positron/_kaon.pyx`,
every one of them a path the purge removed. The skip list is printed,
so the information is not hidden, but it is 25 lines of noise in which
a genuinely wrong citation would not stand out.

## What

Teach the checker to distinguish "never in this repo" from "deleted
from this repo". `git log --diff-filter=D --name-only` over the cited
path, or a check against the merge-base tree, separates the two
cheaply. Then report deleted-path citations as their own category —
at minimum a distinct WARN line, ideally a failure unless the citation
is pinned to a commit (the convention PR #42 adopted:
``lines 427, 447, 456 as of `e94fb21^` ``).

Decide the policy alongside it: a historical analysis doc *should* be
allowed to cite code that no longer exists, so the fix is probably
"require a commit pin", not "forbid the citation".

## Entry points

- `scripts/agents/check_doc_citations.py` (resolution order and the
  EXTERNAL branch)
- `docs/agents/doc-consistency.md` (check 1, the citation sweep)
- `projects/cython-to-rust/references/cython-inventory.md` (the
  densest concentration of purge-deleted citations)

## Risks / open questions

Failing on every deleted-path citation would immediately redden the
cython-to-rust reference docs, which legitimately describe pre-purge
code. Pair the stricter check with the commit-pin convention, or land
the pins first — otherwise the gate gets disabled rather than
satisfied.
