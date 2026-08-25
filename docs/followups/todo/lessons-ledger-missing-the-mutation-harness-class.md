# `[mutation-harness-poisons-its-own-baseline]` is cited but not in the ledger

- **Added:** 2026-08-23
- **Source:** cython-to-rust Task 6.2 (`projects/cython-to-rust/task-notes/phase-06/task-6.2-decay-spectra.md`)
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** none — the fix is one ledger entry plus one
  examples section, and the citation it needs is recoverable from git
  history.

## Why

Nine places across the tree cite
`[mutation-harness-poisons-its-own-baseline]` as a class in
`docs/agents/lessons.md`, two of them saying so in those words:

```text
test/parity/oracles/README.md:51        `docs/agents/lessons.md` `[mutation-harness-poisons-its-own-baseline]` is
test/parity/oracles/capture.py:373      `[mutation-harness-poisons-its-own-baseline]` guard: a harness
projects/cython-to-rust/task-notes/phase-04/README.md:299
projects/cython-to-rust/task-notes/history-findings.md:815
projects/parity-pinned-defect-repair/task-notes/README.md:119
projects/parity-pinned-defect-repair/references/corpus-repinning.md:110
projects/parity-pinned-defect-repair/task-notes/task-2-cython-oracles.md:46,303,512
```

The class is in **neither** `docs/agents/lessons.md` nor
`docs/agents/lessons-examples.md`. Every one of those citations is
dangling, so a reader sent to the ledger to find out what the class says
finds nothing, and the guard `capture.py` names its rationale after has
no stated rule behind it.

`projects/cython-to-rust/task-notes/phase-04/README.md:299` attributes
the class to **Task 3.3**, so the lesson was learned and named; only the
ledger entry is missing. cython-to-rust Task 6.2 hit the same class a
third time — a mutation campaign whose `uv pip install -e .` did not
force a rebuild measured a stale `hazma/_core.abi3.so` and reported its
results lagging the mutations by two iterations — and could not cite the
class it was an instance of.

## What

Recover the citation and add the entry. `projects/cython-to-rust/`'s
Task 3.3 note names the PR; confirm with

```bash
git log --oneline --all -- projects/cython-to-rust/task-notes/phase-03/task-3.3-quadpack.md
```

Then add one `- [mutation-harness-poisons-its-own-baseline] ...` line to
the ledger and its section to `lessons-examples.md`, following the format
`docs/agents/lessons.md` §Format fixes. The rule the three instances
share: **a harness that mutates a source and re-measures must force the
rebuild and prove it happened** — delete the built artifact, reinstall,
and assert the artifact is back — because a build system that decides
nothing changed will hand the harness the previous iteration's binary and
the campaign will report another mutation's result as this one's.

Note `docs/followups/todo/lessons-ledger-over-its-working-set-cap.md`
first: the ledger is already past its stated cap at 40 entries, so this
may want to land as part of that consolidation rather than as a 41st
line.

## Entry points

- `docs/agents/lessons.md` — the `## Ledger` section
- `docs/agents/lessons-examples.md`
- `test/parity/oracles/capture.py:373` — the guard named after the class
- Prerequisite follow-up: `docs/followups/todo/lessons-ledger-over-its-working-set-cap.md`
