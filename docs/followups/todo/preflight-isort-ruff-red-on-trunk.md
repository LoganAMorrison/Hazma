# `preflight.sh` gates 2 and 3 are red on the trunk, so every PR inherits a FAIL

- **Added:** 2026-08-05
- **Source:** `projects/cython-to-rust/` Phase 00 Task 0.5 — a
  docstring-only change to `hazma/spectra/_photon/__init__.py` returned
  `RESULT: FAIL` from an otherwise clean preflight run
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** ripens now; nothing blocks it. It gets more
  expensive the longer it waits, because every task in flight has to
  re-derive the same "is this mine?" analysis.

## Why

`scripts/agents/preflight.sh` is the repo's one-command commit gate, and
[`docs/agents/preflight.md`](../../agents/preflight.md) states that a
non-zero exit is a blocked handoff. Today it cannot return zero for most
touched files, because two of its gates fail on unmodified trunk code:

- **Gate 2, `isort --check-only`.** Import blocks across the package are
  not isort-sorted. Concretely, `hazma/spectra/_photon/__init__.py:12-21`
  lists `_muon, _pion, _rho, _kaon, _eta, _omega, _eta_prime, _phi` in
  physics order, not alphabetical order.
- **Gate 3, `ruff check`** with the configured `[tool.ruff]` rule set.
  The same file yields 17 findings (missing docstring periods, unused
  `typing.List` / `warnings.warn` imports, missing annotations) with no
  edit applied at all. The project's working memory already records the
  package-wide figure: 6844 findings on the trunk.

Neither is caught by CI, which runs only `black --check --diff hazma
test` plus `ruff check --isolated --select E9,F63,F7,F82` — a
deliberately narrow gate (see the comment in
`.github/workflows/ci.yml`). So the repo is simultaneously green in CI
and red in the gate agents are told to trust.

The cost is not the lint debt itself; it is that a red gate carries no
signal. Every task now has to prove its own red rows are pre-existing
(the Task 0.5 note does this by `git stash`-ing its change and re-running
both commands), and a *real* regression introduced next to that noise is
easy to wave through as "same as trunk".

## What

Pick one of three, and record which in this file:

1. **Clean the debt.** Run `isort hazma test`, then work the configured
   ruff findings down to zero — mechanically where `--fix` applies, by
   hand for the annotation and docstring rules. Verify no public value
   moves (removing a genuinely unused import is safe; removing one that
   a module re-exports is not — `hazma/spectra/_photon/__init__.py` and
   the other package `__init__.py` files need checking against
   `__all__` and against what `hazma.spectra` re-exports). Then keep it
   clean by adding both gates to CI so the debt cannot re-accumulate.
2. **Narrow the gate to the diff.** Make gates 2 and 3 compare against
   the merge base rather than assert absolute cleanliness, so a PR fails
   only on findings it introduced. This is the smallest change that
   restores signal, and it is what the gate's own `--paths` scoping was
   reaching for.
3. **Relax the configured rule set** in `pyproject.toml`'s `[tool.ruff]`
   to what the repo actually intends to enforce, and fix the remainder.
   Worth considering if the current selection was aspirational rather
   than chosen.

Option 2 is the cheapest and is probably the right first move; option 1
is the durable end state.

**Add a fourth thing regardless of which is chosen: a way to say "this
diff has no Python."** `preflight.sh:81` defaults `PATHS` to
`hazma test` when `--paths` is empty, so a *docs-only* run silently
widens from "nothing to check" to "check the two reddest paths in the
repo" and reports `FAIL`. There is no flag that means no-Python — the
only workaround is to pass some unrelated-but-clean file, which is
exactly the kind of gaming that erodes a gate's meaning. cython-to-rust
Task 0.4 hit this on a two-file markdown commit (the diff touched
nothing under `hazma/` or `test/` at all, making the red rows provably
pure trunk state) and had to re-run scoped to `setup.py` to get an
honest green. An explicit `--no-python`, or treating an empty `--paths`
as "skip gates 1–3" rather than as a wildcard, would remove the
temptation.

## Entry points

- `scripts/agents/preflight.sh` — gates 2 and 3.
- [`docs/agents/preflight.md`](../../agents/preflight.md) — the
  "non-zero exit is a blocked handoff" rule this contradicts.
- `.github/workflows/ci.yml:32-45` — what CI actually enforces.
- `pyproject.toml` `[tool.ruff]` (the configured rule set) and
  `[dependency-groups]` `lint` (the pins CI installs).
- `hazma/spectra/_photon/__init__.py:12-21` — the isort exemplar.
- `projects/cython-to-rust/task-notes/README.md` §Findings — the
  standing "ruff is red on the trunk and does not block CI" note.
- `projects/cython-to-rust/task-notes/phase-00/task-0.5-gamma-ray-decision.md`
  §"Preflight disposition" — a worked example of the per-task cost.
- `scripts/agents/preflight.sh:81` — the `PATHS="hazma test"` default
  that turns a docs-only run into a trunk-wide lint run.

## Risks / open questions

- **Reformatting churn.** Option 1 touches many files and will conflict
  with anything in flight; land it on a quiet trunk, in its own PR, with
  no behavior change mixed in.
- **Unused-import removal is not always safe** in a package
  `__init__.py`, where an import can be a deliberate re-export. Check
  each `F401` against `__all__` and against `hazma/spectra/__init__.py`
  before deleting.
- **This is formatting, not physics** — `docs/versioning.md` is
  unaffected and no published number moves, so whichever option is taken
  is a `patch`-level change on its own.
