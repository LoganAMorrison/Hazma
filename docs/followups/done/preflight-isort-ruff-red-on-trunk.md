# `preflight.sh` gates 2 and 3 are red on the trunk, so every PR inherits a FAIL

- **Added:** 2026-08-05
- **Source:** `projects/cython-to-rust/` Phase 00 Task 0.5 — a
  docstring-only change to `hazma/spectra/_photon/__init__.py` returned
  `RESULT: FAIL` from an otherwise clean preflight run
- **Scope:** cross-cutting
- **Status:** done
- **Resolved:** 2026-09-06 by **option 2** (narrow the gate to the diff).
  The lint debt itself is untouched and still real; what changed is that
  gates 2 and 3 no longer charge it to the branch that runs them. See
  **Resolution** below, including which of this file's citations had gone
  stale by the time it was picked up.

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

## Resolution

**Option 2, via [`scripts/agents/lint_delta.py`](../../../scripts/agents/lint_delta.py).**
Gates 2 and 3 now run their linter twice — over the working tree, and
over the same paths as they stand at the merge base — and report only the
difference. Each row states the split it measured, so a green gate is
evidence rather than an argument:

```text
PASS   isort --check-only      0 new (72 pre-existing at 3bbd0a0bddf4)
PASS   ruff check              0 new (6091 pre-existing at 3bbd0a0bddf4)
```

Findings are compared as a multiset of `(path, rule, message)` with line
and column dropped, so an edit near the top of a file does not re-report
everything below it; the price is that removing one finding and adding an
identical one in the same file cancels out. `isort` is compared per file,
because `--check-only` names the file rather than the offending import.
Gate 1 stays absolute: black is green on the trunk and CI enforces it
that way.

**The fourth item needed no flag.** A diff that touches no Python leaves
the two trees identical over `--paths`, so the comparison is empty and
the rows PASS on their own — reported as `no Python changed against
<sha>`, in 0.4 s, without the archive step. Adding `--no-python` would
have been a second way to say what the measurement already says, and
treating an empty `--paths` as "skip gates 1–3" was rejected outright: a
forgotten `--paths` would then silently skip the Python gates, trading a
false red for a false green.

**Why not option 1.** Measured at `3bbd0a0b` under ruff 0.16.6: 6091
findings across 70 rule codes, of which only 715 are `--fix`-able. The
top four are `ANN001` (2364), `ANN202` (837), `D205` (445) and `ANN201`
(417) — annotating roughly 3600 function signatures and rewriting several
hundred docstrings across a physics library. That is a project with its
own numerical-review burden, not a task, and it would conflict with
everything in flight. It remains the durable end state, and option 2 is
what makes it optional rather than urgent: the debt no longer costs
anything per-PR, so it can be paid down file by file, each drop credited
by the `N fixed` counter.

**CI was left alone.** Option 1 paired its cleanup with "add both gates
to CI so the debt cannot re-accumulate", and a diff-scoped gate makes
that possible for the first time — but not free. `actions/checkout@v7`
fetches depth 1, so `git merge-base` has no trunk to resolve and the
comparison would fail closed on every run; enforcing this in CI means
`fetch-depth: 0` and a base ref that differs between `push` and
`pull_request`. That is its own change with its own failure modes, and
adding a required check is a contribution-contract decision rather than a
gate repair. `preflight.sh` remains the enforcement point, as
[`preflight.md`](../../agents/preflight.md) already says.

Option 3 was not taken and stays available. Three things noticed while
measuring, none folded in:

- `[tool.ruff]` does not exclude `hazma/experimental/` or `notebooks/`,
  though [`AGENTS.md`](../../../AGENTS.md) says both are outside the lint
  gate; CI passes `--exclude` for them on its own, `--isolated` run. 305
  of the 6091 and 8 of the 72 sit in `hazma/experimental/`. Harmless
  under a diff-scoped gate, and a slice of option 3 rather than of this.
- The `[tool.ruff]` keys are the deprecated top-level spelling, which
  makes ruff print a migration warning on every gate run — carved out to
  [`ruff-config-uses-deprecated-top-level-keys.md`](../todo/ruff-config-uses-deprecated-top-level-keys.md).
- Gate 1 is still absolute, and `scripts/` is formatted by nothing, so
  widening `--paths` to cover a helper you edited fails black on three
  files you did not touch — the same inherited red, in the one gate that
  did not need diffing. Carved out to
  [`scripts-are-outside-the-format-and-lint-gates.md`](../todo/scripts-are-outside-the-format-and-lint-gates.md).

### Citations that had gone stale

This file was filed on 2026-08-05 and picked up on 2026-09-06, and the
cython-to-rust migration rewrote its exemplar in between. Both of these
were checked against the tree before anything was built on them:

- `hazma/spectra/_photon/__init__.py:12-21`, cited as the isort exemplar
  and as yielding 17 ruff findings, is now clean on both counts — the
  module is a thin `hazma._core` wrapper with two imports, and
  `ruff check` on it reports `All checks passed!`.
- `preflight.sh:81`, cited as the `PATHS="hazma test"` default, is now
  line 93.

The *condition* the file describes survived intact: `isort --check-only
hazma test` still reports 72 ERROR lines and `ruff check hazma test`
still reports 6091 errors on an untouched trunk.
