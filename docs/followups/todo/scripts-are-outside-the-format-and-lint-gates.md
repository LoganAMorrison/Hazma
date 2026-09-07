# Nothing formats or lints `scripts/`, including the gate's own code

- **Added:** 2026-09-06
- **Source:** surfaced while resolving
  [`preflight-isort-ruff-red-on-trunk`](../done/preflight-isort-ruff-red-on-trunk.md),
  which added a Python file under `scripts/agents/` and found the
  directory unformatted
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** none, and it is cheapest now — three files,
  no behavior change. It grows with every helper added under `scripts/`.

## Why

Both automated gates are scoped to `hazma test` and neither reaches
`scripts/`:

- CI's formatting step is `black --check --diff hazma test`.
- `preflight.sh` defaults `--paths` to `hazma test`, so gates 1–3 do too.

CI's *narrow* ruff step is the one exception — it runs over `.` — but it
selects only `E9,F63,F7,F82`, so it catches syntax errors and undefined
names and nothing about formatting or the configured rule set.

The result is that `scripts/` has drifted. Measured at `3bbd0a0b`:

```text
$ black --check scripts/agents
would reformat scripts/agents/check_pr_title.py
would reformat scripts/agents/check_doc_citations.py
would reformat scripts/agents/resolve_task.py
3 files would be reformatted, 1 file would be left unchanged.

$ ruff check scripts/agents
Found 9 errors.
```

This matters more than an unformatted helper usually would, because the
code under `scripts/agents/` is what the repo's gates are made of —
`preflight.sh` is the commit gate, `lint_delta.py` decides whether gates
2 and 3 pass, and `resolve_task.py` and `resolve_phase.py` tell agents
which task to work on. The one tree nothing checks is the tree that does
the checking. It is also a quiet trap for anyone widening `--paths` to
cover a script they edited: gate 1 then fails on three files they never
touched, which is the same "inherited red" problem the follow-up above
was filed to remove, in the one gate that is still absolute.

## What

Bring `scripts/` into both gates, in this order so no step lands red:

1. `black scripts` — three files, formatting only. Confirm no behavior
   change: `python -m py_compile` on each, then
   `pytest test/agents` for the helpers that have tests.
2. Add `scripts` to CI's black step (`black --check --diff hazma test
   scripts`) and to `preflight.sh`'s `PATHS` default.
3. Decide what the configured ruff rule set should mean for a `scripts/`
   helper before turning it on there. The nine findings are worth reading
   first: helper scripts are not library code, and some of the rule set
   (`D`, `ANN` on private functions) may not earn its keep. Gate 3 is
   diff-scoped, so this step is not urgent — a new script is already held
   to zero new findings the moment it is passed to `--paths`.

## Entry points

- `.github/workflows/ci.yml` — the `black --check --diff hazma test` step
- `scripts/agents/preflight.sh` — the `PATHS="hazma test"` default
- `scripts/agents/check_pr_title.py`,
  `scripts/agents/check_doc_citations.py`,
  `scripts/agents/resolve_task.py` — the three unformatted files
- [`docs/agents/preflight.md`](../../agents/preflight.md) — gate 1, which
  records this exception
- Related: [`ruff-config-uses-deprecated-top-level-keys.md`](ruff-config-uses-deprecated-top-level-keys.md),
  the other `[tool.ruff]` change worth batching with step 3

## Risks / open questions

- **Reformatting churn conflicts with anything in flight** under
  `scripts/`. Three files, so the window is small; land step 1 on its
  own.
- **`test/agents/` covers only two of the five helpers**
  (`resolve_phase.py` and `lint_delta.py`), so step 1's "no behavior
  change" rests on black's own guarantee for the other three rather than
  on tests. That is the usual bargain for a formatter, but worth stating
  since `check_pr_title.py` is invoked by hand rather than by any gate.
