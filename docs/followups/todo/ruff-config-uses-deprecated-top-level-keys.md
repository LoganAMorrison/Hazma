# `[tool.ruff]` uses the deprecated top-level keys, and ruff says so on every run

- **Added:** 2026-09-06
- **Source:** carved out of
  [`preflight-isort-ruff-red-on-trunk`](../done/preflight-isort-ruff-red-on-trunk.md),
  noticed while measuring the trunk's findings
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** none. It is a mechanical rename that gets
  forced whenever the `ruff<1.0` pin is raised past the release that
  drops the compatibility shim.

## Why

`pyproject.toml` declares the linter's rule set with the pre-0.2 spelling
— `select`, `ignore`, `flake8-tidy-imports` and `pydocstyle` directly
under `[tool.ruff]` rather than under `[tool.ruff.lint]`. ruff still
honors it, and prints a migration warning every time it is asked to:

```text
$ ruff check hazma test
warning: The top-level linter settings are deprecated in favour of their
counterparts in the `lint` section. Please update the following options in
`pyproject.toml`:
  - 'ignore' -> 'lint.ignore'
  - 'select' -> 'lint.select'
  - 'flake8-tidy-imports' -> 'lint.flake8-tidy-imports'
  - 'pydocstyle' -> 'lint.pydocstyle'
warning: `TCH003` has been remapped to `TC003`.
```

Two costs, both small and both growing. The warning is noise on stderr in
every preflight run and every editor invocation, which trains readers to
skim past ruff's stderr — the stream that also carries the errors that
mean ruff could not run at all. And the shim is the only thing keeping
the repo's configured rule set alive: `[dependency-groups] lint` pins
`ruff>=0.1,<1.0`, so a routine bump inside that range can drop it, at
which point ruff falls back to its default rule set and the gate
silently starts asking a different and much weaker question.

## What

Rename the four keys into `[tool.ruff.lint]` and rename `TCH003` to
`TC003` in the `ignore` list. Nothing else moves: `target-version` and
`exclude` are top-level options in the current schema and stay where they
are, and `[tool.ruff.flake8-tidy-imports.banned-api."numpy.typing"]`
becomes `[tool.ruff.lint.flake8-tidy-imports.banned-api."numpy.typing"]`.

Verify that the rule set did not change, rather than assuming the rename
was faithful — the failure mode is a silently narrower gate, which no
test would notice. `ruff check --statistics hazma test` before and after
must report the same counts per rule, and the run must be warning-free.
Under the diff-scoped gates the count is also directly checkable:
`scripts/agents/lint_delta.py --linter ruff --base origin/master hazma
test` must report `0 new` across the rename.

## Entry points

- `pyproject.toml` `[tool.ruff]` — the four keys and the `TCH003` entry
- `pyproject.toml` `[dependency-groups]` `lint` — the `ruff>=0.1,<1.0`
  pin whose upper end forces this
- `scripts/agents/lint_delta.py` — passes the config through `--config`,
  so it reads the same keys
- `.github/workflows/ci.yml` — CI's ruff step is `--isolated` and does
  not read this table, so it neither warns nor protects against the
  fallback

## Risks / open questions

- **A faithful-looking rename that narrows the rule set is the whole
  risk.** The per-rule statistics comparison above is the check that
  catches it; a bare "still 6091 errors" total would not distinguish a
  rule that stopped firing from one that started.
- Worth doing in the same change as any other `[tool.ruff]` edit — for
  instance the `hazma/experimental/` and `notebooks/` exclusions that
  `AGENTS.md` describes but the table does not declare, noted in
  [`preflight-isort-ruff-red-on-trunk`](../done/preflight-isort-ruff-red-on-trunk.md).
