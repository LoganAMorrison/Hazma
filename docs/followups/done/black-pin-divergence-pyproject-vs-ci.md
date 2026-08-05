# `black` pin diverges between `pyproject.toml` and CI

- **Added:** 2026-08-04
- **Source:** cython-to-rust Task 0.3 (PR #37 — CI Lint went red on a
  locally-black-clean tree)
- **Scope:** cross-cutting
- **Status:** done — resolved 2026-08-04 by
  [PR #40](https://github.com/LoganAMorrison/Hazma/pull/40) (direction 2,
  see [Resolution](#resolution)).
- **Triggers / blockers:** none — but it silently breaks contributors
  today, so it ripens now.

## Why

The two places that pin `black` disagree:

<!-- markdownlint-disable MD013 -- pin strings -->
| Location | Pin |
| --- | --- |
| `pyproject.toml:31` (the `dev` extra) | `black>=23.3,<27.0` |
| `.github/workflows/ci.yml:24` (Lint job) | `black>=23.3,<25.0` |
<!-- markdownlint-enable MD013 -->

`pyproject.toml` was widened by
[PR #27](https://github.com/LoganAMorrison/Hazma/pull/27) — "update black
requirement from <25.0,>=23.3 to >=23.3,<27.0", merged 2026-08-04 —
and `ci.yml` was not updated to match.

So `pip install -e '.[dev]'` — the documented dev setup — installs
black 26, while CI checks with black 24. Black's style changed between
those majors (notably the "hug" treatment of a sole multiline string
argument), so **the documented dev environment reformats files into a
style CI rejects.** The failure mode is nasty because it is invisible
locally: `black --check` passes on your machine and Lint fails on the
PR, on lines you did not intend to touch.

Measured on `origin/master` at `cd0be2b`, same tree, same command:

| black | `black --check hazma test` |
| --- | --- |
| 24.10.0 (CI's) | `249 files would be left unchanged` — clean |
| 26.5.1 (what `[dev]` installs) | 34 files would be reformatted |

This also means the "preflight's Python gates are red on `origin/master`
itself" note recorded in cython-to-rust Task 0.1 was measuring an
unpinned newer black, not a real property of the repo. CI Lint is and was
green on the trunk. That claim has been corrected in
`projects/cython-to-rust/task-notes/README.md`.

## What

Pick one pin and make both places use it. Two directions, and the choice
is a maintainer call:

1. **Narrow `pyproject.toml` to `<25.0`** — matches CI today, zero
   reformatting, effectively reverts PR #27's widening. Cheapest.
2. **Widen CI to `<27.0` and reformat** — moves to modern black, but
   requires a repo-wide `black hazma test` commit (34 files) that will
   collide with any open branch. If taken, do it as its own PR with no
   other content.

Whichever is chosen, remove the duplication so they cannot drift again:
have the Lint job install the dev extra (`pip install -e '.[dev]'`)
rather than repeating a literal pin, or add a `.github/dependabot.yml`
rule that updates the workflow pin alongside the pyproject one.

## Entry points

- `pyproject.toml:31` — the `dev` optional-dependency pin.
- `.github/workflows/ci.yml:24` — the Lint job's literal pin.
- `scripts/agents/preflight.sh` — runs whatever `black` is on `PATH`, so
  it inherits the ambiguity; it is the gate agents trust.
- `docs/agents/environment.md` — "Build and imports" now records the
  trap.
- Prior art: [PR #27](https://github.com/LoganAMorrison/Hazma/pull/27)
  (the widening), [PR #37](https://github.com/LoganAMorrison/Hazma/pull/37)
  (where it first bit).

## Risks / open questions

- Direction 2 produces a large formatting-only diff. Land it when no
  long-lived branch is open, or it becomes a merge-conflict generator —
  `projects/cython-to-rust/` has several phases still to go.
- `isort` and `ruff` are installed unpinned in the Lint job's `pip
  install "black>=23.3,<25.0" ruff` line; `ruff` in particular is
  fast-moving. The same class of drift applies to it, and CI's ruff step
  is `--isolated --select E9,F63,F7,F82`, so it is much less exposed —
  worth confirming rather than assuming while fixing this.

## Resolution

**Direction 2**, maintainer's call: CI moves to modern black and the
repo is reformatted to match.

The pins now live in exactly one place —
`pyproject.toml`'s PEP 735 `[dependency-groups]`:

<!-- markdownlint-disable MD013 -- pin strings -->
```toml
[dependency-groups]
lint = ["black>=23.3,<27.0", "isort>=5.12,<9.0", "ruff>=0.1,<1.0"]
dev = [{ include-group = "lint" }, "pytest>=7.0"]
```
<!-- markdownlint-enable MD013 -->

and `.github/workflows/ci.yml`'s Lint job installs that group
(`python -m pip install --group lint`) instead of repeating a literal
version. A group, not the `dev` extra: `--group` installs only those
packages, so the Lint job still does not build the Cython extensions —
`pip install -e '.[dev]'` would have made a formatting check compile 32
extension modules. The old `[project.optional-dependencies] dev` extra
is gone; the documented dev setup is now `pip install -e . --group dev`
(pip >= 25.1), which also brings `pytest` for the preflight gate.

`black hazma test` under black 26.5.1 then reformatted **33 files, +59
/ −109 lines** — all formatting: `(a, b) = f()` → `a, b = f()`,
`# type:ignore` → `# type: ignore`, one-line docstring collapse, blank
lines after imports, and the "hug" of a sole multiline string argument
that started this. Verified against the same tree, same commands:

<!-- markdownlint-disable MD013 -- measurement table -->
| Gate | Before | After |
| --- | --- | --- |
| `black --check hazma test` (26.5.1) | 33 reformat / 191 clean | **224 clean** |
| `ruff check --isolated --select E9,F63,F7,F82` (CI's form) | passes | passes |
| `isort --check-only hazma test` | 141 error lines | 134 |
| `ruff check hazma test` (configured) | 6619 | 6611 |
| `pytest` | 68 passed / 20 skipped | 68 passed / 20 skipped |
<!-- markdownlint-enable MD013 -->

isort and configured-ruff both improved because black collapsed
constructs they were flagging; neither is a gate CI runs.

Two things deliberately **not** done:

- No `.github/dependabot.yml` `ignore` rule. Dependabot's unrestricted
  `pip` rule is what widened the pyproject pin in PR #27, but with the
  workflow no longer carrying a second pin there is nothing left to
  desync — a future widening PR just moves the one pin, and its CI run
  is an honest test of whether the new major reformats the repo.
  Whether Dependabot reads PEP 735 groups at all is untested here; if it
  stops proposing black bumps, that is a downgrade in nagging, not in
  correctness.
- No `CHANGELOG.md` entry. Nothing on the public Python API moved —
  this is formatting and dev tooling only.
