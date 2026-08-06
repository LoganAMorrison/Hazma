# Preflight gate

The canonical gate to run **before every commit and before every PR**.
Every implementing and review-response skill points here instead of
restating it. The one-command form is
[`scripts/agents/preflight.sh`](../../scripts/agents/preflight.sh); this
document is the rationale and the manual fallback.

## The gate

Run these from the worktree root, in order. Each must pass on its own
before you stage anything.

1. **`black --check <paths>`** — the repo is black-formatted at 88
   columns. If it fails, run `black <paths>` and re-run the check.
   (`black` alone is not the gate; the `--check` is.)
2. **`isort --check-only <paths>`** — black profile, configured in
   `pyproject.toml`. Same rule: fix with `isort <paths>`, then re-check.
3. **`ruff check <paths>`** — the configured rule set lives under
   `[tool.ruff]` in `pyproject.toml`. `hazma/experimental/` and
   `notebooks/` are outside the gate.
4. **`pytest <targets>`** — **read the summary line.** `pytest` exits **5**
   when it collects zero tests, and a wrapper that only checks "non-zero
   exit" will happily treat a typo'd path as a failure while a
   `-k` filter matching nothing exits 0 with `no tests ran`. Zero
   collected means the gate FAILED, not passed. Name the real targets
   (`pytest test/spectra`), not a filter you have not verified selects
   something.
5. **`python -c "import hazma"`** — the import smoke. Hazma ships Cython
   extensions; a `.pyx` edit that was never rebuilt, or a rebuild against
   a different interpreter, produces a tree that lints and formats
   cleanly and fails at import. Run this after any change under
   `_utils/`, `_gamma_ray/`, `_phase_space/`, or the `.pyx` in
   `spectra/`, `scalar_mediator/`, `vector_mediator/`, and after any
   `_build.py` change.
6. **`markdownlint --dot <changed .md files>`** — when curated docs
   changed. Word-diff after any `--fix`: it can corrupt code spans.
   Run it from the repo root: the committed
   [`.markdownlint.jsonc`](../../.markdownlint.jsonc) is discovered
   relative to the **current directory**, not to the linted file, so
   the same file lints differently from a subdirectory. See
   [Markdown rules](#markdown-rules) for what the config relaxes —
   reach for an inline `<!-- markdownlint-disable -->` only after
   confirming the config does not already cover your case. Arguments
   are globs: one that matches nothing prints the usage banner and
   exits **0**, so a typo'd path lints nothing and looks green.
   `preflight.sh` checks the `--md` paths exist; a bare `markdownlint`
   run does not.
7. **Version-bump check** — only when the diff flips a
   `projects/<slug>/PLAN.md` `status:` to `Complete`.
   `scripts/agents/preflight.sh --closing` verifies that `VERSION` in
   `hazma/__init__.py` actually moved relative to the trunk and that
   `CHANGELOG.md` carries a matching `## [X.Y.Z]` section. See
   [`../versioning.md`](../versioning.md).
8. **Forbidden-token scan** over the diff: `breakpoint()`,
   `pdb.set_trace()`, `import pdb`, and stray `print()` added to library
   code. Resolve or justify each hit. `git diff origin/master -- '*.py'`
   is the surface.

## Exit-code safety

Never pipe a gate through `head`, `tail`, or `grep` — a pipeline reports
the exit status of the **last** command, which masks the gate's own
failure. Run each gate bare and read its status (and, for pytest, its
summary line). If you must filter output, capture the exit code
separately.

## Markdown rules

Gate 6 runs against the committed
[`.markdownlint.jsonc`](../../.markdownlint.jsonc) at the repo root.
Everything not listed there is at its markdownlint default, including
**MD013's 80-column limit on prose** — the config buys tables and code
blocks room, not paragraphs. What it relaxes, and why:

| Rule | Setting | Because |
| --- | --- | --- |
| MD013 line-length | `tables: false`, `code_blocks: false` | Grounded-fact tables are as wide as their content; rewrapping pasted commands or pasted output falsifies the record. |
| MD025 single-title | `front_matter_title: ""` | A phase file carries `title:` in frontmatter *and* an H1 in the body — that is the schema. |
| MD041 first-line-heading | `name:` accepted as the title | `SKILL.md` declares its title as frontmatter `name:` and opens with prose. |
| MD033 no-inline-html | off | `<angle bracket>` placeholders are the repo's fill-in notation; markdownlint cannot tell one from an HTML tag. |
| MD049 emphasis-style | off | `*emphasis*` inline and `_…_` for standing "nothing here yet" markers are two notations with two meanings. |
| MD060 table-column-style | off | A template's placeholder cells change width once a project fills them in, so an aligned template is a misaligned copy. |

One trap the config cannot cover: `tables: false` and
`code_blocks: false` exempt **parsed** tables and fences, and nothing
inside an HTML comment is parsed. A wide table commented out as a
template alternative therefore trips MD013 anyway. Fence it as a
```` ```markdown ```` block instead of commenting it out — that is why
the phased block in
[`task-notes/README.md`](../../projects/_template/task-notes/README.md)
is a fence. Do not reach for a pragma there: a pragma in
`projects/_template/` is copied into every project that follows.

markdownlint is not pinned by this repo and its rule set grows between
releases (MD060 is a recent addition). If a rule fires that nobody else
sees, compare `markdownlint --version` before assuming the doc is wrong.

## Sequential critical path

The path from a finished edit to a landed commit is strictly sequential:

```text
edit → rebuild (if Cython) → run gates → read results
     → stage → commit → push → verify
```

**Never batch these steps into one parallel tool block.** A failure
partway through a batch lands a half-done commit — the gates never ran,
or ran against the wrong tree. Gate each step on the previous one's
result. After pushing, verify the push landed: `git rev-parse HEAD` must
equal `git rev-parse origin/<branch>`.

## Branch and worktree assertion

Immediately before `git commit`, confirm you are where you think you are:

- `git rev-parse --abbrev-ref HEAD` is the intended branch — **never
  `master`.** Direct commits to `master` are forbidden (see
  [`AGENTS.md`](../../AGENTS.md)). Note the trunk here is `master`, not
  `main`; an assertion written against `main` silently passes on a
  `master` checkout and protects nothing.
- `git rev-parse --show-toplevel` is the intended worktree, not the main
  checkout.

The Bash tool's cwd can reset between calls, so a bare `git commit` may
run against the wrong tree. Prefer `git -C <worktree>` with an absolute
path for every git write.

## Do not trust hooks or CI for this list

There is no committed `.pre-commit-config.yaml` in this repo. CI
(`.github/workflows/ci.yml`) runs an import smoke test, `pytest` on
Python 3.10–3.14, `black --check --diff hazma test`, and a deliberately
narrow lint pass — `ruff check --isolated --select E9,F63,F7,F82` — whose
`--isolated` flag ignores `[tool.ruff]` in `pyproject.toml`. So CI green
means "no syntax errors, no undefined names, black-formatted, tests
pass"; it says nothing about import order or the configured lint rules.
That is a floor, not this gate. Run this gate yourself.

## One-command form

[`scripts/agents/preflight.sh`](../../scripts/agents/preflight.sh) runs
black, isort, ruff, pytest (with the zero-collection guard), the import
smoke, and optionally markdownlint, the version-bump gate, and the
forbidden-token scan:

```bash
scripts/agents/preflight.sh --paths "hazma/spectra test/spectra" --tests "test/spectra"
```

Add `--md "docs/agents/preflight.md"` when curated docs changed and
`--closing` on a project-closing PR. It prints a PASS/FAIL/WARN/SKIP row
per gate and exits non-zero on the first hard failure. A non-zero exit is
a blocked commit — fix and re-run; do not commit around a red gate.

A `WARN` row means a tool is not installed and its gate did **not** run —
it is a hole in your coverage, not a pass. Install the toolchain
(`pip install --group dev`, which pulls black, isort, ruff, and pytest at
the versions CI uses) rather than shipping on an unchecked gate. Do not
`pip install black` on its own: this script runs whatever is on `PATH`,
so a hand-picked version silently formats the tree differently from CI.
