# Environment and test-infra gotchas

Traps in this repo's shell, build, and test infrastructure. Each entry is
a **symptom** followed by the fix. Skim before any task that runs
commands; Reviewer E and the implementer both rely on it.

This file is seeded from what the layout and tooling make predictable.
When you hit a new trap, add it here in the same commit — that is what
keeps it worth reading.

## Shell and filesystem

**The Bash tool shell may be fish, not bash.** `VAR=$(...)` assignment,
`for` loops, and backtick substitution are mangled or rejected. Prefix
one-shot env vars as `env VAR=val cmd`; avoid shell loops — call the tool
once per command, or write a `.sh` and run it with `bash`.

**`ls` may be aliased with color escape codes.** Parsing its output in a
scripting context picks up ANSI garbage; use `command ls` (or a glob)
when the result feeds another command.

**`rg -rl` means `--replace` + `-l`, not recursive-list.** ripgrep has no
`-r`-for-recursive flag (it recurses by default). To rewrite in place use
`grep -rl PAT | xargs perl -pi -e 's/.../.../g'` and verify with
`grep -rc`.

**Bash-tool cwd resets between calls.** A `cd` lasts only for the command
it is part of; the next call starts back at the worktree root. Always use
absolute paths and `git -C <worktree>` for git writes — a bare
`git commit` can land in the wrong tree.

**Merge conflicts edited with `sed` or bulk line-deletes corrupt the
file.** Edit the marked region by hand, then confirm zero markers remain
(`grep -n '^<<<<<<<\|^=======\|^>>>>>>>'`).

## Build and imports

**Editing a `.pyx` / `.pxd` and re-running pytest tests the OLD kernel.**
Cython sources are compiled at build time by `_build.py`. Until you
rebuild (`pip install -e .`), every import resolves the previously-built
extension — so a change can look like it had no effect, or a bug can look
fixed when it isn't. Rebuild, then confirm.

**You may be importing an installed hazma, not the worktree.** A
site-packages install shadows the checkout depending on cwd and how the
env was set up. `python -c "import hazma; print(hazma.__file__)"` before
trusting any result you attribute to your edit — especially inside a git
worktree under `.claude/worktrees/` or `.codex/worktrees/`, which is a
*different directory* from the checkout the editable install points at.

**`pip install -e .` needs Cython, NumPy, and a C compiler.** A missing
toolchain surfaces as a build error deep in a generated `.c` file, not as
a clear "install Cython" message. (Every C++ extension went with
`_gamma_ray/` and `_phase_space/` in cython-to-rust Task 0.2; the tree
builds as C only.)

**Never hand-edit generated `.c` / `.cpp`.** They are cythonize output.
Edit the `.pyx` and rebuild.

## Linters

**Install the linters from the `lint` dependency group, not by hand.**
`pyproject.toml`'s `[dependency-groups]` is the only place the `black`,
`isort`, and `ruff` pins live, and CI's Lint job installs that same
group. Anything else risks a formatter that disagrees with CI:

```sh
uv pip install --group lint     # or: pip install --group lint
```

`--group dev` adds `pytest` on top, for the full preflight toolchain.
Note these are PEP 735 groups, **not** extras — the old
`pip install -e '.[dev]'` no longer resolves, and `--group` needs
pip >= 25.1.

This used to be a live trap and is worth knowing about because
`preflight.sh` invokes whatever `black` is on `PATH`: until 2026-08-04
the pin was written out twice, `black>=23.3,<27.0` in `pyproject.toml`
against `black>=23.3,<25.0` in `.github/workflows/ci.yml`. Black's style
changed across that boundary (a sole multiline string argument is
"hugged" in 26.x, exploded in 24.x), so the documented dev setup
installed a formatter whose output CI rejected — `black --check` clean
locally, Lint red on the PR, on lines nobody touched. The repo is now
formatted with black 26.x and the workflow carries no literal pin. Do
not reintroduce one.

**CI's ruff step is not the configured one.** Lint runs
`ruff check --isolated --select E9,F63,F7,F82 --exclude hazma/experimental
--exclude notebooks .` — syntax errors, undefined names, broken
comparisons and f-strings only. It deliberately ignores the much stricter
`[tool.ruff]` config in `pyproject.toml`, under which the repo carries
thousands of findings. A red `ruff check hazma test` therefore says
nothing about whether CI will pass; run the `--isolated` form to predict
CI, and judge the configured form only as a delta against the trunk.

## Tests

**pytest exit code 5 means zero tests collected.** It is not a pass. A
mistyped path, a `-k` filter that matches nothing, or a `collect_ignore`
entry silently reduces the run to nothing. Read the summary line
(`N passed`), not just the exit status.

**`test/conftest.py` no longer ignores any test module.** Its
`collect_ignore` list holds only the repo's `setup.py`, which is not a
test module. Both entries that used to hide part of the suite are gone
with the code they covered: `test/decay/` alongside `hazma/_decay/`
(cython-to-rust Task 0.3) and `test/test_gamma_ray.py` alongside
`hazma/gamma_ray.py` (Task 0.2). A bare `pytest` still runs a *different*
suite from `pytest test`, but for an unrelated reason: `setup.cfg`'s
`[tool:pytest] testpaths` is `hazma`, so a bare run collects the
in-package `*_test.py` modules (`hazma/form_factors/`,
`hazma/phase_space/`) and never enters `test/`. Cite the command you ran,
not "the full suite".

**The test tree does not mirror the package one-to-one.** `test/` has
`agents/`, `positron/`, `rh_neutrino/`, `scalar_mediator/`, `spectra/`,
`vector_mediator/` plus a few loose `test_*.py`. There is no test package
for several `hazma/` subpackages. Absence of a test directory is not
evidence that an area is untested elsewhere, nor that it is covered —
check before claiming either.

**Floating-point assertions need an explicit tolerance and a reason.**
`np.isclose` defaults (`rtol=1e-5`) are generous enough to hide a real
physics regression. State the tolerance you chose and why in a comment;
a reviewer will ask.

**A test that only checks shape/sign/finiteness pins nothing.** For any
new physics, assert against an analytic limit, a published number (cite
it), or a stored regression array. See the numerical-correctness section
of [`AGENTS.md`](../../AGENTS.md).

## Lint and CI

**CI's lint job does not enforce the repo's own ruff config.**
`.github/workflows/ci.yml` runs
`ruff check --isolated --select E9,F63,F7,F82` — syntax errors, undefined
names, and broken comparisons/f-strings. `--isolated` is the load-bearing
flag: it deliberately ignores `[tool.ruff]` in `pyproject.toml`, so the
stricter rule set you get from a bare local `ruff check` is **not** what
CI runs. Green CI does not mean the tree satisfies the configured rules.
Run `ruff check` locally; do not infer "CI would have caught it".

**CI does check formatting.** The lint job runs
`black --check --diff hazma test`, so a formatting regression turns CI
red. `black` is in both this repo's preflight gate and its CI.
Do not reformat files your task does not touch just because `black`
wants to; that is how an unrelated 85-file diff gets attached to a
one-line change.

**`hazma/experimental/` and `notebooks/` are excluded from CI lint**
(`--exclude` flags on the ruff step). Code there is not held to the
repo's standard. Do not cite it as precedent, and do not import from
`experimental/` in the library.

**The CI test matrix is Python 3.10 through 3.14 on Linux, plus macOS on
3.14**, matching `pyproject.toml`'s `requires-python = ">=3.10"`. CI also
runs an import
smoke test before the suite, so a broken Cython build fails there rather
than as a confusing collection error.

## Git and orchestration

**A shared worktree can sit on detached HEAD after an orchestrated run.**
Check `git symbolic-ref -q HEAD` before committing; if detached,
re-attach to the intended branch rather than committing into limbo.

**The trunk is `master`, not `main`.** Every script here resolves it from
`origin/HEAD` with a `master` fallback. A hand-written `origin/main`
reference fails silently in the worst way: `git diff origin/main` errors,
but `git rev-parse --abbrev-ref HEAD != "main"` *passes* on `master`, so
a branch assertion written against `main` protects nothing.

## Markdown

**`markdownlint --fix` corrupts code spans documenting literal syntax.**
After any `--fix`, word-diff the result and restore any semantic
whitespace it stripped.
