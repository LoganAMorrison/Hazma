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
worktree under `.claude/worktrees/`, which is a *different directory*
from the checkout the editable install points at.

**`pip install -e .` needs Cython, NumPy, and a C/C++ compiler.** A
missing toolchain surfaces as a build error deep in `_gamma_ray`
(compiled as C++), not as a clear "install Cython" message.

**Never hand-edit generated `.c` / `.cpp`.** They are cythonize output.
Edit the `.pyx` and rebuild.

## Tests

**pytest exit code 5 means zero tests collected.** It is not a pass. A
mistyped path, a `-k` filter that matches nothing, or a `collect_ignore`
entry silently reduces the run to nothing. Read the summary line
(`N passed`), not just the exit status.

**`test/conftest.py` deliberately ignores part of the suite.** It builds
a `collect_ignore` list that excludes `test/test_gamma_ray.py` and
everything under `test/decay/`. A bare `pytest` therefore does **not**
run those files, and "the full suite is green" does not cover them. If
your change touches decay spectra or `gamma_ray.py`, run those paths
explicitly and expect to deal with why they were parked.

**The test tree does not mirror the package one-to-one.** `test/` has
`decay/`, `positron/`, `rambo/`, `rh_neutrino/`, `scalar_mediator/`,
`spectra/`, `vector_mediator/` plus a few loose `test_*.py`. There is no
test package for several `hazma/` subpackages. Absence of a test
directory is not evidence that an area is untested elsewhere, nor that it
is covered — check before claiming either.

**Floating-point assertions need an explicit tolerance and a reason.**
`np.isclose` defaults (`rtol=1e-5`) are generous enough to hide a real
physics regression. State the tolerance you chose and why in a comment;
a reviewer will ask.

**A test that only checks shape/sign/finiteness pins nothing.** For any
new physics, assert against an analytic limit, a published number (cite
it), or a stored regression array. See the numerical-correctness section
of [`AGENTS.md`](../../AGENTS.md).

## Lint and CI

**CI's flake8 pass is not a lint gate.** `.github/workflows/python-package.yml`
runs flake8 twice: once with `--select=E9,F63,F7,F82` (syntax errors and
undefined names only) and once with `--exit-zero` (advisory). Neither
enforces the `[tool.ruff]` config. Run `ruff check` locally; do not infer
"CI would have caught it".

**`hazma/experimental/` and `notebooks/` are excluded from CI lint.**
Code there is not held to the repo's standard. Do not cite it as
precedent, and do not import from `experimental/` in the library.

**CI tests on Python 3.9 and 3.10 while `pyproject.toml` requires
≥3.10.** The matrix and the declared floor disagree. Do not use syntax
newer than the lower of the two without checking which is actually
authoritative for the change you are making.

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
