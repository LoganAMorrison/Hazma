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
Cython sources are compiled at build time by `setup.py`. Until you
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

**It also needs `cargo` on `PATH`, and pip cannot supply it.**
`pyproject.toml`'s `[build-system] requires` carries `setuptools-rust`
(cython-to-rust Phase 02), which shells out to cargo to build the
`hazma._core` extension in the same pass as the Cython ones. No
toolchain, no build — of *any* extension, not just the Rust one. Install
it from rustup; edition 2024 needs rustc ≥ 1.85. Both this requirement
and the Cython half disappear at the Phase 07 maturin cutover.

**Editing a `.rs` and re-running pytest tests the OLD extension, exactly
like a `.pyx`.** And the trap has an extra step, because the fast
iteration command is not the publishing one: `cargo build` and
`cargo test` work out of `rust/target/`, which nothing Python imports.
Only `pip install -e .` re-links the crate into the tree as
`hazma/_core.abi3.so`. So iterate with
`cargo test --manifest-path rust/Cargo.toml --no-default-features`, then
reinstall before believing any Python-side result, and confirm with
`python -c "import hazma._core; print(hazma._core.__file__)"` that the
path is inside your worktree.

**Deleting a `.pyx` does not make its module unimportable.** The built
`_name.cpython-*.so` and the generated `_name.c` sit beside the source in
the package directory, both are gitignored, and neither is removed by
deleting the `.pyx`, by `git checkout`, or by `git stash`. Python imports
the stale extension happily. So a test written as
`pytest.raises(ImportError)` to prove a module was deleted is testing
whoever last ran `pip install -e .`, not the change — assert on the
source files and the `setup.py` entry instead
(`test/test_core_photon_rho.py::test_the_cython_twin_is_gone_from_the_tree`
is the worked example). And `rm` the orphaned `.so`/`.c` after any stash
cycle that briefly restored the source, or the next build resurrects the
extension.

**`git checkout <path>` restores from the *index*, not from HEAD.** If
you staged a file earlier with `git add -A` and have since edited it,
`git checkout` on that path silently reverts to the staged version — it
prints only `Updated 1 path from the index`. Mid-task this reads as "my
edit vanished". Use `git checkout HEAD -- <path>` or
`git restore --source=origin/master <path>` when you mean a specific
revision, and re-inspect the file afterwards rather than trusting the
message.

**`cargo test` must be `--no-default-features`.** The crate's default
`extension-module` feature tells PyO3 to leave CPython's symbols
undefined for the interpreter that `dlopen`s the module. A test
executable has no such interpreter, so with the feature on the harness
fails to link — a wall of undefined `_Py*` symbols that reads like a
broken toolchain rather than a wrong flag.

**Never hand-edit generated `.c` / `.cpp`.** They are cythonize output.
Edit the `.pyx` and rebuild.

**A clean wheel is not evidence of a clean sdist.** They are built by
different machinery and neither fix reaches the other: the wheel's
contents come from `[tool.setuptools.packages.find]` in
`pyproject.toml`, the sdist's from `MANIFEST.in`. `MANIFEST.in`'s
`global-include` is a **repo-wide** sweep, so it happily picks up
`.claude/`, `.codex/` and `projects/` — it did, unnoticed, for four
months, because no one ran `build --sdist` (cython-to-rust Task 0.4).
Check the artifact you actually changed. And when probing a `tar tzf`
listing for paths that should be absent, **anchor the pattern** (`^…$`):
an unanchored `_positron` or `gamma_ray` matches dozens of live paths
and buries the real hit.

**A path probe is not a build.** A tarball can list exactly the right
files and still fail to install. The gate is
`uv pip install --no-binary hazma dist/*.tar.gz` into a *fresh* venv,
then import-smoke from outside the repo — `cd /tmp` first, or you will
import the checkout instead of the installed package.

## Linters

**Install the linters from the `lint` dependency group, not by hand.**
`pyproject.toml`'s `[dependency-groups]` is the only place the `black`,
`isort`, and `ruff` pins live, and CI's Lint job installs that same
group. Anything else risks a formatter that disagrees with CI:

```sh
uv pip install --group lint     # or: pip install --group lint
```

`--group dev` adds `pytest` and `pytest-xdist` on top, for the full
preflight toolchain. The plugin is not optional: `pyproject.toml`'s
`addopts` passes `--numprocesses` to every run, and a pytest without
xdist rejects that flag outright.
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
`hazma/gamma_ray.py` (Task 0.2).

**A bare `pytest` is now the whole suite, and it is slow.** pytest is
configured in `pyproject.toml`'s `[tool.pytest.ini_options]` — not
`setup.cfg`, which carries only a pointer comment — and `testpaths` is
`["hazma", "test"]`. Before cython-to-rust Task 1.3 it was `hazma`
alone, so a bare run collected the in-package `*_test.py` modules
(`hazma/form_factors/`, `hazma/phase_space/`) and never entered `test/`;
CI, `preflight.sh`, and a contributor typing `pytest` each ran a
different subset. They now run the same one. The cost is the golden
parity corpus and its nested adaptive quadrature under `test/parity/`.
The work is minutes of CPU — 598s serial for the bare run, measured
idle on macOS/arm64 (2026-08-17) — but the pytest-xdist `addopts` in
`pyproject.toml` spread it across cores: 45s wall at `-n 12` on the
same machine, identical collection and outcomes. `pytest -n 0` restores
the in-process run that `--pdb` and clean sequential output need.
Narrow with an explicit target while iterating, but cite the command
you ran — "the full suite" is only true of the bare form.

**Running the parity suite needs an editable install, not just any
install.** `test/parity/cases.py` refuses a `hazma` that resolves
outside the repository (`cases.assert_module_is_repo_tree`), and running pytest
from the repo root puts the source tree first on `sys.path` regardless,
so a non-editable `pip install .` leaves the corpus looking at a tree
with no compiled extensions in it. `pip install -e .`, then confirm with
`python -c "import hazma.spectra._photon._muon as m; print(m.__file__)"`
that the `.so` is inside your worktree. CI does the non-editable install
first (that is what its outside-the-repo import smoke test checks) and
reinstalls editable before the test step.

**The parity corpus runs on every platform, and it did not always.**
Until 2026-08-18 it only reproduced on the host that captured it
(macOS/arm64): roughly 70-75 of its 626 blocks failed on Linux/glibc,
mostly last-bit `libc.math` differences but some at cancellation points
where the pinned value flipped sign. CI skipped it off macOS. Three
things fixed that and each names what it covers —
`test/parity/stability.py` (494 stored positions that assert nothing),
`tolerances.PLATFORM_EXACT_RTOL` (the `EXACT` class off the capturing
libm) and `tolerances.zero_floor` (stored exact zeros). If a
`test/parity` failure surfaces on a platform you have not seen it on,
read
[`docs/followups/done/parity-corpus-pins-ill-conditioned-points.md`](../followups/done/parity-corpus-pins-ill-conditioned-points.md)
and decide which of the three it belongs in before widening anything;
none of them is a catch-all.

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
3.14**, matching `pyproject.toml`'s `requires-python = ">=3.10"`. Each
entry installs a Rust toolchain (`dtolnay/rust-toolchain@stable`; without
cargo nothing builds — see the `setuptools-rust` note above), installs
hazma non-editable, runs an import smoke test from outside the repo (so a
broken build or a missing package-data entry fails there rather than as a
confusing collection error), reinstalls editable, and then runs a bare
`pytest`, parity corpus included, on every entry.

**CI has a third job, `rust`.** It runs the same three cargo gates
`preflight.sh` does — `cargo fmt --check`,
`cargo clippy --all-targets -- -D warnings`, and
`cargo test --no-default-features` — on ubuntu only, since none of them
is platform-sensitive. It installs a Python for the same reason the flag
exists: the test harness links libpython for real.

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
