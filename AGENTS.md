# AGENTS.md

Repo-wide guidance for any coding agent working in Hazma — Claude Code,
Codex, or otherwise. This file is the tie-breaker: when it conflicts with
a skill, a task note, or a plan, this file wins.

Agent-neutral rules that many skills share live one level down, in
[`docs/agents/`](docs/agents/README.md). This file carries the
repo-specific facts; that directory carries the shared invariants (the
preflight gate, the doc-consistency checklist, the reviewer roster,
environment gotchas, the lessons ledger). Neither restates the other.

## What Hazma is

Hazma computes indirect-detection observables for sub-GeV dark matter:
gamma-ray, electron/positron, and neutrino spectra from dark matter
annihilation; limits from existing gamma-ray data; discovery reach for
future detectors; and CMB constraints. It is a scientific Python library
with performance-critical inner loops in compiled code — historically
Cython, currently mid-migration to a single Rust extension
(`hazma._core`) under the `cython-to-rust` project. Both toolchains build
in the same `pip install -e .` pass and both are gated by
[`scripts/agents/preflight.sh`](scripts/agents/preflight.sh); until that
project closes, expect to meet either one.

The user-facing surface is the **public Python API** — module paths,
function and class names, keyword arguments, return shapes and units, and
the numerical values those functions produce. Everything in
[`docs/versioning.md`](docs/versioning.md) is defined against that
surface.

## Layout

```text
hazma/
├── spectra/            # dnde_photon / dnde_positron / dnde_neutrino
│   ├── _photon/        #   per-final-state photon spectra + data
│   ├── _positron/      #   per-final-state positron spectra + data
│   ├── _neutrino/      #   per-final-state neutrino spectra + data
│   ├── _nbody.py       #   N-body spectra via phase-space integration
│   ├── boost.py        #   boost integrals (rest frame → lab frame)
│   └── altarelli_parisi.py
├── theory/             # Theory ABC: the model interface every model implements
├── scalar_mediator/    # concrete models …
├── vector_mediator/
├── rh_neutrino/
├── single_channel.py
├── form_factors/       # hadronic form factors
├── phase_space/        # RAMBO and friends
├── _utils/             # Cython helpers (boost.pyx, constants.pxd, …)
├── limits/             # limit-setting machinery
├── relic_density/
├── cmb.py, pbh.py, parameters.py, utils.py
├── gamma_ray_data/     # detector response data (*.dat)
└── experimental/       # excluded from lint gates; not a public surface
rust/                   # the hazma._core crate (PyO3); src/kernels.rs is
                        #   PyO3-free, src/dispatch.rs is the PyO3 boundary
test/                   # pytest suite, mirrors the package tree
docs/source/            # Sphinx documentation (the published docs)
notebooks/              # exploratory notebooks + spectrum-generation scripts
examples/
```

### Layering

The dependency direction is one-way; do not invert it.

1. **Compiled kernels** — the `.pyx` under `_utils/`, `spectra/`,
   `scalar_mediator/` and `vector_mediator/`, plus the `rust/` crate that
   is replacing them. Numerics only; they import nothing from the
   pure-Python layers above.
2. **Primitives** (`parameters.py`, `utils.py`, `form_factors/`,
   `phase_space/`) — physical constants, kinematics, shared helpers.
3. **Spectra** (`spectra/`) — builds on 1 and 2. This is the layer most
   new physics lands in.
4. **Theory** (`theory/`) — the abstract model interface, plus the
   gamma-ray-limit, CMB, and constraint mixins.
5. **Models** (`scalar_mediator/`, `vector_mediator/`, `rh_neutrino/`,
   `single_channel.py`) — concrete `Theory` implementations.
6. **Analysis** (`limits/`, `relic_density/`, `cmb.py`, `pbh.py`) —
   consumes models.

A leading underscore on a package (`_utils`, `spectra/_photon`) means
*private implementation*. Public callers go through `hazma.spectra`,
`hazma.theory`, and the model packages.

## Commands

```sh
pip install -e .          # build the Cython + Rust extensions in place
pip install --group dev   # black, isort, ruff, pytest, pytest-xdist, mpmath
pytest                    # full suite (hazma + test), parallel via xdist
pytest -n 0               # same suite in-process (--pdb needs this)
pytest test/spectra -q    # one area
black hazma test          # format
isort hazma test          # import order
ruff check hazma test     # lint
pyright hazma             # types (advisory)
```

The Rust crate has its own three, all run from the repo root against
`rust/Cargo.toml` (`preflight.sh` runs exactly these):

```sh
cargo fmt --manifest-path rust/Cargo.toml --check
cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path rust/Cargo.toml --no-default-features
```

`--no-default-features` on the test command is load-bearing: the crate's
default `extension-module` feature leaves CPython's symbols to be
resolved by the interpreter that loads the shared object, and a test
executable has no interpreter, so with it on the harness will not link.

The one-command pre-commit gate is
[`scripts/agents/preflight.sh`](scripts/agents/preflight.sh) — see
[`docs/agents/preflight.md`](docs/agents/preflight.md). Run it before
every commit; do not assume a hook covers it.

**Editing a `.pyx` or `.pxd` requires a rebuild.** `pip install -e .`
does not pick up Cython edits automatically on every setup — if a change
to a kernel does not show up, rebuild explicitly and confirm with
`python -c "import hazma; print(hazma.__file__)"` that you are importing
the tree you edited, not an installed copy.

**Editing a `.rs` requires the same rebuild, and `cargo build` is not
it.** `cargo build` refreshes `rust/target/`, which nothing imports;
`pip install -e .` is what re-links the crate into the tree as
`hazma/_core.abi3.so`. So the loop is: iterate with
`cargo test --no-default-features` (fast, needs no reinstall, and is
where kernel unit tests belong), then re-run the editable install before
any Python-side check — pytest, the parity corpus, an interactive
`import hazma._core` — is worth believing. Confirm the same way as for
Cython: `python -c "import hazma._core; print(hazma._core.__file__)"`
must land inside your worktree.

A source build now needs `cargo` on `PATH` at all: `pyproject.toml`'s
`[build-system] requires` includes `setuptools-rust`, and pip cannot
install a Rust toolchain for you.

## Conventions

- **Python ≥ 3.10.** Type annotations on public functions; `from
  __future__ import annotations` where it helps. Runtime introspection of
  annotations is supported (`runtime-typing = true`), so do not stringify
  annotations that users may resolve.
- **Formatting is black (88 cols) + isort (black profile).** Do not
  hand-format around them, and do not install them by hand — the pins
  live once, in `pyproject.toml`'s `[dependency-groups]`, and CI
  installs that same `lint` group. A version pinned in two places drifts
  and turns CI red on lines nobody touched.
- **Naming:** modules and functions `snake_case`, classes `UpperCamel`,
  constants `SCREAMING_SNAKE`. Physics symbols keep their conventional
  spelling in docstrings (`dN/dE`, `⟨σv⟩`), not in identifiers.
- **NumPy-style docstrings** with `Parameters` / `Returns` sections, and
  **units stated for every physical quantity**. A returned spectrum
  without its units in the docstring is an incomplete public API.
- **Arrays in, arrays out.** Spectrum functions accept scalars or NumPy
  arrays and broadcast; new public functions follow that contract and
  have a test for both.
- **No bare `except:`.** Raise the errors in `hazma/hazma_errors.py`
  where they fit.
- **`hazma/experimental/` and `notebooks/` are excluded from the lint
  gate** (CI's ruff step passes `--exclude` for both). Do not treat code
  there as a pattern to copy, and do not import from `experimental/` in
  the library.
- **`hazma/deprecated/` stays importable.** Removing or changing anything
  there is a user-facing break — see `docs/versioning.md`. The package is
  empty today (its last module went in cython-to-rust Task 0.2), so the
  rule binds the next module parked there.
- **Never commit generated C/C++.** `setup.py` cythonizes on build; the
  `.c` / `.cpp` output is not the source of truth.
- **No `breakpoint()`, `pdb`, or stray `print()` in library code.** Use
  the returned value or a logger.

## Numerical correctness

This is a physics library — a silently wrong number is worse than a
crash, and neither a type checker nor a linter will catch one.

- **Every new physics function needs a pinned numerical test.** A test
  that only asserts "returns a positive float" is not a test. Pin against
  an analytic limit, a published value (cite it), an independent
  implementation, or a stored regression array.
- **State the tolerance and why.** `np.isclose(..., rtol=1e-6)` with no
  comment invites silent loosening later.
- **Check the boundaries:** threshold energies, the endpoint of a
  spectrum, `E → m/2`, zero mass, equal masses, and the massless limit.
  Spectra routinely go NaN or negative exactly at the kinematic edge.
- **Integrals need their range and units justified in the docstring.**
  A `quad` call whose limits changed without a comment is a review
  finding.
- **Changing a number is a behavior change.** If a fix moves a published
  spectrum, say so in the PR body and in `CHANGELOG.md` — even when every
  test still passes because the tolerance absorbed it.

## Git and branches

- **Never commit directly to `master`.** Branch first. The trunk branch
  is `master`, not `main` — every script and skill in this repo resolves
  it from `origin/HEAD` with a `master` fallback.
- Branches carry the driving agent's identity as their prefix:
  `claude/<...>` for Claude Code, `codex/<...>` for Codex. Both are valid
  and permanent; all tooling parses either.
  - **Project work:** `<agent>/<project-slug>/<task-slug>`.
  - **Ad-hoc work:** `<agent>/<short-description>`.
- Commit messages and PR titles follow
  [`docs/PR_GUIDELINES.md`](docs/PR_GUIDELINES.md) (Conventional
  Commits). Validate a header with
  `scripts/agents/check_pr_title.py "<header>"` rather than counting
  characters by hand.

## Where work is tracked

Multi-commit efforts live under `projects/<slug>/` with a plan, task
notes, ADRs, and learnings. Single-commit changes skip the scaffolding.
The full contract — when to create a project, flat vs phased, ADR
placement, the follow-up backlog, the project lifecycle — is
[`docs/workflow.md`](docs/workflow.md).

Durable decisions go in ADRs ([`docs/adrs/`](docs/adrs/README.md)
repo-wide, `projects/<slug>/adrs/` project-scoped). Deferred work goes in
[`docs/followups/`](docs/followups/README.md), not in a comment and not
only in a PR description.

## Skills

`.claude/skills/` and `.codex/skills/` hold parallel workflow skills for
this loop: `execute-single-task`, `commit-and-pr`, `review-pr`,
`review-respond`, `review-cycle`, `task-pipeline`, `begin-phase`, and
`review-plan`. Each expects the filesystem contract above and points into
`docs/agents/` for shared rules. Read the active agent's `SKILL.md` for
its exact inputs and outputs.
