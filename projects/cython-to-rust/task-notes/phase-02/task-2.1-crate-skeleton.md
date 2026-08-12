# Task 2.1: Crate + setuptools-rust integration

**Date:** 2026-08-08
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-02-rust-scaffold.md` (Task 2.1 and
the phase Goal); `../../rules.md` "Rust conventions" rules 1–3
**Related ADRs:** ADR-0001 (Accepted — Rust + PyO3, single abi3
`hazma._core`, setuptools-rust coexistence during Phases 02–06)
**Depends On:** Phase 01 complete

## Objective

Stand up the walking skeleton: a Rust crate that builds into
`hazma._core` at its final import path, alongside the surviving Cython,
under the existing setuptools backend. No physics — the deliverable is a
proven toolchain, not a kernel.

## Exit Criteria

Copied from `../../phases/phase-02-rust-scaffold.md` ("Task 2.1"):

- `rust/Cargo.toml` (edition 2024) with `pyo3` (`abi3-py310`,
  `extension-module`) and `numpy` crates; `rust/src/lib.rs` defines
  `#[pymodule] _core` with empty submodules (`photon`, `positron`,
  `neutrino`, `scalar_mediator`, `vector_mediator`) plus one trivial
  round-trip function (array in → array out) for plumbing tests.
- `setup.py` gains `RustExtension("hazma._core", ...)`;
  `pip install -e .` (uv env) builds Cython + Rust in one pass;
  `python -c "import hazma._core"` works.
- `hazma/_core.pyi` stub started; py.typed unaffected.
- **(Added by this task, and patched into the phase file.)** The parity
  gate still runs in bit-equality mode — see "The widening" below.

Scope boundary held deliberately: CI, `preflight.sh`'s cargo gates and
the dev-loop docs belong to Task 2.2; the pytest plumbing suite belongs
to Task 2.3. This task ships the crate, the build wiring, and the stub.

## Inputs Reviewed

- `../../PLAN.md` (Goal, Scope, Numerical impact, Phases table).
- `../../phases/phase-02-rust-scaffold.md` — Task 2.1 exit criteria,
  phase Goal and Prerequisites.
- `../README.md` (project working memory) — Findings, Numerical impact
  so far, Handoff.
- `README.md` (phase working memory).
- `../../rules.md` — Rust conventions rules 1–3 (edition 2024; one
  cdylib named `hazma._core` with per-domain submodules; kernels are
  PyO3-free `fn(f64, ...) -> f64`), and parity rules 1–2.
- `../../adrs/ADR-0001-rust-pyo3-maturin-over-pybind11.md`.
- `../../references/numerics-replacements.md` §"Entry-point dispatch
  contract" — the scalar/1-D/error shape the round-trip function
  prototypes.
- `setup.py`, `pyproject.toml`, `MANIFEST.in`, `.gitignore`,
  `.github/workflows/ci.yml`, `.github/workflows/release.yml`.
- `test/parity/cases.py`, `tolerances.py`, `test_parity.py`,
  `README.md` — reached because the scaffold trips their
  "has the port started?" predicate.
- `docs/agents/lessons.md`, `docs/agents/environment.md`,
  `docs/agents/preflight.md`, `docs/agents/doc-consistency.md`.
- `docs/followups/todo/model-spectra-reject-scalar-energies.md`.

## Findings

### The widening: the scaffold silently switched off the parity gate

The one thing in this task that was not foreseeable from the phase file,
and the reason the diff reaches `test/parity/`.

`test/parity/tolerances.provenance` decides whether the corpus runner
demands bit-equality or falls back to the declared per-function budgets
(1e-8 relative for quad-backed entry points). One of its three inputs was
`cases.rust_core_available()` — literally
`importlib.util.find_spec("hazma._core") is not None`. That predicate was
a faithful reading of "has the port started?" only while the extension
did not exist. **The moment this task lands it is true in every build,
and stays true through Phase 03 while every value still comes from
Cython.** Measured on this branch before the fix:

```text
exact: False
detail: …; hazma._core is importable
```

So Task 2.1 would have taken the project's primary numerical gate from
bit-equality to 1e-8 for the whole of Phases 02–03 — the stretch where a
one-ulp regression is both most worth catching and least expected — and
nothing would have turned red. That is `docs/agents/lessons.md`'s
`[gate-disabled-stays-green]` class, which this repo has already been
bitten by twice (PR #52, #53). The runner does report the mode as a skip,
so it is visible rather than silent; it is still wrong, and the skip
reason ("`hazma._core` is importable") reads as a statement about the
values that is not true.

Same predicate, second consequence: `generate.assert_no_rust_core()`
would refuse to regenerate the corpus at all. The corpus *does* need
repair before Phase 04 — the ill-conditioned points in
[`../../../../docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md`](../../../../docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md)
— so an over-strict guard here blocks a task the project has already
committed to.

The fix is to ask the question `rules.md` rule 2 actually asks — is a
kernel *served* by Rust? — instead of the proxy. `cases.rust_core_kernels()`
walks `hazma._core` and its submodules and returns the public callables it
finds, excluding the scaffold's own `roundtrip` probe by name. Empty
today; non-empty from the first Phase 04 swap. After the fix, on the
capturing interpreter:

```text
exact: True | detail: ''
```

### PyO3 0.29 / toolchain facts worth not rediscovering

- **`Bound::downcast` is now `Bound::cast`.** pyo3 0.29.2 renamed it;
  every tutorial and most crates still say `downcast`. The compiler's
  suggestion is correct, but the rename is easy to mistake for a missing
  trait import.
- **A stray `Cargo.toml` anywhere above the checkout captures the crate.**
  `cargo build` failed with `failed to load manifest for workspace member
  /Users/logan.morrison/bazel_utils` — a workspace manifest in the
  developer's home directory, nothing to do with hazma. `rust/Cargo.toml`
  now declares an explicit empty `[workspace]`, which is the documented
  fix and costs one line.
- **PyO3 0.29 does not emit macOS's `-undefined dynamic_lookup` for you.**
  A plain `cargo build` of the cdylib fails with ~40 undefined `_Py*`
  symbols even though the identical crate builds fine through
  setuptools-rust, which passes the flags itself. `rust/build.rs` emits
  them via `cargo::rustc-cdylib-link-arg`, which applies *only* when a
  cdylib is being linked and so leaves the test harness alone.
- **`extension-module` cannot be on for `cargo test`.** With it enabled
  the test executable has no interpreter to resolve CPython's symbols
  against. It is therefore a default-on optional feature and the unit
  tests run as `cargo test --no-default-features`, which links libpython.
  Task 2.2 wires that exact spelling into `preflight.sh`.
- **abi3 is observable in the filename, and across interpreters.** The
  installed file is `hazma/_core.abi3.so` (not
  `_core.cpython-312-darwin.so`), and one such file built under CPython
  3.12.12 imports unchanged under 3.13.7 — see Verification. Per
  `lessons.md` `[wheel-tag-vs-extension-abi]` the *wheel* stays
  CPython-tagged, and it does. **The tag names the building interpreter,
  not the extension:** the same tree yields
  `hazma-2.1.0-cp312-cp312-macosx_11_0_arm64.whl` from the 3.12.12 venv
  and `…-cp313-cp313-…` from a 3.13.7 one, while the shared object inside
  is `_core.abi3.so` in both. The invariant to check is therefore
  "`cp<XY>`, not `abi3`" — quoting one interpreter's tag as *the* tag is
  what let two different figures into this note's first draft.

### The live dispatch contract is not what the reference describes

Building the dispatch helper from
`../../references/numerics-replacements.md` meant checking it against the
Cython, and it diverges in four ways — a 0-d array raises rather than
taking the scalar path, a Python list is *accepted*, shape errors are
`AssertionError` not `ValueError`, and one neutrino module's message says
"Photon energies". All four are measured, cited and written into that
reference under a new "What the Cython actually does today" subsection,
because Task 3.5 implements from it and two of the four would be silent
public-API narrowings. Items 1–2 are the same layer as
[`../../../../docs/followups/todo/model-spectra-reject-scalar-energies.md`](../../../../docs/followups/todo/model-spectra-reject-scalar-energies.md);
resolving that follow-up at the public boundary settles them.

This task implements the reference's *target* contract, not the current
Cython behavior, because that is what the phase file's Task 2.3 exit
criteria ask the plumbing suite to assert. Nothing in `hazma/` calls
`roundtrip`, so no user-visible behavior rides on the choice yet.

### Packaging

- **The sdist needed the crate added by hand.** `MANIFEST.in`'s
  `global-include` covers `*.txt *.rst *.pyx *.pxd *.c *.md` — no `.rs`,
  no `Cargo.toml`. Without the four new lines the sdist would have
  carried a `setup.py` that builds a crate it does not ship. Verified by
  installing the tarball, not by reading it (`docs/agents/environment.md`,
  "a path probe is not a build").
- **`.pyi` stubs ship already**, via setuptools' own handling rather than
  `[tool.setuptools.package-data]`, which lists `*.pyd` where `*.pxd` was
  surely meant. 15 `.pyi` in both wheel and sdist (14 pre-existing + the
  new `_core.pyi`). The `*.pyd` typo is already recorded in
  [`../../../../docs/followups/todo/sdist-ships-generated-c-and-docs.md`](../../../../docs/followups/todo/sdist-ships-generated-c-and-docs.md);
  not touched here.
- **The repo has no `py.typed` marker at all** (`git ls-files | grep -c
  py.typed` → 0), so "py.typed unaffected" is satisfied by there being
  nothing to affect. Worth stating rather than implying: a reader could
  reasonably assume the 14 existing stubs mean hazma is PEP 561 typed.

## Decisions and Implementation Notes

- **Crate layout mirrors `rules.md` rule 8** (Rust conventions rule 3).
  `kernels.rs` is plain GIL-free Rust with its own `#[cfg(test)]` tests
  and no PyO3 types at all; everything PyO3 touches sits above it —
  `dispatch.rs` owns argument conversion, array glue and error mapping,
  `lib.rs` owns module registration, and the five submodule files are
  registration only. The rule is that PyO3 stays *out of the kernels*,
  not that it lives in exactly one file: `lib.rs` and the submodules
  necessarily use it too. Phases 03–06 add kernels to `kernels.rs`-shaped
  modules and call `dispatch::map_unary`, never PyO3 inside a kernel.
- **Submodules are registered in `sys.modules`, not just attached.**
  `add_submodule` alone makes `from hazma._core import photon` work but
  leaves `from hazma._core.photon import dnde_photon_muon` an
  ImportError. Both forms work now, and each child's `__name__` is the
  fully-qualified path so it matches its `sys.modules` key.
- **`roundtrip` is the identity, and that is deliberate.** A
  value-preserving probe lets the Task 2.3 tests assert bit-equality with
  no tolerance to argue about. Identity alone would not prove the Rust ran
  — a passthrough dispatch would satisfy it — so the array path allocates
  a *fresh* array and Task 2.3 asserts the result never aliases the input.
  Measured here already: `np.shares_memory(result, input)` is False.
- **`lto = true` / `codegen-units = 1` on the release profile.** Not a
  performance claim, and nothing is benchmarked yet (`rules.md` Process
  rule 3 would require one): it is the setting under which the Phase 03+
  kernels get inlined across the module boundary, chosen now so no later
  measurement has to be retaken after a profile change.
- **`Cargo.lock` is committed.** This crate builds a shipped binary
  artifact, so the resolved dependency graph is a build input, and the
  sdist carries it.
- **Review round 1 (PR #55) landed three fixes**, all documentation
  except one docstring. (1) The note quoted two different wheel tags —
  `cp313` in Findings, `cp312` in Verification — because the first
  measurement was taken before the venv was rebuilt on 3.12.12 for
  bit-equality mode, and never swept. Fixed by stating the *invariant*
  (`cp<XY>`, not `abi3`) with the mechanism, rather than picking a tag.
  (2) `rules.md` numbers its rules per section while the plan and phase
  files cite them flat, and that mapping was written down nowhere, so
  "rules 6–8" read as a dangling reference; `rules.md` now carries the
  key. (3) "`dispatch.rs` is the single PyO3 boundary" overstated it —
  `lib.rs` and the submodules use PyO3 too. Corrected in the note *and*
  in `rust/src/lib.rs`'s module doc, which carried the same wording.
- **The reviewer's own run is the first off-capture measurement of the
  new predicate, and it corroborates.** On CPython 3.13 they got
  `1008 passed, 14 skipped` against this task's `1009 passed, 13 skipped`
  on 3.12.12 — 1022 collected both ways, with exactly one test moving
  from pass to skip. That test is `test_running_on_the_capturing_tree`,
  which is precisely the designed behavior: an interpreter that is not
  the capturing one drops the suite to declared budgets and *says so*.
  Had `rust_core_kernels()` still keyed on importability, the 3.12.12 run
  would have skipped it too and the two runs would have been
  indistinguishable.
- **`.gitignore` gained an explicit `rust/target/`.** The bare `target/`
  under the PyBuilder heading already matched it by accident; naming it
  means a later cleanup of that unrelated line cannot silently start
  tracking a multi-gigabyte build directory.

## Files Changed

### New — the crate

- `rust/Cargo.toml` — edition 2024, `[lib] name = "_core"`,
  `crate-type = ["cdylib", "rlib"]`, pyo3 0.29.2 (`abi3-py310`) + numpy
  0.29.0, `extension-module` as a default-on optional feature, explicit
  empty `[workspace]`, release profile.
- `rust/Cargo.lock` — resolved graph (25 packages).
- `rust/build.rs` — macOS `-undefined dynamic_lookup` for the cdylib only.
- `rust/src/lib.rs` — `#[pymodule] _core`, the `roundtrip` pyfunction, and
  `add_submodule` (attach + `sys.modules`).
- `rust/src/dispatch.rs` — `map_unary`, the one implementation of the
  entry-point dispatch contract. *(Task 3.5 gave it two siblings,
  `map_flavors` and `require_vector`, over one shared classification.)*
- `rust/src/kernels.rs` — `roundtrip` plus two `cargo test` unit tests.
- `rust/src/{photon,positron,neutrino,scalar_mediator,vector_mediator}.rs`
  — empty `register` hooks.

### Build and packaging

- `setup.py` — imports `setuptools_rust`, declares
  `RustExtension("hazma._core", path="rust/Cargo.toml",
  binding=Binding.PyO3, py_limited_api=True)`, passes it as
  `rust_extensions=`; module docstring rewritten for the two-toolchain
  window.
- `pyproject.toml` — `setuptools-rust` added to `[build-system] requires`.
- `MANIFEST.in` — `rust/Cargo.toml`, `rust/Cargo.lock`, `rust/build.rs`,
  `recursive-include rust/src *.rs`.
- `.gitignore` — explicit `### Rust ###` section.
- `hazma/_core.pyi` — typed stub for `roundtrip` (two overloads) plus a
  note that per-submodule stubs land with their first kernel.

### The parity gate (the widening)

- `test/parity/cases.py` — `rust_core_available` re-documented as
  "extension exists at all"; new `rust_core_kernels()` +
  `_CORE_SCAFFOLD_NAMES`; `assert_no_rust_core` keyed on served kernels
  and its message now names them.
- `test/parity/tolerances.py` — `provenance` keys on served kernels.
- `test/parity/test_parity.py` — three tests pinning the distinction
  (scaffold serves none; one injected callable is found, blocks
  regeneration and leaves exact mode; an imported third-party module is
  not a kernel).
- `test/parity/README.md` — "When *not* to regenerate" repointed at the
  served-kernel predicate.

### Project docs

- `projects/cython-to-rust/phases/phase-02-rust-scaffold.md` — Task 2.1
  exit criteria gained the bit-equality bullet, recording the widening on
  the canonical record rather than leaving it inferable from the diff.
- `projects/cython-to-rust/references/numerics-replacements.md` — new
  "What the Cython actually does today (measured, Task 2.1)" subsection.
- `projects/cython-to-rust/task-notes/phase-02/README.md` — status,
  findings, verification.
- This note.

## Verification

Everything below ran on the corpus's **capturing environment** — CPython
3.12.12, macOS/arm64 — which is the only environment in which the parity
gate can be in bit-equality mode. That mattered enough to rebuild the
venv for: the first pass of this task ran on 3.12's successor and every
parity block was silently on declared budgets.

### The gate

```text
$ scripts/agents/preflight.sh --paths "setup.py hazma/_core.pyi \
    test/parity/cases.py test/parity/test_parity.py test/parity/tolerances.py" \
    --md "projects/cython-to-rust/phases/phase-02-rust-scaffold.md \
    projects/cython-to-rust/references/numerics-replacements.md test/parity/README.md"
PASS   black --check
PASS   isort --check-only
PASS   ruff check
PASS   pytest                  1009 passed, 13 skipped, 5 warnings in 569.45s (0:09:29)
PASS   import hazma            version 2.1.0
PASS   markdownlint
SKIP   version bump            not a closing PR (pass --closing)
PASS   forbidden tokens        none added
RESULT: PASS
```

1022 collected (`pytest --collect-only -q` → `1022 tests collected`),
against 1019 at Phase 01 close. The +3 are this task's three predicate
tests; the skip count is **unchanged at 13**, which is itself the
evidence that the parity suite stayed in bit-equality mode — the mode is
reported as a skip on `test_running_on_the_capturing_tree`, so a
budget-mode run would have shown 14.

Directly, rather than by inference:

```text
$ python -c "…; print(tolerances.provenance(manifest))"
exact: True | detail: ''
$ python -m pytest test/parity/test_parity.py::test_running_on_the_capturing_tree -q -rs
1 passed in 0.97s
$ python test/parity/generate.py --check
corpus OK: 41 cases / 1580 arrays match the manifest
(generated at 010747c6125d, kernel digest f5e6e269be47)
```

### Rust

```text
$ cd rust && cargo fmt --check                       # clean, no output
$ cargo clippy --all-targets -- -D warnings          # Finished, no warnings
$ cargo test --no-default-features
test result: ok. 2 passed; 0 failed; 0 ignored   (unit)
test result: ok. 0 passed; 0 failed; 0 ignored   (doc-tests)
```

The two unit tests are `kernels::tests` — `roundtrip` is the identity on
ordinary values compared by `to_bits()` (so ±0 are distinguished) and
preserves NaN and both infinities. They are GIL-free, per `rules.md` Rust
rule 3.

### The dispatch contract, exercised

`hazma._core.roundtrip`, on the built tree:

| Input | Result |
| --- | --- |
| `1.5` | `1.5` (`float`) |
| `np.float64(2.5)` | `2.5` |
| `np.float32(3.5)` | `3.5` |
| `3` (int) | `3.0` |
| `np.array(3.5)` (0-d) | `3.5` |
| `np.array([1., -2., nan, inf])` | same values, `float64`, `shares_memory` → `False` |
| `np.arange(6.)[::2]` (non-contiguous) | `[0., 2., 4.]` |
| `np.array([], dtype=f8)` | `[]` |
| `np.zeros((2,2))` | `ValueError: Input values must be 0 or 1-dimensional.` |
| `np.array([1,2], dtype=int64)` | `ValueError: Input values must be a float64 array; got dtype int64.` |
| `[1.0]` (list) | `ValueError: Input values must be a float or a NumPy array.` |
| `'x'` | `ValueError: Input values must be a float or a NumPy array.` |

### abi3, proved rather than asserted

The installed file is `hazma/_core.abi3.so`, not `_core.cpython-312-…`.
Built by CPython 3.12.12, then loaded by hand under a bare 3.13.7
interpreter **with no NumPy installed**:

```text
under CPython 3.13.7 (no numpy installed): roundtrip(2.5) = 2.5
```

That run is also what caught the NumPy-capsule panic (see Findings); on
the first attempt the same command aborted with
`pyo3_runtime.PanicException: Failed to access NumPy array API capsule`.

The wheel stays CPython-tagged, which is correct while Cython remains.
Re-derived from the final tree on the 3.12.12 venv, reading the tag out
of the archive rather than off the filename:

```text
$ uv build --wheel
Successfully built dist/hazma-2.1.0-cp312-cp312-macosx_11_0_arm64.whl
$ python -c "…read('hazma-2.1.0.dist-info/WHEEL')…"
Root-Is-Purelib: false
Tag: cp312-cp312-macosx_11_0_arm64
```

21 `.so` inside (20 Cython + `hazma/_core.abi3.so`) and 15 `.pyi`. Build
the same tree on 3.13.7 and the tag reads `cp313-cp313-…` with the same
`_core.abi3.so` inside — the tag follows the interpreter, the extension
does not.

### Packaging

`uv build` on a cleaned tree (`rm -rf dist build .pytest_cache` first —
`lessons.md` `[artifact-inventory-depends-on-cwd-state]`), diffed against
an sdist built the same way from a throwaway `origin/master` worktree:

```text
master sdist: 404   branch sdist: 418
> hazma/_core.pyi
> rust/  rust/Cargo.lock  rust/Cargo.toml  rust/build.rs
> rust/src/  rust/src/{dispatch,kernels,lib,neutrino,photon,positron,
             scalar_mediator,vector_mediator}.rs
```

Nothing else moved, and `^\.venv/|^rust/target/|^\.pytest_cache/` matches
0 entries. Then the real gate — a source install, not a path probe:

```text
$ uv pip install --no-binary hazma dist/hazma-2.1.0.tar.gz   # fresh 3.13.7 venv
$ cd /private/tmp && python -c "…"
hazma : …/site-packages/hazma/__init__.py
_core : …/site-packages/hazma/_core.abi3.so
roundtrip: 1.5 [1. 2.]
cython via public API: [0.00385242 0.00044214]
```

Both toolchains built from the tarball, in a different interpreter than
built it, imported from outside the repo.

### Test validity (stash-proof)

With `test/parity/{cases,tolerances}.py` reverted to `HEAD` and the new
tests left in place:

```text
3 failed, 1 skipped, 625 deselected
E   AttributeError: module 'cases' has no attribute 'rust_core_kernels'
SKIPPED [1] test_parity.py:211: declared budgets in force: hazma._core is importable
```

All three predicate tests fail, and the skip line is the defect itself.
Note the revert was done with `cp`/`git checkout HEAD --`, not
`git stash`: `git stash push -- <paths>` refuses outright once any other
path is staged with `git add -N` (`error: Entry 'hazma/_core.pyi' not
uptodate`), and the tests then run against the *unmodified* tree and pass
— a stash-proof that proves nothing. Same family as
`docs/agents/environment.md`'s note that a stash round-trip un-stages a
deletion.

## Open Questions

- ~~**Will CI stay green before Task 2.2 pins the toolchain?**~~ —
  **answered on PR #55, 2026-08-08: yes, on the runner images as they
  stand today.** `.github/workflows/ci.yml` installs no Rust, so this was
  filed as a dependency rather than a prediction; all seven checks passed
  on the first run, hybrid build included, across every matrix entry
  (`Lint` 19s; ubuntu py3.10/3.11/3.12/3.13/3.14 in 19m59s / 16m29s /
  19m42s / 17m46s / 19m26s; macos py3.14 in 16m49s). So the GitHub-hosted
  images do ship a usable cargo, and `setuptools-rust` finds it with no
  configuration. **That is an observation about today's images, not a
  guarantee** — nothing in the repo pins it, and an image refresh that
  dropped Rust would take the whole matrix down at once. Task 2.2 still
  owns the explicit toolchain step, and now has a measured baseline
  (~16-20 min per entry) to compare its own runs against. `release.yml`
  does *not* run on pull requests (release/`workflow_dispatch` only), so
  its cibuildwheel job — which *will* need a toolchain inside the
  manylinux container — is untested by this PR and remains Task 2.2's.
- **The corpus-repair follow-up now has a deadline it did not have.**
  [`parity-corpus-pins-ill-conditioned-points.md`](../../../../docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md)
  was already "read it before Phase 04". Regeneration is still permitted
  after this task (that is what the `assert_no_rust_core` change buys),
  but it stops being permitted the moment the first Phase 04 kernel
  lands. Repair the corpus before that swap, not after.

## Plan Impact

**Impact Level:** Phase file patched (plus a canonical reference patched).

Two canonical edits, both in the same PR as the code they describe:

1. `phases/phase-02-rust-scaffold.md` — Task 2.1's exit criteria gained
   the "parity gate still runs in bit-equality mode" bullet. The task was
   widened beyond the plan's wording; the plan now says so.
2. `references/numerics-replacements.md` — the entry-point dispatch
   contract gained the measured live behavior it was silent about. Task
   3.5 implements from that section; leaving it describing only the
   target would have made two public-API narrowings look like
   transcription.

No ADR. Nothing here revises ADR-0001: the framework, the single-cdylib
shape, the abi3 choice and the setuptools-rust coexistence window are all
exactly as decided, and this task is their first executable form.

## Stale-state sweep

Run against this branch after every prose edit was frozen.

### Identifier sweep

```sh
rg -n 'rust_core_kernels|rust_core_available|_CORE_SCAFFOLD_NAMES|assert_no_rust_core|map_unary' \
   projects/ docs/ hazma/ test/ rust/src
```

| Hit | Verdict |
| --- | --- |
| `test/parity/{cases,tolerances,test_parity}.py`, `test/parity/README.md` | EDITED — the change itself |
| `test/parity/generate.py:307` (`assert_no_rust_core()` call) | KEPT — call site unchanged; only the predicate behind it moved |
| `rust/src/{lib,dispatch}.rs` (`map_unary`) | ADDED |
| `projects/cython-to-rust/phases/phase-02-rust-scaffold.md`, `task-notes/README.md`, `task-notes/phase-02/*` | EDITED — this task's records |
| `learnings/phase-01-parity-corpus.md:58`, `task-notes/phase-01/README.md:55` | KEPT — both phrase the guard at rule 2's level ("may never be regenerated from a tree where any kernel runs on Rust"), which my change makes *more* literally true, not less. Neither names the importability proxy. |
| `task-notes/phase-01/task-1.{1,2}-*.md` (7 hits) | KEPT — dated records of what those tasks did, including `task-1.1:268`'s `hazma._core is importable: this tree runs Rust kernels…` message string, which was the message *then*. |

```sh
rg -n 'hazma\._core|_core\.abi3|RustExtension|setuptools[-_]rust' <same roots> + build files
```

37 hits. Pre-existing project vocabulary in `PLAN.md`, `rules.md`, ADR-0001
and phase files 02–07 — all KEPT and all still accurate (the import path,
the single-cdylib shape and the coexistence window are exactly as ADR-0001
decided). New hits are `setup.py`, `pyproject.toml`, `MANIFEST.in`,
`hazma/_core.pyi` and this task's notes.

```sh
rg -n 'roundtrip' projects/ docs/ hazma/ test/ rust/src rust/build.rs
```

15 hits, all inside `rust/src/{kernels,lib}.rs` and `hazma/_core.pyi`. No
`hazma/` module and no test outside the crate calls it, which is the
"nothing imports it yet" the phase Exit Criteria require.

### Line-number citation sweep

```text
$ python scripts/agents/check_doc_citations.py --changed-vs origin/master
error: no docs to check (pass paths or --changed-vs REF)
```

That is `lessons.md` `[changed-vs-sees-only-commits]` firing exactly as
written: the work is uncommitted, so the ref diff sees zero files and
returns a not-a-pass. Re-run with explicit paths:

```text
$ python scripts/agents/check_doc_citations.py \
    projects/cython-to-rust/references/numerics-replacements.md \
    projects/cython-to-rust/phases/phase-02-rust-scaffold.md \
    projects/cython-to-rust/task-notes/phase-02/task-2.1-crate-skeleton.md \
    projects/cython-to-rust/task-notes/phase-02/README.md \
    projects/cython-to-rust/task-notes/README.md \
    test/parity/README.md
docs scanned: 6
in-repo citations checked: 17
  resolved by exact: 9
  resolved by suffix: 8
out-of-range or ambiguous: NONE
```

One AMBIGUOUS was caught and fixed en route — a bare `_pion.pyx:261` in
the reference's new subsection matched three files
(`lessons.md` `[elided-doc-paths]`); it is now the full
`hazma/spectra/_neutrino/_pion.pyx:261`.

### Forward-looking phrase sweep

```text
$ rg -n '(Task [0-9.]+ will|will be added|still pending|today: ?stub)' \
     projects/cython-to-rust/task-notes/phase-02/ \
     projects/cython-to-rust/phases/phase-02-rust-scaffold.md rust/src hazma/_core.pyi
(no matches)
```

Forward references that *do* exist are deliberate and scoped to a named
task — "Phase 04 fills it", "Task 2.2 wires that spelling into
`preflight.sh`" — rather than open-ended.

### Count sweep

| Claim location | Command | Actual | Status |
| --- | --- | --- | --- |
| "1009 passed, 13 skipped" (this note, phase README) | `scripts/agents/preflight.sh …` | `1009 passed, 13 skipped … 564.91s` | OK |
| "1022 collected, +3 on Phase 01's 1019" | `pytest --collect-only -q \| tail -1` | `1022 tests collected` | OK |
| "21 `.so` in the tree: 20 Cython + `_core.abi3.so`" (phase README, project README) | `find hazma -name '*.so' \| wc -l` | `21` | OK |
| "21 `.so`, 15 `.pyi` in the wheel" | `unzip -l dist/*.whl \| grep -c` | `21`, `15` | OK |
| "sdist 404 → 418" | `tar tzf … \| wc -l` on both | `404`, `418` | OK |
| "13 rust entries + 1 `.pyi`" | `diff` of the two listings | 14 added lines | OK |
| "2 cargo unit tests" | `cargo test --no-default-features` | `2 passed` | OK |
| "corpus: 41 cases / 1580 arrays" | `python test/parity/generate.py --check` | `41 cases / 1580 arrays` | OK |
| "54 `__len__` dispatch sites across 15 `.pyx`, 17 assert guards" (reference doc) | `rg -c --include`-free `rg -n '__len__' --include='*.pyx'` etc. | `54`, `15`, `17` | OK |
| "25 packages in `Cargo.lock`" (this note) | `grep -c '^name = ' rust/Cargo.lock` | `25` | OK — first written as "12", which was the count of *compiling* lines in cargo's output, not the resolved graph |
| "12 modified, 13 added" (this note, project README) | `git status --short` | 12 `M`, 13 `A` = 25 files | OK — the line counts from `--stat` are deliberately *not* quoted: this note is one of the 25, so any figure taken here goes stale on the next edit to it (`lessons.md` `[measurement-taken-before-the-task-ended]`, and it did: 1680 → 1709 between two passes). File counts are stable. |

### Numerical-impact statement

**No public value changes.** Two independent lines of evidence:

1. `git diff origin/master --stat -- hazma` is a single file,
   `hazma/_core.pyi`, +19 lines — a type stub. No `.py`, no `.pyx`, no
   `.pxd`, no constant, no signature. The one new runtime artifact,
   `hazma/_core.abi3.so`, is not imported by anything under `hazma/`.
2. Stronger, because it measures rather than argues: the parity corpus
   ran in **bit-equality mode** (`rtol=0`) over all 41 consumed entry
   points — 626 blocks, 1,580 arrays, 179,695 pinned values — and passed.
   That is only available because this task fixed the mode switch it
   would otherwise have broken; had it shipped without that fix, this row
   could only have said "within 1e-8".

No grid evaluation is therefore reported separately: the corpus *is* the
grid, and it is a stricter one than any ad-hoc sweep.

The project's `version_bump: major` is unaffected — it is driven by the
Phase 00 API removals, and nothing here removes or renames a public name.

### Exit Criteria → evidence mapping

| Criterion | Evidence |
| --- | --- |
| `rust/Cargo.toml` edition 2024, pyo3 `abi3-py310` + `extension-module`, numpy | `rust/Cargo.toml`; `cargo build` green; `_core.abi3.so` produced |
| `#[pymodule] _core` with five empty submodules + one round-trip fn | `rust/src/lib.rs`; `sys.modules` shows all five under `hazma._core.*`; `roundtrip` present |
| `setup.py` gains `RustExtension("hazma._core", …)` | `setup.py`; `uv pip install -e .` builds both toolchains in one pass |
| `python -c "import hazma._core"` works | Verification §abi3 and §Packaging (in-tree, wheel, and sdist installs) |
| `hazma/_core.pyi` started; py.typed unaffected | new file; `git ls-files \| grep -c py.typed` → 0, so nothing to affect |
| Parity gate still in bit-equality mode *(added)* | `provenance` → `exact: True`; `test_running_on_the_capturing_tree` passes; 13 skips unchanged; three new tests, stash-proved |

### Task-note self-consistency

`**Status:** Complete` matches the phase README row and the project
README's Phases row (`In Progress — Task 2.1 complete`). Every file named
under §Files Changed appears in `git status --short` (11 modified, 12
added, 1 of them this note). Every symbol named in §Findings and
§Decisions — `rust_core_kernels`, `_CORE_SCAFFOLD_NAMES`, `map_unary`,
`add_submodule`, `roundtrip`, `RustExtension` — appears in the diff.

## Handoff to Next Task

**Read first:** `../README.md` (this phase's working memory), then
`../../phases/phase-02-rust-scaffold.md`. Tasks 2.2 and 2.3 both depend
only on 2.1 and share no files, so they can run in either order or in
parallel.

**Safe to assume:**

- The crate builds, imports and ships. `uv pip install -e .` produces
  `hazma/_core.abi3.so` beside the 20 Cython extensions; the wheel and a
  from-source sdist install both carry it.
- The three cargo gates Task 2.2 must wire into `preflight.sh` are
  `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings` and
  **`cargo test --no-default-features`**. The last flag is not optional —
  without it the test harness cannot link, because `extension-module`
  leaves CPython's symbols undefined.
- The dev loop, answering this phase README's old open question: a `.rs`
  edit needs `uv pip install -e .` to reach Python. `cargo build` alone
  updates `rust/target/` and leaves the installed
  `hazma/_core.abi3.so` stale — the Cython trap in
  `docs/agents/environment.md`, one language over. Task 2.2 writes that
  into `AGENTS.md` and `docs/agents/`.
- `dispatch::map_unary` is the single implementation of the dispatch
  contract, and `hazma._core.roundtrip` exercises every branch of it.
  *(Task 3.5, 2026-08-11: three helpers over one classification —
  `map_unary`, `map_flavors`, `require_vector`.)*
  Task 2.3's suite is written against `roundtrip` and the table in
  §Verification is the behavior to assert.
- The parity gate is in bit-equality mode and stays there until a real
  kernel lands. **Do not re-key it on `rust_core_available()`.**

**Still risky / open:**

- ~~CI carries no Rust toolchain step (Task 2.2). See Open Questions —
  a red matrix on this PR is that task's trigger, not a mystery.~~ —
  **closed by Task 2.2 on 2026-08-08**: both workflows now install one,
  and CI grew a `rust` job for the cargo gates.
- The four measured divergences between the reference's dispatch contract
  and the live Cython are now written down but not decided. Task 3.5
  decides each; two are public-API narrowings if taken by default.
- The corpus repair
  ([`parity-corpus-pins-ill-conditioned-points.md`](../../../../docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md))
  is still possible today and impossible from the first Phase 04 swap.
