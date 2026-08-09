# Phase 02 Learnings: Rust scaffold

Synthesized at phase close (2026-08-09) from the three task notes
([2.1](../task-notes/phase-02/task-2.1-crate-skeleton.md),
[2.2](../task-notes/phase-02/task-2.2-ci-devloop.md),
[2.3](../task-notes/phase-02/task-2.3-plumbing-test.md)) and
[`../task-notes/phase-02/README.md`](../task-notes/phase-02/README.md).
Read this instead of the notes; the notes are history.

## 1. Implementation Reality Check

The phase delivered what it promised — `hazma._core` at its final import
path, abi3, built beside the 20 Cython extensions by one
`pip install -e .`, gated in CI and in `preflight.sh`, with the dispatch
contract pinned from Python. No ADR was needed: ADR-0001 already fixed
the framework choice and this phase is its first executable form.

What the plan did **not** anticipate is that all three tasks had to widen
their own exit criteria, and each widening was the same shape — *a
scaffold that is merely present is not the thing the criterion meant*:

- **Task 2.1**: the crate's mere existence switched the parity gate out
  of bit-equality mode, because `tolerances.provenance` keyed on
  "`hazma._core` is importable". Shipped as written, Phases 02–03 would
  have run against 1e-8 budgets with nothing turning red. The predicate
  now asks whether a kernel is *served* (`cases.rust_core_kernels()`).
- **Task 2.2**: two gates that existed but could not fail anything — the
  cargo checks lived only in local discipline, and the wheel/abi3
  criterion lived in a workflow with no pull-request trigger. Both became
  things that fail a job.
- **Task 2.3**: `roundtrip` advertised `(x, /)` while accepting `x=`.
  A template that misdescribes its own signature propagates the error to
  every kernel that copies it.

The generalization, and the one thing to carry into Phases 03–07: **in
this project, "it exists" and "it is load-bearing" are different claims,
and the gap between them is always silent.** Every criterion in the
remaining phases deserves the question "what turns red if this is
wrong?" before it is called done. Three of the three tasks here found a
`[gate-disabled-stays-green]` instance by asking it.

## 2. Critical Context for Future Work

- **`hazma._core` is the final import path, from now on.** One cdylib,
  five submodules (`photon`, `positron`, `neutrino`, `scalar_mediator`,
  `vector_mediator`), each registered in `sys.modules` under its
  fully-qualified name so both `from hazma._core import photon` and
  `from hazma._core.photon import <kernel>` work. Public Python import
  paths never change; wrappers re-export behind the existing names
  (`../rules.md` Rust rule 2).
- **`dispatch::map_unary` is the single implementation of the
  entry-point dispatch contract.** Phases 03–06 call it rather than
  touching PyO3 inside a kernel (`../rules.md` Rust rule 3). Its measured
  behavior, now pinned by `test/test_core_dispatch.py`:
  - `float` / NumPy scalar / 0-d `float64` array → Python `float`;
  - 1-D `float64` array → a **fresh, non-aliasing** 1-D `float64` array;
  - anything else → `ValueError`, prefixed with the `quantity` string the
    kernel passes.
  - **Rank is checked before dtype** — a 2-D int64 array reports the
    dimension error.
  - **A 0-d array still enforces dtype**, where a Python `int` does not:
    the 0-d path is inside the array branch behind the typed view.
  - **Non-`float` NumPy scalars are accepted** (`np.float32`, `np.int64`,
    `np.uint8`, `np.bool_`) via the `extract::<f64>` arm.
- **The `PyFloat` fast path is ordered first on purpose.** The `numpy`
  crate *panics* rather than raising when NumPy cannot be imported —
  `cast::<PyUntypedArray>` reaches for the array-API capsule. Every
  kernel inherits that ordering; do not reorder it for tidiness.
- **Kernels are PyO3-free.** `src/kernels.rs` is plain
  `fn(f64, …) -> f64`, which is what keeps `cargo test` GIL-free and
  interpreter-free. The PyO3 layer is `dispatch.rs` plus registration.
- **`pip install -e .` is the only thing that republishes a `.rs` edit.**
  `cargo build` and `cargo test` work out of `rust/target/`, which
  nothing Python imports. Iterate with
  `cargo test --manifest-path rust/Cargo.toml --no-default-features`,
  reinstall before quoting any Python-side number, confirm with
  `python -c "import hazma._core; print(hazma._core.__file__)"`.
- **The parity gate is in bit-equality mode and stays there** until the
  first Phase 04 kernel is served. Do not re-key it on
  `rust_core_available()`. Both `tolerances.provenance` and
  `assert_no_rust_core` flip permanently at that first swap, so the
  [ill-conditioned-points corpus repair](../../../docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md)
  must land **before** it.

## 3. Quirk Log & Edge Cases

- **PyO3 0.29.2 renamed `Bound::downcast` to `Bound::cast`.** Every
  tutorial still says `downcast`.
- **`cargo build` needs three things setuptools-rust supplies for you:**
  an explicit empty `[workspace]` in `rust/Cargo.toml` (a stray
  `Cargo.toml` anywhere above the checkout captures the crate — a
  home-directory one did), macOS's `-undefined dynamic_lookup` (emitted
  from `rust/build.rs` as a cdylib-only link arg; PyO3 0.29 does not add
  it), and `extension-module` *off* for `cargo test` — it is a default-on
  optional feature and the harness will not link with it.
- **`text_signature` is a claim, not a constraint.** It sets what
  `inspect.signature` reports and nothing else; enforcing positional-only
  needs `#[pyo3(signature = (x, /))]`. The Cython entry points being
  replaced are `def` functions and accept keywords, so keyword-accepting
  is the correct target and a `/` in a `text_signature` is a latent
  public-API narrowing.
- **A "fresh" array from the `numpy` crate does not own its data.**
  `owndata` is `False` and `.base` is a `PySliceContainer` wrapping the
  Rust `Vec`. Assert non-aliasing (`is not`, `.base is not`,
  `np.shares_memory`, mutate-and-check), never `owndata` — that assertion
  is red on correct code.
- **The two wheel platforms need two different toolchains.**
  cibuildwheel builds macOS wheels on the runner (a host
  `dtolnay/rust-toolchain` step covers them) and Linux wheels inside a
  manylinux container that cannot see the host (`CIBW_BEFORE_ALL_LINUX`
  installs rustup in the container, `CIBW_ENVIRONMENT_LINUX` puts it on
  `PATH`). Same shape as Phase 00's `MANIFEST.in`-vs-wheel lesson: two
  artifacts, two mechanisms, and fixing one has never fixed the other.
  **Phase 07 Task 7.1 rewrites this job for maturin and inherits it.**
- **`release.yml` has no pull-request trigger**, so nothing about it is
  verifiable from a PR check however green the PR is. Closing one of its
  criteria takes an explicit `gh workflow run release.yml --ref <branch>`
  (safe because `publish` is gated on
  `github.event_name == 'release'` — check that gate before dispatching).
  Measured cost ~17 min for both platforms.
- **The live Cython dispatch is not the contract the reference
  described** — four measured divergences, now written into
  `../references/numerics-replacements.md`: a 0-d array raises rather
  than taking the scalar path; a Python list is accepted; shape errors
  are `AssertionError` (and vanish under `python -O`), not `ValueError`;
  and `hazma/spectra/_neutrino/_muon.pyx:205` says "Photon energies".
  **Task 3.5 decides each one**, and two of the four are user-visible
  narrowings if transcribed from the design instead of the code.

## 4. Test Infrastructure State

- **`test/test_core_dispatch.py` is the template every kernel swap
  copies** (54 tests, 0.27s, platform-independent). Copy it per kernel:
  swap `roundtrip` for the kernel and `QUANTITY` for the wording that
  kernel passes to `map_unary`, keep every test, and add the kernel's
  *numerical* tests beside them rather than merged in — plumbing failures
  and physics failures should not need the same debugging. The copy
  instructions live in the module docstring so they travel with the file.
- **It deliberately does not re-pin values against Cython.** The parity
  corpus does that at bit-equality across all 41 entry points; a second,
  looser numerical gate would only be one more thing to keep in sync.
- **No `importorskip` on `hazma._core`.** From this phase on the
  extension is in every build, so a missing `_core` is a build failure
  that must fail loudly.
- **The three cargo gates run themselves, in two places**:
  `scripts/agents/preflight.sh` (gates 4–6, ahead of pytest because they
  cost seconds and the bare suite costs minutes) and CI's `rust` job.
  Exact spellings are in the phase file's Task 2.2 exit criteria; note
  `--no-default-features` on the test one and `--manifest-path
  rust/Cargo.toml` so they run from the repo root. Absence rules: no
  `rust/` → SKIP, `rust/` but no `cargo` → WARN.
- **Suite size across the phase**, all on the corpus's capturing
  environment (CPython 3.12.12, macOS/arm64), parity in bit-equality mode
  throughout: 1006/13 at Phase 01 close → 1009/13 (Task 2.1, +3
  served-kernel predicate tests) → 1009/13 (Task 2.2, byte-identical,
  which is what showed no test outcome moved) → 1063/13 (Task 2.3, +54).
  The skip count never moved; that is the tell that the parity gate
  stayed in exact mode — budget mode is reported as a *skip* on
  `test_running_on_the_capturing_tree`, so a budget-mode run would have
  shown 14.
- **Mutation campaigns are cheap here and worth running.** Task 2.3 ran
  six mutations of `dispatch.rs`/`lib.rs`, each rebuilt with
  `uv pip install -e .` and re-run (~45s per cycle), and every one was
  caught by the test whose name claimed it. For a module whose assertions
  all pass against unmodified code by construction, that is the only real
  evidence it tests anything.

## 5. Follow-on seeds

None new. Phase 02 opened no follow-up that is not already filed, and
closed two of its own open questions in-phase (CI's unpinned toolchain,
`release.yml` unexercised). The live items this phase hands forward all
predate it and are already tracked:

- [parity corpus pins ill-conditioned points](../../../docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md)
  — **read before Phase 04**, and land before the first swap, because
  that swap flips the corpus out of repairable, bit-equality mode
  permanently.
- [model spectra reject scalar energies](../../../docs/followups/todo/model-spectra-reject-scalar-energies.md)
  — the model-level half of dispatch divergences 1 and 2; resolving it by
  normalizing at the public boundary also settles the 0-d case, and
  deciding the two separately risks two different answers.
- [positron spectrum `nan` at the legacy electron mass](../../../docs/followups/todo/positron-spectrum-nan-at-legacy-electron-mass.md)
  — ripens before Phases 05/06.
- [sdist ships generated C and docs](../../../docs/followups/todo/sdist-ships-generated-c-and-docs.md)
  — time-boxed to before Phase 07 Task 7.1, since maturin reads neither
  `MANIFEST.in` nor `[tool.setuptools]`.
