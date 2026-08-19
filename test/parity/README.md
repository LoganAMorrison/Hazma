# Golden parity corpus

Pinned reference values for every compiled entry point that Hazma's
Python layer consumes, captured from the pre-port Cython. The corpus is
the gate the Cython-to-Rust port swaps against: a kernel is repointed at
the Rust implementation only once it reproduces these arrays within that
function's budget.

| File | What it is |
| --- | --- |
| [`cases.py`](cases.py) | The specification — which entry points, on which grids, invoked how. No numbers. |
| [`generate.py`](generate.py) | Evaluates the specification and writes `data/`; also verifies `data/` against its manifest. |
| [`tolerances.py`](tolerances.py) | How far a replacement implementation may move each entry point, and why that far. |
| [`stability.py`](stability.py) | Which pinned values are rounding residue rather than physics, and are therefore skipped. |
| [`reference.py`](reference.py) | Arbitrary-precision copies of the four cancellation-prone kernels. Used only to rebuild the mask. |
| [`test_parity.py`](test_parity.py) | The gate — re-evaluates every entry point and compares against `data/`. |
| `data/*.npz` | One file per entry point. Reference arrays, stored exactly as the library returned them. |
| `data/manifest.json` | Provenance and per-array hashes. |
| `data/unpinnable.json` | The mask `stability.py` builds: 494 stored positions that assert nothing. |

## Commands

Run the gate. One test per corpus block (623 of them) plus 15 guards;
around five minutes of single-core work, nearly all of it the nested
adaptive quadrature in the rho and mediator-spectrum kernels. The
pytest-xdist `addopts` in `pyproject.toml` spread it across cores, so
the wall-clock is that cost divided by the machine:

```bash
pytest test/parity
```

A bare `pytest` runs it too — `pyproject.toml`'s `testpaths` is
`["hazma", "test"]` (cython-to-rust Task 1.3) — and every CI matrix
entry pays it. Between PR #52 and 2026-08-18 the Linux entries passed
`--ignore=test/parity` instead, because the corpus did not survive a
change of libm; `stability.py`, `tolerances.PLATFORM_EXACT_RTOL`,
`tolerances.PLATFORM_SPECFUN_RTOL` and `tolerances.zero_floor` are what
took that scoping out again. That work
is the standing price of the gate. Note that
the suite needs the extensions built **inside the repository**:
`cases.assert_module_is_repo_tree` refuses a `hazma` resolving anywhere
else, so `pip install -e .`, not `pip install .`.

Verify the committed data is intact and complete. Imports no kernel and
evaluates nothing, so it is fast and works on an unbuilt tree:

```bash
python test/parity/generate.py --check
```

Regenerate everything. Takes a few minutes, for the same reason the gate
does:

```bash
python test/parity/generate.py
```

Generation is deterministic: two runs on the same tree and environment
produce a byte-identical manifest.

Rebuild the unpinnable-point mask. Reads the committed corpus and
`reference.py` and touches no kernel, so it needs no build — but it does
need `mpmath` from the `dev` group:

```bash
python test/parity/stability.py --regenerate
```

## When *not* to regenerate

`projects/cython-to-rust/rules.md` rule 2: **the corpus is generated
only from pre-port Cython.** Regenerating it from a tree in which any
kernel already runs on Rust would pin the port against itself and the
gate would pass vacuously. `generate.py` enforces this — it refuses to
run once `hazma._core` *serves* a kernel — but the rule matters beyond
what the check can see. The distinction is load-bearing from
cython-to-rust Phase 02 on: the Rust extension exists in every build from
Task 2.1, while every value still comes from Cython until the first
Phase 04 swap, so keying on importability alone would both block a
legitimate corpus repair and drop the runner out of bit-equality mode two
phases early. `cases.rust_core_kernels()` is the predicate; it returns the
kernels the extension actually exposes, ignoring the scaffold's
`roundtrip` probe and the test-only submodules listed in
`cases._CORE_TEST_ONLY_MODULES` (today five: `hazma._core.special`, the
Phase 03 Task 3.2 specfun shim that `test/test_core_special.py` sweeps
against scipy; `hazma._core.quad`, the Task 3.3 QUADPACK port that
`test/test_core_quad.py` compares against `scipy.integrate.quad`;
`hazma._core.interp` and `hazma._core.boost`, the Task 3.4 interpolation
and boost foundation, swept by `test/test_core_interp.py` against
`np.interp` and by `test/test_core_boost.py` against the Cython twin
itself through `hazma._utils.boost.__pyx_capi__`; and
`hazma._core.dispatch`, the Task 3.5 argument-and-error layer, whose
messages `test/test_core_dispatch.py` compares byte for byte against the
strings it extracts from the `.pyx` sources). That
second exemption is held honest by
`test_test_only_core_submodules_have_no_importer`, which fails the moment
anything under `hazma/` imports one of them — at which point it is a
served kernel and belongs back in the count. If a swap changes a number,
the fix is a declared tolerance in the parity suite plus an entry in the
project's numerical record, never a regenerated array.

## What the corpus pins

**Coverage** is checked, not asserted: `assert_full_coverage` walks the
surviving `.pyx` for top-level `def`s and fails if any lacks a case, or
if a case names a `def` that no longer exists. The two `sigma_xx_to_all`
exports are the only exclusions, and `assert_unconsumed_exports_are_unimported`
re-derives at generation time that nothing imports them.

**Edge behavior is part of the contract.** Grids deliberately run past
thresholds and endpoints, and values are stored exactly as returned —
including `nan`, `inf` and negative entries. Where an entry point
*raises* rather than returning (`sigma_xx_to_v_to_pipi` and
`sigma_xx_to_v_to_pi0v` raise `TypeError` exactly at `e_cm = 2 mx`), the
stored value is `nan` and the manifest's `raises` block records the
index, the argument and the exception type.

## What the gate compares

Per block: the grid `cases.py` produces against the grid the values were
captured on (bit-exact on the capturing tree, one ulp elsewhere — see
`tolerances.abscissa_budget`); the values themselves against the budget
[`tolerances.py`](tolerances.py) selects; and the manifest's `raises`
records, **replayed** — the entry point must still raise the same type at
the same argument, and must not raise anywhere new. Evaluation goes
through `generate.evaluate_block`, the same function that produced the
stored numbers, so the kernel is the only thing that can differ.

Two carve-outs, both narrow and both declared:

- **Unpinnable positions are dropped** before the value comparison. Four
  scalar elastic cross sections evaluate a difference of two `atan`s that
  cancels every significant bit near `e_cm = 2 mx` and throughout
  `closed_resonance`, so what the corpus stored there is one platform's
  rounding. [`stability.py`](stability.py) names them — 494 positions
  across 12 blocks, established against `reference.py` rather than
  guessed — and says why the obvious cheaper detectors do not work.
- **Four declared stored zeros get an absolute floor** rather than a
  relative one, at `tolerances.ZERO_FLOOR_FRACTION` of the array's own
  median non-zero magnitude. A quadrature whose integrand sits at *its*
  threshold lands on exactly `0.0` on one libm and on 2.6e-13 on another;
  with `atol` at zero that reads as an infinite relative error.
  `stability.PORTABILITY_ZEROS` names the four — every other stored zero,
  66,836 of them, keeps the exact-zero contract.

The budget depends on which tree you are on. When the kernel digest, the
toolchain and the numerics libraries all match what the manifest records,
every case is held to **bit-equality** — the corpus pins that
implementation against itself, so any difference is a regression. Once
anything diverges (a ported kernel, a different platform, a newer scipy)
the declared per-function budgets in `tolerances.py` take over, which is
the situation they were written for. `pytest test/parity -rs` names the
mode: a skipped `test_running_on_the_capturing_tree` means budget mode,
and its reason says what differed.

**Provenance.** The manifest carries the generating commit and whether
that tree was dirty, the versions of numpy/scipy/cython/hazma, and
`kernel_digest` — a hash over every `.pyx`, `.pxd` and photon data file
in the tree. The digest, not the SHA, is what identifies the kernels
that produced the numbers: the generating commit does not exist yet when
the data is written, so `dirty` is normally `true`.

See [`cases.py`](cases.py)'s module docstring for the grid design and
`projects/cython-to-rust/phases/phase-01-parity-corpus.md` for where
this fits in the port.
