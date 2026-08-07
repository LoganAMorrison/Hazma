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
| [`test_parity.py`](test_parity.py) | The gate — re-evaluates every entry point and compares against `data/`. |
| `data/*.npz` | One file per entry point. Reference arrays, stored exactly as the library returned them. |
| `data/manifest.json` | Provenance and per-array hashes. |

## Commands

Run the gate. One test per corpus block (623 of them) plus three guards;
takes around five minutes, nearly all of it the nested adaptive
quadrature in the rho and mediator-spectrum kernels:

```bash
pytest test/parity
```

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

## When *not* to regenerate

`projects/cython-to-rust/rules.md` rule 2: **the corpus is generated
only from pre-port Cython.** Regenerating it from a tree in which any
kernel already runs on Rust would pin the port against itself and the
gate would pass vacuously. `generate.py` enforces this — it refuses to
run once `hazma._core` is importable — but the rule matters beyond what
the check can see. If a swap changes a number, the fix is a declared
tolerance in the parity suite plus an entry in the project's numerical
record, never a regenerated array.

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
captured on (exact, always); the values themselves against the budget
[`tolerances.py`](tolerances.py) selects; and the manifest's `raises`
records, **replayed** — the entry point must still raise the same type at
the same argument, and must not raise anywhere new. Evaluation goes
through `generate.evaluate_block`, the same function that produced the
stored numbers, so the kernel is the only thing that can differ.

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
