# Corrected-value oracles from the Cython twins

Reference arrays for the four numerical defects that still had a Cython
twin when this capture was taken, evaluated from a **patched** copy of
that twin on the corpus's own grids. The parity corpus next door pins
what hazma 2.1.0 shipped, wrong values included, and is never rewritten
([`projects/parity-pinned-defect-repair/rules.md`](../../../projects/parity-pinned-defect-repair/rules.md)
rule 1); these arrays are what a repair is checked *against* instead.

They exist on a deadline that has already partly passed. `cython-to-rust`
deletes the twins in three waves — Task 4.6, Tasks 6.2/6.3, then Task 6.4
— and after the last one the only source of a corrected value is the
repaired Rust, which pins the port against its own answer.
[`projects/parity-pinned-defect-repair/references/corpus-repinning.md`](../../../projects/parity-pinned-defect-repair/references/corpus-repinning.md)
is the spec this implements, and
[`task-notes/task-2-cython-oracles.md`](../../../projects/parity-pinned-defect-repair/task-notes/task-2-cython-oracles.md)
records what this particular run measured.

| File | What it is |
| --- | --- |
| [`defects.py`](defects.py) | The four Group A defects: which `.pyx` each patches and which corpus cases it reaches. |
| [`entry_points.py`](entry_points.py) | Where each case's *Cython* value comes from now that the port has moved or deleted most entry points. |
| [`capture.py`](capture.py) | Drives the capture, assembles the manifest, and verifies the committed arrays. |
| [`patches/*.patch`](patches) | One unified diff per defect. This is the repair, stated exactly, for the repair tasks to match. |
| `data/*.npz` | One file per defect. Only `values` and `scalar_values` — the grids are inputs and are already committed next door. |
| `data/manifest.json` | Provenance, per-array hashes, and the measured diff against the corpus. |

## Commands

Verify the committed arrays against the manifest. Imports no kernel and
evaluates nothing, so it works on an unbuilt tree — the same contract
`python test/parity/generate.py --check` has:

```bash
python test/parity/oracles/capture.py --check
```

The gate that runs in `pytest` is
[`../test_oracles.py`](../test_oracles.py): it runs that check, holds the
oracle manifest's platform against the corpus manifest's, and requires
every captured key to name a real corpus array of the same shape.

## Recapturing

Only needed if a patch changes. It cannot be done at all once
`cython-to-rust` Task 6.4 lands — that is the whole point of the arrays
being committed.

Everything below mutates tracked sources. Snapshot them outside the tree
first, `cmp` before each step, and verify every restore:
`docs/agents/lessons.md` `[mutation-harness-poisons-its-own-baseline]` is
this exact loop, and it has already cost `cython-to-rust` twice.

**1. Restore the sources the port deleted.** Nine of the twenty cases run
through `.pyx` that no longer exist; `defects.RESTORED_SOURCES` names each
one and the revision it comes from, and `capture.py` refuses to run if a
restored file's bytes differ from that revision's. Add the restored
modules to `setup.py`'s `["spectra", "_photon"]` extension list.

```bash
git show 0954e5a^:hazma/spectra/_photon/_eta.pyx > hazma/spectra/_photon/_eta.pyx
```

**2. Prove the harness before trusting it.** Build unpatched
(`pip install -e .`, not `cargo build` — see
[`AGENTS.md`](../../../AGENTS.md)) and require every case to come back
bit-for-bit identical to the corpus:

```bash
python test/parity/oracles/capture.py --baseline
```

This is the load-bearing step. Three cases are driven through a shim that
rebuilds a `def` the port deleted and nine through resurrected modules;
bit-equality against the committed corpus is what says both are faithful,
rather than an argument that they ought to be.

**3. Per defect: patch, rebuild, capture, revert.** One patch at a time,
so a captured array carries one repair rather than a combination of them.
`capture.py` re-derives the working tree's diff and refuses to capture
unless it is byte-identical to the committed patch.

```bash
git apply test/parity/oracles/patches/A1-boost-integral-window.patch
```

**4. Assemble, then revert everything.** `--assemble` merges the
per-defect parts and measures each capture against the corpus. Then
remove the restored sources, restore `setup.py`, rebuild, and confirm
`git diff -- hazma` is empty: this directory ships no library behavior.

```bash
python test/parity/oracles/capture.py --assemble
```

## What these arrays are not

They are not a proof that the physics is right. Two independent
implementations agreeing on a wrong number is a real failure mode, and it
is why every repair task's gate names a physics invariant — a yield, a
unit, an endpoint, a normalization integral — alongside its comparison
against the arrays here
([`rules.md`](../../../projects/parity-pinned-defect-repair/rules.md)
rule 4).

They are also not a statement about any platform but the one they were
captured on. The corpus scopes its `EXACT` budget class the same way and
for the same reason, and `test_oracles.py` reads both platforms out of the
two manifests rather than probing for either
(`docs/agents/lessons.md` `[platform-scoped-oracle-asserted-globally]`).
