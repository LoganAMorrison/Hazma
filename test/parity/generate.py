#!/usr/bin/env python
"""Generate and verify Hazma's golden parity corpus.

The corpus pins the output of every compiled entry point Hazma's Python
layer consumes, captured from the pre-port Cython, so the Cython-to-Rust
port has something to swap against. `cases` holds the specification —
which entry points, on which grids, called how — and this module turns it
into ``data/*.npz`` plus a ``data/manifest.json`` recording where the
numbers came from.

Usage
-----
Regenerate everything (refuses to run once ``hazma._core`` exists, per
``projects/cython-to-rust/rules.md`` rule 2)::

    python test/parity/generate.py

Verify the committed data still hashes to what the manifest says --
a pure integrity check that imports no kernel and evaluates nothing::

    python test/parity/generate.py --check

Provenance
----------
The manifest records the git SHA of the tree that generated the data,
whether that tree was dirty, and the versions of every package whose
numerics can move a value. Because the generating commit is by
construction not yet made when the data is written, the SHA alone is not
enough to identify the kernels that produced it; the manifest therefore
also carries ``kernel_digest``, a hash over every ``.pyx``, ``.pxd`` and
photon data file in the tree. That digest, not the SHA, is what a later
reader should compare when asking "was this corpus captured from these
kernels?".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import warnings
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cases as corpus  # (imported after the sys.path entry above)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"
REPO_ROOT = corpus.REPO_ROOT

#: Bumped when the on-disk layout changes in a way a reader must notice.
SCHEMA_VERSION = 1

#: Guardrail from the phase file: the corpus is a test fixture, not a
#: dataset. Generation fails rather than quietly committing something huge.
MAX_TOTAL_BYTES = 10 * 1024 * 1024


# ---------------------------------------------------------------------------
# ---- Hashing and provenance -----------------------------------------------
# ---------------------------------------------------------------------------


def hash_array(array: np.ndarray) -> dict[str, Any]:
    """Describe an array by shape, dtype and a hash of its exact bytes.

    ``dtype.str`` carries byte order (``<f8``), so a mismatch between the
    generating and checking platforms shows up as a dtype difference
    rather than as a spurious hash failure. NaNs hash fine here because
    the bytes, not the values, are hashed — which is the point, since NaN
    at a kinematic edge is part of the pinned contract.
    """
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def git_provenance() -> dict[str, Any]:
    """The generating commit, and whether the tree was clean at the time."""

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        return {
            "sha": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        raise RuntimeError(
            "cannot record the generating git SHA, which rules.md rule 2 "
            "requires; regenerate the corpus from inside a git checkout"
        ) from err


def kernel_digest() -> dict[str, Any]:
    """Hash the compiled-layer sources and the data they read at import.

    Covers every ``.pyx`` and ``.pxd`` under ``hazma/`` plus the photon
    CSVs the tabulated kernels load. Two trees with the same digest
    produce the same corpus; a differing digest means the numbers could
    have moved, whatever the git SHA says.
    """
    package = REPO_ROOT / "hazma"
    paths = sorted(
        set(package.rglob("*.pyx"))
        | set(package.rglob("*.pxd"))
        | set(corpus.PHOTON_DATA_DIR.glob("*.csv"))
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO_ROOT)).encode())
        digest.update(path.read_bytes())
    return {"sha256": digest.hexdigest(), "n_files": len(paths)}


def cython_version() -> str | None:
    """The Cython that generated the extensions the values came from.

    Not `metadata.version("cython")`: the build backend runs in an
    isolated environment, so Cython is normally absent from the
    environment that imports hazma and the metadata lookup returns
    nothing. The generated ``.c`` carries the real answer on its first
    line (``/* Generated by Cython 3.2.9 */``), which is the version that
    actually shaped the numbers.
    """
    for path in sorted((REPO_ROOT / "hazma").rglob("*.c")):
        with path.open() as handle:
            match = re.match(r"/\* Generated by Cython (\S+) \*/", handle.readline())
        if match:
            return match.group(1)
    return None


def environment() -> dict[str, Any]:
    """Versions of everything whose numerics can move a captured value."""
    versions: dict[str, Any] = {}
    for name in ("numpy", "scipy", "hazma"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    versions["cython"] = cython_version()
    versions["python"] = platform.python_version()
    versions["platform"] = platform.platform()
    versions["machine"] = platform.machine()
    return versions


# ---------------------------------------------------------------------------
# ---- Generation -----------------------------------------------------------
# ---------------------------------------------------------------------------


def _sweep_pointwise(
    evaluate: Callable[[int], Any], grid: np.ndarray
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Evaluate `evaluate` at each grid point, recording what raises.

    Used when a batched call raised: a single bad point takes the whole
    array down with it, so the surviving values have to be recovered
    individually.

    Raising points are filled with ``nan`` shaped like a successful
    result — which matters for the neutrino entry points, whose result
    per point is one value per flavor rather than a scalar.

    Parameters
    ----------
    evaluate : callable
        ``index -> result`` for one grid point.
    grid : numpy.ndarray
        The swept argument, used only to record the offending value.

    Returns
    -------
    results : list of numpy.ndarray
        One per grid point, all the same shape.
    raises : list of dict
        ``{"index", "argument", "type"}`` for each point that raised. The
        exception *message* is deliberately not recorded — Cython rewords
        its errors between releases, and it is the fact of the raise and
        its type that a port has to reproduce.
    """
    results: list[np.ndarray | None] = []
    raises: list[dict[str, Any]] = []
    for index, x in enumerate(grid):
        try:
            results.append(np.asarray(evaluate(index), dtype=np.float64))
        except Exception as err:  # noqa: BLE001 - the type is what we record
            raises.append(
                {"index": index, "argument": float(x), "type": type(err).__name__}
            )
            results.append(None)

    shape = next((r.shape for r in results if r is not None), ())
    filled = [np.full(shape, np.nan) if r is None else r for r in results]
    return filled, raises


def evaluate_block(
    fn: corpus.EntryPoint, block: corpus.Block
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Capture one block: the grid, the array-path values, the scalar probe.

    Values are stored exactly as returned — ``nan``, ``inf`` and negative
    entries included — because the kinematic-edge behavior is part of
    what the corpus pins. When an entry point *raises* at a grid point
    (which happens: ``sigma_xx_to_v_to_pipi`` raises ``TypeError``
    exactly at ``e_cm = 2 mx``), that is pinned too: ``nan`` goes into
    the stored array and the manifest records the index, the argument and
    the exception type.

    Returns
    -------
    arrays : dict
        ``grid`` and ``values`` always; ``scalar_grid`` and
        ``scalar_values`` only when the entry point has a scalar branch.
        Neutrino entry points return one row per flavor, so their
        ``values`` is ``(3, N)`` and their ``scalar_values`` is ``(K, 3)``.
    notes : dict
        Any raises observed, keyed by the array they belong to. Empty
        when the block evaluated cleanly throughout.
    """
    notes: dict[str, Any] = {}
    with warnings.catch_warnings():
        # scipy raises IntegrationWarning on the deliberately pathological
        # grid points (below threshold, at an endpoint). That the warning
        # fires is not part of the contract; the value it returns is, and
        # that is what gets stored.
        warnings.simplefilter("ignore")

        grid = np.asarray(block.grid, dtype=np.float64)
        arrays = {"grid": grid}
        try:
            arrays["values"] = np.asarray(
                block.array_call(fn, block.grid), dtype=np.float64
            )
        except Exception:  # noqa: BLE001 - recovered point-by-point below
            # Each point still goes through the batched call, as a
            # length-1 array, so it travels the same code path.
            per_point, raises = _sweep_pointwise(
                lambda i: block.array_call(fn, block.grid[i : i + 1]), grid
            )
            arrays["values"] = np.concatenate(per_point, axis=-1)
            notes["values"] = raises

        probe = block.scalar_probe
        if probe.size:
            per_point, raises = _sweep_pointwise(
                lambda i, p=probe: block.scalar_call(fn, float(p[i])), probe
            )
            arrays["scalar_grid"] = probe
            arrays["scalar_values"] = np.stack(per_point)
            if raises:
                notes["scalar_values"] = raises
    return arrays, notes


def generate() -> int:
    """Regenerate every ``.npz`` and the manifest. Returns a process exit code."""
    corpus.assert_no_rust_core()
    corpus.assert_unconsumed_exports_are_unimported()

    cases = corpus.build_cases()
    corpus.assert_full_coverage(cases)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for stale in DATA_DIR.glob("*.npz"):
        stale.unlink()

    manifest: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "generated_by": "test/parity/generate.py",
        "git": git_provenance(),
        "kernel_digest": kernel_digest(),
        "environment": environment(),
        "cases": {},
    }

    total_blocks = 0
    for name, case in cases.items():
        fn = case.resolve()
        payload: dict[str, np.ndarray] = {}
        blocks: list[dict[str, Any]] = []
        for index, block in enumerate(case.blocks):
            arrays, notes = evaluate_block(fn, block)
            keys = {}
            for suffix, array in arrays.items():
                key = f"b{index}_{suffix}"
                payload[key] = array
                keys[suffix] = {"key": key, **hash_array(array)}
            entry: dict[str, Any] = {
                "label": block.label,
                "params": block.params,
                "arrays": keys,
            }
            if notes:
                entry["raises"] = notes
            blocks.append(entry)
        total_blocks += len(blocks)

        filename = f"{name}.npz"
        np.savez_compressed(DATA_DIR / filename, **payload)
        manifest["cases"][name] = {
            "entry_point": case.entry_point,
            "summary": case.summary,
            "file": filename,
            "blocks": blocks,
        }

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    total_bytes = sum(path.stat().st_size for path in DATA_DIR.iterdir())
    print(
        f"wrote {len(cases)} cases / {total_blocks} blocks / "
        f"{total_bytes / 1024:.1f} KiB to {DATA_DIR.relative_to(REPO_ROOT)}"
    )
    if total_bytes > MAX_TOTAL_BYTES:
        print(
            f"ERROR: corpus is {total_bytes / 1024 / 1024:.1f} MiB, over the "
            f"{MAX_TOTAL_BYTES / 1024 / 1024:.0f} MiB budget",
            file=sys.stderr,
        )
        return 1
    return 0


# ---------------------------------------------------------------------------
# ---- Verification ---------------------------------------------------------
# ---------------------------------------------------------------------------


def _check_case(
    name: str, case: dict[str, Any], stored: np.lib.npyio.NpzFile
) -> tuple[list[str], int]:
    """Verify one case's npz against its manifest entry.

    Returns
    -------
    failures : list of str
        Human-readable problems; empty when the case is intact.
    n_arrays : int
        How many arrays the manifest claimed, checked or not.
    """
    failures: list[str] = []
    expected_keys = {
        entry["key"] for block in case["blocks"] for entry in block["arrays"].values()
    }
    unexpected = set(stored.files) - expected_keys
    if unexpected:
        failures.append(f"{name}: {case['file']} has extra arrays {sorted(unexpected)}")

    n_arrays = 0
    for block in case["blocks"]:
        for suffix, entry in block["arrays"].items():
            n_arrays += 1
            key = entry["key"]
            if key not in stored:
                failures.append(f"{name}[{block['label']}].{suffix}: missing {key}")
                continue
            actual = hash_array(stored[key])
            failures.extend(
                f"{name}[{block['label']}].{suffix}: {field} "
                f"{actual[field]!r} != manifest {entry[field]!r}"
                for field in ("shape", "dtype", "sha256")
                if actual[field] != entry[field]
            )
    return failures, n_arrays


def check() -> int:
    """Re-hash the stored corpus against the manifest. Returns an exit code.

    Deliberately imports no kernel: this answers "is the committed data
    intact and complete?", not "does the current build still reproduce
    it?". The latter is the parity test suite's job.
    """
    if not MANIFEST_PATH.exists():
        print(f"ERROR: no manifest at {MANIFEST_PATH}", file=sys.stderr)
        return 1

    manifest = json.loads(MANIFEST_PATH.read_text())
    if manifest.get("schema") != SCHEMA_VERSION:
        print(
            f"ERROR: manifest schema {manifest.get('schema')} != "
            f"{SCHEMA_VERSION}; regenerate the corpus",
            file=sys.stderr,
        )
        return 1

    failures: list[str] = []
    n_arrays = 0
    for name, case in manifest["cases"].items():
        path = DATA_DIR / case["file"]
        if not path.exists():
            failures.append(f"{name}: missing {case['file']}")
            continue
        with np.load(path) as stored:
            case_failures, case_arrays = _check_case(name, case, stored)
        failures.extend(case_failures)
        n_arrays += case_arrays

    orphans = sorted(
        path.name
        for path in DATA_DIR.glob("*.npz")
        if path.name not in {case["file"] for case in manifest["cases"].values()}
    )
    if orphans:
        failures.append(f"npz files with no manifest entry: {orphans}")

    if failures:
        print(f"corpus check FAILED ({len(failures)} problems):", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(
        f"corpus OK: {len(manifest['cases'])} cases / {n_arrays} arrays match the "
        f"manifest (generated at {manifest['git']['sha'][:12]}, "
        f"kernel digest {manifest['kernel_digest']['sha256'][:12]})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the stored corpus against the manifest instead of regenerating it",
    )
    args = parser.parse_args(argv)
    return check() if args.check else generate()


if __name__ == "__main__":
    raise SystemExit(main())
