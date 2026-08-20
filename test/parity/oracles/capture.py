#!/usr/bin/env python
r"""Capture and verify the corrected-value oracles from the Cython twins.

The parity corpus in ``test/parity/data`` pins what hazma 2.1.0 shipped,
including seven values that are wrong. It is never rewritten
(`projects/parity-pinned-defect-repair/rules.md` rule 1), so a repair has
to be checked against a *corrected* reference captured somewhere else.
This module is that somewhere else, for the four defects that still have a
Cython twin: it patches one ``.pyx``, drives the corpus's own grids
through the patched Cython, and stores the result beside a manifest
recording exactly which source produced it.

Usage
-----
Verify the committed oracles against their manifest hashes. Imports no
kernel, evaluates nothing, and needs no build — the same contract
``python test/parity/generate.py --check`` has::

    python test/parity/oracles/capture.py --check

Prove the harness reproduces the corpus before trusting anything it
captures. Requires an *unpatched* build with the restored sources in
place; every case in `defects.DEFECTS` must come back bit-for-bit
identical to its stored array::

    python test/parity/oracles/capture.py --baseline

Capture one defect. Requires a build in which exactly that defect's patch
is applied::

    python test/parity/oracles/capture.py --defect A1

Merge the per-defect captures into ``data/manifest.json`` and record the
per-defect diff against the corpus::

    python test/parity/oracles/capture.py --assemble

`README.md` carries the whole loop, including the restore and revert steps
the capture cannot do for itself.

Why this is not circular
------------------------
The corrected values come from a source tree and a compiler that both
predate the Rust port, driven through an FFI boundary the port does not
use. What it is *not* is a proof that the physics is right: two
independent wrong implementations would still agree. Each repair task
carries a physics invariant alongside its oracle comparison for that —
`projects/parity-pinned-defect-repair/rules.md` rule 4.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
import time
import warnings
from collections.abc import Callable
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import defects as roster  # noqa: E402  (needs the sys.path entries above)
import entry_points  # noqa: E402

DATA_DIR = HERE / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"
PART_SUFFIX = ".part.json"
REPO_ROOT = HERE.parents[2]
CORPUS_MANIFEST = REPO_ROOT / "test" / "parity" / "data" / "manifest.json"
CORPUS_DATA = REPO_ROOT / "test" / "parity" / "data"

#: Bumped when the on-disk layout changes in a way a reader must notice.
SCHEMA_VERSION = 1

#: Same guardrail `test/parity/generate.py` carries, at the same figure:
#: these are test fixtures, not a dataset.
MAX_TOTAL_BYTES = 10 * 1024 * 1024

#: Arrays worth storing. ``grid`` and ``scalar_grid`` are inputs and are
#: byte-identical to the corpus by construction — the capture drives the
#: corpus's own `Block` objects — so storing them again would only be a
#: second copy of something already committed.
CAPTURED_SUFFIXES = ("values", "scalar_values")


# ---------------------------------------------------------------------------
# ---- Hashing and provenance -----------------------------------------------
# ---------------------------------------------------------------------------


def hash_array(array: np.ndarray) -> dict[str, Any]:
    """Describe an array by shape, dtype and a hash of its exact bytes.

    Deliberately the same three fields, computed the same way, as
    `test/parity/generate.py`'s function of this name: the two manifests
    are compared field-by-field by `test/parity/test_oracles.py`, and a
    second spelling of "what a stored array is" would let them drift.
    """
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def hash_file(path: Path) -> str:
    """SHA-256 of a file's exact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_provenance() -> dict[str, Any]:
    """The capturing commit, and whether the tree was clean at the time.

    Expected to report ``dirty`` here where `generate.py` might not: a
    capture runs on a tree with a patch applied and deleted sources
    restored, by construction. What identifies the numbers is
    ``patched_source`` below, not this.
    """

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "sha": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def cython_version() -> str | None:
    """The Cython that generated the extensions the values came from.

    Read off the generated ``.c`` rather than from package metadata, for
    the reason `test/parity/generate.py` gives: the build backend runs in
    an isolated environment, so the metadata lookup normally finds
    nothing.
    """
    for path in sorted((REPO_ROOT / "hazma").rglob("*.c")):
        with path.open() as handle:
            first = handle.readline()
        if first.startswith("/* Generated by Cython "):
            return first.split()[4].rstrip("*/ ")
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


def hazma_package_path() -> Path:
    """Where ``import hazma`` actually resolved, as a repo-relative path.

    Delegates the assertion to `cases.hazma_package_path`, which already
    refuses a `hazma` resolving outside the repository — one spelling of
    that check, shared with the corpus generator. What this adds is the
    *record*: per `docs/agents/lessons.md`
    ``[measured-tree-vs-imported-module]``, an assertion guards this
    capture and a recorded path makes a past one auditable.
    """
    import cases as corpus  # noqa: PLC0415 (kept off --check's import path)

    return corpus.hazma_package_path().relative_to(REPO_ROOT)


# ---------------------------------------------------------------------------
# ---- Driving the corpus's own blocks --------------------------------------
# ---------------------------------------------------------------------------


def _sweep_pointwise(evaluate: Callable[[int], Any], size: int) -> np.ndarray:
    """Evaluate point by point, filling anything that raises with ``nan``.

    The corpus does the same and for the same reason: a single bad point
    takes a batched call down with it, and what a kinematic edge returns
    (or raises) is part of what is pinned. Unlike the corpus this does not
    record the exception type — the fact of the raise is the corpus's
    contract to keep, and a capture that reproduced it differently would
    have failed the baseline pass long before reaching here.
    """
    results: list[np.ndarray | None] = []
    for index in range(size):
        try:
            results.append(np.asarray(evaluate(index), dtype=np.float64))
        except Exception:  # noqa: BLE001 - a raising edge is a stored nan
            results.append(None)
    live = [r for r in results if r is not None]
    if not live:
        raise RuntimeError(
            f"every one of the {size} grid points raised; the entry point is "
            "broken, not merely singular at an edge"
        )
    shape = live[0].shape
    return np.stack([np.full(shape, np.nan) if r is None else r for r in results])


@lru_cache(maxsize=1)
def _corpus_cases() -> dict[str, Any]:
    """`cases.build_cases()`, built once.

    Cached because constructing it solves for each mediator model's ``vs``
    and integrates its partial widths, and a capture walks up to seven
    cases.
    """
    import cases as corpus  # noqa: PLC0415  (import cost is the caller's)

    return corpus.build_cases()


def evaluate_case(name: str) -> dict[str, dict[str, np.ndarray]]:
    """Capture every block of one corpus case through its Cython source.

    Uses `test/parity/cases.py`'s own `Block` objects, so the grids and the
    call shapes are the corpus's rather than a second transcription of
    them. Only the entry point differs — `entry_points.SOURCES` resolves
    the Cython the corpus was captured through, which for most of these
    cases is no longer what the corpus manifest's ``entry_point`` names.

    Returns
    -------
    dict
        Block label to ``{suffix: array}`` over
        :data:`CAPTURED_SUFFIXES`.
    """
    fn = entry_points.resolve(entry_points.SOURCES[name])
    case = _corpus_cases()[name]

    captured: dict[str, dict[str, np.ndarray]] = {}
    for block in case.blocks:
        with warnings.catch_warnings():
            # scipy's IntegrationWarning fires on the deliberately
            # pathological grid points. That it fires is not part of the
            # contract; the value is, and that is what gets stored.
            warnings.simplefilter("ignore")
            arrays: dict[str, np.ndarray] = {}
            try:
                arrays["values"] = np.asarray(
                    block.array_call(fn, block.grid), dtype=np.float64
                )
            except Exception:  # noqa: BLE001 - recovered point by point
                per_point = _sweep_pointwise(
                    lambda i, b=block: b.array_call(fn, b.grid[i : i + 1]),
                    block.grid.size,
                )
                arrays["values"] = np.concatenate(list(per_point), axis=-1)

            probe = block.scalar_probe
            if probe.size:
                arrays["scalar_values"] = _sweep_pointwise(
                    lambda i, b=block, p=probe: b.scalar_call(fn, float(p[i])),
                    probe.size,
                )
        captured[block.label] = arrays
    return captured


def load_corpus_arrays(name: str) -> dict[str, dict[str, np.ndarray]]:
    """The stored corpus arrays for one case, keyed the same as a capture."""
    manifest = json.loads(CORPUS_MANIFEST.read_text())
    case = manifest["cases"][name]
    out: dict[str, dict[str, np.ndarray]] = {}
    with np.load(CORPUS_DATA / case["file"]) as stored:
        for block in case["blocks"]:
            out[block["label"]] = {
                suffix: stored[entry["key"]]
                for suffix, entry in block["arrays"].items()
                if suffix in CAPTURED_SUFFIXES
            }
    return out


# ---------------------------------------------------------------------------
# ---- The baseline pass ----------------------------------------------------
# ---------------------------------------------------------------------------


def baseline() -> int:
    """Require an unpatched build to reproduce the corpus bit-for-bit.

    This is what makes the rest of the module believable. Three of the
    twenty cases are driven through a shim that rebuilds a ``def`` the
    port deleted, and nine through modules resurrected from git history;
    neither is assumed faithful. Bit-equality is the right bar here rather
    than a tolerance because the capture runs on the corpus's own
    capturing platform against the same Cython version — anything less
    than identical means the harness, not the libm, is different.

    Returns a process exit code.
    """
    hazma_path = hazma_package_path()
    print(f"hazma resolved at {hazma_path}")

    names = sorted({name for d in roster.DEFECTS.values() for name in d.cases})
    failures: list[str] = []
    n_arrays = 0
    n_values = 0
    for name in names:
        started = time.monotonic()
        got = evaluate_case(name)
        want = load_corpus_arrays(name)
        if set(got) != set(want):
            failures.append(f"{name}: block labels {sorted(set(got) ^ set(want))}")
            continue
        mismatched = 0
        for label, arrays in got.items():
            for suffix, array in arrays.items():
                n_arrays += 1
                n_values += array.size
                stored = want[label][suffix]
                if hash_array(array) != hash_array(stored):
                    mismatched += 1
                    failures.append(
                        f"{name}[{label}].{suffix}: differs from the stored array"
                    )
        print(
            f"  {name:<62} {'OK' if not mismatched else 'MISMATCH'} "
            f"({time.monotonic() - started:.1f}s)"
        )

    if failures:
        print(f"baseline FAILED ({len(failures)} problems):", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print(
        f"baseline OK: {len(names)} cases / {n_arrays} arrays / {n_values} values "
        "reproduce the corpus bit-for-bit"
    )
    return 0


# ---------------------------------------------------------------------------
# ---- Capturing one defect -------------------------------------------------
# ---------------------------------------------------------------------------


def _patched_source(defect: roster.Defect) -> dict[str, Any]:
    """Prove the tree carries exactly this defect's patch, and record it.

    Raises
    ------
    RuntimeError
        If the working tree's diff against ``HEAD`` for the patched
        ``.pyx`` is not byte-identical to the committed patch. That is the
        `[mutation-harness-poisons-its-own-baseline]` guard: a harness
        that patches and reverts repeatedly and never checks accumulates
        edits while reporting independent measurements.
    """
    committed = (REPO_ROOT / defect.patch).read_text()
    actual = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "--", defect.source],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if actual != committed:
        raise RuntimeError(
            f"the working tree's diff for {defect.source} is not "
            f"{defect.patch}; build the patched tree first, and make sure no "
            "other edit rode along"
        )
    return {
        "path": defect.source,
        "patch": defect.patch,
        "patch_sha256": hash_file(REPO_ROOT / defect.patch),
        "patched_sha256": hash_file(REPO_ROOT / defect.source),
    }


def _restored_sources() -> list[dict[str, Any]]:
    """Record which resurrected sources are in the tree, and from where.

    Raises
    ------
    RuntimeError
        If a restored file's bytes differ from what its recorded revision
        holds — the capture would then describe a source nobody can
        recover.
    """
    recorded = []
    for path, rev in roster.RESTORED_SOURCES.items():
        target = REPO_ROOT / path
        if not target.exists():
            continue
        blob = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", f"{rev}:{path}"],
            check=True,
            capture_output=True,
        ).stdout
        digest = hashlib.sha256(blob).hexdigest()
        if hash_file(target) != digest:
            raise RuntimeError(
                f"{path} in the tree differs from {rev}:{path}; restore it "
                "again rather than capturing from an edited copy"
            )
        recorded.append({"path": path, "revision": rev, "sha256": digest})
    return recorded


def capture(label: str) -> int:
    """Capture one defect's oracle. Returns a process exit code."""
    defect = roster.DEFECTS[label]
    hazma_path = hazma_package_path()
    provenance = {
        "label": defect.label,
        "summary": defect.summary,
        "follow_up": defect.follow_up,
        "repair_task": defect.repair_task,
        "patched_source": _patched_source(defect),
        "restored_sources": _restored_sources(),
        "hazma_package": str(hazma_path),
        "environment": environment(),
        "git": git_provenance(),
        "cases": {},
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for name in defect.cases:
        captured = evaluate_case(name)
        blocks = []
        for index, (block_label, arrays) in enumerate(captured.items()):
            keys = {}
            for suffix, array in arrays.items():
                key = f"{name}|b{index}_{suffix}"
                payload[key] = array
                keys[suffix] = {"key": key, **hash_array(array)}
            blocks.append({"label": block_label, "arrays": keys})
        provenance["cases"][name] = {
            "source": dataclasses.asdict(entry_points.SOURCES[name]),
            "blocks": blocks,
        }
        print(f"  captured {name} ({len(blocks)} blocks)")

    np.savez_compressed(DATA_DIR / f"{label}.npz", **payload)
    (DATA_DIR / f"{label}{PART_SUFFIX}").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {label}.npz ({len(payload)} arrays) and {label}{PART_SUFFIX}")
    return 0


# ---------------------------------------------------------------------------
# ---- Assembling the manifest ----------------------------------------------
# ---------------------------------------------------------------------------


def summarize_case_diff(
    blocks: list[dict[str, Any]],
    oracle: np.lib.npyio.NpzFile,
    corpus_arrays: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Measure one case's captured blocks against the corpus arrays.

    Positions where both sides are ``nan`` are equal for this purpose; a
    position where exactly one is ``nan`` counts as moved, since the port
    has to reproduce which energies are kinematically absent.

    Split out from `diff_against_corpus` so the capture and
    `test/parity/test_oracles.py` derive these numbers with one
    implementation rather than two. The test re-runs it over the
    *committed* arrays and requires the result to equal what the manifest
    stores — which is what pins the manifest's numbers to the bytes
    beside them.

    Parameters
    ----------
    blocks : list of dict
        A case's ``blocks`` entries, from the manifest or a part file.
    oracle : numpy.lib.npyio.NpzFile
        The defect's open ``.npz``.
    corpus_arrays : dict
        `load_corpus_arrays` output for the same case.

    Returns
    -------
    dict
        How many values moved, out of how many, the largest absolute and
        relative shift, the sign pattern, and how many of the moved
        positions have no finite magnitude (a ``nan`` or an ``inf`` on one
        side only).
    """
    moved = total = unmeasurable = 0
    max_abs = max_rel = 0.0
    up = down = 0
    for block in blocks:
        shipped = corpus_arrays[block["label"]]
        for suffix, entry in block["arrays"].items():
            got = oracle[entry["key"]]
            want = shipped[suffix]
            both_nan = np.isnan(got) & np.isnan(want)
            differs = ~(both_nan | (got == want))
            total += got.size
            moved += int(differs.sum())
            if not differs.any():
                continue
            delta = np.abs(got[differs] - want[differs])
            # A position where exactly one side is nan counts as moved but
            # has no finite magnitude, and so does one where either side is
            # inf. Both are real findings and are counted separately rather
            # than folded into a `max` that would come back nan and say
            # nothing.
            scale = np.abs(want[differs])
            finite = np.isfinite(delta)
            unmeasurable += int(np.sum(~finite))
            if finite.any():
                max_abs = max(max_abs, float(np.max(delta[finite])))
            relative = finite & (scale > 0)
            if relative.any():
                max_rel = max(max_rel, float(np.max(delta[relative] / scale[relative])))
            up += int(np.sum(got[differs] > want[differs]))
            down += int(np.sum(got[differs] < want[differs]))
    return {
        "values_moved": moved,
        "values_total": total,
        "max_abs_shift": max_abs,
        "max_rel_shift": max_rel,
        "moved_up": up,
        "moved_down": down,
        "unmeasurable": unmeasurable,
    }


def summarize_defect_diff(defect: dict[str, Any]) -> dict[str, Any]:
    """Re-derive one defect's whole `diff_against_corpus` from the arrays.

    Every case, every block, every recorded field — the manifest's own
    numbers are never read. `test/parity/test_oracles.py` compares the
    result against what the manifest stores.
    """
    with np.load(DATA_DIR / defect["file"]) as oracle:
        return {
            name: summarize_case_diff(case["blocks"], oracle, load_corpus_arrays(name))
            for name, case in defect["cases"].items()
        }


def diff_against_corpus(label: str) -> dict[str, Any]:
    """Measure a freshly captured defect against the corpus it corrects.

    This is the deliverable the repair tasks consume: the first
    measurement of each defect's size, taken from Cython rather than from
    the Rust that will be repaired. Reads the part file `capture` wrote,
    since `assemble` has not built the manifest entry yet.
    """
    part = json.loads((DATA_DIR / f"{label}{PART_SUFFIX}").read_text())
    with np.load(DATA_DIR / f"{label}.npz") as oracle:
        return {
            name: summarize_case_diff(case["blocks"], oracle, load_corpus_arrays(name))
            for name, case in part["cases"].items()
        }


def assemble() -> int:
    """Merge the per-defect captures into one manifest. Returns an exit code."""
    parts = sorted(DATA_DIR.glob(f"*{PART_SUFFIX}"))
    missing = sorted(set(roster.DEFECTS) - {p.name.split(".")[0] for p in parts})
    if missing:
        print(f"ERROR: no capture for {missing}", file=sys.stderr)
        return 1

    manifest: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "generated_by": "test/parity/oracles/capture.py",
        "corpus_manifest_sha256": hash_file(CORPUS_MANIFEST),
        "defects": {},
    }
    for part_path in parts:
        label = part_path.name.split(".")[0]
        part = json.loads(part_path.read_text())
        part["file"] = f"{label}.npz"
        part["diff_against_corpus"] = diff_against_corpus(label)
        manifest["defects"][label] = part

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    # Only now, with the merged manifest on disk. A part file is the sole
    # record of which patched build produced its `.npz`, and each one costs
    # a rebuild and a full sweep to recreate — so nothing is discarded
    # until the thing that supersedes it exists.
    for part_path in parts:
        part_path.unlink()

    total_bytes = sum(path.stat().st_size for path in DATA_DIR.iterdir())
    print(
        f"wrote {len(manifest['defects'])} defects / "
        f"{total_bytes / 1024:.1f} KiB to {DATA_DIR.relative_to(REPO_ROOT)}"
    )
    if total_bytes > MAX_TOTAL_BYTES:
        print(
            f"ERROR: oracles are {total_bytes / 1024 / 1024:.1f} MiB, over the "
            f"{MAX_TOTAL_BYTES / 1024 / 1024:.0f} MiB budget",
            file=sys.stderr,
        )
        return 1
    return 0


# ---------------------------------------------------------------------------
# ---- Verification ---------------------------------------------------------
# ---------------------------------------------------------------------------


def load_manifest() -> dict[str, Any]:
    """Read the committed oracle manifest.

    Shared with `test/parity/test_oracles.py` so there is one spelling of
    where it lives and how it is parsed.
    """
    return json.loads(MANIFEST_PATH.read_text())


def check() -> int:
    """Re-hash the stored oracles against the manifest. Returns an exit code.

    Imports no kernel and evaluates nothing: this answers "is the
    committed capture intact and complete?", not "does the current build
    reproduce it?" — which it could not, since reproducing it needs a
    patched build. `test/parity/generate.py --check` draws the same line.
    """
    if not MANIFEST_PATH.exists():
        print(f"ERROR: no manifest at {MANIFEST_PATH}", file=sys.stderr)
        return 1

    manifest = load_manifest()
    if manifest.get("schema") != SCHEMA_VERSION:
        print(
            f"ERROR: manifest schema {manifest.get('schema')} != "
            f"{SCHEMA_VERSION}; recapture the oracles",
            file=sys.stderr,
        )
        return 1

    failures: list[str] = []
    n_arrays = 0
    for label, defect in manifest["defects"].items():
        path = DATA_DIR / defect["file"]
        if not path.exists():
            failures.append(f"{label}: missing {defect['file']}")
            continue
        expected = {
            entry["key"]
            for case in defect["cases"].values()
            for block in case["blocks"]
            for entry in block["arrays"].values()
        }
        with np.load(path) as stored:
            unexpected = set(stored.files) - expected
            if unexpected:
                failures.append(
                    f"{label}: {defect['file']} has extra arrays {sorted(unexpected)}"
                )
            for name, case in defect["cases"].items():
                for block in case["blocks"]:
                    for suffix, entry in block["arrays"].items():
                        n_arrays += 1
                        key = entry["key"]
                        if key not in stored:
                            failures.append(
                                f"{label}/{name}[{block['label']}].{suffix}: "
                                f"missing {key}"
                            )
                            continue
                        actual = hash_array(stored[key])
                        failures.extend(
                            f"{label}/{name}[{block['label']}].{suffix}: {field} "
                            f"{actual[field]!r} != manifest {entry[field]!r}"
                            for field in ("shape", "dtype", "sha256")
                            if actual[field] != entry[field]
                        )

    orphans = sorted(
        path.name
        for path in DATA_DIR.glob("*.npz")
        if path.name not in {d["file"] for d in manifest["defects"].values()}
    )
    if orphans:
        failures.append(f"npz files with no manifest entry: {orphans}")

    if failures:
        print(f"oracle check FAILED ({len(failures)} problems):", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(
        f"oracles OK: {len(manifest['defects'])} defects / {n_arrays} arrays match "
        f"the manifest (corpus manifest {manifest['corpus_manifest_sha256'][:12]})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="verify the stored oracles against the manifest; needs no build",
    )
    mode.add_argument(
        "--baseline",
        action="store_true",
        help="require an unpatched build to reproduce the corpus bit-for-bit",
    )
    mode.add_argument(
        "--defect",
        choices=sorted(roster.DEFECTS),
        help="capture one defect from a build carrying exactly its patch",
    )
    mode.add_argument(
        "--assemble",
        action="store_true",
        help="merge the per-defect captures into the manifest",
    )
    args = parser.parse_args(argv)
    if args.check:
        return check()
    if args.baseline:
        return baseline()
    if args.assemble:
        return assemble()
    return capture(args.defect)


if __name__ == "__main__":
    raise SystemExit(main())
