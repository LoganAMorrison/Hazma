"""The corrected-value captures under ``oracles/``, as a delta relation.

``oracles/data/{A1,A2,A3,A4}.npz`` holds what each of the four Group A
defects should have returned, evaluated from a *patched* copy of the
Cython twin on the corpus's own grids before ``cython-to-rust`` deleted
that twin. ``oracles/README.md`` is the capture's own account of itself
and ``test_oracles.py`` is its gate; this module is the one line between
those arrays and `deltas.DECLARED_DELTAS`.

A Group A repair is therefore a `deltas.Reference`: the stored value is
superseded rather than corrected, and what supersedes it was computed by
an implementation that predates the port. That is the independent oracle
`projects/parity-pinned-defect-repair/rules.md` rule 3 asks for, and it
is why these repairs carry no closed-form model — each defect is a lost
or doubled *term* of a quadrature, which no transform of the stored
array can rebuild.

A capture is keyed by corpus case, block label and array suffix, while a
relation is handed only the entry point and the block. The entry point is
what closes the gap: `_cases` maps it back to the case name through
`cases.build_cases()`, so the case list stays derived from the corpus
specification rather than transcribed beside it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from cases import Block

#: Where `oracles/capture.py` writes, and `oracles/README.md` documents.
DATA_DIR = Path(__file__).resolve().parent / "oracles" / "data"


@cache
def _blocks(label: str) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """One defect's capture, as ``(case, block label) -> {suffix: array}``.

    Read through the oracle manifest rather than by rebuilding the npz
    key, because that key carries the block's *index* and the manifest is
    what maps an index back to a label.

    The arrays come back read-only: they are shared by every comparison
    that reads them and are the committed record of a measurement, not a
    scratch buffer.
    """
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    captured: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    with np.load(DATA_DIR / f"{label}.npz") as data:
        for case_name, case in manifest["defects"][label]["cases"].items():
            for block in case["blocks"]:
                arrays = {}
                for suffix, entry in block["arrays"].items():
                    array = data[entry["key"]]
                    array.setflags(write=False)
                    arrays[suffix] = array
                captured[case_name, block["label"]] = arrays
    return captured


@cache
def _cases(label: str) -> dict[tuple[str, str], str]:
    """``(module, qualname) -> case name`` over the cases one capture covers.

    Built by resolving the live entry points rather than by reading the
    manifests' ``entry_point``, which records where each case was
    *captured* from — the Cython modules the port has since deleted.
    """
    import cases as corpus  # noqa: PLC0415  (0.2s of model building, paid once)

    built = corpus.build_cases()
    out: dict[tuple[str, str], str] = {}
    for case_name in sorted({case for case, _ in _blocks(label)}):
        fn = built[case_name].resolve()
        key = (fn.__module__, fn.__qualname__)
        assert key not in out, f"{label}: {key} serves {out.get(key)} and {case_name}"
        out[key] = case_name
    return out


def captured(label: str) -> Callable[..., dict[str, np.ndarray]]:
    """A `deltas.Reference` callable serving one defect's whole capture.

    Parameters
    ----------
    label : str
        The roster label, from `deltas.REPAIRS`. Only the four Group A
        defects have a capture.

    Returns
    -------
    callable
        ``(fn, block) -> {suffix: array}``. Unlike the other relations
        this one reads ``fn`` — but only for its identity, never by
        calling it, since a reference that ran the kernel it supersedes
        would not be one.
    """

    def reference(fn: Callable[..., Any], block: Block) -> dict[str, np.ndarray]:
        entry = (getattr(fn, "__module__", None), getattr(fn, "__qualname__", None))
        try:
            case_name = _cases(label)[entry]  # type: ignore[index]
        except KeyError:
            msg = f"the {label} capture covers no entry point {entry[0]}:{entry[1]}"
            raise KeyError(msg) from None
        try:
            return _blocks(label)[case_name, block.label]
        except KeyError:
            msg = (
                f"the {label} capture holds no {case_name}[{block.label}]; "
                f"see test/parity/oracles/data/manifest.json"
            )
            raise KeyError(msg) from None

    return reference
