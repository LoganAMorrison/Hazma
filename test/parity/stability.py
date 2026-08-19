"""Which pinned corpus values are rounding residue rather than physics.

The problem
-----------
`generate.py` stores what the pre-port Cython returned at every sampled
point. At almost every point that is a well-conditioned number and any
faithful reimplementation reproduces it. At some it is not: the kernel
forms a difference of two nearly-equal quantities, every significant bit
cancels, and what got pinned is one platform's ``atan`` rounding. Task
1.3 found this the hard way — enabling the corpus in CI produced its
first non-macOS run, and
``cross_sections.scalar.sigma_xl_to_xl[closed_resonance.mu]`` at scalar
probe index 5 came back ``+5.624212846110624e-07`` on Linux/glibc against
the ``-1.504080817723100e-02`` the corpus holds. Same source, same
commit, opposite sign and seven orders of magnitude.

A tolerance cannot fix that, and the cross-platform failure is only the
symptom. The corpus exists to gate the Rust port, and a faithful Rust
rewrite with a different instruction order lands somewhere else in the
same cancellation region too. Those values assert nothing on any
platform, so this module stops them from pretending to.

What "unpinnable" means here
----------------------------
A point is unpinnable when the value the corpus stores differs from the
value the same closed form is mathematically worth by more than
`UNPINNABLE_RTOL`. "Mathematically worth" is `reference.py`: the same
expressions, copied verbatim from the ``.pyx``, evaluated at 60 decimal
digits. So this is a statement about the stored number, established
against ground truth — not a guess from a condition-number proxy and not
an archaeology of which platforms happen to disagree.

Both alternatives were tried first and are recorded here because they
look reasonable and are not:

- **Perturbing the inputs by an ulp** (the mechanism
  ``docs/followups/done/parity-corpus-pins-ill-conditioned-points.md``
  proposed) measures *conditioning*, and these points are perfectly
  well conditioned — the true function is smooth across
  ``e_cm = 2 mx``, with no pole at all. At the exemplar point a 1-ulp
  nudge of ``e_cm`` moves the result by 1.6e-10 relative while the
  stored value is wrong by 2.4e4. What is broken is the *stability* of
  the algorithm, which no input perturbation can see.
- **Thresholding the atan difference itself** does not separate either.
  In ``closed_resonance`` the correct points carry
  ``|atan(u) - atan(v)|`` between 2.2e-17 and 2.8e-16 and the wrong ones
  between 3.3e-25 and 1.8e-17 — overlapping ranges, because whether the
  cancelled term matters depends on how it compares with the
  ``ms * width_s * (...)`` term beside it, which the difference alone
  does not know.

Which entry points are affected
-------------------------------
The four scalar-mediator elastic cross sections that evaluate
``P * atan(ms/width_s) - P * atan((ms**2 - 4 mx**2 + s)/(ms * width_s))``:
`sigma_xl_to_xl`, `sigma_xpi_to_xpi`, `sigma_xpi0_to_xpi0` and
`sigma_xg_to_xg`. `sigma_xs_to_xs` shares the family but not the
construction and is not affected. Two regimes reach the cancellation:

``e_cm -> 2 mx``
    the two ``atan`` arguments become equal, so the difference goes to
    zero while each term stays O(1). Hits the ``2 mx`` anchors of every
    model point, four grid points wide.
``width_s -> 0``
    both arguments exceed ~9e15 and each ``atan`` rounds to the double
    nearest ``pi/2``, so the difference is 0 or +-1 ulp regardless of the
    physics. This is ``closed_resonance`` (``width_s`` = 3.7e-15), where
    it spoils everything above ``e_cm ~ 595`` — about 29% of the block.

Below ``e_cm = sqrt(4 mx**2 - ms**2)`` the second argument is *negative*
and large, so the two ``atan``s sit at opposite ends and their difference
is ~pi. That half of ``closed_resonance`` is fine and stays pinned.

Regenerating
------------
.. code-block:: sh

    python test/parity/stability.py --regenerate

Needs `mpmath` (the ``dev`` dependency group). The test path does not:
it reads the committed ``data/unpinnable.json``, so a normal ``pytest``
run imports neither `mpmath` nor `reference`.

``projects/cython-to-rust/rules.md`` rule 2 governs regeneration the same
way it governs the corpus itself. The mask is derived from the *stored*
corpus and from `reference.py`, and touches no live kernel, so a tree
with Rust kernels served cannot skew it — but the stored corpus it reads
must still be the pre-port one, which is what the recorded
``kernel_digest`` pins.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MASK_PATH = HERE / "data" / "unpinnable.json"

#: Schema of ``data/unpinnable.json``. Bump when its shape changes.
SCHEMA_VERSION = 1

#: How far a stored value may sit from what the closed form is worth
#: before it stops counting as a pin. Not a tolerance on a *port* — the
#: budgets in `tolerances.py` are that — but the line between "the
#: corpus recorded this number" and "the corpus recorded a rounding
#: residue". 1e-9 is seven decades looser than the tightest declared
#: budget (`EXACT_RTOL`) and three decades tighter than the loosest
#: (`NESTED_RTOL`), so nothing lands here that any budget could have
#: carried.
#:
#: It is also where the data puts it. Binning all 4,675 stored grid values of
#: the four affected cases by their disagreement with `reference` gives a
#: bimodal histogram, and 1e-9 is its minimum:
#:
#: .. code-block:: text
#:
#:     decade   1e-13  1e-12  1e-11  1e-10  1e-9   1e-8  1e-7  1e-6  1e-5
#:     points     245     80     19     11      4     23    35    30    42
#:
#: The left mode is accumulated rounding through a long expression and
#: peaks at 1e-16; the right mode is the cancellation and runs on out to
#: 1e+24. Four points sit in the valley decade — the only ones the choice
#: of threshold could move either way.
UNPINNABLE_RTOL = 1e-9

#: Entry points whose closed form contains the cancelling ``atan``
#: difference, mapped to the `reference` function that evaluates it
#: exactly. Only these are examined; every other corpus case is pinned in
#: full. Adding a row here is what it takes to mask a new case, and
#: `test_parity.test_only_the_declared_cases_are_masked` holds the two
#: sides together.
AFFECTED_CASES: dict[str, str] = {
    "cross_sections.scalar.sigma_xl_to_xl": "sigma_xl_to_xl",
    "cross_sections.scalar.sigma_xpi_to_xpi": "sigma_xpi_to_xpi",
    "cross_sections.scalar.sigma_xpi0_to_xpi0": "sigma_xpi0_to_xpi0",
    "cross_sections.scalar.sigma_xg_to_xg": "sigma_xg_to_xg",
}


def load_mask() -> dict[str, Any]:
    """The committed unpinnable-point mask.

    Raises
    ------
    FileNotFoundError
        If the mask has not been generated. It is committed data, like
        the corpus itself, so this means the checkout is incomplete
        rather than that the mask is optional.
    """
    with MASK_PATH.open(encoding="utf-8") as handle:
        mask = json.load(handle)
    if mask.get("schema") != SCHEMA_VERSION:
        raise ValueError(
            f"{MASK_PATH.name} schema {mask.get('schema')} != {SCHEMA_VERSION}; "
            "regenerate with `python test/parity/stability.py --regenerate`"
        )
    return mask


def unpinnable_indices(
    mask: dict[str, Any], case_name: str, block_label: str, array_suffix: str
) -> frozenset[int]:
    """Positions of one block array that assert nothing and are skipped.

    Parameters
    ----------
    mask : dict
        From `load_mask`.
    case_name, block_label : str
        A `cases.build_cases()` key and one of its `cases.Block` labels.
    array_suffix : str
        ``"values"`` or ``"scalar_values"``. Indices are into that array,
        not into the grid, so the caller needs no mapping.

    Returns
    -------
    frozenset of int
        Empty for every case not in `AFFECTED_CASES`.
    """
    block = mask["cases"].get(case_name, {}).get(block_label, {})
    return frozenset(block.get(array_suffix, ()))


def total_masked(mask: dict[str, Any]) -> int:
    """How many array positions the mask removes, across every block."""
    return sum(
        len(indices)
        for case in mask["cases"].values()
        for block in case.values()
        for indices in block.values()
    )


# ===========================================================================
# ---- Regeneration (needs mpmath; not imported by the test path) -----------
# ===========================================================================


def _relative_to_reference(
    stored: float,
    exact: Any,  # noqa: ANN401 (an `mpmath.mpf`; this module must not
    # import mpmath at module scope -- the test path never needs it)
) -> float:
    """``|stored - exact| / |exact|``, in arbitrary precision.

    Both ends of the comparison get their degenerate cases named rather
    than silently bucketed:

    - a non-finite stored value (the corpus holds one ``-inf``, where the
      Cython divided by a denominator that rounded to exactly zero) can
      never be reproduced within a relative budget, so it is unpinnable
      by definition;
    - an exactly-zero reference against a non-zero stored value is too,
      since no relative measure exists;
    - zero against zero agrees exactly.
    """
    import mpmath as mp  # noqa: PLC0415 (regeneration-only dependency)

    if not math.isfinite(stored):
        return math.inf
    if exact == 0:
        return 0.0 if stored == 0.0 else math.inf
    return float(abs(mp.mpf(stored) - exact) / abs(exact))


def regenerate() -> int:
    """Rebuild ``data/unpinnable.json`` from the corpus and `reference`.

    Compares every stored value of every `AFFECTED_CASES` block against
    the arbitrary-precision evaluation of the same closed form and
    records the positions that disagree by more than `UNPINNABLE_RTOL`.
    Returns a process exit code.
    """
    import cases as corpus  # noqa: PLC0415
    import generate as corpus_generate  # noqa: PLC0415
    import mpmath as mp  # noqa: PLC0415 (regeneration-only dependency)
    import numpy as np  # noqa: PLC0415 (matches the import style above)
    import reference  # noqa: PLC0415

    mp.mp.dps = reference.DPS
    manifest = corpus_generate.load_manifest()
    built = corpus.build_cases()

    cases_out: dict[str, dict[str, dict[str, list[int]]]] = {}
    kept_worst = 0.0
    masked_best = float("inf")
    for case_name, reference_name in sorted(AFFECTED_CASES.items()):
        exact_fn = getattr(reference, reference_name)
        case = built[case_name]
        path = corpus_generate.DATA_DIR / manifest["cases"][case_name]["file"]
        with np.load(path) as npz:
            stored = {key: npz[key] for key in npz.files}
        blocks_out: dict[str, dict[str, list[int]]] = {}
        for index, block in enumerate(case.blocks):
            manifest_block = manifest["cases"][case_name]["blocks"][index]
            args = block.params["args"]
            arrays_out: dict[str, list[int]] = {}
            for suffix, abscissae in (
                ("values", block.grid),
                ("scalar_values", block.scalar_probe),
            ):
                if suffix not in manifest_block["arrays"]:
                    continue
                values = stored[manifest_block["arrays"][suffix]["key"]]
                flagged = []
                for position, abscissa in enumerate(abscissae):
                    try:
                        exact = exact_fn(float(abscissa), *args)
                    except ZeroDivisionError:
                        # `e_cm = 2 mx` exactly: the closed form is 0/0
                        # there (both the atan difference and the whole
                        # log tail vanish with the `4 mx**2 - s`
                        # denominator), so the limit is finite but no
                        # evaluation of *this* expression reaches it.
                        # Whatever a double produced is its own rounding.
                        flagged.append(position)
                        masked_best = min(masked_best, math.inf)
                        continue
                    rel = _relative_to_reference(float(values[position]), exact)
                    if rel > UNPINNABLE_RTOL:
                        flagged.append(position)
                        masked_best = min(masked_best, rel)
                    else:
                        kept_worst = max(kept_worst, rel)
                if flagged:
                    arrays_out[suffix] = flagged
            if arrays_out:
                blocks_out[block.label] = arrays_out
        cases_out[case_name] = blocks_out

    mask = {
        "schema": SCHEMA_VERSION,
        "unpinnable_rtol": UNPINNABLE_RTOL,
        "kernel_digest": manifest["kernel_digest"]["sha256"],
        "reference_dps": reference.DPS,
        "cases": cases_out,
    }
    MASK_PATH.write_text(json.dumps(mask, indent=1) + "\n", encoding="utf-8")
    print(
        f"wrote {MASK_PATH.relative_to(HERE.parent.parent)}: "
        f"{total_masked(mask)} positions masked across "
        f"{sum(len(blocks) for blocks in cases_out.values())} blocks"
    )
    print(
        f"  separation: worst kept point {kept_worst:.3e}, "
        f"best masked point {masked_best:.3e} (threshold {UNPINNABLE_RTOL:.0e})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI: ``--regenerate`` rewrites the mask; otherwise summarise it."""
    import argparse  # noqa: PLC0415 (CLI-only)

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="rebuild data/unpinnable.json from the corpus and reference.py",
    )
    args = parser.parse_args(argv)
    if args.regenerate:
        return regenerate()
    mask = load_mask()
    print(
        f"{total_masked(mask)} positions masked; rtol "
        f"{mask['unpinnable_rtol']:.0e}; kernel digest "
        f"{mask['kernel_digest'][:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
