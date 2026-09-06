"""Declared deltas: which pinned positions a repair moved, and how.

The corpus under ``data/`` records what 2.1.0 shipped, and
``projects/cython-to-rust/rules.md`` rule 2 forbids regenerating it from a
tree whose kernels run on Rust. Some of what it pins is *wrong* — the
defects filed under ``docs/followups/`` and repaired under
``projects/parity-pinned-defect-repair`` — and a repair still has to get
past the gate. It does so by declaring, for each stored array it moves,
how the repaired value relates to the stored one. ``test_parity.py``
compares a declared array against that relation and every other array
against the stored values unchanged, so a repair proves it moved only
what it meant to and nothing else moved at all.

The schema is the one ``projects/parity-pinned-defect-repair/references/corpus-repinning.md``
specifies, keyed the way ``stability.PORTABILITY_ZEROS`` is keyed. It is
an allowlist of arrays, not a rule over positions of a given shape: every
declaration names the repair, the positions, the relation, the
measurement that justifies its budget, and where the evidence lives.

Relations
---------
``Additive`` — the repaired value is the stored value plus a term the
declaration knows how to compute. The term here is evaluated live, from
the repaired kernel, and is its own adaptive quadrature; the budget each
relation carries is measured, not assumed, and says so in its ``why``.

Staleness
---------
A declaration that no longer describes a change is a hole in the gate
(spec rule 3), so the runner also asserts that a declared array *has*
moved: at the positions where the term exceeds the relation's own budget,
the live value must differ from the stored one. Reverting a repair fails
that assertion; the declaration cannot outlive the repair it describes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from cases import Block

#: The closed set of repair labels a declaration may carry: the roster in
#: ``projects/parity-pinned-defect-repair/references/defect-blast-radius.md``.
REPAIRS = frozenset({"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"})

#: The sentinel for "every position of the array".
ALL = "all"

Positions = tuple[int, ...] | Literal["all"]

#: ``(entry point, block) -> {array suffix: term}``. Evaluates whatever
#: the relation adds to the stored array, on the block's own grids, for
#: every value suffix the block stores (``values`` and, when the entry
#: point has a scalar branch, ``scalar_values``).
TermFn = Callable[[Callable[..., Any], "Block"], dict[str, np.ndarray]]


@dataclass(frozen=True)
class Additive:
    """``repaired == stored + term(block)``, within ``rtol``.

    Parameters
    ----------
    term : TermFn
        Computes the additive term for one block.
    rtol : float
        Relative budget the relation holds to. Measured, and ``why`` says
        how; a term that is its own quadrature cannot be held to the
        case's bit-level budget.
    why : str
        One-line justification of ``rtol``.
    """

    term: TermFn
    rtol: float
    why: str


@dataclass(frozen=True)
class Delta:
    """One declared change to one stored array.

    Parameters
    ----------
    repair : str
        Which roster entry moved it; drawn from `REPAIRS`.
    positions : tuple of int or ALL
        Which positions the relation covers. Undeclared positions are
        still compared against the stored value under the case's budget.
    relation : Additive
        How the repaired value relates to the stored one.
    measured : str
        The measurement behind the declaration, so the table is not a
        list of assertions nobody re-derived.
    evidence : str
        Repo-relative path of the note that holds the measurement.
    """

    repair: str
    positions: Positions
    relation: Additive
    measured: str
    evidence: str


# ---------------------------------------------------------------------------
# B4 -- the scalar mediator decay spectrum's FSR was half its size
# ---------------------------------------------------------------------------

#: The three FSR channels of ``scalar_mediator_decay_spectrum``, in the
#: entry point's own mode spelling.
SCALAR_DECAY_FSR_MODES = ("e e g", "pi pi g", "mu mu g")


def _scalar_decay_fsr_half(
    fn: Callable[..., Any], block: Block
) -> dict[str, np.ndarray]:
    """Half the repaired kernel's FSR-only spectrum, on the block's grids.

    The repair doubled every rest-frame FSR coefficient and touched no
    decay channel, so ``stored = decay + fsr_old`` and
    ``repaired = decay + 2 fsr_old``: the term is ``fsr_old``, which the
    repaired kernel returns as exactly twice itself when asked for the
    FSR modes alone (the factor is a power of two, applied last). The
    block's ``partial_widths`` are the ones the stored values were
    captured with (``cases._scalar_decay_spectrum_blocks``).
    """
    params = block.params
    args = (
        params["mediator_energy"],
        params["mediator_mass"],
        np.asarray(params["partial_widths"], dtype=np.float64),
        list(SCALAR_DECAY_FSR_MODES),
    )
    terms = {"values": 0.5 * np.asarray(fn(block.grid, *args), dtype=np.float64)}
    probe = block.scalar_probe
    if probe.size:
        terms["scalar_values"] = 0.5 * np.array(
            [fn(float(x), *args) for x in probe], dtype=np.float64
        )
    return terms


_B4 = Delta(
    repair="B4",
    positions=ALL,
    relation=Additive(
        term=_scalar_decay_fsr_half,
        rtol=1e-3,
        why="the term is its own cos(theta) quadrature (epsrel 1e-5) over a "
        "different integrand than the stored total, and the repaired total "
        "is a third; measured 3.1e-4 worst relative over all 4,305 "
        "declared positions, at ms_550.boosted_strong E=2696 MeV where the "
        "boost window is narrow and the integrator's own error estimate is "
        "what moves. Three times headroom; the pre-repair arrays sit a "
        "factor of two away where FSR is the only open channel.",
    ),
    measured="the FSR-only spectrum at rest is 0.5000000000 x the annihilation-"
    "side ScalarMediator.dnde_xx_to_s_to_ffg / dnde_xx_to_s_to_pipig at the "
    "same invariant mass, in every channel, to ten digits after removing the "
    "legacy/PDG alpha ratio; the vector twin is 1.000 x its own. 3,061 of the "
    "4,305 pinned positions move, 2,874 by more than 0.1%, up to exactly "
    "double.",
    evidence="docs/followups/done/scalar-decay-fsr-half-normalized.md",
)

#: Every declared array. The ``mu_mu_only`` blocks of the same case are
#: deliberately absent: they open no FSR channel and must still match the
#: stored arrays bit for bit, which is the "moved only what it intended"
#: half of the proof.
DECLARED_DELTAS: dict[tuple[str, str, str], Delta] = {
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_250.rest.default",
        "values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_250.rest.default",
        "scalar_values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_250.rest_plus_eps.default",
        "values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_250.rest_plus_eps.default",
        "scalar_values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_250.near_rest.default",
        "values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_250.near_rest.default",
        "scalar_values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_250.boosted_mild.default",
        "values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_250.boosted_mild.default",
        "scalar_values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_250.boosted_strong.default",
        "values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_250.boosted_strong.default",
        "scalar_values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_550.rest.default",
        "values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_550.rest.default",
        "scalar_values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_550.rest_plus_eps.default",
        "values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_550.rest_plus_eps.default",
        "scalar_values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_550.near_rest.default",
        "values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_550.near_rest.default",
        "scalar_values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_550.boosted_mild.default",
        "values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_550.boosted_mild.default",
        "scalar_values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_550.boosted_strong.default",
        "values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_550.boosted_strong.default",
        "scalar_values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_900.rest.default",
        "values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_900.rest.default",
        "scalar_values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_900.rest_plus_eps.default",
        "values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_900.rest_plus_eps.default",
        "scalar_values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_900.near_rest.default",
        "values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_900.near_rest.default",
        "scalar_values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_900.boosted_mild.default",
        "values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_900.boosted_mild.default",
        "scalar_values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_900.boosted_strong.default",
        "values",
    ): _B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_900.boosted_strong.default",
        "scalar_values",
    ): _B4,
}


def declared(case_name: str, block_label: str, array_suffix: str) -> Delta | None:
    """The declaration covering one stored array, or ``None``."""
    return DECLARED_DELTAS.get((case_name, block_label, array_suffix))
