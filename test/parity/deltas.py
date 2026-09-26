"""Declared deltas: which pinned positions a repair moved, and how.

The corpus under ``data/`` records what 2.1.0 shipped, and
``projects/cython-to-rust/rules.md`` rule 2 forbids regenerating it from a
tree whose kernels run on Rust. Some of what it pins is *wrong* — the
defects filed under ``docs/followups/``, repaired first under
``projects/parity-pinned-defect-repair`` and since then one at a time —
and a repair still has to get past the gate. The rules are
``docs/adrs/ADR-0003-corpus-repairs-are-declared-deltas.md``. A repair
passes by declaring, for each stored array it moves,
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
A relation answers one question — what the repaired array should be —
and all relations return it from `Relation.expected`.

``Additive`` — the repaired value is the stored value plus a term the
declaration knows how to compute. The term may be evaluated live, from
the repaired kernel, and be its own adaptive quadrature; the budget each
relation carries is measured, not assumed, and says so in its ``why``.

``Exact`` — the repaired value is a closed-form transform of the stored
value, so nothing is recomputed and the prediction is limited by the
transform's own arithmetic rather than by a quadrature. It is the
strongest relation the spec offers and the one to reach for first.

``Reference`` — the stored value is superseded outright by one a second
implementation computes. For a defect that lost or doubled a term of a
quadrature there is no term to add and no transform to apply, so the only
statement available is what an implementation that does not carry the
defect returns: `thermal_reference` integrates the same integrand with
scipy's QUADPACK, and `oracle_reference` reads the arrays captured from
the Cython twins before the port deleted them.

``Composed`` — one relation for an array **more than one** repair moves.
``rules.md`` rule 7 forbids leaving that as overlapping declarations, and
one key holds one `Delta` in any case, so they collapse: a base relation
predicts the array as the first repair leaves it, and each further repair
adds a term or applies a closed-form transform to that prediction. The
`Delta` then names them all, as ``"A1+B1"``,
and `repair_labels` is what splits a composite spelling back into the
roster entries it is made of.

Absolute floors
---------------
A prediction is compared to the repaired kernel relatively, with ``atol``
at zero, because an absolute floor is scale-dependent and a spectrum runs
over twenty decades (`tolerances`, "``atol`` is 0.0 everywhere"). One
relation cannot hold to that. A `Composed` whose addend *relocates* a
line subtracts the line back out of the base, and where the base is a
captured array that subtraction annihilates whatever the cancelled term's
last bit could not hold -- so the prediction is exactly zero at positions
where the kernel still returns the continuum underneath, and no relative
budget can describe it. Such a relation declares an ``atol`` at one ulp of
the term it cancels, ``why`` says which term, and `test_parity` caps it
below the array's own zero floor so it cannot grow into a budget.

Positions
---------
A declaration covers exactly the positions its mechanism reaches
(``rules.md`` rule 5): either an explicit tuple, every entry of which the
relation must move, or `MOVED`, which resolves at comparison time to the
positions where the predicted array differs from the stored one. Every
other position of a declared array is compared against the stored value
under the case's own budget, exactly as if nothing had been declared, so
a regression where the repair changed nothing is still caught at that
budget.

Staleness
---------
A declaration that no longer describes a change is a hole in the gate
(spec rule 3), so the runner also asserts that a declared array *has*
moved: at the positions where the prediction departs from the stored
value by more than the relation's own budget, the live value must differ
from the stored one. Reverting a repair fails that assertion; the
declaration cannot outlive the repair it describes.

Models without a declaration
----------------------------
`DELTA_MODELS` holds registered models of roster repairs and their
composites, including consumer variants. `DECLARED_DELTAS` holds only the arrays a
*landed* repair moves. The two differ while a model is established ahead
of its repair: declaring an array the tree has not yet moved would fail
the staleness rule above, so the model waits in `DELTA_MODELS` and the
repair adds the keys.
Model keys normally equal the roster label. An optional ``/variant``
suffix distinguishes consumer-specific budgets without inventing a new
repair: ``A3/nested`` still carries ``Delta.repair == "A3"``. Every
variant is registered and covered by the same shape and evidence gates.
`test_delta_models.py` gates the closed-form models.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import oracle_reference
import thermal_reference
import tolerances

from hazma import parameters
from hazma._core import boost as core_boost

if TYPE_CHECKING:
    from cases import Block

#: The closed set of repair labels a declaration may carry. ``A1``-``A4``
#: and ``B1``-``B6`` are the roster in
#: ``projects/parity-pinned-defect-repair/references/defect-blast-radius.md``;
#: ``C1`` onward are the repairs that landed after that project closed,
#: numbered in landing order and listed in ``README.md``, "Repairs". See
#: ``docs/adrs/ADR-0003-corpus-repairs-are-declared-deltas.md``.
REPAIRS = frozenset({"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "B5", "B6", "C1"})

#: The sentinel for "every position the relation actually moves", resolved
#: against the prediction at comparison time. A position the relation
#: leaves at its stored value has not moved, and a declaration that
#: covered it anyway would be wider than its mechanism.
MOVED = "moved"

Positions = tuple[int, ...] | Literal["moved"]

#: ``(entry point, block) -> {array suffix: term}``. Evaluates whatever
#: the relation adds to the stored array, on the block's own grids, for
#: every value suffix the block stores (``values`` and, when the entry
#: point has a scalar branch, ``scalar_values``).
TermFn = Callable[[Callable[..., Any], "Block"], dict[str, np.ndarray]]

#: ``(block, stored) -> {array suffix: repaired}``. Rebuilds the repaired
#: arrays from the stored ones, which arrive keyed by suffix exactly as
#: the corpus holds them (``grid`` and ``scalar_grid`` included, so a
#: transform can read the abscissae it is a function of).
TransformFn = Callable[["Block", dict[str, np.ndarray]], dict[str, np.ndarray]]


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
        One-line justification of ``rtol``, and of ``atol`` where it is
        set.
    atol : float, optional
        Absolute floor, added to ``rtol`` when the prediction is compared.
        ``0.0`` everywhere the relation's arithmetic resolves the repaired
        value at every magnitude it takes; see "Absolute floors" in the
        module docstring for the one composition that cannot.
    """

    term: TermFn
    rtol: float
    why: str
    atol: float = 0.0

    def expected(
        self,
        fn: Callable[..., Any],
        block: Block,
        stored: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """The repaired arrays this relation predicts, by suffix."""
        return {
            suffix: stored[suffix] + term
            for suffix, term in self.term(fn, block).items()
        }


@dataclass(frozen=True)
class Exact:
    """``repaired == transform(stored)``, within ``rtol``.

    Parameters
    ----------
    transform : TransformFn
        Rebuilds the repaired arrays from the stored ones.
    rtol : float
        Relative budget the relation holds to. A closed-form transform of
        the stored array reaches the last few bits, so this is orders
        tighter than an `Additive` whose term is a quadrature; ``why``
        says which operations set it.
    why : str
        One-line justification of ``rtol``, and of ``atol`` where it is
        set.
    atol : float, optional
        Absolute floor, added to ``rtol`` when the prediction is compared.
        ``0.0`` everywhere the relation's arithmetic resolves the repaired
        value at every magnitude it takes; see "Absolute floors" in the
        module docstring for the one composition that cannot.
    """

    transform: TransformFn
    rtol: float
    why: str
    atol: float = 0.0

    def expected(
        self,
        fn: Callable[..., Any],
        block: Block,
        stored: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """The repaired arrays this relation predicts, by suffix."""
        del fn  # a closed-form transform reads the stored arrays, not a kernel
        return self.transform(block, stored)


@dataclass(frozen=True)
class Reference:
    """``repaired == reference(block)``, within ``rtol``.

    The stored array is superseded rather than corrected: the repaired
    kernel is compared against a value reached without it. Use this where
    the defect left the stored value *wrong* by an amount only a second
    implementation can say, rather than shifted by a term the physics
    names (`Additive`) or transformed by a closed form (`Exact`).

    Parameters
    ----------
    reference : TermFn
        Computes the superseding arrays for one block.
    rtol : float
        Relative budget the relation holds to. Measured, and ``why`` says
        how.
    why : str
        One-line justification of ``rtol``, and of ``atol`` where it is
        set.
    atol : float, optional
        Absolute floor, added to ``rtol`` when the prediction is compared.
        ``0.0`` everywhere the relation's arithmetic resolves the repaired
        value at every magnitude it takes; see "Absolute floors" in the
        module docstring for the one composition that cannot.
    """

    reference: TermFn
    rtol: float
    why: str
    atol: float = 0.0

    def expected(
        self,
        fn: Callable[..., Any],
        block: Block,
        stored: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """The repaired arrays this relation predicts, by suffix."""
        del stored  # superseded outright; that is the point of the relation
        return self.reference(fn, block)


@dataclass(frozen=True)
class Composed:
    """``repaired == base``'s prediction with each ``added`` repair applied.

    What two repairs moving the same stored array declare instead of two
    overlapping declarations (``rules.md`` rule 7): ``base`` predicts the
    array as the first repair leaves it, and every repair after that
    adds a term or transforms the preceding prediction. Further repairs
    are `Additive` or `Exact`; a `Reference` that supersedes the array
    would discard the preceding repair and cannot be composed here.
    Every step must return only suffixes present in the base prediction;
    stored arrays supply abscissae, never a missing predicted output.

    Parameters
    ----------
    base : Additive, Exact or Reference
        Predicts the array with the first repair applied and no other.
    added : tuple of Additive or Exact
        One per further repair, in the order they landed.
    rtol : float
        Relative budget the composition holds to. Measured against the
        repaired kernel; ``why`` says what sets it.
    why : str
        One-line justification of ``rtol``, and of ``atol`` where it is
        set.
    atol : float, optional
        Absolute floor, added to ``rtol`` when the prediction is compared.
        A composition whose addend *relocates* a term rather than adding
        one subtracts it back out of the base, and where the base is a
        captured array the subtraction annihilates whatever the cancelled
        term's last bit could not hold; see "Absolute floors" in the
        module docstring.
    """

    base: Additive | Exact | Reference
    added: tuple[Additive | Exact, ...]
    rtol: float
    why: str
    atol: float = 0.0

    def expected(
        self,
        fn: Callable[..., Any],
        block: Block,
        stored: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """The repaired arrays this relation predicts, by suffix."""
        # Copied, and every entry rebound rather than updated in place: a
        # `Reference` base hands back `oracle_reference`'s cached arrays,
        # which are read-only and shared by every comparison that reads
        # them.
        predicted = dict(self.base.expected(fn, block, stored))
        for addend in self.added:
            # Preserve the stored abscissae for Exact transforms while
            # replacing every value with the preceding prediction.
            updated = addend.expected(fn, block, {**stored, **predicted})
            missing = updated.keys() - predicted.keys()
            assert not missing, (
                "composition step returned suffixes missing from the base prediction: "
                f"{sorted(missing)}"
            )
            predicted.update(updated)
        return predicted


#: How a repaired value may relate to the stored one.
Relation = Additive | Exact | Reference | Composed


def repair_labels(spelling: str) -> tuple[str, ...]:
    """The roster entries one ``Delta.repair`` names, in declared order.

    A composite declaration (`Composed`) names every repair that moved the
    array, joined by ``+``; every other declaration names exactly one. The
    close aggregates per roster entry, so it needs the parts rather than
    the spelling.
    """
    return tuple(spelling.split("+"))


@dataclass(frozen=True)
class Delta:
    """One declared change to one stored array.

    Parameters
    ----------
    repair : str
        Which roster entry moved it, drawn from `REPAIRS` — or, where two
        repairs moved the same array and `Composed` collapsed them, every
        one of them joined by ``+``. `repair_labels` splits it.
    positions : tuple of int or MOVED
        Which positions the relation covers: an explicit tuple, every
        entry of which the relation must move, or `MOVED` for wherever
        the prediction differs from the stored value. Undeclared
        positions are still compared against the stored value under the
        case's budget.
    relation : Additive, Exact, Reference or Composed
        How the repaired value relates to the stored one.
    measured : str
        The measurement behind the declaration, so the table is not a
        list of assertions nobody re-derived.
    evidence : str
        Repo-relative path of the note that holds the measurement.
    """

    repair: str
    positions: Positions
    relation: Relation
    measured: str
    evidence: str


# ---------------------------------------------------------------------------
# Shared: the tabulated photon family's boosted line terms
# ---------------------------------------------------------------------------

#: `rust/src/constants.rs`, module ``pdg`` — the table the tabulated
#: photon kernels read. Spelled out rather than imported from
#: `hazma.parameters`, which carries the masses but none of the branching
#: ratios, so that a future consolidation of the two constant tables
#: cannot silently move a declaration with the code. That is the
#: convention `test/test_core_photon_tables.py` already sets for the same
#: constants.
MASS_ETA = 547.862
MASS_ETAP = 957.78
MASS_PHI = 1019.461
MASS_RHO = 775.26
BR_ETAP_TO_A_A = 2.307e-2
BR_PHI_TO_ETA_A = 1.303e-2
BR_PHI_TO_ETAP_A = 6.22e-5


def _photon_energy(parent: float, daughter: float) -> float:
    """Rest-frame photon energy in ``X -> Y gamma``, MeV.

    ``(M**2 - m**2) / (2 M)``, in the operation order
    ``photon_tables::photon_line_energy`` writes it — for the phi's two
    lines as well as the omega's, since B2. See `_daughter_energy` for
    what the phi's shipped with.
    """
    return (parent * parent - daughter * daughter) / (2.0 * parent)


def _daughter_energy(parent: float, daughter: float) -> float:
    """Rest-frame energy of the *daughter meson* in ``X -> Y gamma``, MeV.

    ``(M**2 + m**2) / (2 M)``, in the operation order the shipped
    ``photon_tables::PHI_TO_ETA_A_ENERGY`` wrote it. That constant fed it
    to the boost as if it were the photon's energy, which is what B2
    repairs; the two differ by ``m**2 / M``. The model still needs it,
    because the corpus is still pinned at the shipped energies.
    """
    return (parent * parent + daughter * daughter) / (2.0 * parent)


def _parent_beta(block: Block) -> float:
    """The boost velocity the tabulated kernel runs this block at.

    Mirrors ``photon_tables::branch``: the rest-frame arm is taken when
    the parent energy is within one epsilon MeV of the mass, and it adds
    no line at all, so the term is identically zero there. Anywhere else
    the velocity is ``boost::boost_beta``, taken from the crate so that
    the window edges a declaration resolves `MOVED` against are the
    kernel's own rather than a re-rounded copy of them.
    """
    energy = block.params["parent_energy"]
    mass = block.params["parent_mass"]
    if energy - mass < np.finfo(np.float64).eps:
        return 0.0
    return float(core_boost.boost_beta(energy, mass))


def _boosted_line(
    e0: float, weight: float, grid: np.ndarray, beta: float
) -> np.ndarray:
    """One rest-frame line's contribution to the boosted spectrum, MeV⁻¹.

    ``weight`` photons per decay spread flat across the window the boost
    opens, and exactly zero outside it — `boost::boost_delta_function`,
    which the tabulated kernel calls once per line.
    """
    return weight * np.asarray(
        core_boost.boost_delta_function(
            e0, np.asarray(grid, dtype=np.float64), 0.0, beta
        ),
        dtype=np.float64,
    )


def _on_value_grids(
    block: Block, of_energy: Callable[[np.ndarray], np.ndarray]
) -> dict[str, np.ndarray]:
    """Evaluate a function of photon energy on a block's value grids."""
    terms = {"values": of_energy(block.grid)}
    probe = block.scalar_probe
    if probe.size:
        terms["scalar_values"] = of_energy(probe)
    return terms


# ---------------------------------------------------------------------------
# B1 -- the eta-prime two-photon line carries one branching ratio, not two
# ---------------------------------------------------------------------------


def _eta_prime_line_second_copy(
    fn: Callable[..., Any], block: Block
) -> dict[str, np.ndarray]:
    """The copy of the ``eta' -> gamma gamma`` line the shipped weight drops.

    Two photons leave the decay, so the line's weight is ``2 BR``, which
    is what the eta and both neutral kaons write. The eta-prime writes a
    bare ``BR`` (``photon_tables::ETAP_TO_A_A_WEIGHT``), so the repaired
    spectrum is the stored one plus a second copy of the same term — and
    the term is closed-form, needing neither the repaired kernel nor a
    Cython twin.
    """
    del fn  # the term is a closed form, not a re-evaluation of the spectrum
    beta = _parent_beta(block)
    return _on_value_grids(
        block,
        lambda grid: _boosted_line(MASS_ETAP / 2.0, BR_ETAP_TO_A_A, grid, beta),
    )


_B1 = Delta(
    repair="B1",
    positions=MOVED,
    relation=Additive(
        term=_eta_prime_line_second_copy,
        rtol=1e-11,
        why="the term is closed form, so what sets this is the continuum "
        "underneath it: the unchanged tabulated boost is already allowed "
        "tolerances.TABULATED_RTOL = 1e-12 off the capturing tree, and the "
        "relation compares totals. One decade of headroom over that. The "
        "line term itself reproduces the corpus's own plateau step to "
        "4.9e-13 worst over 22 line/block pairs "
        "(test_delta_models.py::"
        "test_a_tabulated_line_carries_the_weight_the_corpus_stores).",
    ),
    measured="the plateau the stored eta-prime spectrum carries at the "
    "boosted image of M/2 is BR_ETAP_TO_A_A / (2 gamma beta e0) — one "
    "branching ratio, where its three correct siblings (eta, K_L, K_S) "
    "carry two. Recovered from the committed arrays alone, to 4.9e-13, in "
    "all four boosted blocks. The rest block takes the kernel's rest-frame "
    "arm, which adds no line, so nothing moves there: 189 positions over "
    "six arrays.",
    evidence="projects/parity-pinned-defect-repair/task-notes/task-3-closed-form-deltas.md",
)


# ---------------------------------------------------------------------------
# B2 -- both phi lines sit at the daughter meson's energy, not the photon's
# ---------------------------------------------------------------------------

#: ``phi -> Y gamma`` for the two daughters the kernel gives a line, with
#: the branching ratio each line carries.
PHI_LINE_DAUGHTERS = ((MASS_ETA, BR_PHI_TO_ETA_A), (MASS_ETAP, BR_PHI_TO_ETAP_A))


def _phi_lines_relocated(fn: Callable[..., Any], block: Block) -> dict[str, np.ndarray]:
    """Both phi lines moved from the daughter's energy to the photon's.

    Neither weight changes and no channel opens or closes: the repair
    subtracts each line where the shipped kernel puts it and adds it back
    at ``(M**2 - m**2) / (2 M)``. Both energies are closed forms, so the
    term needs no kernel evaluation — and because the total yield is
    conserved, a declaration that named only a magnitude would pass on an
    unrepaired kernel. This one names the positions instead.
    """
    del fn  # both line energies are closed forms, not kernel outputs

    beta = _parent_beta(block)

    def relocation(grid: np.ndarray) -> np.ndarray:
        moved = np.zeros(np.shape(grid), dtype=np.float64)
        for daughter, weight in PHI_LINE_DAUGHTERS:
            moved += _boosted_line(
                _photon_energy(MASS_PHI, daughter), weight, grid, beta
            )
            moved -= _boosted_line(
                _daughter_energy(MASS_PHI, daughter), weight, grid, beta
            )
        return moved

    return _on_value_grids(block, relocation)


_B2 = Delta(
    repair="B2",
    positions=MOVED,
    relation=Additive(
        term=_phi_lines_relocated,
        rtol=1e-11,
        why="closed form on both sides of the move, so the budget is the "
        "continuum's, exactly as for B1: tolerances.TABULATED_RTOL = 1e-12 "
        "with a decade of headroom.",
    ),
    measured="the stored phi spectrum's top is a two-tread staircase, "
    "3.1077e-05 then 1.0122e-07 then exactly 0.0, whose treads are the two "
    "shipped lines' plateau heights at 656.942002472385 and "
    "959.6459594437648 MeV — recovered from the committed arrays to 0.0 "
    "and 1.8e-15 relative. Both windows sit above the boosted continuum's "
    "support, where the repaired kernel has none: 305 positions over six "
    "arrays, 233 up and 72 down. The two rest blocks do not move — no "
    "grid point falls inside either window at beta = 1.414e-06.",
    evidence="projects/parity-pinned-defect-repair/task-notes/task-3-closed-form-deltas.md",
)


# ---------------------------------------------------------------------------
# B3 -- the rho rest-frame branch returns the boost integrand
# ---------------------------------------------------------------------------


def _rho_rest_frame_spectrum(
    block: Block, stored: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """The rest-frame spectrum the stored integrand is missing its ``E`` from.

    ``photon_rho::boosted`` short circuits at ``E_rho == m_rho`` and
    returns ``integrand(E)``, which carries the ``1 / E'`` belonging to
    the boost kernel rather than to the spectrum — MeV⁻² where the other
    branch is MeV⁻¹. Multiplying it back is the whole repair, so the
    repaired array is a transform of the stored one and no kernel is
    evaluated to predict it.

    Off that branch the repair changes nothing, and this returns the
    stored arrays unchanged rather than a transform of them: a block the
    guard does not admit moves at no position, which is what a
    declaration keyed on one would fail as stale.
    """
    energy = block.params["parent_energy"]
    on_the_rest_branch = (
        energy >= MASS_RHO and energy - MASS_RHO < np.finfo(np.float64).eps
    )
    return {
        values: stored[values] * stored[grid] if on_the_rest_branch else stored[values]
        for values, grid in (("values", "grid"), ("scalar_values", "scalar_grid"))
        if values in stored
    }


_B3 = Delta(
    repair="B3",
    positions=MOVED,
    relation=Exact(
        transform=_rho_rest_frame_spectrum,
        rtol=1e-9,
        why="one multiplication by the abscissa the repaired kernel "
        "multiplies by, so on the capturing tree the prediction is "
        "bit-identical and the relation adds no error of its own. Off it, "
        "what is left is the nested pion quadrature the integrand calls, "
        "which is the case's own tolerances.PORTED_NESTED_RTOL = 1e-9 — "
        "so the relation is held to the budget the case already holds "
        "rather than to a looser one.",
    ),
    measured="stored x E_gamma reproduces the same case's rest_plus_eps "
    "block — the beta -> 0 limit of the branch that is correct — to "
    "6.8e-11 (charged) and 1.8e-09 (neutral) worst over the 170 paired "
    "non-zero positions each, with matching zero patterns. The ratio to "
    "the stored value is E_gamma at every position, spanning 7.75e-03 to "
    "7.75e+04 MeV: 350 positions over four arrays, all of them the rest "
    "block, which is the only parent energy the guard admits.",
    evidence="projects/parity-pinned-defect-repair/task-notes/task-3-closed-form-deltas.md",
)


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
    positions=MOVED,
    relation=Additive(
        term=_scalar_decay_fsr_half,
        rtol=1e-3,
        why="the term is its own cos(theta) quadrature (epsrel 1e-5) over a "
        "different integrand than the stored total, and the repaired total "
        "is a third; measured 3.1e-4 worst relative over the 3,065 "
        "declared positions, at ms_550.boosted_strong E=2696 MeV where the "
        "boost window is narrow and the integrator's own error estimate is "
        "what moves. Three times headroom; the pre-repair arrays sit a "
        "factor of two away where FSR is the only open channel.",
    ),
    measured="the FSR-only spectrum at rest is 0.5000000000 x the annihilation-"
    "side ScalarMediator.dnde_xx_to_s_to_ffg / dnde_xx_to_s_to_pipig at the "
    "same invariant mass, in every channel, to ten digits after removing the "
    "legacy/PDG alpha ratio; the vector twin is 1.000 x its own. The FSR "
    "term is non-zero at 3,065 of the 4,305 pinned positions; 2,874 of "
    "those move by more than 0.1%, up to exactly double.",
    evidence="docs/followups/done/scalar-decay-fsr-half-normalized.md",
)

# ---------------------------------------------------------------------------
# B5 -- the charged pion's prompt "pi -> e nu" neutrino line was added twice
# ---------------------------------------------------------------------------

#: ``BR(pi -> e nu_e)``, as ``rust/src/constants.rs``'s ``pdg`` module
#: spells it. A literal because ``hazma.parameters`` has no branching
#: ratios in it -- the masses the term needs do come from there, and agree
#: with the crate's ``pdg`` constants bit for bit, which is what keeps the
#: window decision below on the same side of every grid point.
BR_PI_TO_E_NUE = 1.230e-4


def _pion_electron_line(_fn: Callable[..., Any], block: Block) -> dict[str, np.ndarray]:
    """Minus one boosted ``pi -> e nu_e`` line, on the block's grids.

    A rest-frame line at ``e0`` boosts to a flat plateau of height
    ``1 / (2 gamma beta e0)`` across the lab energies whose boost window
    ``[gamma E (1 - beta), gamma E (1 + beta)]`` straddles ``e0``, and zero
    elsewhere; the shipped kernel added ``BR_e`` of it in both halves of
    the sum, so ``stored = repaired + BR_e * plateau``. Only the electron
    row carries it -- the ``pi -> e nu_e`` half writes nothing to the other
    two -- so the term is zero everywhere else, and `MOVED` leaves those
    positions at the case's own budget.
    """
    epi = block.params["parent_energy"]
    mpi = block.params["parent_mass"]
    e0 = (mpi * mpi - parameters.electron_mass**2) / (2.0 * mpi)
    ratio = mpi / epi
    beta = math.sqrt(1.0 - ratio * ratio)

    def line(energies: np.ndarray) -> np.ndarray:
        # A line has no rest-frame representation and no width to boost,
        # so a parent at rest carries none -- the same answer, and for the
        # same reason, as ``boost.boost_delta_function``'s ``beta <= 0``
        # guard. Without this the height below divides by zero.
        if beta <= 0.0:
            return np.zeros_like(energies)
        gamma = 1.0 / math.sqrt(1.0 - beta * beta)
        inside = (gamma * energies * (1.0 - beta) < e0) & (
            e0 < gamma * energies * (1.0 + beta)
        )
        return np.where(inside, -BR_PI_TO_E_NUE / (2.0 * gamma * beta * e0), 0.0)

    # Row 0 of a (3, n) values array and column 0 of an (n, 3) scalar one:
    # both are the electron flavor, `cases` stores the two orientations.
    terms = {}
    values = np.zeros((3, block.grid.size), dtype=np.float64)
    values[0] = line(block.grid)
    terms["values"] = values
    probe = block.scalar_probe
    if probe.size:
        scalar_values = np.zeros((probe.size, 3), dtype=np.float64)
        scalar_values[:, 0] = line(probe)
        terms["scalar_values"] = scalar_values
    return terms


_B5 = Delta(
    repair="B5",
    positions=MOVED,
    relation=Additive(
        term=_pion_electron_line,
        rtol=3e-12,
        why="three times the case's own PORTED_QUAD_RTOL (1e-12), derived "
        "rather than fitted: the term carries no quadrature, so the only "
        "slack the relation needs is the platform drift already between the "
        "stored value and the live one, and the comparison denominator "
        "stored + term is at most 2x smaller than stored (the doubled line "
        "cannot exceed the stored value), so that budget is amplified by at "
        "most two. The closed form's own disagreement with the kernel's "
        "boost_delta_function -- an FMA in the gamma fold this expression "
        "does not spell -- adds under 1.5e-15 of the compared value. Worst "
        "measured over the 215 declared positions: 1.494e-15, in "
        "boosted_strong.",
    ),
    measured="the repair removes exactly one BR_e = 1.230e-4 from the "
    "electron-neutrino row per pion, and moves 215 of the case's 4,305 pinned "
    "values, all downward and all in the electron row: 35 in near_rest, 69 in "
    "boosted_mild, 111 in boosted_strong, and none in rest or rest_plus_eps, "
    "where beta is too small for any grid point's window to straddle the "
    "line. The drop runs from 4.716e-5 relative, on the plateau where the "
    "muon-decay continuum dominates, to exactly 0.500000000000 at the 14 "
    "positions where that continuum's quadrature returns zero and the "
    "doubled line was the entire value.",
    evidence="docs/followups/done/neutrino-pion-electron-line-counted-twice.md",
)

# ---------------------------------------------------------------------------
# B6 -- the thermal averages never converged
# ---------------------------------------------------------------------------


_B6 = Delta(
    repair="B6",
    positions=MOVED,
    relation=Reference(
        reference=thermal_reference.reference_values,
        rtol=1e-7,
        why="the reference is scipy's QUADPACK at epsrel 1e-12 over the same "
        "integrand, so what bounds agreement is the repaired kernels' own "
        "epsrel of 1.49e-8, not the reference: a platform whose libm steers "
        "QUADPACK to a different accepted partition may land anywhere inside "
        "it. Measured 3.6e-9 worst relative over the 540 positions the "
        "reference integrates. 1e-7 is 6.7x the bound that has to hold "
        "everywhere, rather than 28x the figure this platform happens to "
        "give.",
    ),
    measured="the stored arrays are the initial-partition estimate: against "
    "the reference they are wrong by up to 1.00 relative (scalar and vector "
    "closed_resonance, where the shipped value retains none of the true "
    "one), with per-block medians from 7.2e-6 to 8.1e-2. 539 of the 570 "
    "pinned positions move. Of the 31 that do not, 30 are the ten points per "
    "scalar block above x = 300, where that kernel returns 0.0 outright and "
    "the quadrature is never reached; the last is vector narrow_resonance at "
    "x = 0.1367, small enough that the relative criterion already bound "
    "before the repair.",
    evidence="docs/followups/done/thermal-cross-section-quadrature-never-converges.md",
)

# ---------------------------------------------------------------------------
# A1 -- the boost integral mis-covers its window at both ends
# ---------------------------------------------------------------------------


#: The seven tabulated photon cases `boost_integrate_linear_interp`
#: reaches. Each contributes four keys below -- ``rest_plus_eps``,
#: ``near_rest``, ``boosted_mild`` and ``boosted_strong``, both value
#: suffixes. ``rest`` is deliberately absent from all seven: every one of
#: these kernels short-circuits to its rest-frame spectrum at
#: ``E - m < DBL_EPSILON``, so the integral never runs at ``beta = 0`` and
#: those 1,841 pinned positions must still match the stored arrays.
A1_CASES = (
    "spectra.photon.eta",
    "spectra.photon.eta_prime",
    "spectra.photon.omega",
    "spectra.photon.phi",
    "spectra.photon.charged_kaon",
    "spectra.photon.long_kaon",
    "spectra.photon.short_kaon",
)

_A1 = Delta(
    repair="A1",
    positions=MOVED,
    relation=Reference(
        reference=oracle_reference.captured("A1"),
        rtol=1e-12,
        why="the reference is the same routine with the same window "
        "coverage, compiled from the pre-port Cython, so the only thing "
        "between it and the repaired kernel is the port's own arithmetic "
        "-- which is what `tolerances.TABULATED_RTOL` already budgets for "
        "these seven cases, and this is that number. Measured 0.0: the "
        "repaired kernel reproduces the capture at all 10,045 pinned "
        "positions bit for bit, as the unrepaired port reproduced the "
        "corpus. Not tightened to zero, because the capture is one "
        "platform's (`test_oracles.py` holds it against the corpus "
        "manifest's) and a libm that moves the boost integral has to be "
        "held exactly as tightly here as the undeclared positions are, "
        "not tighter.",
    ),
    measured="4,154 of the 10,045 pinned positions move, in four of the "
    "five blocks of each case and in none of the `rest` blocks. The sign "
    "splits by block rather than by case: `rest_plus_eps` moves down at "
    "all 1,156 of its positions, where the two partial-cell terms "
    "overlapped and the shipped value is a median 9,768x and up to "
    "360,507x too high; `near_rest`, `boosted_mild` and `boosted_strong` "
    "move up at 2,997 of their 2,998, by up to 98.7%, where the dropped "
    "interior cell left the value low. Per case: eta 560, eta_prime 552, "
    "omega 633, phi 551, charged_kaon 631, long_kaon 631, short_kaon 596.",
    evidence="projects/parity-pinned-defect-repair/task-notes/task-4-boost-window.md",
)


# A2 -- restore the muon rest-frame endpoint, retaining the signed formula.
_A2 = Delta(
    repair="A2",
    positions=(161, 162, 163, 164),
    relation=Reference(
        reference=oracle_reference.captured("A2"),
        rtol=1e-9,
        why="the independent patched-Cython capture uses the same rest-frame "
        "arithmetic; the endpoint polynomial cancels, so reserve the case's "
        "existing PLATFORM_SPECFUN_RTOL across libm implementations, with "
        "no absolute floor. "
        "The four declared values agree bit for bit on the capturing platform.",
    ),
    measured="Only spectra.photon.muon/rest/values moves: positions 161-164 "
    "change from zero to -2.996e-9 through -2.916e-9 MeV^-1. All six "
    "candidate composed corpus cases remain unchanged. The signed tail "
    "is retained to preserve the rest-to-flight boost identity.",
    evidence="projects/parity-pinned-defect-repair/task-notes/task-7-photon-muon-endpoint.md",
)


# A3 -- restrict the charged-pion angular integral to its physical support.
_A3 = Delta(
    repair="A3",
    positions=MOVED,
    relation=Reference(
        reference=oracle_reference.captured("A3"),
        rtol=tolerances.PORTED_QUAD_RTOL,
        why="independently compiled Task 2 Cython capture of the same clipped "
        "pion quadrature; measured at most 4.12e-16 relative on macOS. "
        "Retains the pion case's existing 1e-12 arithmetic budget.",
    ),
    measured="The before/after builds move 6,359 of 71,570 values in six "
    "corpus cases. The pion moves 245 positions, each rho 528, each vector "
    "entry point 2,013, and the scalar mediator 1,032. Rho rest positions "
    "move too: Task 9 composes B3 with A3. The outer rho support "
    "defect is a separate follow-up.",
    evidence="projects/parity-pinned-defect-repair/task-notes/task-8-charged-pion-cone.md",
)

# The nested consumers retain their existing case budget. One-platform
# agreement with a compiled capture cannot tighten their portability contract.
_A3_NESTED = replace(
    _A3,
    relation=Reference(
        reference=oracle_reference.captured("A3"),
        rtol=tolerances.PORTED_NESTED_RTOL,
        why="rho and vector mediator consumers retain their existing 1e-9 "
        "nested-quadrature case budget. PR #97 CI run 35423602586 measured "
        "9.61566611e-12 relative on Linux for mv_900.rest.total, versus "
        "at most 1.55e-13 on macOS. The pion-only model keeps 1e-12; "
        "no corpus tolerance or quadrature option changes.",
    ),
)


_A3_B3 = Delta(
    repair="A3+B3",
    positions=MOVED,
    relation=Composed(
        base=_A3_NESTED.relation,
        added=(_B3.relation,),
        rtol=1e-9,
        why="B3 multiplies the A3 Cython capture by E_gamma. The single "
        "multiplication retains the rho consumers' existing nested 1e-9 "
        "budget; no absolute floor is needed.",
    ),
    measured="B3 moves 350 positions in four rho rest arrays, 170 vector "
    "and five scalar positions per species. All other blocks are unchanged. "
    "The A3 capture times E_gamma agrees within 4.16e-16 relative on the "
    "capturing platform; the nested portability budget stays 1e-9.",
    evidence="projects/parity-pinned-defect-repair/task-notes/task-9-rho-rest-frame.md",
)


_A3_B4 = Delta(
    repair="A3+B4",
    positions=MOVED,
    relation=Composed(
        base=_A3.relation,
        added=(_B4.relation,),
        rtol=_B4.relation.rtol,
        why="A3's captured scalar decay includes the original half-size FSR; "
        "add B4's independently evaluated FSR half. The total and term use "
        "different adaptive partitions, retaining B4's measured 1e-3 "
        "budget (3.08e-4 worst here). A dedicated rest-frame comparison "
        "holds A3 to 1e-12 without this quadrature cancellation.",
    ),
    measured="A3 changes 1,032 positions already covered by B4, all in "
    "default-mode blocks at scalar masses 550 and 900 MeV. Their 20 "
    "arrays compose; B4 keeps the ten arrays at 250 MeV alone.",
    evidence="projects/parity-pinned-defect-repair/task-notes/task-8-charged-pion-cone.md",
)


# A4 -- multiply the Michel spectrum by its normalization instead of dividing.
# One model per budget class, as for A3: each case keeps its own corpus
# budget, because agreement with a one-platform capture cannot tighten a
# portability contract and must not loosen one.
_A4 = Delta(
    repair="A4",
    positions=MOVED,
    relation=Reference(
        reference=oracle_reference.captured("A4"),
        rtol=tolerances.EXACT_RTOL,
        why="the Task 2 capture evaluates the same closed form from a "
        "patched build of the pre-port Cython, and the repaired kernel "
        "spells the two changed expressions in the same operation order, so "
        "spectra.positron.muon keeps its EXACT case budget, platform branch "
        "included. Measured 0.0 on the capturing macOS/arm64 libm, "
        "bit-for-bit at all 1,370 pinned values; up to 6.8e-12 on Linux "
        "x86_64 CI, inside PLATFORM_EXACT_RTOL and far below the 3.7e-4 "
        "the repair moves.",
    ),
    measured="Every nonzero value rises by exactly R_FACTOR**2 = "
    "1.000374206647938 wherever the muon kernel is the only contributor: "
    "502 positions in all ten value arrays of spectra.positron.muon, 525 "
    "in eight of spectra.positron.charged_pion's ten (a pion exactly at "
    "rest returns zero), and 5,237 in 40 arrays of each of the four "
    "mediator positron entry points -- 21,975 of the 69,830 values the "
    "capture covers, the count Task 2 predicted. No other corpus value "
    "moves.",
    evidence="projects/parity-pinned-defect-repair/task-notes/task-10-positron-muon-norm.md",
)

_A4_PION = replace(
    _A4,
    relation=Reference(
        reference=oracle_reference.captured("A4"),
        rtol=tolerances.PORTED_QUAD_RTOL,
        why="the pion boosts the muon spectrum through one quad, and the "
        "capture ran that quadrature in Cython; measured at most 5.30e-15 "
        "relative, within the case's existing 1e-12 budget.",
    ),
)

_A4_NESTED = replace(
    _A4,
    relation=Reference(
        reference=oracle_reference.captured("A4"),
        rtol=tolerances.PORTED_NESTED_RTOL,
        why="the mediator positron spectra nest the pion quadrature inside "
        "an angular one; measured at most 3.55e-12 (scalar) and 6.40e-12 "
        "(vector) relative against the capture, within the cases' existing "
        "1e-9 nested budget. The unrepaired port sat 2.33e-12 and 1.50e-12 "
        "from the corpus on the same arrays.",
    ),
)

#: The blocks every A4 case samples, in corpus order.
_A4_BLOCKS = ("rest", "rest_plus_eps", "near_rest", "boosted_mild", "boosted_strong")

#: The mediator decay channels whose positrons come from a muon, by mediator
#: mass in MeV. ``e_e`` is a line the muon kernel never reaches, and
#: ``pi_pi`` is closed at 250 MeV, below twice the charged-pion mass.
_A4_MEDIATOR_CHANNELS = {
    250: ("total", "mu_mu"),
    550: ("total", "mu_mu", "pi_pi"),
    900: ("total", "mu_mu", "pi_pi"),
}

#: The 178 arrays A4 moves: the allowlist ``rules.md`` rule 5 asks for,
#: written as the product it is rather than transcribed key by key.
_A4_DECLARATIONS: dict[tuple[str, str, str], Delta] = {
    **{
        ("spectra.positron.muon", block, suffix): _A4
        for block in _A4_BLOCKS
        for suffix in ("values", "scalar_values")
    },
    **{
        ("spectra.positron.charged_pion", block, suffix): _A4_PION
        for block in _A4_BLOCKS
        if block != "rest"
        for suffix in ("values", "scalar_values")
    },
    **{
        (
            f"mediator_spectra.{kind}.positron.dnde_decay_{m}{pt}",
            f"m{m}_{mass}.{block}.{channel}",
            "values",
        ): _A4_NESTED
        for kind, m in (("scalar", "s"), ("vector", "v"))
        for pt in ("", "_pt")
        for mass, channels in _A4_MEDIATOR_CHANNELS.items()
        for block in _A4_BLOCKS
        for channel in channels
    },
}


# ---------------------------------------------------------------------------
# C1 -- the mediator positron line carries the electron's velocity
# ---------------------------------------------------------------------------

#: `rust/src/constants.rs`, module ``legacy`` -- the electron mass the
#: mediator positron kernel reads. Spelled out for the reason the photon
#: masses above are: ``hazma.parameters.electron_mass`` is the modern
#: 0.5109989461, and a consolidation of the two must not move this model.
LEGACY_MASS_E = 0.510998928


def _mul_add(a: float, b: float, c: float) -> float:
    """``a * b + c`` rounded once, as Rust's ``f64::mul_add`` rounds it.

    `math.fma` needs Python 3.13. `Fraction` arithmetic is exact, and
    converting the exact result back to a float rounds it once, so this
    is the fused value rather than an approximation of it.
    """
    return float(Fraction(a) * Fraction(b) + Fraction(c))


def _electron_line_velocity(
    fn: Callable[..., Any], block: Block
) -> dict[str, np.ndarray]:
    """The part of the ``S/V -> e+ e-`` box the shipped height left out.

    Boosted, the rest-frame line at ``m/2`` is a flat box between
    ``E (1 -+ r beta) / 2``, where ``r = sqrt(1 - 4 m_e**2 / m**2)`` is the
    electron's rest-frame velocity. The box is ``E r beta`` wide, so one
    positron per decay makes it ``pw_ee / (E r beta)`` tall; the shipped
    kernel wrote ``pw_ee / (E beta)``, and the repaired kernel divides
    that by ``r``. The term is the difference, inside the window and
    nowhere else. Every recognised mode string adds the line, so the term
    is the same for all four of a block's modes.

    The window edges are the kernel's own -- ``mediator_decay_positron::
    spectrum_point`` fuses ``r beta + 1`` -- because the corpus grids
    carry anchors on exactly those edges. With the mediator at rest the
    box is a single point where both heights are infinite, so nothing
    moves; below its mass the kernel returns zero.
    """
    del fn  # closed form in the block's parameters
    energy = block.params["mediator_energy"]
    mass = block.params["mediator_mass"]
    pw_ee = block.params["partial_widths"][0]
    if energy <= mass:
        return _on_value_grids(block, np.zeros_like)
    ratio = mass / energy
    beta = math.sqrt(1.0 - ratio * ratio)
    r = math.sqrt(1.0 - ((4.0 * LEGACY_MASS_E) * LEGACY_MASS_E) / (mass * mass))
    eplus = (energy * _mul_add(r, beta, 1.0)) / 2.0
    eminus = (energy * _mul_add(-r, beta, 1.0)) / 2.0
    shipped = pw_ee / (energy * beta)
    missing = shipped / r - shipped
    return _on_value_grids(
        block, lambda e: np.where((eminus <= e) & (e <= eplus), missing, 0.0)
    )


_C1 = Delta(
    repair="C1",
    positions=MOVED,
    relation=Additive(
        term=_electron_line_velocity,
        rtol=tolerances.PORTED_NESTED_RTOL,
        why="the case's own budget. The term is closed form and the arrays "
        "C1 declares alone have no continuum under the line -- every `e_e` "
        "block, and `pi_pi` at 250 MeV, below the pion threshold -- so the "
        "stored line plus the term reproduces the repaired kernel bit for "
        "bit: measured 0.0 over all 3,000 positions it moves.",
    ),
    measured="Every position inside the boosted `e+ e-` window moves "
    "by the line's height times `1/r - 1`: 8.356e-06 at 250 MeV, 1.726e-06 "
    "at 550 MeV and 6.447e-07 at 900 MeV, relative to the line. Over the "
    "four mediator positron cases that is 8,912 positions in 192 arrays -- "
    "2,228 per case, in all four modes of the `rest_plus_eps`, "
    "`near_rest`, `boosted_mild` and `boosted_strong` blocks of all three "
    "masses. 6,596 of them move by more than the cases' 1e-9 budget; the "
    "rest sit under a continuum large enough to hide the shift. Nothing "
    "moves at `rest`, where the box is one point of infinite height.",
    evidence="docs/followups/done/mediator-positron-line-misses-the-electron-velocity.md",
)

_A4_C1 = Delta(
    repair="A4+C1",
    positions=MOVED,
    relation=Composed(
        base=_A4_NESTED.relation,
        added=(_C1.relation,),
        rtol=tolerances.PORTED_NESTED_RTOL,
        why="the base is the A4 capture at A4's nested budget and the "
        "addend is closed form, so the composition inherits A4's residual: "
        "measured at most 3.55e-12 (scalar) and 6.40e-12 (vector) relative "
        "over the 5,912 positions C1 moves in these arrays, the figures "
        "A4 alone measured on them.",
    ),
    measured="C1 moves 5,912 positions in 128 arrays that A4 already "
    "declares: `total` and `mu_mu` at every mass, and `pi_pi` at 550 and "
    "900 MeV, in the four blocks above rest. They compose rather than "
    "declaring separately because one key holds one declaration. The "
    "`rest` block of each keeps A4's declaration alone.",
    evidence=_C1.evidence,
)

#: Every block the mediator positron line moves in, with the mode strings'
#: labels. ``rest`` is absent: see `_electron_line_velocity`.
_C1_BLOCKS = ("rest_plus_eps", "near_rest", "boosted_mild", "boosted_strong")

#: The arrays C1 moves. Where A4 already declares the array, the two
#: compose, since one key holds one `Delta`; the rest -- every ``e_e``
#: block, and ``pi_pi`` at 250 MeV where the pion channel is closed and
#: the line is all the array holds -- carry C1 alone.
_C1_DECLARATIONS: dict[tuple[str, str, str], Delta] = {
    key: (_A4_C1 if key in _A4_DECLARATIONS else _C1)
    for key in (
        (
            f"mediator_spectra.{kind}.positron.dnde_decay_{m}{pt}",
            f"m{m}_{mass}.{block}.{channel}",
            "values",
        )
        for kind, m in (("scalar", "s"), ("vector", "v"))
        for pt in ("", "_pt")
        for mass in (250, 550, 900)
        for block in _C1_BLOCKS
        for channel in ("total", "e_e", "mu_mu", "pi_pi")
    )
}


# ---------------------------------------------------------------------------
# A1 + B1 -- the six eta-prime arrays both repairs move
# ---------------------------------------------------------------------------


_A1_B1 = Delta(
    repair="A1+B1",
    positions=MOVED,
    relation=Composed(
        base=_A1.relation,
        added=(_B1.relation,),
        rtol=1e-12,
        why="the base is the A1 capture and the addend is closed form, so "
        "what separates the prediction from the repaired kernel is the "
        "order the two line copies are summed in: the kernel folds one "
        "`2 BR` weight into the boost's own fused multiply-add, the "
        "prediction adds a second `BR` copy afterwards. Measured 2.1e-16 "
        "worst over the six arrays, which is one ulp. Held at the case's "
        "own `tolerances.TABULATED_RTOL` rather than tightened to that, "
        "for the reason A1 gives: the capture is one platform's, and a "
        "libm that moves the boost integral must not fail here while the "
        "undeclared positions of the same block still pass.",
    ),
    measured="B1 moves 189 positions over six of this case's ten value "
    "arrays -- 9 of `rest_plus_eps.values`, 20 of `near_rest.values`, 57 "
    "and 2 of `boosted_mild.{values,scalar_values}`, 98 and 3 of "
    "`boosted_strong.{values,scalar_values}` -- every one of them upward, "
    "by 7.7e-04 to 1.0 relative. 18 of the 189 are positions A1 moves too, "
    "which is why the two compose rather than declaring separately. The "
    "case's other four value arrays keep A1's declaration alone: B1 moves "
    "nothing in either `rest` array, where the kernel takes its rest-frame "
    "arm and adds no line, and nothing at the scalar probe of "
    "`rest_plus_eps` or `near_rest`, where the probe falls outside the "
    "line's window.",
    evidence="projects/parity-pinned-defect-repair/task-notes/task-5-eta-prime-line.md",
)


# ---------------------------------------------------------------------------
# A1 + B2 -- the six phi arrays both repairs move
# ---------------------------------------------------------------------------


_A1_B2 = Delta(
    repair="A1+B2",
    positions=MOVED,
    relation=Composed(
        base=_A1.relation,
        added=(_B2.relation,),
        rtol=1e-12,
        atol=1e-20,
        why="the base is the A1 capture and the addend is closed form, so "
        "the relative budget is the case's own `tolerances.TABULATED_RTOL`, "
        "for the reason A1 gives: the capture is one platform's, and a libm "
        "that moves the boost integral must not fail here while the "
        "undeclared positions of the same block still pass. Measured 2.9e-14 "
        "worst over the six arrays. The floor is what relocation costs: the "
        "addend subtracts the shipped line back out of the capture, and at "
        "the two positions inside the shipped `phi -> eta gamma` window but "
        "outside both repaired windows that cancellation takes the continuum "
        "with it -- the plateau there is 3.107723e-05, whose last bit is "
        "6.8e-21, and the continuum underneath it is 3.3e-21. So the "
        "prediction reads exactly 0.0 where the kernel returns 3.3e-21 and "
        "3.1e-22, and one ulp of the cancelled plateau is the tightest "
        "statement available.",
    ),
    measured="B2 moves 305 positions over six of this case's ten value "
    "arrays -- 57 and 1 of `near_rest.{values,scalar_values}`, 102 and 3 of "
    "`boosted_mild.{values,scalar_values}`, 139 and 3 of "
    "`boosted_strong.{values,scalar_values}` -- 233 of them up and 72 down, "
    "which is what relocating a line rather than adding one looks like. 90 "
    "of the 305 are positions A1 moves too, so the two compose rather than "
    "declaring separately. The case's other four value arrays keep A1's "
    "declaration alone: both `rest` arrays take the kernel's rest-frame arm, "
    "which adds no line at all, and at `rest_plus_eps` the boost opens a "
    "window 2.8e-06 wide in relative energy at beta = 1.414e-06, too narrow "
    "for any grid point to fall inside a shipped or a repaired line.",
    evidence="projects/parity-pinned-defect-repair/task-notes/task-6-phi-lines.md",
)


#: Every declared array. The two blocks of the same case that are absent --
#: ``rest`` and ``rest_plus_eps`` -- must still match the stored arrays under
#: the case's own budget, which is the "moved only what it intended" half of
#: the B5 proof: at rest the kernel drops both prompt lines, and one epsilon
#: above it no grid point's boost window is wide enough to straddle the line.
DECLARED_DELTAS: dict[tuple[str, str, str], Delta] = {
    # A3: captured inner-pion repair, including the rho rest blocks.
    (
        "mediator_spectra.vector.photon.dnde_decay_v",
        "mv_550.rest.total",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v",
        "mv_550.rest.pi_pi",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v",
        "mv_550.rest_plus_eps.total",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v",
        "mv_550.rest_plus_eps.pi_pi",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v",
        "mv_550.near_rest.total",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v",
        "mv_550.near_rest.pi_pi",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v",
        "mv_550.boosted_mild.total",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v",
        "mv_550.boosted_mild.pi_pi",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v",
        "mv_550.boosted_strong.total",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v",
        "mv_550.boosted_strong.pi_pi",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v",
        "mv_900.rest.total",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v",
        "mv_900.rest.pi_pi",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v",
        "mv_900.rest_plus_eps.total",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v",
        "mv_900.rest_plus_eps.pi_pi",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v",
        "mv_900.near_rest.total",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v",
        "mv_900.near_rest.pi_pi",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v",
        "mv_900.boosted_mild.total",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v",
        "mv_900.boosted_mild.pi_pi",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v",
        "mv_900.boosted_strong.total",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v",
        "mv_900.boosted_strong.pi_pi",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v_pt",
        "mv_550.rest.total",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v_pt",
        "mv_550.rest.pi_pi",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v_pt",
        "mv_550.rest_plus_eps.total",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v_pt",
        "mv_550.rest_plus_eps.pi_pi",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v_pt",
        "mv_550.near_rest.total",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v_pt",
        "mv_550.near_rest.pi_pi",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v_pt",
        "mv_550.boosted_mild.total",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v_pt",
        "mv_550.boosted_mild.pi_pi",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v_pt",
        "mv_550.boosted_strong.total",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v_pt",
        "mv_550.boosted_strong.pi_pi",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v_pt",
        "mv_900.rest.total",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v_pt",
        "mv_900.rest.pi_pi",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v_pt",
        "mv_900.rest_plus_eps.total",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v_pt",
        "mv_900.rest_plus_eps.pi_pi",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v_pt",
        "mv_900.near_rest.total",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v_pt",
        "mv_900.near_rest.pi_pi",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v_pt",
        "mv_900.boosted_mild.total",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v_pt",
        "mv_900.boosted_mild.pi_pi",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v_pt",
        "mv_900.boosted_strong.total",
        "values",
    ): _A3_NESTED,
    (
        "mediator_spectra.vector.photon.dnde_decay_v_pt",
        "mv_900.boosted_strong.pi_pi",
        "values",
    ): _A3_NESTED,
    ("spectra.photon.charged_pion", "near_rest", "scalar_values"): _A3,
    ("spectra.photon.charged_pion", "near_rest", "values"): _A3,
    ("spectra.photon.charged_pion", "boosted_mild", "scalar_values"): _A3,
    ("spectra.photon.charged_pion", "boosted_mild", "values"): _A3,
    ("spectra.photon.charged_pion", "boosted_strong", "scalar_values"): _A3,
    ("spectra.photon.charged_pion", "boosted_strong", "values"): _A3,
    ("spectra.photon.charged_rho", "rest", "scalar_values"): _A3_B3,
    ("spectra.photon.charged_rho", "rest", "values"): _A3_B3,
    ("spectra.photon.charged_rho", "rest_plus_eps", "scalar_values"): _A3_NESTED,
    ("spectra.photon.charged_rho", "rest_plus_eps", "values"): _A3_NESTED,
    ("spectra.photon.charged_rho", "near_rest", "scalar_values"): _A3_NESTED,
    ("spectra.photon.charged_rho", "near_rest", "values"): _A3_NESTED,
    ("spectra.photon.charged_rho", "boosted_mild", "scalar_values"): _A3_NESTED,
    ("spectra.photon.charged_rho", "boosted_mild", "values"): _A3_NESTED,
    ("spectra.photon.charged_rho", "boosted_strong", "scalar_values"): _A3_NESTED,
    ("spectra.photon.charged_rho", "boosted_strong", "values"): _A3_NESTED,
    ("spectra.photon.neutral_rho", "rest", "scalar_values"): _A3_B3,
    ("spectra.photon.neutral_rho", "rest", "values"): _A3_B3,
    ("spectra.photon.neutral_rho", "rest_plus_eps", "scalar_values"): _A3_NESTED,
    ("spectra.photon.neutral_rho", "rest_plus_eps", "values"): _A3_NESTED,
    ("spectra.photon.neutral_rho", "near_rest", "scalar_values"): _A3_NESTED,
    ("spectra.photon.neutral_rho", "near_rest", "values"): _A3_NESTED,
    ("spectra.photon.neutral_rho", "boosted_mild", "scalar_values"): _A3_NESTED,
    ("spectra.photon.neutral_rho", "boosted_mild", "values"): _A3_NESTED,
    ("spectra.photon.neutral_rho", "boosted_strong", "scalar_values"): _A3_NESTED,
    ("spectra.photon.neutral_rho", "boosted_strong", "values"): _A3_NESTED,
    ("spectra.photon.muon", "rest", "values"): _A2,
    # A1.
    ("spectra.photon.eta", "rest_plus_eps", "values"): _A1,
    ("spectra.photon.eta", "rest_plus_eps", "scalar_values"): _A1,
    ("spectra.photon.eta", "near_rest", "values"): _A1,
    ("spectra.photon.eta", "near_rest", "scalar_values"): _A1,
    ("spectra.photon.eta", "boosted_mild", "values"): _A1,
    ("spectra.photon.eta", "boosted_mild", "scalar_values"): _A1,
    ("spectra.photon.eta", "boosted_strong", "values"): _A1,
    ("spectra.photon.eta", "boosted_strong", "scalar_values"): _A1,
    ("spectra.photon.eta_prime", "rest_plus_eps", "values"): _A1_B1,
    ("spectra.photon.eta_prime", "rest_plus_eps", "scalar_values"): _A1,
    ("spectra.photon.eta_prime", "near_rest", "values"): _A1_B1,
    ("spectra.photon.eta_prime", "near_rest", "scalar_values"): _A1,
    ("spectra.photon.eta_prime", "boosted_mild", "values"): _A1_B1,
    ("spectra.photon.eta_prime", "boosted_mild", "scalar_values"): _A1_B1,
    ("spectra.photon.eta_prime", "boosted_strong", "values"): _A1_B1,
    ("spectra.photon.eta_prime", "boosted_strong", "scalar_values"): _A1_B1,
    ("spectra.photon.omega", "rest_plus_eps", "values"): _A1,
    ("spectra.photon.omega", "rest_plus_eps", "scalar_values"): _A1,
    ("spectra.photon.omega", "near_rest", "values"): _A1,
    ("spectra.photon.omega", "near_rest", "scalar_values"): _A1,
    ("spectra.photon.omega", "boosted_mild", "values"): _A1,
    ("spectra.photon.omega", "boosted_mild", "scalar_values"): _A1,
    ("spectra.photon.omega", "boosted_strong", "values"): _A1,
    ("spectra.photon.omega", "boosted_strong", "scalar_values"): _A1,
    ("spectra.photon.phi", "rest_plus_eps", "values"): _A1,
    ("spectra.photon.phi", "rest_plus_eps", "scalar_values"): _A1,
    ("spectra.photon.phi", "near_rest", "values"): _A1_B2,
    ("spectra.photon.phi", "near_rest", "scalar_values"): _A1_B2,
    ("spectra.photon.phi", "boosted_mild", "values"): _A1_B2,
    ("spectra.photon.phi", "boosted_mild", "scalar_values"): _A1_B2,
    ("spectra.photon.phi", "boosted_strong", "values"): _A1_B2,
    ("spectra.photon.phi", "boosted_strong", "scalar_values"): _A1_B2,
    ("spectra.photon.charged_kaon", "rest_plus_eps", "values"): _A1,
    ("spectra.photon.charged_kaon", "rest_plus_eps", "scalar_values"): _A1,
    ("spectra.photon.charged_kaon", "near_rest", "values"): _A1,
    ("spectra.photon.charged_kaon", "near_rest", "scalar_values"): _A1,
    ("spectra.photon.charged_kaon", "boosted_mild", "values"): _A1,
    ("spectra.photon.charged_kaon", "boosted_mild", "scalar_values"): _A1,
    ("spectra.photon.charged_kaon", "boosted_strong", "values"): _A1,
    ("spectra.photon.charged_kaon", "boosted_strong", "scalar_values"): _A1,
    ("spectra.photon.long_kaon", "rest_plus_eps", "values"): _A1,
    ("spectra.photon.long_kaon", "rest_plus_eps", "scalar_values"): _A1,
    ("spectra.photon.long_kaon", "near_rest", "values"): _A1,
    ("spectra.photon.long_kaon", "near_rest", "scalar_values"): _A1,
    ("spectra.photon.long_kaon", "boosted_mild", "values"): _A1,
    ("spectra.photon.long_kaon", "boosted_mild", "scalar_values"): _A1,
    ("spectra.photon.long_kaon", "boosted_strong", "values"): _A1,
    ("spectra.photon.long_kaon", "boosted_strong", "scalar_values"): _A1,
    ("spectra.photon.short_kaon", "rest_plus_eps", "values"): _A1,
    ("spectra.photon.short_kaon", "rest_plus_eps", "scalar_values"): _A1,
    ("spectra.photon.short_kaon", "near_rest", "values"): _A1,
    ("spectra.photon.short_kaon", "near_rest", "scalar_values"): _A1,
    ("spectra.photon.short_kaon", "boosted_mild", "values"): _A1,
    ("spectra.photon.short_kaon", "boosted_mild", "scalar_values"): _A1,
    ("spectra.photon.short_kaon", "boosted_strong", "values"): _A1,
    ("spectra.photon.short_kaon", "boosted_strong", "scalar_values"): _A1,
    # B4. The ``mu_mu_only`` blocks of this case are deliberately absent:
    # they open no FSR channel and must still match the stored arrays bit
    # for bit. Within a declared array the same holds position by position:
    # wherever the FSR term is zero -- above a channel's endpoint, below the
    # soft cut, outside the boost window -- `MOVED` leaves the position at
    # the case's own budget.
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
    ): _A3_B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_550.rest.default",
        "scalar_values",
    ): _A3_B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_550.rest_plus_eps.default",
        "values",
    ): _A3_B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_550.rest_plus_eps.default",
        "scalar_values",
    ): _A3_B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_550.near_rest.default",
        "values",
    ): _A3_B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_550.near_rest.default",
        "scalar_values",
    ): _A3_B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_550.boosted_mild.default",
        "values",
    ): _A3_B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_550.boosted_mild.default",
        "scalar_values",
    ): _A3_B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_550.boosted_strong.default",
        "values",
    ): _A3_B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_550.boosted_strong.default",
        "scalar_values",
    ): _A3_B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_900.rest.default",
        "values",
    ): _A3_B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_900.rest.default",
        "scalar_values",
    ): _A3_B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_900.rest_plus_eps.default",
        "values",
    ): _A3_B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_900.rest_plus_eps.default",
        "scalar_values",
    ): _A3_B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_900.near_rest.default",
        "values",
    ): _A3_B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_900.near_rest.default",
        "scalar_values",
    ): _A3_B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_900.boosted_mild.default",
        "values",
    ): _A3_B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_900.boosted_mild.default",
        "scalar_values",
    ): _A3_B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_900.boosted_strong.default",
        "values",
    ): _A3_B4,
    (
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
        "ms_900.boosted_strong.default",
        "scalar_values",
    ): _A3_B4,
    # B5.
    ("spectra.neutrino.charged_pion", "near_rest", "values"): _B5,
    ("spectra.neutrino.charged_pion", "near_rest", "scalar_values"): _B5,
    ("spectra.neutrino.charged_pion", "boosted_mild", "values"): _B5,
    ("spectra.neutrino.charged_pion", "boosted_mild", "scalar_values"): _B5,
    ("spectra.neutrino.charged_pion", "boosted_strong", "values"): _B5,
    ("spectra.neutrino.charged_pion", "boosted_strong", "scalar_values"): _B5,
    # B6.
    (
        "cross_sections.scalar.thermal_cross_section",
        "open_resonance",
        "values",
    ): _B6,
    (
        "cross_sections.scalar.thermal_cross_section",
        "narrow_resonance",
        "values",
    ): _B6,
    (
        "cross_sections.scalar.thermal_cross_section",
        "closed_resonance",
        "values",
    ): _B6,
    (
        "cross_sections.vector.thermal_cross_section",
        "open_resonance",
        "values",
    ): _B6,
    (
        "cross_sections.vector.thermal_cross_section",
        "narrow_resonance",
        "values",
    ): _B6,
    (
        "cross_sections.vector.thermal_cross_section",
        "closed_resonance",
        "values",
    ): _B6,
    # A4.
    **_A4_DECLARATIONS,
    # C1, after A4: it replaces the A4 keys it shares with their composite.
    **_C1_DECLARATIONS,
}


#: Every modelled delta, by roster label and optional /variant — including
#: the ones whose repair
#: has not landed and which therefore hold no key in `DECLARED_DELTAS`
#: yet. A repair task moves its model into that table by adding the arrays
#: it measured moving; see the module docstring.
DELTA_MODELS: dict[str, Delta] = {
    "A1": _A1,
    "A2": _A2,
    "A3": _A3,
    "A3/nested": _A3_NESTED,
    "A3+B4": _A3_B4,
    "A3+B3": _A3_B3,
    "A4": _A4,
    "A4/pion": _A4_PION,
    "A4/nested": _A4_NESTED,
    "A4+C1": _A4_C1,
    "A1+B1": _A1_B1,
    "A1+B2": _A1_B2,
    "B1": _B1,
    "B2": _B2,
    "B3": _B3,
    "B4": _B4,
    "B5": _B5,
    "B6": _B6,
    "C1": _C1,
}


def declared(case_name: str, block_label: str, array_suffix: str) -> Delta | None:
    """The declaration covering one stored array, or ``None``."""
    return DECLARED_DELTAS.get((case_name, block_label, array_suffix))
