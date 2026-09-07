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
Every relation answers the same question — how far is the stored array
from what it should hold? — through `Relation.term_for`, so the runner
never has to know which kind it is holding. Relations differ in how they
reach that term, not in what the runner does with it.

``Additive`` — the repaired value is the stored value plus a term the
declaration knows how to compute. The term is evaluated live, from the
repaired kernel, and is its own adaptive quadrature.

``Reference`` — the stored value is superseded outright by one computed
without going through the kernel under repair. A defect that made an
array *wrong*, rather than shifting it by a knowable amount, has no
additive term to name: `thermal_reference` integrates the same integrand
with a different QUADPACK at a convergent tolerance, and the repaired
kernel is held to that.

The budget each relation carries is measured, not assumed, and says so
in its ``why``.

Positions
---------
A declaration covers exactly the positions its mechanism reaches
(``rules.md`` rule 5): either an explicit tuple, every entry of which the
term must move, or `MOVED`, which resolves at comparison time to the
positions where the term is non-zero. Every other position of a declared
array is compared against the stored value under the case's own budget,
exactly as if nothing had been declared, so a regression where the
repair changed nothing is still caught at that budget.

Staleness
---------
A declaration that no longer describes a change is a hole in the gate
(spec rule 3), so the runner also asserts that a declared array *has*
moved: at the positions where the term exceeds the relation's own budget,
the live value must differ from the stored one. Reverting a repair fails
that assertion; the declaration cannot outlive the repair it describes.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import thermal_reference

from hazma import parameters

if TYPE_CHECKING:
    from cases import Block

#: The closed set of repair labels a declaration may carry: the roster in
#: ``projects/parity-pinned-defect-repair/references/defect-blast-radius.md``.
REPAIRS = frozenset({"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "B5", "B6"})

#: The sentinel for "every position the term is non-zero at", resolved
#: against the term at comparison time. A term that is zero at a position
#: has not moved it, and a declaration that covered it anyway would be
#: wider than its mechanism.
MOVED = "moved"

Positions = tuple[int, ...] | Literal["moved"]

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

    def term_for(
        self,
        fn: Callable[..., Any],
        block: Block,
        suffix: str,
        pinned: np.ndarray,
    ) -> np.ndarray:
        """The declared term itself; ``pinned`` is not consulted."""
        del pinned
        return self.term(fn, block)[suffix]


@dataclass(frozen=True)
class Reference:
    """``repaired == reference(block)``, within ``rtol``.

    The stored array is superseded rather than corrected: the repaired
    kernel is compared against a value reached without it. Use this where
    the defect made the stored value wrong by an amount only a second
    implementation can say, rather than by a term the physics names.

    Parameters
    ----------
    reference : TermFn
        Computes the superseding values for one block.
    rtol : float
        Relative budget the relation holds to. Measured, and ``why`` says
        how.
    why : str
        One-line justification of ``rtol``.
    """

    reference: TermFn
    rtol: float
    why: str

    def term_for(
        self,
        fn: Callable[..., Any],
        block: Block,
        suffix: str,
        pinned: np.ndarray,
    ) -> np.ndarray:
        """How far the stored array is from the reference."""
        return self.reference(fn, block)[suffix] - pinned


@dataclass(frozen=True)
class Delta:
    """One declared change to one stored array.

    Parameters
    ----------
    repair : str
        Which roster entry moved it; drawn from `REPAIRS`.
    positions : tuple of int or MOVED
        Which positions the relation covers: an explicit tuple, every
        entry of which the term must move, or `MOVED` for wherever the
        term is non-zero. Undeclared positions are still compared against
        the stored value under the case's budget.
    relation : Additive or Reference
        How the repaired value relates to the stored one.
    measured : str
        The measurement behind the declaration, so the table is not a
        list of assertions nobody re-derived.
    evidence : str
        Repo-relative path of the note that holds the measurement.
    """

    repair: str
    positions: Positions
    relation: Additive | Reference
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
    evidence="docs/followups/todo/neutrino-pion-electron-line-counted-twice.md",
)


# B6 -- the thermal averages never converged
# ---------------------------------------------------------------------------

#: The three model points both thermal cases sweep, in corpus block order.
THERMAL_BLOCKS = ("open_resonance", "narrow_resonance", "closed_resonance")


def _thermal(model: str) -> Reference:
    """The `Reference` relation for one mediator family's thermal case."""
    return Reference(
        reference=thermal_reference.reference_values(model),
        rtol=1e-7,
        why="the reference is scipy's QUADPACK at epsrel 1e-12 over the same "
        "integrand, so what bounds agreement is the repaired kernels' own "
        "epsrel of 1.49e-8, not the reference: a platform whose libm steers "
        "QUADPACK to a different accepted partition may land anywhere inside "
        "it. Measured 3.6e-9 worst relative over the 540 positions the "
        "reference integrates. 1e-7 is 6.7x the bound that has to hold "
        "everywhere, rather than 28x the figure this platform happens to "
        "give.",
    )


#: Both cases carry the same measurement, so the two declarations differ
#: only in which model the reference integrates.
_B6_MEASURED = (
    "the stored arrays are the initial-partition estimate: against the "
    "reference they are wrong by up to 1.00 relative (scalar and vector "
    "closed_resonance, where the shipped value retains none of the true "
    "one), with per-block medians from 7.2e-6 to 8.1e-2. 539 of the 570 "
    "pinned positions move. Of the 31 that do not, 30 are the ten points "
    "per scalar block above x = 300, where that kernel returns 0.0 outright "
    "and the quadrature is never reached; the last is vector "
    "narrow_resonance at x = 0.1367, small enough that the relative "
    "criterion already bound before the repair."
)

_B6_SCALAR = Delta(
    repair="B6",
    positions=MOVED,
    relation=_thermal("scalar"),
    measured=_B6_MEASURED,
    evidence="docs/followups/done/thermal-cross-section-quadrature-never-converges.md",
)

_B6_VECTOR = Delta(
    repair="B6",
    positions=MOVED,
    relation=_thermal("vector"),
    measured=_B6_MEASURED,
    evidence="docs/followups/done/thermal-cross-section-quadrature-never-converges.md",
)

#: Every declared array, keyed as `stability.PORTABILITY_ZEROS` is.
#: What each repair leaves *out* is half its proof that it moved only
#: what it intended:
#:
#: * B4 -- the 15 ``mu_mu_only`` blocks of the scalar decay case open no
#:   FSR channel and must still match the stored arrays bit for bit.
#:   Within a declared array the same holds position by position:
#:   wherever the FSR term is zero -- above a channel's endpoint, below
#:   the soft cut, outside the boost window -- `MOVED` leaves the
#:   position at the case's own budget.
#: * B5 -- ``rest`` and ``rest_plus_eps`` are absent: at rest the kernel
#:   drops both prompt lines, and one epsilon above it no grid point's
#:   boost window is wide enough to straddle the line.
#: * B6 -- the 30 scalar positions above ``x = 300`` are absent, because
#:   that kernel returns ``0.0`` before it integrates and the repair
#:   cannot reach them.
DECLARED_DELTAS: dict[tuple[str, str, str], Delta] = {
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
    ): _B6_SCALAR,
    (
        "cross_sections.scalar.thermal_cross_section",
        "narrow_resonance",
        "values",
    ): _B6_SCALAR,
    (
        "cross_sections.scalar.thermal_cross_section",
        "closed_resonance",
        "values",
    ): _B6_SCALAR,
    (
        "cross_sections.vector.thermal_cross_section",
        "open_resonance",
        "values",
    ): _B6_VECTOR,
    (
        "cross_sections.vector.thermal_cross_section",
        "narrow_resonance",
        "values",
    ): _B6_VECTOR,
    (
        "cross_sections.vector.thermal_cross_section",
        "closed_resonance",
        "values",
    ): _B6_VECTOR,
}


def declared(case_name: str, block_label: str, array_suffix: str) -> Delta | None:
    """The declaration covering one stored array, or ``None``."""
    return DECLARED_DELTAS.get((case_name, block_label, array_suffix))
