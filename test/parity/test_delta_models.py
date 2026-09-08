"""The closed-form delta models, checked against the arrays that pin the defect.

Three of the defects under
``projects/parity-pinned-defect-repair`` have no Cython twin left to
capture a corrected value from: B1 (the eta-prime two-photon line's
weight), B2 (both phi line energies) and B3 (the rho rest-frame branch).
Their twins died in cython-to-rust Tasks 4.2 and 4.5, so the corrected
value cannot be measured the way `test_oracles.py` measures Group A's.

What they have instead is a closed form, and this module is the argument
that the form is right. The argument runs backwards, which is what makes
it non-circular: rather than predicting the repaired value and checking
it against the repaired kernel, each test takes the **corrected** form,
applies the **named defect** to it, and requires the result to be the
number the corpus already stores. Nothing here evaluates a spectrum
kernel; every comparison is against ``data/*.npz``, captured from Cython
at kernel digest ``f5e6e269be47`` and never rewritten
(``../rules.md`` rule 1). So these tests were falsifiable on the tree
that carried all three defects, and they stay falsifiable as the repairs
land one at a time: B1's has, and B2's and B3's have not.

The three arguments
-------------------
**B1** — a two-photon final state contributes two photons per decay, so
the line's weight is ``2 BR``. Boosted, a line of weight ``w`` is a flat
plateau of height ``w / (2 gamma beta e0)`` across the window it opens,
and the plateau's upper edge is the spectrum's endpoint, because no
photon from the decay carries more than ``M / 2`` in the rest frame. The
step across that edge therefore measures ``w`` directly. Measured on the
stored arrays, the eta reads ``2 BR`` and so do both neutral kaons — and
the eta-prime reads ``1 BR``, which is the defect.

**B2** — in ``X -> Y gamma`` the photon carries ``(M**2 - m**2) / (2 M)``
and the meson ``(M**2 + m**2) / (2 M)``. The phi kernel boosts the
second as if it were the first, which puts both lines *above* the
spectrum's true endpoint, in a region where the boosted continuum is
identically zero. The stored arrays have a two-tread staircase there
whose treads are exactly the two plateau heights at the *shipped*
energies — support the repaired kernel will not have, at energies the
corrected form does not produce.

**B3** — the boost carries a ``1 / E'`` that belongs to its kernel, not
to the spectrum, and the rho's rest-frame short circuit returns the
integrand with that factor still attached. So the corrected value is the
stored one times ``E_gamma``. The corpus checks it against itself: the
``rest_plus_eps`` block of the same case runs the *other* branch at
``beta = 1.4e-06``, whose ``beta -> 0`` limit is the rest-frame spectrum.

Deliberately not arbitrary precision
------------------------------------
``../rules.md`` rule 3 asks for an `mpmath` reference in the shape of
`reference.py` where a Group B closed form is analytic. `reference.py`
exists because its four kernels lose about 33 decimal digits to a
catastrophic ``atan`` cancellation; measured against 60-digit `mpmath`,
these three forms lose under half a digit — the two-body energies are
good to 15.6 to 16.2 digits and a boosted line's ``height * width`` is
``1.0`` to within one ulp. There is nothing for extra precision to
resolve, and the corpus, not the closed form, is what these tests trust.
The measurement is in
``projects/parity-pinned-defect-repair/task-notes/task-3-closed-form-deltas.md``;
`mpmath` stays out of the test path, as `pyproject.toml` and
`stability.py` both say it is.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import deltas  # (imported after the sys.path entry above)
import generate as corpus_generate

MANIFEST = corpus_generate.load_manifest()

#: How close a plateau recovered from the stored arrays sits to the
#: modelled one. The residue is the continuum's own change across the
#: straddling grid step, which the corpus bounds two ways: the four
#: ``X -> gamma gamma`` lines sit at ``M / 2``, an anchor, so their step
#: is 1e-09 relative; the phi's two sit above the support, where the
#: continuum is exactly zero on both sides. Worst over the 22 line/block
#: pairs below is 4.9e-13, so this is 20x headroom.
PLATEAU_RTOL = 1e-11

#: How far a repaired rho rest value may sit from the ``beta -> 0`` limit
#: the same case's ``rest_plus_eps`` block holds. The limit is approached
#: as ``O(beta**2) = 2e-12``, and the two blocks' grids are paired to
#: 1e-09 rather than shared, so a smooth spectrum contributes its log-log
#: slope times that. Worst measured is 1.8e-09, on the neutral rho.
REST_LIMIT_RTOL = 1e-8

#: How closely two grids must agree to be the same sampling point. The
#: two rho blocks differ only in the parent energy's last 1e-12, which
#: moves every base-grid point by at most that and leaves the anchor
#: points — including the ones offset by ``+/-1e-09`` — untouched.
PAIRING_RTOL = 1e-9


def _stored(case_name: str) -> dict[str, dict[str, np.ndarray]]:
    """Every block of one case, as ``label -> {suffix: array}``."""
    case = MANIFEST["cases"][case_name]
    payload = np.load(corpus_generate.DATA_DIR / case["file"])
    return {
        block["label"]: {
            suffix: payload[entry["key"]] for suffix, entry in block["arrays"].items()
        }
        for block in case["blocks"]
    }


def _params(case_name: str, label: str) -> dict[str, float]:
    """One block's fixed arguments, from the manifest."""
    blocks = {b["label"]: b for b in MANIFEST["cases"][case_name]["blocks"]}
    return blocks[label]["params"]


@dataclass(frozen=True)
class _Block:
    """The part of `cases.Block` a delta model reads: params and grids.

    Rebuilt from the manifest and the stored arrays rather than from
    `cases.build_cases`, so nothing here needs the live specification —
    these tests read the corpus and the models, and no kernel.
    """

    params: dict[str, float]
    grid: np.ndarray
    scalar_probe: np.ndarray


def _block(case_name: str, label: str, stored: dict[str, np.ndarray]) -> _Block:
    """One corpus block, in the shape a `deltas` term function expects."""
    return _Block(
        params=_params(case_name, label),
        grid=stored["grid"],
        scalar_probe=stored.get("scalar_grid", np.empty(0, dtype=np.float64)),
    )


def _beta(case_name: str, label: str) -> float:
    """The boost velocity the tabulated kernel runs one block at."""
    params = _params(case_name, label)
    return deltas._parent_beta(_Block(params, np.empty(0), np.empty(0)))


def _plateau_step(
    grid: np.ndarray, values: np.ndarray, e0: float, weight: float, beta: float
) -> tuple[float, float] | None:
    """Recover a boosted line's plateau height from a stored spectrum.

    Returns ``(measured, modelled)`` in MeV⁻¹, read across the **upper**
    edge of the line's window: the last position inside it minus the
    first outside. Above that edge the decay puts no photons, so the
    difference is the plateau and nothing else. ``None`` when the window
    has no edge to read — no grid point inside it, which is the two phi
    lines at ``beta = 1.4e-06`` where it is 2.8e-06 wide in relative
    energy and the grid steps straight over, or nothing outside it,
    which no corpus block reaches because every grid runs to a hundred
    times the parent energy.
    """
    term = deltas._boosted_line(e0, weight, grid, beta)
    inside = np.flatnonzero(term != 0.0)
    if inside.size == 0 or inside[-1] + 1 >= grid.size:
        return None
    last = inside[-1]
    return float(abs(values[last] - values[last + 1])), float(term[last])


#: ``(case, line label, rest-frame energy, weight)`` for every line the
#: tabulated photon family carries whose plateau is readable off the
#: stored arrays — the four ``X -> gamma gamma`` lines, which sit at the
#: spectrum's endpoint, and the phi's two, which the defect puts above
#: it. The omega's two are the same shape but sit *inside* the boosted
#: continuum, where a one-step difference measures the continuum's slope
#: rather than the plateau; they are the control for the energy formula
#: (`test_a_two_body_decay_splits_the_parent_mass`), not for a weight.
#:
#: The three defect rows take their energy and weight from `deltas`, so a
#: line that moves in the model moves here. The three control rows carry
#: literals from the same `rust/src/constants.rs` ``pdg`` table, because
#: no model needs them — being independent of `deltas` is the point of a
#: control.
TABULATED_LINES = [
    ("spectra.photon.eta", "eta -> gamma gamma", 547.862 / 2.0, 2.0 * 39.41e-2),
    ("spectra.photon.long_kaon", "K_L -> gamma gamma", 497.611 / 2.0, 2.0 * 5.47e-4),
    ("spectra.photon.short_kaon", "K_S -> gamma gamma", 497.611 / 2.0, 2.0 * 2.63e-6),
    (
        "spectra.photon.eta_prime",
        "eta' -> gamma gamma",
        deltas.MASS_ETAP / 2.0,
        deltas.BR_ETAP_TO_A_A,
    ),
    (
        "spectra.photon.phi",
        "phi -> eta gamma",
        deltas._daughter_energy(deltas.MASS_PHI, deltas.MASS_ETA),
        deltas.BR_PHI_TO_ETA_A,
    ),
    (
        "spectra.photon.phi",
        "phi -> eta' gamma",
        deltas._daughter_energy(deltas.MASS_PHI, deltas.MASS_ETAP),
        deltas.BR_PHI_TO_ETAP_A,
    ),
]

#: The four parent energies with a line to read. The ``rest`` block takes
#: ``photon_tables::branch``'s rest-frame arm, which adds no line at all.
BOOSTED_LABELS = ("rest_plus_eps", "near_rest", "boosted_mild", "boosted_strong")

#: How many (line, block) pairs `TABULATED_LINES` actually yields a
#: reading for. Held as a literal so that a block quietly falling out of
#: the measurement — a grid change, a window that stops being sampled —
#: shows up as a failure rather than as a smaller silent sweep.
EXPECTED_PLATEAU_READINGS = 22


@pytest.mark.parametrize(
    ("case_name", "line", "e0", "weight"),
    TABULATED_LINES,
    ids=[line for _, line, _, _ in TABULATED_LINES],
)
def test_a_tabulated_line_carries_the_weight_the_corpus_stores(
    case_name: str, line: str, e0: float, weight: float
) -> None:
    """The stored plateau is the weight the shipped kernel gives the line.

    This is the measurement B1 and B2 both rest on, run over three lines
    that are already right (the eta and both neutral kaons) as well as
    the three that are not, so a method that could only ever answer "yes"
    would be caught by the controls.
    """
    blocks = _stored(case_name)
    readings = 0
    for label in BOOSTED_LABELS:
        step = _plateau_step(
            blocks[label]["grid"],
            blocks[label]["values"],
            e0,
            weight,
            _beta(case_name, label),
        )
        if step is None:
            continue
        readings += 1
        measured, modelled = step
        assert measured == pytest.approx(modelled, rel=PLATEAU_RTOL), (
            f"{case_name}[{label}] {line}: the stored spectrum steps by "
            f"{measured:.9e} MeV^-1 across the top of the line's window, "
            f"where a weight of {weight!r} predicts {modelled:.9e}"
        )
    assert readings, f"{case_name} {line}: no block sampled the line's window"


def test_the_line_measurement_covers_every_pair_it_is_expected_to() -> None:
    """The sweep above reads 22 (line, block) pairs, not fewer.

    `_plateau_step` returns ``None`` for a window no grid point falls
    inside, which is legitimate for the two phi lines at
    ``beta = 1.4e-06`` and would silently shrink the sweep anywhere else.
    """
    readings = 0
    for case_name, _, e0, weight in TABULATED_LINES:
        blocks = _stored(case_name)
        for label in BOOSTED_LABELS:
            step = _plateau_step(
                blocks[label]["grid"],
                blocks[label]["values"],
                e0,
                weight,
                _beta(case_name, label),
            )
            readings += step is not None
    assert readings == EXPECTED_PLATEAU_READINGS


def test_the_eta_prime_line_is_stored_at_one_branching_ratio_not_two() -> None:
    """B1, and its falsification, against the model that will declare it.

    The corrected weight is ``2 BR``, the same as the eta's and both
    neutral kaons'. So the delta is one more copy of the line the stored
    spectrum already carries — which means the B1 model's term must equal
    the plateau the corpus holds, plateau for plateau, and not twice or
    half of it. Reading it off the model rather than recomputing it here
    is what makes this a test of `deltas`.
    """
    case_name = "spectra.photon.eta_prime"
    model = deltas.DELTA_MODELS["B1"]
    blocks = _stored(case_name)
    for label in BOOSTED_LABELS:
        stored = blocks[label]
        term = model.relation.term(None, _block(case_name, label, stored))["values"]
        inside = np.flatnonzero(term != 0.0)
        assert inside.size, f"eta_prime[{label}]: the B1 term reaches no position"
        last = inside[-1]
        measured = abs(stored["values"][last] - stored["values"][last + 1])
        assert measured == pytest.approx(term[last], rel=PLATEAU_RTOL), (
            f"eta_prime[{label}]: the stored spectrum steps by {measured:.9e} "
            f"MeV^-1 across the top of the line's window, where the B1 model "
            f"adds {term[last]:.9e} — the delta is one more copy of the "
            "stored line, so the two are the same number"
        )
        # Falsification: a factor of two either way is 50% or 100% out,
        # eleven decades outside the budget the equality is held to.
        assert measured != pytest.approx(2.0 * term[last], rel=1e-6)
        assert measured != pytest.approx(0.5 * term[last], rel=1e-6)


def test_the_phi_lines_are_stored_at_the_daughter_mesons_energy() -> None:
    """B2, and its falsification.

    Both lines' shipped windows lie above the boosted continuum's
    support, so out there the stored spectrum is a two-tread staircase
    and nothing else: ``h_eta + h_etap``, then ``h_etap``, then exactly
    zero. Put the lines where the corrected form puts them and that
    structure is not there — the corrected windows sit down inside the
    live continuum, and the stored spectrum is smoothly varying across
    both of their edges.
    """
    blocks = _stored("spectra.photon.phi")
    eta_shipped = deltas._daughter_energy(deltas.MASS_PHI, deltas.MASS_ETA)
    etap_shipped = deltas._daughter_energy(deltas.MASS_PHI, deltas.MASS_ETAP)
    eta_fixed = deltas._photon_energy(deltas.MASS_PHI, deltas.MASS_ETA)
    etap_fixed = deltas._photon_energy(deltas.MASS_PHI, deltas.MASS_ETAP)

    for label in ("near_rest", "boosted_mild", "boosted_strong"):
        beta = _beta("spectra.photon.phi", label)
        grid, values = blocks[label]["grid"], blocks[label]["values"]
        eta_term = deltas._boosted_line(eta_shipped, deltas.BR_PHI_TO_ETA_A, grid, beta)
        etap_term = deltas._boosted_line(
            etap_shipped, deltas.BR_PHI_TO_ETAP_A, grid, beta
        )
        # The eta line's window is the narrower of the two and sits
        # inside the eta-prime's, so the three treads are: both lines,
        # the eta-prime alone, and nothing.
        both = (eta_term != 0.0) & (etap_term != 0.0)
        outer = (etap_term != 0.0) & (eta_term == 0.0) & (grid > grid[both][-1])
        above = grid > grid[etap_term != 0.0][-1]
        assert both.any() and outer.any() and above.any()
        np.testing.assert_allclose(
            values[both][-3:],
            (eta_term + etap_term)[both][-3:],
            rtol=PLATEAU_RTOL,
            err_msg=f"phi[{label}]: the top of the stored spectrum is not "
            "the two shipped lines' plateaus alone",
        )
        np.testing.assert_allclose(
            values[outer],
            etap_term[outer],
            rtol=PLATEAU_RTOL,
            err_msg=f"phi[{label}]: between the two shipped windows the "
            "stored spectrum is not the eta-prime line alone",
        )
        assert not values[above].any(), (
            f"phi[{label}]: the stored spectrum is non-zero above the "
            "eta-prime line's window, so the staircase is not the whole "
            "of the support out there"
        )
        # The B2 model's own statement of the same thing: the repair
        # subtracts both plateaus from where they are and adds them a
        # long way below, so wherever the stored array is nothing but
        # staircase the array the model predicts is identically zero.
        # Read off the model rather than recomputed, so a term that
        # relocated only one of the two lines fails here.
        pure = outer | above
        pure[np.flatnonzero(both)[-3:]] = True
        repaired = deltas.DELTA_MODELS["B2"].relation.expected(
            None, _block("spectra.photon.phi", label, blocks[label]), blocks[label]
        )["values"]
        assert not repaired[pure].any(), (
            f"phi[{label}]: the B2 model leaves "
            f"{int((repaired[pure] != 0.0).sum())} non-zero value(s) where "
            "the stored spectrum is nothing but the two shipped plateaus — "
            "above the phi's true photon endpoint a repaired kernel has no "
            "support at all"
        )
        # Falsification: at the corrected energies there is no staircase,
        # because those windows lie inside the live continuum.
        for fixed, weight in (
            (eta_fixed, deltas.BR_PHI_TO_ETA_A),
            (etap_fixed, deltas.BR_PHI_TO_ETAP_A),
        ):
            step = _plateau_step(grid, values, fixed, weight, beta)
            assert step is not None
            measured, modelled = step
            assert measured != pytest.approx(modelled, rel=1e-3), (
                f"phi[{label}]: the stored spectrum steps by the corrected "
                f"line's plateau at {fixed:.6f} MeV, which would mean the "
                "corpus was captured from a repaired kernel"
            )


def _pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs where ``left`` and ``right`` sample the same energy."""
    nearest = np.searchsorted(right, left)
    rows: list[tuple[int, int]] = []
    for i, energy in enumerate(left):
        for j in (nearest[i] - 1, nearest[i], nearest[i] + 1):
            if 0 <= j < right.size and abs(right[j] / energy - 1.0) <= PAIRING_RTOL:
                rows.append((i, j))
                break
    return np.array([i for i, _ in rows]), np.array([j for _, j in rows])


@pytest.mark.parametrize(
    "case_name", ["spectra.photon.charged_rho", "spectra.photon.neutral_rho"]
)
def test_the_repaired_rho_rest_block_is_the_boosted_branchs_limit(
    case_name: str,
) -> None:
    """B3, and its falsification.

    ``rest`` and ``rest_plus_eps`` are one part in 1e12 apart in parent
    energy and on opposite sides of the ``E_rho - m_rho < DBL_EPSILON``
    guard, so the corpus holds both branches of the same physical point.
    The boosted one is right, and its ``beta -> 0`` limit is the
    rest-frame spectrum; the short circuit returns that divided by
    ``E_gamma``. Multiplying it back reproduces the boosted block. Not
    multiplying it back does not, by a factor that runs from 7.8e-03 to
    7.8e+04.
    """
    blocks = _stored(case_name)
    rest, boosted = blocks["rest"], blocks["rest_plus_eps"]
    i, j = _pair(rest["grid"], boosted["grid"])
    assert i.size == rest["grid"].size, (
        f"{case_name}: only {i.size} of {rest['grid'].size} rest points pair "
        "with a rest_plus_eps point"
    )

    # The model's own prediction, not a re-spelling of it: mutating the
    # transform to any other power of E_gamma has to fail here.
    predicted = deltas.DELTA_MODELS["B3"].relation.expected(
        None, _block(case_name, "rest", rest), rest
    )
    repaired = predicted["values"][i]
    limit = boosted["values"][j]
    assert ((repaired == 0.0) == (limit == 0.0)).all(), (
        f"{case_name}: the two branches disagree about where the spectrum " "vanishes"
    )
    live = limit != 0.0
    np.testing.assert_allclose(
        repaired[live],
        limit[live],
        rtol=REST_LIMIT_RTOL,
        err_msg=f"{case_name}: stored rest values times E_gamma are not the "
        "rest_plus_eps block, so the rest-frame branch's defect is not a "
        "missing factor of E_gamma",
    )
    # The same statement as a ratio, which is what the declaration says:
    # the boosted limit is E_gamma times the stored value, at every live
    # position and nowhere near any other power.
    np.testing.assert_allclose(
        limit[live] / rest["values"][i][live],
        rest["grid"][i][live],
        rtol=REST_LIMIT_RTOL,
        err_msg=f"{case_name}: the ratio of the two branches is not E_gamma",
    )
    # And the model reaches only the rest block: every other parent energy
    # is past the guard, where the repair changes nothing.
    for label, stored in _stored(case_name).items():
        off_branch = deltas.DELTA_MODELS["B3"].relation.expected(
            None, _block(case_name, label, stored), stored
        )
        moved = any((off_branch[s] != stored[s]).any() for s in off_branch)
        assert moved == (label == "rest"), (
            f"{case_name}[{label}]: the B3 model "
            f"{'moves' if moved else 'leaves'} this block, and the "
            "E_rho - m_rho < DBL_EPSILON guard admits only rest"
        )
    # Falsification: nothing in the corpus already agrees. The unrepaired
    # value would coincide with the limit only at E_gamma = 1 MeV, and the
    # nearest grid point to that is 0.9959 MeV -- 0.4% away, six decades
    # outside REST_LIMIT_RTOL.
    coincidences = int(
        np.isclose(
            rest["values"][i][live], limit[live], rtol=REST_LIMIT_RTOL, atol=0.0
        ).sum()
    )
    assert coincidences == 0, (
        f"{case_name}: {coincidences} stored rest values already equal the "
        "boosted limit, so the corpus does not pin the defect there"
    )


def test_a_boosted_line_delivers_the_photons_its_weight_declares() -> None:
    """The physics invariant behind both line models (`../rules.md` rule 4).

    A line of weight ``w`` is ``w`` photons per decay in every frame: the
    boost widens the window by exactly the factor it lowers the plateau,
    so ``height * width == w``. A corpus comparison cannot say this — it
    is what makes ``2 BR`` the right weight for a two-photon mode and
    ``BR`` the right one for ``X -> Y gamma``.
    """
    for e0 in (deltas.MASS_ETAP / 2.0, deltas.MASS_ETA / 2.0, 248.8055):
        for beta in (1e-06, 0.3049106779729929, 0.8660254037844386, 0.99):
            gamma = 1.0 / np.sqrt(1.0 - beta * beta)
            lower, upper = gamma * e0 * (1.0 - beta), gamma * e0 * (1.0 + beta)
            # `2 gamma beta e0` rather than `upper - lower`, which is the
            # same width and loses six digits to cancellation at
            # beta = 1e-06. The support check below is what ties the
            # uncancelled spelling back to the edges the kernel uses.
            width = 2.0 * gamma * beta * e0
            for weight in (2.307e-2, 2.0 * 39.41e-2):
                probe = np.array(
                    [
                        lower * (1.0 - 1e-12),
                        0.5 * (lower + upper),
                        upper * (1.0 + 1e-12),
                    ]
                )
                term = deltas._boosted_line(e0, weight, probe, beta)
                assert term[0] == 0.0 and term[2] == 0.0, (
                    f"e0={e0} beta={beta}: the plateau does not stop at "
                    "gamma e0 (1 -/+ beta)"
                )
                assert term[1] * width == pytest.approx(weight, rel=1e-15)


def test_a_parent_at_rest_carries_no_boosted_line() -> None:
    """At ``E == M`` the model adds nothing, and only at ``E == M``.

    `deltas._parent_beta` mirrors ``photon_tables::branch``, including its
    ``E - M < DBL_EPSILON`` arm, which returns the bare table with no
    line. That arm is unreachable at these masses and the kernel says so:
    ``DBL_EPSILON`` is an absolute 2.2e-16 MeV while one ulp at 957.78 MeV
    is 1.1e-13, so the half-open interval it covers holds no double but
    ``M`` itself — where ``boost_beta`` is zero anyway. Asserted here
    rather than exercised, because there is no input that exercises it,
    and the next parent energy up is the corpus's own ``rest_plus_eps``.
    """
    eps = np.finfo(np.float64).eps
    for mass in (deltas.MASS_ETAP, deltas.MASS_PHI):
        assert np.nextafter(mass, np.inf) - mass > eps, (
            f"{mass} MeV: an ulp is smaller than DBL_EPSILON here, so the "
            "guard's interval is reachable and needs a case of its own"
        )
        block = _Block(
            params={"parent_energy": mass, "parent_mass": mass},
            grid=np.array([mass / 2.0]),
            scalar_probe=np.empty(0, dtype=np.float64),
        )
        assert deltas._parent_beta(block) == 0.0
        assert not deltas.DELTA_MODELS["B1"].relation.term(None, block)["values"].any()
        # One ulp up is the smallest parent energy that is not at rest,
        # and it does open a window -- so the model's "no line at rest" is
        # a statement about one point, not a blunt threshold.
        moving = _Block(
            params={"parent_energy": np.nextafter(mass, np.inf), "parent_mass": mass},
            grid=np.array([mass / 2.0]),
            scalar_probe=np.empty(0, dtype=np.float64),
        )
        assert deltas._parent_beta(moving) > 0.0


def test_a_two_body_decay_splits_the_parent_mass() -> None:
    """The corrected form B2 restores, stated as the kinematics it comes from.

    In ``X -> Y gamma`` the photon and the meson share the parent's rest
    mass and carry equal and opposite momentum. Both of the kernel's
    expressions are here, and what separates them is ``m**2 / M`` — small
    for the omega's pi-zero, and 900 MeV for the phi's eta-prime, which
    is why B2 is a factor of 16 there and 1.8 at the other line.
    """
    for parent, daughter in (
        (deltas.MASS_PHI, deltas.MASS_ETA),
        (deltas.MASS_PHI, deltas.MASS_ETAP),
        (782.66, 134.9768),
        (782.66, deltas.MASS_ETA),
    ):
        photon = deltas._photon_energy(parent, daughter)
        meson = deltas._daughter_energy(parent, daughter)
        assert photon + meson == pytest.approx(parent, rel=1e-15)
        assert photon**2 == pytest.approx(meson**2 - daughter**2, rel=1e-12)
        assert meson - photon == pytest.approx(daughter**2 / parent, rel=1e-12)
        assert photon < meson


#: ``repair -> (arrays, positions)`` each model moves, re-derived by
#: `test_each_model_moves_what_it_says_it_moves` and quoted in the model's
#: own ``measured``. Held as literals so that a model whose reach changes
#: — a grid change, a constant that moves — has to be re-measured rather
#: than silently re-scoped (`../rules.md` rule 11).
EXPECTED_REACH = {"B1": (6, 189), "B2": (6, 305), "B3": (4, 350)}

#: Which corpus cases each unlanded model reaches, from the roster in
#: ``references/defect-blast-radius.md``. A repair task turns these into
#: `deltas.DECLARED_DELTAS` keys.
MODEL_CASES = {
    "B1": ("spectra.photon.eta_prime",),
    "B2": ("spectra.photon.phi",),
    "B3": ("spectra.photon.charged_rho", "spectra.photon.neutral_rho"),
}


@pytest.mark.parametrize("repair", sorted(EXPECTED_REACH))
def test_each_model_moves_what_it_says_it_moves(repair: str) -> None:
    """A model's reach is measured, and it is the blast radius's cases.

    Counting it here is what lets Tasks 5, 6 and 9 declare rather than
    re-derive, and what makes a model that quietly grows a block fail
    before it is ever keyed.
    """
    model = deltas.DELTA_MODELS[repair]
    arrays = positions = 0
    for case_name in MODEL_CASES[repair]:
        for label, stored in _stored(case_name).items():
            predicted = model.relation.expected(
                None, _block(case_name, label, stored), stored
            )
            for suffix, values in predicted.items():
                moved = int((values != stored[suffix]).sum())
                if moved:
                    arrays += 1
                    positions += moved
    assert (arrays, positions) == EXPECTED_REACH[repair]
