"""``hazma._core.interp`` against ``numpy.interp``.

Twelve ``cdef`` functions in the compiled layer call ``np.interp`` on a
shipped table: the five rest-frame photon spectra
(``hazma/spectra/_photon/{_eta,_eta_prime,_kaon,_omega,_phi}.pyx``) and
the four mediator spectrum modules. cython-to-rust Task 3.4 reimplements
it in ``rust/src/interp.rs``, and this module is the gate on that.

The oracle is ``numpy.interp`` itself, and the bar is **bit-equality**,
not a tolerance. That is achievable because the Rust reproduces NumPy's
arithmetic exactly, fused multiply-add included -- see
:class:`TestFusedArithmetic` for the measurement that made the fused form
mandatory rather than stylistic.

The comparison is scoped to a contracting platform
--------------------------------------------------
Whether ``np.interp`` fuses ``slope * (x - xp[j]) + fp[j]`` is a property
of *the NumPy binary that happens to be installed*, not of this port. On
macOS/arm64 -- the platform the parity corpus was captured on, and whose
numbers this port targets -- the C compiler contracts it and the
comparison is bit-exact. On a target built without hardware FMA
(baseline x86-64, which is what the Linux wheels are built for) NumPy
computes the unfused values instead, and "does the Rust match the local
NumPy bit-for-bit" stops being a question about the port.

So :data:`NUMPY_CONTRACTS` is *measured at import* and the
cross-implementation tests skip where it is false. That is the same
scoping the parity corpus already has -- CI runs
``pytest --ignore=test/parity`` off macOS for the same reason -- and it
is preferred over loosening the assertion to a tolerance, because the
worst *relative* gap between the two forms lands at a catastrophic
cancellation point (the eta table's tail, where the interpolant is
``2.4e-26`` against a table whose scale is ``0.2``, an absolute gap of
``1.4e-30``). A tolerance wide enough to admit that point would be wide
enough to hide a real defect. Everything platform-independent -- the
clamping contract, NumPy's quirks, the error paths, dispatch -- runs
everywhere.

Lifetime
--------
Nothing here parses Cython, so this module outlives the ``.pyx``. After
Phase 06 it remains the standing check that the Rust interpolation still
tracks NumPy -- a property the parity corpus cannot see, because the
corpus pins spectra rather than the routines underneath them.
"""

from __future__ import annotations

import numpy as np
import pytest

from hazma._core import interp as core_interp
from hazma.spectra._photon import _eta, _eta_prime, _kaon, _omega, _phi

interp = core_interp.interp

#: The smallest grid the multi-point path handles; below it NumPy takes
#: its one-point branch, whose behavior differs (see :class:`TestQuirks`).
MULTI_POINT = 2


def photon_tables() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The live tables ``np.interp`` is called on, keyed by kernel."""
    return {
        "eta": (_eta.eta_data_energies, _eta.eta_data_dnde),
        "eta_prime": (
            _eta_prime.eta_prime_data_energies,
            _eta_prime.eta_prime_data_dnde,
        ),
        "charged_kaon": (
            _kaon.charged_kaon_data_energies,
            _kaon.charged_kaon_data_dnde,
        ),
        "long_kaon": (_kaon.long_kaon_data_energies, _kaon.long_kaon_data_dnde),
        "short_kaon": (_kaon.short_kaon_data_energies, _kaon.short_kaon_data_dnde),
        "omega": (_omega.omega_data_energies, _omega.omega_data_dnde),
        "phi": (_phi.phi_data_energies, _phi.phi_data_dnde),
    }


def numpy_contracts() -> bool:
    """Whether the installed NumPy fuses the interpolation step.

    Compares ``np.interp`` against the unfused form on interior points
    only, where the cell index is unambiguous and no clamp or node
    short circuit is in play -- so a disagreement can only be the
    contraction.
    """
    xp, fp = photon_tables()["eta"]
    rng = np.random.default_rng(0)
    x = rng.uniform(xp[0], xp[-1], 4096)
    j = np.clip(np.searchsorted(xp, x, side="right") - 1, 0, xp.size - 2)
    slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])
    return bool(np.any(slope * (x - xp[j]) + fp[j] != np.interp(x, xp, fp)))


#: True where the installed NumPy contracts, i.e. where a bit-for-bit
#: comparison against it is a statement about this port rather than
#: about the platform's instruction selection.
NUMPY_CONTRACTS = numpy_contracts()

requires_a_contracting_numpy = pytest.mark.skipif(
    not NUMPY_CONTRACTS,
    reason=(
        "this NumPy does not fuse the interpolation step, so it computes "
        "different values than the macOS/arm64 build this port targets; "
        "the bit-for-bit comparison is scoped to a contracting platform "
        "exactly as the parity corpus is"
    ),
)


def sweep_abscissae(xp: np.ndarray, seed: int) -> np.ndarray:
    """Abscissae covering every branch of the interpolation.

    Random interior points find generic cells; the nodes themselves hit
    the exact-node short circuit; the nodes nudged by one part in 1e13
    land just inside the cells on either side of a node, which is where a
    wrong cell index shows up; and the four out-of-range points exercise
    the clamps.
    """
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [
            rng.uniform(xp[0], xp[-1], 20_000),
            xp,
            xp * (1.0 + 1e-13),
            xp * (1.0 - 1e-13),
            [xp[0] - 1.0, xp[-1] + 1.0, xp[0], xp[-1]],
        ]
    )


@requires_a_contracting_numpy
class TestAgainstNumpy:
    """Bit-equality with ``np.interp`` on every live table.

    Scoped to a contracting NumPy -- see the module docstring.
    """

    @pytest.mark.parametrize("name", list(photon_tables()))
    def test_matches_numpy_bit_for_bit(self, name: str) -> None:
        xp, fp = photon_tables()[name]
        x = sweep_abscissae(xp, seed=hash(name) % 2**32)
        got = interp(x, xp, fp)
        want = np.interp(x, xp, fp)
        mismatched = int(np.count_nonzero(got != want))
        # `want` is exactly zero past the spectrum endpoint, so the
        # relative gap is reported against a floor rather than dividing
        # by it -- otherwise the diagnostic reads `nan` precisely when
        # it is needed.
        scale = np.where(want == 0.0, 1.0, np.abs(want))
        assert mismatched == 0, (
            f"{name}: {mismatched} of {x.size} points differ from np.interp; "
            f"worst relative {np.max(np.abs(got - want) / scale):.3e}"
        )

    def test_matches_numpy_on_a_random_grid(self) -> None:
        """A grid NumPy has never seen, with cells of wildly unequal width.

        The live tables are smooth and near-geometric; this one is not,
        so a cell-index or slope error that the tables happen to forgive
        has somewhere to show.
        """
        rng = np.random.default_rng(20260810)
        for _ in range(50):
            n = int(rng.integers(2, 60))
            xp = np.sort(rng.uniform(-1e3, 1e3, n) * 10.0 ** rng.uniform(-6, 6, n))
            xp = np.unique(xp)
            if xp.size < MULTI_POINT:
                continue
            fp = rng.standard_normal(xp.size) * 10.0 ** rng.uniform(-8, 8, xp.size)
            x = sweep_abscissae(xp, seed=int(rng.integers(2**31)))
            assert np.array_equal(interp(x, xp, fp), np.interp(x, xp, fp))


@requires_a_contracting_numpy
class TestFusedArithmetic:
    """The interpolation step is fused, and it has to be.

        NumPy computes ``slope * (x - xp[j]) + fp[j]`` in C, where the
        default ``-ffp-contract=on`` lets the compiler emit a fused
        multiply-add -- and on this project's reference platform
        (macOS/arm64) it does. Rust never contracts on its own, so
        ``rust/src/interp.rs`` spells the fusion out with ``mul_add``.

    It computes the unfused value in Python and asserts that where the
        two forms differ, the Rust sides with NumPy. The class only runs
        where :data:`NUMPY_CONTRACTS`, so "the forms differ somewhere" is a
        precondition rather than a hope.
    """

    def test_the_rust_sides_with_numpy_where_the_forms_differ(self) -> None:
        xp, fp = photon_tables()["eta"]
        rng = np.random.default_rng(4)
        x = rng.uniform(xp[0], xp[-1], 20_000)

        j = np.searchsorted(xp, x, side="right") - 1
        j = np.clip(j, 0, xp.size - 2)
        slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])
        unfused = slope * (x - xp[j]) + fp[j]

        want = np.interp(x, xp, fp)
        differ = unfused != want
        assert differ.any(), (
            "no point distinguishes the fused and unfused forms on this "
            "platform, so this test proves nothing here"
        )
        assert np.array_equal(interp(x, xp, fp), want)


#: A three-node toy grid with cells of unequal width, shared by the
#: contract tests below. The names let the assertions say *which* node
#: they expect rather than repeating its value.
TOY_XP = np.array([1.0, 2.0, 4.0])
TOY_FP = np.array([10.0, 20.0, -5.0])
TOY_FIRST, TOY_MIDDLE, TOY_LAST = (float(value) for value in TOY_FP)
#: Midpoints of the two cells: (10 + 20)/2 and (20 - 5)/2.
TOY_FIRST_CELL_MID = 15.0
TOY_SECOND_CELL_MID = 7.5


class TestClamping:
    """Outside the grid ``np.interp`` clamps; it never extrapolates."""

    XP = TOY_XP
    FP = TOY_FP

    @pytest.mark.parametrize("x", [0.0, -1e300, -np.inf, 0.999_999])
    def test_below_the_grid_returns_the_first_value(self, x: float) -> None:
        assert interp(x, self.XP, self.FP) == TOY_FIRST

    @pytest.mark.parametrize("x", [5.0, 1e300, np.inf, 4.000_001])
    def test_above_the_grid_returns_the_last_value(self, x: float) -> None:
        assert interp(x, self.XP, self.FP) == TOY_LAST

    def test_nodes_return_their_own_values(self) -> None:
        assert np.array_equal(interp(self.XP, self.XP, self.FP), self.FP)

    def test_the_midpoint_of_a_cell_is_the_mean_of_its_ends(self) -> None:
        assert interp(1.5, self.XP, self.FP) == TOY_FIRST_CELL_MID
        assert interp(3.0, self.XP, self.FP) == TOY_SECOND_CELL_MID


#: Values the quirk tests assert on by name.
ONE_POINT_VALUE = 7.0
LAST_DUPLICATE = 9.0
INFINITE_NODE_VALUE = 1.0
FLAT_CELL_VALUE = 3.0


class TestQuirks:
    """Behaviors that are NumPy's rather than linear interpolation's.

    Each is reproduced deliberately and checked against NumPy in the same
    assertion, so the pin cannot drift away from the thing it pins.
    """

    def test_nan_propagates_on_a_multi_point_grid(self) -> None:
        xp, fp = np.array([1.0, 2.0]), np.array([ONE_POINT_VALUE, 8.0])
        assert np.isnan(interp(np.nan, xp, fp))
        assert np.isnan(np.interp(np.nan, xp, fp))

    def test_a_one_point_grid_answers_everything_with_its_one_value(self) -> None:
        """NumPy's one-point branch runs before its NaN check.

        So a NaN abscissa returns ``fp[0]`` there while it returns NaN on
        any longer grid -- an asymmetry with no principle behind it,
        carried because the corpus is pinned to what NumPy does.
        """
        xp, fp = np.array([2.0]), np.array([ONE_POINT_VALUE])
        for x in (np.nan, -1.0, 2.0, 5.0):
            assert interp(x, xp, fp) == ONE_POINT_VALUE
            assert np.interp(x, xp, fp) == ONE_POINT_VALUE

    def test_duplicate_nodes_resolve_to_the_last_copy(self) -> None:
        xp = np.array([0.0, 1.0, 1.0, 2.0])
        fp = np.array([0.0, 5.0, LAST_DUPLICATE, LAST_DUPLICATE])
        assert interp(1.0, xp, fp) == np.interp(1.0, xp, fp) == LAST_DUPLICATE

    def test_an_infinite_ordinate_falls_back_to_the_cells_other_end(self) -> None:
        xp, fp = np.array([0.0, 1.0]), np.array([np.inf, 0.0])
        assert interp(0.5, xp, fp) == np.interp(0.5, xp, fp) == np.inf

    def test_an_infinite_node_returns_its_own_value(self) -> None:
        """The exact-node short circuit, in the only place it is visible.

        At an ordinary node the interpolation gives ``slope * 0 + fp[j]``
        = ``fp[j]`` anyway, so the guard NumPy carries to "avoid potential
        non-finite interpolation" is unobservable — until the cell is
        infinitely wide. Here ``slope`` is 0 and ``x - xp[j]`` is
        ``-inf - -inf`` = NaN, so the product is NaN and both NaN rescues
        fail; only the short circuit returns a number.
        """
        xp = np.array([-np.inf, 0.0, 1.0])
        fp = np.array([INFINITE_NODE_VALUE, 2.0, 3.0])
        assert interp(-np.inf, xp, fp) == INFINITE_NODE_VALUE
        assert np.interp(-np.inf, xp, fp) == INFINITE_NODE_VALUE

    def test_an_infinitely_wide_cell_is_rescued_only_when_flat(self) -> None:
        xp = np.array([-np.inf, np.inf])
        flat = np.array([FLAT_CELL_VALUE, FLAT_CELL_VALUE])
        sloped = np.array([FLAT_CELL_VALUE, FLAT_CELL_VALUE + 1.0])
        assert interp(0.0, xp, flat) == FLAT_CELL_VALUE
        assert np.interp(0.0, xp, flat) == FLAT_CELL_VALUE
        assert np.isnan(interp(0.0, xp, sloped))
        assert np.isnan(np.interp(0.0, xp, sloped))


class TestErrors:
    """The two grids NumPy refuses, refused with NumPy's own wording."""

    def test_an_empty_grid_raises(self) -> None:
        empty = np.array([], dtype=np.float64)
        with pytest.raises(ValueError, match="array of sample points is empty"):
            interp(1.0, empty, empty)
        with pytest.raises(ValueError, match="array of sample points is empty"):
            np.interp(1.0, empty, empty)

    def test_mismatched_lengths_raise(self) -> None:
        xp = np.array([1.0, 2.0])
        fp = np.array([1.0])
        with pytest.raises(ValueError, match="not of the same length"):
            interp(1.0, xp, fp)
        with pytest.raises(ValueError, match="not of the same length"):
            np.interp(1.0, xp, fp)


class TestDispatch:
    """The abscissa follows the contract every ported entry point uses.

    The full branch-by-branch pinning lives in
    ``test/test_core_dispatch.py``; this only checks that ``interp`` is
    wired into it rather than re-deriving it.
    """

    XP = TOY_XP
    FP = TOY_FP

    def test_scalar_in_float_out(self) -> None:
        got = interp(1.5, self.XP, self.FP)
        assert isinstance(got, float)
        assert got == TOY_FIRST_CELL_MID

    def test_array_in_fresh_array_out(self) -> None:
        x = np.array([1.5, 3.0])
        got = interp(x, self.XP, self.FP)
        assert isinstance(got, np.ndarray)
        assert got.dtype == np.float64
        assert np.array_equal(got, [TOY_FIRST_CELL_MID, TOY_SECOND_CELL_MID])
        assert not np.shares_memory(got, x)

    def test_array_path_equals_the_scalar_path(self) -> None:
        x = np.linspace(0.5, 4.5, 101)
        assert np.array_equal(
            interp(x, self.XP, self.FP),
            [interp(float(v), self.XP, self.FP) for v in x],
        )

    def test_empty_array_round_trips(self) -> None:
        got = interp(np.array([]), self.XP, self.FP)
        assert isinstance(got, np.ndarray)
        assert got.size == 0

    def test_two_dimensional_abscissae_raise(self) -> None:
        with pytest.raises(ValueError, match="0 or 1-dimensional"):
            interp(np.zeros((2, 2)), self.XP, self.FP)

    def test_non_float64_abscissae_raise(self) -> None:
        with pytest.raises(ValueError, match="float64 array"):
            interp(np.array([1, 2], dtype=np.int64), self.XP, self.FP)
