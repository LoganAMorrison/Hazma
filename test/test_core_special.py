"""``hazma._core.special`` against scipy: the three cimported specfuns.

The Cython layer reaches outside itself for exactly three special
functions, all through ``from scipy.special.cython_special cimport ...``:

===========================  ======================================
``spence(xm) - spence(xp)``  ``hazma/spectra/_photon/_muon.pyx:113``
``k1(x * z)``                both mediators' thermal-average integrand
``kn(2, x)``                 both mediators' thermal-average prefactor
===========================  ======================================

cython-to-rust Task 3.2 reimplements them in ``rust/src/special.rs`` over
the cephes-lineage ``spec_math`` crate, which is also what drops the
``scipy>=1.13`` build-ABI pin. This module is the gate on that: it holds
the Rust side to ``scipy.special`` at ``rtol <= 1e-13`` over each
function's live domain, and pins the two things that are conventions
rather than values -- ``spence``'s argument (``Li2(1 - z)``, not
``Li2(z)``) and ``kn``'s order.

The oracle chain
----------------
Every comparison below is against the ``scipy.special`` *ufunc*, while
the code being replaced calls the ``scipy.special.cython_special``
*C function*. :class:`TestOracleIdentity` is what makes that
substitution legitimate: it calls the C functions through their
``__pyx_capi__`` capsules and asserts they return bit-identical values
to the ufuncs on the same grids. If scipy ever splits the two, that
class fails first and everything below it becomes suspect at once.

Why ``rtol = 1e-13``
--------------------
Task 3.2's exit criteria set it, and it is loose relative to what these
functions actually do: ``spence`` and ``k1`` are the same cephes
routines on both sides, so they agree to a few ulp (measured max
``2.4e-15`` and ``1.2e-15``), and ``kn`` -- which is *not* the same
routine, see :class:`TestBesselKn` -- agrees to ``9.8e-16`` over
hazma's live domain. 1e-13 is therefore a ceiling with two orders of
headroom, not a fitted tolerance: anything approaching it is a defect.

Lifetime
--------
Unlike ``test/test_core_constants.py``, nothing here parses Cython, so
this module outlives the ``.pyx``. It should keep running after Phase 06
as the standing check that the Rust specfuns still track scipy -- which
is the property the parity corpus cannot see, since the corpus pins
*spectra*, not the functions underneath them.
"""

from __future__ import annotations

import ctypes
import functools
import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
import scipy.special as sp
import scipy.special.cython_special as cython_special

from hazma._core import special as rust_special

if TYPE_CHECKING:
    from collections.abc import Callable

#: Task 3.2's tolerance for every scipy comparison. See the module
#: docstring for why this is a ceiling and not a fit.
RTOL = 1e-13

#: Tolerance for the one test that exists to *distinguish* the
#: implementation from cephes' own ``kn``, which misses by 5.1e-9 there.
#: Tighter than :data:`RTOL` so it cannot pass on a near-miss.
CEPHES_DISCRIMINATOR_RTOL = 1e-14

#: A grid segment thinner than this is a slice expression that stopped
#: selecting what its name says.
MIN_GRID_POINTS = 100

#: Above this the mediator models short-circuit
#: (``thermal_cross_section`` returns ``0.0`` for ``x > 300``), so it is
#: the top of the only ``kn`` domain hazma can reach.
X_MAX_THERMAL = 300.0

#: Where ``scipy.special.kn(2, .)`` first flushes to zero, and where this
#: crate's ``bessel_kn`` first does. Measured on scipy 1.18.0 over
#: ``np.linspace(600, 760, 16001)``; both are grid points of that sweep,
#: so they are upper bounds on the true crossings to within its 0.01
#: spacing. See :class:`TestBesselKnUnderflowTail`.
SCIPY_KN_FLUSHES_AT = 697.88
RUST_KN_FLUSHES_AT = 742.09


def max_relative_error(got: np.ndarray, want: np.ndarray) -> tuple[float, float]:
    """Return ``(max relative error, the abscissa where it occurred)``.

    Compared pointwise on the entries where ``want`` is finite and
    non-zero; a zero reference has no relative error to speak of and is
    covered by the exact-value tests instead.
    """
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    usable = np.isfinite(want) & (want != 0.0) & np.isfinite(got)
    assert usable.any(), "grid produced no comparable points"
    error = np.abs(got[usable] - want[usable]) / np.abs(want[usable])
    return float(error.max()), int(np.argmax(error))


def assert_tracks_scipy(got: np.ndarray, want: np.ndarray, grid: np.ndarray) -> float:
    """Assert ``got`` matches ``want`` to :data:`RTOL`; return the max error."""
    error, index = max_relative_error(got, want)
    usable = np.isfinite(want) & (want != 0.0) & np.isfinite(got)
    where = np.asarray(grid, dtype=float)[usable][index]
    assert error <= RTOL, (
        f"max relative error {error:.3e} > {RTOL:.0e} at x = {where!r} "
        f"(rust {np.asarray(got)[usable][index]!r} vs scipy "
        f"{np.asarray(want)[usable][index]!r})"
    )
    return error


def dilogarithm(z: float, terms: int = 2000) -> float:
    """``Li2(z)`` from its defining series, for ``|z| <= 1``.

    ``sum_{k>=1} z**k / k**2`` (DLMF 25.12.1). Deliberately independent
    of both scipy and the crate: it is what lets
    :class:`TestSpenceConvention` say *which* dilogarithm ``spence`` is,
    rather than only that two implementations agree with each other.
    """
    if abs(z) > 1.0:
        msg = "the series converges only on the closed unit disk"
        raise ValueError(msg)
    return math.fsum(z**k / k**2 for k in range(1, terms + 1))


# ---------------------------------------------------------------------
# Grids. Built once at import; each is named for the exit criterion or
# call site it covers, so a later reader can tell which are load-bearing.
# ---------------------------------------------------------------------

#: Task 3.2's ``spence`` grid: "(0,1), [1,inf), z->0+, z=1, z=2".
SPENCE_GRID = np.unique(
    np.concatenate(
        [
            np.linspace(1e-12, 1.0, 5001),  # (0, 1) and the branch point z = 1
            np.geomspace(1.0, 1e12, 5001),  # [1, inf)
            np.geomspace(1e-300, 1e-3, 1001),  # z -> 0+
            [0.0, 1.0, 2.0],
        ]
    )
)

#: ``k1``'s live argument is ``x * z`` with ``z >= 2`` and ``x <= 300``,
#: so it spans many decades; the linear patch straddles cephes' internal
#: ``x = 2`` branch switch.
K1_GRID = np.unique(
    np.concatenate([np.geomspace(1e-8, 690.0, 20001), np.linspace(1.9, 2.1, 401)])
)

#: The large-argument region, up to where both implementations have
#: flushed to zero.
K1_UNDERFLOW_GRID = np.linspace(690.0, 745.0, 2001)

#: ``kn(2, .)`` sees ``x = m_chi / T`` in ``(0, 300]``. The linear patch
#: straddles cephes ``kn``'s ``x = 9.55`` branch switch, which is where
#: that routine is least accurate -- see :class:`TestBesselKn`.
KN_GRID = np.unique(
    np.concatenate(
        [np.geomspace(1e-8, X_MAX_THERMAL, 20001), np.linspace(1.0, 20.0, 2001)]
    )
)


class TestOracleIdentity:
    """``scipy.special.<f>`` is the same function the Cython cimports.

    Everything else in this module compares the crate against the ufunc,
    because the ufunc is what a test can call. The Cython calls the
    ``cython_special`` C symbol. These three tests close that gap by
    calling the C symbols directly through their capsules.

    They are also the reason the rest of the module can be read as a
    parity gate rather than as a plausibility check.
    """

    @staticmethod
    def _c_function(
        name: str,
        signature: str,
        restype: type,
        argtypes: list[type],
    ) -> Callable[..., float]:
        """Resolve a ``cython_special`` capsule by name **and** signature.

        Cython mangles fused-type entry points as
        ``__pyx_fuse_<i><name>``, and the index depends on the order the
        fused types are declared -- not something to hardcode. Matching
        on the capsule's own signature string instead survives a
        reordering and still fails loudly if the symbol goes away.
        """
        get_name = ctypes.pythonapi.PyCapsule_GetName
        get_name.restype = ctypes.c_char_p
        get_name.argtypes = [ctypes.py_object]
        get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
        get_pointer.restype = ctypes.c_void_p
        get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

        matches = [
            capsule
            for key, capsule in cython_special.__pyx_capi__.items()
            if (key == name or key.endswith(name))
            and get_name(capsule).decode() == signature
        ]
        assert len(matches) == 1, (
            f"expected exactly one cython_special capsule named {name!r} with "
            f"signature {signature!r}; found {len(matches)}. scipy's fused-type "
            f"export layout has changed -- re-derive it before trusting the "
            f"scipy.special comparisons in this module."
        )
        address = get_pointer(matches[0], get_name(matches[0]))
        return ctypes.CFUNCTYPE(restype, *argtypes)(address)

    def test_cython_special_spence_is_the_ufunc(self) -> None:
        spence = self._c_function(
            "spence",
            "double (double, int __pyx_skip_dispatch)",
            ctypes.c_double,
            [ctypes.c_double, ctypes.c_int],
        )
        grid = SPENCE_GRID[::7]
        through_c = np.array([spence(float(x), 0) for x in grid])
        np.testing.assert_array_equal(through_c, sp.spence(grid))

    def test_cython_special_k1_is_the_ufunc(self) -> None:
        k1 = self._c_function(
            "k1",
            "double (double, int __pyx_skip_dispatch)",
            ctypes.c_double,
            [ctypes.c_double, ctypes.c_int],
        )
        grid = K1_GRID[::7]
        through_c = np.array([k1(float(x), 0) for x in grid])
        np.testing.assert_array_equal(through_c, sp.k1(grid))

    def test_cython_special_kn_is_the_ufunc(self) -> None:
        # The ``.pyx`` write ``kn(2, x)`` with an integer literal, which
        # Cython resolves to the ``long`` overload.
        kn = self._c_function(
            "kn",
            "double (long, double, int __pyx_skip_dispatch)",
            ctypes.c_double,
            [ctypes.c_long, ctypes.c_double, ctypes.c_int],
        )
        grid = KN_GRID[::7]
        through_c = np.array([kn(2, float(x), 0) for x in grid])
        np.testing.assert_array_equal(through_c, sp.kn(2, grid))


class TestSpenceConvention:
    """``spence`` is ``Li2(1 - z)``. Not ``Li2(z)``.

    This is the trap ``references/numerics-replacements.md`` flags and
    the one that would be cheapest to get wrong: ``spec_math`` exposes
    the routine as ``Polylog::li2``, a name that says the *other*
    convention. ``dnde_photon_muon`` subtracts two of these, so the wrong
    convention does not blow up -- it returns a smooth, plausible,
    incorrect spectrum.
    """

    def test_closed_forms(self) -> None:
        # Li2(1 - 0) = Li2(1) = pi^2/6; Li2(1 - 1) = Li2(0) = 0;
        # Li2(1 - 2) = Li2(-1) = -pi^2/12.
        assert rust_special.spence(0.0) == pytest.approx(math.pi**2 / 6, rel=RTOL)
        assert rust_special.spence(1.0) == 0.0
        assert rust_special.spence(2.0) == pytest.approx(-(math.pi**2) / 12, rel=RTOL)

    def test_argument_is_reflected_about_one(self) -> None:
        # The discriminator. At z = 0.25 the two conventions differ by a
        # factor of ~3.7, so this cannot pass under either reading by
        # accident -- unlike z = 0.5, where Li2(1 - z) and Li2(z) are the
        # same number.
        assert rust_special.spence(0.25) == pytest.approx(dilogarithm(0.75), rel=RTOL)
        assert rust_special.spence(0.75) == pytest.approx(dilogarithm(0.25), rel=RTOL)
        assert rust_special.spence(0.25) != pytest.approx(dilogarithm(0.25), rel=1e-3)

    def test_reflection_identity(self) -> None:
        # Li2(z) + Li2(1 - z) = pi^2/6 - ln(z) ln(1 - z), which in this
        # convention reads spence(x) + spence(1 - x)
        #   = pi^2/6 - ln(1 - x) ln(x).
        grid = np.linspace(0.02, 0.98, 97)
        lhs = rust_special.spence(grid) + rust_special.spence(1.0 - grid)
        rhs = math.pi**2 / 6 - np.log1p(-grid) * np.log(grid)
        assert_tracks_scipy(lhs, rhs, grid)


class TestSpenceAgreement:
    """The exit-criterion sweep, segment by segment."""

    @pytest.mark.parametrize(
        ("label", "grid"),
        [
            ("(0, 1)", SPENCE_GRID[(SPENCE_GRID > 0.0) & (SPENCE_GRID < 1.0)]),
            ("[1, inf)", SPENCE_GRID[SPENCE_GRID >= 1.0]),
            ("z -> 0+", np.geomspace(1e-300, 1e-3, 1001)),
            ("whole grid", SPENCE_GRID),
        ],
    )
    def test_tracks_scipy(self, label: str, grid: np.ndarray) -> None:
        assert (
            grid.size > MIN_GRID_POINTS
        ), f"{label} segment collapsed to {grid.size} points"
        assert_tracks_scipy(rust_special.spence(grid), sp.spence(grid), grid)

    def test_exact_at_the_named_points(self) -> None:
        # z -> 0+, z = 1 and z = 2 are named individually in the exit
        # criteria; the first two are cephes' own special cases, so they
        # should be bit-equal rather than merely close.
        for x in (0.0, 1.0, 2.0):
            assert rust_special.spence(x) == float(sp.spence(x)), f"spence({x})"


class TestSpenceEdges:
    """Domain errors and non-finite inputs, matching cephes and scipy."""

    @pytest.mark.parametrize("x", [-1.0, -1e-300, -np.inf, np.inf, np.nan])
    def test_returns_nan_where_scipy_does(self, x: float) -> None:
        assert np.isnan(rust_special.spence(x))
        assert np.isnan(float(sp.spence(x)))


class TestBesselK1:
    """``k1`` over the thermal-average domain, including underflow."""

    def test_tracks_scipy_over_the_thermal_domain(self) -> None:
        assert_tracks_scipy(rust_special.bessel_k1(K1_GRID), sp.k1(K1_GRID), K1_GRID)

    def test_tracks_scipy_through_the_underflow_tail(self) -> None:
        # Both sides are cephes here, with no explicit cutoff on either:
        # the large-argument branch decays into the subnormals and
        # reaches zero when its exp(-x) does. They therefore agree
        # through the whole tail, zeros included.
        grid = K1_UNDERFLOW_GRID
        assert_tracks_scipy(rust_special.bessel_k1(grid), sp.k1(grid), grid)
        np.testing.assert_array_equal(
            rust_special.bessel_k1(grid) == 0.0, np.asarray(sp.k1(grid)) == 0.0
        )

    def test_is_subnormal_before_it_is_zero(self) -> None:
        # Guards the claim above: if this were a hard cutoff rather than
        # a natural decay, the tail sweep would be comparing two zeros.
        assert 0.0 < rust_special.bessel_k1(730.0) < np.finfo(float).tiny

    @pytest.mark.parametrize(
        ("x", "expected"),
        [(0.0, np.inf), (np.inf, 0.0), (1e4, 0.0)],
    )
    def test_edges_match_scipy(self, x: float, expected: float) -> None:
        assert rust_special.bessel_k1(x) == expected
        assert float(sp.k1(x)) == expected

    @pytest.mark.parametrize("x", [-1.0, np.nan])
    def test_returns_nan_where_scipy_does(self, x: float) -> None:
        assert np.isnan(rust_special.bessel_k1(x))
        assert np.isnan(float(sp.k1(x)))


class TestBesselKn:
    """``kn`` over the live domain, and why it is not cephes' ``kn``.

    ``spec_math`` ships a faithful cephes ``kn`` and the crate does not
    use it, because **scipy's ``kn`` is not cephes' ``kn``** -- scipy
    dispatches integer orders to ``kv`` and keeps only ``k0``/``k1`` on
    cephes. ``rust/src/special.rs`` therefore builds ``K_n`` from the
    upward recurrence seeded on ``k0``/``k1``.
    """

    def test_tracks_scipy_over_the_live_domain(self) -> None:
        got = rust_special.bessel_kn(2, KN_GRID)
        assert_tracks_scipy(got, sp.kn(2, KN_GRID), KN_GRID)

    @pytest.mark.parametrize("order", [0, 1, 2, 3, 4, 5])
    def test_tracks_scipy_at_every_small_order(self, order: int) -> None:
        # Only order 2 is live, but the recurrence is general and a
        # seeding or step-count error shows up first at its neighbours.
        grid = np.geomspace(1e-6, X_MAX_THERMAL, 5001)
        got = rust_special.bessel_kn(order, grid)
        assert_tracks_scipy(got, sp.kn(order, grid), grid)

    def test_negative_order_folds(self) -> None:
        grid = np.geomspace(1e-3, X_MAX_THERMAL, 501)
        np.testing.assert_array_equal(
            rust_special.bessel_kn(-2, grid), rust_special.bessel_kn(2, grid)
        )

    def test_beats_cephes_kn_at_its_worst_argument(self) -> None:
        # The discriminator for the implementation choice. cephes' own
        # kn misses scipy by up to 5.1e-9 relative on the low side of its
        # x = 9.55 branch switch -- four orders past this module's gate,
        # and inside the parity corpus's 1e-8 budget for
        # thermal_cross_section, whose prefactor squares this value.
        # Swap the recurrence in rust/src/special.rs back to
        # `x.bessel_kn(n as isize)` and this fails while every other test
        # in the class still passes.
        for x in (8.0, 9.0, 9.5, 9.54, 12.0):
            got = rust_special.bessel_kn(2, x)
            want = float(sp.kn(2, x))
            assert (
                abs(got - want) / abs(want) <= CEPHES_DISCRIMINATOR_RTOL
            ), f"kn(2, {x})"

    @pytest.mark.parametrize(
        ("x", "expected"),
        [(0.0, np.inf), (np.inf, 0.0)],
    )
    def test_edges_match_scipy(self, x: float, expected: float) -> None:
        assert rust_special.bessel_kn(2, x) == expected
        assert float(sp.kn(2, x)) == expected

    @pytest.mark.parametrize("x", [-1.0, np.nan])
    def test_returns_nan_where_scipy_does(self, x: float) -> None:
        assert np.isnan(rust_special.bessel_kn(2, x))
        assert np.isnan(float(sp.kn(2, x)))


class TestBesselKnUnderflowTail:
    """The one measured divergence from scipy, and its distance from hazma.

    scipy's ``kn`` inherits ``kv``'s underflow handling and flushes to
    zero from about ``x = 698``; the crate's recurrence decays with its
    ``exp(-x)`` seeds and reaches zero only when they do, at about
    ``x = 742``. On the ~44-wide window between them the two disagree
    wholesale -- scipy says ``0``, the crate returns everything from
    ``4e-305`` down to the smallest subnormal.

    Note where scipy gives up: ``K2(697.88)`` is ``3.9e-305``, a
    perfectly ordinary normal double. The flush point is ``kv``'s own
    conservative exponent limit, not the end of the representable range,
    so it discards about three decades of real values.

    Nothing in hazma reaches it: ``thermal_cross_section`` short-circuits
    above ``x = 300``, where ``K2 ~ 3.7e-132``. Pinned here so that a
    later caller which widens that domain meets the divergence in a test
    rather than in a spectrum.
    """

    def test_agrees_up_to_scipys_flush_point(self) -> None:
        grid = np.linspace(600.0, SCIPY_KN_FLUSHES_AT, 2001)[:-1]
        got = rust_special.bessel_kn(2, grid)
        want = sp.kn(2, grid)
        assert np.all(want > 0.0), "grid strayed past scipy's flush point"
        # Still RTOL, but with far less headroom than anywhere else in
        # this module: measured max is 5.7e-14 here against 9.8e-16 over
        # the live domain, because the top of this window is deep in the
        # subnormals where the reference itself has lost mantissa bits.
        # Not a reason to loosen anything -- hazma stops at x = 300.
        error, _ = max_relative_error(got, want)
        assert error <= RTOL, f"max relative error {error:.3e}"

    def test_scipy_flushes_first(self) -> None:
        assert float(sp.kn(2, SCIPY_KN_FLUSHES_AT)) == 0.0
        assert float(sp.kn(2, SCIPY_KN_FLUSHES_AT - 0.01)) > 0.0

    def test_the_crate_is_still_returning_values_there(self) -> None:
        window = np.linspace(SCIPY_KN_FLUSHES_AT, RUST_KN_FLUSHES_AT - 0.01, 501)
        got = rust_special.bessel_kn(2, window)
        assert np.all(got > 0.0)
        assert np.all(np.asarray(sp.kn(2, window)) == 0.0)
        # The window opens on a normal double and closes in the
        # subnormals: scipy is discarding representable values, not
        # rounding an already-lost one to zero.
        assert got[0] > np.finfo(float).tiny
        assert got[-1] < np.finfo(float).tiny

    def test_both_are_zero_past_the_crates_flush_point(self) -> None:
        grid = np.linspace(RUST_KN_FLUSHES_AT, 800.0, 101)
        np.testing.assert_array_equal(
            rust_special.bessel_kn(2, grid), np.zeros_like(grid)
        )
        np.testing.assert_array_equal(np.asarray(sp.kn(2, grid)), np.zeros_like(grid))


#: The three bindings as one-argument callables, paired with the
#: ``quantity`` wording each passes to ``dispatch::map_unary``. ``kn``'s
#: order is bound rather than wrapped in a lambda so the parametrized
#: ids below name real functions.
PROBES: dict[str, tuple[Callable[..., object], str]] = {
    "spence": (rust_special.spence, "Spence arguments"),
    "bessel_k1": (rust_special.bessel_k1, "Bessel arguments"),
    "bessel_kn": (
        functools.partial(rust_special.bessel_kn, 2),
        "Bessel arguments",
    ),
}


class TestDispatch:
    """The probes follow the crate's scalar-or-1D contract.

    ``test/test_core_dispatch.py`` pins that contract in full through
    ``hazma._core.roundtrip``; these four tests only confirm the three
    ``special`` bindings are wired through the same helper, and -- the
    part that matters here -- that the array path and the scalar path
    return **the same bits**, since every sweep above is taken through
    the array path.
    """

    @pytest.mark.parametrize("name", list(PROBES))
    def test_array_path_equals_the_scalar_path(self, name: str) -> None:
        call, _ = PROBES[name]
        grid = np.geomspace(1e-6, 500.0, 997)
        np.testing.assert_array_equal(
            call(grid), np.array([call(float(x)) for x in grid])
        )

    @pytest.mark.parametrize("name", list(PROBES))
    def test_scalar_in_float_out(self, name: str) -> None:
        call, _ = PROBES[name]
        assert type(call(1.5)) is float

    @pytest.mark.parametrize("name", list(PROBES))
    def test_empty_array_round_trips(self, name: str) -> None:
        call, _ = PROBES[name]
        out = call(np.array([], dtype=np.float64))
        assert isinstance(out, np.ndarray)
        assert out.shape == (0,)

    @pytest.mark.parametrize("name", list(PROBES))
    def test_two_dimensional_input_raises(self, name: str) -> None:
        call, quantity = PROBES[name]
        with pytest.raises(ValueError, match=f"^{quantity} must be 0 or 1-dimensional"):
            call(np.zeros((2, 2)))
