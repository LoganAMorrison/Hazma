r"""``hazma._core.boost`` against the Cython it replaces.

``hazma/_utils/boost.pyx`` is the only compiled module the spectra share:
:func:`boost_beta` and :func:`boost_gamma` turn a parent's energy and mass
into boost parameters, ``boost_delta_function`` boosts a two-body line,
and ``boost_integrate_linear_interp`` boosts a tabulated continuum.
cython-to-rust Task 3.4 reimplements all four in ``rust/src/boost.rs``.

The oracle is the Cython itself
-------------------------------
The phase file asked for "micro-fixtures captured in Phase 01". There are
none: the parity corpus enumerates top-level ``def``\ s in the surviving
``.pyx``, and every routine here is ``cdef`` -- private to the C level and
invisible to the corpus by construction. What exists instead is stronger.
``boost.pxd`` declares the ``cdef``\ s, which makes Cython export them
through ``hazma._utils.boost.__pyx_capi__`` as ``PyCapsule``\ s, and
:func:`cython_boost` calls them through ``ctypes``. So the comparisons
below are against the *live* kernel at whatever arguments the test picks,
rather than against a frozen sample of it.

Two mechanical points about that shim. It must use ``ctypes.PYFUNCTYPE``
rather than ``CFUNCTYPE``: the latter releases the GIL, and
``boost_integrate_linear_interp`` touches Python objects (it calls
``np.trapezoid``), so a ``CFUNCTYPE`` call segfaults. And the capsule
name *is* the C signature, so :class:`TestOracle` checks it rather than
trusting that the argument list has not changed.

The bar is bit-equality
-----------------------
Not a tolerance. ``rust/src/boost.rs`` reproduces the shipped Cython's
arithmetic exactly, fused multiply-adds included --
:class:`TestFusedArithmetic` is the measurement that made that mandatory:
written the obvious unfused way, the boost integral misses the corpus by
up to 3.6e-12 relative on the corpus's own grids, against the 1e-12
``TABULATED`` budget in ``test/parity/tolerances.py``.

Lifetime
--------
Everything comparing against ``__pyx_capi__`` dies with the ``.pyx`` in
Phase 06 Task 6.4. :class:`TestTrapezoidSummation`,
:class:`TestDroppedInteriorCell`, :class:`TestErrors` and
:class:`TestDispatch` do not, and are what remains as the standing check
afterwards.
"""

from __future__ import annotations

import ctypes
import math
from collections.abc import Callable
from fractions import Fraction

import numpy as np
import pytest

from hazma._core import boost as core_boost
from hazma._utils import boost as cython_module
from hazma.parameters import (
    charged_kaon_mass,
    eta_mass,
    eta_prime_mass,
    neutral_kaon_mass,
    omega_mass,
    phi_mass,
)
from hazma.spectra._photon import _eta, _eta_prime, _kaon, _omega, _phi

boost_beta = core_boost.boost_beta
boost_gamma = core_boost.boost_gamma
boost_delta_function = core_boost.boost_delta_function
boost_integrate_linear_interp = core_boost.boost_integrate_linear_interp


# --------------------------------------------------------------------------
# The Cython oracle
# --------------------------------------------------------------------------

_SIGNATURES = {
    "boost_integrate_linear_interp": (
        b"double (double, double, PyArrayObject *, PyArrayObject *)"
    ),
    "boost_delta_function": b"double (double, double, double, double)",
    "boost_eng": b"double (double, double, double, double, double)",
    "boost_jac": b"double (double, double, double, double, double)",
}

_ARGTYPES: dict[str, tuple[type, ...]] = {
    "boost_integrate_linear_interp": (
        ctypes.c_double,
        ctypes.c_double,
        ctypes.py_object,
        ctypes.py_object,
    ),
    "boost_delta_function": (ctypes.c_double,) * 4,
    "boost_eng": (ctypes.c_double,) * 5,
    "boost_jac": (ctypes.c_double,) * 5,
}


def cython_boost(name: str) -> Callable[..., float]:
    """The live Cython ``cdef`` of that name, callable from Python.

    Parameters
    ----------
    name : str
        A key of ``hazma._utils.boost.__pyx_capi__``.

    Returns
    -------
    callable
        Returns a ``float``. ``PYFUNCTYPE``, not ``CFUNCTYPE`` -- see the
        module docstring.
    """
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.restype = ctypes.c_void_p
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

    capsule = cython_module.__pyx_capi__[name]
    address = get_pointer(capsule, _SIGNATURES[name])
    prototype = ctypes.PYFUNCTYPE(ctypes.c_double, *_ARGTYPES[name])
    return prototype(address)


def fma(a: float, b: float, c: float) -> float:
    """``a * b + c`` with a single rounding, without ``math.fma``.

    ``math.fma`` arrived in Python 3.13 and this suite supports 3.10, so
    the fused product is computed exactly as a rational and rounded once
    by ``float()``, which rounds to nearest. Used to reproduce the
    compiler's contraction when a test needs to predict a bound.
    """
    return float(Fraction(a) * Fraction(b) + Fraction(c))


def rust_gamma(beta: float) -> float:
    """``1 / sqrt(1 - beta**2)`` as ``rust/src/boost.rs`` computes it."""
    return 1.0 / math.sqrt(fma(-beta, beta, 1.0))


def photon_tables() -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    """The seven live ``(energies, dnde, parent_mass)`` triples."""
    return {
        "eta": (_eta.eta_data_energies, _eta.eta_data_dnde, eta_mass),
        "eta_prime": (
            _eta_prime.eta_prime_data_energies,
            _eta_prime.eta_prime_data_dnde,
            eta_prime_mass,
        ),
        "charged_kaon": (
            _kaon.charged_kaon_data_energies,
            _kaon.charged_kaon_data_dnde,
            charged_kaon_mass,
        ),
        "long_kaon": (
            _kaon.long_kaon_data_energies,
            _kaon.long_kaon_data_dnde,
            neutral_kaon_mass,
        ),
        "short_kaon": (
            _kaon.short_kaon_data_energies,
            _kaon.short_kaon_data_dnde,
            neutral_kaon_mass,
        ),
        "omega": (_omega.omega_data_energies, _omega.omega_data_dnde, omega_mass),
        "phi": (_phi.phi_data_energies, _phi.phi_data_dnde, phi_mass),
    }


#: Parent-energy multiples spanning the regimes the parity corpus uses --
#: just off rest, near rest, mildly boosted, strongly boosted.
BOOST_REGIMES = (1.000_000_001, 1.05, 1.5, 2.0, 3.0, 10.0)

#: A flat toy table with ``y / x == 1`` everywhere, so every branch's
#: contribution is a length and can be predicted by hand.
FLAT_X = np.arange(1.0, 9.0)
FLAT_Y = FLAT_X.copy()

#: The Cython's absolute tolerance for "this bound sits on a node"
#: (``hazma/_utils/boost.pyx:212``), reused where a test has to predict
#: which branch fires.
EDGE_ATOL = 1e-6
#: Enough bracketed window edges for the sweep to mean something.
MIN_EDGES_CHECKED = 100
#: The smallest miss the unfused arithmetic is recorded as producing; a
#: smaller one means the platform stopped contracting.
MIN_RECORDED_UNFUSED_MISS = 1e-13


class TestOracle:
    """The shim really is the Cython, and calls it correctly."""

    def test_every_cdef_is_exported(self) -> None:
        assert set(cython_module.__pyx_capi__) == set(_SIGNATURES)

    @pytest.mark.parametrize("name", sorted(_SIGNATURES))
    def test_the_capsule_signature_is_the_one_the_shim_assumes(self, name: str) -> None:
        """A changed argument list would otherwise corrupt the stack."""
        get_name = ctypes.pythonapi.PyCapsule_GetName
        get_name.restype = ctypes.c_char_p
        get_name.argtypes = [ctypes.py_object]
        assert get_name(cython_module.__pyx_capi__[name]) == _SIGNATURES[name]

    def test_the_shim_returns_the_documented_closed_form(self) -> None:
        """``boost_eng(ep, mp, 1, 0, 0)`` is ``gamma`` and nothing else.

        With ``md = 0`` and ``ed = 1`` the transverse factor is 1 and the
        angular factor is ``1 + 0``, so the Cython computes ``g * 1 * 1``.
        That makes it an exact readout of the inlined ``boost_gamma``,
        which is what :class:`TestBoostParameters` uses it for.
        """
        eng = cython_boost("boost_eng")
        assert eng(1000.0, 139.57039, 1.0, 0.0, 0.0) == 1000.0 / 139.57039


class TestBoostParameters:
    """``boost_beta`` and ``boost_gamma``."""

    CASES = (
        (1000.0, 139.57039),
        (547.9, 547.862),
        (5e4, 0.510_998_946_1),
        (1200.0, 493.677),
        (139.570_390_000_001, 139.57039),
    )

    @pytest.mark.parametrize(("energy", "mass"), CASES)
    def test_gamma_matches_the_cython_bit_for_bit(
        self, energy: float, mass: float
    ) -> None:
        eng = cython_boost("boost_eng")
        assert boost_gamma(energy, mass) == eng(energy, mass, 1.0, 0.0, 0.0)

    @pytest.mark.parametrize(("energy", "mass"), CASES)
    def test_beta_matches_the_unfused_closed_form_bit_for_bit(
        self, energy: float, mass: float
    ) -> None:
        """The definition, evaluated in Python's (never-fused) arithmetic.

        This is the pin on the *absence* of a fused multiply-add in
        ``boost_beta``: the Cython spells it ``(mass / energy) ** 2``,
        whose rounded product completes before the subtraction, and none
        of its ten inlining call sites contract it (checked by
        disassembly, Task 3.4 task note). Writing it fused would move
        every boosted spectrum, and this assertion is what fails if
        someone does.
        """
        ratio = mass / energy
        assert boost_beta(energy, mass) == math.sqrt(1.0 - ratio * ratio)

    @pytest.mark.parametrize(("energy", "mass"), CASES)
    def test_beta_agrees_with_what_the_cython_can_be_asked_for(
        self, energy: float, mass: float
    ) -> None:
        """Cross-check through ``boost_eng``, at the precision it allows.

        ``boost_eng(ep, mp, 1, 0, 1) = gamma * (1 + beta)``, so dividing
        out ``gamma`` and subtracting 1 recovers ``beta`` -- but the
        subtraction cancels, leaving an absolute error of order ``eps``
        rather than a relative one. The tolerance says exactly that: a few
        ulp of 1, loosened by nothing else.
        """
        eng = cython_boost("boost_eng")
        gamma = eng(energy, mass, 1.0, 0.0, 0.0)
        recovered = eng(energy, mass, 1.0, 0.0, 1.0) / gamma - 1.0
        assert boost_beta(energy, mass) == pytest.approx(
            recovered, abs=4.0 * np.finfo(np.float64).eps, rel=0.0
        )

    def test_a_particle_at_rest_has_zero_velocity_and_unit_gamma(self) -> None:
        assert boost_beta(139.57039, 139.57039) == 0.0
        assert boost_gamma(139.57039, 139.57039) == 1.0

    def test_below_rest_energy_beta_is_nan(self) -> None:
        assert np.isnan(boost_beta(100.0, 139.57039))


class TestBoostDeltaFunction:
    """The boosted two-body line, branch by branch."""

    def test_matches_the_cython_over_a_random_sweep(self) -> None:
        """40,000 draws at both live product masses, bit for bit.

        ``m = 0`` is the photon and neutrino case; ``m = MASS_E`` is
        ``hazma/spectra/_positron/_pion.pyx``. The product energy is drawn
        within a factor of 2.5 of the line, which straddles both window
        edges at every boost.
        """
        cython = cython_boost("boost_delta_function")
        rng = np.random.default_rng(20_260_810)
        masses = [0.0, 0.510_998_928]
        mismatched = 0
        total = 0
        for _ in range(40_000):
            m = float(rng.choice(masses))
            beta = float(rng.uniform(1e-6, 1.0 - 1e-9))
            e0 = float(10.0 ** rng.uniform(-1.0, 3.0))
            e = float(e0 * 10.0 ** rng.uniform(-0.4, 0.4))
            total += 1
            if boost_delta_function(e0, e, m, beta) != cython(e0, e, m, beta):
                mismatched += 1
        assert mismatched == 0, f"{mismatched} of {total} draws differ from the Cython"

    def test_the_window_edges_agree_with_the_cython(self) -> None:
        """Both edges, sampled just inside and just outside.

        A one-ulp difference in the bound flips the answer between a
        finite value and zero rather than moving it slightly.
        """
        cython = cython_boost("boost_delta_function")
        e0, m, beta = 200.0, 0.0, 0.6
        gamma = rust_gamma(beta)
        for edge in (gamma * e0 * (1.0 - beta), gamma * e0 * (1.0 + beta)):
            for scale in (1.0 - 1e-15, 1.0, 1.0 + 1e-15, 0.99, 1.01):
                e = edge * scale
                assert boost_delta_function(e0, e, m, beta) == cython(e0, e, m, beta)

    @staticmethod
    def _window_edges(
        cython: Callable[..., float], e0: float, m: float, beta: float
    ) -> list[tuple[float, float]]:
        """The adjacent doubles the Cython's support starts and ends between.

        Returns one `(below, above)` pair per edge that could be
        bracketed — at most two, and possibly none if the analytic
        brackets do not straddle a transition for these parameters.
        """

        def bisect(lo: float, hi: float) -> tuple[float, float]:
            lo_bits = int(np.float64(lo).view(np.int64))
            hi_bits = int(np.float64(hi).view(np.int64))
            lo_zero = cython(e0, lo, m, beta) == 0.0
            while hi_bits - lo_bits > 1:
                mid_bits = (lo_bits + hi_bits) // 2
                mid = float(np.int64(mid_bits).view(np.float64))
                if (cython(e0, mid, m, beta) == 0.0) == lo_zero:
                    lo_bits = mid_bits
                else:
                    hi_bits = mid_bits
            return (
                float(np.int64(lo_bits).view(np.float64)),
                float(np.int64(hi_bits).view(np.float64)),
            )

        gamma = rust_gamma(beta)
        span = 4.0 * gamma * (1.0 + beta)
        brackets = [(max(m, e0 / span), e0), (e0, e0 * span)]
        return [
            bisect(lo, hi)
            for lo, hi in brackets
            if (cython(e0, lo, m, beta) == 0.0) != (cython(e0, hi, m, beta) == 0.0)
        ]

    def test_the_window_edges_sit_on_the_same_double_as_the_cython(self) -> None:
        """Both edges of the support, located to the last bit, 400 times.

        ``eminus`` and ``eplus`` never appear in the returned value —
        they only decide whether it is ``1/(2 gamma beta k0)`` or zero.
        So a one-ulp change in how they are computed is invisible to any
        test that samples ``e`` on a grid: it moves the edge by a single
        double, and no random draw lands there. Bisecting on the bit
        pattern does land there.

        The sweep is what makes this a pin rather than an anecdote, and
        the massive product is the half that matters: with ``m = 0`` the
        fused and unfused ``k = sqrt(e^2 - m^2)`` agree identically, so
        only the ``m = MASS_E`` draws can catch a change in it. An
        earlier version of this test used three fixed parameter sets and
        missed an unfused ``k`` entirely; the random sweep catches it
        (Task 3.4's mutation campaign, round 3).
        """
        cython = cython_boost("boost_delta_function")
        rng = np.random.default_rng(20_260_810)
        checked = 0
        for _ in range(400):
            m = float(rng.choice([0.0, 0.510_998_928]))
            e0 = float(10.0 ** rng.uniform(0.0, 3.0))
            beta = float(rng.uniform(0.02, 0.999))
            if cython(e0, e0, m, beta) == 0.0:
                continue
            for below, above in self._window_edges(cython, e0, m, beta):
                # A flip, so the assertions below compare a zero against a
                # non-zero rather than two zeros.
                assert (cython(e0, below, m, beta) == 0.0) != (
                    cython(e0, above, m, beta) == 0.0
                )
                for e in (below, above):
                    assert boost_delta_function(e0, e, m, beta) == cython(
                        e0, e, m, beta
                    ), f"edge differs at e0={e0!r}, e={e!r}, m={m!r}, beta={beta!r}"
                checked += 1
        assert (
            checked > MIN_EDGES_CHECKED
        ), f"only {checked} edges were bracketed; the sweep is thin"

    def test_the_line_integrates_to_one_across_its_window(self) -> None:
        """The boost spreads a normalised delta; it does not renormalise it.

        Tolerance is the arithmetic's, not the method's: the height and
        the width are each a handful of roundings, so a part in 1e-13 is
        already three orders of headroom.
        """
        e0, m, beta = 200.0, 0.0, 0.6
        gamma = rust_gamma(beta)
        lo, hi = gamma * e0 * (1.0 - beta), gamma * e0 * (1.0 + beta)
        height = boost_delta_function(e0, 0.5 * (lo + hi), m, beta)
        assert height * (hi - lo) == pytest.approx(1.0, rel=1e-13)

    @pytest.mark.parametrize(
        ("e0", "e", "m", "beta"),
        [
            (200.0, 200.0, 0.0, 1.5),  # beta > 1
            (200.0, 200.0, 0.0, 1.0),  # beta == 1
            (200.0, 200.0, 0.0, 0.0),  # beta == 0, the singular guard
            (200.0, 200.0, 0.0, -0.3),  # beta < 0
            (200.0, 0.1, 0.510_998_946_1, 0.6),  # product below its own mass
            (200.0, 1e9, 0.0, 0.6),  # far above the window
            (200.0, 1e-9, 0.0, 0.6),  # far below the window
        ],
    )
    def test_unphysical_and_outside_arguments_return_zero(
        self, e0: float, e: float, m: float, beta: float
    ) -> None:
        cython = cython_boost("boost_delta_function")
        assert boost_delta_function(e0, e, m, beta) == 0.0
        assert cython(e0, e, m, beta) == 0.0


class TestBoostIntegrateLinearInterp:
    """The tabulated continuum boost, branch by branch.

    Every case asserts bit-equality with the Cython *and* pins the branch
    that fired -- either by a closed form the branch alone can produce, or
    by a sensitivity check on a table entry only that branch reads.
    """

    def test_the_whole_window_above_the_table_is_zero(self) -> None:
        cython = cython_boost("boost_integrate_linear_interp")
        energy, beta = 1e6, 0.5
        assert boost_integrate_linear_interp(energy, beta, FLAT_X, FLAT_Y) == 0.0
        assert cython(energy, beta, FLAT_X, FLAT_Y) == 0.0

    def test_the_whole_window_below_the_table_is_the_analytic_tail(self) -> None:
        """``y0 * x0 / E``, a closed form no other branch can produce."""
        cython = cython_boost("boost_integrate_linear_interp")
        energy, beta = 1e-6, 0.5
        got = boost_integrate_linear_interp(energy, beta, FLAT_X, FLAT_Y)
        assert got == FLAT_Y[0] * FLAT_X[0] / energy
        assert got == cython(energy, beta, FLAT_X, FLAT_Y)

    def test_a_window_straddling_the_table_floor_adds_the_tail(self) -> None:
        """`lb` below the table, `ub` inside it.

        Pinned by sensitivity: only the tail term reads ``y[0]`` when
        ``ilow`` is 0, and only through the ``y0 * (1 - rat) / rat``
        factor, so scaling ``y[0]`` must scale the tail's contribution.
        """
        cython = cython_boost("boost_integrate_linear_interp")
        energy, beta = 1.4, 0.6
        base = boost_integrate_linear_interp(energy, beta, FLAT_X, FLAT_Y)
        assert base == cython(energy, beta, FLAT_X, FLAT_Y)

        bumped = FLAT_Y.copy()
        bumped[0] *= 2.0
        moved = boost_integrate_linear_interp(energy, beta, FLAT_X, bumped)
        assert moved != base
        assert moved == cython(energy, beta, FLAT_X, bumped)

    def test_a_window_above_the_table_ceiling_clamps(self) -> None:
        """`ub` past the table's top, but `lb` inside it.

        The clamp is what keeps this from returning zero, and the value
        matches the Cython. That the clamp *also* skips the upper
        partial-cell term — and with it the table's last row — is pinned
        separately in :class:`TestDroppedInteriorCell`.
        """
        cython = cython_boost("boost_integrate_linear_interp")
        energy, beta = 5.0, 0.6
        got = boost_integrate_linear_interp(energy, beta, FLAT_X, FLAT_Y)
        assert got != 0.0
        assert got == cython(energy, beta, FLAT_X, FLAT_Y)

    @pytest.mark.parametrize("index", [0, 7])
    def test_both_partial_cells_are_integrated(self, index: int) -> None:
        """`lb` and `ub` both strictly inside cells.

        ``y[0]`` is read only by the lower partial cell here (``ilow`` is
        1, so the tail does not fire) and ``y[7]`` only by the upper one
        (``ihigh`` is 6). Perturbing either must move the answer, and the
        Cython must move with it.
        """
        cython = cython_boost("boost_integrate_linear_interp")
        energy, beta = 3.7, 0.6
        base = boost_integrate_linear_interp(energy, beta, FLAT_X, FLAT_Y)
        assert base == cython(energy, beta, FLAT_X, FLAT_Y)

        bumped = FLAT_Y.copy()
        bumped[index] += 1.0
        moved = boost_integrate_linear_interp(energy, beta, FLAT_X, bumped)
        assert moved != base
        assert moved == cython(energy, beta, FLAT_X, bumped)

    def test_the_interior_sum_is_integrated(self) -> None:
        """The trapezoidal sum contributes.

        ``y[3]`` is read by that sum and by nothing else at these bounds.
        """
        cython = cython_boost("boost_integrate_linear_interp")
        energy, beta = 3.7, 0.6
        base = boost_integrate_linear_interp(energy, beta, FLAT_X, FLAT_Y)
        bumped = FLAT_Y.copy()
        bumped[3] += 1.0
        moved = boost_integrate_linear_interp(energy, beta, FLAT_X, bumped)
        assert moved != base
        assert moved == cython(energy, beta, FLAT_X, bumped)

    @pytest.mark.parametrize("name", list(photon_tables()))
    def test_matches_the_cython_on_the_live_tables(self, name: str) -> None:
        """The seven shipped tables, bit for bit.

        Six boost regimes and 400 energies each -- the sweep Phase 04's
        swap will be graded on.
        """
        cython = cython_boost("boost_integrate_linear_interp")
        x, y, mass = photon_tables()[name]
        energies = np.geomspace(1e-3, 1e4, 400)
        mismatched = 0
        total = 0
        for multiple in BOOST_REGIMES:
            beta = boost_beta(mass * multiple, mass)
            got = boost_integrate_linear_interp(energies, beta, x, y)
            want = np.array([cython(float(e), beta, x, y) for e in energies])
            total += energies.size
            mismatched += int(np.count_nonzero(got != want))
        assert (
            mismatched == 0
        ), f"{name}: {mismatched} of {total} points differ from the Cython"


class TestFusedArithmetic:
    """The fused multiply-adds are load-bearing, and this is the proof.

    :func:`unfused_reference` is the same algorithm written the obvious
    way -- ``a * b + c``, as Rust would compile it without ``mul_add``.
    It is an independent implementation as well as a discriminator: it
    reproduces the Cython everywhere the contraction does not bite.
    """

    @staticmethod
    def unfused_reference(
        photon_energy: float, beta: float, x: np.ndarray, y: np.ndarray
    ) -> float:
        """``boost_integrate_linear_interp`` with no contraction anywhere."""
        npts = len(x)
        xmax, x0, y0 = float(x[-1]), float(x[0]), float(y[0])
        gamma = 1.0 / math.sqrt(1.0 - beta * beta)
        lb = photon_energy * gamma * (1.0 - beta)
        ub = photon_energy * gamma * (1.0 + beta)
        if lb > xmax:
            return 0.0
        if ub < x0:
            return y0 * x0 / photon_energy

        integral = 0.0
        ilow = ihigh = -1
        if ub > xmax:
            ub, ihigh = xmax, npts - 1
        if lb < x0:
            rat = (1.0 - beta) * photon_energy * gamma / x0
            integral += y0 * (1.0 - rat) / rat
            lb, ilow = x0, 0
        yy = y / x
        if ilow == -1:
            ilow = int(np.flatnonzero(lb <= x)[0])
        if ihigh == -1:
            ihigh = int(np.flatnonzero(ub <= x)[0])
            if abs(float(x[ihigh]) - ub) > EDGE_ATOL:
                ihigh -= 1
        if ilow < ihigh:
            integral += float(np.trapezoid(yy[ilow:ihigh], x=x[ilow:ihigh]))
        if ilow > 0 and abs(float(x[ilow]) - lb) > EDGE_ATOL:
            x2, x1 = float(x[ilow]), float(x[ilow - 1])
            m = (float(yy[ilow]) - float(yy[ilow - 1])) / (x2 - x1)
            b = float(yy[ilow - 1]) - m * x1
            integral += (x2 - lb) * (0.5 * m * (x2 + lb) + b)
        if ihigh < npts - 1 and abs(ub - float(x[ihigh])) > EDGE_ATOL:
            x1 = float(x[ihigh])
            m = (float(yy[ihigh + 1]) - float(yy[ihigh])) / (float(x[ihigh + 1]) - x1)
            b = float(yy[ihigh]) - m * x1
            integral += (ub - x1) * (0.5 * m * (ub + x1) + b)
        return integral / (2.0 * gamma * beta)

    def test_the_unfused_form_misses_the_cython_and_the_rust_does_not(self) -> None:
        """On the eta table, over the corpus's boost regimes.

        The recorded figure is the point of this test: writing the port
        without ``mul_add`` costs up to a few parts in 1e12, which the
        1e-12 ``TABULATED`` budget does not cover. If a future platform
        stops contracting, ``differ`` goes empty and the assertion says so
        rather than passing silently.
        """
        x, y, mass = photon_tables()["eta"]
        energies = np.geomspace(1e-2, 1e3, 300)
        cython = cython_boost("boost_integrate_linear_interp")

        worst = 0.0
        differ = 0
        for multiple in BOOST_REGIMES:
            beta = boost_beta(mass * multiple, mass)
            for energy in energies:
                want = cython(float(energy), beta, x, y)
                assert boost_integrate_linear_interp(float(energy), beta, x, y) == want
                unfused = self.unfused_reference(float(energy), beta, x, y)
                if want not in (unfused, 0.0):
                    differ += 1
                    worst = max(worst, abs(unfused - want) / abs(want))
        assert differ > 0, (
            "the unfused form matched the Cython everywhere, so this "
            "platform does not contract and the test proves nothing here"
        )
        assert (
            worst > MIN_RECORDED_UNFUSED_MISS
        ), f"unfused worst miss {worst:.3e} is smaller than recorded"


class TestDroppedInteriorCell:
    """The interior sum stops one cell short, and the port keeps it that way.

    ``np.trapezoid(yy[ilow:ihigh], x=x[ilow:ihigh])`` has an exclusive
    upper bound, so the cell ``[x[ihigh - 1], x[ihigh]]`` is covered by
    neither the sum nor the upper partial-cell term. Reproduced rather
    than repaired (``projects/cython-to-rust/rules.md`` rule 1); the
    repair is tracked in
    ``docs/followups/todo/boost-integral-drops-last-interior-cell.md``.
    """

    X = np.array([1.0, 2.0, 3.0, 4.0])
    Y = np.array([1.0, 2.0, 3.0, 4.0])  # y / x == 1, so integrals are lengths

    def test_the_hand_computed_value_omits_one_cell(self) -> None:
        """``E = 2.2``, ``beta = 0.6``: ``lb = 1.1``, ``ub`` clamped to 4.

        The Cython covers ``[1.1, 2]`` (lower partial cell) and ``[2, 3]``
        (the interior sum) for ``1.9 / (2 gamma beta)``. Covering
        ``[1.1, 4]`` as intended would give ``2.9`` -- a 53% difference,
        far too large to be roundoff.
        """
        cython = cython_boost("boost_integrate_linear_interp")
        got = boost_integrate_linear_interp(2.2, 0.6, self.X, self.Y)
        assert got == cython(2.2, 0.6, self.X, self.Y)
        assert got == pytest.approx(1.9 / 1.5, rel=1e-15)
        assert got != pytest.approx(2.9 / 1.5, rel=1e-3)

    def test_a_clamped_window_never_reads_the_tables_last_row(self) -> None:
        """The sharpest form of the drop.

        When the window reaches past the table, ``ihigh`` is the last
        index, the upper partial-cell term is skipped, and the interior
        sum stops one short -- so the final row contributes to nothing.
        Replacing it with a value six orders larger leaves the answer
        bit-identical, in the port and in the Cython alike.
        """
        cython = cython_boost("boost_integrate_linear_interp")
        spoiled = self.Y.copy()
        spoiled[-1] = 1e6
        base = boost_integrate_linear_interp(2.2, 0.6, self.X, self.Y)
        assert boost_integrate_linear_interp(2.2, 0.6, self.X, spoiled) == base
        assert cython(2.2, 0.6, self.X, spoiled) == base


class TestTrapezoidSummation:
    """The interior sum reproduces ``np.trapezoid``, reduction order included.

    ``ndarray.sum`` is pairwise, not sequential, so a left-to-right
    accumulation in Rust would be a different number -- up to 1.8e-15
    relative on the 500-row tables. ``rust/src/boost.rs`` mirrors NumPy's
    blocking; this is the pin, and it survives the ``.pyx``'s deletion.

    The construction puts both bounds exactly on nodes, which switches off
    the tail, the clamp and both partial-cell terms, leaving the interior
    sum as the whole answer.
    """

    @pytest.mark.parametrize("nodes", [12, 41, 130, 260, 1001])
    def test_the_interior_sum_is_numpys_trapezoid(self, nodes: int) -> None:
        beta = 0.6
        gamma = rust_gamma(beta)
        energy = 40.0
        lb = energy * gamma * (1.0 - beta)
        ub = energy * gamma * (1.0 + beta)

        # A table whose nodes include `lb` and `ub` exactly, with a node
        # outside each bound so neither the tail nor the clamp fires.
        interior = np.linspace(lb, ub, nodes)
        x = np.concatenate([[lb * 0.5], interior, [ub * 1.5]])
        rng = np.random.default_rng(nodes)
        y = rng.uniform(0.1, 10.0, x.size) * x

        got = boost_integrate_linear_interp(energy, beta, x, y)
        yy = y / x
        # `ilow` is 1 (the node at `lb`), `ihigh` the node at `ub`; the
        # Cython's slice stops one short of `ihigh`.
        ihigh = x.size - 2
        want = np.trapezoid(yy[1:ihigh], x=x[1:ihigh]) / (2.0 * gamma * beta)
        assert got == want

    def test_a_sequential_sum_would_be_a_different_number(self) -> None:
        """Otherwise the test above would pass against either reduction."""
        rng = np.random.default_rng(11)
        values = rng.standard_normal(500) * 10.0 ** rng.uniform(-6, 6, 500)
        sequential = 0.0
        for value in values:
            sequential += float(value)
        assert sequential != float(values.sum())


class TestErrors:
    """Guards the Cython states as ``assert`` become raises (rule 9)."""

    def test_beta_outside_the_open_unit_interval_raises(self) -> None:
        for beta in (0.0, 1.0, -0.5, 1.5, float("nan")):
            with pytest.raises(ValueError, match="0 < beta < 1"):
                boost_integrate_linear_interp(1.0, beta, FLAT_X, FLAT_Y)

    def test_the_cython_also_refuses_those_betas(self) -> None:
        """The port tightens an ``assert`` into a raise.

        It does not invent a restriction the Cython does not have.
        """
        cython = cython_boost("boost_integrate_linear_interp")
        for beta in (0.0, 1.0, -0.5, 1.5):
            with pytest.raises(AssertionError):
                cython(1.0, beta, FLAT_X, FLAT_Y)

    def test_mismatched_columns_raise(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            boost_integrate_linear_interp(1.0, 0.5, FLAT_X, FLAT_Y[:-1])

    def test_an_empty_table_raises(self) -> None:
        """New in the port.

        The Cython reads ``x[npts - 1]`` unchecked, so an empty table is
        undefined behavior there rather than an error.
        """
        empty = np.array([], dtype=np.float64)
        with pytest.raises(ValueError, match="must not be empty"):
            boost_integrate_linear_interp(1.0, 0.5, empty, empty)


class TestDispatch:
    """The swept argument follows the usual scalar-or-1-D contract.

    ``test/test_core_dispatch.py`` pins that contract branch by branch;
    this only checks the three wrappers are wired into it.
    """

    @pytest.mark.parametrize(
        ("call", "scalar"),
        [
            (lambda x: boost_beta(x, 139.57039), 1000.0),
            (lambda x: boost_gamma(x, 139.57039), 1000.0),
            (lambda x: boost_delta_function(200.0, x, 0.0, 0.6), 220.0),
            (
                lambda x: boost_integrate_linear_interp(x, 0.6, FLAT_X, FLAT_Y),
                3.7,
            ),
        ],
        ids=["beta", "gamma", "delta", "integral"],
    )
    def test_scalar_and_array_paths_agree(
        self, call: Callable[[object], object], scalar: float
    ) -> None:
        grid = np.array([scalar, scalar * 1.3, scalar * 0.7])
        single = call(scalar)
        assert isinstance(single, float)
        swept = call(grid)
        assert isinstance(swept, np.ndarray)
        assert swept.dtype == np.float64
        assert swept[0] == single
        assert np.array_equal(swept, [call(float(v)) for v in grid])

    @pytest.mark.parametrize(
        ("call", "scalar"),
        [
            (lambda x: boost_beta(x, 139.57039), 1000.0),
            (lambda x: boost_gamma(x, 139.57039), 1000.0),
            (lambda x: boost_delta_function(200.0, x, 0.0, 0.6), 220.0),
            (
                lambda x: boost_integrate_linear_interp(x, 0.6, FLAT_X, FLAT_Y),
                3.7,
            ),
        ],
        ids=["beta", "gamma", "delta", "integral"],
    )
    def test_bad_shapes_and_dtypes_raise(
        self, call: Callable[[object], object], scalar: float
    ) -> None:
        del scalar
        with pytest.raises(ValueError, match="0 or 1-dimensional"):
            call(np.zeros((2, 2)))
        with pytest.raises(ValueError, match="float64 array"):
            call(np.array([1, 2], dtype=np.int64))
