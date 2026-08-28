r"""``hazma._core.boost`` against an independent reference.

``rust/src/boost.rs`` carries the four routines the spectra share:
:func:`boost_beta` and :func:`boost_gamma` turn a parent's energy and mass
into boost parameters, ``boost_delta_function`` boosts a two-body line,
and ``boost_integrate_linear_interp`` boosts a tabulated continuum.
cython-to-rust Task 3.4 ported all four from ``hazma/_utils/boost.pyx``,
and Task 6.4 deleted that file.

The oracle is a reference implementation
----------------------------------------
The phase file asked for "micro-fixtures captured in Phase 01". There are
none, and there never could be: the parity corpus enumerates top-level
``def``\ s, and every routine here was ``cdef`` -- private to the C level
and invisible to the corpus by construction. Nor does the corpus reach
them indirectly in a way this module could rely on, because
``hazma._core.boost`` is a test-only submodule (``cases.py``'s
``_CORE_TEST_ONLY_MODULES``): the kernels that use it call it natively, in
Rust.

What stands in place of both is :func:`delta_function_reference` and
:func:`integrate_reference` -- the two kernels transcribed into Python and
NumPy from ``rust/src/boost.rs``, site for site, with their multiply-adds
injected as a parameter. Passed ``mul_add=fma`` they are what the port
must equal **bit for bit on every platform**, because ``f64::mul_add`` is
correctly rounded whether or not the target has an FMA instruction and
``sqrt`` is exact; passed ``mul_add=unfused`` they are the same algorithm
written the obvious way, which is how :class:`TestFusedArithmetic` pins
the fusion in both directions rather than asserting it.

That is a stronger oracle than the one this module used until Task 6.4,
and deliberately so. Until then the comparison was against the live
``.pyx`` through ``hazma._utils.boost.__pyx_capi__``, which made every
claim a statement about the C compiler that built it: bit-equality held
only on the platform the corpus was captured on, and off it the module
carried a measured tolerance instead. The reference removes the platform
from the claim entirely.

What the compiled twin cost, while it existed
---------------------------------------------
The measurement is kept because it is what set the surviving tolerances
elsewhere and it is not recoverable now. Built for linux/amd64 (Debian,
gcc, glibc, CPython 3.12.13, NumPy 2.5.1), the Cython and the port
differed by:

============================== ============= ================
comparison                     max relative  max ``|Δ|``/peak
============================== ============= ================
``boost_delta_function``       1.9e-13       7.3e-17
``…_interp``, eta              9.9e-13       1.2e-15
``…_interp``, eta_prime        6.3e-13       7.2e-16
``…_interp``, charged_kaon     3.6e-13       3.2e-15
``…_interp``, long_kaon        1.3e-13       3.1e-15
``…_interp``, short_kaon       1.2e-13       3.4e-15
``…_interp``, omega            2.6e-13       3.3e-15
``…_interp``, phi              9.1e-14       5.9e-16
============================== ============= ================

Over 40,000 delta-function draws and 16,800 tabulated points: no sign
flip, no NaN, and -- the statement rounding cannot excuse -- **no
disagreement anywhere about which energies are zero**. The magnitudes are
what the unfused arithmetic costs: baseline x86-64 has no hardware FMA,
and the worst delta-function disagreement measured against the Linux
Cython (1.8683e-13) is the number
:meth:`TestFusedArithmetic.test_the_delta_function_port_is_the_fused_reference`
recovers from the unfused Python reference, on any platform, over the
same 40,000 draws. The two agreeing is the direct evidence that a
compiler with no FMA to contract into simply computes that reference --
which is why deleting the ``.pyx`` costs this module no coverage.

Lifetime
--------
Nothing here reads the Cython any more, so nothing here has a deadline.
:class:`TestFusedArithmetic` and :class:`TestTrapezoidSummation` pin the
arithmetic, :class:`TestDroppedInteriorCell` pins the one defect this
module knows the size of (53%, far above any rounding),
:class:`TestBoostDeltaFunction` and
:class:`TestBoostIntegrateLinearInterp` pin each branch by a closed form
or a sensitivity check as well as against the reference, and
:class:`TestErrors` and :class:`TestDispatch` pin the boundary. Rule 9's
tightening -- the ``.pyx``'s ``assert``\ s became unconditional raises --
is stated by :class:`TestErrors`; that the Cython also refused those
``beta`` values was asserted against the twin until Task 6.4, and is now
recorded here rather than executed.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from hazma._core import boost as core_boost
from hazma.parameters import (
    charged_kaon_mass,
    eta_mass,
    eta_prime_mass,
    neutral_kaon_mass,
    omega_mass,
    phi_mass,
)

boost_beta = core_boost.boost_beta
boost_gamma = core_boost.boost_gamma
boost_delta_function = core_boost.boost_delta_function
boost_integrate_linear_interp = core_boost.boost_integrate_linear_interp

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# The Cython oracle
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Which platform this is, and what the comparison costs off it
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Reference implementations, parameterised by their multiply-add
# --------------------------------------------------------------------------


def fma(a: float, b: float, c: float) -> float:
    """``a * b + c`` with a single rounding, without ``math.fma``.

    ``math.fma`` arrived in Python 3.13 and this suite supports 3.10, so
    the fused product is computed exactly as a rational and rounded once
    by ``float()``, which rounds to nearest. Passed to the references
    below to reproduce ``f64::mul_add``, which is correctly rounded on
    every target Rust supports whether or not the hardware has an FMA.
    """
    return float(Fraction(a) * Fraction(b) + Fraction(c))


def unfused(a: float, b: float, c: float) -> float:
    """``a * b + c`` with the product rounded before the sum."""
    return a * b + c


#: The Cython's absolute tolerance for "this bound sits on a node"
#: (``hazma/_utils/boost.pyx:212``), reused by
#: :func:`integrate_reference` and wherever a test has to predict which
#: branch fires.
EDGE_ATOL = 1e-6


def rust_gamma(beta: float) -> float:
    """``1 / sqrt(1 - beta**2)`` as ``rust/src/boost.rs`` computes it."""
    return 1.0 / math.sqrt(fma(-beta, beta, 1.0))


def delta_function_reference(
    e0: float,
    e: float,
    m: float,
    beta: float,
    *,
    mul_add: Callable[[float, float, float], float],
) -> float:
    """``boost_delta_function`` with its five multiply-adds injected.

    ``mul_add=fma`` is ``rust/src/boost.rs:164-170`` transcribed site for
    site; ``mul_add=unfused`` is the same algorithm written the obvious
    way. :class:`TestFusedArithmetic` asserts the port is the first and not
    the second, which pins the fusion without asking what any compiler did.
    """
    if beta > 1.0 or beta <= 0.0 or e < m:
        return 0.0
    gamma = 1.0 / math.sqrt(mul_add(-beta, beta, 1.0))
    k = math.sqrt(mul_add(e, e, -(m * m)))
    if gamma * mul_add(-beta, k, e) < e0 < gamma * mul_add(beta, k, e):
        return 1.0 / (2.0 * gamma * beta * math.sqrt(mul_add(e0, e0, -(m * m))))
    return 0.0


def integrate_reference(
    photon_energy: float,
    beta: float,
    x: np.ndarray,
    y: np.ndarray,
    *,
    mul_add: Callable[[float, float, float], float],
) -> float:
    """``boost_integrate_linear_interp`` with its seven multiply-adds injected.

    The counterpart of :func:`delta_function_reference` for the tabulated
    branch; the fused sites are ``rust/src/boost.rs:257`` and ``:326-328`` /
    ``:336-338``.
    """
    npts = len(x)
    xmax, x0, y0 = float(x[-1]), float(x[0]), float(y[0])
    gamma = 1.0 / math.sqrt(mul_add(-beta, beta, 1.0))
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
        b = mul_add(-m, x1, float(yy[ilow - 1]))
        inner = mul_add(0.5 * m, x2 + lb, b)
        integral = mul_add(x2 - lb, inner, integral)
    if ihigh < npts - 1 and abs(ub - float(x[ihigh])) > EDGE_ATOL:
        x2, x1 = float(x[ihigh + 1]), float(x[ihigh])
        m = (float(yy[ihigh + 1]) - float(yy[ihigh])) / (x2 - x1)
        b = mul_add(-m, x1, float(yy[ihigh]))
        inner = mul_add(0.5 * m, ub + x1, b)
        integral = mul_add(ub - x1, inner, integral)
    return integral / (2.0 * gamma * beta)


# --------------------------------------------------------------------------
# Fixtures and constants
# --------------------------------------------------------------------------


#: ``kernel name -> shipped CSV``. Until cython-to-rust Task 4.2 these
#: tables were read off the ``.pyx`` modules' import-time globals
#: (``_eta.eta_data_energies`` and friends); that task moved the parse
#: into Rust and deleted the five extensions, so the tables are now read
#: the way those modules read them — which is also what keeps this an
#: oracle independent of the Rust that consumes it.
PHOTON_CSVS = {
    "eta": "eta_photon.csv",
    "eta_prime": "eta_prime_photon.csv",
    "charged_kaon": "charged_kaon_photon.csv",
    "long_kaon": "long_kaon_photon.csv",
    "short_kaon": "short_kaon_photon.csv",
    "omega": "omega_photon.csv",
    "phi": "phi_photon.csv",
}

PHOTON_DATA_DIR = REPO_ROOT / "hazma" / "spectra" / "_photon" / "data"


def load_photon_table(csv: str) -> tuple[np.ndarray, np.ndarray]:
    """``np.loadtxt(...).T`` then ``np.sum(rows[1:], axis=0)``.

    The two lines every one of the five deleted ``.pyx`` files opened
    with, reproduced verbatim so the tables are the same doubles they
    always were.
    """
    data = np.loadtxt(PHOTON_DATA_DIR / csv, delimiter=",").T
    return data[0], np.sum(data[1:], axis=0)


#: ``kernel name -> parent mass in MeV``, the boost's second argument.
PHOTON_MASSES = {
    "eta": eta_mass,
    "eta_prime": eta_prime_mass,
    "charged_kaon": charged_kaon_mass,
    "long_kaon": neutral_kaon_mass,
    "short_kaon": neutral_kaon_mass,
    "omega": omega_mass,
    "phi": phi_mass,
}


def photon_tables() -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    """The seven live ``(energies, dnde, parent_mass)`` triples."""
    return {
        name: (*load_photon_table(csv), PHOTON_MASSES[name])
        for name, csv in PHOTON_CSVS.items()
    }


#: Parent-energy multiples spanning the regimes the parity corpus uses --
#: just off rest, near rest, mildly boosted, strongly boosted.
BOOST_REGIMES = (1.000_000_001, 1.05, 1.5, 2.0, 3.0, 10.0)

#: A flat toy table with ``y / x == 1`` everywhere, so every branch's
#: contribution is a length and can be predicted by hand.
FLAT_X = np.arange(1.0, 9.0)
FLAT_Y = FLAT_X.copy()

#: The smallest miss the unfused arithmetic is recorded as producing; a
#: smaller one means :func:`integrate_reference` stopped discriminating.
MIN_RECORDED_UNFUSED_MISS = 1e-13


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
        someone does. Platform-independent: both sides are Python and Rust
        arithmetic, neither of which contracts on its own.
        """
        ratio = mass / energy
        assert boost_beta(energy, mass) == math.sqrt(1.0 - ratio * ratio)

    def test_a_particle_at_rest_has_zero_velocity_and_unit_gamma(self) -> None:
        assert boost_beta(139.57039, 139.57039) == 0.0
        assert boost_gamma(139.57039, 139.57039) == 1.0

    def test_below_rest_energy_beta_is_nan(self) -> None:
        assert np.isnan(boost_beta(100.0, 139.57039))


class TestBoostDeltaFunction:
    """The boosted two-body line, branch by branch."""

    @staticmethod
    def _support_edge(
        fn: Callable[..., float],
        line: tuple[float, float, float],
        bracket: tuple[float, float],
    ) -> float | None:
        """The double at which ``fn``'s support flips inside ``bracket``.

        Bisects on the bit pattern, so the answer is the boundary itself
        rather than a nearby sample. ``None`` when the bracket does not
        straddle a transition for these parameters.

        Parameters
        ----------
        fn : callable
            ``fn(e0, e, m, beta)`` -- either implementation.
        line : tuple of float
            ``(e0, m, beta)``, the parameters held fixed while ``e`` moves.
        bracket : tuple of float
            ``(lo, hi)``, the product energies to bisect between.
        """
        e0, m, beta = line
        lo, hi = bracket
        lo_bits = int(np.float64(lo).view(np.int64))
        hi_bits = int(np.float64(hi).view(np.int64))
        lo_zero = fn(e0, lo, m, beta) == 0.0
        if lo_zero == (fn(e0, hi, m, beta) == 0.0):
            return None
        while hi_bits - lo_bits > 1:
            mid_bits = (lo_bits + hi_bits) // 2
            mid = float(np.int64(mid_bits).view(np.float64))
            if (fn(e0, mid, m, beta) == 0.0) == lo_zero:
                lo_bits = mid_bits
            else:
                hi_bits = mid_bits
        return float(np.int64(hi_bits).view(np.float64))

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
        assert boost_delta_function(e0, e, m, beta) == 0.0
        if beta == 1.0:
            # `beta == 1` passes the shared guard (`beta > 1.0`, not `>=`)
            # in both implementations, and the two then differ in language
            # rather than in algorithm: `gamma` is `1 / sqrt(0)`, which is
            # `+inf` under IEEE-754 in Rust -- so the height `1 / (2 gamma
            # beta k0)` underflows to the 0.0 asserted above -- while
            # Python raises `ZeroDivisionError` instead of producing the
            # infinity. The reference is only an oracle where it is
            # defined; the port's answer is pinned above regardless.
            with pytest.raises(ZeroDivisionError):
                delta_function_reference(e0, e, m, beta, mul_add=fma)
        else:
            assert delta_function_reference(e0, e, m, beta, mul_add=fma) == 0.0


class TestBoostIntegrateLinearInterp:
    """The tabulated continuum boost, branch by branch.

    Every case asserts agreement with :func:`integrate_reference` *and*
    pins the branch that fired -- either by a closed form the branch alone
    can produce, or by a sensitivity check on a table entry only that
    branch reads.
    """

    def test_the_whole_window_above_the_table_is_zero(self) -> None:
        energy, beta = 1e6, 0.5
        got = boost_integrate_linear_interp(energy, beta, FLAT_X, FLAT_Y)
        assert got == 0.0
        assert got == integrate_reference(energy, beta, FLAT_X, FLAT_Y, mul_add=fma)

    def test_the_whole_window_below_the_table_is_the_analytic_tail(self) -> None:
        """``y0 * x0 / E``, a closed form no other branch can produce."""
        energy, beta = 1e-6, 0.5
        got = boost_integrate_linear_interp(energy, beta, FLAT_X, FLAT_Y)
        assert got == FLAT_Y[0] * FLAT_X[0] / energy
        assert got == integrate_reference(energy, beta, FLAT_X, FLAT_Y, mul_add=fma)

    def test_a_window_straddling_the_table_floor_adds_the_tail(self) -> None:
        """`lb` below the table, `ub` inside it.

        Pinned by sensitivity: only the tail term reads ``y[0]`` when
        ``ilow`` is 0, and only through the ``y0 * (1 - rat) / rat``
        factor, so scaling ``y[0]`` must scale the tail's contribution.
        """
        energy, beta = 1.4, 0.6
        base = boost_integrate_linear_interp(energy, beta, FLAT_X, FLAT_Y)
        bumped = FLAT_Y.copy()
        bumped[0] *= 2.0
        moved = boost_integrate_linear_interp(energy, beta, FLAT_X, bumped)
        assert moved != base
        assert base == integrate_reference(energy, beta, FLAT_X, FLAT_Y, mul_add=fma)
        assert moved == integrate_reference(energy, beta, FLAT_X, bumped, mul_add=fma)

    def test_a_window_above_the_table_ceiling_clamps(self) -> None:
        """`ub` past the table's top, but `lb` inside it.

        The clamp is what keeps this from returning zero, and the value
        matches the reference. That the clamp *also* skips the upper
        partial-cell term — and with it the table's last row — is pinned
        separately in :class:`TestDroppedInteriorCell`.
        """
        energy, beta = 5.0, 0.6
        got = boost_integrate_linear_interp(energy, beta, FLAT_X, FLAT_Y)
        assert got != 0.0
        assert got == integrate_reference(energy, beta, FLAT_X, FLAT_Y, mul_add=fma)

    @pytest.mark.parametrize("index", [0, 7])
    def test_both_partial_cells_are_integrated(self, index: int) -> None:
        """`lb` and `ub` both strictly inside cells.

        ``y[0]`` is read only by the lower partial cell here (``ilow`` is
        1, so the tail does not fire) and ``y[7]`` only by the upper one
        (``ihigh`` is 6). Perturbing either must move the answer, and the
        reference must move with it.
        """
        energy, beta = 3.7, 0.6
        base = boost_integrate_linear_interp(energy, beta, FLAT_X, FLAT_Y)
        bumped = FLAT_Y.copy()
        bumped[index] += 1.0
        moved = boost_integrate_linear_interp(energy, beta, FLAT_X, bumped)
        assert moved != base
        assert base == integrate_reference(energy, beta, FLAT_X, FLAT_Y, mul_add=fma)
        assert moved == integrate_reference(energy, beta, FLAT_X, bumped, mul_add=fma)

    def test_the_interior_sum_is_integrated(self) -> None:
        """The trapezoidal sum contributes.

        ``y[3]`` is read by that sum and by nothing else at these bounds.
        """
        energy, beta = 3.7, 0.6
        base = boost_integrate_linear_interp(energy, beta, FLAT_X, FLAT_Y)
        bumped = FLAT_Y.copy()
        bumped[3] += 1.0
        moved = boost_integrate_linear_interp(energy, beta, FLAT_X, bumped)
        assert moved != base
        assert moved == integrate_reference(energy, beta, FLAT_X, bumped, mul_add=fma)


class TestFusedArithmetic:
    """The fused multiply-adds are load-bearing, and this is the proof.

    :func:`integrate_reference` and :func:`delta_function_reference` are
    the two kernels written twice over -- once with ``fma`` at exactly the
    sites ``rust/src/boost.rs`` spells ``mul_add``, and once the obvious
    way. Both are pure Python and NumPy, so both are the same numbers on
    every platform, and so is the port: ``f64::mul_add`` is correctly
    rounded whether or not the target has an FMA instruction, and
    ``sqrt`` is exact. That makes "the port fuses here and only here" a
    claim this class can assert **everywhere**.

    It could not before. Until 2026-08-12 the discriminator was the
    *Cython*, which fuses only where its compiler chose to, so the class
    was gated on a probe and skipped wherever the probe said no -- which
    on Linux/x86-64 was always. Comparing against a reference instead
    removes the platform from the claim entirely, and the class now
    outlives the ``.pyx``.
    """

    @pytest.mark.parametrize("name", list(photon_tables()))
    def test_the_tabulated_port_is_the_fused_reference(self, name: str) -> None:
        """Bit-for-bit against ``mul_add=fma``, on every platform."""
        x, y, mass = photon_tables()[name]
        energies = np.geomspace(1e-2, 1e3, 300)
        for multiple in BOOST_REGIMES:
            beta = boost_beta(mass * multiple, mass)
            for energy in energies:
                assert boost_integrate_linear_interp(
                    float(energy), beta, x, y
                ) == integrate_reference(float(energy), beta, x, y, mul_add=fma), (
                    f"{name} at {multiple}x rest, {energy=!r}: the port is not "
                    f"the fused reference"
                )

    def test_the_unfused_tabulated_form_would_be_a_different_number(self) -> None:
        """Otherwise the test above would pass against either arithmetic.

        The recorded figure is the point: writing the port without
        ``mul_add`` costs up to a few parts in 1e12, which the 1e-12
        ``TABULATED`` budget in ``test/parity/tolerances.py`` does not
        cover. It is also, to the digit, the disagreement measured against
        the *Linux* Cython (module docstring) -- which is what a compiler
        with no hardware FMA to contract into produces.
        """
        x, y, mass = photon_tables()["eta"]
        energies = np.geomspace(1e-2, 1e3, 300)

        worst = 0.0
        differ = 0
        for multiple in BOOST_REGIMES:
            beta = boost_beta(mass * multiple, mass)
            for energy in energies:
                want = integrate_reference(float(energy), beta, x, y, mul_add=fma)
                unfused_value = integrate_reference(
                    float(energy), beta, x, y, mul_add=unfused
                )
                if want not in (unfused_value, 0.0):
                    differ += 1
                    worst = max(worst, abs(unfused_value - want) / abs(want))
        assert differ > 0, "the two arithmetics agreed everywhere on the eta table"
        assert (
            worst > MIN_RECORDED_UNFUSED_MISS
        ), f"unfused worst miss {worst:.3e} is smaller than recorded"

    def test_the_delta_function_port_is_the_fused_reference(self) -> None:
        """The same statement for the two-body line, over 40,000 draws.

        The recorded worst miss is what the module docstring's
        delta-function row compares against: the unfused form costs up to
        1.87e-13 relative here, and 1.8683e-13 is what the *Linux* Cython
        was measured to cost against the port over these same draws. The
        two agreeing is the direct evidence that a compiler with no
        hardware FMA to contract into simply computes this reference.
        """
        rng = np.random.default_rng(20_260_810)
        masses = [0.0, 0.510_998_928]
        differ = 0
        worst = 0.0
        for _ in range(40_000):
            m = float(rng.choice(masses))
            beta = float(rng.uniform(1e-6, 1.0 - 1e-9))
            e0 = float(10.0 ** rng.uniform(-1.0, 3.0))
            e = float(e0 * 10.0 ** rng.uniform(-0.4, 0.4))
            got = boost_delta_function(e0, e, m, beta)
            assert got == delta_function_reference(e0, e, m, beta, mul_add=fma)
            unfused_value = delta_function_reference(e0, e, m, beta, mul_add=unfused)
            if got != unfused_value:
                differ += 1
                if got:
                    worst = max(worst, abs(unfused_value - got) / abs(got))
        assert differ > 0, (
            "the fused and unfused forms agreed on all 40,000 draws, so this "
            "test cannot distinguish them"
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

        The kernel covers ``[1.1, 2]`` (lower partial cell) and ``[2, 3]``
        (the interior sum) for ``1.9 / (2 gamma beta)``. Covering
        ``[1.1, 4]`` as intended would give ``2.9`` -- a 53% difference,
        far too large to be roundoff.
        """
        got = boost_integrate_linear_interp(2.2, 0.6, self.X, self.Y)
        assert got == pytest.approx(1.9 / 1.5, rel=1e-15)
        assert got != pytest.approx(2.9 / 1.5, rel=1e-3)
        assert got == integrate_reference(2.2, 0.6, self.X, self.Y, mul_add=fma)

    def test_a_clamped_window_never_reads_the_tables_last_row(self) -> None:
        """The sharpest form of the drop.

        When the window reaches past the table, ``ihigh`` is the last
        index, the upper partial-cell term is skipped, and the interior
        sum stops one short -- so the final row contributes to nothing.
        Replacing it with a value six orders larger leaves the answer
        bit-identical, in the port and in the reference alike.
        """
        spoiled = self.Y.copy()
        spoiled[-1] = 1e6
        base = boost_integrate_linear_interp(2.2, 0.6, self.X, self.Y)
        assert boost_integrate_linear_interp(2.2, 0.6, self.X, spoiled) == base
        assert integrate_reference(
            2.2, 0.6, self.X, spoiled, mul_add=fma
        ) == integrate_reference(2.2, 0.6, self.X, self.Y, mul_add=fma)


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
