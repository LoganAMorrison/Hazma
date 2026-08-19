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

Why the comparison has two modes
--------------------------------
Bit-equality against a *compiled* twin is a statement about the build that
produced it, not about this port. Until 2026-08-12 this module tried to
*detect* that condition instead of declaring it: ``cython_contracts()``
compared the compiled ``boost_delta_function`` against an unfused Python
transcription and skipped every claim against the Cython where the two
agreed, on the theory that a build which does not fuse its multiply-adds
is simply a different arithmetic. Task 4.1 showed the same mechanism to be
unsound in ``test/test_core_positron_muon.py`` (PR #63, runs 31562223329
and 31564747071): a probe over one mechanism cannot see the others, so it
answers "contracts" on platforms where nothing does and the bit-equality
assertions then fail, or answers "does not contract" and silently voids
the whole comparison. **This module was resolving the second way.** On
Linux/x86-64 the probe returned ``False``, all 19 of its
cross-implementation claims skipped, and that gate had been vacuous on
every CI entry but macOS since PR #61 -- measured directly, and
corroborated by the skip counts of master run 31619425557
(``projects/cython-to-rust/task-notes/phase-03/task-3.4-interp-boost.md``
carries the arithmetic).

So the *mode* is declared from the platform, and the divergence off it was
**measured rather than assumed** -- by building this worktree for
linux/amd64 (Debian, gcc, glibc, CPython 3.12.13, NumPy 2.5.1) and
comparing the two implementations directly:

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
what the unfused arithmetic costs, which is the expected finding here:
baseline x86-64 has no hardware FMA, and the worst delta-function
disagreement measured against the Linux Cython (1.8683e-13) is the number
:meth:`TestFusedArithmetic.test_the_delta_function_port_is_the_fused_reference`
recovers from an unfused Python reference, on any platform, over the same
40,000 draws.

:data:`OFF_PLATFORM_BUDGET` is scaled to the **peak of the compared
array** rather than applied pointwise, and the two columns above are why:
they differ by three orders of magnitude, because the worst *relative*
disagreements land where the integrand cancels. On the eta table the two
maxima together bound the value carrying the worst relative gap at
1.2e-3 of the peak -- so a pointwise ``rtol`` wide enough to admit it
would be a thousand times wider than what the spectrum actually is, which
is the shape Task 3.4 rejected a tolerance over. Against the peak -- what
a downstream integral or limit sees -- the whole population fits in
3.4e-15. Both arms are asserted (``atol = BUDGET * peak`` with
``rtol = BUDGET``), and the figure is set from the tighter reading: 1e-10
is 100x the worst measured relative disagreement and 2.9e4x the worst
measured peak-relative one. A wrong branch, a dropped term or a bad
constant lands at O(1) against that -- :class:`TestDroppedInteriorCell`
measures the one defect this module knows the size of at 53%.

Support edges get their own budget, in ulps
-------------------------------------------
``eminus`` and ``eplus`` never appear in a returned value; they only
decide whether it is ``1/(2 gamma beta k0)`` or zero. So the port's agreement
with the Cython about *where* the window ends cannot be expressed as a
tolerance on the output -- one implementation returns a finite number and
the other returns zero. It is expressed instead as a distance in ulps
between the two support boundaries, located by bisection on the bit
pattern. Same two modes: the same double on the capturing platform, and
within :data:`OFF_PLATFORM_EDGE_ULPS` elsewhere.

The parity corpus (``test/parity/``) draws the same line in the same
place: its ``EXACT`` budget class is bit-equality on the capturing
platform and :data:`tolerances.PLATFORM_EXACT_RTOL` once the libm
changes. :data:`CAPTURE_MACHINE` is read out of the corpus manifest so
the two scopes cannot drift apart.

Lifetime
--------
Everything comparing against ``__pyx_capi__`` dies with the ``.pyx`` in
Phase 06 Task 6.4. :class:`TestFusedArithmetic`,
:class:`TestTrapezoidSummation`, :class:`TestDroppedInteriorCell`,
:class:`TestErrors` and :class:`TestDispatch` do not, and are what remains
as the standing check afterwards. :class:`TestFusedArithmetic` earns that
by comparing the port against a Python reference rather than against the
Cython: ``rust/src/boost.rs``'s twelve ``mul_add`` sites are pinned in
both directions on every platform, which is the claim the old
Cython-versus-unfused form could only make on one.
"""

from __future__ import annotations

import ctypes
import json
import math
import platform
import sys
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

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

boost_beta = core_boost.boost_beta
boost_gamma = core_boost.boost_gamma
boost_delta_function = core_boost.boost_delta_function
boost_integrate_linear_interp = core_boost.boost_integrate_linear_interp

REPO_ROOT = Path(__file__).resolve().parents[1]


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


# --------------------------------------------------------------------------
# Which platform this is, and what the comparison costs off it
# --------------------------------------------------------------------------

#: The platform the parity corpus was captured on, read from its own
#: manifest so the two can never drift apart. `test/parity` demands
#: bit-equality of its `EXACT` class only on this platform, for exactly the
#: reason in the module docstring; this module is the same kind of oracle
#: and carries the same scope.
CAPTURE_MACHINE = json.loads(
    (REPO_ROOT / "test" / "parity" / "data" / "manifest.json").read_text()
)["environment"]["machine"]

ON_THE_CAPTURING_PLATFORM = (
    sys.platform == "darwin" and platform.machine() == CAPTURE_MACHINE
)

#: The off-platform budget, as a fraction of the peak of the array being
#: compared. Set from the linux/amd64 measurement in the module docstring:
#: 100x the worst *relative* disagreement observed there (9.9e-13, the eta
#: table at the strongest boost) and 2.9e4x the worst peak-relative one
#: (3.4e-15). The relative reading is the tighter of the two and is what
#: chose the figure; the peak scaling is what keeps the tolerance from
#: having to admit the catastrophic-cancellation points. Applied as
#: `assert_allclose`'s `atol` scaled by the peak, with `rtol` at the same
#: figure so a large value is held to the same standard.
OFF_PLATFORM_BUDGET = 1e-10

#: How far the two implementations' support boundaries may sit apart off
#: the capturing platform, in ulps of the boundary itself. Measured at
#: **55 ulps worst** over 800 bracketed edges on linux/amd64 (547 of them
#: on the identical double); 4096 leaves 74x headroom and still corresponds
#: to a relative displacement of 9.1e-13, eleven orders below anything a
#: wrong edge formula would produce.
OFF_PLATFORM_EDGE_ULPS = 4096


def _as_array(values: object) -> np.ndarray:
    """A 1-D float64 view of a scalar or array, for the comparisons below."""
    return np.atleast_1d(np.asarray(values, dtype=np.float64))


def assert_within_the_off_platform_budget(
    got: object, want: object, context: str
) -> None:
    """Assert two results agree to :data:`OFF_PLATFORM_BUDGET` of the peak.

    Split out from :func:`assert_matches_the_cython` so the budget can be
    exercised on *every* platform, including the one where the caller would
    otherwise take the bit-equality branch and leave this untested --
    :meth:`TestOffPlatformBudgets.test_the_value_budget_rejects_a_real_error`.

    ``atol`` is scaled by the peak rather than left at zero because the
    relative error is unbounded where the integrand cancels: see "Why the
    comparison has two modes".
    """
    got, want = _as_array(got), _as_array(want)
    finite = np.isfinite(want)
    peak = float(np.abs(want[finite]).max()) if finite.any() else 0.0
    np.testing.assert_allclose(
        got,
        want,
        rtol=OFF_PLATFORM_BUDGET,
        atol=OFF_PLATFORM_BUDGET * peak,
        err_msg=(
            f"{context}: the port left the Cython's budget of "
            f"{OFF_PLATFORM_BUDGET:.0e} x the peak ({peak:.6e}). Rounding "
            f"between two builds was measured at 3.4e-15 x peak, so this is "
            f"a defect, not a platform difference."
        ),
    )


def assert_matches_the_cython(got: object, want: object, context: str) -> None:
    """The oracle, in whichever of its two modes this platform gets.

    Bit-for-bit where the corpus was captured -- the port was written
    against *this* build's arithmetic and reproduces it exactly, which is a
    far stronger statement than any tolerance. A budget elsewhere, because
    off it the comparison measures the C compiler rather than the port.
    """
    if ON_THE_CAPTURING_PLATFORM:
        assert _as_array(got).tobytes() == _as_array(want).tobytes(), (
            f"{context}: not bit-equal to the Cython on the platform the "
            f"corpus was captured on, where the port is written to reproduce "
            f"it exactly"
        )
        return
    assert_within_the_off_platform_budget(got, want, context)


def ulps_between(a: float, b: float) -> int:
    """How many doubles separate ``a`` and ``b``, both finite and positive."""
    return abs(int(np.float64(a).view(np.int64)) - int(np.float64(b).view(np.int64)))


def assert_the_edge_is_within_the_off_platform_budget(
    got: float, want: float, context: str
) -> None:
    """Assert two support boundaries sit within :data:`OFF_PLATFORM_EDGE_ULPS`.

    Split out from :func:`assert_the_support_edge_matches_the_cython` for
    the same reason the value budget is -- on the capturing platform the
    caller takes the exact branch, so only a direct call can show this one
    still bites
    (:meth:`TestOffPlatformBudgets.test_the_edge_budget_rejects_a_real_error`).
    """
    apart = ulps_between(got, want)
    assert apart <= OFF_PLATFORM_EDGE_ULPS, (
        f"{context}: the port's support boundary is {apart} ulp(s) from the "
        f"Cython's ({got!r} vs {want!r}), past the {OFF_PLATFORM_EDGE_ULPS}-ulp "
        f"budget. Rounding between two builds was measured at 55 ulps."
    )


def assert_the_support_edge_matches_the_cython(
    got: float, want: float, context: str
) -> None:
    """The two implementations' window boundaries sit on the same double.

    Off the capturing platform they need only sit within
    :data:`OFF_PLATFORM_EDGE_ULPS` of each other, because the Cython
    computes ``gamma * (e -+ beta * k)`` with whatever contraction its
    compiler chose, and the port computes it with the contraction the
    shipped macOS build used. A budget on the *values* cannot express
    this: across the boundary one implementation returns a finite number
    and the other returns zero.
    """
    if ON_THE_CAPTURING_PLATFORM:
        apart = ulps_between(got, want)
        assert apart == 0, (
            f"{context}: the port's support boundary is {apart} ulp(s) from "
            f"the Cython's ({got!r} vs {want!r}) on the platform the corpus "
            f"was captured on, where the port reproduces it exactly"
        )
        return
    assert_the_edge_is_within_the_off_platform_budget(got, want, context)


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

#: Enough bracketed window edges for the sweep to mean something.
MIN_EDGES_CHECKED = 100
#: The smallest miss the unfused arithmetic is recorded as producing; a
#: smaller one means :func:`integrate_reference` stopped discriminating.
MIN_RECORDED_UNFUSED_MISS = 1e-13
#: How far off a value has to be for the off-platform budget to reject it,
#: as a fraction of the peak. 100x :data:`OFF_PLATFORM_BUDGET`, and ~3e6x
#: the largest disagreement ever measured between two builds (3.4e-15).
BUDGET_PROBE_ERROR = 1e-8
#: The same for a support boundary, in ulps: 2**32 doubles is a relative
#: displacement of 9.5e-7, a million times :data:`OFF_PLATFORM_EDGE_ULPS`
#: and eight million times the 55 ulps measured between two builds --
#: while still far too small to be a physical change. Written as an
#: absolute figure, not as a multiple of the budget, so the guard fails if
#: the budget itself is ever loosened past it.
EDGE_PROBE_ERROR_ULPS = 2**32


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
    def test_gamma_matches_the_cython(self, energy: float, mass: float) -> None:
        eng = cython_boost("boost_eng")
        assert_matches_the_cython(
            boost_gamma(energy, mass),
            eng(energy, mass, 1.0, 0.0, 0.0),
            f"gamma at {energy=}, {mass=}",
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

    @pytest.mark.parametrize(("energy", "mass"), CASES)
    def test_beta_agrees_with_what_the_cython_can_be_asked_for(
        self, energy: float, mass: float
    ) -> None:
        """Cross-check through ``boost_eng``, at the precision it allows.

        ``boost_eng(ep, mp, 1, 0, 1) = gamma * (1 + beta)``, so dividing
        out ``gamma`` and subtracting 1 recovers ``beta`` -- but the
        subtraction cancels, leaving an absolute error of order ``eps``
        rather than a relative one. The tolerance says exactly that: a few
        ulp of 1, loosened by nothing else. Looser than
        :data:`OFF_PLATFORM_BUDGET` and so in force on every platform.
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
        """40,000 draws at both live product masses.

        ``m = 0`` is the photon and neutrino case; ``m = MASS_E`` is
        ``hazma/spectra/_positron/_pion.pyx``. The product energy is drawn
        within a factor of 2.5 of the line, which straddles both window
        edges at every boost.

        The support check runs before the values and on every platform:
        which draws are *zero* is structural, and no rounding difference
        explains a disagreement about it. Measured at 0 disagreements over
        these same 40,000 draws on linux/amd64.
        """
        cython = cython_boost("boost_delta_function")
        rng = np.random.default_rng(20_260_810)
        masses = [0.0, 0.510_998_928]
        got = np.empty(40_000)
        want = np.empty(40_000)
        for i in range(40_000):
            m = float(rng.choice(masses))
            beta = float(rng.uniform(1e-6, 1.0 - 1e-9))
            e0 = float(10.0 ** rng.uniform(-1.0, 3.0))
            e = float(e0 * 10.0 ** rng.uniform(-0.4, 0.4))
            got[i] = boost_delta_function(e0, e, m, beta)
            want[i] = cython(e0, e, m, beta)
        flips = int(np.count_nonzero((got == 0.0) != (want == 0.0)))
        assert flips == 0, (
            f"{flips} of 40,000 draws put the port and the Cython on "
            f"opposite sides of the window, which no rounding explains"
        )
        assert_matches_the_cython(got, want, "40,000-draw sweep")

    def test_the_window_is_where_the_cython_puts_it(self) -> None:
        """Both edges, sampled clear of the boundary on either side.

        "Clear" is with respect to :data:`OFF_PLATFORM_EDGE_ULPS`: 1e-11
        relative is about 45,000 ulps, so these samples stay on their own
        side of the boundary under any displacement this module tolerates.
        Locating the boundary itself to the last bit is
        :meth:`test_the_window_edges_sit_where_the_cython_puts_them`'s job,
        and it is expressed in ulps because a value tolerance cannot span a
        finite-versus-zero disagreement.
        """
        cython = cython_boost("boost_delta_function")
        e0, m, beta = 200.0, 0.0, 0.6
        gamma = rust_gamma(beta)
        for edge in (gamma * e0 * (1.0 - beta), gamma * e0 * (1.0 + beta)):
            for scale in (1.0 - 1e-11, 1.0 + 1e-11, 0.99, 1.01):
                e = edge * scale
                assert_matches_the_cython(
                    boost_delta_function(e0, e, m, beta),
                    cython(e0, e, m, beta),
                    f"window sample at {e=!r}",
                )

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

    def test_the_window_edges_sit_where_the_cython_puts_them(self) -> None:
        """Both edges of the support, located to the last bit, 400 times.

        ``eminus`` and ``eplus`` never appear in the returned value --
        they only decide whether it is ``1/(2 gamma beta k0)`` or zero.
        So a one-ulp change in how they are computed is invisible to any
        test that samples ``e`` on a grid: it moves the edge by a single
        double, and no random draw lands there. Bisecting on the bit
        pattern does land there, in each implementation separately, and
        the two boundaries are then compared as a distance.

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
            gamma = rust_gamma(beta)
            span = 4.0 * gamma * (1.0 + beta)
            line = (e0, m, beta)
            for bracket in ((max(m, e0 / span), e0), (e0, e0 * span)):
                want = self._support_edge(cython, line, bracket)
                got = self._support_edge(boost_delta_function, line, bracket)
                if want is None or got is None:
                    continue
                assert_the_support_edge_matches_the_cython(
                    got, want, f"edge at {e0=!r}, {m=!r}, {beta=!r}"
                )
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

    Every case asserts agreement with the Cython *and* pins the branch
    that fired -- either by a closed form the branch alone can produce, or
    by a sensitivity check on a table entry only that branch reads.
    """

    def test_the_whole_window_above_the_table_is_zero(self) -> None:
        cython = cython_boost("boost_integrate_linear_interp")
        energy, beta = 1e6, 0.5
        got = boost_integrate_linear_interp(energy, beta, FLAT_X, FLAT_Y)
        assert got == 0.0
        assert_matches_the_cython(got, cython(energy, beta, FLAT_X, FLAT_Y), "clamp")

    def test_the_whole_window_below_the_table_is_the_analytic_tail(self) -> None:
        """``y0 * x0 / E``, a closed form no other branch can produce."""
        cython = cython_boost("boost_integrate_linear_interp")
        energy, beta = 1e-6, 0.5
        got = boost_integrate_linear_interp(energy, beta, FLAT_X, FLAT_Y)
        assert got == FLAT_Y[0] * FLAT_X[0] / energy
        assert_matches_the_cython(got, cython(energy, beta, FLAT_X, FLAT_Y), "tail")

    def test_a_window_straddling_the_table_floor_adds_the_tail(self) -> None:
        """`lb` below the table, `ub` inside it.

        Pinned by sensitivity: only the tail term reads ``y[0]`` when
        ``ilow`` is 0, and only through the ``y0 * (1 - rat) / rat``
        factor, so scaling ``y[0]`` must scale the tail's contribution.
        """
        cython = cython_boost("boost_integrate_linear_interp")
        energy, beta = 1.4, 0.6
        base = boost_integrate_linear_interp(energy, beta, FLAT_X, FLAT_Y)
        bumped = FLAT_Y.copy()
        bumped[0] *= 2.0
        moved = boost_integrate_linear_interp(energy, beta, FLAT_X, bumped)
        assert moved != base
        assert_matches_the_cython(
            base, cython(energy, beta, FLAT_X, FLAT_Y), "tail base"
        )
        assert_matches_the_cython(
            moved, cython(energy, beta, FLAT_X, bumped), "tail bumped"
        )

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
        assert_matches_the_cython(got, cython(energy, beta, FLAT_X, FLAT_Y), "ceiling")

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
        bumped = FLAT_Y.copy()
        bumped[index] += 1.0
        moved = boost_integrate_linear_interp(energy, beta, FLAT_X, bumped)
        assert moved != base
        assert_matches_the_cython(
            base, cython(energy, beta, FLAT_X, FLAT_Y), "edge base"
        )
        assert_matches_the_cython(
            moved, cython(energy, beta, FLAT_X, bumped), "edge bumped"
        )

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
        assert_matches_the_cython(
            moved, cython(energy, beta, FLAT_X, bumped), "interior sum"
        )

    @pytest.mark.parametrize("name", list(photon_tables()))
    def test_matches_the_cython_on_the_live_tables(self, name: str) -> None:
        """The seven shipped tables, six boost regimes, 400 energies each.

        The sweep Phase 04's swap is graded on, and the measurement
        :data:`OFF_PLATFORM_BUDGET` was set from. The zeros are compared
        first and everywhere, for the same reason as in the delta-function
        sweep: where the spectrum vanishes is structural.
        """
        cython = cython_boost("boost_integrate_linear_interp")
        x, y, mass = photon_tables()[name]
        energies = np.geomspace(1e-3, 1e4, 400)
        for multiple in BOOST_REGIMES:
            beta = boost_beta(mass * multiple, mass)
            got = boost_integrate_linear_interp(energies, beta, x, y)
            want = np.array([cython(float(e), beta, x, y) for e in energies])
            assert np.array_equal(got == 0.0, want == 0.0), (
                f"{name} at {multiple}x rest: the port and the Cython "
                f"disagree about where the boosted spectrum vanishes"
            )
            assert_matches_the_cython(got, want, f"{name} at {multiple}x rest")


class TestOffPlatformBudgets:
    """The two off-platform budgets are not vacuous.

    Asserted where the budgets are *not* used: on the capturing platform
    every comparison above takes its exact branch, so nothing else would
    exercise either tolerance and both could rot to ``inf`` unnoticed.
    That is the failure mode Task 4.1 recorded -- "the capturing platform
    cannot see a bug in its own skip logic" -- one level down.
    """

    def test_the_value_budget_rejects_a_real_error(self) -> None:
        """A perturbation of :data:`BUDGET_PROBE_ERROR` of the peak must fail.

        Six orders of magnitude above the largest rounding difference
        measured between two builds (3.4e-15 of the peak), and still far
        too small to see in a plot.
        """
        x, y, mass = photon_tables()["eta"]
        beta = boost_beta(mass * 2.0, mass)
        energies = np.geomspace(1e-3, 1e4, 400)
        want = boost_integrate_linear_interp(energies, beta, x, y)
        nudged = want.copy()
        nudged[nudged.argmax()] += BUDGET_PROBE_ERROR * want.max()

        assert_within_the_off_platform_budget(want, want, "unperturbed")
        with pytest.raises(AssertionError):
            assert_within_the_off_platform_budget(nudged, want, "perturbed")

    def test_the_edge_budget_rejects_a_real_error(self) -> None:
        """An edge displaced past the ulp budget must fail.

        Called through the budget predicate rather than through
        :func:`assert_the_support_edge_matches_the_cython`, so it exercises
        the tolerance on every platform -- going through the dispatcher
        would take the exact branch here and prove nothing about
        :data:`OFF_PLATFORM_EDGE_ULPS`, which is the whole point.
        """

        def displaced(ulps: int) -> float:
            moved = int(np.float64(300.0).view(np.int64)) + ulps
            return float(np.int64(moved).view(np.float64))

        edge = 300.0
        at_the_budget = displaced(OFF_PLATFORM_EDGE_ULPS)
        a_real_error = displaced(EDGE_PROBE_ERROR_ULPS)

        assert ulps_between(at_the_budget, edge) == OFF_PLATFORM_EDGE_ULPS
        assert_the_edge_is_within_the_off_platform_budget(edge, edge, "identical")
        # The budget's own boundary is inclusive.
        assert_the_edge_is_within_the_off_platform_budget(
            at_the_budget, edge, "at the budget"
        )
        with pytest.raises(AssertionError):
            assert_the_edge_is_within_the_off_platform_budget(
                a_real_error, edge, "displaced by a real error"
            )

    def test_this_platform_gets_the_mode_it_is_supposed_to(self) -> None:
        """One ulp: rejected where the corpus was captured, tolerated off it.

        The guard on the *strict* branch, and on the dispatch into it,
        which the two above do not cover. The failure mode is silent in
        one direction: an :data:`ON_THE_CAPTURING_PLATFORM` that had
        rotted to ``False`` would route every comparison in this module
        through the budget and every one of them would still pass. So the
        expected mode is re-derived here from ``sys``/``platform`` rather
        than read back out of the module -- reading it back would agree
        with the dispatcher by construction and assert nothing.
        """
        expected_strict = (
            sys.platform == "darwin" and platform.machine() == CAPTURE_MACHINE
        )
        value = 1.0
        nudged = float(np.nextafter(value, np.inf))
        assert_matches_the_cython(value, value, "identical")
        if expected_strict:
            with pytest.raises(AssertionError):
                assert_matches_the_cython(nudged, value, "one ulp")
        else:
            assert_matches_the_cython(nudged, value, "one ulp")


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

        The Cython covers ``[1.1, 2]`` (lower partial cell) and ``[2, 3]``
        (the interior sum) for ``1.9 / (2 gamma beta)``. Covering
        ``[1.1, 4]`` as intended would give ``2.9`` -- a 53% difference,
        far too large to be roundoff, and the scale
        :data:`OFF_PLATFORM_BUDGET` is set nine orders of magnitude below.
        """
        cython = cython_boost("boost_integrate_linear_interp")
        got = boost_integrate_linear_interp(2.2, 0.6, self.X, self.Y)
        assert got == pytest.approx(1.9 / 1.5, rel=1e-15)
        assert got != pytest.approx(2.9 / 1.5, rel=1e-3)
        assert_matches_the_cython(got, cython(2.2, 0.6, self.X, self.Y), "dropped cell")

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
        assert cython(2.2, 0.6, self.X, spoiled) == cython(2.2, 0.6, self.X, self.Y)


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
