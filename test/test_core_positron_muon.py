"""``hazma._core.positron.dnde_positron_muon`` — the first ported kernel.

cython-to-rust Phase 04 Task 4.1. This module is the **per-kernel template**
Tasks 4.2-4.6 and Phases 05-06 copy, and it is deliberately *not* a copy of
``test/test_core_dispatch.py``.

Why not a copy
--------------
Task 2.3 wrote ``test/test_core_dispatch.py`` when the plan was one dispatch
implementation per kernel, and its own docstring says to copy every test.
Task 3.5 then replaced that with three shared helpers -- ``map_unary``,
``map_flavors``, ``require_vector`` -- so the 118 branch tests there now cover
code that *every* kernel routes through unchanged. Transcribing them sixteen
more times would re-test one function sixteen times and leave sixteen copies
to keep in sync.

What is genuinely per-kernel is which helper the wrapper reached for, which
quantity wording it passed, and whether the numbers are right. So this module
has three parts, and a later swap copies its *shape*:

1. :class:`TestDispatchWiring` -- one assertion per contract branch, enough to
   prove this entry point goes through ``map_unary`` with the wording its
   Cython twin used. Branch-by-branch reasoning about the helper itself stays
   in ``test/test_core_dispatch.py``.
2. :class:`TestAgainstTheCythonTwin` -- bit-equality against the ``cdef``
   ``dnde_positron_muon_point`` that is still exported through
   ``hazma/spectra/_positron/_muon.pyx``'s ``__pyx_capi__``. This is the
   strongest available oracle and, like ``test/test_core_boost.py``'s, it is
   **scoped to a platform whose C compiler contracts multiply-adds** -- the
   Task 3.4 lesson ``[platform-scoped-oracle-asserted-globally]``. It dies
   with the ``.pyx`` in Phase 06 Task 6.4.
3. :class:`TestPhysics` -- statements about the spectrum that owe nothing to
   the implementation being replaced: thresholds, support, the normalization,
   and the boost's conservation of positron number. These outlive the Cython.

The parity corpus (``test/parity/``) is still the gate that governs the swap;
it holds this entry point to ``rtol = 0`` against 179,695 pinned pre-port
values. Nothing here duplicates that.

The normalization defect
------------------------
:class:`TestPhysics` asserts the spectrum integrates to ``1/N**2``, not to 1.
The shipped Cython divides by the Michel normalization where it should
multiply, so every value is low by 0.0374%. That is a live defect in hazma
2.1.0 which the port reproduces on purpose
(``projects/cython-to-rust/rules.md`` rule 1) and which
``docs/followups/todo/positron-muon-spectrum-normalization-inverted.md``
tracks. Asserting the correct normalization here would contradict the corpus.
"""

from __future__ import annotations

import ctypes
import math
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest

from hazma import spectra
from hazma._core import positron as core_positron
from hazma.spectra import _positron as wrapper
from hazma.spectra._positron import _muon as cython_module

if TYPE_CHECKING:
    from collections.abc import Callable

dnde = core_positron.dnde_positron_muon

QUANTITY = "Positron energies"
DIMENSION_ERROR = f"{QUANTITY} must be 0 or 1-dimensional."
TYPE_ERROR = f"{QUANTITY} must be a float or a NumPy array."

#: `hazma/_utils/constants.pxd`, which is the table this kernel's `.pyx`
#: `include`s. Spelled out rather than imported from `hazma.parameters` so a
#: future consolidation of the two tables cannot silently move the tests with
#: the code (`projects/cython-to-rust/rules.md` rule 4).
MASS_E = 0.5109989461
MASS_MU = 105.6583745
R = MASS_E / MASS_MU
R_FACTOR = 1.0001870858234163

#: The signature string that is also the capsule's *name*, so a changed
#: `cdef` prototype fails loudly rather than being called through the wrong
#: ABI (the Task 3.4 constraint).
_POINT_SIGNATURE = b"double (double, double)"

#: The Cython's ``DBL_EPSILON``, which is the same double Rust spells
#: ``f64::EPSILON``. Named because :func:`unfused_point` has to reproduce the
#: near-rest branch guard exactly.
DBL_EPSILON = sys.float_info.epsilon

#: How far below 1 the shipped normalization sits: ``1 - 1/R_FACTOR**2``, or
#: 3.74e-4. Named so the assertion that the integral is *not* 1 states the
#: separation it relies on rather than a bare literal.
NORMALIZATION_DEFICIT = 1.0 - 1.0 / R_FACTOR**2

#: Muon energies spanning rest, just-off-rest, and increasing boosts.
MUON_ENERGIES = (MASS_MU, MASS_MU + 1e-9, 110.0, 150.0, 500.0, 1500.0, 1e5)


def cython_point() -> Callable[[float, float], float]:
    """The live Cython ``dnde_positron_muon_point``, callable from Python.

    ``PYFUNCTYPE``, never ``CFUNCTYPE``: the latter releases the GIL, and
    anything that calls back into Python then segfaults with no Python-level
    error (``test/test_core_boost.py`` documents the same constraint).
    """
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.restype = ctypes.c_void_p
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

    capsule = cython_module.__pyx_capi__["dnde_positron_muon_point"]
    address = get_pointer(capsule, _POINT_SIGNATURE)
    return ctypes.PYFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)(address)


def unfused_point(e: float, emu: float) -> float:
    """``dnde_positron_muon`` with no multiply-add contraction anywhere.

    Used only to decide whether the local Cython build fuses; the port's own
    arithmetic is in ``rust/src/kernels/positron_muon.rs``.

    **Every association here is the Cython's**, not merely every operation.
    ``pre * (numerator / denominator)`` and ``pre * numerator / denominator``
    are different doubles, and writing the second is what made this probe
    report a *contracting* build on Linux, where nothing contracts: the
    reference then differed from the Cython for a reason that was not
    contraction, `cython_contracts` said True, and every assertion this guard
    exists to skip ran and failed. `test_the_reference_is_the_cython_where_
    nothing_contracts` is the assertion that now catches that directly.
    """
    r2 = R * R
    two_r, one_plus_r2 = 2.0 * R, 1.0 + r2

    def rest_frame(x: float) -> float:
        if x <= two_r or x >= one_plus_r2:
            return 0.0
        root = math.sqrt(x * x - 4.0 * r2)
        return -2.0 * root * (4.0 * r2 + x * (-3.0 - 3.0 * r2 + 2.0 * x)) / R_FACTOR

    if emu < MASS_MU or e <= MASS_E:
        return 0.0
    if emu - MASS_MU < DBL_EPSILON:
        pre = 2.0 / MASS_MU
        return pre * rest_frame(pre * e)

    beta = math.sqrt(1.0 - (MASS_MU / emu) ** 2)
    pre = 2.0 / emu
    x = pre * e
    if beta < 0.0 or beta > 1.0:
        return 0.0
    gamma2 = 1.0 / (1.0 - beta**2)
    r22 = 4.0 * r2 * (1.0 - beta**2)
    root = math.sqrt(x * x - r22)
    xm = max(gamma2 * (x - beta * root), two_r)
    xp = min(gamma2 * (x + beta * root), one_plus_r2)
    if xm > xp:
        return 0.0
    numerator = xm * (8.0 * r2 + xm * (-3.0 - 3.0 * r2 + (4.0 * xm) / 3.0)) + xp * (
        -8.0 * r2 + (3.0 + 3.0 * r2 - (4.0 * xp) / 3.0) * xp
    )
    # The Cython divides inside `dndx_positron_muon` and multiplies by `pre`
    # in its caller, so the division completes first. Folding the two into
    # one expression moves the last bit.
    dndx = numerator / (2.0 * beta * R_FACTOR)
    return pre * dndx


def cython_contracts() -> bool:
    """Whether this build's Cython fuses its multiply-adds.

    Drawn until the two forms are distinguishable rather than decided at one
    point: at most arguments both roundings agree.
    """
    point = cython_point()
    rng = np.random.default_rng(0)
    for _ in range(4096):
        emu = float(MASS_MU * 10.0 ** rng.uniform(0.001, 2.0))
        e = float(emu * rng.uniform(0.01, 0.99))
        if point(e, emu) != unfused_point(e, emu):
            return True
    return False


CYTHON_CONTRACTS = cython_contracts()

requires_a_contracting_cython = pytest.mark.skipif(
    not CYTHON_CONTRACTS,
    reason=(
        "this Cython build does not fuse its multiply-adds, so it computes "
        "different values than the macOS/arm64 build this port targets; the "
        "bit-for-bit comparison is scoped to a contracting platform exactly "
        "as the parity corpus is"
    ),
)


def test_the_reference_is_the_cython_where_nothing_contracts() -> None:
    """`unfused_point` differs from the Cython *only* by contraction.

    The other half of :data:`CYTHON_CONTRACTS`, and the one that catches a
    mistake in the reference rather than in the port. Where the guard says
    this build does not contract, the two must agree **bit for bit** at
    every argument -- so any other divergence in `unfused_point` (a moved
    association, a constant off by an ulp) turns this red instead of
    silently flipping the guard to True and un-skipping
    :class:`TestAgainstTheCythonTwin`.

    Vacuous on a contracting platform, which is why it is a plain
    ``skipif`` rather than the negation of the class-level marker: there
    the guard's *other* direction is what
    :meth:`TestAgainstTheCythonTwin.test_the_unfused_form_actually_differs_somewhere`
    checks. Between them the probe is pinned in both directions.
    """
    if CYTHON_CONTRACTS:
        pytest.skip(
            "this build contracts, so the unfused reference is expected to "
            "differ; the contracting direction is checked inside "
            "TestAgainstTheCythonTwin"
        )

    point = cython_point()
    rng = np.random.default_rng(7)
    for emu in (MASS_MU, *MUON_ENERGIES[1:]):
        for e in rng.uniform(0.0, emu * 1.1, 500):
            got, want = point(float(e), emu), unfused_point(float(e), emu)
            assert got == want, (
                f"at e={e!r}, emu={emu!r} the unfused reference gives {want!r} "
                f"and the Cython {got!r}, on a build the probe says does not "
                "contract — so the reference has a bug that is not about FMA"
            )


class TestDispatchWiring:
    """The entry point goes through ``map_unary`` with its own wording.

    One assertion per contract branch. The branch-by-branch argument about
    ``map_unary`` itself is ``test/test_core_dispatch.py``'s; what is specific
    to this kernel is that it reached that helper at all, and with the
    quantity string its Cython twin's ``assert`` carried.
    """

    def test_a_scalar_returns_a_python_float(self) -> None:
        value = dnde(10.0, 500.0)
        assert type(value) is float
        assert value > 0.0

    def test_a_numpy_scalar_and_a_zero_dim_array_take_the_scalar_path(self) -> None:
        expected = dnde(10.0, 500.0)
        assert dnde(np.float64(10.0), 500.0) == expected
        assert dnde(np.float32(10.0), 500.0) == dnde(float(np.float32(10.0)), 500.0)
        assert dnde(np.array(10.0), 500.0) == expected
        assert type(dnde(np.array(10.0), 500.0)) is float

    def test_an_array_returns_a_fresh_float64_array_of_the_same_length(self) -> None:
        energies = np.geomspace(1.0, 400.0, 64)
        values = dnde(energies, 500.0)
        assert values.dtype == np.float64
        assert values.shape == energies.shape
        assert not np.shares_memory(values, energies)

    def test_the_array_path_agrees_with_the_scalar_path_bit_for_bit(self) -> None:
        # The property the two paths must share: `map_unary` calls the same
        # kernel either way, so a broadcasting bug shows up here and nowhere
        # else the corpus looks.
        energies = np.geomspace(0.4, 600.0, 257)
        batched = dnde(energies, 500.0)
        one_at_a_time = np.array([dnde(float(e), 500.0) for e in energies])
        assert batched.tobytes() == one_at_a_time.tobytes()

    def test_a_sequence_is_accepted(self) -> None:
        assert dnde([10.0, 20.0], 500.0).tolist() == [
            dnde(10.0, 500.0),
            dnde(20.0, 500.0),
        ]

    def test_an_empty_grid_returns_an_empty_grid(self) -> None:
        values = dnde(np.array([], dtype=np.float64), 500.0)
        assert values.shape == (0,)

    def test_a_higher_rank_array_names_this_kernel_s_quantity(self) -> None:
        with pytest.raises(ValueError, match=r"^Positron energies must be 0 or 1-"):
            dnde(np.ones((2, 2)), 500.0)

    def test_the_rank_message_is_the_cython_assert_verbatim(self) -> None:
        # The wording is user-visible API. Compared against the string as the
        # `.pyx` sources still spell it, not against a copy in this file.
        with pytest.raises(ValueError) as excinfo:
            dnde(np.ones((2, 2)), 500.0)
        assert str(excinfo.value) == DIMENSION_ERROR

    def test_a_non_float64_array_is_a_value_error(self) -> None:
        with pytest.raises(ValueError, match="must be a float64 array"):
            dnde(np.arange(4), 500.0)

    def test_a_non_number_is_a_type_error(self) -> None:
        with pytest.raises(TypeError) as excinfo:
            dnde(None, 500.0)
        assert str(excinfo.value) == TYPE_ERROR

    def test_both_arguments_are_accepted_by_keyword(self) -> None:
        # The Cython twin was a `def` and took keywords; a positional-only
        # port would be a silent public-API narrowing.
        assert dnde(positron_energies=10.0, muon_energy=500.0) == dnde(10.0, 500.0)


class TestWrapperAndPublicApi:
    """The swap is wired all the way out to what users import."""

    def test_the_private_wrapper_returns_the_core_kernel_s_values(self) -> None:
        energies = np.geomspace(1.0, 400.0, 32)
        assert wrapper.dnde_positron_muon(energies, 500.0).tobytes() == (
            dnde(energies, 500.0).tobytes()
        )

    def test_the_public_spectra_name_resolves_to_the_same_function(self) -> None:
        assert spectra.dnde_positron_muon(10.0, 500.0) == dnde(10.0, 500.0)

    def test_the_cython_module_no_longer_exports_a_python_entry_point(self) -> None:
        # rules.md rule 1, as far as the capi exception allows: the extension
        # is still built for its `cdef` capsules, but no Python caller can
        # reach the implementation the swap replaced.
        assert not hasattr(cython_module, "dnde_positron_muon")

    def test_the_cdef_capsules_the_mediator_modules_cimport_are_intact(self) -> None:
        # Phase 06 Task 6.4 deletes these; until then `_positron/_pion.pyx`
        # and both mediator positron spectrum modules cimport them, so
        # removing the `def` must not have disturbed them.
        exported = cython_module.__pyx_capi__
        assert set(exported) == {
            "dnde_positron_muon_point",
            "dnde_positron_muon_array",
        }

    def test_the_capsule_name_is_the_expected_c_signature(self) -> None:
        get_name = ctypes.pythonapi.PyCapsule_GetName
        get_name.restype = ctypes.c_char_p
        get_name.argtypes = [ctypes.py_object]
        capsule = cython_module.__pyx_capi__["dnde_positron_muon_point"]
        assert get_name(capsule) == _POINT_SIGNATURE


@requires_a_contracting_cython
class TestAgainstTheCythonTwin:
    """Bit-equality against the ``cdef`` the swap left behind.

    The parity corpus pins 179,695 values at the grids it chose; this reaches
    the same kernel at arbitrary arguments, which is what lets the edges be
    probed directly. Scoped to a contracting platform for the reason the class
    docstring of ``test/test_core_boost.py`` gives at length.
    """

    @pytest.mark.parametrize("emu", MUON_ENERGIES)
    def test_a_swept_grid_is_bit_equal(self, emu: float) -> None:
        point = cython_point()
        energies = np.geomspace(MASS_E * 0.5, emu * 1.5, 2001)
        assert dnde(energies, emu).tobytes() == (
            np.array([point(float(e), emu) for e in energies]).tobytes()
        )

    @pytest.mark.parametrize("emu", MUON_ENERGIES)
    def test_random_arguments_are_bit_equal(self, emu: float) -> None:
        point = cython_point()
        rng = np.random.default_rng(4)
        energies = rng.uniform(0.0, emu * 1.1, 4000)
        assert dnde(energies, emu).tobytes() == (
            np.array([point(float(e), emu) for e in energies]).tobytes()
        )

    def test_the_kinematic_edges_are_bit_equal(self) -> None:
        point = cython_point()
        for emu in (MASS_MU, MASS_MU * (1 + 1e-17), MASS_MU + 1e-16, 500.0, 1e9):
            edges = np.array(
                [
                    MASS_E,
                    np.nextafter(MASS_E, np.inf),
                    MASS_E * 1.0000001,
                    emu / 2.0,
                    np.nextafter(emu, 0.0),
                    emu,
                    np.nextafter(emu, np.inf),
                    0.0,
                    -1.0,
                    np.inf,
                ]
            )
            assert dnde(edges, emu).tobytes() == (
                np.array([point(float(e), emu) for e in edges]).tobytes()
            )

    def test_a_nan_energy_does_not_propagate_and_both_agree_on_that(self) -> None:
        """A ``NaN`` energy comes back as a *number*, in both implementations.

        Surprising enough to pin rather than leave to the corpus, which does
        not sample it. Neither threshold test fires on a ``NaN``, so it reaches
        the boosted branch, where ``fmax``/``fmin`` (``fmaxnm``/``fminnm``, and
        Rust's ``f64::max``/``min``) return their *non*-``NaN`` operand. Both
        limits therefore collapse onto the rest-frame support, the window
        survives, and a finite value comes out. The port reproduces it because
        Rust picked the same NaN convention, not by arrangement.
        """
        point = cython_point()
        from_rust = dnde(float("nan"), 500.0)
        assert not math.isnan(from_rust)
        assert from_rust == point(float("nan"), 500.0)

        # The rest-frame branch has no fmax/fmin, so there a NaN does survive.
        assert math.isnan(dnde(float("nan"), MASS_MU))
        assert math.isnan(point(float("nan"), MASS_MU))

    def test_a_below_threshold_muon_is_zero_in_both(self) -> None:
        point = cython_point()
        assert dnde(10.0, MASS_MU * 0.999_999) == 0.0
        assert point(10.0, MASS_MU * 0.999_999) == 0.0

    def test_the_unfused_form_actually_differs_somewhere(self) -> None:
        # Guards the guard: if `unfused_point` ever agreed with the Cython
        # everywhere, `CYTHON_CONTRACTS` would read False on a contracting
        # platform and every assertion in this class would skip silently.
        point = cython_point()
        rng = np.random.default_rng(11)
        differ = sum(
            point(e, emu) != unfused_point(e, emu)
            for emu in (150.0, 500.0, 1500.0)
            for e in rng.uniform(1.0, 400.0, 500)
        )
        assert differ > 0, (
            "the unfused reference matches the Cython everywhere, so the "
            "contraction probe cannot distinguish the two arithmetics"
        )


class TestPhysics:
    """Statements about the spectrum, not about the code it replaced.

    These are what survives Phase 06 Task 6.4 deleting the ``.pyx``.
    """

    #: The endpoint of the rest-frame spectrum in energy: ``x = 1 + r**2``
    #: scaled back by ``m_mu / 2``.
    REST_FRAME_ENDPOINT = 0.5 * MASS_MU * (1.0 + R * R)

    def test_the_spectrum_vanishes_below_both_thresholds(self) -> None:
        assert dnde(10.0, MASS_MU * 0.999) == 0.0
        assert dnde(MASS_E, 500.0) == 0.0
        assert dnde(MASS_E * 0.5, 500.0) == 0.0
        assert dnde(-1.0, 500.0) == 0.0

    def test_a_muon_at_rest_has_the_michel_endpoint(self) -> None:
        assert dnde(np.nextafter(self.REST_FRAME_ENDPOINT, np.inf), MASS_MU) == 0.0
        assert dnde(self.REST_FRAME_ENDPOINT * 0.999, MASS_MU) > 0.0

    def test_a_boosted_muon_extends_the_endpoint(self) -> None:
        # A moving parent pushes the endpoint up by roughly gamma(1 + beta).
        emu = 500.0
        assert dnde(self.REST_FRAME_ENDPOINT * 5.0, emu) > 0.0
        assert dnde(emu, emu) == 0.0

    @pytest.mark.parametrize("emu", [MASS_MU, 110.0, 500.0, 1500.0])
    def test_the_spectrum_is_non_negative_everywhere_it_is_defined(
        self, emu: float
    ) -> None:
        energies = np.geomspace(MASS_E * 1.000_001, emu, 20_001)
        values = dnde(energies, emu)
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)

    @pytest.mark.parametrize("emu", [MASS_MU, 150.0, 500.0, 1500.0])
    def test_the_integral_is_the_shipped_inverted_normalization(
        self, emu: float
    ) -> None:
        """``int dN/dE dE = 1/N**2`` at every parent energy.

        Two statements at once. That the integral is the *same* at every
        ``emu`` is the physics -- the boost moves positrons around in energy
        but creates none -- and it is what a wrong Jacobian or a wrong
        ``x``-scaling would break. That the shared value is ``1/N**2``
        rather than 1 is the shipped normalization defect this port
        reproduces (see the module docstring).

        Trapezoid on 200_001 points. 5e-5 relative: the in-flight spectrum has
        a kink where the two kinematic branches meet and a square-root edge at
        the endpoint, so a composite rule of this order gets no closer --
        measured, not chosen. The gap between ``1/N**2`` and 1 is 3.7e-4, so
        the bound still separates them by a factor of seven.
        """
        energies = np.linspace(MASS_E, emu, 200_001)
        integral = np.trapezoid(dnde(energies, emu), energies)
        shipped = 1.0 / R_FACTOR**2
        assert integral == pytest.approx(shipped, rel=5e-5)
        assert abs(integral - 1.0) > 0.9 * NORMALIZATION_DEFICIT

    def test_the_rest_frame_limit_is_continuous_in_the_parent_energy(self) -> None:
        """The ``E - m < eps`` branch is a removable singularity.

        The Cython short-circuits to the rest frame within one epsilon MeV of
        rest because the in-flight form carries a ``1/(2 beta)`` prefactor.
        The two must agree there, or the guard would be a step in a published
        spectrum rather than a numerical safeguard. 1e-6 relative: the
        cancellation in the boosted form grows as ``beta -> 0``, and at
        ``E - m = 1e-9 MeV`` (``beta ~ 1.4e-6``) that is what it reaches.
        """
        energies = np.linspace(MASS_E * 1.01, self.REST_FRAME_ENDPOINT * 0.99, 501)
        at_rest = dnde(energies, MASS_MU)
        just_moving = dnde(energies, MASS_MU + 1e-9)
        np.testing.assert_allclose(just_moving, at_rest, rtol=1e-6)
