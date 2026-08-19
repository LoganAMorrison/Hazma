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
2. :class:`TestAgainstTheCythonTwin` -- the ``cdef``
   ``dnde_positron_muon_point`` that is still exported through
   ``hazma/spectra/_positron/_muon.pyx``'s ``__pyx_capi__``, which is the
   strongest available oracle. It is compared **bit-for-bit on the platform
   the parity corpus was captured on, and within a measured budget
   everywhere else** -- see below. It dies with the ``.pyx`` in Phase 06
   Task 6.4.
3. :class:`TestPhysics` -- statements about the spectrum that owe nothing to
   the implementation being replaced: thresholds, support, the normalization,
   and the boost's conservation of positron number. These outlive the Cython.

Why the comparison has two modes
--------------------------------
Bit-equality against a *compiled* twin is a statement about the build that
produced it, not about the port. The first version of this module tried to
detect that condition instead of declaring it: it compared the compiled
kernel against an unfused Python transcription and skipped where the two
agreed, on the theory that a build which does not contract its
multiply-adds is simply a different arithmetic. CI refuted it twice. On
Linux/x86-64 -- with no ``-march`` flag, so none of the hardware FMA the
probe went looking for -- the compiled kernel diverges from a faithful
unfused reference anyway, and the mechanism was never localized. It does
not need to be: a compiler contracting a different set of expressions, or
a libm rounding one call differently, breaks bit-equality just as
thoroughly, and no probe over one mechanism can see the others.

So the *mode* is declared from the platform, and the divergence off it was
**measured rather than assumed** (PR #63, run 31564747071; Linux/glibc,
py3.10-3.14, 21,953 differing values decoded from the failure output):

======================= ==================== ====================
``emu`` / MeV           max relative         max ``|Δ|`` / peak
======================= ==================== ====================
105.6583745 (``m_mu``)  4.2e-16              3.7e-16
105.658374501           6.0e-11              1.9e-11
110                     2.7e-14              7.8e-16
150                     3.7e-13              5.7e-16
500                     6.4e-12              3.6e-15
1500                    2.2e-11              3.0e-14
100000                  1.5e-07              1.3e-10
======================= ==================== ====================

Median relative difference 7.2e-15; no sign flip, no NaN, no disagreement
about support or zeros. This is rounding amplified by the kernel's own
conditioning: both bad regimes -- ``beta -> 0`` just off rest, and
``gamma >> 1`` -- form ``xm``/``xp`` as ``gamma**2 * (x -+ beta * root)``
and then difference nearly-equal terms. The amplification belongs to the
*formula*; two Cython builds would show the same spread.

Which is why the off-platform budget is scaled to the **peak of the
spectrum** and not applied pointwise. Pointwise, the worst case is 1.5e-7
-- but it sits at a value 4.3e-4 of the peak, and a pointwise ``rtol`` loose
enough to admit it (>=1e-6) would be loose enough to hide a real defect.
Against the peak -- which is what a downstream integral or limit actually
sees -- the worst disagreement anywhere is 1.3e-10, and 1.9e-11 within the
sub-GeV domain this library is for. A wrong branch, a dropped term or a bad
constant lands at O(1) against that, so the budget below still fails on
anything structural.

The parity corpus (``test/parity/``) draws the same line in the same
place: its ``EXACT`` budget class is bit-equality on the capturing
platform and :data:`tolerances.PLATFORM_EXACT_RTOL` once the libm
changes. :data:`CAPTURE_MACHINE` is read out of the corpus manifest so
the two scopes cannot drift apart. It remains the gate that governs the
swap, holding this entry point to ``rtol = 0`` against 179,695 pinned
pre-port values; nothing here duplicates that.

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
import json
import math
import platform
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from hazma import spectra
from hazma._core import positron as core_positron
from hazma.spectra import _positron as wrapper
from hazma.spectra._positron import _muon as cython_module

if TYPE_CHECKING:
    from collections.abc import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]

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


#: The platform the parity corpus was captured on, read from its own
#: manifest so the two can never drift apart. `test/parity` demands
#: bit-equality of its `EXACT` class only on this platform, for exactly
#: the reason below; this module is the same kind of oracle and carries
#: the same scope.
CAPTURE_MACHINE = json.loads(
    (REPO_ROOT / "test" / "parity" / "data" / "manifest.json").read_text()
)["environment"]["machine"]

ON_THE_CAPTURING_PLATFORM = (
    sys.platform == "darwin" and platform.machine() == CAPTURE_MACHINE
)

#: The off-platform budget, as a fraction of the peak of the spectrum being
#: compared. Replaying every Linux array recovered from run 31564747071
#: through the assertion below clears it by **84x at the tightest** (both
#: `emu = 1e5` blocks; everything within the sub-GeV domain clears by 1e5x
#: or more). That headroom is deliberate -- a libm this port has not met
#: yet should not turn the suite red for rounding -- and it is still seven
#: orders of magnitude tighter than any physically meaningful change.
#: Applied as `assert_allclose`'s `atol`, with `rtol` at the same figure so
#: a large value is held to the same standard.
OFF_PLATFORM_BUDGET = 1e-8


def assert_within_the_off_platform_budget(
    got: np.ndarray, want: np.ndarray, context: str
) -> None:
    """Assert two spectra agree to :data:`OFF_PLATFORM_BUDGET` of the peak.

    Split out from :func:`assert_matches_the_cython` so the budget can be
    exercised on *every* platform, including the one where the caller would
    otherwise take the bit-equality branch and leave this untested --
    :func:`test_the_off_platform_budget_rejects_a_real_error`.

    ``atol`` is scaled by the peak rather than left at zero because the
    kernel's relative error is unbounded where it cancels: see "Why the
    comparison has two modes".
    """
    finite = np.isfinite(want)
    peak = float(np.abs(want[finite]).max()) if finite.any() else 0.0
    np.testing.assert_allclose(
        got,
        want,
        rtol=OFF_PLATFORM_BUDGET,
        atol=OFF_PLATFORM_BUDGET * peak,
        err_msg=(
            f"{context}: the port left the Cython's budget of "
            f"{OFF_PLATFORM_BUDGET:.0e} x the spectrum peak ({peak:.6e}). "
            f"Rounding between two builds was measured at 1.3e-10 x peak, so "
            f"this is a defect, not a platform difference."
        ),
    )


def assert_matches_the_cython(got: np.ndarray, want: np.ndarray, context: str) -> None:
    """The oracle, in whichever of its two modes this platform gets.

    Bit-for-bit where the corpus was captured -- the port was written
    against *this* build's arithmetic and reproduces it exactly, which is a
    far stronger statement than any tolerance. A budget elsewhere, because
    off it the comparison measures the C library rather than the port.
    """
    if ON_THE_CAPTURING_PLATFORM:
        assert got.tobytes() == want.tobytes(), (
            f"{context}: not bit-equal to the Cython on the platform the "
            f"corpus was captured on, where the port is written to reproduce "
            f"it exactly"
        )
        return
    assert_within_the_off_platform_budget(got, want, context)


def cython_spectrum(emu: float, energies: np.ndarray) -> np.ndarray:
    """The Cython twin evaluated pointwise over ``energies``."""
    point = cython_point()
    return np.array([point(float(e), emu) for e in energies])


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


class TestAgainstTheCythonTwin:
    """The ``cdef`` the swap left behind, as an oracle.

    The parity corpus pins 179,695 values at the grids it chose; this reaches
    the same kernel at arbitrary arguments, which is what lets the edges be
    probed directly. Bit-for-bit on the capturing platform and within
    :data:`OFF_PLATFORM_BUDGET` elsewhere — "Why the comparison has two
    modes" in the module docstring derives both.
    """

    @pytest.mark.parametrize("emu", MUON_ENERGIES)
    def test_a_swept_grid_matches(self, emu: float) -> None:
        energies = np.geomspace(MASS_E * 0.5, emu * 1.5, 2001)
        assert_matches_the_cython(
            dnde(energies, emu), cython_spectrum(emu, energies), f"swept grid, {emu=}"
        )

    @pytest.mark.parametrize("emu", MUON_ENERGIES)
    def test_random_arguments_match(self, emu: float) -> None:
        rng = np.random.default_rng(4)
        energies = rng.uniform(0.0, emu * 1.1, 4000)
        assert_matches_the_cython(
            dnde(energies, emu),
            cython_spectrum(emu, energies),
            f"random arguments, {emu=}",
        )

    def test_the_kinematic_edges_match(self) -> None:
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
            assert_matches_the_cython(
                dnde(edges, emu),
                cython_spectrum(emu, edges),
                f"kinematic edges, {emu=}",
            )

    def test_the_support_is_identical_everywhere(self) -> None:
        """Which energies are *zero* is structural, so it holds on any build.

        The budget above is a statement about rounding; this is the statement
        rounding cannot excuse. A port that moved a threshold or a kinematic
        limit by one grid point turns this red on every platform, including
        the ones where the tolerance branch is in force.
        """
        for emu in MUON_ENERGIES:
            energies = np.geomspace(MASS_E * 0.5, emu * 1.5, 2001)
            got, want = dnde(energies, emu), cython_spectrum(emu, energies)
            assert np.array_equal(got == 0.0, want == 0.0), (
                f"the port and the Cython disagree about where the spectrum "
                f"vanishes at {emu=}, which no rounding difference explains"
            )

    def test_the_off_platform_budget_rejects_a_real_error(self) -> None:
        """The budget is not vacuous, asserted where the budget is not used.

        On the capturing platform :func:`assert_matches_the_cython` takes its
        bit-equality branch, so nothing else here would exercise the
        tolerance at all and it could rot to `inf` unnoticed. A perturbation
        of 1e-6 of the peak — four orders of magnitude above the largest
        rounding difference ever measured between two builds, and still far
        too small to see in a plot — must be rejected.
        """
        energies = np.geomspace(MASS_E * 0.5, 750.0, 2001)
        want = cython_spectrum(500.0, energies)
        nudged = want.copy()
        nudged[nudged.argmax()] += 1e-6 * want.max()

        assert_within_the_off_platform_budget(want, want, "unperturbed")
        with pytest.raises(AssertionError):
            assert_within_the_off_platform_budget(nudged, want, "perturbed")

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
        assert_matches_the_cython(
            np.array([from_rust]),
            np.array([point(float("nan"), 500.0)]),
            "a NaN energy in the boosted branch",
        )

        # The rest-frame branch has no fmax/fmin, so there a NaN does survive.
        assert math.isnan(dnde(float("nan"), MASS_MU))
        assert math.isnan(point(float("nan"), MASS_MU))

    def test_a_below_threshold_muon_is_zero_in_both(self) -> None:
        point = cython_point()
        assert dnde(10.0, MASS_MU * 0.999_999) == 0.0
        assert point(10.0, MASS_MU * 0.999_999) == 0.0


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
