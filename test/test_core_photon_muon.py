"""``hazma._core.photon.dnde_photon_muon`` -- the radiative muon spectrum.

cython-to-rust Phase 04 Task 4.3. Shaped after
``test/test_core_positron_muon.py``, which is the per-kernel template, and
deliberately not a copy of ``test/test_core_dispatch.py`` -- the 118 branch
tests there cover the three shared dispatch helpers every kernel routes
through unchanged. What is per-kernel is which helper the wrapper reached
for, which quantity wording it passed, and whether the numbers are right.

Three parts, same as the template:

1. :class:`TestDispatchWiring` -- one assertion per contract branch.
2. :class:`TestAgainstTheCythonTwin` -- the ``cdef``
   ``dnde_photon_muon_point`` still exported through
   ``hazma/spectra/_photon/_muon.pyx``'s ``__pyx_capi__``, which is the
   strongest available oracle. Bit-for-bit on the platform the parity
   corpus was captured on, within a peak-scaled budget elsewhere. It dies
   with the ``.pyx`` in Phase 06 Task 6.4.
3. :class:`TestPhysics` -- statements about the spectrum that outlive the
   Cython.

Why this kernel needed ``spence`` transcribed
---------------------------------------------
This is the project's first ``SPECFUN``-class swap, and it is the reason
``rust/src/special.rs`` stopped calling ``spec_math::Polylog::li2``. The
closed form carries ``(5/beta) * (spence(xm) - spence(xp))``, and the
parity corpus samples ``E_mu = m_mu * (1 + 1e-12)``, i.e. ``beta = 1.4e-6``:
the ``1/beta`` turns a two-ulp difference in ``spence`` into a **3.2e-11**
relative difference in the spectrum, 320x the ``SPECFUN`` budget of 1e-13.
Measured, not estimated -- every one of the 24 differing corpus points was
reproduced to a ratio of 1.000 by ``(5/beta) * delta_spence * alpha /
(3 pi E_mu)`` alone, which is also the evidence that the rest of the port
was already bit-equal.

The fix was to transcribe cephes ``spence`` with the contraction scipy's C
build uses, which makes it bit-identical to ``scipy.special.spence`` (0
mismatches in 13,000 points across all four branches). With that in place
the whole kernel is bit-equal to the Cython twin at **144,000** sampled
points, so the ``SPECFUN`` budget goes unused on the capturing platform --
kept rather than tightened, for the same reason Task 4.2 kept
``TABULATED``: it is the right contract for a platform where scipy's cephes
is compiled without contraction, which is where it will be needed.

Why the comparison still has two modes
--------------------------------------
Bit-equality against a *compiled* twin is a statement about the build that
produced it. ``test/test_core_positron_muon.py``'s docstring derives this
at length and CI refuted the alternative twice; the short version is that a
compiler contracting a different set of expressions, or a libm rounding one
call differently, breaks the comparison just as thoroughly as a bad port
and no probe over one mechanism sees the others. So the mode is declared
from the platform, read out of ``test/parity/data/manifest.json`` -- the
same mechanism ``test/parity`` and ``ci.yml`` use.

Here the platform scope has a second reason on top of that one. scipy's
cephes is *itself* contracted only where the C toolchain has an FMA to
contract into, so off macOS/arm64 ``spence`` is expected to differ by a few
ulp again, and the amplification above turns that into a relative
difference no pointwise ``rtol`` should admit. Which is why the
off-platform budget is scaled to the **peak of the spectrum**: against what
a downstream integral or limit actually sees, the whole effect is bounded
by 2.2e-13 absolute against a peak of order 10.

The endpoint defect
-------------------
:class:`TestPhysics` asserts the rest-frame branch returns zero over the
top 0.2543 MeV of the spectrum's support, where the spectrum is not zero.
That is a live defect in hazma 2.1.0 which the port reproduces on purpose
(``projects/cython-to-rust/rules.md`` rule 1) and which
``docs/followups/todo/photon-muon-rest-frame-endpoint-uses-the-wrong-power-of-r.md``
tracks. Asserting the correct endpoint here would contradict the corpus.
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
from hazma._core import photon as core_photon
from hazma.spectra import _photon as wrapper
from hazma.spectra._photon import _muon as cython_module

if TYPE_CHECKING:
    from collections.abc import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]

dnde = core_photon.dnde_photon_muon

QUANTITY = "Photon energies"
DIMENSION_ERROR = f"{QUANTITY} must be 0 or 1-dimensional."
TYPE_ERROR = f"{QUANTITY} must be a float or a NumPy array."

#: `hazma/_utils/constants.pxd`, which is the table this kernel's `.pyx`
#: `include`s. Spelled out rather than imported from `hazma.parameters` so a
#: future consolidation of the two tables cannot silently move the tests
#: with the code (`projects/cython-to-rust/rules.md` rule 4).
MASS_E = 0.5109989461
MASS_MU = 105.6583745
R = (MASS_E / MASS_MU) ** 2

#: Where the shipped rest-frame branch stops, in MeV: `y = 1 - m_e/m_mu`.
SHIPPED_REST_FRAME_CUT = 0.5 * MASS_MU * (1.0 - MASS_E / MASS_MU)

#: Where it should stop, in MeV: `(m_mu**2 - m_e**2) / (2 m_mu)`, i.e.
#: `y = 1 - r`. This is the edge the *boosted* branch uses, and the value
#: `hazma/spectra/_photon/_pion.pyx:16` hard-codes as `ENG_GAM_MAX_MURF`
#: (52.82795006985128, from the legacy mass table -- 1.5e-6 MeV below the
#: figure here, which uses the PDG one).
TRUE_REST_FRAME_ENDPOINT = 0.5 * MASS_MU * (1.0 - R)

#: The size of the gap between them, MeV. Named so the assertions state
#: the separation they rely on rather than a bare literal.
ENDPOINT_GAP = TRUE_REST_FRAME_ENDPOINT - SHIPPED_REST_FRAME_CUT

#: Muon energies spanning rest, the corpus's just-off-rest probe, and
#: increasing boosts.
MUON_ENERGIES = (MASS_MU, MASS_MU * (1.0 + 1e-12), 110.0, 150.0, 500.0, 1500.0, 1e5)

#: The signature string that is also the capsule's *name*, so a changed
#: `cdef` prototype fails loudly rather than being called through the wrong
#: ABI (the Task 3.4 constraint).
_POINT_SIGNATURE = b"double (double, double)"


def cython_point() -> Callable[[float, float], float]:
    """The live Cython ``dnde_photon_muon_point``, callable from Python.

    ``PYFUNCTYPE``, never ``CFUNCTYPE``: the latter releases the GIL, and
    this kernel calls back into ``scipy.special.cython_special``'s
    ``spence``, which segfaults without it.
    """
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.restype = ctypes.c_void_p
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

    capsule = cython_module.__pyx_capi__["dnde_photon_muon_point"]
    address = get_pointer(capsule, _POINT_SIGNATURE)
    return ctypes.PYFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)(address)


#: The platform the parity corpus was captured on, read from its own
#: manifest so the two can never drift apart.
CAPTURE_MACHINE = json.loads(
    (REPO_ROOT / "test" / "parity" / "data" / "manifest.json").read_text()
)["environment"]["machine"]

ON_THE_CAPTURING_PLATFORM = (
    sys.platform == "darwin" and platform.machine() == CAPTURE_MACHINE
)

#: The off-platform budget, as a fraction of the peak of the spectrum being
#: compared. **Derived** rather than measured, unlike the same constant in
#: ``test/test_core_positron_muon.py``, which carries a Linux measurement
#: from PR #63. The derivation below is what set the figure; PR #67's CI
#: then **held it green on Linux/glibc across py3.10-3.14**, which
#: confirms the budget is not too tight but measures no margin -- the
#: assertion reports nothing on success. Someone wanting the actual Linux
#: spread has to provoke a failure, as PR #63 did by accident.
#:
#: Two contributions, both bounded above:
#:
#: * ``spence``: off macOS/arm64 the C toolchain has no FMA to contract
#:   into, so scipy's cephes runs unfused and differs from this one by
#:   <=2.0e-15 relative (measured, both spellings, 13,000 points). Through
#:   ``(5/beta) * dspence * alpha / (3 pi E_mu)`` at the corpus's smallest
#:   ``beta`` of 1.4e-6 that is **2.2e-13 absolute**, against a peak of
#:   order 10 -- 2e-14 of peak.
#: * libm and contraction elsewhere: the same conditioning structure as the
#:   positron muon kernel, whose Linux spread was measured at **1.3e-10 of
#:   peak** at the worst point of a comparable sweep.
#:
#: 1e-8 of peak therefore leaves ~75x headroom over the larger of the two,
#: and is the same figure the positron kernel uses, so the two do not need
#: separate justification. It is still seven orders of magnitude tighter
#: than any physically meaningful change: a wrong branch, a dropped term or
#: a bad constant lands at O(1) against the peak.
OFF_PLATFORM_BUDGET = 1e-8


def assert_within_the_off_platform_budget(
    got: np.ndarray, want: np.ndarray, context: str
) -> None:
    """Assert two spectra agree to :data:`OFF_PLATFORM_BUDGET` of the peak.

    Split out from :func:`assert_matches_the_cython` so the budget can be
    exercised on *every* platform, including the one where the caller would
    otherwise take the bit-equality branch and leave this untested --
    :func:`TestAgainstTheCythonTwin.test_the_off_platform_budget_rejects_a_real_error`.

    ``atol`` is scaled by the peak rather than left at zero because this
    kernel's relative error is unbounded where it cancels: see "Why the
    comparison still has two modes".
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
            f"The largest effect this budget is meant to absorb was bounded "
            f"at 1.3e-10 x peak, so this is a defect, not a platform "
            f"difference."
        ),
    )


def assert_matches_the_cython(got: np.ndarray, want: np.ndarray, context: str) -> None:
    """The oracle, in whichever of its two modes this platform gets."""
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


def boosted_endpoint(emu: float) -> float:
    """The forward-cone endpoint ``(1 - r) E_mu (1 + beta) / 2``, in MeV."""
    beta = math.sqrt(max(1.0 - (MASS_MU / emu) ** 2, 0.0))
    return 0.5 * (1.0 - R) * emu * (1.0 + beta)


class TestDispatchWiring:
    """The entry point goes through ``map_unary`` with its own wording.

    One assertion per contract branch. The branch-by-branch argument about
    ``map_unary`` itself is ``test/test_core_dispatch.py``'s; what is
    specific to this kernel is that it reached that helper at all, and with
    the quantity string its Cython twin's ``assert`` carried.
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
        with pytest.raises(ValueError, match=r"^Photon energies must be 0 or 1-"):
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
        assert dnde(photon_energies=10.0, muon_energy=500.0) == dnde(10.0, 500.0)


class TestWrapperAndPublicApi:
    """The swap is wired all the way out to what users import."""

    def test_the_private_wrapper_returns_the_core_kernel_s_values(self) -> None:
        energies = np.geomspace(1.0, 400.0, 32)
        assert wrapper.dnde_photon_muon(energies, 500.0).tobytes() == (
            dnde(energies, 500.0).tobytes()
        )

    def test_the_public_spectra_name_resolves_to_the_same_function(self) -> None:
        assert spectra.dnde_photon_muon(10.0, 500.0) == dnde(10.0, 500.0)

    def test_the_cython_module_no_longer_exports_a_python_entry_point(self) -> None:
        # rules.md rule 1, as far as the capi exception allows: the extension
        # is still built for its `cdef` capsules, but no Python caller can
        # reach the implementation the swap replaced. Note the `.pyx` spelled
        # its entry point `dnde_photon`, not `dnde_photon_muon`.
        assert not hasattr(cython_module, "dnde_photon")

    def test_the_cdef_capsules_the_cimporters_need_are_intact(self) -> None:
        # Phase 06 Task 6.4 deletes these; until then
        # `hazma/spectra/_photon/_pion.pyx` cimports the point function and
        # both mediator decay-spectrum modules cimport one or the other, so
        # removing the `def` must not have disturbed them.
        exported = cython_module.__pyx_capi__
        assert set(exported) == {
            "dnde_photon_muon_point",
            "dnde_photon_muon_array",
        }

    def test_the_capsule_name_is_the_expected_c_signature(self) -> None:
        get_name = ctypes.pythonapi.PyCapsule_GetName
        get_name.restype = ctypes.c_char_p
        get_name.argtypes = [ctypes.py_object]
        capsule = cython_module.__pyx_capi__["dnde_photon_muon_point"]
        assert get_name(capsule) == _POINT_SIGNATURE

    def test_the_still_cython_photon_siblings_import(self) -> None:
        # The capi exception is only worth anything if the modules it exists
        # for still load. `_pion` cimports this kernel at C level.
        assert wrapper.dnde_photon_charged_pion(20.0, 200.0) >= 0.0


class TestAgainstTheCythonTwin:
    """The ``cdef`` the swap left behind, as an oracle.

    The parity corpus pins the values at the grids it chose; this reaches
    the same kernel at arbitrary arguments, which is what lets the edges be
    probed directly.
    """

    @pytest.mark.parametrize("emu", MUON_ENERGIES)
    def test_a_swept_grid_matches(self, emu: float) -> None:
        energies = np.geomspace(1e-4, boosted_endpoint(emu) * 1.5, 2001)
        assert_matches_the_cython(
            dnde(energies, emu), cython_spectrum(emu, energies), f"swept grid, {emu=}"
        )

    @pytest.mark.parametrize("emu", MUON_ENERGIES)
    def test_random_arguments_match(self, emu: float) -> None:
        rng = np.random.default_rng(4)
        energies = rng.uniform(0.0, boosted_endpoint(emu) * 1.1, 4000)
        assert_matches_the_cython(
            dnde(energies, emu),
            cython_spectrum(emu, energies),
            f"random arguments, {emu=}",
        )

    def test_the_kinematic_edges_match(self) -> None:
        for emu in (MASS_MU, MASS_MU * (1 + 1e-17), MASS_MU + 1e-16, 500.0, 1e9):
            endpoint = boosted_endpoint(emu)
            edges = np.array(
                [
                    SHIPPED_REST_FRAME_CUT,
                    np.nextafter(SHIPPED_REST_FRAME_CUT, np.inf),
                    TRUE_REST_FRAME_ENDPOINT,
                    endpoint,
                    np.nextafter(endpoint, 0.0),
                    np.nextafter(endpoint, np.inf),
                    emu / 2.0,
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
            energies = np.geomspace(1e-4, boosted_endpoint(emu) * 1.5, 2001)
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
        of 1e-6 of the peak -- four orders of magnitude above the largest
        difference this budget is meant to absorb, and still far too small to
        see in a plot -- must be rejected.
        """
        energies = np.geomspace(1e-4, 750.0, 2001)
        want = cython_spectrum(500.0, energies)
        nudged = want.copy()
        nudged[nudged.argmax()] += 1e-6 * want.max()

        assert_within_the_off_platform_budget(want, want, "unperturbed")
        with pytest.raises(AssertionError):
            assert_within_the_off_platform_budget(nudged, want, "perturbed")

    def test_a_nan_energy_propagates_and_both_agree_on_that(self) -> None:
        """A ``NaN`` energy comes back as ``NaN``, in both implementations.

        Worth pinning because the sibling ``dnde_positron_muon`` does the
        *opposite*: there ``fmax``/``fmin`` swallow the ``NaN`` and a finite
        number comes back. Nothing clips here, so a port that "helpfully"
        guarded would differ at an input the corpus never samples.
        """
        point = cython_point()
        for emu in (MASS_MU, 500.0):
            assert math.isnan(dnde(float("nan"), emu))
            assert math.isnan(point(float("nan"), emu))

    def test_a_zero_energy_is_nan_in_both(self) -> None:
        """``E_gamma = 0`` passes both support tests and then divides by it.

        Not a guard the ``.pyx`` has: ``x = 0`` fails ``x < 0`` and fails
        ``x >= (1-r)/(1-beta)``, so the closed form runs with
        ``xm = xp = 0`` and takes ``ln 0``. The rest frame is the branch that
        *does* return zero there.
        """
        point = cython_point()
        assert math.isnan(dnde(0.0, 500.0))
        assert math.isnan(point(0.0, 500.0))
        assert dnde(0.0, MASS_MU) == 0.0
        assert point(0.0, MASS_MU) == 0.0

    def test_a_below_threshold_muon_is_zero_in_both(self) -> None:
        point = cython_point()
        assert dnde(10.0, MASS_MU * 0.999_999) == 0.0
        assert point(10.0, MASS_MU * 0.999_999) == 0.0


class TestPhysics:
    """Statements about the spectrum, not about the code it replaced.

    These are what survives Phase 06 Task 6.4 deleting the ``.pyx``.
    """

    def test_the_spectrum_vanishes_below_the_muon_threshold(self) -> None:
        assert dnde(10.0, MASS_MU * 0.999) == 0.0
        assert dnde(-1.0, 500.0) == 0.0

    def test_a_muon_at_rest_stops_at_the_shipped_cut(self) -> None:
        assert dnde(np.nextafter(SHIPPED_REST_FRAME_CUT, np.inf), MASS_MU) == 0.0
        assert dnde(SHIPPED_REST_FRAME_CUT * 0.999, MASS_MU) > 0.0

    def test_the_rest_frame_cut_is_short_of_the_kinematic_endpoint(self) -> None:
        """The shipped defect, stated as the step it leaves behind.

        ``hazma/spectra/_photon/_muon.pyx:41`` guards the rest frame with
        ``y >= 1 - m_e/m_mu``; the kinematic endpoint, which the boosted
        branch and ``_pion.pyx``'s ``ENG_GAM_MAX_MURF`` both use, is
        ``y = 1 - r``. So the rest-frame spectrum is a hard zero over the
        top 0.2543 MeV of its support while an infinitesimally moving muon
        still radiates 5.34e-7 MeV^-1 there.

        Reproduced rather than repaired (``rules.md`` rule 1) and tracked in
        ``docs/followups/todo/photon-muon-rest-frame-endpoint-uses-the-wrong-power-of-r.md``.
        The just-off-rest value is compared at 1e-4 relative: the boosted
        form differences nearly-equal logarithms at ``beta = 1.4e-6``, which
        is how much of the value that cancellation has eaten.
        """
        assert ENDPOINT_GAP == pytest.approx(0.2542637928, rel=1e-9)
        assert dnde(SHIPPED_REST_FRAME_CUT * (1 + 1e-12), MASS_MU) == 0.0

        just_moving = dnde(
            SHIPPED_REST_FRAME_CUT * (1 + 1e-12), MASS_MU * (1.0 + 1e-12)
        )
        assert just_moving == pytest.approx(5.3356e-7, rel=1e-4)

        # And the boosted branch keeps radiating all the way to 1 - r.
        inside_the_gap = np.linspace(
            SHIPPED_REST_FRAME_CUT * 1.000_001, TRUE_REST_FRAME_ENDPOINT * 0.999, 64
        )
        assert np.all(dnde(inside_the_gap, MASS_MU) == 0.0)
        assert np.all(dnde(inside_the_gap, MASS_MU * (1.0 + 1e-12)) > 0.0)

    @pytest.mark.parametrize("emu", [110.0, 150.0, 500.0, 1500.0])
    def test_a_boosted_muon_ends_at_the_forward_cone_endpoint(self, emu: float) -> None:
        endpoint = boosted_endpoint(emu)
        assert dnde(endpoint * (1.0 - 1e-3), emu) > 0.0
        assert dnde(endpoint * (1.0 + 1e-9), emu) == 0.0

    @pytest.mark.parametrize("emu", [MASS_MU, 110.0, 500.0, 1500.0])
    def test_the_spectrum_is_finite_and_falls_monotonically_over_a_decade(
        self, emu: float
    ) -> None:
        """``dN/dE ~ 1/E`` in the infrared, so the spectrum decreases.

        Restricted to the well-inside region: within the last 0.1% of the
        support the closed form's terms cancel to a residual that can go
        slightly negative in both implementations, which
        :func:`test_the_endpoint_cancellation_is_bounded` bounds instead.
        """
        endpoint = boosted_endpoint(emu)
        energies = np.geomspace(1e-3 * endpoint, 0.9 * endpoint, 512)
        values = dnde(energies, emu)
        assert np.all(np.isfinite(values))
        assert np.all(values > 0.0)
        assert np.all(np.diff(values) < 0.0)

    @pytest.mark.parametrize("emu", [110.0, 500.0, 1e5])
    def test_the_endpoint_cancellation_is_bounded(self, emu: float) -> None:
        """Inside the last 0.1% the residual may be negative, but not large.

        Measured against the Cython twin on a 4001-point grid: the dip
        reaches 2.78e-4 of the value at 0.99 of the endpoint, and the *same*
        fraction at every parent energy from 110 MeV to 1e5 MeV because it
        depends only on the scaled variable. The bound is 1e-3, so it has
        3.6x headroom and still rejects anything structural.
        """
        endpoint = boosted_endpoint(emu)
        reference = dnde(0.99 * endpoint, emu)
        assert reference > 0.0
        energies = endpoint * (np.arange(4001) + 0.5) / 4001
        values = dnde(energies, emu)
        assert np.all(np.isfinite(values))
        assert np.all(values >= -1e-3 * reference)

    def test_the_rest_frame_limit_is_continuous_below_the_shipped_cut(self) -> None:
        """The ``E - m < eps`` branch is a removable singularity there.

        The Cython short-circuits to the rest frame within one epsilon MeV
        of rest because the in-flight form carries ``1/beta`` prefactors. The
        two must agree below the shipped cut, or the guard would be a step in
        a published spectrum rather than a numerical safeguard -- and above
        the cut they famously do not, which is
        :func:`test_the_rest_frame_cut_is_short_of_the_kinematic_endpoint`.
        2e-5 relative: the cancellation in the boosted form grows as
        ``beta -> 0``, and at ``E - m = 1e-9 MeV`` (``beta ~ 4.3e-6``) that is
        what it reaches.
        """
        energies = np.geomspace(1e-3, SHIPPED_REST_FRAME_CUT * 0.99, 501)
        at_rest = dnde(energies, MASS_MU)
        just_moving = dnde(energies, MASS_MU + 1e-9)
        np.testing.assert_allclose(just_moving, at_rest, rtol=2e-5)

    def test_the_boost_moves_the_endpoint_by_gamma_one_plus_beta(self) -> None:
        """A moving parent pushes the endpoint up by ``gamma(1 + beta)``.

        The one statement about the boost that needs no quadrature: the
        forward-cone endpoint scales with the parent's energy, so a spectrum
        that stopped at the rest-frame endpoint regardless of ``E_mu`` --
        a wrong Jacobian, or the rest-frame branch taken in flight -- fails
        here.
        """
        for emu in (150.0, 500.0, 1500.0):
            assert dnde(TRUE_REST_FRAME_ENDPOINT * 2.0, emu) > 0.0
            assert boosted_endpoint(emu) > TRUE_REST_FRAME_ENDPOINT
