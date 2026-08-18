"""``hazma._core.photon``'s two pion spectra -- charged and neutral.

cython-to-rust Phase 04 Task 4.4. Shaped after
``test/test_core_photon_muon.py``, which is shaped after
``test/test_core_positron_muon.py``, the per-kernel template; deliberately
not a copy of ``test/test_core_dispatch.py``, whose 118 branch tests cover
the three shared dispatch helpers every kernel routes through unchanged.

Four parts:

1. :class:`TestDispatchWiring` -- one assertion per contract branch, for
   both entry points.
2. :class:`TestWrapperAndPublicApi` -- the swap wired out to what users
   import, plus the four ``cdef`` capsules ``_rho.pyx`` and both mediator
   decay-spectrum modules still cimport.
3. :class:`TestNeutralPionAgainstTheCythonTwin` and
   :class:`TestChargedPionAgainstTheCythonTwin` -- the two ``cdef``s the
   swap left behind, as oracles. They are separate classes because they
   are held to **different** standards; see below.
4. :class:`TestPhysics` -- statements about the spectra that outlive the
   Cython.

Why the two entry points get different oracles
----------------------------------------------
``dnde_photon_neutral_pion`` is closed-form arithmetic, so bit-equality is
available and the corpus pins it at ``EXACT`` (``rtol = 0``). It gets the
template's two-mode comparison: bit-for-bit on the platform the parity
corpus was captured on, a peak-scaled budget elsewhere.

``dnde_photon_charged_pion`` is an adaptive quadrature, and the port
replaces *scipy's* QUADPACK with the in-tree one (Phase 03 Task 3.3). Two
independent adaptive integrators are not bit-equal on any platform, so
there is no capturing-platform branch to take -- the comparison is a
budget everywhere, and the budget is a measurement rather than a
concession. Measured on this tree: **6.5e-15** worst relative over 8,000
sampled points spanning eight parent energies, and **2.6e-15** over the
1,500 values the corpus pins. Task 4.4 tightened
``test/parity/tolerances.py``'s budget for that case from ``QUAD_RTOL``
(1e-8) to ``PORTED_QUAD_RTOL`` (1e-12) on those numbers.

``cdef float`` is not a rounding nicety
---------------------------------------
``dnde_photon_neutral_pion_point`` declares ``cdef float beta`` and
``cdef float ret_val`` in a file where every other local is a ``double``,
and the shipped object confirms it with two ``fcvt`` round trips. The port
reproduces both as ``as f32 as f64``. Without them the spectrum moves in
the eighth significant figure -- four decades past the ``EXACT`` budget --
which is why :meth:`TestNeutralPionAgainstTheCythonTwin.test_the_single_precision_rounding_is_what_makes_it_bit_equal`
asserts the all-``f64`` spelling *fails*.

Where the quadrature stops converging
-------------------------------------
:class:`TestChargedPionAgainstTheCythonTwin` pins the other half of
Task 3.3's obligation: the port tracks scipy where QUADPACK converges and
may separate without bound where it does not, so each consumer has to say
whether any live shape reaches the second regime. This one does, but only
at ``E_pi >= 4e4`` MeV (``gamma_pi >= 290``) -- 40 GeV, against a library
whose domain is sub-GeV dark matter and a corpus whose most boosted block
is ``10 m_pi = 1396`` MeV. There the port's own termination flag equals
scipy's at all 88 sampled arguments (asserted in the Rust), but the
*values* are entitled to separate without bound -- so
:meth:`TestChargedPionAgainstTheCythonTwin.test_the_two_regimes_behave_as_task_3_3_said_they_would`
partitions the grid by scipy's verdict and holds each half to what is true
of it: 1e-12 where QUADPACK converged, sign and order of magnitude where
it did not. See :data:`DIVERGENT_REGIME_FACTOR` for why the loose half is
loose, and what two CI rounds cost to establish it.
"""

from __future__ import annotations

import ctypes
import json
import math
import platform
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from hazma import spectra
from hazma._core import photon as core_photon
from hazma.spectra import _photon as wrapper
from hazma.spectra._photon import _pion as cython_module

if TYPE_CHECKING:
    from collections.abc import Callable

    #: What the `entry` fixture hands a dispatch test: one of the two
    #: `hazma._core.photon` entry points, and a parent energy in support.
    EntryPoint = tuple[Callable[..., object], float]

REPO_ROOT = Path(__file__).resolve().parents[1]

dnde_charged = core_photon.dnde_photon_charged_pion
dnde_neutral = core_photon.dnde_photon_neutral_pion

QUANTITY = "Photon energies"
DIMENSION_ERROR = f"{QUANTITY} must be 0 or 1-dimensional."
TYPE_ERROR = f"{QUANTITY} must be a float or a NumPy array."

#: `hazma/_utils/constants.pxd`, the table this kernel's `.pyx` `include`s.
#: Spelled out rather than imported from `hazma.parameters` so a future
#: consolidation of the two tables cannot silently move the tests with the
#: code (`projects/cython-to-rust/rules.md` rule 4).
MASS_PI = 139.57039
MASS_PI0 = 134.9768
MASS_MU = 105.6583745
MASS_E = 0.5109989461
BR_PI0_TO_A_A = 98.823e-2

#: The three literals `hazma/spectra/_photon/_pion.pyx:16-18` hard-codes.
#: They are *legacy*-table values in a file that `include`s the PDG one --
#: the mixed provenance Phase 03 Task 3.1 recorded and rule 4 preserves.
ENG_GAM_MAX_MURF = 52.82795006985128
ENG_GAM_MAX_PIRG = 69.78345771948752
ENG_MU_PIRF = 109.77820123634007

#: Pion energies spanning rest, the corpus's just-off-rest probe, and
#: increasing boosts. The corpus itself stops at `10 * mass`.
CHARGED_ENERGIES = (
    MASS_PI,
    MASS_PI * (1.0 + 1e-12),
    MASS_PI * 1.05,
    200.0,
    500.0,
    1500.0,
    1e4,
)
NEUTRAL_ENERGIES = (
    MASS_PI0,
    MASS_PI0 * (1.0 + 1e-12),
    MASS_PI0 * 1.05,
    200.0,
    500.0,
    1500.0,
    1e4,
    1e5,
)

#: The signature string that is also the capsule's *name*, so a changed
#: `cdef` prototype fails loudly rather than being called through the
#: wrong ABI (the Task 3.4 constraint).
_POINT_SIGNATURE = b"double (double, double)"


def cython_point(name: str) -> Callable[[float, float], float]:
    """A live Cython ``*_point`` ``cdef``, callable from Python.

    ``PYFUNCTYPE``, never ``CFUNCTYPE``: the latter releases the GIL, and
    the charged-pion kernel calls back into Python (``scipy.integrate.quad``
    and, through it, ``scipy.special.cython_special``), which segfaults
    without it.
    """
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.restype = ctypes.c_void_p
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

    capsule = cython_module.__pyx_capi__[name]
    address = get_pointer(capsule, _POINT_SIGNATURE)
    return ctypes.PYFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)(address)


#: The platform the parity corpus was captured on, read from its own
#: manifest so the two can never drift apart. Used only by the *neutral*
#: pion: the charged one has no bit-equality mode on any platform.
CAPTURE_MACHINE = json.loads(
    (REPO_ROOT / "test" / "parity" / "data" / "manifest.json").read_text()
)["environment"]["machine"]

ON_THE_CAPTURING_PLATFORM = (
    sys.platform == "darwin" and platform.machine() == CAPTURE_MACHINE
)

#: The neutral pion's off-platform budget, as a fraction of the peak.
#: Derived rather than measured, and generous for what it has to absorb:
#: the kernel reaches only `sqrt` and the four arithmetic operations, and
#: `sqrt` is correctly rounded on every IEEE-754 platform, so the only
#: portable source of difference is `1 - (m/E)^2` cancelling near rest.
#: Even there the `f32` truncation that follows discards ~29 bits, so a
#: last-ulp `f64` difference has to be ~1e-8 relative before it can change
#: the rounded `beta` at all. 1e-8 of peak is therefore the scale at which
#: a real difference could first appear, and it is still seven decades
#: tighter than any physically meaningful change: a wrong branch, a lost
#: branching fraction or a dropped factor of two lands at O(1) against the
#: peak. Same figure as the muon kernels use, so the three do not need
#: separate justification.
OFF_PLATFORM_BUDGET = 1e-8

#: The charged pion's budget against the Cython, on **every** platform.
#: A measurement, not a concession: two independent adaptive quadratures
#: are never bit-equal, and this tree measures the gap at 6.5e-15 relative
#: over 8,000 points. 1e-12 leaves ~150x headroom and is the same figure
#: Task 4.4 gave the parity corpus for this case.
CHARGED_PION_BUDGET = 1e-12

#: The window the neutral pion's `f32` truncations put between the shipped
#: value and an all-`f64` transcription of the same three lines. `f32` has
#: 24 significant bits, so a truncation shows up around `2**-24 = 6e-8`
#: relative; the bounds are a decade either side of that, wide enough not
#: to be brittle and tight enough that "no truncation at all" (0) and "a
#: different formula" (>1e-6) both fail.
F32_TRUNCATION_WINDOW = (1e-9, 1e-6)

#: How far apart the port and scipy may be in the **non-converging**
#: regime, as a ratio rather than a relative tolerance.
#:
#: Phase 03 Task 3.3 measured that the two are entitled to separate
#: *without bound* once QUADPACK stops converging, because Wynn's
#: epsilon-algorithm is chaotic on a non-converging sequence. So there is
#: no honest tolerance to assert there, and PR #68 proved it the hard way:
#: 1e-10 (from a 2.8e-11 macOS measurement) failed on Linux at 6.2998e-10,
#: and 1e-8 then failed at the *next* point, 3.0552e-08. Chasing that one
#: measurement at a time is how a gate becomes vacuous.
#:
#: A factor of 2 asserts the thing that is actually true and actually
#: worth knowing: the port has not wandered off, it is still computing the
#: same integral to the same size and sign. The precision claim lives in
#: the converged half of the same test, at `CHARGED_PION_BUDGET`, and in
#: the three sweep classes -- all of which hold at 1e-12 on Linux as well
#: as macOS, which is the substantive result of those CI rounds.
DIVERGENT_REGIME_FACTOR = 2.0

#: The band the two `pi -> l nu gamma` channels occupy as a fraction of
#: the charged pion's rest-frame photon spectrum. Wide, because it is a
#: presence check on two terms a port could silently drop, not a pinned
#: value -- the pinned values are the parity corpus's job.
RADIATIVE_FRACTION_BAND = (1e-3, 1e-1)


def assert_within_the_off_platform_budget(
    got: np.ndarray, want: np.ndarray, context: str
) -> None:
    """Assert two spectra agree to :data:`OFF_PLATFORM_BUDGET` of the peak.

    Split out from :func:`assert_neutral_matches_the_cython` so the budget
    is exercised on *every* platform, including the one where the caller
    would otherwise take the bit-equality branch and leave this untested.
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
            f"This kernel reaches only sqrt and arithmetic, so a platform "
            f"cannot explain a difference this large -- it is a defect."
        ),
    )


def assert_neutral_matches_the_cython(
    got: np.ndarray, want: np.ndarray, context: str
) -> None:
    """The neutral-pion oracle, in whichever of its two modes applies."""
    if ON_THE_CAPTURING_PLATFORM:
        assert got.tobytes() == want.tobytes(), (
            f"{context}: not bit-equal to the Cython on the platform the "
            f"corpus was captured on, where the port is written to reproduce "
            f"it exactly"
        )
        return
    assert_within_the_off_platform_budget(got, want, context)


def assert_charged_matches_the_cython(
    got: np.ndarray, want: np.ndarray, context: str
) -> None:
    """The charged-pion oracle: one budget, no platform branch.

    ``atol`` is scaled by the peak for the same reason the muon kernels
    scale theirs: the spectrum passes through zero at the endpoint, and a
    relative bound alone is unbounded at a cancellation.
    """
    finite = np.isfinite(want)
    peak = float(np.abs(want[finite]).max()) if finite.any() else 0.0
    np.testing.assert_allclose(
        got,
        want,
        rtol=CHARGED_PION_BUDGET,
        atol=CHARGED_PION_BUDGET * peak,
        err_msg=(
            f"{context}: the port left the in-tree QUADPACK's measured "
            f"agreement with scipy's ({CHARGED_PION_BUDGET:.0e}, against a "
            f"measured 6.5e-15 over 8,000 points and a peak of "
            f"{peak:.6e}). Phase 03 Task 3.3's envelope for *converged* "
            f"runs is 8.2e-11 relative, so a failure here is either a "
            f"defect or a shape that stopped converging -- check the "
            f"termination flag before touching this number."
        ),
    )


def cython_spectrum(name: str, parent_energy: float, grid: np.ndarray) -> np.ndarray:
    """A Cython twin evaluated pointwise over ``grid``."""
    point = cython_point(name)
    return np.array([point(float(e), parent_energy) for e in grid])


def charged_endpoint(epi: float) -> float:
    """The forward-cone photon endpoint from a charged pion, MeV.

    ``ENG_GAM_MAX_PIRG`` is the pion-rest-frame maximum (the muon's own
    endpoint boosted out of the muon frame); boosting it forward by the
    pion's own ``gamma (1 + beta)`` gives the lab-frame edge. This is
    exactly what the ``.pyx``'s unreferenced ``eng_gam_max`` computes.
    """
    beta = math.sqrt(max(1.0 - (MASS_PI / epi) ** 2, 0.0))
    return ENG_GAM_MAX_PIRG * (epi / MASS_PI) * (1.0 + beta)


def neutral_edges(epi: float) -> tuple[float, float]:
    """The box edges ``E_pi (1 -+ beta) / 2``, MeV."""
    beta = math.sqrt(max(1.0 - (MASS_PI0 / epi) ** 2, 0.0))
    return 0.5 * epi * (1.0 - beta), 0.5 * epi * (1.0 + beta)


class TestDispatchWiring:
    """Both entry points go through ``map_unary`` with their own wording.

    One assertion per contract branch. The branch-by-branch argument about
    ``map_unary`` itself is ``test/test_core_dispatch.py``'s; what is
    specific to these kernels is that they reached that helper at all, and
    with the quantity string their Cython twin's ``assert`` carried.
    """

    #: Both entry points, at a parent energy that puts each in support.
    #: 500 MeV is 3.6 times the charged pion's mass and 3.7 times the
    #: neutral one's, so a 250 MeV photon is interior to both.
    PARENT_ENERGY = 500.0

    @pytest.fixture(params=[dnde_charged, dnde_neutral], ids=["charged", "neutral"])
    def entry(
        self, request: pytest.FixtureRequest
    ) -> tuple[Callable[..., object], float]:
        return request.param, self.PARENT_ENERGY

    def test_a_scalar_returns_a_python_float(self, entry: EntryPoint) -> None:
        dnde, parent = entry
        value = dnde(250.0, parent)
        assert type(value) is float
        assert value > 0.0

    def test_a_numpy_scalar_and_a_zero_dim_array_take_the_scalar_path(
        self, entry: EntryPoint
    ) -> None:
        dnde, parent = entry
        expected = dnde(250.0, parent)
        assert dnde(np.float64(250.0), parent) == expected
        assert dnde(np.array(250.0), parent) == expected
        assert type(dnde(np.array(250.0), parent)) is float

    def test_an_array_returns_a_fresh_float64_array_of_the_same_length(
        self, entry: EntryPoint
    ) -> None:
        dnde, parent = entry
        energies = np.geomspace(1.0, 400.0, 64)
        values = dnde(energies, parent)
        assert values.dtype == np.float64
        assert values.shape == energies.shape
        assert not np.shares_memory(values, energies)

    def test_the_array_path_agrees_with_the_scalar_path_bit_for_bit(
        self, entry: EntryPoint
    ) -> None:
        # `map_unary` calls the same kernel either way, so a broadcasting
        # bug shows up here and nowhere else the corpus looks.
        dnde, parent = entry
        energies = np.geomspace(0.4, 600.0, 129)
        batched = dnde(energies, parent)
        one_at_a_time = np.array([dnde(float(e), parent) for e in energies])
        assert batched.tobytes() == one_at_a_time.tobytes()

    def test_a_sequence_is_accepted(self, entry: EntryPoint) -> None:
        dnde, parent = entry
        assert dnde([100.0, 250.0], parent).tolist() == [
            dnde(100.0, parent),
            dnde(250.0, parent),
        ]

    def test_an_empty_grid_returns_an_empty_grid(self, entry: EntryPoint) -> None:
        dnde, parent = entry
        assert dnde(np.array([], dtype=np.float64), parent).shape == (0,)

    def test_the_rank_message_is_the_cython_assert_verbatim(
        self, entry: EntryPoint
    ) -> None:
        # The wording is user-visible API. Both `.pyx` entry points spelled
        # it "Photon energies"; a reworded port is a silent break no
        # numerical gate can see.
        dnde, parent = entry
        with pytest.raises(ValueError) as excinfo:
            dnde(np.ones((2, 2)), parent)
        assert str(excinfo.value) == DIMENSION_ERROR

    def test_a_non_float64_array_is_a_value_error(self, entry: EntryPoint) -> None:
        dnde, parent = entry
        with pytest.raises(ValueError, match="must be a float64 array"):
            dnde(np.arange(4), parent)

    def test_a_non_number_is_a_type_error(self, entry: EntryPoint) -> None:
        dnde, parent = entry
        with pytest.raises(TypeError) as excinfo:
            dnde(None, parent)
        assert str(excinfo.value) == TYPE_ERROR

    def test_both_arguments_are_accepted_by_keyword(self) -> None:
        # The Cython twins were `def`s and took keywords; a positional-only
        # port would be a silent public-API narrowing.
        assert dnde_charged(photon_energies=20.0, pion_energy=500.0) == dnde_charged(
            20.0, 500.0
        )
        assert dnde_neutral(photon_energies=250.0, pion_energy=500.0) == dnde_neutral(
            250.0, 500.0
        )


class TestWrapperAndPublicApi:
    """The swap is wired all the way out to what users import."""

    def test_the_private_wrappers_return_the_core_kernels_values(self) -> None:
        energies = np.geomspace(1.0, 400.0, 32)
        assert wrapper.dnde_photon_charged_pion(energies, 500.0).tobytes() == (
            dnde_charged(energies, 500.0).tobytes()
        )
        assert wrapper.dnde_photon_neutral_pion(energies, 500.0).tobytes() == (
            dnde_neutral(energies, 500.0).tobytes()
        )

    def test_the_public_spectra_names_resolve_to_the_same_functions(self) -> None:
        assert spectra.dnde_photon_charged_pion(20.0, 500.0) == dnde_charged(
            20.0, 500.0
        )
        assert spectra.dnde_photon_neutral_pion(250.0, 500.0) == dnde_neutral(
            250.0, 500.0
        )

    def test_the_cython_module_no_longer_exports_a_python_entry_point(self) -> None:
        # rules.md rule 1, as far as the capi exception allows: the extension
        # is still built for its `cdef` capsules, but no Python caller can
        # reach the implementation the swap replaced.
        assert not hasattr(cython_module, "dnde_photon_charged_pion")
        assert not hasattr(cython_module, "dnde_photon_neutral_pion")

    def test_the_cdef_capsules_the_cimporters_need_are_intact(self) -> None:
        # Phase 06 Task 6.4 deletes these. Until then all four are live:
        # `hazma/spectra/_photon/_rho.pyx` cimports both point functions,
        # `hazma/vector_mediator/vector_mediator_decay_spectrum.pyx` both
        # array functions plus the neutral point one, and
        # `hazma/scalar_mediator/scalar_mediator_decay_spectrum.pyx` all
        # four. Removing the `def`s must not have disturbed any of them.
        assert set(cython_module.__pyx_capi__) == {
            "dnde_photon_charged_pion_point",
            "dnde_photon_charged_pion_array",
            "dnde_photon_neutral_pion_point",
            "dnde_photon_neutral_pion_array",
        }

    def test_the_capsule_names_are_the_expected_c_signatures(self) -> None:
        get_name = ctypes.pythonapi.PyCapsule_GetName
        get_name.restype = ctypes.c_char_p
        get_name.argtypes = [ctypes.py_object]
        for name in (
            "dnde_photon_charged_pion_point",
            "dnde_photon_neutral_pion_point",
        ):
            assert get_name(cython_module.__pyx_capi__[name]) == _POINT_SIGNATURE

    def test_the_still_cython_dependents_import_and_run(self) -> None:
        # The capi exception is only worth anything if the modules it exists
        # for still load and still produce numbers.
        assert wrapper.dnde_photon_charged_rho(20.0, 900.0) >= 0.0
        assert wrapper.dnde_photon_neutral_rho(20.0, 900.0) >= 0.0

        from hazma.scalar_mediator import HiggsPortal  # noqa: PLC0415
        from hazma.vector_mediator import VectorMediator  # noqa: PLC0415

        model = VectorMediator(
            mx=250.0,
            mv=1000.0,
            gvxx=1.0,
            gvuu=1.0,
            gvdd=-1.0,
            gvss=0.0,
            gvee=0.0,
            gvmumu=0.0,
        )
        total = model.total_spectrum(np.array([10.0, 100.0]), 600.0)
        assert np.all(np.isfinite(total))

        scalar = HiggsPortal(mx=250.0, ms=1000.0, gsxx=1.0, stheta=1e-1)
        assert np.all(
            np.isfinite(scalar.total_spectrum(np.array([10.0, 100.0]), 600.0))
        )


class TestNeutralPionAgainstTheCythonTwin:
    """``dnde_photon_neutral_pion_point`` as an oracle: bit-equal.

    Closed-form arithmetic, so the template's two-mode comparison applies
    unchanged.
    """

    @pytest.mark.parametrize("epi", NEUTRAL_ENERGIES)
    def test_a_swept_grid_matches(self, epi: float) -> None:
        _, upper = neutral_edges(epi)
        energies = np.geomspace(1e-4, upper * 1.5, 2001)
        assert_neutral_matches_the_cython(
            dnde_neutral(energies, epi),
            cython_spectrum("dnde_photon_neutral_pion_point", epi, energies),
            f"swept grid, {epi=}",
        )

    @pytest.mark.parametrize("epi", NEUTRAL_ENERGIES)
    def test_random_arguments_match(self, epi: float) -> None:
        rng = np.random.default_rng(44)
        _, upper = neutral_edges(epi)
        energies = rng.uniform(0.0, upper * 1.1, 4000)
        assert_neutral_matches_the_cython(
            dnde_neutral(energies, epi),
            cython_spectrum("dnde_photon_neutral_pion_point", epi, energies),
            f"random arguments, {epi=}",
        )

    def test_the_box_edges_match(self) -> None:
        for epi in (MASS_PI0, MASS_PI0 * (1 + 1e-15), 200.0, 1e5):
            lower, upper = neutral_edges(epi)
            edges = np.array(
                [
                    lower,
                    np.nextafter(lower, 0.0),
                    np.nextafter(lower, np.inf),
                    upper,
                    np.nextafter(upper, 0.0),
                    np.nextafter(upper, np.inf),
                    epi / 2.0,
                    0.0,
                    -1.0,
                    np.inf,
                ]
            )
            assert_neutral_matches_the_cython(
                dnde_neutral(edges, epi),
                cython_spectrum("dnde_photon_neutral_pion_point", epi, edges),
                f"box edges, {epi=}",
            )

    def test_the_single_precision_rounding_is_what_makes_it_bit_equal(self) -> None:
        """The ``cdef float`` declarations are load-bearing, not cosmetic.

        An all-``f64`` transcription of the same three lines differs from the
        Cython in the eighth significant figure -- four decades past the
        ``EXACT`` budget the corpus holds this case to. Asserted so that a
        future reader who finds ``as f32 as f64`` odd sees what removing it
        costs before removing it.
        """
        epi = 500.0
        egam = 250.0
        beta_f64 = math.sqrt(1.0 - (MASS_PI0 / epi) ** 2)
        all_f64 = (BR_PI0_TO_A_A * 2.0) / (epi * beta_f64)

        shipped = dnde_neutral(egam, epi)
        point = cython_point("dnde_photon_neutral_pion_point")
        assert shipped == point(egam, epi)
        assert shipped != all_f64
        relative = abs((shipped - all_f64) / all_f64)
        low, high = F32_TRUNCATION_WINDOW
        assert low < relative < high, (
            f"the f32 truncation should show up near 6e-8 relative, got "
            f"{relative:e}"
        )

    def test_the_support_is_identical_everywhere(self) -> None:
        """Which energies are *zero* is structural, so it holds on any build.

        The budget above is a statement about rounding; this is the statement
        rounding cannot excuse. A port that moved a box edge by one grid
        point turns this red on every platform.
        """
        for epi in NEUTRAL_ENERGIES:
            _, upper = neutral_edges(epi)
            energies = np.geomspace(1e-4, upper * 1.5, 2001)
            got = dnde_neutral(energies, epi)
            want = cython_spectrum("dnde_photon_neutral_pion_point", epi, energies)
            assert np.array_equal(got == 0.0, want == 0.0), (
                f"the port and the Cython disagree about where the box "
                f"vanishes at {epi=}, which no rounding difference explains"
            )

    def test_the_off_platform_budget_rejects_a_real_error(self) -> None:
        """The budget is not vacuous, asserted where the budget is not used.

        On the capturing platform :func:`assert_neutral_matches_the_cython`
        takes its bit-equality branch, so nothing else here would exercise
        the tolerance at all and it could rot to ``inf`` unnoticed.
        """
        energies = np.geomspace(1e-4, 750.0, 2001)
        want = cython_spectrum("dnde_photon_neutral_pion_point", 500.0, energies)
        nudged = want.copy()
        nudged[nudged.argmax()] += 1e-6 * want.max()

        assert_within_the_off_platform_budget(want, want, "unperturbed")
        with pytest.raises(AssertionError):
            assert_within_the_off_platform_budget(nudged, want, "perturbed")

    def test_a_nan_energy_is_zero_in_both(self) -> None:
        """A ``NaN`` photon energy leaves the box, in both implementations.

        The ``.pyx`` writes a chained ``lower <= E <= upper``, both halves of
        which are false for a ``NaN``, so the box's ``0.0`` seed survives.
        Opposite to the *charged* pion below, whose guard is a rejection --
        the two live in one file and disagree, which is exactly the kind of
        thing a port invents a house style for and gets wrong.
        """
        point = cython_point("dnde_photon_neutral_pion_point")
        for epi in (MASS_PI0, 500.0):
            assert dnde_neutral(float("nan"), epi) == 0.0
            assert point(float("nan"), epi) == 0.0


class TestChargedPionAgainstTheCythonTwin:
    """``dnde_photon_charged_pion_point`` as an oracle: one budget.

    No capturing-platform branch: the port replaces scipy's QUADPACK with
    the in-tree one, and two independent adaptive integrators are not
    bit-equal anywhere. See the module docstring.
    """

    @pytest.mark.parametrize("epi", CHARGED_ENERGIES)
    def test_a_swept_grid_matches(self, epi: float) -> None:
        energies = np.geomspace(1e-4, charged_endpoint(epi) * 1.5, 401)
        assert_charged_matches_the_cython(
            dnde_charged(energies, epi),
            cython_spectrum("dnde_photon_charged_pion_point", epi, energies),
            f"swept grid, {epi=}",
        )

    @pytest.mark.parametrize("epi", CHARGED_ENERGIES)
    def test_random_arguments_match(self, epi: float) -> None:
        rng = np.random.default_rng(444)
        energies = rng.uniform(0.0, charged_endpoint(epi) * 1.1, 300)
        assert_charged_matches_the_cython(
            dnde_charged(energies, epi),
            cython_spectrum("dnde_photon_charged_pion_point", epi, energies),
            f"random arguments, {epi=}",
        )

    def test_the_kinematic_edges_match(self) -> None:
        for epi in (MASS_PI, MASS_PI * (1 + 1e-15), 200.0, 1e4):
            edge = charged_endpoint(epi)
            edges = np.array(
                [
                    edge,
                    np.nextafter(edge, 0.0),
                    np.nextafter(edge, np.inf),
                    ENG_GAM_MAX_MURF,
                    ENG_GAM_MAX_PIRG,
                    ENG_MU_PIRF,
                    epi / 2.0,
                    0.0,
                    -1.0,
                ]
            )
            assert_charged_matches_the_cython(
                dnde_charged(edges, epi),
                cython_spectrum("dnde_photon_charged_pion_point", epi, edges),
                f"kinematic edges, {epi=}",
            )

    def test_the_budget_rejects_a_real_error(self) -> None:
        """The budget is used on every platform, but assert it is not vacuous.

        1e-9 of the peak is four decades above the measured 6.5e-15 and
        still far too small to see in a plot.
        """
        energies = np.geomspace(1e-4, 400.0, 401)
        want = cython_spectrum("dnde_photon_charged_pion_point", 500.0, energies)
        nudged = want.copy()
        nudged[nudged.argmax()] += 1e-9 * want.max()

        assert_charged_matches_the_cython(want, want, "unperturbed")
        with pytest.raises(AssertionError):
            assert_charged_matches_the_cython(nudged, want, "perturbed")

    def test_below_threshold_is_exactly_zero_in_both(self) -> None:
        point = cython_point("dnde_photon_charged_pion_point")
        for epi in (0.0, 1.0, MASS_PI * 0.999, np.nextafter(MASS_PI, 0.0)):
            assert dnde_charged(20.0, epi) == 0.0
            assert point(20.0, epi) == 0.0

    def test_a_nan_energy_propagates_and_both_agree_on_that(self) -> None:
        """A ``NaN`` photon energy comes back ``NaN``, in both.

        ``dnde_pi_to_lnug``'s guard is a *rejection* -- ``x < 0 or (1-r) <
        x`` -- and both comparisons are false for a ``NaN``, so it falls
        through to the arithmetic rather than returning zero. Opposite to
        the neutral pion above, in the same file.
        """
        point = cython_point("dnde_photon_charged_pion_point")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert math.isnan(dnde_charged(float("nan"), 500.0))
            assert math.isnan(point(float("nan"), 500.0))

    def test_the_two_regimes_behave_as_task_3_3_said_they_would(self) -> None:
        """Phase 03 Task 3.3's obligation, discharged for the first `qagp`.

        Task 3.3 measured that the QUADPACK port tracks scipy **wherever
        QUADPACK converges, and only there**: past that, Wynn's
        epsilon-algorithm is chaotic on a non-converging sequence and the
        two can separate without bound. Each consumer has to say whether a
        live shape reaches the second regime.

        This one does, far outside hazma's domain -- around ``E_pi = 4e4``
        MeV (``gamma_pi ~ 290``), 40 GeV against a sub-GeV library and two
        decades above the corpus's ``10 m_pi`` ceiling. So this test
        partitions the grid by scipy's *own* verdict and asserts what is
        true of each half:

        * **converged** (scipy issues no ``IntegrationWarning``) -- 51 of
          the 64 points, held to :data:`CHARGED_PION_BUDGET` on the
          capturing platform and :data:`OFF_PLATFORM_BUDGET` elsewhere.
          The scope is not decoration: this grid runs to ``E_pi = 1e5``
          MeV, a decade past the swept-grid classes, and **the capturing
          platform cannot measure what that costs off it** -- macOS/arm64
          reports 2.22e-16 (one ulp) at *every* boost from 1e3 to 1e5,
          while Linux/glibc reached **1.2916e-12** at
          ``E_gam = 1e-3, E_pi = 3e4``. A flat 1e-12 passed macOS and
          three-quarters of CI, which is exactly how this class of mistake
          survives (``docs/agents/lessons.md``
          ``[platform-scoped-oracle-asserted-globally]``). The swept-grid
          classes keep their flat 1e-12 because they stop at
          ``E_pi = 1e4``, where three CI rounds have held it;
        * **not converged** -- the other 13, held only to "the port has
          not wandered off": same sign, and within
          :data:`DIVERGENT_REGIME_FACTOR`. Measured worst on the capturing
          platform: 2.75e-11, and see that constant for what Linux
          reported.

        The loose half is loose on purpose, and two CI rounds are why.
        PR #68 first asserted 1e-10 there, having measured 2.8e-11 on
        macOS/arm64; Linux reported **6.2998e-10** at ``E_gam = 1.0,
        E_pi = 4e4``. Raising it to 1e-8 simply moved the failure to the
        next-worst point -- **3.0552e-08** at ``E_gam = 0.01, E_pi = 6e4``.
        That is the regime behaving exactly as Task 3.3 documented, and
        chasing it one measurement at a time is how a gate becomes vacuous
        (the warning
        ``docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md``
        makes about widening until it passes). A bound the numerics do not
        support is not worth asserting; a sign-and-magnitude check is.

        **This test does not compare termination flags, and an earlier
        draft's name said it did** (PR #68 review round 1). It reads
        scipy's warning only to choose which assertion to apply, and it
        never sees the port's `Ier` -- `hazma._core.photon` returns the
        value alone, exactly as the ``.pyx`` does, so no Python caller
        can. A regression that changed the port's flag without changing
        its value would pass here.

        That half is gated where it is observable, in
        ``rust/src/kernels/photon_pion.rs``: one test asserts the whole
        live domain returns ``ier = 0``, a second asserts the other regime
        is still reachable and the options never go invalid. What is
        deliberately *not* gated anywhere is the point-by-point flag map
        above the boundary -- measured once as equal to scipy's at all 88
        points of an 11 x 8 grid, on the capturing platform, and left as a
        measurement because the values there are demonstrably
        platform-dependent and a flag is a discrete decision on the same
        chaotic sequence.
        """
        point = cython_point("dnde_photon_charged_pion_point")
        pion_energies = (1e3, 1e4, 3e4, 4e4, 5e4, 6e4, 8e4, 1e5)
        photon_energies = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)

        converged = diverged = 0
        for epi in pion_energies:
            for egam in photon_energies:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    want = point(egam, epi)
                quadrature_converged = not caught
                got = dnde_charged(egam, epi)

                if want == 0.0:
                    assert got == 0.0, f"port is nonzero at {egam=}, {epi=}"
                    converged += quadrature_converged
                    diverged += not quadrature_converged
                    continue

                ratio = got / want
                if quadrature_converged:
                    converged += 1
                    budget = (
                        CHARGED_PION_BUDGET
                        if ON_THE_CAPTURING_PLATFORM
                        else OFF_PLATFORM_BUDGET
                    )
                    assert abs(ratio - 1.0) < budget, (
                        f"port and scipy disagree where QUADPACK *converged* "
                        f"at {egam=}, {epi=}: {got!r} vs {want!r}. This half "
                        f"is a real precision claim -- widen it only with a "
                        f"measurement from the platform that failed, and "
                        f"never the capturing one, which reports one ulp "
                        f"here at every boost."
                    )
                else:
                    diverged += 1
                    assert (
                        1.0 / DIVERGENT_REGIME_FACTOR
                        < ratio
                        < (DIVERGENT_REGIME_FACTOR)
                    ), (
                        f"port and scipy are not even the same size where "
                        f"QUADPACK did *not* converge, at {egam=}, {epi=}: "
                        f"{got!r} vs {want!r}. Separation is licensed here, "
                        f"but this much of it means the port stopped "
                        f"computing the same integral."
                    )

        # Neither half may be empty, or the assertions above are vacuous --
        # the whole point is that this kernel reaches both regimes.
        assert converged > 0 and diverged > 0, (
            f"the grid no longer spans both regimes ({converged=}, "
            f"{diverged=}); it is meant to straddle E_pi ~ 4e4 MeV"
        )
        assert converged + diverged == len(pion_energies) * len(photon_energies)


class TestPhysics:
    """Statements about the spectra that outlive the Cython."""

    def test_the_neutral_pion_box_carries_two_photons_per_decay(self) -> None:
        """``int dN/dE dE = 2 BR(pi0 -> gamma gamma)``, at every boost.

        The one number this kernel exists to produce, and the corpus cannot
        state it -- the corpus pins values, not an integral over them. The
        box is exact, so the area is width x height with no quadrature
        error; the tolerance is set by the ``f32`` rounding of ``beta`` and
        the height, which is ~1e-7 relative.
        """
        for epi in (MASS_PI0 * 1.001, 200.0, 500.0, 5000.0):
            lower, upper = neutral_edges(epi)
            height = dnde_neutral(0.5 * (lower + upper), epi)
            area = (upper - lower) * height
            assert area == pytest.approx(
                2.0 * BR_PI0_TO_A_A, rel=1e-6
            ), f"the pi0 box does not carry 2 x BR photons at {epi=}"

    def test_the_neutral_pion_box_is_flat(self) -> None:
        epi = 500.0
        lower, upper = neutral_edges(epi)
        interior = lower + np.linspace(0.05, 0.95, 32) * (upper - lower)
        values = dnde_neutral(interior, epi)
        assert (
            len(set(values.tobytes()[i : i + 8] for i in range(0, values.nbytes, 8)))
            == 1
        )

    def test_the_neutral_pion_box_is_symmetric_about_half_the_pion_energy(self) -> None:
        # A boosted two-body line is symmetric in E about E_pi/2. Nothing in
        # the implementation enforces it, so it is a real check on the edges.
        epi = 500.0
        offsets = np.linspace(0.0, 0.45, 16) * epi
        left = dnde_neutral(epi / 2.0 - offsets, epi)
        right = dnde_neutral(epi / 2.0 + offsets, epi)
        assert left.tobytes() == right.tobytes()

    def test_a_charged_pion_at_rest_reproduces_its_own_rest_frame_sum(self) -> None:
        """At rest the ``cos theta`` integral is trivial and can be done by hand.

        ``gamma = 1``, ``beta = 0``, the Jacobian is ``1/2`` and the
        integrand is independent of the angle, so the integral over
        ``[-1, 1]`` is the integrand itself. That makes the whole boost
        machinery checkable against a value the quadrature never touches --
        a statement the original never made.
        """
        for egam in (1.0, 10.0, 30.0, 60.0):
            integrated = dnde_charged(egam, MASS_PI)
            by_hand = (
                0.9998770 * spectra.dnde_photon_muon(egam, ENG_MU_PIRF)
                + 0.9998770 * _pi_to_lnug(egam, MASS_MU)
                + 1.230e-4 * _pi_to_lnug(egam, MASS_E)
            )
            assert integrated == pytest.approx(by_hand, rel=1e-12), (
                f"the rest-frame charged-pion spectrum is not its own "
                f"integrand at {egam=}"
            )

    def test_the_charged_pion_spectrum_vanishes_above_its_boosted_endpoint(
        self,
    ) -> None:
        # `ENG_GAM_MAX_PIRG` boosted forward. Above it every quadrature node
        # is outside the muon's and the radiative decays' support, so the
        # integral is identically zero rather than merely small.
        for epi in (MASS_PI * 1.05, 200.0, 500.0, 5000.0):
            edge = charged_endpoint(epi)
            assert dnde_charged(edge * 1.01, epi) == 0.0
            assert dnde_charged(edge * 10.0, epi) == 0.0

    def test_the_charged_pion_spectrum_is_positive_across_its_bulk(self) -> None:
        # Bounded well below the boosted endpoint on purpose -- see
        # `test_the_forward_cone_is_a_hard_zero_the_quadrature_invented`
        # for why "the bulk" stops where it does.
        for epi in (MASS_PI * 1.05, 200.0, 500.0, 5000.0):
            grid = np.geomspace(0.5, ENG_GAM_MAX_PIRG, 40)
            assert np.all(dnde_charged(grid, epi) > 0.0), f"{epi=}"

    def test_the_forward_cone_is_a_hard_zero_the_quadrature_invented(self) -> None:
        """A live 2.1.0 defect the port reproduces on purpose.

        `hazma/spectra/_photon/_pion.pyx:123` integrates over the whole of
        ``cos theta``, but the integrand is nonzero only where the
        pion-rest-frame photon energy stays below ``ENG_GAM_MAX_PIRG``.
        Above roughly ``0.77`` of the boosted endpoint at ``gamma = 7``,
        that window is narrower than QUADPACK's largest first-rule
        abscissa (~0.9956), so every node returns zero, the error estimate
        is zero, and the routine terminates *successfully* with ``0.0``.

        Asserting the physically correct value here would contradict the
        parity corpus (`projects/cython-to-rust/rules.md` rule 1), so what
        is asserted is the defect itself and the fact that both
        implementations have it identically. The repair is
        ``docs/followups/todo/charged-pion-photon-spectrum-misses-the-forward-cone.md``,
        blocked until after Phase 06 Task 6.4.
        """
        point = cython_point("dnde_photon_charged_pion_point")
        # (parent energy, photon energy) inside the true support where the
        # shipped answer is nevertheless exactly zero, with the reference
        # value the follow-up records.
        for epi, egam in ((1000.0, 800.0), (1396.0, 900.0), (1396.0, 1200.0)):
            assert egam < charged_endpoint(epi), "sample must be inside support"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                assert dnde_charged(egam, epi) == 0.0
                assert point(egam, epi) == 0.0

        # And the zeros are in the *same places* in both, which is the
        # statement that makes this faithfulness rather than coincidence.
        for epi in (500.0, 1000.0, 2000.0, 5000.0, 1e4):
            grid = np.geomspace(0.5, charged_endpoint(epi) * 0.99, 60)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                want = cython_spectrum("dnde_photon_charged_pion_point", epi, grid)
            got = dnde_charged(grid, epi)
            assert np.array_equal(got == 0.0, want == 0.0), (
                f"the port and the Cython disagree about where the "
                f"quadrature loses the forward cone at {epi=}"
            )

    def test_the_radiative_channels_are_a_percent_level_correction(self) -> None:
        """``pi -> l nu gamma`` is small next to the boosted muon spectrum.

        Both are in the integrand and a port that dropped either would still
        produce a plausible-looking spectrum, so the *ratio* is what pins
        them. Measured at rest, where the integrand is the spectrum.
        """
        egam = 20.0
        muon_only = 0.9998770 * spectra.dnde_photon_muon(egam, ENG_MU_PIRF)
        total = dnde_charged(egam, MASS_PI)
        radiative = (total - muon_only) / total
        low, high = RADIATIVE_FRACTION_BAND
        assert low < radiative < high, (
            f"the radiative channels contribute {radiative:.3%} of the "
            f"rest-frame charged-pion spectrum at {egam} MeV, which is "
            f"outside the percent-level band this kernel is known to have"
        )


def _pi_to_lnug(egam: float, ml: float) -> float:
    """``dN/dE`` for ``pi -> l nu gamma`` in the pion rest frame, MeV^-1.

    An independent Python transcription of the ``.pyx``'s ``dnde_pi_to_lnug``,
    written from the same closed form and *without* the FMA map the Rust
    reproduces -- so it agrees only to ordinary double precision. It exists
    so :meth:`TestPhysics.test_a_charged_pion_at_rest_reproduces_its_own_rest_frame_sum`
    can assemble the rest-frame sum from parts, which is a statement about
    the boost machinery rather than about the arithmetic.
    """
    alpha = 1.0 / 137.035999084
    f_a, f_v0, slope = 0.0119, 0.0254, 0.10
    fpi = 130.41 * math.sqrt(0.5)

    x = 2.0 * egam / MASS_PI
    r = (ml / MASS_PI) ** 2
    if x < 0.0 or (1.0 - r) < x:
        return 0.0

    f_v = f_v0 * (1.0 + slope * (1.0 - x))
    f = (r + x - 1.0) * (
        MASS_PI**2 * x**4 * (f_a**2 + f_v**2) * (r * r - r * x + r - 2 * (x - 1) ** 2)
        - 12
        * math.sqrt(2)
        * fpi
        * MASS_PI
        * r
        * (x - 1)
        * x**2
        * (f_a * (r - 2 * x + 1) + f_v * x)
        - 24 * fpi**2 * r * (x - 1) * (4 * r * (x - 1) + (x - 2) ** 2)
    )
    g = (
        12
        * math.sqrt(2)
        * fpi
        * r
        * (x - 1) ** 2
        * math.log(r / (1.0 - x))
        * (
            MASS_PI * x**2 * (f_a * (x - 2 * r) - f_v * x)
            + math.sqrt(2) * fpi * (2 * r * r - 2 * r * x - x * x + 2 * x - 2)
        )
    )
    return (
        alpha
        * (f + g)
        / (24 * math.pi * MASS_PI * fpi**2 * (r - 1) ** 2 * (x - 1) ** 2 * r * x)
    )
