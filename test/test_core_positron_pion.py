"""``hazma._core.positron``'s charged-pion spectrum — the last positron kernel.

cython-to-rust Phase 04 Task 4.6. Shaped after
``test/test_core_positron_muon.py`` rather than after
``test/test_core_photon_rho.py``, and the choice is forced the same way it
was for Task 4.4's pion: ``hazma/spectra/_positron/_pion.pyx`` **survives**
this swap as a capi provider, because both mediator positron-spectrum
modules ``cimport`` its ``dnde_positron_charged_pion_array``
(``scalar_mediator_positron_spec.pyx:2``,
``vector_mediator_positron_spec.pyx:10``). Only its ``def`` went. So the
live ``cdef`` is still reachable through ``__pyx_capi__`` and is the
strongest available oracle; Phase 06 Task 6.4 deletes it and this module's
:class:`TestAgainstTheCythonTwin` with it.

Four parts:

1. :class:`TestDispatchWiring` — one assertion per contract branch, enough
   to prove this entry point goes through ``map_unary`` with the wording
   its Cython twin used. Branch-by-branch reasoning about the helper
   itself stays in ``test/test_core_dispatch.py``.
2. :class:`TestWrapperAndPublicApi` — the swap wired out to what users
   import, and the ``def`` gone while the capsules stay.
3. :class:`TestAgainstTheCythonTwin` — the surviving ``cdef``, compared
   within one measured budget on **every** platform. There is no
   bit-equality mode here and there is none to be had: the port replaces
   *scipy's* QUADPACK with the in-tree one (Phase 03 Task 3.3), and two
   independent adaptive integrators are not bit-equal anywhere. That is
   the same call ``test/test_core_photon_pion.py`` makes for its charged
   pion, and unlike ``test/test_core_positron_muon.py``'s two-mode
   comparison, which its closed-form kernel earns.
4. :class:`TestPhysics` — statements about the spectrum that outlive the
   Cython.

What this kernel is
-------------------
A charged pion decays to ``mu nu`` (BR 0.9998770) and to ``e nu``
(BR 1.230e-4), and both put a positron in the final state. The muon
channel contributes a continuum — Task 4.1's Michel spectrum for a muon
carrying ``E_mu^rf = 109.778`` MeV, boosted into the lab by one adaptive
``quad`` over the positron's rest-frame energy — and the electron channel
a line at ``E_e^rf = 69.786`` MeV, boosted by ``boost_delta_function``.

Measured drift
--------------
Against the live twin on this tree: **5.5e-15** worst relative over the
1,460 values the corpus pins, 1,304 of them bit-equal, and **3.5e-13** on
a denser off-corpus sweep (3,200 points over eight pion energies, worst at
``E_pi = 1e4`` MeV). ``test/parity/tolerances.py`` tightened this case from
``QUAD_RTOL`` (1e-8) to ``PORTED_QUAD_RTOL`` (1e-12) on those numbers.

The inherited normalization defect
----------------------------------
The muon-channel continuum is Task 4.1's kernel, which divides by the
Michel normalization where it should multiply, so it is low by ``1/N**2``
— 0.0374%. :class:`TestPhysics` asserts the integral the shipped code
produces, not the correct one; see
``docs/followups/todo/positron-muon-spectrum-normalization-inverted.md``.
"""

from __future__ import annotations

import ctypes
import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from hazma import spectra
from hazma._core import positron as core_positron
from hazma.scalar_mediator import scalar_mediator_positron_spec as scalar_mediator
from hazma.spectra import _nbody
from hazma.spectra import _positron as wrapper
from hazma.spectra._positron import _pion as cython_module
from hazma.vector_mediator import vector_mediator_positron_spec as vector_mediator

if TYPE_CHECKING:
    from collections.abc import Callable

dnde = core_positron.dnde_positron_charged_pion

QUANTITY = "Positron energies"
DIMENSION_ERROR = f"{QUANTITY} must be 0 or 1-dimensional."
TYPE_ERROR = f"{QUANTITY} must be a float or a NumPy array."

#: `hazma/_utils/constants.pxd`, the table this kernel's `.pyx` `include`s.
#: Spelled out rather than imported from `hazma.parameters` so a future
#: consolidation of the two tables cannot silently move the tests with the
#: code (`projects/cython-to-rust/rules.md` rule 4).
MASS_E = 0.5109989461
MASS_MU = 105.6583745
MASS_PI = 139.57039
BR_PI_TO_MU_NUMU = 0.9998770
BR_PI_TO_E_NUE = 1.230e-4

#: The muon's energy in the pion rest frame, MeV — the `.pyx`'s
#: `eng_mu_pi_rf`, and the second argument every integrand evaluation
#: passes to the Michel spectrum.
ENG_MU_PI_RF = 0.5 * (MASS_PI * MASS_PI + MASS_MU * MASS_MU) / MASS_PI

#: The positron energy of the two-body `pi -> e nu` line in the pion rest
#: frame, MeV — the `.pyx`'s `eng_e_pi_rf`. Also, to within one ulp, the
#: endpoint of the muon-channel continuum: the most energetic positron
#: from the chain is emitted forward at every step and carries exactly
#: this. `rust/src/kernels/positron_pion.rs` pins the one-ulp gap.
ENG_E_PI_RF = 0.5 * (MASS_PI * MASS_PI + MASS_E * MASS_E) / MASS_PI

#: The Michel normalization `_positron/_muon.pyx` divides by where it
#: should multiply. Named here because the muon channel's continuum is
#: that kernel, so this file's integral inherits the defect.
R_FACTOR = 1.0001870858234163

#: How far below one positron per pion the shipped total sits, from the
#: muon channel's inverted normalization: ``1 - 1/R_FACTOR**2``, or
#: 3.74e-4. Named so the assertion that the integral is *not* the
#: un-defected value states the separation it relies on.
NORMALIZATION_DEFICIT = 1.0 - 1.0 / R_FACTOR**2

#: This kernel's budget against the Cython, on **every** platform.
#:
#: A measurement, not a concession: the port replaces scipy's QUADPACK
#: with the in-tree one, and two independent adaptive integrators are not
#: bit-equal anywhere. Over every grid this module sweeps -- 7 pion
#: energies x (401 swept + 400 random points), plus the kinematic edges at
#: four more -- the worst `|got - want| / (peak + |want|)` is **6.9e-15**,
#: so 1e-12 leaves ~145x headroom. It is the same figure and the same
#: headroom Task 4.4 gave `test/test_core_photon_pion.py`'s charged pion
#: and `test/parity/tolerances.py` gives this case.
#:
#: `atol` is scaled by the peak alongside it, for the same reason the muon
#: kernels scale theirs: the spectrum passes through zero at its endpoint,
#: and a relative bound alone is unbounded at a cancellation.
CHARGED_PION_BUDGET = 1e-12

#: The signature string that is also the capsule's *name*, so a changed
#: `cdef` prototype fails loudly rather than being called through the wrong
#: ABI (the Task 3.4 constraint).
_POINT_SIGNATURE = b"double (double, double)"

#: Pion energies spanning rest, just-off-rest, and increasing boosts. `m_pi`
#: itself is included because this kernel is the one that returns *zero*
#: there rather than a rest-frame spectrum.
PION_ENERGIES = (MASS_PI, MASS_PI + 1e-9, 145.0, 200.0, 500.0, 1500.0, 1e4)


def cython_point() -> Callable[[float, float], float]:
    """The live Cython ``dnde_positron_charged_pion_point``, from Python.

    ``PYFUNCTYPE``, never ``CFUNCTYPE``: this ``cdef`` calls back into
    Python (``scipy.integrate.quad``), and ``CFUNCTYPE`` releases the GIL,
    so the call would segfault with no Python-level error.
    """
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.restype = ctypes.c_void_p
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

    capsule = cython_module.__pyx_capi__["dnde_positron_charged_pion_point"]
    address = get_pointer(capsule, _POINT_SIGNATURE)
    return ctypes.PYFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)(address)


def cython_spectrum(epi: float, energies: np.ndarray) -> np.ndarray:
    """The Cython twin evaluated pointwise over ``energies``."""
    point = cython_point()
    return np.array([point(float(e), epi) for e in energies])


def assert_matches_the_cython(got: np.ndarray, want: np.ndarray, context: str) -> None:
    """Assert the port agrees with the twin to :data:`CHARGED_PION_BUDGET`.

    One budget, no platform branch — see the module docstring for why
    there is no bit-equality mode to branch on.
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
            f"measured 6.9e-15 over these grids and a peak of {peak:.6e}). "
            f"Phase 03 Task 3.3's envelope for *converged* runs is 8.2e-11 "
            f"relative, so a failure here is either a defect or a shape "
            f"that stopped converging -- check the termination flag before "
            f"touching this number."
        ),
    )


class TestDispatchWiring:
    """The entry point goes through ``map_unary`` with its own wording."""

    def test_a_scalar_returns_a_python_float(self) -> None:
        value = dnde(10.0, 500.0)
        assert type(value) is float
        assert value > 0.0

    def test_a_numpy_scalar_and_a_zero_dim_array_take_the_scalar_path(self) -> None:
        expected = dnde(10.0, 500.0)
        assert dnde(np.float64(10.0), 500.0) == expected
        assert dnde(np.array(10.0), 500.0) == expected
        assert type(dnde(np.array(10.0), 500.0)) is float

    def test_an_array_returns_a_fresh_float64_array_of_the_same_length(self) -> None:
        energies = np.geomspace(1.0, 400.0, 64)
        values = dnde(energies, 500.0)
        assert values.dtype == np.float64
        assert values.shape == energies.shape
        assert not np.shares_memory(values, energies)

    def test_the_array_path_agrees_with_the_scalar_path_bit_for_bit(self) -> None:
        energies = np.geomspace(0.6, 600.0, 129)
        batched = dnde(energies, 500.0)
        one_at_a_time = np.array([dnde(float(e), 500.0) for e in energies])
        assert batched.tobytes() == one_at_a_time.tobytes()

    def test_a_sequence_is_accepted(self) -> None:
        assert dnde([10.0, 20.0], 500.0).tolist() == [
            dnde(10.0, 500.0),
            dnde(20.0, 500.0),
        ]

    def test_an_empty_grid_returns_an_empty_grid(self) -> None:
        assert dnde(np.array([], dtype=np.float64), 500.0).shape == (0,)

    def test_the_rank_message_is_the_cython_assert_verbatim(self) -> None:
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
        assert dnde(positron_energies=10.0, pion_energy=500.0) == dnde(10.0, 500.0)


class TestWrapperAndPublicApi:
    """The swap is wired all the way out to what users import."""

    def test_the_private_wrapper_returns_the_core_kernel_s_values(self) -> None:
        energies = np.geomspace(1.0, 400.0, 32)
        assert wrapper.dnde_positron_charged_pion(energies, 500.0).tobytes() == (
            dnde(energies, 500.0).tobytes()
        )

    def test_the_public_spectra_name_resolves_to_the_same_function(self) -> None:
        assert spectra.dnde_positron_charged_pion(10.0, 500.0) == dnde(10.0, 500.0)

    def test_the_cython_module_no_longer_exports_a_python_entry_point(self) -> None:
        # rules.md rule 1, as far as the capi exception allows: the
        # extension is still built for its `cdef` capsules, but no Python
        # caller can reach the implementation the swap replaced.
        assert not hasattr(cython_module, "dnde_positron_charged_pion")

    def test_the_cdef_capsules_the_mediator_modules_cimport_are_intact(self) -> None:
        # Phase 06 Task 6.4 deletes these; until then both mediator
        # positron spectrum modules cimport `_array`, so removing the
        # `def` must not have disturbed them.
        assert set(cython_module.__pyx_capi__) == {
            "dnde_positron_charged_pion_point",
            "dnde_positron_charged_pion_array",
        }

    def test_the_capsule_name_is_the_expected_c_signature(self) -> None:
        get_name = ctypes.pythonapi.PyCapsule_GetName
        get_name.restype = ctypes.c_char_p
        get_name.argtypes = [ctypes.py_object]
        capsule = cython_module.__pyx_capi__["dnde_positron_charged_pion_point"]
        assert get_name(capsule) == _POINT_SIGNATURE

    def test_the_nbody_dispatch_table_reaches_the_ported_entry_point(self) -> None:
        # `_nbody.py` maps final-state names to spectrum functions; a swap
        # that repointed the wrapper but not this table would leave the
        # N-body path on the implementation the swap replaced.
        assert _nbody._dnde_positron_dict["pi"] is wrapper.dnde_positron_charged_pion

    def test_the_mediator_positron_spectra_still_import(self) -> None:
        # The whole reason the file survives. An import is the cheap end of
        # the check; the parity corpus pins their values.
        assert hasattr(scalar_mediator, "dnde_decay_s")
        assert hasattr(vector_mediator, "dnde_decay_v")


class TestAgainstTheCythonTwin:
    """The ``cdef`` the swap left behind, as an oracle.

    The parity corpus pins this entry point at the grids it chose; this
    reaches the same kernel at arbitrary arguments, which is what lets the
    edges be probed directly.
    """

    @pytest.mark.parametrize("epi", PION_ENERGIES)
    def test_a_swept_grid_matches(self, epi: float) -> None:
        energies = np.geomspace(MASS_E * 0.5, epi * 1.5, 401)
        assert_matches_the_cython(
            dnde(energies, epi), cython_spectrum(epi, energies), f"swept grid, {epi=}"
        )

    @pytest.mark.parametrize("epi", PION_ENERGIES)
    def test_random_arguments_match(self, epi: float) -> None:
        rng = np.random.default_rng(46)
        energies = rng.uniform(0.0, epi * 1.1, 400)
        assert_matches_the_cython(
            dnde(energies, epi),
            cython_spectrum(epi, energies),
            f"random arguments, {epi=}",
        )

    def test_the_kinematic_edges_match(self) -> None:
        for epi in (MASS_PI, np.nextafter(MASS_PI, np.inf), 500.0, 1e6):
            edges = np.array(
                [
                    MASS_E,
                    np.nextafter(MASS_E, np.inf),
                    np.nextafter(MASS_E, 0.0),
                    ENG_E_PI_RF,
                    ENG_MU_PI_RF,
                    epi / 2.0,
                    epi,
                    np.nextafter(epi, np.inf),
                    0.0,
                    -1.0,
                ]
            )
            assert_matches_the_cython(
                dnde(edges, epi),
                cython_spectrum(epi, edges),
                f"kinematic edges, {epi=}",
            )

    def test_the_support_is_identical_everywhere(self) -> None:
        """Which energies are *zero* is structural, so it holds on any build.

        The budget is a statement about rounding; this is the statement
        rounding cannot excuse. A port that moved a threshold or a boost
        limit by one grid point turns this red on every platform,
        including the ones where the tolerance branch is in force.
        """
        for epi in PION_ENERGIES:
            energies = np.geomspace(MASS_E * 0.5, epi * 1.5, 401)
            got, want = dnde(energies, epi), cython_spectrum(epi, energies)
            assert np.array_equal(got == 0.0, want == 0.0), (
                f"the port and the Cython disagree about where the spectrum "
                f"vanishes at {epi=}, which no rounding difference explains"
            )

    def test_the_budget_rejects_a_real_error(self) -> None:
        """The budget is not vacuous.

        Every assertion above passes; that alone does not show the bound
        is doing any work. A perturbation of 1e-9 of the peak — three
        decades above the largest disagreement measured on these grids,
        and far too small to see in a plot — must be rejected.
        """
        energies = np.geomspace(MASS_E * 0.5, 750.0, 401)
        want = cython_spectrum(500.0, energies)
        nudged = want.copy()
        nudged[nudged.argmax()] += 1e-9 * want.max()

        assert_matches_the_cython(want, want, "unperturbed")
        with pytest.raises(AssertionError):
            assert_matches_the_cython(nudged, want, "perturbed")

    def test_a_pion_at_rest_is_zero_in_both(self) -> None:
        """The guard this kernel does *not* share with its siblings.

        Every other boosted kernel in the crate answers a rest-frame value
        within one ``DBL_EPSILON`` MeV of rest; this one returns exactly
        zero (``_pion.pyx:49-50``). The corpus's ``rest`` block for this
        case is therefore a block of zeros, and a port that "fixed" the
        asymmetry would fail the gate.
        """
        point = cython_point()
        for energy in (1.0, 10.0, 50.0, 69.0):
            assert dnde(energy, MASS_PI) == 0.0
            assert point(energy, MASS_PI) == 0.0
        # And one representable step up, both come alive together.
        just_above = np.nextafter(MASS_PI, np.inf)
        assert dnde(10.0, just_above) > 0.0
        assert point(10.0, just_above) > 0.0

    def test_a_nan_energy_agrees_in_both(self) -> None:
        """A ``NaN`` argument is not sampled by the corpus, so pin it here.

        Neither threshold comparison fires on a ``NaN``, so it reaches the
        quadrature over ``NaN`` limits in both implementations. What comes
        back is whatever the two integrators make of that, and they must
        make the same thing of it.
        """
        point = cython_point()
        assert math.isnan(dnde(float("nan"), 500.0)) == math.isnan(
            point(float("nan"), 500.0)
        )
        assert math.isnan(dnde(10.0, float("nan"))) == math.isnan(
            point(10.0, float("nan"))
        )


class TestPhysics:
    """Statements about the spectrum, not about the code it replaced.

    These are what survives Phase 06 Task 6.4 deleting the ``.pyx``.
    """

    def test_the_spectrum_vanishes_below_every_threshold(self) -> None:
        assert dnde(10.0, MASS_PI * 0.999) == 0.0
        assert dnde(MASS_E * 0.5, 500.0) == 0.0
        assert dnde(-1.0, 500.0) == 0.0
        assert dnde(10.0, MASS_PI) == 0.0

    def test_a_boosted_pion_extends_the_endpoint_past_the_rest_frame_line(
        self,
    ) -> None:
        epi = 500.0
        gamma = epi / MASS_PI
        beta = math.sqrt(1.0 - (MASS_PI / epi) ** 2)
        endpoint = gamma * ENG_E_PI_RF * (1.0 + beta)
        assert dnde(endpoint * 0.99, epi) > 0.0
        assert dnde(endpoint * 1.01, epi) == 0.0
        # And it really is a boost: the lab endpoint is gamma(1 + beta)
        # times the rest-frame one, which is 7.02 at this pion energy.
        assert endpoint / ENG_E_PI_RF == pytest.approx(gamma * (1.0 + beta))
        assert endpoint > 7.0 * ENG_E_PI_RF

    @pytest.mark.parametrize("epi", [145.0, 200.0, 500.0, 1500.0])
    def test_the_spectrum_is_finite_and_non_negative_everywhere(
        self, epi: float
    ) -> None:
        energies = np.geomspace(MASS_E * 1.000_001, epi * 2.0, 1001)
        values = dnde(energies, epi)
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)

    @pytest.mark.parametrize("epi", [145.0, 200.0, 500.0])
    def test_the_boost_conserves_positron_number(self, epi: float) -> None:
        """``int dN/dE dE`` is the same at every pion energy.

        Two statements at once. That the integral does not depend on
        ``epi`` is the physics — the boost moves positrons around in energy
        but creates none — and it is what a wrong Jacobian or a wrong
        prefactor would break. The value it takes is
        ``BR_mu / N**2 + BR_e``: one positron per pion, with the muon
        channel carrying Task 4.1's inverted normalization.

        Trapezoid on 200_001 points. 2e-4 relative: the spectrum has a step
        where the boosted electron line's window opens and a kink at the
        continuum endpoint, and a composite rule of this order gets no
        closer — measured, not chosen. The ``1/N**2`` deficit is 3.7e-4, so
        the bound still separates the shipped total from the correct one.
        """
        energies = np.linspace(MASS_E, 12.0 * epi, 200_001)
        integral = np.trapezoid(dnde(energies, epi), energies)
        shipped = BR_PI_TO_MU_NUMU / R_FACTOR**2 + BR_PI_TO_E_NUE
        assert integral == pytest.approx(shipped, rel=2e-4)
        # And it is not the un-defected total, which the muon channel's
        # normalization would give: that sits 3.7e-4 away, well outside
        # the quadrature bound above.
        undefected = BR_PI_TO_MU_NUMU + BR_PI_TO_E_NUE
        assert abs(integral - undefected) > NORMALIZATION_DEFICIT / 2.0

    def test_the_electron_line_is_a_visible_plateau_on_the_continuum(self) -> None:
        """``pi -> e nu`` boosts to a flat step, and it is where it should be.

        The line's lab-frame window is
        ``[E_rf/(gamma(1+beta)), E_rf/(gamma(1-beta))]``; inside it the
        spectrum carries an extra ``BR_e/(2 gamma beta E_rf)``, and outside
        it carries none. Pinning the *edge* rather than the height is what
        makes this a statement about the boost rather than about a
        constant.
        """
        epi = 500.0
        gamma = epi / MASS_PI
        beta = math.sqrt(1.0 - (MASS_PI / epi) ** 2)
        lower = ENG_E_PI_RF / (gamma * (1.0 + beta))
        plateau = BR_PI_TO_E_NUE / (2.0 * gamma * beta * ENG_E_PI_RF)

        just_below = dnde(lower * 0.999, epi)
        just_above = dnde(lower * 1.001, epi)
        # The continuum is smooth across the edge, so the jump is the line.
        assert just_above - just_below == pytest.approx(plateau, rel=5e-2)
        assert plateau > 0.0

    def test_a_faster_pion_spreads_the_same_positrons_over_more_energy(self) -> None:
        """The peak falls as the parent is boosted, at fixed total.

        A statement the corpus cannot make, because it compares each parent
        energy only against itself.
        """
        peaks = []
        for epi in (200.0, 500.0, 1500.0):
            energies = np.geomspace(MASS_E * 1.01, 12.0 * epi, 4001)
            peaks.append(float(dnde(energies, epi).max()))
        assert peaks[0] > peaks[1] > peaks[2]
