"""``hazma._core.positron``'s charged-pion spectrum — the last positron kernel.

cython-to-rust Phase 04 Task 4.6. Shaped after
``test/test_core_positron_muon.py`` rather than after
``test/test_core_photon_rho.py``, because when it was written
``hazma/spectra/_positron/_pion.pyx`` **survived** this swap as a capi
provider: only its ``def`` went, and the live ``cdef`` stayed reachable
through ``__pyx_capi__``. What kept the file on disk was the two mediator
positron-spectrum modules, which reached its
``dnde_positron_charged_pion_array`` through a ``cimport``; Task 6.3
ported and deleted both, and Task 6.4 then deleted the file.

Three parts:

1. :class:`TestDispatchWiring` — one assertion per contract branch, enough
   to prove this entry point goes through ``map_unary`` with the wording
   its Cython twin used. Branch-by-branch reasoning about the helper
   itself stays in ``test/test_core_dispatch.py``.
2. :class:`TestWrapperAndPublicApi` — the swap wired out to what users
   import.
3. :class:`TestPhysics` — statements about the spectrum itself.

``TestAgainstTheCythonTwin`` was the third of four until Task 6.4.
It compared the surviving ``cdef`` within one measured budget on **every**
platform; there was no bit-equality mode to be had, because the port
replaces *scipy's* QUADPACK with the in-tree one (Phase 03 Task 3.3) and
two independent adaptive integrators are not bit-equal anywhere. The
``spectra.positron.charged_pion`` corpus case carries that comparison
now, at the same budget class and against arrays captured from the same
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

Those figures are the capturing platform's, and the range they hold over
is part of the claim: see :data:`CHARGED_PION_BUDGET` and
:meth:`TestPhysics.test_the_boost_window_is_ill_conditioned_at_extreme_boosts`
for why every grid here stops at ``E_pi = 1e4`` MeV.

The inherited normalization defect
----------------------------------
The muon-channel continuum is Task 4.1's kernel, which divides by the
Michel normalization where it should multiply, so it is low by ``1/N**2``
— 0.0374%. :class:`TestPhysics` asserts the integral the shipped code
produces, not the correct one; see
``docs/followups/todo/positron-muon-spectrum-normalization-inverted.md``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from hazma import spectra
from hazma._core import positron as core_positron
from hazma.scalar_mediator import (
    _scalar_mediator_positron_spectra as scalar_mediator_wrapper,
)
from hazma.spectra import _nbody
from hazma.spectra import _positron as wrapper
from hazma.vector_mediator import (
    _vector_mediator_positron_spectra as vector_mediator_wrapper,
)

dnde = core_positron.dnde_positron_charged_pion

REPO_ROOT = Path(__file__).resolve().parents[1]

QUANTITY = "Positron energies"
DIMENSION_ERROR = f"{QUANTITY} must be 0 or 1-dimensional."
TYPE_ERROR = f"{QUANTITY} must be a float or a NumPy array."

#: `hazma/_utils/constants.pxd`, the table this kernel's `.pyx` `include`s.
#: Spelled out rather than imported from `hazma.parameters` so a future
#: consolidation of the two tables cannot silently move the tests with the
#: code (`projects/cython-to-rust/rules.md` rule 4).
MASS_E = 0.5109989461
MASS_PI = 139.57039
BR_PI_TO_MU_NUMU = 0.9998770
BR_PI_TO_E_NUE = 1.230e-4


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

#: This kernel's budget against the Cython, on **every** platform, within
#: the boost range :data:`PION_ENERGIES` covers.
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
#:
#: **The range is part of the claim** -- see
#: :data:`ULTRARELATIVISTIC_PION_ENERGY` and
#: :meth:`TestPhysics.test_the_boost_window_is_ill_conditioned_at_extreme_boosts`.
#: The lower boost limit `gamma (E - beta k)` is a catastrophic
#: cancellation whose relative error grows like `2 gamma**2 eps`, so past
#: `gamma ~ 1e3` *no* tolerance against a second implementation is
#: honest. PR #74's first CI run proved it: green on macOS/arm64 and red
#: on all five Linux jobs at `E_pi = 1e6` (gamma = 7165), where the
#: predicted envelope is 2.3e-8 and the measured disagreement was 7.5e-9
#: plus a delta-function branch flip. The grids stop at 1e4 for that
#: reason and not to make a red test green: 1e4 is already 71x the pion
#: mass, seven times what the parity corpus samples, and far past the
#: sub-GeV domain hazma is for.
CHARGED_PION_BUDGET = 1e-12

#: A pion energy past which this kernel is ill-conditioned rather than
#: this port inaccurate: `gamma = 7165`, so `2 gamma**2 eps = 2.3e-8`.
#: Used only by the test that documents the regime.
ULTRARELATIVISTIC_PION_ENERGY = 1e6


#: Pion energies spanning rest, just-off-rest, and increasing boosts. `m_pi`
#: itself is included because this kernel is the one that returns *zero*
#: there rather than a rest-frame spectrum.
PION_ENERGIES = (MASS_PI, MASS_PI + 1e-9, 145.0, 200.0, 500.0, 1500.0, 1e4)


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

    def test_the_nbody_dispatch_table_reaches_the_ported_entry_point(self) -> None:
        # `_nbody.py` maps final-state names to spectrum functions; a swap
        # that repointed the wrapper but not this table would leave the
        # N-body path on the implementation the swap replaced.
        assert _nbody._dnde_positron_dict["pi"] is wrapper.dnde_positron_charged_pion

    def test_the_mediator_positron_spectra_keep_their_public_names(self) -> None:
        # These two names were `def`s in the mediator `.pyx` this file
        # used to feed by capsule. Task 6.3 moved them to `hazma._core`
        # under different spellings and had the wrappers re-export them
        # under these, so the check is that the rename stayed invisible.
        # An attribute is the cheap end of it; the parity corpus pins
        # their values.
        assert hasattr(scalar_mediator_wrapper, "dnde_decay_s")
        assert hasattr(vector_mediator_wrapper, "dnde_decay_v")

    def test_the_extension_this_module_replaces_is_gone(self) -> None:
        # This scanned the surviving `.pyx` for a `_positron._pion
        # cimport` while any remained, and guarded against its own sweep
        # going vacuous. Task 6.4 deleted the last of them, so the sweep
        # became the stronger claim it was approaching -- no Cython at all
        # -- which `test/test_no_cython_remains.py` now makes tree-wide.
        # What stays here is this module's own half of it.
        package = REPO_ROOT / "hazma" / "spectra" / "_positron"
        for suffix in (".pyx", ".pxd"):
            assert list(package.glob(f"*{suffix}")) == [], suffix


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

    def test_the_boost_window_is_ill_conditioned_at_extreme_boosts(self) -> None:
        """Why every grid in this module stops at ``E_pi = 1e4`` MeV.

        The boost integral runs from ``emin = gamma (E - beta k)`` with
        ``k = sqrt(E**2 - m_e**2)``. As ``beta -> 1`` that difference is a
        catastrophic cancellation: ``E - beta k`` falls like
        ``E / (2 gamma**2)`` while ``E`` and ``beta k`` stay ``O(E)``, so
        the *relative* error in ``emin`` grows like ``2 gamma**2 eps``.
        That is a property of the formula, not of either implementation,
        and it is why no cross-implementation tolerance is honest past
        ``gamma ~ 1e3``.

        PR #74's first CI run is the worked example: identical on
        macOS/arm64 (7.9e-16 at ``E_pi = 1e6``) and adrift on all five
        Linux jobs, where the shipped Cython is compiled without an FMA —
        x86-64's baseline has none — so its ``E - beta k`` is unfused
        while the port's ``mul_add`` is fused. One ulp there, amplified by
        ``2 gamma**2``, came out as 7.5e-9 at ``E = E_pi / 2`` and as a
        *branch flip* on the ``pi -> e nu`` line at ``E = E_pi``.

        So this test asserts the **mechanism** rather than a value, which
        is what survives a change of platform:

        1. the cancellation really does scale as ``E / (2 gamma**2)``;
        2. the envelope it implies is *quadratic* in the boost — a decade
           of ``gamma`` costs two decades of precision in ``emin``;
        3. at :data:`ULTRARELATIVISTIC_PION_ENERGY` that envelope is four
           decades outside :data:`CHARGED_PION_BUDGET`, so the grids'
           upper bound is derived rather than chosen.

        The envelope bounds what *can* propagate, not what does: it is
        2.3e-12 already at ``E_pi = 1e4``, where every sweep in this
        module passes on Linux at 1e-12, because the integrand vanishes at
        its own threshold and damps a wobble in the lower limit. What made
        1e6 different is that the same wobble also flipped the
        ``pi -> e nu`` delta function's ``eminus < e0 < eplus`` branch,
        turning a rounding difference into a support difference — which no
        tolerance should absorb. The empirical bracket from that CI run is
        therefore: covered and green at 1e4, ill-conditioned at 1e6.
        """
        eps = float(np.finfo(np.float64).eps)

        def envelope(epi: float) -> float:
            """``2 gamma**2 eps`` — the relative error ``emin`` inherits."""
            return 2.0 * (epi / MASS_PI) ** 2 * eps

        # 1. The cancellation is the one the derivation names. At E = E_pi
        #    the exact difference tends to E/(2 gamma**2); check it over
        #    three decades of boost.
        for epi in (1e4, 1e5, 1e6):
            beta = math.sqrt(1.0 - (MASS_PI / epi) ** 2)
            k = math.sqrt(epi * epi - MASS_E * MASS_E)
            assert epi - beta * k == pytest.approx(
                epi / (2.0 * (epi / MASS_PI) ** 2), rel=1e-3
            )

        # 2. Quadratic, not linear: ten times the boost is a hundred times
        #    the envelope. This is what makes the regime arrive suddenly.
        assert envelope(1e5) == pytest.approx(100.0 * envelope(1e4), rel=1e-12)

        # 3. Four decades outside the budget at 1e6, which is the derived
        #    reason no grid in this module goes there.
        assert envelope(ULTRARELATIVISTIC_PION_ENERGY) > 1e4 * CHARGED_PION_BUDGET
        assert ULTRARELATIVISTIC_PION_ENERGY > 10.0 * max(PION_ENERGIES)

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
