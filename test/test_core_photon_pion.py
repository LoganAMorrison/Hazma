"""``hazma._core.photon``'s two pion spectra -- charged and neutral.

cython-to-rust Phase 04 Task 4.4. Shaped after
``test/test_core_photon_muon.py``, which is shaped after
``test/test_core_positron_muon.py``, the per-kernel template; deliberately
not a copy of ``test/test_core_dispatch.py``, whose 118 branch tests cover
the three shared dispatch helpers every kernel routes through unchanged.

Three parts:

1. :class:`TestDispatchWiring` -- one assertion per contract branch, for
   both entry points.
2. :class:`TestWrapperAndPublicApi` -- the swap wired out to what users
   import.
3. :class:`TestPhysics` -- statements about the spectra themselves.

Two more classes sat between them until cython-to-rust Task 6.4:
``TestNeutralPionAgainstTheCythonTwin`` and
``TestChargedPionAgainstTheCythonTwin``, driving the two ``cdef``s this
swap left behind through ``__pyx_capi__`` (the file stayed on disk for
them, and for the mediator decay modules that cimported them, until Task
6.2 took those and 6.4 the file). They were separate classes because they
were held to **different** standards, and the reasoning is kept below
because it is what set two surviving corpus budgets.

Why the two entry points got different oracles
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
which the corpus now pins: ``spectra.photon.neutral_pion`` is an
``EXACT`` case, so dropping either narrowing moves it.

Repaired angular support
------------------------
The charged-pion quadrature integrates over the widest channel's
support. The energy-variable boost identity below supplies an independent
check and partitions comparisons by scipy's convergence verdict. Rust
unit tests inspect the production quadrature's own termination flags.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.integrate import quad

from hazma import spectra
from hazma._core import photon as core_photon
from hazma.spectra import _photon as wrapper

if TYPE_CHECKING:
    from collections.abc import Callable

    #: What the `entry` fixture hands a dispatch test: one of the two
    #: `hazma._core.photon` entry points, and a parent energy in support.
    EntryPoint = tuple[Callable[..., object], float]


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

ENG_GAM_MAX_PIRG = 69.78345771948752
PHOTON_ENDPOINT_PIRF = (MASS_PI**2 - MASS_E**2) / (2 * MASS_PI)
ENG_MU_PIRF = 109.77820123634007


#: The band the two `pi -> l nu gamma` channels occupy as a fraction of
#: the charged pion's rest-frame photon spectrum. Wide, because it is a
#: presence check on two terms a port could silently drop, not a pinned
#: value -- the pinned values are the parity corpus's job.
RADIATIVE_FRACTION_BAND = (1e-3, 1e-1)


def charged_endpoint(epi: float) -> float:
    """The electron radiative edge boosted from the pion rest frame, MeV."""
    beta = math.sqrt(max(1.0 - (MASS_PI / epi) ** 2, 0.0))
    return PHOTON_ENDPOINT_PIRF * (epi / MASS_PI) * (1.0 + beta)


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
        # The widest radiative edge boosted forward. Above it every quadrature node
        # is outside the muon's and the radiative decays' support, so the
        # integral is identically zero rather than merely small.
        for epi in (MASS_PI * 1.05, 200.0, 500.0, 5000.0):
            edge = charged_endpoint(epi)
            assert dnde_charged(edge * 1.01, epi) == 0.0
            assert dnde_charged(edge * 10.0, epi) == 0.0

    def test_the_charged_pion_spectrum_is_positive_across_its_bulk(self) -> None:
        # Soft and intermediate photons complement the forward-tail probes.
        for epi in (MASS_PI * 1.05, 200.0, 500.0, 5000.0):
            grid = np.geomspace(0.5, ENG_GAM_MAX_PIRG, 40)
            assert np.all(dnde_charged(grid, epi) > 0.0), f"{epi=}"

    def test_the_forward_cone_reproduces_the_captured_repair(self) -> None:
        """The forward-cone value matches the independent corrected Cython capture."""
        # Six significant digits recorded by the capture, so 2e-7 relative
        # covers rounding the reference rather than the integrator's error.
        assert dnde_charged(900.0, 1396.0) == pytest.approx(3.585860e-7, rel=2e-7)
        for epi in (500.0, 1000.0, 2000.0, 5000.0, 1e4):
            grid = np.geomspace(0.5, charged_endpoint(epi) * 0.99, 60)
            assert np.all(dnde_charged(grid, epi) > 0.0)

    @pytest.mark.parametrize("epi", [MASS_PI, 500.0, 1396.0])
    def test_the_electron_radiative_sliver_is_not_clipped(self, epi: float) -> None:
        """The electron channel survives beyond the narrower legacy muon edge."""
        edge = charged_endpoint(epi)
        energy = edge * (1 + ENG_GAM_MAX_PIRG / PHOTON_ENDPOINT_PIRF) / 2
        value = dnde_charged(energy, epi)
        assert value > 0.0
        assert dnde_charged(np.array([energy]), epi)[0] == value
        if epi == MASS_PI:
            # Only the electron channel remains; the independent expression
            # cancels near the endpoint, hence the 1e-6 relative budget.
            assert value == pytest.approx(
                1.230e-4 * _pi_to_lnug(energy, MASS_E), rel=1e-6
            )
        else:
            expected, converged = _energy_boost_reference(energy, epi)
            assert converged
            # The electron formula cancels at its endpoint; the two
            # integration coordinates agree within 2e-4 in this tiny tail.
            assert value == pytest.approx(expected, rel=2e-4)
        assert dnde_charged(edge * (1 + 1e-10), epi) == 0.0

    def test_the_boost_matches_an_energy_integral_where_scipy_converges(self) -> None:
        """Change variables to E' and compare only converged reference integrals.

        The energy interval is intersected with physical support in MeV;
        integrating (dN/dE')/E' then dividing by 2 gamma beta yields MeV^-1.
        This formulation has no shrinking angular interval to miss.
        """
        converged = 0
        for epi in (150.0, 500.0, 1396.0, 5000.0):
            for fraction in (0.01, 0.25, 0.6, 0.9, 0.99):
                energy = fraction * charged_endpoint(epi)
                expected, success = _energy_boost_reference(energy, epi)
                if success:
                    converged += 1
                    # The production angular quad retains epsabs=1e-10;
                    # reference uses epsabs=0 and epsrel=1e-9. The absolute
                    # budget is its production stopping accuracy, not a
                    # parity budget; the captured repair separately pins it.
                    assert dnde_charged(energy, epi) == pytest.approx(
                        expected, rel=2e-5, abs=2e-10
                    )
        assert converged > 0, "the scipy-success comparison must execute"

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


def _energy_boost_reference(energy: float, epi: float) -> tuple[float, bool]:
    """Independent energy-variable boost of the rest spectrum, in MeV^-1."""
    gamma = epi / MASS_PI
    beta = math.sqrt(1 - (MASS_PI / epi) ** 2)
    lower = energy / (gamma * (1 + beta))
    upper = min(PHOTON_ENDPOINT_PIRF, energy * gamma * (1 + beta))
    if lower >= upper:
        return 0.0, True

    def integrand(rest_energy: float) -> float:
        rest = (
            0.9998770 * spectra.dnde_photon_muon(rest_energy, ENG_MU_PIRF)
            + 0.9998770 * _pi_to_lnug(rest_energy, MASS_MU)
            + 1.230e-4 * _pi_to_lnug(rest_energy, MASS_E)
        )
        return rest / rest_energy

    result = quad(
        integrand,
        lower,
        upper,
        epsabs=0,
        epsrel=1e-9,
        points=[
            x
            for x in ((MASS_PI**2 - MASS_MU**2) / (2 * MASS_PI), ENG_GAM_MAX_PIRG)
            if lower < x < upper
        ],
        limit=200,
        full_output=True,
    )
    successful_result_length = 3  # scipy appends a message on non-convergence
    return result[0] / (2 * gamma * beta), len(result) == successful_result_length
