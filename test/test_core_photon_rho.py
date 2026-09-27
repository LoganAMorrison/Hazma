"""``hazma._core.photon``'s two rho spectra — charged and neutral.

cython-to-rust Phase 04 Task 4.5, the phase's declared numerical stress
test: an adaptive quadrature whose integrand is an adaptive quadrature.

Shaped after ``test/test_core_photon_tables.py`` rather than after
``test/test_core_photon_pion.py``, and the choice is forced. Task 4.4's
pion module could call a live ``cdef`` because ``_pion.pyx`` survives as a
capi provider; **nothing cimported** ``_photon/_rho.pyx``, so rules.md
rule 1 applies without its capi exception and the whole file went in this
swap's PR. There is no twin left to call, so the against-the-Cython
evidence is the parity corpus plus the direct comparison run *before* the
deletion — see "What replaced the Cython oracle" below.

Five parts:

1. :class:`TestDispatchWiring` — one assertion per contract branch, for
   both entry points.
2. :class:`TestWrapperAndPublicApi` — the swap wired out to what users
   import, and the twin gone from the tree.
3. :class:`TestAgainstAnIndependentBoostIntegral` — the outer integral
   recomputed in Python with ``scipy.integrate.quad`` over the *ported*
   pion kernels. A genuine second opinion on the layer this task added,
   using a different QUADPACK binding over the same integrand.
4. :class:`TestTheBoostedTail` — the top of the boosted spectrum, which
   the Cython's whole-window integral lost, against a log-energy
   reference and the ``gamma`` scaling of the photon energy.
5. :class:`TestPhysics` — statements about the spectra that outlive the
   Cython.

What replaced the Cython oracle
-------------------------------
Measured against the live twin on this tree before ``_rho.pyx`` was
deleted, at the parent energies the corpus samples and at four it does
not:

* on the **1,395 values the corpus pins** for each entry point,
  ``charged_rho`` moved by at most **1.5e-13** relative and
  ``neutral_rho`` by **3.2e-15**, with roughly three quarters of the
  points bit-equal;
* on a denser off-corpus sweep — 3,200 points over eight parent energies
  — the worst was **2.5e-11**, at a photon energy whose boost window
  straddles the pi0 box's upper edge, where a jump discontinuity sits
  inside the interval and a single bisection decision can flip. Even
  there the difference is five decades below the ``abserr`` the
  integrator itself reports for that call.

Task 4.5 tightened ``test/parity/tolerances.py``'s budget for both cases
from ``NESTED_RTOL`` (1e-6) to ``PORTED_NESTED_RTOL`` (1e-9) on those
numbers. :data:`INDEPENDENT_BUDGET` here is the same figure for the same
reason.

The nesting the module docs of ``rust/src/kernels/photon_rho.rs``
describe is real and it is why this file is slower than its siblings:
every value costs an outer ``qags`` whose every integrand evaluation
costs an inner ``qagp``. The grids below are sized accordingly.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.integrate import quad

from hazma import spectra
from hazma._core import photon as core_photon
from hazma.spectra import _nbody
from hazma.spectra import _photon as wrapper

if TYPE_CHECKING:
    from collections.abc import Callable

    #: What the `entry` fixture hands a dispatch test: one of the two
    #: `hazma._core.photon` entry points, and a parent energy in support.
    EntryPoint = tuple[Callable[..., object], float]

REPO_ROOT = Path(__file__).resolve().parents[1]

dnde_charged = core_photon.dnde_photon_charged_rho
dnde_neutral = core_photon.dnde_photon_neutral_rho

QUANTITY = "Photon energies"
DIMENSION_ERROR = f"{QUANTITY} must be 0 or 1-dimensional."
TYPE_ERROR = f"{QUANTITY} must be a float or a NumPy array."

#: `hazma/_utils/constants.pxd`, the table this kernel's `.pyx` `include`d.
#: Spelled out rather than imported from `hazma.parameters` so a future
#: consolidation of the two tables cannot silently move the tests with the
#: code (`projects/cython-to-rust/rules.md` rule 4).
MASS_RHO = 775.26
MASS_PI = 139.57039
MASS_PI0 = 134.9768

MASS_E = 0.5109989461

#: The pion-rest-frame photon endpoint, MeV: ``(m_pi^2 - m_e^2) / (2 m_pi)``,
#: the edge of the ``pi -> e nu gamma`` channel and the widest of the
#: charged pion's three. ``PHOTON_ENDPOINT_PIRF`` in
#: ``rust/src/kernels/photon_pion.rs``.
PHOTON_ENDPOINT_PIRF = (MASS_PI * MASS_PI - MASS_E * MASS_E) / (2.0 * MASS_PI)


def two_body_energy(q: float, m1: float, m2: float) -> float:
    """``hazma/_utils/kinematics.pxd``'s helper, in Python.

    ``E1 = (q^2 + m1^2 - m2^2) / (2 q)`` — the energy of particle 1 in the
    rest frame of a parent of mass ``q`` decaying to ``m1 + m2``, MeV.
    """
    return (q * q + m1 * m1 - m2 * m2) / (2.0 * q)


#: The daughter energies in the rho rest frame, MeV. The charged rho's
#: two differ; the neutral rho's coincide at ``m_rho/2``.
ENG_PI_CHARGED_RHO = two_body_energy(MASS_RHO, MASS_PI, MASS_PI0)
ENG_PI0_CHARGED_RHO = two_body_energy(MASS_RHO, MASS_PI0, MASS_PI)
ENG_PI_NEUTRAL_RHO = MASS_RHO / 2.0

#: The rho energies the tests sweep: exactly at rest, one step past the
#: ``E - m < DBL_EPSILON`` short circuit, and three boosts. The corpus
#: uses the same ladder up to ``10 m_rho``; the nesting makes anything
#: denser expensive, so the grids are short instead of the ladder.
RHO_ENERGIES = (
    MASS_RHO,
    MASS_RHO * (1.0 + 1e-12),
    MASS_RHO * 1.05,
    MASS_RHO * 2.0,
    MASS_RHO * 10.0,
)

#: ``scipy.integrate.quad``'s settings at both of the deleted ``.pyx``'s
#: call sites, and therefore at ``RHO_QUAD`` in the Rust. Reproduced here
#: so :class:`TestAgainstAnIndependentBoostIntegral` integrates the same
#: problem the port does rather than a better-resolved one.
QUAD_EPSABS = 1e-10
QUAD_EPSREL = 1e-5

#: How far the port may sit from the same integral evaluated through
#: scipy's QUADPACK binding. The same 1e-9 the parity corpus now gives
#: these two cases, and derived the same way — 6,600x headroom over the
#: worst drift measured against the Cython on the corpus's own points
#: (1.5e-13) and 40x over the worst found anywhere off it (2.5e-11).
#: Tighter than that would be pinning one platform's bisection decisions;
#: looser would stop separating a real error, since the coarsest thing
#: that can go wrong here — a swapped daughter energy — moves the answer
#: by O(1).
INDEPENDENT_BUDGET = 1e-9

#: The lower edge of the charged rho's pi0 box, MeV: ``E_pi0 (1 - beta)/2``
#: at ``E_pi0 =`` :data:`ENG_PI0_CHARGED_RHO`. The sharpest structure
#: either integrand has, and the energy either side of which the two
#: integrands' ratio inverts.
PI0_BOX_LOWER_EDGE = 12.156854062150506

#: Both edges of that box, MeV: the lower one above and ``E_pi0 (1 + beta)/2``.
#: Jump discontinuities of the charged rho's integrand, which
#: :func:`log_energy_reference` hands to scipy as break points.
PI0_BOX_EDGES = (PI0_BOX_LOWER_EDGE, 374.6597689891406)

#: The band the charged-to-neutral integrand ratio occupies **below**
#: :data:`PI0_BOX_LOWER_EDGE`, where the charged rho has one charged pion
#: against the neutral rho's two. Exactly 0.5 in the limit where the two
#: daughter energies coincide; they differ by 1.6 MeV, so the measured
#: ratio runs 0.5000-0.5002 over the probes used. +-1% is wide enough not
#: to be brittle and far too tight for a dropped factor of two.
BELOW_BOX_RATIO_BAND = (0.49, 0.51)


def charged_pion_dnde(egam: float, epi: float) -> float:
    """The ported charged-pion spectrum, MeV^-1."""
    return float(core_photon.dnde_photon_charged_pion(egam, epi))


def neutral_pion_dnde(egam: float, epi: float) -> float:
    """The ported neutral-pion spectrum, MeV^-1."""
    return float(core_photon.dnde_photon_neutral_pion(egam, epi))


def charged_integrand(e: float) -> float:
    """``hazma/spectra/_photon/_rho.pyx``'s ``integrand_charged_rho``."""
    return (
        charged_pion_dnde(e, ENG_PI_CHARGED_RHO)
        + neutral_pion_dnde(e, ENG_PI0_CHARGED_RHO)
    ) / e


def neutral_integrand(e: float) -> float:
    """``hazma/spectra/_photon/_rho.pyx``'s ``integrand_neutral_rho``."""
    return 2.0 * charged_pion_dnde(e, ENG_PI_NEUTRAL_RHO) / e


def reference(
    egam: float, erho: float, integrand: Callable[[float], float], endpoint: float
) -> float:
    """The outer boost in Python over scipy, as the repaired kernel poses it.

    The boosted branch transcribes the deleted Cython's window clipped at
    the integrand's rest-frame ``endpoint``; the rest branch removes its
    erroneous 1/E factor. Two independent QUADPACK bindings given the same
    integrand, interval and tolerances land on the same number. The inner
    pion spectra are shared — they are the ported Rust either way — so
    this oracle tests the outer integration and the branch structure, and
    nothing below them. :func:`log_energy_reference` is the one that does
    not share the kernel's interval.
    """
    if erho < MASS_RHO:
        return 0.0
    if erho - MASS_RHO < np.finfo(np.float64).eps:
        return egam * integrand(egam)

    beta = math.sqrt(1.0 - (MASS_RHO / erho) ** 2)
    gamma = erho / MASS_RHO
    emin = gamma * egam * (1.0 - beta)
    emax = min(gamma * egam * (1.0 + beta), endpoint)
    if emin >= emax:
        return 0.0
    pre = 0.5 / (beta * gamma)
    return pre * quad(integrand, emin, emax, epsabs=QUAD_EPSABS, epsrel=QUAD_EPSREL)[0]


def log_energy_reference(egam: float, erho: float, charged: bool) -> float:
    """The boosted spectrum integrated over ``u = ln E'``, MeV^-1.

    ``dE'/E' = du``, so the boost integral is ``1/(2 beta gamma)`` times
    ``int f(e^u) du`` over the window's intersection with the support,
    with the pi0 box's two edges handed to scipy as break points and a
    tolerance three decades tighter than the kernel's, the tightest at which
    scipy reports no roundoff against the inner quadrature's noise. It shares the pion
    spectra with the kernel and nothing else: not the variable, not the
    interval's parametrization, and not the break-point-free rule the
    kernel's ``qags`` applies across the box edges.
    """
    beta = math.sqrt(1.0 - (MASS_RHO / erho) ** 2)
    gamma = erho / MASS_RHO
    lower = math.log(gamma * egam * (1.0 - beta))
    upper = math.log(min(gamma * egam * (1.0 + beta), rho_rest_frame_endpoint(charged)))
    if lower >= upper:
        return 0.0
    integrand = charged_integrand if charged else neutral_integrand
    edges = PI0_BOX_EDGES if charged else ()
    points = [math.log(e) for e in edges if lower < math.log(e) < upper]
    value = quad(
        lambda u: math.exp(u) * integrand(math.exp(u)),
        lower,
        upper,
        points=points or None,
        epsabs=0.0,
        epsrel=1e-8,
        limit=200,
    )[0]
    return 0.5 / (beta * gamma) * value


def daughter_endpoint(rest_endpoint: float, energy: float, mass: float) -> float:
    """A rest-frame photon endpoint boosted into the parent's frame, MeV."""
    beta = math.sqrt(max(1.0 - (mass / energy) ** 2, 0.0))
    return rest_endpoint * (energy / mass) * (1.0 + beta)


def rho_rest_frame_endpoint(charged: bool) -> float:
    """The highest photon energy a rho at rest can emit, MeV."""
    from_charged_pion = daughter_endpoint(
        PHOTON_ENDPOINT_PIRF,
        ENG_PI_CHARGED_RHO if charged else ENG_PI_NEUTRAL_RHO,
        MASS_PI,
    )
    if not charged:
        return from_charged_pion
    from_neutral_pion = daughter_endpoint(MASS_PI0 / 2.0, ENG_PI0_CHARGED_RHO, MASS_PI0)
    return max(from_charged_pion, from_neutral_pion)


def probe_grid(erho: float, charged: bool, npoints: int = 60) -> np.ndarray:
    """A short log grid spanning a rho's whole photon support.

    Short on purpose: each point is a nested double quadrature. 60 points
    over five decades still resolves the box edge, the endpoint and the
    low-energy tail, which is what these comparisons need.
    """
    endpoint = daughter_endpoint(rho_rest_frame_endpoint(charged), erho, MASS_RHO)
    return np.geomspace(1e-2, endpoint * 0.999, npoints)


@pytest.fixture(
    params=[("charged", MASS_RHO * 2.0), ("neutral", MASS_RHO * 2.0)],
    ids=["charged_rho", "neutral_rho"],
)
def entry(request: pytest.FixtureRequest) -> EntryPoint:
    """One of the two entry points, with a parent energy in its support."""
    which, parent = request.param
    return (dnde_charged if which == "charged" else dnde_neutral), parent


class TestDispatchWiring:
    """Both entry points go through ``map_unary`` with their own wording.

    One assertion per contract branch. The branch-by-branch argument about
    ``map_unary`` itself is ``test/test_core_dispatch.py``'s; what is
    specific to these kernels is that they reached that helper at all, and
    with the ``"Photon energies"`` wording the deleted ``.pyx``'s
    ``assert`` used.
    """

    def test_a_scalar_returns_a_python_float(self, entry: EntryPoint) -> None:
        fn, parent = entry
        assert type(fn(100.0, parent)) is float

    def test_a_numpy_scalar_and_a_zero_dim_array_take_the_scalar_path(
        self, entry: EntryPoint
    ) -> None:
        fn, parent = entry
        want = fn(100.0, parent)
        assert type(fn(np.float64(100.0), parent)) is float
        assert fn(np.float64(100.0), parent) == want
        assert type(fn(np.array(100.0), parent)) is float
        assert fn(np.array(100.0), parent) == want

    def test_an_array_returns_a_fresh_float64_array_of_the_same_length(
        self, entry: EntryPoint
    ) -> None:
        fn, parent = entry
        grid = np.array([50.0, 100.0, 200.0])
        out = fn(grid, parent)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float64
        assert out.shape == grid.shape
        assert out is not grid

    def test_the_array_path_agrees_with_the_scalar_path_bit_for_bit(
        self, entry: EntryPoint
    ) -> None:
        # The two paths must be the same arithmetic, not merely close:
        # `map_unary` is one closure applied element by element.
        fn, parent = entry
        grid = np.array([13.0, 50.0, 137.0, 400.0])
        swept = np.asarray(fn(grid, parent))
        pointwise = np.array([fn(float(e), parent) for e in grid])
        assert swept.tobytes() == pointwise.tobytes()

    def test_a_sequence_is_accepted(self, entry: EntryPoint) -> None:
        fn, parent = entry
        assert np.asarray(fn([50.0, 100.0], parent)).shape == (2,)

    def test_an_empty_grid_returns_an_empty_grid(self, entry: EntryPoint) -> None:
        fn, parent = entry
        out = fn(np.array([], dtype=np.float64), parent)
        assert isinstance(out, np.ndarray)
        assert out.shape == (0,)

    def test_the_rank_message_is_the_cython_assert_verbatim(
        self, entry: EntryPoint
    ) -> None:
        # rules.md rule 9: the `assert`'s *type* becomes a ValueError, its
        # *message* does not change. The roster itself is pinned in
        # `test/test_core_dispatch.py`.
        fn, parent = entry
        with pytest.raises(ValueError) as excinfo:
            fn(np.ones((2, 2)), parent)
        assert str(excinfo.value) == DIMENSION_ERROR

    def test_a_non_float64_array_is_a_value_error(self, entry: EntryPoint) -> None:
        fn, parent = entry
        with pytest.raises(ValueError):
            fn(np.array([1, 2, 3], dtype=np.int64), parent)

    def test_a_non_number_is_a_type_error(self, entry: EntryPoint) -> None:
        fn, parent = entry
        with pytest.raises(TypeError) as excinfo:
            fn(None, parent)
        assert str(excinfo.value) == TYPE_ERROR

    def test_both_arguments_are_accepted_by_keyword(self) -> None:
        # The twin was a `def` and took both by keyword; the `text_signature`
        # PyO3 advertises is a claim, so it is exercised rather than read.
        assert dnde_charged(
            photon_energies=100.0, rho_energy=MASS_RHO * 2.0
        ) == dnde_charged(100.0, MASS_RHO * 2.0)
        assert dnde_neutral(
            photon_energies=100.0, rho_energy=MASS_RHO * 2.0
        ) == dnde_neutral(100.0, MASS_RHO * 2.0)


class TestWrapperAndPublicApi:
    """The swap is wired all the way out, and the twin is gone."""

    def test_the_private_wrappers_return_the_core_kernels_values(self) -> None:
        erho, grid = MASS_RHO * 2.0, np.array([13.0, 100.0, 400.0])
        for wrapped, kernel in (
            (wrapper.dnde_photon_charged_rho, dnde_charged),
            (wrapper.dnde_photon_neutral_rho, dnde_neutral),
        ):
            assert (
                np.asarray(wrapped(grid, erho)).tobytes()
                == np.asarray(kernel(grid, erho)).tobytes()
            )

    def test_the_public_spectra_names_resolve_to_the_same_functions(self) -> None:
        assert spectra.dnde_photon_charged_rho is wrapper.dnde_photon_charged_rho
        assert spectra.dnde_photon_neutral_rho is wrapper.dnde_photon_neutral_rho
        assert "dnde_photon_charged_rho" in spectra.__all__
        assert "dnde_photon_neutral_rho" in spectra.__all__

    def test_the_cython_twin_is_gone_from_the_tree(self) -> None:
        """rules.md rule 1, in its unqualified form.

        Nothing cimported ``_rho.pyx``, so it is not a Phase 06 capi
        survivor and the swap PR removes the module outright rather than
        only its ``def``s.

        Asserted against the **source tree and the build declaration**,
        not against importability. A stale `_rho.cpython-*.so` from an
        older build sits in the package directory until someone cleans it
        and imports perfectly well (``docs/agents/environment.md``), so an
        ``ImportError`` assertion tests whoever last ran ``pip install``
        rather than this change.
        """
        package = REPO_ROOT / "hazma" / "spectra" / "_photon"
        for suffix in (".pyx", ".pxd", ".pyi"):
            assert not (package / f"_rho{suffix}").exists(), suffix
        # Nothing can rebuild it either: the build declares no Cython
        # step at all, which `test/test_no_cython_remains.py` asserts
        # tree-wide against `[build-system] requires`. That replaced the
        # per-module `setup.py` extension-list check this test used to
        # carry, which went with `setup.py` at the maturin cutover.

    def test_the_nbody_dispatch_table_reaches_the_ported_entry_points(self) -> None:
        # `_nbody.py` maps final-state names to spectrum functions; a swap
        # that repointed the wrapper but not this table would leave the
        # N-body path on a module that no longer exists.
        assert _nbody._dnde_photon_dict["rho"] is wrapper.dnde_photon_charged_rho
        assert _nbody._dnde_photon_dict["rho0"] is wrapper.dnde_photon_neutral_rho


class TestAgainstAnIndependentBoostIntegral:
    """The outer integral, recomputed through scipy's QUADPACK binding.

    Not a re-implementation of the Rust: :func:`reference` transcribes the
    deleted ``.pyx``'s three branches, with the window clipped at the
    support as the kernel clips it, and hands the same integrand and the
    same ``epsabs``/``epsrel`` to ``scipy.integrate.quad``. What it can
    catch is the branch structure, the boost window, the
    ``1/(2 beta gamma)`` prefactor, which daughter energies go into which
    integrand — against an integrator that shares no code with the port.
    Because it shares the kernel's interval it cannot see a wrong clip;
    :class:`TestTheBoostedTail` can. What it cannot catch is an error in the *pion* kernels,
    which both sides call; those are Task 4.4's, gated by its own module
    and by the corpus.
    """

    @pytest.mark.parametrize("erho", RHO_ENERGIES, ids=lambda e: f"{e / MASS_RHO:.3g}m")
    @pytest.mark.parametrize("charged", [True, False], ids=["charged", "neutral"])
    def test_a_swept_grid_matches(self, erho: float, charged: bool) -> None:
        fn = dnde_charged if charged else dnde_neutral
        integrand = charged_integrand if charged else neutral_integrand
        endpoint = rho_rest_frame_endpoint(charged)
        grid = probe_grid(erho, charged)
        got = np.asarray(fn(grid, erho))
        want = np.array([reference(float(e), erho, integrand, endpoint) for e in grid])
        peak = float(np.abs(want).max())
        np.testing.assert_allclose(
            got,
            want,
            rtol=INDEPENDENT_BUDGET,
            atol=INDEPENDENT_BUDGET * peak,
            err_msg=(
                f"E_rho = {erho} MeV: the port and scipy's QUADPACK "
                f"disagree beyond {INDEPENDENT_BUDGET:.0e} of the peak "
                f"({peak:.6e}) on the same integrand at the same "
                f"tolerances. Task 4.5 measured 1.5e-13 against the Cython "
                f"on the corpus's own points, so a failure here is a "
                f"defect in the outer integral rather than method error."
            ),
        )

    def test_the_budget_is_not_vacuous(self) -> None:
        # A wrong prefactor is the cheapest way to be wrong here, and it
        # must be caught by the same comparison the test above makes.
        erho = MASS_RHO * 2.0
        endpoint = rho_rest_frame_endpoint(charged=True)
        grid = probe_grid(erho, charged=True, npoints=12)
        want = np.array(
            [reference(float(e), erho, charged_integrand, endpoint) for e in grid]
        )
        # `1/(beta gamma)` instead of `1/(2 beta gamma)`.
        mutated = 2.0 * want
        peak = float(np.abs(want).max())
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                mutated, want, rtol=INDEPENDENT_BUDGET, atol=INDEPENDENT_BUDGET * peak
            )

    def test_the_daughter_energies_are_not_interchangeable(self) -> None:
        # The charged rho's two daughters differ by 1.6 MeV, and swapping
        # which mass each integrand uses is a transcription error a
        # relative budget would have to be O(1) to miss.
        erho = MASS_RHO * 2.0
        endpoint = rho_rest_frame_endpoint(charged=True)
        grid = probe_grid(erho, charged=True, npoints=12)
        want = np.array(
            [reference(float(e), erho, charged_integrand, endpoint) for e in grid]
        )

        def swapped(e: float) -> float:
            return (
                charged_pion_dnde(e, ENG_PI0_CHARGED_RHO)
                + neutral_pion_dnde(e, ENG_PI_CHARGED_RHO)
            ) / e

        # Clipped at the unswapped endpoint, which cuts the swapped box's
        # last MeV; the two differ by far more than that anyway.
        got = np.array([reference(float(e), erho, swapped, endpoint) for e in grid])
        assert not np.allclose(got, want, rtol=1e-6)


class TestTheBoostedTail:
    """The top of the boosted spectrum, which the whole-window rule lost.

    Near the lab-frame endpoint the boost window spans decades while the
    integrand survives only on ``[gamma E (1 - beta), E'_max]`` at its
    bottom, and a rule over the whole window can place every node above
    that sliver. With the charged-pion angular integral already repaired,
    that returned exactly zero over the top 0.2% of the spectrum at
    ``1.05 m_rho`` and the top 46% at ``10 m_rho``. The kernel now clips
    the window at ``E'_max``; these tests hold it to references that do
    not share its interval, and to an invariant no quadrature choice can
    satisfy by accident.
    """

    #: How far the kernel may sit from :func:`log_energy_reference`. The
    #: kernel's outer and inner quadratures both request ``epsrel = 1e-5``
    #: and the reference cannot resolve the inner one's noise, so the two
    #: agree to that noise and no better: 5.3e-6 worst, measured over the
    #: grid below. Five times the requested tolerance; the defect this
    #: gates is a zero where the reference is positive.
    LOG_REFERENCE_BUDGET = 5e-5

    #: How far ``int E dN/dE`` may sit from ``gamma`` times its rest-frame
    #: value. Measured 8.3e-7 worst over the four parametrizations; the
    #: whole-window rule missed it by 1.3e-3 for the charged rho at
    #: ``2 m_rho``, by 0.35 at ``10 m_rho``, and by 6.3e-3 for the neutral
    #: rho at ``10 m_rho``.
    ENERGY_BUDGET = 1e-5

    #: Fractions of the lab-frame endpoint the tail tests probe. The top
    #: one sits inside the 0.2% the whole-window rule lost at ``1.05 m_rho``.
    FRACTIONS = (1e-3, 0.1, 0.3, 0.6, 0.9, 0.99, 0.9999)

    @pytest.mark.parametrize("factor", [1.05, 2.0, 10.0])
    @pytest.mark.parametrize("charged", [True, False], ids=["charged", "neutral"])
    def test_the_tail_matches_a_log_energy_reference(
        self, charged: bool, factor: float
    ) -> None:
        fn = dnde_charged if charged else dnde_neutral
        erho = factor * MASS_RHO
        lab_endpoint = daughter_endpoint(
            rho_rest_frame_endpoint(charged), erho, MASS_RHO
        )
        grid = np.array(self.FRACTIONS) * lab_endpoint
        want = np.array([log_energy_reference(float(e), erho, charged) for e in grid])
        assert np.all(want > 0.0)
        np.testing.assert_allclose(
            fn(grid, erho), want, rtol=self.LOG_REFERENCE_BUDGET, atol=0.0
        )

    def test_the_whole_window_loses_the_tail_the_reference_keeps(self) -> None:
        # The defect, reproduced through scipy rather than the kernel: the
        # same rule over the unclipped window returns a converged zero
        # where the log-energy reference, and now the kernel, do not.
        erho = 10.0 * MASS_RHO
        endpoint = rho_rest_frame_endpoint(charged=True)
        egam = 0.9 * daughter_endpoint(endpoint, erho, MASS_RHO)
        assert reference(egam, erho, charged_integrand, math.inf) == 0.0
        assert log_energy_reference(egam, erho, charged=True) > 0.0
        assert dnde_charged(egam, erho) > 0.0

    @pytest.mark.parametrize("factor", [2.0, 10.0])
    @pytest.mark.parametrize("charged", [True, False], ids=["charged", "neutral"])
    def test_the_lab_frame_photon_energy_is_gamma_times_the_rest_frame_energy(
        self, charged: bool, factor: float
    ) -> None:
        """``int E dN/dE dE = gamma int E' f(E') dE'``, exactly.

        Integrating the flat boost kernel against ``E`` over the lab
        energies that see a given ``E'`` gives ``gamma E'``: an isotropic
        rest-frame source carries its energy into the lab multiplied by
        ``gamma``. The top of the spectrum carries the most energy, so this
        is the invariant a lost tail violates most.
        """
        fn = dnde_charged if charged else dnde_neutral
        integrand = charged_integrand if charged else neutral_integrand
        endpoint = rho_rest_frame_endpoint(charged)
        edges = PI0_BOX_EDGES if charged else ()
        rest = quad(
            lambda e: e * e * integrand(e),
            0.0,
            endpoint,
            points=edges or None,
            epsabs=0.0,
            epsrel=1e-8,
            limit=200,
        )[0]

        erho = factor * MASS_RHO
        gamma = erho / MASS_RHO
        beta = math.sqrt(1.0 - 1.0 / (gamma * gamma))
        lab_endpoint = daughter_endpoint(endpoint, erho, MASS_RHO)
        # The boosted box edges are kinks of the lab spectrum.
        kinks = sorted(
            x * gamma * (1.0 + sign * beta)
            for x in (*edges, endpoint)
            for sign in (-1.0, 1.0)
            if x * gamma * (1.0 + sign * beta) < lab_endpoint
        )
        lab = quad(
            lambda e: e * float(fn(e, erho)),
            0.0,
            lab_endpoint,
            points=kinks,
            epsabs=0.0,
            epsrel=1e-6,
            limit=200,
        )[0]
        assert lab / (gamma * rest) == pytest.approx(1.0, rel=self.ENERGY_BUDGET)


class TestPhysics:
    """Statements about the spectra that owe nothing to the Cython."""

    @pytest.mark.parametrize("erho", [0.0, 1.0, MASS_RHO * 0.5, MASS_RHO - 1e-9])
    def test_a_rho_below_its_rest_mass_radiates_exactly_nothing(
        self, erho: float
    ) -> None:
        # Exactly zero, not nearly: the corpus compares with atol 0, so a
        # port returning 1e-300 here would fail it.
        assert dnde_charged(100.0, erho) == 0.0
        assert dnde_neutral(100.0, erho) == 0.0

    @pytest.mark.parametrize("charged", [True, False], ids=["charged", "neutral"])
    def test_the_rest_frame_branch_returns_the_daughter_spectra(
        self, charged: bool
    ) -> None:
        """The spectrum sums daughter yields with units MeV^-1."""
        fn = (
            spectra.dnde_photon_charged_rho
            if charged
            else spectra.dnde_photon_neutral_rho
        )
        grid = np.array([0.01, 13.0, 50.0, 200.0, 300.0, MASS_RHO])
        if charged:
            want = np.array(
                [
                    charged_pion_dnde(e, ENG_PI_CHARGED_RHO)
                    + neutral_pion_dnde(e, ENG_PI0_CHARGED_RHO)
                    for e in grid
                ]
            )
        else:
            want = np.array(
                [2.0 * charged_pion_dnde(e, ENG_PI_NEUTRAL_RHO) for e in grid]
            )
        # Only the division and multiplication by E differ from the sum.
        np.testing.assert_allclose(
            fn(grid, MASS_RHO), want, rtol=2 * np.finfo(float).eps, atol=0.0
        )
        np.testing.assert_array_equal(
            fn(grid, MASS_RHO), [fn(float(e), MASS_RHO) for e in grid]
        )

        next_double = float(np.nextafter(MASS_RHO, np.inf))
        assert next_double - MASS_RHO > np.finfo(np.float64).eps
        for e in (13.0, 50.0, 200.0, 300.0):
            # A decade above the outer quad's 1e-5, allowing near-zero
            # boost-window rounding, and far below the old factor of E.
            assert fn(e, next_double) / fn(e, MASS_RHO) == pytest.approx(1.0, rel=1e-4)

    def test_the_rest_frame_neutral_pion_box_has_two_photons_per_decay(self) -> None:
        """Subtracting the charged daughter leaves the pi0 -> gamma gamma box."""
        beta = math.sqrt(1.0 - (MASS_PI0 / ENG_PI0_CHARGED_RHO) ** 2)
        lo, hi = (
            ENG_PI0_CHARGED_RHO * (1.0 - beta) / 2,
            ENG_PI0_CHARGED_RHO * (1.0 + beta) / 2,
        )
        grid = np.linspace(lo, hi, 11)[1:-1]
        residual = dnde_charged(grid, MASS_RHO) - np.array(
            [charged_pion_dnde(e, ENG_PI_CHARGED_RHO) for e in grid]
        )
        # 0.98823 is BR(pi0 -> gamma gamma), the captured PDG branching
        # fraction used by photon_pion.rs.
        # The pion kernel retains its captured f32 box height, whose
        # relative rounding error is at most 2^-24; allow 1e-7.
        np.testing.assert_allclose(
            residual * (hi - lo), 2 * 0.98823, rtol=1e-7, atol=0.0
        )

    @pytest.mark.parametrize("charged", [True, False], ids=["charged", "neutral"])
    def test_the_spectrum_vanishes_above_its_kinematic_endpoint(
        self, charged: bool
    ) -> None:
        # The endpoint is the daughters' rest-frame photon endpoint boosted
        # twice: out of the pion frame into the rho's, and out of the rho's
        # into the lab.
        fn = dnde_charged if charged else dnde_neutral
        erho = MASS_RHO * 2.0
        endpoint = daughter_endpoint(rho_rest_frame_endpoint(charged), erho, MASS_RHO)
        assert fn(endpoint * 1.01, erho) == 0.0
        assert fn(endpoint * 10.0, erho) == 0.0

    def test_the_pi0_box_edge_reverses_which_rho_is_brighter(self) -> None:
        """The sharpest structure either spectrum has, at rest.

        Below :data:`PI0_BOX_LOWER_EDGE` the charged rho has only its one
        charged pion against the neutral rho's two, so it is half as
        bright; above the edge the pi0's ``gamma gamma`` box dominates by
        more than a factor of two. An implementation that swapped the
        daughters, dropped the factor of two, or used the wrong pion mass
        in the box fails one half or the other.
        """
        for e in (1.0, 5.0, 10.0, 12.0):
            assert e < PI0_BOX_LOWER_EDGE
            ratio = charged_integrand(e) / neutral_integrand(e)
            low, high = BELOW_BOX_RATIO_BAND
            assert low < ratio < high, f"below the box, at {e} MeV: {ratio}"
        for e in (13.0, 20.0, 50.0, 100.0):
            assert e > PI0_BOX_LOWER_EDGE
            assert charged_integrand(e) > 2.0 * neutral_integrand(
                e
            ), f"the pi0 box should dominate at {e} MeV"

    @pytest.mark.parametrize("charged", [True, False], ids=["charged", "neutral"])
    def test_boosting_the_rho_pushes_flux_past_the_rest_frame_endpoint(
        self, charged: bool
    ) -> None:
        """The direction of the boost, asserted where it cannot be argued.

        Above the rho's *rest-frame* endpoint a rho at rest emits exactly
        nothing; a boosted one emits there and its own endpoint rises like
        ``gamma (1 + beta)``. An implementation that applied the boost
        backwards — ``1 - beta`` for ``1 + beta``, or the reciprocal
        prefactor — fails this regardless of the source spectrum's shape,
        which is what makes it the right invariant to pin. See
        :meth:`test_the_charged_rho_plateau_falls_as_the_box_widens` for
        the shape-dependent statement.
        """
        fn = dnde_charged if charged else dnde_neutral
        rest_endpoint = rho_rest_frame_endpoint(charged)
        probe = rest_endpoint * 1.5
        assert fn(probe, MASS_RHO * (1.0 + 1e-12)) == 0.0

        previous_endpoint = rest_endpoint
        for factor in (1.5, 2.0, 4.0, 8.0):
            erho = factor * MASS_RHO
            endpoint = daughter_endpoint(rest_endpoint, erho, MASS_RHO)
            assert endpoint > previous_endpoint, f"endpoint at {factor} m_rho"
            assert fn(probe, erho) > 0.0, f"flux past the rest endpoint at {factor}"
            previous_endpoint = endpoint

    def test_the_charged_rho_plateau_falls_as_the_box_widens(self) -> None:
        """The pi0 box carries fixed area, so boosting lowers its height.

        The charged rho's spectrum above :data:`PI0_BOX_LOWER_EDGE` is
        dominated by the pi0's ``gamma gamma`` box — a fixed number of
        photons spread over a window that widens like ``gamma (1 + beta)``
        — so at a fixed energy inside the plateau the spectrum falls
        monotonically with the parent energy.

        The **neutral** rho does *not* do this, and the asymmetry is the
        physics rather than a wrinkle: its only source is the charged
        pion's radiative spectrum, which is soft and steeply falling, so
        at these energies a boost moves more flux *into* the probe than
        out of it. Both directions are asserted so that an implementation
        which swapped the two entry points fails here.
        """
        e = 40.0
        charged_values = [dnde_charged(e, f * MASS_RHO) for f in (1.5, 2.0, 4.0, 8.0)]
        neutral_values = [dnde_neutral(e, f * MASS_RHO) for f in (1.5, 2.0, 4.0, 8.0)]
        assert all(v > 0.0 for v in charged_values + neutral_values)
        assert charged_values == sorted(charged_values, reverse=True), charged_values
        assert neutral_values == sorted(neutral_values), neutral_values

    @pytest.mark.parametrize("charged", [True, False], ids=["charged", "neutral"])
    def test_the_spectrum_is_non_negative_everywhere_it_is_defined(
        self, charged: bool
    ) -> None:
        # A spectrum is a probability density. The boost integral is a
        # positive kernel against non-negative daughter spectra, so any
        # negative value is a sign error rather than rounding.
        fn = dnde_charged if charged else dnde_neutral
        erho = MASS_RHO * 2.0
        values = np.asarray(fn(probe_grid(erho, charged), erho))
        assert np.all(values >= 0.0)
        assert np.all(np.isfinite(values))

    def test_a_nan_rho_energy_propagates_rather_than_reading_as_below_threshold(
        self,
    ) -> None:
        # Both guards compare with `<`, false for NaN, so a NaN parent
        # reaches the quadrature and comes back NaN. Worth pinning because
        # "returns 0 below threshold" does not describe it.
        assert math.isnan(dnde_charged(100.0, math.nan))
        assert math.isnan(dnde_neutral(100.0, math.nan))
