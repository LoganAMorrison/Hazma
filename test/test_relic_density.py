import unittest
import warnings
from collections.abc import Callable, Iterator
from typing import Any, ClassVar

from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.special import k1, kn

import hazma.vector_mediator._gev.thermal_cross_section as gev_site
from hazma.parameters import omega_h2_cdm
from hazma.relic_density import relic_density
from hazma.relic_density._thermal_functions import (
    thermal_cross_section,
    thermal_cross_section_integrand,
)
from hazma.scalar_mediator import HiggsPortal
from hazma.vector_mediator import KineticMixing, VectorMediatorGeV

warnings.filterwarnings("ignore")


class ToyModel:
    def __init__(self, mx: float, sigmav: float) -> None:
        self.mx = mx
        self.sigmav = sigmav

    def thermal_cross_section(self, _: float) -> float:
        """Compute the thermal cross section at a given mass-to-temperature ratio.

        Parameters
        ----------
        x: float
            DM mass over temperature.

        Returns
        -------
        sigmav: float
            Dark matter thermmal cross section.
        """
        return self.sigmav


class TestRelicDensity(unittest.TestCase):
    def setUp(self) -> None:
        mx1, sigmav1 = 10.313897683787216e3, 1.966877938634266e-15
        mx2, sigmav2 = 104.74522360006331e3, 1.7597967261428258e-15
        mx3, sigmav3 = 1063.764854316313e3, 1.837766552668581e-15
        mx4, sigmav4 = 10000.0e3, 1.8795945459427076e-15

        self.models = [
            ToyModel(mx1, sigmav1),
            ToyModel(mx2, sigmav2),
            ToyModel(mx3, sigmav3),
            ToyModel(mx4, sigmav4),
        ]

    def test_relic_density(self) -> None:
        for model in self.models:
            # check that semi-analytical esult is within 6% omega_h2_cdm
            rd_semianalytic = relic_density(model, semi_analytic=True)
            assert_allclose(rd_semianalytic, omega_h2_cdm, rtol=0.06)

            # check that semi-analytical esult is within 0.5% omega_h2_cdm
            rd_numeric = relic_density(model, semi_analytic=False)
            assert_allclose(rd_numeric, omega_h2_cdm, rtol=0.005)


class TestMediatorRelicDensity(unittest.TestCase):
    """End-to-end relic densities through the mediator ``thermal_cross_section``.

    ``ToyModel`` above short-circuits `hazma.relic_density`'s only coupling
    to the compiled layer — it supplies a constant ``sigmav`` — so nothing
    else in the suite drives `relic_density` through a real
    ``thermal_cross_section``.  These six scenarios do, and they pin the
    values the converged kernel produces.  They held the pre-port Cython's
    values until ``B5`` (cython-to-rust Task 5.3 captured those at
    ``14f1c66``); that quadrature never converged, so the numbers it
    produced pinned a defect rather than a physical prediction.

    The six model points are the ones `test/parity/cases.py` uses for the
    cross-section corpus, so a failure here and a failure there implicate
    the same kernels.
    """

    #: name -> (mx, mmed, coupling) for `HiggsPortal` / `KineticMixing`.
    SCALAR_POINTS: ClassVar = {
        "open_resonance": dict(mx=100.0, ms=300.0, gsxx=1.0, stheta=1e-1),
        "narrow_resonance": dict(mx=200.0, ms=550.0, gsxx=1.0, stheta=1e-4),
        "closed_resonance": dict(mx=300.0, ms=200.0, gsxx=1.0, stheta=1e-2),
    }
    VECTOR_POINTS: ClassVar = {
        "open_resonance": dict(mx=100.0, mv=300.0, gvxx=1.0, eps=1e-1),
        "narrow_resonance": dict(mx=200.0, mv=550.0, gvxx=1.0, eps=1e-4),
        "closed_resonance": dict(mx=300.0, mv=200.0, gvxx=1.0, eps=1e-2),
    }

    #: Relic densities, (semi_analytic, boltzmann).  Dimensionless
    #: (Omega h^2).  Not physical abundances — these model points were chosen
    #: to stress the cross sections, not to sit on the observed value.
    #:
    #: Derived from the converged ``thermal_cross_section`` (roster entry
    #: ``B5``), not from the pre-port Cython: the shipped kernels returned
    #: their integrator's initial partition, so the values these replace
    #: were wrong by up to 100% on <sigma v> and, since freeze-out
    #: abundance goes as 1/<sigma v>, by up to two orders of magnitude
    #: here.  The
    #: closed-resonance points are where the old quadrature missed most of
    #: the integrand's mass: they fall 91.9% and 99.9%.
    PINNED: ClassVar = {
        "scalar.open_resonance": (26.667787923392634, 34.42028079003851),
        "scalar.narrow_resonance": (6905.347000480099, 8282.819772041152),
        "scalar.closed_resonance": (9.359827513207474e-08, 9.849973690303772e-08),
        "vector.open_resonance": (6.142001344063203e-07, 6.408469699348235e-07),
        "vector.narrow_resonance": (0.3081076863117194, 0.3277204010101092),
        "vector.closed_resonance": (5.6437976262658464e-09, 5.9113937835982655e-09),
    }

    #: Solver tolerances for the Boltzmann pins above.  *Not* the
    #: `relic_density` defaults (``rtol=1e-5, atol=1e-3``) — see
    #: `BOLTZMANN_RTOL` for why the defaults cannot be pinned portably.
    BOLTZMANN_SOLVER_RTOL = 1e-10
    BOLTZMANN_SOLVER_ATOL = 1e-8

    #: The semi-analytic path is a closed-form composition of
    #: `thermal_cross_section` with no adaptive solver in it, so whatever
    #: that kernel's own error is arrives essentially undamped.  Since
    #: ``B5`` the kernel subdivides until it meets ``epsrel = 1.49e-8``,
    #: which is four decades looser than the <= 2.06e-14 port drift this
    #: budget used to be set from, and a platform whose libm steers
    #: QUADPACK to a different accepted partition may land anywhere inside
    #: it.  Measured against `test/parity/thermal_reference.py` — scipy's
    #: QUADPACK on the same integrand at ``epsrel = 1e-12`` — the repaired
    #: kernel is within 3.6e-9 at all 540 corpus positions it integrates,
    #: but 1.49e-8 is the bound that has to hold off this platform.  1e-6
    #: is ~6.7x that bound: still ~10,000x
    #: tighter than the smallest shift ``B5`` itself produced (0.071%, at
    #: ``scalar.open_resonance``), so a real kernel regression cannot hide
    #: under it.
    SEMI_ANALYTIC_RTOL = 1e-6

    #: The Boltzmann path integrates the same kernel with
    #: `scipy.integrate.solve_ivp`, whose adaptive stepping does not
    #: depend continuously on its input: a last-bit change in
    #: `thermal_cross_section` flips a step-acceptance decision and the
    #: whole step sequence differs.  The answer then moves at the
    #: *solver's* tolerance rather than the kernel's, which makes a pin
    #: taken at the `relic_density` default ``rtol=1e-5`` both loose and
    #: platform-dependent — cython-to-rust Task 5.3 measured 3.82e-5
    #: pre-port vs ported on macOS/arm64 and CI then found 1.22e-4 for
    #: the same comparison on Linux/glibc, because a different libm
    #: perturbs the step sequence differently.
    #:
    #: So these pins are taken at ``rtol=1e-10`` instead, where the
    #: physics dominates the step noise: the same comparison is 1.93e-8
    #: at worst (`scalar.open_resonance`), a ~2000x improvement in what
    #: the pin can resolve, for ~1.5 s of extra solve time across the six
    #: scenarios.  1e-5 is ~500x that measured worst case, leaving room
    #: for the platform spread while still catching any kernel error
    #: large enough to matter physically.
    BOLTZMANN_RTOL = 1e-5

    def _models(self) -> Iterator[tuple[str, object]]:
        for name, kwargs in self.SCALAR_POINTS.items():
            yield f"scalar.{name}", HiggsPortal(**kwargs)
        for name, kwargs in self.VECTOR_POINTS.items():
            yield f"vector.{name}", KineticMixing(**kwargs)

    def test_semi_analytic_matches_converged_kernel(self) -> None:
        for name, model in self._models():
            with self.subTest(model=name):
                assert_allclose(
                    relic_density(model, semi_analytic=True),
                    self.PINNED[name][0],
                    rtol=self.SEMI_ANALYTIC_RTOL,
                )

    def test_boltzmann_matches_converged_kernel(self) -> None:
        for name, model in self._models():
            with self.subTest(model=name):
                assert_allclose(
                    relic_density(
                        model,
                        semi_analytic=False,
                        rtol=self.BOLTZMANN_SOLVER_RTOL,
                        atol=self.BOLTZMANN_SOLVER_ATOL,
                    ),
                    self.PINNED[name][1],
                    rtol=self.BOLTZMANN_RTOL,
                )


class TestThermalQuadratureConverges(unittest.TestCase):
    r"""The two pure-Python ``thermal_cross_section`` sites resolve their integral.

    Both pass ``epsabs=0.0`` so that the relative criterion is the one
    that binds.  Neither is reachable from the parity corpus or from
    `TestMediatorRelicDensity`: those go through
    ``hazma._core``'s scalar and vector kernels, and the mediator models
    define their own ``thermal_cross_section``, which
    `hazma.relic_density._thermal_functions.thermal_cross_section`
    short-circuits to.  Without the tests below, reverting ``epsabs`` at
    either Python site would leave the whole suite green.

    Each site is compared against the same integrand integrated to
    ``epsrel = 1e-12``, and each test also asserts that scipy's default
    ``epsabs`` would *not* pass — the assertion that makes this a
    regression test for the tolerance rather than a generic accuracy
    check.  Measured worst relative error at the default, over the grid
    below: **0.765** for the generic fallback (``scalar.open`` at
    ``x = 5``) and **3.6e-3** for the GeV vector site.
    """

    #: Budget for "the site agrees with a converged integral".  Both sites
    #: run at scipy's default ``epsrel = 1.49e-8``, so that — not the
    #: measured figure — is what a pin here has to survive on a platform
    #: whose libm steers QUADPACK to a different accepted partition.
    #: 1e-6 is ~67x it, and still two decades under the smallest error the
    #: default ``epsabs`` produces anywhere on this grid (2.2e-5).
    CONVERGED_RTOL = 1e-6

    #: ``x = mx/T`` sample points.  Capped below 25 deliberately: both
    #: sites integrate to ``50/x``, which reaches the lower limit of 2 at
    #: exactly ``x = 25`` and inverts above it, so there is no integral to
    #: check there.  That is a separate, pre-existing defect —
    #: ``docs/followups/todo/thermal-fallback-upper-limit-collapses-at-x-25.md``.
    X_GRID: ClassVar = (1.0, 5.0, 10.0, 20.0, 24.0)

    @staticmethod
    def _converged(integrand: Callable[..., float], x: float, args: tuple) -> float:
        """``<sigma v>(x)`` from the same integrand at ``epsrel = 1e-12``."""
        prefactor = x / (2.0 * kn(2, x)) ** 2
        value, _ = quad(
            integrand,
            2.0,
            50.0 / x,
            args=args,
            points=[2.0],
            epsabs=0.0,
            epsrel=1e-12,
            limit=200,
        )
        return prefactor * value

    @staticmethod
    def _at_scipy_defaults(
        integrand: Callable[..., float], x: float, args: tuple
    ) -> float:
        """The same integral with no tolerances passed: what the sites did."""
        prefactor = x / (2.0 * kn(2, x)) ** 2
        value, _ = quad(integrand, 2.0, 50.0 / x, args=args, points=[2.0])
        return prefactor * value

    def test_generic_fallback_converges(self) -> None:
        """`_thermal_functions.thermal_cross_section`, the no-kernel path."""

        class NoThermalCrossSection:
            """A model the fallback cannot short-circuit past.

            `thermal_cross_section` defers to ``model.thermal_cross_section``
            whenever the model defines one, and every mediator model does,
            so reaching the generic path needs a model that does not.
            """

            def __init__(self, inner: HiggsPortal | KineticMixing) -> None:
                self._inner = inner
                self.mx: float = inner.mx

            def annihilation_cross_sections(self, e_cm: float) -> dict:
                return self._inner.annihilation_cross_sections(e_cm)

        points = {
            "scalar.open": HiggsPortal(mx=100.0, ms=300.0, gsxx=1.0, stheta=1e-1),
            "scalar.closed": HiggsPortal(mx=300.0, ms=200.0, gsxx=1.0, stheta=1e-2),
            "vector.open": KineticMixing(mx=100.0, mv=300.0, gvxx=1.0, eps=1e-1),
            "vector.closed": KineticMixing(mx=300.0, mv=200.0, gvxx=1.0, eps=1e-2),
        }
        worst_default = 0.0
        for name, inner in points.items():
            model = NoThermalCrossSection(inner)
            for x in self.X_GRID:
                with self.subTest(model=name, x=x):
                    reference = self._converged(
                        thermal_cross_section_integrand, x, (x, model)
                    )
                    assert_allclose(
                        thermal_cross_section(x, model),
                        reference,
                        rtol=self.CONVERGED_RTOL,
                    )
                    default = self._at_scipy_defaults(
                        thermal_cross_section_integrand, x, (x, model)
                    )
                    worst_default = max(
                        worst_default, abs(default - reference) / abs(reference)
                    )
        assert worst_default > self.CONVERGED_RTOL, (
            "scipy's default epsabs now resolves this integral, so `epsabs=0.0` "
            f"at the call site pins nothing (worst relative error {worst_default:.2e} "
            f"against a budget of {self.CONVERGED_RTOL:.0e})"
        )

    def test_gev_vector_site_converges(self) -> None:
        """The `VectorMediatorGeV.relic_density` closure.

        The closure is built inside the method and handed to
        `hazma.relic_density.relic_density`, so it is reached by
        intercepting that call rather than by solving the Boltzmann
        equation — which for this model returns ``nan`` for reasons that
        predate the tolerance fix (see the follow-up cited on `X_GRID`).
        """
        model = VectorMediatorGeV(
            mx=5e3,
            mv=2e3,
            gvxx=1.0,
            gvuu=3.0,
            gvdd=1.0,
            gvss=-1.0,
            gvee=0.0,
            gvmumu=0.0,
            gvveve=0.0,
            gvvmvm=0.0,
            gvvtvt=0.0,
        )

        captured: dict[str, Any] = {}
        original = gev_site.rd
        try:
            gev_site.rd = lambda model, **_: captured.setdefault("model", model)
            model.relic_density(semi_analytic=True, three_body=False, four_body=False)
        finally:
            gev_site.rd = original
        site = captured["model"].thermal_cross_section

        # The integrand the closure built, rebuilt here from the same
        # channel filter so the reference integrates the same function.
        channel_fns = {
            key: fn
            for key, fn in model.annihilation_cross_section_funcs().items()
            if key in gev_site.TWO_BODY
        }

        def integrand(z: float, x: float) -> float:
            sigma = sum(fn(model.mx * z) for fn in channel_fns.values())
            return sigma * z**2 * (z**2 - 4.0) * k1(x * z)

        worst_default = 0.0
        for x in self.X_GRID:
            with self.subTest(x=x):
                reference = self._converged(integrand, x, (x,))
                assert_allclose(site(x), reference, rtol=self.CONVERGED_RTOL)
                default = self._at_scipy_defaults(integrand, x, (x,))
                worst_default = max(
                    worst_default, abs(default - reference) / abs(reference)
                )
        assert worst_default > self.CONVERGED_RTOL, (
            "scipy's default epsabs now resolves this integral, so `epsabs=0.0` "
            f"at the call site pins nothing (worst relative error {worst_default:.2e} "
            f"against a budget of {self.CONVERGED_RTOL:.0e})"
        )
