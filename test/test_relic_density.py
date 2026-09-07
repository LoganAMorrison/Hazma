import unittest
import warnings
from collections.abc import Iterator
from typing import ClassVar

from numpy.testing import assert_allclose

from hazma.parameters import omega_h2_cdm
from hazma.relic_density import relic_density
from hazma.scalar_mediator import HiggsPortal
from hazma.vector_mediator import KineticMixing

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

    def test_semi_analytic_matches_pre_port(self) -> None:
        for name, model in self._models():
            with self.subTest(model=name):
                assert_allclose(
                    relic_density(model, semi_analytic=True),
                    self.PINNED[name][0],
                    rtol=self.SEMI_ANALYTIC_RTOL,
                )

    def test_boltzmann_matches_pre_port(self) -> None:
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
