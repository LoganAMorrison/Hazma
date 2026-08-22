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
    values the pre-port Cython produced (cython-to-rust Task 5.3,
    captured at ``14f1c66``, the commit before the Phase 05 swaps).

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

    #: Pre-port relic densities, (semi_analytic, boltzmann).  Dimensionless
    #: (Omega h^2).  Not physical abundances — these model points were chosen
    #: to stress the cross sections, not to sit on the observed value.
    PINNED: ClassVar = {
        "scalar.open_resonance": (26.68685642281613, 34.4452749510159),
        "scalar.narrow_resonance": (6767.372700752017, 8088.965075110545),
        "scalar.closed_resonance": (1.148403342097341e-06, 1.3502945042316372e-06),
        "vector.open_resonance": (6.105824110025352e-07, 6.371224032649989e-07),
        "vector.narrow_resonance": (0.3074889583129119, 0.3270447354727253),
        "vector.closed_resonance": (4.1185334301418195e-06, 4.98145477734309e-06),
    }

    #: The semi-analytic path is a closed-form composition of
    #: `thermal_cross_section` with no adaptive solver in it, so the port's
    #: <= 2.06e-14 drift on that kernel (numerical-impact.md, Tasks 5.1/5.2)
    #: arrives essentially undamped: measured <= 4.2e-16 over these six
    #: points.  1e-12 is ~2000x that, tight enough to catch a real kernel
    #: regression and loose enough to survive a libm difference.
    SEMI_ANALYTIC_RTOL = 1e-12

    #: The Boltzmann path runs `scipy.integrate.solve_ivp` with the
    #: `relic_density` default ``rtol=1e-5``.  A last-bit change in
    #: `thermal_cross_section` flips a step-acceptance decision and the
    #: whole step sequence differs, so the answer moves at the *solver's*
    #: tolerance, not the kernel's: measured <= 3.83e-5 here.  Tightening
    #: the solve collapses it (Task 5.3 measured 2.8e-7 at ``rtol=1e-8``
    #: and 3.8e-9 at ``rtol=1e-10``), which is what identifies it as step
    #: selection rather than drift.  1e-4 bounds the measured spread with
    #: ~3x headroom.
    BOLTZMANN_RTOL = 1e-4

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
                    relic_density(model, semi_analytic=False),
                    self.PINNED[name][1],
                    rtol=self.BOLTZMANN_RTOL,
                )
