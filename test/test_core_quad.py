"""``hazma._core.quad`` against ``scipy.integrate.quad``: the QUADPACK port.

Every compiled kernel in hazma that integrates does it by calling
``scipy.integrate.quad`` from Cython with a ``cdef`` callback — twelve
live call sites, all over finite intervals, listed in
``projects/cython-to-rust/references/numerics-replacements.md``.
cython-to-rust Task 3.3 replaces that with a translation of the netlib
QUADPACK Fortran in ``rust/src/quad.rs`` (``qk15``, ``qk21``, ``qelg``,
``qpsrt``, ``qagse``, ``qagpe``), which is what lets the extension stop
linking against scipy's C API. This module is the gate on that.

What is actually being compared
-------------------------------
``hazma._core.quad.quad`` takes a *Python callable*, so every test here
hands the **same** Python integrand to scipy and to the port. Any
difference is therefore the quadrature algorithm and nothing else — not a
Rust reimplementation of the integrand, and not a different grid. That is
also what the Cython does today, since ``quad`` re-enters Python once per
node there too.

The two things this module pins are different in kind:

1. **The break-point preprocessing contract** — what
   ``quad(points=...)`` does with unsorted, duplicated,
   endpoint-coincident and out-of-interval entries. This is *not* derived
   from the QUADPACK documentation; it is scipy's own three lines of
   Python (``np.unique``, then ``a < p``, then ``p < b``) and
   :class:`TestBreakPointPreprocessing` re-derives each clause from scipy
   at run time. Both degeneracies occur live: the five ``points=[-1, 1]``
   calls on ``[-1, 1]`` have **every** break point coincident with an
   endpoint, and the thermal-average calls pass ``[2, m/mx, 2 m/mx]`` whose
   mediator entries fall outside the interval for a heavy mediator.
2. **The numbers** — agreement with scipy on QUADPACK's own reference
   problems and on every live integrand shape.

Why the tolerances are what they are
------------------------------------
Task 3.3's exit criteria ask for agreement within 10x the requested
tolerance, and within 1e-12 relative on smooth cases. Measured over
11,274 random (integrand, tolerance, limit, points) combinations against
scipy 1.18.0, the port does far better on the 4,461 that **converged**:
it reproduced scipy's ``neval`` and ``last`` on all but 5 (0.11%),
agreed on the termination flag every time, and landed within 3.6e-2 of
the requested tolerance (8.2e-11 relative at worst). Every case in this
file reproduces the subdivision exactly, so ``SMOOTH_RTOL = 1e-13`` is a
ceiling with orders of headroom rather than a fitted tolerance —
anything approaching it is a defect, not rounding.

The ``neval``/``last`` equality is the sharper assertion of the two and is
made everywhere it can be. A value can agree by luck; an identical
subdivision count on a singular integrand cannot.

Two classes deliberately assert *loosely* — :class:`TestDivergenceRegime`
and :class:`TestAdaptiveHeuristics`, both of which use inputs that
exhaust ``limit``. Their docstrings say why.

Lifetime
--------
Nothing here parses Cython, so this module outlives the ``.pyx``. It
should keep running after Phase 06 as the standing check that the port
still tracks scipy — a property the parity corpus cannot see, because the
corpus pins *spectra*, not the integrator underneath them.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import ClassVar

import numpy as np
import pytest
import scipy.integrate as si
from scipy.special import k1

from hazma.scalar_mediator import _c_scalar_mediator_cross_sections as scalar_xs
from hazma.vector_mediator import _c_vector_mediator_cross_sections as vector_xs

#: A quadrature integrand: one float in, one float out.
Integrand = Callable[[float], float]

#: The keyword arguments both integrators accept — `epsabs`, `epsrel`,
#: `limit`, `points`. Spelled out rather than `Any` so a typo in a test
#: is a type error rather than a silently ignored keyword.
QuadKwarg = float | int | list[float] | None

core_quad = pytest.importorskip(
    "hazma._core.quad",
    reason="hazma._core is not built; run `pip install -e .`",
)

#: Ceiling for smooth integrands. It is a ceiling, not a fitted
#: tolerance: every comparison in this file also asserts that the port
#: reproduced scipy's `neval` and `last` exactly, so a value that drifted
#: anywhere near 1e-13 while the subdivision stayed identical would mean
#: the arithmetic had changed, not the algorithm.
SMOOTH_RTOL = 1e-13

#: Length of ``scipy.integrate.quad``'s ``full_output`` tuple when it
#: converged: value, abserr, infodict. A fourth entry is the
#: abnormal-termination message, which is the only place scipy exposes a
#: non-zero ``ier``.
_CONVERGED_TUPLE_LEN = 3

#: Ceiling for integrands with an algebraic or logarithmic singularity,
#: where scipy and the port both reach the answer through the epsilon
#: algorithm. Extrapolation sums the same terms in the same order but the
#: two builds' compilers are free to contract differently, so a few more
#: ulp are expected here than on a single-panel result.
SINGULAR_RTOL = 1e-12

#: Below this, a `qk21` result that should vanish by symmetry counts as
#: zero: the rule sums 21 weighted values of order 1, so cancellation
#: leaves a few ulp.
_SYMMETRY_ZERO = 1e-16

#: `resabs` is the rule applied to |f|, not the true integral of |f|. For
#: f(x) = x on [-1, 1] the 21-point rule misses the true value 1 by about
#: 4e-3; this is the floor that distinguishes "different number" from
#: "rounding".
_RESABS_GAP = 1e-3

#: An oscillatory integrand whose exact value is zero is checked
#: absolutely, not relatively; this is the ceiling at epsabs = 1e-12.
_OSCILLATORY_ABS = 1e-13

#: `qagpe` and `qagse` must disagree by at least this much on the
#: dispatch-discriminating integrand, or that test proves nothing.
_QAGP_QAGSE_MIN_GAP = 0.01

#: Bounds on the port-vs-scipy gap in the limit-exhausted regime, and on
#: how far either lands from the exact value there. See
#: :class:`TestDivergenceRegime`.
_DIVERGENCE_GAP_MIN = 1e-6
_DIVERGENCE_GAP_MAX = 1e-2


def scipy_quad(
    f: Integrand, a: float, b: float, **kwargs: QuadKwarg
) -> tuple[float, float, int, int, bool]:
    """``scipy.integrate.quad`` with ``full_output``, warnings silenced.

    Returns ``(value, abserr, neval, last, converged)``. ``converged`` is
    ``False`` when scipy appended an abnormal-termination message, which
    is the only way ``full_output`` exposes a non-zero ``ier``.

    An abnormal termination is a *value* here rather than a warning, and
    that is the point: hazma's call sites take ``quad(...)[0]`` and never
    see the warning, so a test that let ``IntegrationWarning`` fail the
    run would be testing something no caller experiences.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", si.IntegrationWarning)
        out = si.quad(f, a, b, full_output=1, **kwargs)
    value, abserr, info = out[0], out[1], out[2]
    return value, abserr, info["neval"], info["last"], len(out) == _CONVERGED_TUPLE_LEN


def assert_matches_scipy(
    f: Integrand,
    a: float,
    b: float,
    *,
    rtol: float = SMOOTH_RTOL,
    label: str = "",
    **kwargs: QuadKwarg,
) -> float:
    """Run `f` through both integrators and assert they agree.

    Asserts the subdivision bookkeeping (`neval`, `last`) is *identical*
    before it looks at the value, because that is the assertion a lucky
    number cannot pass. Returns the measured relative difference so a
    caller can report it.
    """
    s_value, _s_abserr, s_neval, s_last, s_converged = scipy_quad(f, a, b, **kwargs)
    r_value, _r_abserr, r_neval, r_last, r_ier = core_quad.quad(f, a, b, **kwargs)

    assert (r_ier == 0) is s_converged, (
        f"{label}: termination flag differs from scipy — "
        f"ier {r_ier} against scipy converged={s_converged}"
    )

    assert (r_neval, r_last) == (s_neval, s_last), (
        f"{label}: subdivision differs from scipy — "
        f"neval {r_neval} vs {s_neval}, last {r_last} vs {s_last}"
    )
    rel = abs(r_value - s_value) / abs(s_value) if s_value else abs(r_value - s_value)
    assert rel <= rtol, f"{label}: got {r_value!r}, scipy {s_value!r}, rel {rel:.3e}"
    return rel


# ---------------------------------------------------------------------
# The break-point preprocessing contract
# ---------------------------------------------------------------------


class TestBreakPointPreprocessing:
    """What ``quad(points=...)`` does before QUADPACK sees the list.

    Every test here uses an integrand with a **genuine interior
    singularity**, so a break point that survives filtering changes
    ``neval`` and ``last`` visibly. On a smooth integrand the whole
    contract is unobservable and these tests would pass against any
    filtering rule at all.
    """

    @staticmethod
    def singular(centre: float) -> Integrand:
        """``|x - centre|**-1/2`` — integrable, but unbounded at `centre`."""

        def f(x: float) -> float:
            d = abs(x - centre)
            return 0.0 if d == 0.0 else d**-0.5

        return f

    def test_break_points_are_sorted_and_deduplicated(self) -> None:
        # np.unique does both. An unsorted, duplicated list must give the
        # same subdivision as the clean one — including through the port.
        f = self.singular(1.0)
        clean = core_quad.quad(f, 0.0, 3.0, points=[1.0, 2.0], epsabs=1e-10)
        messy = core_quad.quad(f, 0.0, 3.0, points=[2.0, 1.0, 2.0, 1.0], epsabs=1e-10)
        assert messy == clean
        assert_matches_scipy(
            f,
            0.0,
            3.0,
            points=[2.0, 1.0, 2.0, 1.0],
            epsabs=1e-10,
            rtol=SINGULAR_RTOL,
            label="unsorted duplicated points",
        )

    def test_endpoint_coincident_break_points_are_dropped(self) -> None:
        # The live shape: `points=[-1, 1]` on the interval [-1, 1], at four
        # spectra call sites and both mediator spectrum modules. Nothing
        # survives the filter, so QUADPACK starts from a single interval.
        f = self.singular(0.0)
        with_points = core_quad.quad(
            f, -1.0, 1.0, points=[-1.0, 1.0], epsabs=1e-10, epsrel=1e-5
        )
        empty = core_quad.quad(f, -1.0, 1.0, points=[], epsabs=1e-10, epsrel=1e-5)
        assert with_points == empty
        assert_matches_scipy(
            f,
            -1.0,
            1.0,
            points=[-1.0, 1.0],
            epsabs=1e-10,
            epsrel=1e-5,
            rtol=SINGULAR_RTOL,
            label="endpoint-coincident points",
        )

    def test_out_of_interval_break_points_are_dropped(self) -> None:
        # The live shape: a heavy mediator pushes `m/mx` and `2 m/mx` past
        # the upper bound of the thermal integral. The interior point must
        # still be honoured.
        f = self.singular(1.0)
        mixed = core_quad.quad(f, 0.0, 3.0, points=[-5.0, 1.0, 9.0], epsabs=1e-10)
        interior_only = core_quad.quad(f, 0.0, 3.0, points=[1.0], epsabs=1e-10)
        assert mixed == interior_only
        assert_matches_scipy(
            f,
            0.0,
            3.0,
            points=[-5.0, 1.0, 9.0],
            epsabs=1e-10,
            rtol=SINGULAR_RTOL,
            label="out-of-interval points",
        )

    def test_all_break_points_out_of_interval_is_not_an_error(self) -> None:
        f = self.singular(1.0)
        assert_matches_scipy(
            f,
            0.0,
            3.0,
            points=[-5.0, 9.0],
            epsabs=1e-10,
            rtol=SINGULAR_RTOL,
            label="every point out of interval",
        )

    def test_nan_break_points_are_dropped(self) -> None:
        # Every comparison against NaN is false, so `a < p` discards it.
        # Pinned because "sorts last" and "is discarded" are different
        # outcomes and only one of them is scipy's.
        f = self.singular(1.0)
        with_nan = core_quad.quad(f, 0.0, 3.0, points=[1.0, float("nan")], epsabs=1e-10)
        without = core_quad.quad(f, 0.0, 3.0, points=[1.0], epsabs=1e-10)
        assert with_nan == without
        assert_matches_scipy(
            f,
            0.0,
            3.0,
            points=[1.0, float("nan")],
            epsabs=1e-10,
            rtol=SINGULAR_RTOL,
            label="NaN in points",
        )

    def test_an_empty_points_list_still_selects_qagp(self) -> None:
        # scipy dispatches on `points is None` *before* it filters, so an
        # empty-after-filtering list runs qagpe, not qagse. That is the
        # branch all six `points=[-1, 1]` call sites take, and it is
        # normally invisible: over 3,776 random (integrand, tolerance,
        # limit) combinations the two routines returned identical values,
        # identical `neval` and identical `last` in every case that
        # converged. They diverge only once the subdivision limit is hit,
        # because they enter extrapolation at different depths — qagse
        # when an interval falls below 0.375x the original length, qagpe
        # one bisection earlier, at subdivision level 1.
        #
        # This integrand is chosen to sit in that gap: |x - 1/3|^-0.9
        # cos(50x) at limit=10 exhausts the subdivisions, and the two
        # routines then disagree by 11%. So it distinguishes the branches,
        # which the smooth and mildly-singular cases above cannot.
        def f(x: float) -> float:
            d = abs(x - 1.0 / 3.0)
            return 0.0 if d == 0.0 else d**-0.9 * math.cos(50.0 * x)

        settings = dict(epsabs=1e-12, epsrel=1e-10, limit=10)
        with_points = core_quad.quad(f, 0.0, 1.0, points=[], **settings)
        without_points = core_quad.quad(f, 0.0, 1.0, **settings)
        rel = abs(with_points[0] - without_points[0]) / abs(without_points[0])
        assert rel > _QAGP_QAGSE_MIN_GAP, (
            "this case no longer distinguishes qagpe from qagse, so the "
            "assertion below proves nothing — find another integrand"
        )

        assert_matches_scipy(
            f,
            0.0,
            1.0,
            points=[],
            rtol=SINGULAR_RTOL,
            label="points=[] (qagpe path)",
            **settings,
        )
        assert_matches_scipy(
            f,
            0.0,
            1.0,
            rtol=SINGULAR_RTOL,
            label="points omitted (qagse path)",
            **settings,
        )

    def test_reversed_limits_negate_the_result(self) -> None:
        f = self.singular(1.0)
        forward = core_quad.quad(f, 0.0, 3.0, points=[1.0], epsabs=1e-10)
        backward = core_quad.quad(f, 3.0, 0.0, points=[1.0], epsabs=1e-10)
        assert backward[0] == -forward[0]
        assert_matches_scipy(
            f,
            3.0,
            0.0,
            points=[1.0],
            epsabs=1e-10,
            rtol=SINGULAR_RTOL,
            label="reversed limits",
        )

    def test_too_many_break_points_for_the_limit_raises(self) -> None:
        # QUADPACK's own `limit <= npts` check, counted after filtering.
        f = self.singular(1.0)
        points = list(np.linspace(0.1, 2.9, 50))
        with pytest.raises(ValueError):
            si.quad(f, 0.0, 3.0, points=points, limit=50)
        with pytest.raises(ValueError):
            core_quad.quad(f, 0.0, 3.0, points=points, limit=50)
        # One fewer break point and it is legal on both sides.
        assert core_quad.quad(f, 0.0, 3.0, points=points[:-1], limit=50)[0] > 0.0


# ---------------------------------------------------------------------
# QUADPACK's own reference problems
# ---------------------------------------------------------------------


class TestReferenceProblems:
    """Analytic values, so scipy is not the only oracle in the room.

    Everything else in this module compares two implementations; if both
    were wrong the same way, nothing above would notice. These have closed
    forms taken from the QUADPACK book's own worked examples (Piessens,
    de Doncker-Kapenga, Überhuber, Kahaner, *QUADPACK: A Subroutine
    Package for Automatic Integration*, Springer 1983, §1.2 and §3.2).
    """

    @pytest.mark.parametrize("alpha", [-0.9, -0.5, 0.0, 1.0, 2.0, 5.0])
    def test_endpoint_logarithmic_singularity(self, alpha: float) -> None:
        # int_0^1 x^alpha ln(1/x) dx = 1/(alpha+1)^2.
        def f(x: float) -> float:
            return 0.0 if x <= 0.0 else x**alpha * math.log(1.0 / x)

        want = 1.0 / (alpha + 1.0) ** 2
        got = core_quad.quad(f, 0.0, 1.0, epsabs=1e-12, epsrel=1e-12)[0]
        assert got == pytest.approx(want, rel=1e-11)

    def test_interior_algebraic_singularity_with_a_break_point(self) -> None:
        # int_0^1 |x - 1/3|^(-1/2) dx = 2(sqrt(1/3) + sqrt(2/3)).
        centre = 1.0 / 3.0

        def f(x: float) -> float:
            d = abs(x - centre)
            return 0.0 if d == 0.0 else d**-0.5

        want = 2.0 * (math.sqrt(centre) + math.sqrt(1.0 - centre))
        got = core_quad.quad(f, 0.0, 1.0, points=[centre], epsabs=1e-12, epsrel=1e-12)[
            0
        ]
        assert got == pytest.approx(want, rel=1e-10)

    def test_a_smooth_integrand_needs_one_panel(self) -> None:
        # int_0^1 exp(x) dx = e - 1, resolved by a single 21-point rule.
        value, _abserr, neval, last, ier = core_quad.quad(
            math.exp, 0.0, 1.0, epsabs=1e-10, epsrel=1e-10
        )
        assert (neval, last, ier) == (21, 1, 0)
        assert value == pytest.approx(math.e - 1.0, rel=1e-15)

    def test_an_oscillatory_integrand(self) -> None:
        # int_0^{2 pi} sin(50 x) dx = 0 by periodicity — a case where the
        # answer is a cancellation, so an absolute rather than relative
        # check is the honest one.
        got = core_quad.quad(
            lambda x: math.sin(50.0 * x), 0.0, 2.0 * math.pi, epsabs=1e-12
        )[0]
        assert abs(got) < _OSCILLATORY_ABS


# ---------------------------------------------------------------------
# The live integrand shapes
# ---------------------------------------------------------------------


class TestLiveIntegrandShapes:
    """One case per row of the call-site table, at that row's settings.

    Two of these run the **actual** integrand — the thermal-average one,
    reachable because ``sigma_xx_to_all`` is a public Cython export. The
    rest reproduce the *shape*: the boost Jacobian ``1/(2 gamma |1 - beta
    cos t|)`` that the cos-theta sites integrate against
    (``hazma/spectra/_photon/_pion.pyx:94-99``), and the smooth
    ``[E gamma (1 - beta), E gamma (1 + beta)]`` energy window the rho,
    positron-pion and neutrino-pion sites use. What is being tested is the
    integrator, not the physics; the physics is the parity corpus's job.
    """

    @staticmethod
    def boost_jacobian_integrand(beta: float, gamma: float, eng: float) -> Integrand:
        """The cos-theta shape: a Jacobian with a `1 - beta*cl` denominator.

        `hazma/spectra/_photon/_pion.pyx:94-99` computes
        ``jac = 1/(2 gamma |1 - beta cl|)`` and evaluates the rest-frame
        spectrum at ``eng * gamma * (1 - beta cl)``. The stand-in below
        keeps both, with a smooth positive stand-in for the spectrum, so
        the integrand has the same near-endpoint peaking the real one has.
        """

        def f(cl: float) -> float:
            shifted = eng * gamma * (1.0 - beta * cl)
            jac = 1.0 / (2.0 * gamma * abs(1.0 - beta * cl))
            return jac * math.exp(-shifted) * shifted

        return f

    @pytest.mark.parametrize("beta", [0.1, 0.9, 0.999, 0.999999])
    def test_cos_theta_sites(self, beta: float) -> None:
        # `points=[-1, 1]`, epsabs=1e-10, epsrel=1e-5 — the settings shared
        # by _photon/_pion.pyx:123 and all four mediator spectrum modules.
        gamma = 1.0 / math.sqrt(1.0 - beta * beta)
        f = self.boost_jacobian_integrand(beta, gamma, 1.0)
        assert_matches_scipy(
            f,
            -1.0,
            1.0,
            points=[-1.0, 1.0],
            epsabs=1e-10,
            epsrel=1e-5,
            rtol=SINGULAR_RTOL,
            label=f"cos-theta site, beta={beta}",
        )

    @pytest.mark.parametrize(
        ("epsabs", "epsrel", "site"),
        [
            (1e-10, 1e-5, "spectra/_photon/_rho.pyx:52,123"),
            (1e-10, 1e-4, "spectra/_positron/_pion.pyx:58"),
            (1.49e-8, 1.49e-8, "spectra/_neutrino/_pion.pyx:124,127"),
        ],
    )
    def test_boosted_energy_window_sites(
        self, epsabs: float, epsrel: float, site: str
    ) -> None:
        # The three sites that integrate over [E gamma (1 - beta),
        # E gamma (1 + beta)] with no break points, each at its own
        # tolerances. The neutrino row is scipy's defaults, which the
        # `.pyx` reaches by passing neither keyword.
        beta, eng = 0.98, 30.0
        gamma = 1.0 / math.sqrt(1.0 - beta * beta)
        emin, emax = eng * gamma * (1.0 - beta), eng * gamma * (1.0 + beta)

        def f(e: float) -> float:
            # A decay spectrum's shape: falls off, vanishes at the endpoint.
            return math.sqrt(max(emax - e, 0.0)) * math.exp(-e / emax) / e

        assert_matches_scipy(
            f,
            emin,
            emax,
            epsabs=epsabs,
            epsrel=epsrel,
            rtol=SINGULAR_RTOL,
            label=f"boosted energy window, {site}",
        )

    def test_the_nested_rho_integral(self) -> None:
        # `spectra/_photon/_rho.pyx` integrates a function that itself
        # calls quad — the reference file calls it the stress test. Both
        # levels run on the port here, which is the configuration the port
        # will actually be in.
        def inner_scipy(y: float) -> float:
            return si.quad(
                lambda t: math.exp(-t * t), 0.0, y, epsabs=1e-10, epsrel=1e-5
            )[0]

        def inner_port(y: float) -> float:
            return core_quad.quad(
                lambda t: math.exp(-t * t), 0.0, y, epsabs=1e-10, epsrel=1e-5
            )[0]

        s_value = si.quad(inner_scipy, 0.0, 2.0, epsabs=1e-10, epsrel=1e-5)[0]
        r_value = core_quad.quad(inner_port, 0.0, 2.0, epsabs=1e-10, epsrel=1e-5)[0]
        assert r_value == pytest.approx(s_value, rel=SMOOTH_RTOL)

    @staticmethod
    def thermal_integrand(
        model: str, x: float, mx: float, m_med: float, width: float
    ) -> Integrand:
        """The *actual* mediator thermal integrand, for either model.

        Both are ``sigma_xx_to_all(mx*z, ...) * z**2 * (z**2 - 4) *
        k1(x*z)`` —
        ``hazma/scalar_mediator/_c_scalar_mediator_cross_sections.pyx:1354-1361``
        and
        ``hazma/vector_mediator/_c_vector_mediator_cross_sections.pyx:598-606``
        — and ``sigma_xx_to_all`` is a public export of each module, so
        this is the integrand rather than a stand-in for it.
        """
        if model == "scalar":
            cross_sections = scalar_xs
            # (e_cm, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs)
            args = (1.0, 1.0, 0.1, 0.1, 1e4, width, 0.0)
        else:
            cross_sections = vector_xs
            # (e_cm, mx, mv, gvxx, gvuu, gvdd, gvss, gvee, gvmumu, width_v)
            args = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, width)

        def f(z: float) -> float:
            sigma = cross_sections.sigma_xx_to_all(mx * z, mx, m_med, *args)
            return sigma * z**2 * (z**2 - 4.0) * k1(x * z)

        return f

    #: Upper limit of each model's thermal integral, as a function of x.
    #: `max(50/x, 100)` for the scalar
    #: (`_c_scalar_mediator_cross_sections.pyx:1412`) and `max(50/x, 150)`
    #: for the vector (`_c_vector_mediator_cross_sections.pyx:657`).
    THERMAL_FLOOR: ClassVar[dict[str, float]] = {"scalar": 100.0, "vector": 150.0}

    @pytest.mark.parametrize("model", ["scalar", "vector"])
    @pytest.mark.parametrize(
        ("mx", "m_med", "expected_last", "regime"),
        [
            (100.0, 250.0, 3, "break points interior (resonance active)"),
            (100.0, 200.0, 2, "the lower break point sits on the lower limit"),
            (1.0, 500.0, 1, "both mediator break points outside the interval"),
        ],
    )
    def test_thermal_cross_section_site(
        self,
        model: str,
        mx: float,
        m_med: float,
        expected_last: int,
        regime: str,
    ) -> None:
        # `points=[2, m/mx, 2 m/mx]` over [2, max(50/x, floor)] at scipy's
        # default tolerances — both mediator sites verbatim, in the three
        # regimes Task 3.3's exit criteria name. At x = 20 the upper limit
        # is the floor (100 or 150), so:
        #   mx=100, m=250 -> [2, 2.5, 5]; 2 equals the lower limit and is
        #                    dropped, leaving 2 interior points, 3 intervals;
        #   mx=100, m=200 -> [2, 2, 4]; the duplicate collapses and the
        #                    survivor 2 is dropped, leaving 1 point,
        #                    2 intervals;
        #   mx=1,   m=500 -> [2, 500, 1000]; both mediator points exceed the
        #                    upper limit, leaving nothing and 1 interval.
        # `last` is asserted directly, not only against scipy: it is the
        # observable that says the filtering produced the partition this
        # comment claims, and two implementations could agree on a wrong
        # one.
        x, width = 20.0, 2.5
        f = self.thermal_integrand(model, x, mx, m_med, width)
        upper = max(50.0 / x, self.THERMAL_FLOOR[model])
        points = [2.0, m_med / mx, 2.0 * m_med / mx]

        assert_matches_scipy(
            f,
            2.0,
            upper,
            points=points,
            rtol=SINGULAR_RTOL,
            label=f"{model} thermal cross section, {regime}",
        )
        _value, _abserr, _neval, last, ier = core_quad.quad(
            f, 2.0, upper, points=points
        )
        assert last == expected_last
        assert ier == 0, "no live call site should terminate abnormally"

        # The live tolerances are absolute 1.49e-8 against an integrand of
        # order 1e-33, so QUADPACK stops on the initial partition and never
        # subdivides — which is what happens in production too. Guard that
        # the integrand is not simply zero, or all three regimes would
        # agree for the wrong reason.
        sampled = [f(z) for z in np.linspace(2.001, min(upper, 30.0), 200)]
        assert max(sampled) > 0.0, "the live thermal integrand vanished"


# ---------------------------------------------------------------------
# The bare Gauss-Kronrod rules
# ---------------------------------------------------------------------


class TestKronrodRules:
    """`qk15` and `qk21` applied once, without subdivision.

    ``rust/src/quad.rs`` pins the tables by degree of exactness in
    ``cargo test``; what these add is the cross-language check that the
    compiled extension exposes the same rule the Rust unit tests
    exercised, and that ``qk21`` is the rule ``quad`` runs on.
    """

    def test_qk21_is_what_quad_uses_on_a_single_panel(self) -> None:
        # A smooth integrand converges in one panel, so `quad`'s answer
        # must be *bit-identical* to one application of qk21. This is what
        # ties the two entry points together; nothing else here would
        # notice if `quad` quietly used qk15.
        value, abserr, _resabs, _resasc = core_quad.qk21(math.exp, 0.0, 1.0)
        q_value, q_abserr, neval, last, _ier = core_quad.quad(
            math.exp, 0.0, 1.0, epsabs=1e-10, epsrel=1e-10
        )
        assert (neval, last) == (21, 1)
        assert q_value == value
        assert q_abserr == abserr

    @pytest.mark.parametrize("rule", ["qk15", "qk21"])
    def test_a_rule_integrates_a_low_degree_polynomial_exactly(self, rule: str) -> None:
        # Both rules are exact well past degree 5, so this is an analytic
        # pin rather than a comparison: int_{-1}^{1} (3x^4 - 2x^2 + 1) dx
        # = 6/5 - 4/3 + 2.
        f = getattr(core_quad, rule)
        got = f(lambda x: 3.0 * x**4 - 2.0 * x**2 + 1.0, -1.0, 1.0)[0]
        assert got == pytest.approx(6.0 / 5.0 - 4.0 / 3.0 + 2.0, rel=1e-15)

    def test_resabs_is_the_rule_applied_to_the_absolute_value(self) -> None:
        # `resabs` is not int|f|: |x| is not a polynomial, so the rule
        # misses the true value 1 by ~4e-3. Pinning it against the rule
        # run on |x| states the invariant exactly, and distinguishes
        # `resabs` from `result` for a sign-changing integrand.
        result, _abserr, resabs, _resasc = core_quad.qk21(lambda x: x, -1.0, 1.0)
        assert abs(result) < _SYMMETRY_ZERO
        assert resabs == core_quad.qk21(abs, -1.0, 1.0)[0]
        assert abs(resabs - 1.0) > _RESABS_GAP

    def test_the_two_rules_agree_on_a_smooth_integrand(self) -> None:
        want = math.e - 1.0
        assert core_quad.qk15(math.exp, 0.0, 1.0)[0] == pytest.approx(want, rel=1e-14)
        assert core_quad.qk21(math.exp, 0.0, 1.0)[0] == pytest.approx(want, rel=1e-14)


# ---------------------------------------------------------------------
# Errors and abnormal termination
# ---------------------------------------------------------------------


#: scipy reports ``ier`` only through the message text ``full_output``
#: appends, so decoding it is the only way to compare termination flags.
#: The fragments come from ``scipy.integrate._quadpack_py.quad``'s own
#: ``msgs`` dictionary.
_SCIPY_MESSAGE_TO_IER = {
    "maximum number of subdivisions": 1,
    "roundoff error is detected": 2,
    "Extremely bad integrand behavior": 3,
    "not converge": 4,
    "probably divergent, or slowly convergent": 5,
}


def scipy_ier(f: Integrand, a: float, b: float, **kwargs: QuadKwarg) -> int:
    """The ``ier`` scipy would have reported, decoded from its message."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", si.IntegrationWarning)
        out = si.quad(f, a, b, full_output=1, **kwargs)
    if len(out) == _CONVERGED_TUPLE_LEN:
        return 0
    message = out[3]
    for fragment, code in _SCIPY_MESSAGE_TO_IER.items():
        if fragment in message:
            return code
    raise AssertionError(f"unrecognised scipy message: {message!r}")


class TestTerminationFlags:
    """Every ``ier`` QUADPACK can return, mapped against scipy's own.

    The port returns ``ier`` where scipy raises an ``IntegrationWarning``,
    so "mapped" has to mean *agrees with scipy's code*, not merely
    "non-zero". Each case below is an input that drives QUADPACK down that
    branch; six of the seven codes are reachable this way, and the
    seventh (``ier = 6``) is the invalid-input case
    :class:`TestErrorBehavior` covers as a raise.
    """

    @staticmethod
    def reciprocal(x: float) -> float:
        return 0.0 if x == 0.0 else 1.0 / x

    @pytest.mark.parametrize(
        ("ier", "func", "a", "b", "kwargs"),
        [
            (0, math.exp, 0.0, 1.0, dict(epsabs=1e-10, epsrel=1e-10)),
            # 1 — the subdivision limit is reached on a divergent integral.
            (1, reciprocal, 0.0, 1.0, dict(epsabs=1e-10, epsrel=1e-10, limit=10)),
            # 2 — roundoff prevents an unattainable absolute tolerance.
            (2, math.sin, 0.0, 1.0, dict(epsabs=1e-300, epsrel=1e-16)),
            # 3 — "extremely bad integrand behaviour": the subintervals
            #     collapse toward zero magnitude.
            (
                3,
                reciprocal,
                0.0,
                1e-300,
                dict(epsabs=1e-320, epsrel=1e-14, limit=200),
            ),
            # 4 — the extrapolation table stops converging.
            (
                4,
                lambda x: 0.0 if x <= 0.0 else x**-0.999,
                0.0,
                1.0,
                dict(epsabs=1e-14, epsrel=1e-14, limit=200),
            ),
            # 5 — probably divergent: 1/x^2 has a non-integrable pole.
            (
                5,
                lambda x: 0.0 if x == 0.0 else x**-2,
                0.0,
                1.0,
                dict(epsabs=1e-10, epsrel=1e-10),
            ),
        ],
    )
    def test_ier_matches_scipy(
        self,
        ier: int,
        func: Integrand,
        a: float,
        b: float,
        kwargs: dict[str, QuadKwarg],
    ) -> None:
        assert scipy_ier(func, a, b, **kwargs) == ier, (
            "this input no longer drives scipy down the branch it was "
            "chosen for, so the assertion below proves nothing"
        )
        assert core_quad.quad(func, a, b, **kwargs)[4] == ier


class TestDivergenceRegime:
    """Where the port and scipy can disagree, and why no caller reaches it.

    Measured over 11,274 random (integrand, tolerance, limit, points)
    combinations against scipy 1.18.0. On the 4,461 that **converged**,
    the port reproduced scipy's ``neval`` and ``last`` on all but 5
    (0.11%) and landed within 3.6e-2 of the requested tolerance, 8.2e-11
    relative at worst. Termination flags agreed on all 11,274.

    On the 6,813 that **exhausted** ``limit`` the values can separate
    without bound — 4.5e-5 relative in that sweep, and 11% on the
    hand-picked case below. That is the epsilon algorithm being chaotic on
    a sequence that is not converging, not a defect in the translation:
    identical subdivision plus a few ulp of difference in the
    extrapolation table is enough, and QUADPACK is not claiming an answer
    there — it is reporting one it says it could not reach.

    hazma never enters that regime. Every live shape in
    :class:`TestLiveIntegrandShapes` returns ``ier = 0``, and the class
    asserts it.
    """

    @staticmethod
    def slowly_convergent(x: float) -> float:
        # int_0^{1/2} dx / (x ln^2 x) = 1/ln 2 = 1.4427, approached far
        # too slowly for QUADPACK to certify at any usable limit.
        return 0.0 if x <= 0.0 or x >= 1.0 else 1.0 / (x * math.log(x) ** 2)

    def test_the_subdivision_still_matches_when_the_limit_is_hit(self) -> None:
        # The divergence is in the extrapolated value, not in the path:
        # both implementations subdivide identically right up to the
        # limit. Pinned so that a future translation bug — which would
        # move `neval` — cannot hide behind "that regime is chaotic".
        for limit in (50, 100, 200, 400):
            settings = dict(epsabs=1e-10, epsrel=1e-10, limit=limit)
            _sv, _sa, s_neval, s_last, s_converged = scipy_quad(
                self.slowly_convergent, 0.0, 0.5, **settings
            )
            _v, _a, neval, last, ier = core_quad.quad(
                self.slowly_convergent, 0.0, 0.5, **settings
            )
            assert not s_converged and ier != 0, "expected an abnormal end"
            assert (neval, last) == (s_neval, s_last)
            assert last == limit

    def test_the_extrapolated_values_may_differ_there(self) -> None:
        # Documented rather than tolerated silently: this is the one shape
        # in this file where the two implementations give visibly
        # different numbers, and the assertion records how different so a
        # change in the gap shows up as a test failure.
        settings = dict(epsabs=1e-10, epsrel=1e-10, limit=100)
        s_value = scipy_quad(self.slowly_convergent, 0.0, 0.5, **settings)[0]
        value = core_quad.quad(self.slowly_convergent, 0.0, 0.5, **settings)[0]
        rel = abs(value - s_value) / abs(s_value)
        assert _DIVERGENCE_GAP_MIN < rel < _DIVERGENCE_GAP_MAX, f"gap moved: {rel:.3e}"
        # Both are still the same integral, roughly: the true value is
        # 1/ln 2 and both land ~4e-3 short of it, which is what "QUADPACK
        # could not certify this" looks like.
        exact = 1.0 / math.log(2.0)
        for got in (s_value, value):
            assert abs(got - exact) / exact < _DIVERGENCE_GAP_MAX


class TestAdaptiveHeuristics:
    """The two branches inside the adaptive loop that nothing else reaches.

    `qagpe`'s `ndin` flag and `qagse`'s roundoff counters are pure
    heuristics: they change *which* subinterval is bisected next, never
    the arithmetic, so on ordinary integrands they are invisible — a
    mutation campaign found both survive every other test in this file.
    The two inputs below were found by applying each mutation and
    searching for a case whose answer moved: dropping `ndin` moves the
    first by a factor of 48, and relaxing the roundoff threshold moves the
    second by a factor of 2,800.

    Both sit in the limit-exhausted regime (see
    :class:`TestDivergenceRegime`), so both are asserted at a deliberately
    **coarse** ``rtol = 1e-6``. That is not a weak gate for what it is
    guarding: the defects it exists to catch move the answer by factors,
    not by ulps, and a tight tolerance here would be pinning digits the
    two implementations are entitled to disagree about.
    """

    def test_the_ndin_flag_promotes_a_degenerate_initial_panel(self) -> None:
        # `qagpe` marks an initial subinterval whose 21-point rule reports
        # `abserr == resasc` — a panel the rule cannot resolve at all — and
        # replaces its error estimate with the *total* initial error, which
        # sends it to the front of the subdivision order. sin(w/x) over a
        # 39-point uniform grid has exactly one such panel, the one
        # touching the essential singularity at 0.
        w = 293.25358911967305
        grid = [i / 40 for i in range(1, 40)]

        def f(x: float) -> float:
            return math.sin(w / max(x, 1e-14))

        assert_matches_scipy(
            f,
            0.0,
            1.0,
            points=grid,
            epsabs=1e-10,
            epsrel=1e-10,
            limit=50,
            rtol=1e-6,
            label="ndin-sensitive oscillatory integrand",
        )

    def test_the_roundoff_counters_gate_the_extrapolation(self) -> None:
        # `iroff1`/`iroff2` count bisections that barely improved the
        # estimate (`erro12 >= 0.99 * errmax`), and ten of them set
        # `ier = 2` and stop the extrapolation. A near-delta spike with a
        # break point on the wrong side of it produces exactly that
        # sequence.
        centre = 0.16308536762834644

        def f(x: float) -> float:
            return 1.0 / ((x - centre) ** 2 + 1e-12)

        assert_matches_scipy(
            f,
            0.0,
            1.0,
            points=[0.5],
            epsabs=1e-12,
            epsrel=1e-12,
            limit=20,
            rtol=1e-6,
            label="roundoff-counter-sensitive spike",
        )


class TestErrorBehavior:
    """Never panic across FFI; raise where scipy raises, warn where it warns."""

    def test_invalid_tolerances_raise_on_both_sides(self) -> None:
        with pytest.raises(ValueError):
            si.quad(math.exp, 0.0, 1.0, epsabs=0.0, epsrel=0.0)
        with pytest.raises(ValueError):
            core_quad.quad(math.exp, 0.0, 1.0, epsabs=0.0, epsrel=0.0)

    def test_a_zero_limit_raises_on_both_sides(self) -> None:
        with pytest.raises(ValueError):
            si.quad(math.exp, 0.0, 1.0, limit=0)
        with pytest.raises(ValueError):
            core_quad.quad(math.exp, 0.0, 1.0, limit=0)

    def test_hitting_the_subdivision_limit_reports_and_returns(self) -> None:
        # int_0^1 dx/x diverges. scipy warns and hands back the partial
        # sum; the port returns the same partial sum with a non-zero ier
        # instead of warning, because hazma's call sites read `[0]` and
        # never see a warning.
        def f(x: float) -> float:
            return 0.0 if x == 0.0 else 1.0 / x

        s_value, _s_abserr, s_neval, s_last, s_converged = scipy_quad(
            f, 0.0, 1.0, epsabs=1e-10, epsrel=1e-10, limit=10
        )
        assert not s_converged, "scipy was expected to terminate abnormally here"
        value, _abserr, neval, last, ier = core_quad.quad(
            f, 0.0, 1.0, epsabs=1e-10, epsrel=1e-10, limit=10
        )
        assert ier != 0
        assert (neval, last) == (s_neval, s_last)
        assert value == pytest.approx(s_value, rel=SINGULAR_RTOL)
        assert math.isfinite(value)

    def test_an_exception_in_the_integrand_propagates(self) -> None:
        sentinel = RuntimeError("integrand exploded")
        calls = []

        def f(x: float) -> float:
            calls.append(x)
            raise sentinel

        with pytest.raises(RuntimeError) as excinfo:
            core_quad.quad(f, 0.0, 1.0)
        assert excinfo.value is sentinel
        # The integrand short-circuits after the first failure rather than
        # being called once per node for the rest of the run.
        assert len(calls) == 1

    def test_a_non_float_return_from_the_integrand_is_an_error(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            core_quad.quad(lambda x: "not a number", 0.0, 1.0)

    def test_a_nan_integrand_does_not_crash(self) -> None:
        value = core_quad.quad(lambda x: float("nan"), 0.0, 1.0)[0]
        assert math.isnan(value) or value == 0.0

    def test_a_zero_width_interval_is_zero(self) -> None:
        value, _abserr, _neval, _last, ier = core_quad.quad(math.exp, 1.5, 1.5)
        assert value == 0.0
        assert ier == 0
        assert si.quad(math.exp, 1.5, 1.5)[0] == 0.0
