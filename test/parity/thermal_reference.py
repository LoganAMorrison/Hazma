"""A converged, independently integrated value for the thermal averages.

``cross_sections.scalar.thermal_cross_section`` and its vector twin are
the two corpus cases whose stored arrays hold the *unconverged* estimate
QUADPACK returns when its default ``epsabs`` of 1.49e-8 is applied to an
integral of order 1e-27: the absolute criterion is satisfied by the very
first Gauss-Kronrod pass, so the initial three-interval partition comes
back unrefined and no subdivision ever happens. Roster entry ``B5`` in
``projects/parity-pinned-defect-repair`` zeroes that ``epsabs`` in both
kernels, which leaves the relative criterion binding.

This module supplies the value the repaired kernels are then held to.
It rebuilds each model's ``sigma_xx_to_all`` from the per-channel
kernels ``hazma._core`` exports -- the same six summands in the same
order the Rust integrand uses internally (``sigma_xx_to_all`` in
``rust/src/kernels/vector_xs.rs`` and in ``scalar_xs.rs``) -- applies
the Boltzmann weight, and integrates with **scipy's** QUADPACK at a
convergent tolerance.

Independence is in the integrator, which is exactly the component the
repair changes: scipy ships its own QUADPACK, the crate a Rust
transcription of it. The integrand is the controlled variable and is
deliberately shared, because it is not what moved -- the five
closed-form vector kernels reproduce the pre-port Cython bit for bit at
every one of the 5,811 corpus positions that sample them.

The two large-``x`` rules below are the kernels' own and are **not**
what ``B5`` repairs: the scalar hard-returns ``0.0`` above ``x = 300``
while the vector clips ``x`` to 300 and saturates. That divergence
between the two models predates the port and is untouched here.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.integrate as si
from scipy.integrate import IntegrationWarning
from scipy.special import k1, kn

from hazma._core import scalar_mediator as _core_scalar
from hazma._core import vector_mediator as _core_vector

if TYPE_CHECKING:
    from cases import Block

#: Electron and muon masses in MeV, spelled as the two cross-section
#: kernels carry them (``rust/src/kernels/vector_xs.rs:123,125`` and
#: ``rust/src/kernels/scalar_xs.rs:104,106``), which hard-code them
#: rather than reading ``constants::pdg``. Hard-coded here for the same
#: reason the kernels do: an oracle built on a different electron mass
#: would be integrating a different function.
ME = 0.510998928
MMU = 105.6583715

#: Floor on the upper integration limit, per model. The kernels
#: integrate to ``max(floor, 50/x)`` -- 100 for the scalar, 150 for the
#: vector -- so the floor binds everywhere above ``x = 0.5`` and
#: ``x = 1/3`` respectively.
UPPER_FLOOR = {"scalar": 100.0, "vector": 150.0}

#: Where each kernel stops integrating in ``x = m_x / T``. Above it the
#: Bessel prefactor ``x / (2 K_2(x))^2`` overflows a double.
X_CLIP = 300.0

#: Lower limit of the integral, in ``z = e_cm / m_x``: the pair
#: production threshold ``e_cm = 2 m_x``, where every cross section
#: divides by zero and the ``(z^2 - 4)`` weight is zero.
Z_THRESHOLD = 2.0

#: Relative tolerance for the reference quadrature, with ``epsabs`` off
#: so it is the criterion that binds. Four decades tighter than the
#: 1.49e-8 the repaired kernels run at, which is what makes this a
#: reference for them rather than a second opinion at the same accuracy.
REFERENCE_EPSREL = 1e-12

#: Subdivision limit. The integrand is a near-exponential spike against
#: an interval running out to 150, so the converged partition is deep;
#: 200 is twice the kernels' own ``THERMAL_LIMIT`` of 100, which keeps
#: the reference from being the shallower of the two quadratures.
REFERENCE_LIMIT = 200


def _sigma_all(
    model: str, args: list[float]
) -> tuple[Callable[[float], float], float, float]:
    """``sigma_xx_to_all`` for one model, plus its ``m_x`` and mediator mass.

    Parameters
    ----------
    model : {'scalar', 'vector'}
        Which mediator family.
    args : list of float
        The block's stored argument tuple, i.e. everything the entry
        point takes after the swept ``x``.

    Returns
    -------
    sigma_all : callable
        ``e_cm`` in MeV to the summed annihilation cross section in
        MeV^-2.
    mx, m_med : float
        Dark matter and mediator masses in MeV.
    """
    mx, m_med = args[0], args[1]
    if model == "scalar":
        # (gsxx, gsff, gsGG, gsFF, lam, width_s, vs), and the .pyx's own
        # summation order: e, mu, gg, pi0pi0, pipi, ss.
        rest = tuple(args[2:9])

        def sigma_all(e_cm: float) -> float:
            return (
                _core_scalar.sigma_xx_to_s_to_ff(e_cm, mx, m_med, *rest, ME)
                + _core_scalar.sigma_xx_to_s_to_ff(e_cm, mx, m_med, *rest, MMU)
                + _core_scalar.sigma_xx_to_s_to_gg(e_cm, mx, m_med, *rest)
                + _core_scalar.sigma_xx_to_s_to_pi0pi0(e_cm, mx, m_med, *rest)
                + _core_scalar.sigma_xx_to_s_to_pipi(e_cm, mx, m_med, *rest)
                + _core_scalar.sigma_xx_to_ss(e_cm, mx, m_med, *rest)
            )

        return sigma_all, mx, m_med

    # (gvxx, gvuu, gvdd, gvss, gvee, gvmumu, width_v), and the .pyx's own
    # summation order: e, mu, pipi, pi0g, pi0v, vv. The lepton entry
    # point takes the one coupling it uses; the rest take the full tuple
    # and drop what they do not.
    rest = tuple(args[2:9])
    gvxx, _gvuu, _gvdd, _gvss, gvee, gvmumu, width_v = rest

    def sigma_all(e_cm: float) -> float:
        return (
            _core_vector.sigma_xx_to_v_to_ff(e_cm, mx, m_med, gvxx, gvee, width_v, ME)
            + _core_vector.sigma_xx_to_v_to_ff(
                e_cm, mx, m_med, gvxx, gvmumu, width_v, MMU
            )
            + _core_vector.sigma_xx_to_v_to_pipi(e_cm, mx, m_med, *rest)
            + _core_vector.sigma_xx_to_v_to_pi0g(e_cm, mx, m_med, *rest)
            + _core_vector.sigma_xx_to_v_to_pi0v(e_cm, mx, m_med, *rest)
            + _core_vector.sigma_xx_to_vv(e_cm, mx, m_med, *rest)
        )

    return sigma_all, mx, m_med


def thermal_cross_section(model: str, args: list[float], x: float) -> float:
    """``<sigma v>(x)`` in MeV^-2, integrated to `REFERENCE_EPSREL`.

    Parameters
    ----------
    model : {'scalar', 'vector'}
        Which mediator family.
    args : list of float
        The block's stored argument tuple.
    x : float
        ``m_x / T``, dimensionless.

    Returns
    -------
    float
        The thermally averaged cross section in MeV^-2.
    """
    sigma_all, mx, m_med = _sigma_all(model, args)
    if model == "scalar":
        if x > X_CLIP:
            return 0.0
        xnew = x
    else:
        xnew = min(x, X_CLIP)

    prefactor = xnew / (2.0 * kn(2, xnew)) ** 2
    upper = max(UPPER_FLOOR[model], 50.0 / xnew)
    ratio = m_med / mx
    # QUADPACK drops break points on or outside the interval; doing it
    # here keeps scipy from raising on the duplicates the kernels pass.
    points = [p for p in (Z_THRESHOLD, ratio, 2.0 * ratio) if Z_THRESHOLD < p < upper]

    def integrand(z: float) -> float:
        return sigma_all(mx * z) * z * z * (z * z - 4.0) * k1(xnew * z)

    with warnings.catch_warnings():
        # 16 of the 540 positions this integrates report roundoff in
        # the extrapolation table at this tolerance. The warning is about
        # what QUADPACK can still certify, not about the value it
        # returns: asking for epsrel 1e-9, 1e-10, 1e-11 and 1e-12 in
        # turn moves the answer by at most 7.8e-10 relative, two decades
        # under the budget `deltas.py` holds the repair to. Suppressed so
        # the parity run stays quiet; widen the tolerance instead if that
        # stability ever stops holding.
        warnings.simplefilter("ignore", IntegrationWarning)
        value, _abserr = si.quad(
            integrand,
            Z_THRESHOLD,
            upper,
            points=points or None,
            epsabs=0.0,
            epsrel=REFERENCE_EPSREL,
            limit=REFERENCE_LIMIT,
        )
    return prefactor * value


ReferenceFn = Callable[[Callable[..., Any], "Block"], dict[str, np.ndarray]]


def reference_values(model: str) -> ReferenceFn:
    """A `deltas.Reference` callable for one model's thermal case.

    The entry point is not consulted: the whole point of the reference is
    that it reaches the same numbers without going through the kernel
    under repair.
    """

    def evaluate(_fn: Callable[..., Any], block: Block) -> dict[str, np.ndarray]:
        args = block.params["args"]
        return {
            "values": np.array(
                [thermal_cross_section(model, args, float(x)) for x in block.grid],
                dtype=np.float64,
            )
        }

    return evaluate
