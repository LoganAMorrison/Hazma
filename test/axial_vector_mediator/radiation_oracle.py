"""NumPy Monte Carlo of the LO pi+ pi- pi0 gamma emission.

The oracle of ``rust/src/kernels/axial_radiation.rs``. It samples four-body
phase space with Hazma's RAMBO and builds the radiative amplitude from the
two external-leg diagrams of sigma-model coordinates, each with the Born
current of ``references/lo-three-pion.md`` at its own vertex momenta. It
shares none of the kernel's analytic pair-angle integral, its recoil-frame
variables or its quadrature rules. Momenta are in MeV with metric
``diag(1, -1, -1, -1)``, and every current drops the common ``2 g3/F``.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate

from hazma.phase_space import Rambo


def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the Minkowski product over the leading axis."""
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


def current(
    q: np.ndarray, neutral: np.ndarray, charged: np.ndarray, mass: float
) -> np.ndarray:
    """Return the Born current ``-2p0 + q(q^2 - s)/(q^2 - m^2)``.

    ``charged`` is the total momentum entering the vertex on the two
    charged legs, so ``s`` is its square, and ``mass`` is the pion-pole
    mass.
    """
    q2 = dot(q, q)
    return -2.0 * neutral + q * (q2 - dot(charged, charged)) / (q2 - mass**2)


def events(
    energy: float, masses: tuple[float, float], n: int, seed: int
) -> tuple[np.ndarray, ...]:
    """Return ``(p0, p+, p-, k, weight)`` of ``n`` events at rest.

    The mean weight is the four-body phase-space volume.
    """
    neutral, charged = masses
    momenta, weights = Rambo(energy, [neutral, charged, charged, 0.0]).generate(
        n, seed=seed
    )
    return momenta[:, 0], momenta[:, 1], momenta[:, 2], momenta[:, 3], weights


def amplitude(
    momenta: tuple[np.ndarray, ...], polarization: np.ndarray, neutral_mass: float
) -> np.ndarray:
    """Return the vector ``M^nu`` with ``M = M^nu eps_gamma*_nu``, over ``e``.

    ``momenta`` is ``(p0, p+, p-, k)``. It sums the emission from each
    charged leg, a scalar-QED vertex and propagator times the Born current
    at that leg's off-shell momentum.
    """
    p0, plus, minus, k = momenta
    q = p0 + plus + minus + k
    from_plus = dot(polarization, current(q, p0, plus + k + minus, neutral_mass))
    from_minus = dot(polarization, current(q, p0, plus + minus + k, neutral_mass))
    return (2.0 * plus + k) / (2.0 * dot(plus, k)) * from_plus - (2.0 * minus + k) / (
        2.0 * dot(minus, k)
    ) * from_minus


def photon_sum(plus: np.ndarray, minus: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Return the photon polarization sum ``-J.J`` of the eikonal current."""
    j = plus / dot(plus, k) - minus / dot(minus, k)
    return -dot(j, j)


def densities(
    momenta: tuple[np.ndarray, ...], neutral_mass: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the transverse and longitudinal Born weights of each event."""
    p0, plus, minus, k = momenta
    q = p0 + plus + minus + k
    h = current(q, p0, q - p0, neutral_mass)
    q2 = dot(q, q)
    longitudinal = dot(q, h) ** 2 / q2
    return (-dot(h, h) + longitudinal) / 3.0, longitudinal


def born(energy: float, masses: tuple[float, float]) -> tuple[float, float]:
    """Return the Born ``(A_T, A_L)`` over ``(2 g3/F)^2``, in MeV^2."""
    neutral, charged = masses
    q2 = energy**2

    def measure(s: float) -> float:
        kallen = (s - (energy - neutral) ** 2) * (s - (energy + neutral) ** 2)
        return np.sqrt(max(kallen, 0.0) * (1.0 - 4.0 * charged**2 / s)) / (
            128.0 * np.pi**3 * q2
        )

    def transverse(s: float) -> float:
        kallen = (s - (energy - neutral) ** 2) * (s - (energy + neutral) ** 2)
        return measure(s) * kallen / (3.0 * q2)

    def longitudinal(s: float) -> float:
        return (
            measure(s) * (neutral**2 * (s - neutral**2) / (q2 - neutral**2)) ** 2 / q2
        )

    bounds = (4.0 * charged**2, (energy - neutral) ** 2)
    return (
        integrate.quad(transverse, *bounds, epsrel=1e-12)[0],
        integrate.quad(longitudinal, *bounds, epsrel=1e-12)[0],
    )


def photon_counts(  # noqa: PLR0913 - the source, the binning and the sample
    energy: float,
    masses: tuple[float, float],
    edges: np.ndarray,
    *,
    alpha: float,
    cut: float,
    n: int,
    seed: int,
    transverse_fraction: float = 1.0,
    lab: tuple[float, tuple[float, float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return photons per Born decay in each bin, and their standard errors.

    The cut is on the rest-frame photon energy. With ``lab`` set to
    ``(gamma, diagonal)``, the source is an on-shell mediator with Lorentz
    factor ``gamma`` along z and helicity fractions ``diagonal``
    ``(F+, F0, F-)``; its transverse weight is ``sum F_l |eps_l . H|^2``
    and the photons are binned in lab energy.
    """
    p0, plus, minus, k, weights = events(energy, masses, n, seed)
    a_transverse, a_longitudinal = born(energy, masses)
    transverse, longitudinal = densities((p0, plus, minus, k), masses[0])
    if lab is None:
        photon = k[0]
    else:
        gamma, (f_plus, f_zero, f_minus) = lab
        h = current(p0 + plus + minus + k, p0, plus + minus + k, masses[0])
        # |eps_0 . H|^2 = H_z^2 and |eps_+- . H|^2 = (H_x^2 + H_y^2)/2.
        transverse = (
            f_zero * h[3] ** 2 + (f_plus + f_minus) * (h[1] ** 2 + h[2] ** 2) / 2
        )
        beta = np.sqrt(1.0 - 1.0 / gamma**2)
        photon = gamma * (k[0] + beta * k[3])
    weight = (
        weights
        * 4.0
        * np.pi
        * alpha
        * photon_sum(plus, minus, k)
        * (
            transverse_fraction * transverse / a_transverse
            + (1.0 - transverse_fraction) * longitudinal / a_longitudinal
        )
        * (k[0] > cut)
    )
    blocks = np.array(
        [
            np.histogram(photon[i::8], bins=edges, weights=weight[i::8])[0] / (n / 8)
            for i in range(8)
        ]
    )
    return blocks.mean(axis=0), blocks.std(axis=0, ddof=1) / np.sqrt(8)
