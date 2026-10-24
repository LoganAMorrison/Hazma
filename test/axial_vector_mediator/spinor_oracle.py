"""Explicit-spinor oracle for dark-matter annihilation into a fermion pair.

It builds Dirac spinors in the Dirac representation, contracts the dark and
fermion currents through the Proca propagator split into its transverse
pole and its contact, sums the squared amplitude over all sixteen spin
states and integrates over the scattering angle. It never forms ``A_T`` or
``A_L``, so it checks the model's decomposition rather than repeating it.
Nothing under ``hazma/`` imports it.
"""

from __future__ import annotations

import numpy as np

_PAULI = (
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
)
_I2 = np.eye(2, dtype=complex)
_Z2 = np.zeros((2, 2), dtype=complex)
_GAMMA = (np.block([[_I2, _Z2], [_Z2, -_I2]]),) + tuple(
    np.block([[_Z2, s], [-s, _Z2]]) for s in _PAULI
)
_GAMMA5 = np.block([[_Z2, _I2], [_I2, _Z2]])
_METRIC = np.diag([1.0, -1.0, -1.0, -1.0])
_BASIS = (np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex))


def _sigma_dot(p: np.ndarray) -> np.ndarray:
    return sum(component * s for component, s in zip(p[1:], _PAULI, strict=True))


def _u(p: np.ndarray, mass: float, xi: np.ndarray) -> np.ndarray:
    root = np.sqrt(p[0] + mass)
    return np.concatenate([root * xi, _sigma_dot(p) @ xi / root])


def _v(p: np.ndarray, mass: float, eta: np.ndarray) -> np.ndarray:
    root = np.sqrt(p[0] + mass)
    return np.concatenate([_sigma_dot(p) @ eta / root, root * eta])


def _helicity_spinors(p: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Return ``(helicity, two-spinor)`` for the eigenstates of ``sigma . p_hat``."""
    values, vectors = np.linalg.eigh(_sigma_dot(p / np.linalg.norm(p[1:])))
    return [(round(value), vectors[:, column]) for column, value in enumerate(values)]


def _bar(spinor: np.ndarray) -> np.ndarray:
    return spinor.conj() @ _GAMMA[0]


def _current(left: np.ndarray, right: np.ndarray, gv: float, ga: float) -> np.ndarray:
    """Return ``left gamma^mu (gv + ga gamma5) right`` with an upper index."""
    vertex = gv * np.eye(4) + ga * _GAMMA5
    return np.array([left @ gamma @ vertex @ right for gamma in _GAMMA])


def sigma_v_fermion_pair(  # noqa: PLR0913 - one parameter point
    e_cm: float,
    *,
    mx: float,
    gvxx: float,
    gaxx: float,
    mv: float,
    width: float,
    mf: float,
    gvf: float,
    gaf: float,
    nodes: int = 8,
) -> float:
    r"""Return :math:`\sigma v_\mathrm{Mol}` in MeV⁻² for one fermion pair.

    The propagator is ``P_T/(s - M^2 + i M Gamma) - k k/(s M^2)`` with
    lower indices, which at zero width is ``(g - k k/M^2)/(s - M^2)``.
    The squared amplitude is a quadratic in ``cos(theta)``, so an
    ``nodes``-point Gauss-Legendre rule integrates it exactly for
    ``nodes >= 2``.
    """
    return _sigma_v_by_helicity(
        e_cm,
        mx=mx,
        gvxx=gvxx,
        gaxx=gaxx,
        mv=mv,
        width=width,
        mf=mf,
        gvf=gvf,
        gaf=gaf,
        nodes=nodes,
    )[0]


def fermion_polarization(  # noqa: PLR0913 - one parameter point
    e_cm: float,
    *,
    mx: float,
    gvxx: float,
    gaxx: float,
    mv: float,
    width: float,
    mf: float,
    gvf: float,
    gaf: float,
    nodes: int = 8,
) -> float:
    """Return the mean helicity of the fermion (not the antifermion) of the pair.

    It is the rate weighted by the fermion's helicity over the rate, with the
    fermion's spinors taken as helicity eigenstates along its momentum and the
    antifermion's summed in any basis.
    """
    total, weighted = _sigma_v_by_helicity(
        e_cm,
        mx=mx,
        gvxx=gvxx,
        gaxx=gaxx,
        mv=mv,
        width=width,
        mf=mf,
        gvf=gvf,
        gaf=gaf,
        nodes=nodes,
    )
    return weighted / total


def _sigma_v_by_helicity(  # noqa: PLR0913 - one parameter point
    e_cm: float,
    *,
    mx: float,
    gvxx: float,
    gaxx: float,
    mv: float,
    width: float,
    mf: float,
    gvf: float,
    gaf: float,
    nodes: int,
) -> tuple[float, float]:
    """Return ``sigma v`` and the same weighted by the fermion's helicity."""
    s = e_cm**2
    energy = e_cm / 2.0
    p = np.sqrt(energy**2 - mx**2)
    pf = np.sqrt(energy**2 - mf**2)
    k = np.array([e_cm, 0.0, 0.0, 0.0])
    k_lower = _METRIC @ k
    transverse = _METRIC - np.outer(k_lower, k_lower) / s
    propagator = transverse / (s - mv**2 + 1j * mv * width) - np.outer(
        k_lower, k_lower
    ) / (s * mv**2)

    p1 = np.array([energy, 0.0, 0.0, p])
    p2 = np.array([energy, 0.0, 0.0, -p])
    dark = [
        _current(_bar(_v(p2, mx, _BASIS[b])), _u(p1, mx, _BASIS[a]), gvxx, gaxx)
        for a in range(2)
        for b in range(2)
    ]

    cosines, weights = np.polynomial.legendre.leggauss(nodes)
    integral = 0.0
    weighted = 0.0
    for cosine, weight in zip(cosines, weights, strict=True):
        sine = np.sqrt(1.0 - cosine**2)
        k1 = np.array([energy, pf * sine, 0.0, pf * cosine])
        k2 = np.array([energy, -pf * sine, 0.0, -pf * cosine])
        squared = 0.0
        squared_weighted = 0.0
        for helicity, xi in _helicity_spinors(k1):
            for eta in _BASIS:
                fermion = _current(_bar(_u(k1, mf, xi)), _v(k2, mf, eta), gvf, gaf)
                for current in dark:
                    term = abs(current @ propagator @ fermion) ** 2
                    squared += term
                    squared_weighted += helicity * term
        integral += weight * squared / 4.0
        weighted += weight * squared_weighted / 4.0
    beta = pf / energy
    # dPhi_2 = beta/(8 pi) d(cos theta)/2, and sigma v_Mol divides by
    # 4 E1 E2 = s in the CM.
    factor = beta / (16.0 * np.pi) / s
    return float(factor * integral), float(factor * weighted)
