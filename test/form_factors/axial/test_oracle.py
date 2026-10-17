"""Rust-backed axial numerics against the NumPy oracle in `axial_oracle`.

Each tolerance is a count of units in the last place of the scale both
implementations round against, with the largest difference measured on
macOS/arm64 (NumPy 2.5.4) beside it. NumPy fuses the two products of a
complex multiply into one FMA there, and the kernels do not, so the
complex-valued paths differ by rounding while the real-valued three-pion
currents are bit-identical.
"""

from __future__ import annotations

import axial_oracle as oracle
import numpy as np
import pytest
from test_three_pi import SOURCE, boost_z, point

from hazma.form_factors.axial._tensors import (
    current_projections,
    hadronic_tensor,
    integrate_spectral_functions,
    tensor_projections,
)
from hazma.form_factors.axial._three_pi import (
    AxialFormFactorPi0Pi0Pi0,
    AxialFormFactorPiPiPi0,
)

GAMMAS = [1.0, 1.5, 3.0, 10.0, 40.0]
G3 = float(SOURCE.gau)


def _ulps(difference: np.ndarray, scale: np.ndarray) -> float:
    return float(np.max(np.abs(difference) / np.spacing(np.abs(scale))))


def _boosted(gamma: float, batch: tuple[int, ...] = (64,)) -> tuple:
    """Random complex currents and a 500 MeV source, boosted together."""
    rng = np.random.default_rng(int(gamma * 10))
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    boost = np.eye(4)
    boost[0, 0] = gamma
    boost[0, 1:] = boost[1:, 0] = np.sqrt(gamma**2 - 1) * direction
    boost[1:, 1:] += (gamma - 1) * np.outer(direction, direction)
    current = rng.normal(size=(4, *batch)) + 1j * rng.normal(size=(4, *batch))
    rest = np.zeros((4, *batch))
    rest[0] = 500.0
    return np.tensordot(boost, current, 1), np.tensordot(boost, rest, 1)


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("batch", [(), (64,), (3, 5)])
def test_hadronic_tensor(gamma: float, batch: tuple[int, ...]) -> None:
    current, _ = _boosted(gamma, batch)
    magnitude = np.abs(current)[:, None] * np.abs(current)[None, :]
    # One rounding per entry, fused or not: measured 1.4 ulps.
    limit = 2
    difference = hadronic_tensor(current) - oracle.hadronic_tensor(current)
    assert _ulps(difference, magnitude) <= limit


@pytest.mark.parametrize("gamma", GAMMAS)
def test_current_projections(gamma: float) -> None:
    current, q = _boosted(gamma)
    expected = oracle.current_projections(current, q)
    # Both carry the gamma**2 eps error the docstring states, so the scale is
    # gamma**2 times the rest-frame norm: measured 2 ulps.
    scale = gamma**2 * (expected[1] + 3 * expected[0])
    limit = 4
    for actual, reference in zip(
        current_projections(current, q), expected, strict=True
    ):
        assert _ulps(actual - reference, scale) <= limit


@pytest.mark.parametrize("gamma", [1.0, 1.5, 3.0, 10.0])
@pytest.mark.parametrize("batch", [(), (64,)])
def test_tensor_projections(gamma: float, batch: tuple[int, ...]) -> None:
    current, q = _boosted(gamma, batch)
    tensor = oracle.hadronic_tensor(current)
    s = q[0] ** 2 - np.sum(q[1:] ** 2, axis=0)
    covariant = q * np.array([1.0, -1.0, -1.0, -1.0]).reshape((4,) + (1,) * len(batch))
    terms = covariant[:, None] * covariant[None, :] * tensor / s
    diagonal = np.abs(np.diagonal(tensor, axis1=0, axis2=1)).sum(axis=-1)
    budget = 128 * np.finfo(float).eps * (np.abs(terms).sum(axis=(0, 1)) + diagonal)
    # Each lies within the guard's roundoff budget of the exact projection,
    # so 2 * budget bounds their difference; summation order and FMA leave
    # it at 3.7e-3 of the budget, measured.
    for actual, reference in zip(
        tensor_projections(tensor, q), oracle.tensor_projections(tensor, q), strict=True
    ):
        assert np.all(np.abs(actual - reference) <= budget / 32)


def _q_proportional(gamma: float) -> tuple[np.ndarray, np.ndarray]:
    q = np.array([gamma, 0.0, 0.0, np.sqrt(gamma**2 - 1)])
    return oracle.hadronic_tensor(q), q


def _negative() -> tuple[np.ndarray, np.ndarray]:
    tensor = np.zeros((4, 4, 2), dtype=complex)
    tensor[0, 0] = 1.0
    tensor[1, 1] = [0.0, -1.0]
    return tensor, np.array([1.0, 0.0, 0.0, 0.0])


def _overflow() -> tuple[np.ndarray, np.ndarray]:
    tensor = np.zeros((4, 4), dtype=complex)
    tensor[1, 1] = tensor[2, 2] = np.finfo(float).max
    return tensor, np.array([1.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("arguments", "rtol"),
    [
        (_q_proportional(1e4), 1e-6),
        (_q_proportional(1e4), 0.5),
        (_q_proportional(1.0), 1e-16),
        (_negative(), 1e-6),
        (_overflow(), 1e-6),
    ],
)
def test_tensor_projection_errors_match(arguments: tuple, rtol: float) -> None:
    with pytest.raises(ValueError) as expected:
        oracle.tensor_projections(*arguments, rtol=rtol)
    with pytest.raises(ValueError) as actual:
        tensor_projections(*arguments, rtol=rtol)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize(
    ("batch", "sample_axis", "limit"), [((5, 64), -1, 4), ((64, 5), 0, 8)]
)
def test_integrate_spectral_functions(
    batch: tuple[int, int], sample_axis: int, limit: int
) -> None:
    current, q = _boosted(1.5, batch)
    weights = np.random.default_rng(3).random(64).reshape((64, 1)[: 2 + sample_axis])
    actual = integrate_spectral_functions(current, q, weights, sample_axis=sample_axis)
    expected = oracle.integrate_spectral_functions(
        current, q, weights, sample_axis=sample_axis
    )
    # The densities carry the current_projections rounding above. Pairwise
    # summation reproduces NumPy's reduction of a last axis and adds none,
    # measured 2 ulps; NumPy sums any other axis sequentially, 5 ulps over 64.
    for value, reference in zip(actual, expected, strict=True):
        assert _ulps(value - reference, reference) <= limit


def _events(mass: float) -> np.ndarray:
    return np.stack(
        [
            boost_z(point(mass=mass, fraction=fraction, cosine=cosine))
            for fraction in (0.1, 0.4, 0.9)
            for cosine in (-0.7, 0.3)
        ],
        axis=-1,
    )


@pytest.mark.parametrize("cls", [AxialFormFactorPiPiPi0, AxialFormFactorPi0Pi0Pi0])
@pytest.mark.parametrize("mass", [135.0, 0.0])
def test_three_pi_current_and_weight(cls: type, mass: float) -> None:
    form_factor = cls(pion_mass=mass)
    momenta = _events(mass)
    arguments = (momenta, G3, mass, 92.2, form_factor._neutral)
    # The same IEEE operations in the same order: measured bit-identical.
    current = oracle.three_pi_current(*arguments)
    assert _ulps(form_factor.current(momenta, SOURCE) - current, current) <= 1
    weight = oracle.three_pi_differential_weight(*arguments)
    assert _ulps(form_factor.differential_weight(momenta, SOURCE) - weight, weight) <= 1


@pytest.mark.parametrize("cls", [AxialFormFactorPiPiPi0, AxialFormFactorPi0Pi0Pi0])
@pytest.mark.parametrize("mass", [135.0, 0.0])
@pytest.mark.parametrize("order", [8, 64, 512])
def test_three_pi_spectral_functions(cls: type, mass: float, order: int) -> None:
    form_factor = cls(pion_mass=mass)
    energies = np.append(np.linspace(0.0, 600.0, 241), 3 * mass * (1 + 1e-9))
    result = form_factor.spectral_functions(energies, SOURCE, quadrature_order=order)
    expected = oracle.three_pi_spectral_functions(
        energies, G3, mass, 92.2, form_factor._neutral, quadrature_order=order
    )
    # The kernel finds the Gauss-Legendre nodes by Newton's method and NumPy
    # by an eigensolver; the rules differ by an ulp or two: measured 7 ulps.
    limit = 8
    for actual, reference in zip(
        (result.transverse, result.source_longitudinal), expected, strict=True
    ):
        assert _ulps(actual - reference, reference) <= limit
