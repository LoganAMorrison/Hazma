"""Independent spin traces and rest-frame oracles for axial projectors."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hazma.form_factors.axial._base import AxialSpectralResult
from hazma.form_factors.axial._sources import SourceCoverage
from hazma.form_factors.axial._tensors import (
    current_projections,
    hadronic_tensor,
    integrate_spectral_functions,
    tensor_projections,
)
from hazma.utils import ComplexArray, RealArray


def _gamma_matrices() -> tuple[ComplexArray, ComplexArray]:
    """Dirac representation; the oracle does not use the projector formulas."""
    identity = np.eye(2)
    zero = np.zeros((2, 2))
    pauli = (
        np.array([[0, 1], [1, 0]]),
        np.array([[0, -1j], [1j, 0]]),
        np.array([[1, 0], [0, -1]]),
    )
    gamma = [np.block([[identity, zero], [zero, -identity]])]
    gamma.extend(np.block([[zero, sigma], [-sigma, zero]]) for sigma in pauli)
    gamma = np.array(gamma)
    gamma5 = 1j * gamma[0] @ gamma[1] @ gamma[2] @ gamma[3]
    return gamma, gamma5


def _axial_spin_trace(
    momentum: RealArray, antiparticle: RealArray, mass: float
) -> ComplexArray:
    gamma, gamma5 = _gamma_matrices()
    pslash = momentum[0] * gamma[0] - np.einsum("i,ijk->jk", momentum[1:], gamma[1:])
    kslash = antiparticle[0] * gamma[0] - np.einsum(
        "i,ijk->jk", antiparticle[1:], gamma[1:]
    )
    vertices = gamma @ gamma5
    result = np.empty((4, 4), dtype=complex)
    for mu in range(4):
        for nu in range(4):
            # Dirac conjugate of gamma^nu gamma5 equals gamma^nu gamma5.
            result[mu, nu] = np.trace(
                (pslash + mass * np.eye(4))
                @ vertices[mu]
                @ (kslash - mass * np.eye(4))
                @ vertices[nu]
            )
    return result


@pytest.mark.parametrize("mass", [0.0, 25.0, 199.9])
def test_free_fermion_axial_optical_normalization(mass: float) -> None:
    """Spin-summed unit-coupling fermion pair, with no color or cut 1/2.

    Direct Dirac traces and dPhi2 = beta dOmega/(32 pi²) integrate to
    A_T = s beta³/(6 pi), A_L = m² beta/pi. The 00 and spatial blocks
    are independently read in the CM, so a swapped projector fails.
    """
    energy = 400.0
    beta = np.sqrt(1 - (2 * mass / energy) ** 2)
    k = energy * beta / 2
    # Four Gauss nodes integrate the quadratic angular spin trace exactly;
    # eight equally spaced azimuths integrate its harmonics exactly.
    cosines, weights = np.polynomial.legendre.leggauss(4)
    integrated = np.zeros((4, 4), dtype=complex)
    for cosine, weight in zip(cosines, weights, strict=True):
        sine = np.sqrt(1 - cosine**2)
        for phi in np.arange(8) * 2 * np.pi / 8:
            spatial = k * np.array([sine * np.cos(phi), sine * np.sin(phi), cosine])
            p = np.concatenate(([energy / 2], spatial))
            antiparticle = np.concatenate(([energy / 2], -spatial))
            integrated += (
                _axial_spin_trace(p, antiparticle, mass)
                * weight
                * (2 * np.pi / 8)
                * beta
                / (32 * np.pi**2)
            )
    transverse, longitudinal = tensor_projections(integrated, [energy, 0, 0, 0])
    expected_t = energy**2 * beta**3 / (6 * np.pi)
    expected_l = mass**2 * beta / np.pi
    # Double-precision traces/quadrature; near threshold cancellation incurs
    # ~1e-13 relative error, far below the chosen 2e-11 budget.
    assert_allclose(transverse, expected_t, rtol=2e-11, atol=1e-10)
    assert_allclose(longitudinal, expected_l, rtol=2e-11, atol=1e-10)
    assert_allclose(integrated[0, 0], expected_l, rtol=2e-11, atol=1e-10)
    assert_allclose(integrated[1:, 1:], np.eye(3) * expected_t, rtol=2e-11, atol=1e-10)
    if mass == 0:
        # The axial massless lepton current is conserved in the free theory.
        assert_allclose(integrated[0], 0, rtol=0, atol=1e-10)


def test_complex_interference_tensor_and_lorentz_invariance() -> None:
    current = np.array([2 + 3j, 1 - 5j, -4 + 2j, 7 + 1j])
    rest_q = np.array([13.0, 0, 0, 0])
    direction = np.array([2.0, -1.0, 3.0]) / np.sqrt(14)
    velocity = 0.63
    gamma = 1 / np.sqrt(1 - velocity**2)
    # An explicit Lorentz matrix independent of the implementation's boost.
    boost = np.eye(4)
    boost[0, 0] = gamma
    boost[0, 1:] = boost[1:, 0] = gamma * velocity * direction
    boost[1:, 1:] += (gamma - 1) * np.outer(direction, direction)
    boosted = boost @ current
    q = boost @ rest_q
    expected = (np.vdot(current[1:], current[1:]).real / 3, abs(current[0]) ** 2)
    # The transformation and contractions each take only a few float ops.
    assert_allclose(current_projections(boosted, q), expected, rtol=3e-14, atol=1e-13)
    tensor = hadronic_tensor(boosted)
    assert_allclose(
        tensor, boost @ hadronic_tensor(current) @ boost.T, rtol=3e-14, atol=1e-13
    )
    # Complex multiplication can leave a last-bit imaginary diagonal residue.
    assert_allclose(tensor, tensor.conj().T, rtol=2e-15, atol=1e-14)
    assert_allclose(tensor_projections(tensor, q), expected, rtol=3e-14, atol=1e-13)
    assert tensor[0, 1].imag != 0
    assert_allclose(hadronic_tensor(current)[0, 1], (2 + 3j) * (1 + 5j), rtol=0, atol=0)


def test_source_longitudinal_does_not_mean_helicity_zero() -> None:
    q = np.array([10.0, 0, 0, 6.0])
    # A q-proportional off-shell source (e.g. a neutral two-photon response).
    pure_source = (2 - 3j) * q
    transverse, longitudinal = current_projections(pure_source, q)
    assert_allclose(transverse, 0, rtol=0, atol=1e-27)
    assert_allclose(longitudinal, 13 * 64, rtol=2e-15, atol=0)
    # Physical helicity-zero massive-vector polarization: q.epsilon = 0.
    epsilon_zero = np.array([6.0, 0, 0, 10.0]) / 8
    transverse, longitudinal = current_projections(epsilon_zero, q)
    assert_allclose(transverse, 1 / 3, rtol=2e-15, atol=0)
    assert_allclose(longitudinal, 0, rtol=0, atol=0)


def test_batch_axes_broadcast_and_phase_space_weights() -> None:
    current = np.array([2j, 3, 4j, 0])[:, None, None] * np.array([[1, 2, 3], [2, 4, 6]])
    q = np.array([10.0, 0, 0, 0])
    transverse, longitudinal = current_projections(current, q)
    scale_squared = np.array([[1, 4, 9], [4, 16, 36]])
    assert_allclose(transverse, 25 * scale_squared / 3, rtol=0, atol=1e-13)
    assert_allclose(longitudinal, 4 * scale_squared, rtol=0, atol=0)
    assert hadronic_tensor(current).shape == (4, 4, 2, 3)
    weights = np.array([0.1, 0.2, 0.3])
    integrated = integrate_spectral_functions(current, q, weights)
    # Direct CM components and weighted sum independently pin the measure,
    # including the absence of an extraneous 1/2 optical-cut factor.
    expected = np.array([25 / 3, 4])[:, None] * np.array([3.6, 14.4])
    assert_allclose(integrated, expected, rtol=2e-15, atol=0)
    assert_allclose(
        tensor_projections(hadronic_tensor(current), q),
        (transverse, longitudinal),
        rtol=2e-15,
        atol=0,
    )
    # Batch-only q dimensions broadcast before their Lorentz axis.
    batched_q = np.array([[10.0, 20.0, 30.0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert_allclose(
        current_projections(current, batched_q),
        (transverse, longitudinal),
        rtol=2e-15,
        atol=0,
    )
    assert np.shape(current_projections([1, 2, 3, 4], q)[0]) == ()
    assert_allclose(
        integrate_spectral_functions(current, q, weights, sample_axis=0),
        np.stack((transverse * weights, longitudinal * weights)).sum(axis=1),
        rtol=2e-15,
        atol=0,
    )


@pytest.mark.parametrize(
    "q",
    [
        [0, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 2, 0, 0],
        [-1, 0, 0, 0],
        [np.nan, 0, 0, 0],
        [np.inf, 0, 0, 0],
        [1e308, 0, 0, 0],
        [1, 2, 3],
        [1j, 0, 0, 0],
    ],
)
def test_invalid_momentum_is_rejected(q: list[float | complex]) -> None:
    with pytest.raises(ValueError):
        current_projections([1, 2, 3, 4], q)
    with pytest.raises(ValueError):
        tensor_projections(np.eye(4), q)


@pytest.mark.parametrize("current", [[1, 2, 3], [np.nan, 0, 0, 0], [np.inf, 0, 0, 0]])
def test_invalid_current_is_rejected(current: list[float]) -> None:
    with pytest.raises(ValueError):
        hadronic_tensor(current)
    with pytest.raises(ValueError):
        current_projections(current, [1, 0, 0, 0])


def test_invalid_tensors_weights_axes_and_overflow() -> None:
    q = [2, 0, 0, 0]
    for tensor in (np.zeros((3, 3)), np.full((4, 4), np.nan), np.eye(4) * -1):
        with pytest.raises(ValueError):
            tensor_projections(tensor, q)
    nonhermitian = np.eye(4, dtype=complex)
    nonhermitian[0, 1] = 1j
    with pytest.raises(ValueError, match="Hermitian"):
        tensor_projections(nonhermitian, q)
    for weights in ([-1], [np.nan], [np.inf], [1j]):
        with pytest.raises(ValueError):
            integrate_spectral_functions(np.ones((4, 1)), q, weights)
    for axis in (1, -2, True, 0.5):
        with pytest.raises(ValueError):
            integrate_spectral_functions(np.ones((4, 1)), q, [1], sample_axis=axis)
    with pytest.raises(ValueError, match="sample_axis"):
        integrate_spectral_functions(np.ones(4), q, 1)
    with pytest.raises(ValueError, match="overflowed"):
        hadronic_tensor([1e308, 0, 0, 0])
    with pytest.raises(ValueError, match="overflowed"):
        current_projections([0, 1e308, 0, 0], q)


def _coverage() -> SourceCoverage:
    return SourceCoverage("test", 0, 500, "lo", provenance="analytic test input")


def test_spectral_result_scalars_arrays_units_and_immutability() -> None:
    original = np.array([0.0, 2.0])
    result = AxialSpectralResult(
        [100, 300],
        original,
        [0, 4],
        threshold_mev=200,
        coverage=_coverage(),
        units="MeV^2",
    )
    original[1] = 9
    assert_allclose(result.transverse, [0, 2], rtol=0, atol=0)
    with pytest.raises(ValueError):
        result.transverse[1] = 7
    with pytest.raises(ValueError):
        result.transverse.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        result.units = "other"
    scalar = AxialSpectralResult(
        300, 2, 4, threshold_mev=200, coverage=_coverage(), units="dimensionless"
    )
    assert isinstance(scalar.transverse, float)
    assert scalar.units == "dimensionless"
    broadcast = AxialSpectralResult(
        [300, 400], 2, 4, threshold_mev=200, coverage=_coverage(), units="MeV^4"
    )
    assert_allclose(broadcast.transverse, [2, 2], rtol=0, atol=0)


@pytest.mark.parametrize(
    "changes",
    [
        {"q_mev": 501},
        {"q_mev": np.nan},
        {"q_mev": -1},
        {"q_mev": 200},
        {"transverse": -1},
        {"source_longitudinal": np.inf},
        {"threshold_mev": -1},
        {"threshold_mev": np.nan},
        {"threshold_mev": 200 + 0j},
        {"threshold_mev": True},
        {"threshold_mev": "200"},
        {"units": ""},
        {"transverse": 1j},
    ],
)
def test_spectral_result_rejects_invalid_metadata_or_values(
    changes: dict[str, object],
) -> None:
    args = dict(
        q_mev=300,
        transverse=2,
        source_longitudinal=4,
        threshold_mev=200,
        coverage=_coverage(),
        units="MeV^2",
    )
    args.update(changes)
    with pytest.raises(ValueError):
        AxialSpectralResult(**args)


def _boost_along_z(gamma: float) -> RealArray:
    beta_gamma = np.sqrt(gamma**2 - 1)
    boost = np.eye(4)
    boost[0, 0] = boost[3, 3] = gamma
    boost[0, 3] = boost[3, 0] = beta_gamma
    return boost


def test_boosted_tensor_projection_rejects_lost_precision() -> None:
    # Review case: a q-proportional source at gamma = 1e4. The exact answer is
    # transverse = 0 and source_longitudinal = s = 1; the old contraction
    # returned (1.667, 6.0).
    q = np.array([np.sqrt(100000001.0), 0.0, 0.0, 10000.0])
    # The amplitude path keeps the error near gamma**2 * eps.
    assert_allclose(current_projections(q, q), (0.0, 1.0), rtol=0, atol=3e-8)
    with pytest.raises(ValueError, match="ill-conditioned"):
        tensor_projections(hadronic_tensor(q), q)
    # Loosening rtol does not admit a result with no correct digit.
    with pytest.raises(ValueError, match="ill-conditioned"):
        tensor_projections(hadronic_tensor(q), q, rtol=0.5)


def test_moderately_boosted_tensor_projection_is_accepted() -> None:
    current = np.array([1.5 - 0.5j, 0.3 + 2j, -1.2 + 0.4j, 0.7 - 1.1j])
    boost = _boost_along_z(10.0)
    boosted = boost @ current
    q = boost @ np.array([3.0, 0.0, 0.0, 0.0])
    expected = current_projections(boosted, q)
    norm = expected[1] + 3 * expected[0]
    # At gamma = 10 the measured error is about 1e-12 of the norm.
    assert_allclose(
        tensor_projections(hadronic_tensor(boosted), q),
        expected,
        rtol=0,
        atol=1e-10 * norm,
    )
    # The same input fails a tolerance tighter than its roundoff bound.
    with pytest.raises(ValueError, match="ill-conditioned"):
        tensor_projections(hadronic_tensor(boosted), q, rtol=1e-12)


@pytest.mark.parametrize("rtol", [0, 1, -1e-6, np.nan, np.inf, True, "1e-6"])
def test_invalid_projection_tolerance_is_rejected(rtol: object) -> None:
    with pytest.raises(ValueError, match="rtol"):
        tensor_projections(np.eye(4), [2, 0, 0, 0], rtol=rtol)  # type: ignore[arg-type]
