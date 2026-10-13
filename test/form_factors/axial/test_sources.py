"""Independent generator normalization and explicit source-input contracts."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from hazma.form_factors.axial._sources import AxialCurrent, MatchingCard, SourceCoverage

METADATA = {"qcd_scheme": "reference", "scale_mev": 1000.0, "provenance": "test"}
# Exact generator algebra only accumulates floating-point square-root rounding.
ALGEBRA_ATOL = 2.0e-15
SHA256_HEX_LENGTH = 64


def generators() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.diag([1.0, -1.0, 0.0]),
        np.diag([1.0, 1.0, -2.0]) / np.sqrt(3.0),
        np.sqrt(2.0 / 3.0) * np.eye(3),
    )


def test_generator_trace_normalization() -> None:
    basis = generators()
    gram = [[np.trace(a @ b) for b in basis] for a in basis]
    np.testing.assert_allclose(gram, 2.0 * np.eye(3), rtol=0, atol=ALGEBRA_ATOL)


@pytest.mark.parametrize(
    "flavors", [(1, -1, 0), (1, 1, -2), (1, 1, 1), (1, -1, -1), (0.7, -0.2, 1.3)]
)
def test_projection_and_inverse_against_matrix_trace(
    flavors: tuple[float, float, float],
) -> None:
    source = AxialCurrent(*flavors, **METADATA)
    matrix = np.diag(flavors)
    basis = generators()
    independent = np.array([np.trace(matrix @ generator) / 2.0 for generator in basis])
    np.testing.assert_allclose(
        source.generator_coefficients, independent, rtol=0, atol=ALGEBRA_ATOL
    )
    reconstructed = sum(
        c * t for c, t in zip(source.generator_coefficients, basis, strict=True)
    )
    np.testing.assert_allclose(reconstructed, matrix, rtol=0, atol=ALGEBRA_ATOL)
    inverse = AxialCurrent.from_generator_coefficients(*independent, **METADATA)
    np.testing.assert_allclose(
        [inverse.gau, inverse.gad, inverse.gas], flavors, rtol=0, atol=ALGEBRA_ATOL
    )


def test_traceful_hypercharge_direction_is_not_projected_out() -> None:
    source = AxialCurrent(1, -1, -1, **METADATA)
    assert source.trace == -1.0
    np.testing.assert_allclose(
        source.generator_coefficients,
        [1.0, 1.0 / np.sqrt(3.0), -1.0 / np.sqrt(6.0)],
        rtol=0,
        atol=ALGEBRA_ATOL,
    )


def test_finite_singlet_convention_requires_compensating_residue() -> None:
    source = AxialCurrent(1, -1, -1, **METADATA)
    factor = 1.7
    rescaled = source.rescale_singlet(
        factor, qcd_scheme="finite-singlet-rescaling", provenance="declared convention"
    )
    residues = np.array([0.7 + 0.3j, -0.2j, 0.8 - 0.4j])
    transformed = residues.copy()
    transformed[2] *= factor
    original_amplitude = np.dot(source.generator_coefficients, residues)
    new_amplitude = np.dot(rescaled.generator_coefficients, transformed)
    np.testing.assert_allclose(
        new_amplitude, original_amplitude, rtol=0, atol=ALGEBRA_ATOL
    )
    assert rescaled.scale_mev == source.scale_mev
    assert rescaled.qcd_scheme == "finite-singlet-rescaling"


def test_source_immutable_hash_and_provenance() -> None:
    source = AxialCurrent(1, -1, -1, **METADATA)
    with pytest.raises(FrozenInstanceError):
        source.gau = 2.0
    assert hash(source) == hash(AxialCurrent(1.0, -1.0, -1.0, **METADATA))
    assert source.fingerprint == AxialCurrent(1, -1, -1, **METADATA).fingerprint
    assert len(source.fingerprint) == SHA256_HEX_LENGTH
    assert source.fingerprint != replace(source, provenance="another input").fingerprint
    assert source.fingerprint != replace(source, scale_mev=2000.0).fingerprint


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1j, "1", [1], True])
@pytest.mark.parametrize("name", ["gau", "gad", "gas"])
def test_illegal_flavor_coefficients(name: str, bad: object) -> None:
    args = {"gau": 1, "gad": -1, "gas": 0, **METADATA}
    args[name] = bad
    with pytest.raises(ValueError):
        AxialCurrent(**args)


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf, 1j, True])
def test_illegal_source_scale(bad: object) -> None:
    with pytest.raises(ValueError):
        AxialCurrent(1, -1, 0, **{**METADATA, "scale_mev": bad})


@pytest.mark.parametrize("name", ["qcd_scheme", "provenance"])
@pytest.mark.parametrize("bad", ["", " ", None, 3])
def test_metadata_required_and_nonempty(name: str, bad: object) -> None:
    with pytest.raises(ValueError):
        AxialCurrent(1, -1, 0, **{**METADATA, name: bad})


@pytest.mark.parametrize("bad", [0, np.nan, np.inf, 1j, True])
def test_illegal_singlet_rescaling(bad: object) -> None:
    with pytest.raises(ValueError):
        AxialCurrent(1, -1, -1, **METADATA).rescale_singlet(
            bad, qcd_scheme="new", provenance="test"
        )


def test_generator_overflow_is_a_controlled_domain_error() -> None:
    largest = np.finfo(float).max
    with pytest.raises(ValueError):
        _ = AxialCurrent(largest, largest, largest, **METADATA).generator_coefficients
    with pytest.raises(ValueError):
        AxialCurrent.from_generator_coefficients(largest, largest, largest, **METADATA)


def test_matching_card_distinguishes_input_scales_and_completion() -> None:
    source = AxialCurrent(1, -1, -1, **METADATA)
    chiral_scale = 770
    card = MatchingCard(
        source, chiral_scale, "explicit axial convention", None, None, "lo", "test"
    )
    assert card.current.scale_mev == source.scale_mev
    assert card.chiral_scale_mev == chiral_scale
    assert card.anomaly_matching is None
    assert card.uv_completion is None
    assert card.input_status == "lo"
    with pytest.raises(FrozenInstanceError):
        card.input_status = "fitted"
    for changes in (
        {"chiral_scale_mev": 0},
        {"normalization": ""},
        {"anomaly_matching": ""},
        {"uv_completion": ""},
        {"input_status": "precision"},
        {"current": None},
    ):
        with pytest.raises(ValueError):
            replace(card, **changes)


def test_coverage_domain_arrays_and_immutable_missing_inputs() -> None:
    directions = ["octet", "singlet"]
    coverage = SourceCoverage("3pi", 400, 600, "lo", directions, provenance="test")
    directions.append("unknown")
    assert coverage.missing_directions == ("octet", "singlet")
    assert coverage.contains(400) is True
    assert coverage.contains(600) is True
    assert coverage.contains(399) is False
    np.testing.assert_array_equal(
        coverage.contains([[0, 400], [600, 601]]), [[False, True], [True, False]]
    )
    with pytest.raises(FrozenInstanceError):
        coverage.q_max_mev = 800
    for changes in (
        {"q_min_mev": -1},
        {"q_max_mev": 399},
        {"q_max_mev": np.inf},
        {"input_status": "validated"},
        {"missing_directions": "singlet"},
        {"missing_directions": [""]},
    ):
        with pytest.raises(ValueError):
            replace(coverage, **changes)


@pytest.mark.parametrize("bad", [-1, np.nan, np.inf, [400, np.nan], 1j, "400", True])
def test_coverage_rejects_invalid_masses(bad: object) -> None:
    coverage = SourceCoverage("3pi", 400, 600, "lo", provenance="test")
    with pytest.raises(ValueError):
        coverage.contains(bad)
