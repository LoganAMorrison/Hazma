"""Independent A3 repair checks beyond the ordinary corpus comparison."""

from __future__ import annotations

import json
from pathlib import Path

import cases
import deltas
import generate
import numpy as np
import oracle_reference
import pytest


@pytest.mark.parametrize("mass", [550, 900])
def test_scalar_rest_composition_resolves_a3_beneath_b4_budget(mass: int) -> None:
    """At rest the FSR addition has no adaptive-partition error to absorb A3."""
    case = cases.build_cases()[
        "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum"
    ]
    block = next(b for b in case.blocks if b.label == f"ms_{mass}.rest.default")
    live, _ = generate.evaluate_block(case.resolve(), block)
    predicted = deltas.DELTA_MODELS["A3+B4"].relation.expected(
        case.resolve(), block, {}
    )
    for suffix, reference in predicted.items():
        # Measured <6e-16; 1e-12 is the port's arithmetic budget. B4's
        # 1e-3 flight-quadrature budget would hide the smaller A3 correction.
        np.testing.assert_allclose(live[suffix], reference, rtol=1e-12, atol=0)


def test_pion_stored_zeros_survive_only_outside_physical_support() -> None:
    """Every previously missed positive-energy interior point is restored."""
    name = "spectra.photon.charged_pion"
    case = cases.build_cases()[name]
    manifest = json.loads((Path(__file__).parent / "data/manifest.json").read_text())
    missed = 0
    with np.load(generate.DATA_DIR / manifest["cases"][name]["file"]) as data:
        for block, stored in zip(
            case.blocks, manifest["cases"][name]["blocks"], strict=True
        ):
            parent = block.params["parent_energy"]
            mass, electron = 139.57039, 0.5109989461
            gamma = parent / mass
            beta = np.sqrt(1 - gamma**-2)
            endpoint = (mass**2 - electron**2) / (2 * mass) * gamma * (1 + beta)
            original = data[stored["arrays"]["values"]["key"]]
            live, _ = generate.evaluate_block(case.resolve(), block)
            inside = (block.grid > 0) & (block.grid < endpoint * (1 - 1e-10))
            lost = (original == 0) & inside
            missed += np.count_nonzero(lost)
            assert np.all(live["values"][lost] > 0)
            assert np.all(live["values"][block.grid > endpoint * (1 + 1e-10)] == 0)
    assert missed > 0, "the grid must exercise a formerly missed forward cone"


def test_a3_rho_rest_and_b4_positions_require_composition() -> None:
    """Measure the overlapping positions instead of assuming disjoint repairs."""
    for name in ("spectra.photon.charged_rho", "spectra.photon.neutral_rho"):
        declaration = deltas.declared(name, "rest", "values")
        assert "A3" in deltas.repair_labels(declaration.repair)
    name = "mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum"
    case = cases.build_cases()[name]
    manifest = json.loads((Path(__file__).parent / "data/manifest.json").read_text())
    overlapping = 0
    with np.load(generate.DATA_DIR / manifest["cases"][name]["file"]) as data:
        for block, stored in zip(
            case.blocks, manifest["cases"][name]["blocks"], strict=True
        ):
            captured = oracle_reference.captured("A3")(case.resolve(), block)
            fsr = deltas.DELTA_MODELS["B4"].relation.term(case.resolve(), block)
            for suffix, reference in captured.items():
                original = data[stored["arrays"][suffix]["key"]]
                shared = (original != reference) & (fsr[suffix] != 0)
                if shared.any():
                    overlapping += np.count_nonzero(shared)
                    declaration = deltas.declared(name, block.label, suffix)
                    assert set(deltas.repair_labels(declaration.repair)) == {"A3", "B4"}
    assert overlapping > 0, "the A3/B4 overlap must actually be exercised"
