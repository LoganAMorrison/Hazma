"""Compare maintained public outputs with the captured baseline grid.

Run from the worktree root with its .venv/bin/python. The snapshot was captured
before integrating public axial exports, in the same environment and worktree.
"""

import json
from pathlib import Path

import numpy as np

import hazma
from hazma import spectra
from hazma.form_factors.vector import VectorFormFactorPiPi

root = Path(__file__).resolve().parents[3]
assert Path(hazma.__file__).resolve().is_relative_to(root)
base = json.loads(Path(__file__).with_name("public-values-baseline.json").read_text())
e = np.array(base["energy_mev"])
new = {
    "photon_pi0_200": spectra.dnde_photon_neutral_pion(e, 200.0),
    "positron_muon_150": spectra.dnde_positron_muon(e, 150.0),
    "neutrino_muon_150": spectra.dnde_neutrino_muon(e, 150.0),
    "vector_widths_mev": VectorFormFactorPiPi().width(
        np.array(base["vector_width_q_mev"]), (1.0, -1.0, 0.0)
    ),
}
count = 0
max_difference = 0.0
for name, values in new.items():
    old = np.array(base[name])
    values = np.asarray(values)
    assert old.shape == values.shape
    assert np.all(np.isfinite(values))
    # Same captured environment: arithmetic did not change on these paths.
    assert np.array_equal(values, old), name
    count += values.size
    max_difference = max(max_difference, float(np.max(np.abs(values - old))))
print(
    json.dumps(
        {
            "imported_hazma": hazma.__file__,
            "existing_output_samples": count,
            "photon_positron_neutrino_grid_points": e.size,
            "energy_grid_mev": [float(e[0]), float(e[-1])],
            "vector_width_grid_mev": base["vector_width_q_mev"],
            "maximum_absolute_difference": max_difference,
            "status": "unchanged on the recorded grid",
        },
        indent=2,
    )
)
