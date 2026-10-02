"""Pin the table-backed decay spectra to zero past their endpoint at rest.

A parent at rest takes the rest-frame branch, which evaluates its table
pointwise rather than integrating it over a Lorentz window. Each table ends at
the decay's endpoint, below half the parent mass, so an energy of 0.51 or 1.0
times the parent mass lies past it and must give exactly zero for scalar and
array input alike.
"""

from collections.abc import Callable

import numpy as np
import pytest

from hazma import spectra
from hazma.parameters import standard_model_masses as sm_masses

_PARENTS = {
    "charged_kaon": "k",
    "long_kaon": "kl",
    "short_kaon": "ks",
    "eta": "eta",
    "eta_prime": "etap",
    "omega": "omega",
    "phi": "phi",
    "charged_rho": "rho",
    "neutral_rho": "rho0",
}

_CASES = [
    pytest.param(getattr(spectra, f"dnde_{kind}_{name}"), key, id=f"{kind}-{name}")
    for name, key in _PARENTS.items()
    for kind in ("positron", "neutrino")
]


@pytest.mark.parametrize(("dnde", "key"), _CASES)
def test_rest_frame_spectrum_vanishes_past_the_endpoint(
    dnde: Callable, key: str
) -> None:
    mass = sm_masses[key]
    energies = np.array([0.51, 1.0]) * mass

    np.testing.assert_array_equal(dnde(energies, mass), 0.0)
    for energy in energies:
        np.testing.assert_array_equal(dnde(float(energy), mass), 0.0)
