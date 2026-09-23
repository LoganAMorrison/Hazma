"""Positron yield per annihilation of the legacy mediator models' channels.

``ScalarMediator`` and the ``VectorMediator`` family count positrons only:
a ``mu+ mu-`` or ``pi+ pi-`` pair carries one positive particle, whose decay
chain ends in exactly one positron. So each of those channel spectra must
integrate to one positron per annihilation. The pin catches a channel wired
to the wrong kernel — a photon spectrum integrates to under 0.1 over the
same range — and a stray factor of two alike.

The GeV kinetic-mixing model in ``hazma.vector_mediator._gev`` counts
positrons and electrons together, and so carries a factor of two these
channels do not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.integrate import trapezoid

from hazma.parameters import electron_mass, vh
from hazma.scalar_mediator import ScalarMediator
from hazma.vector_mediator import VectorMediator

if TYPE_CHECKING:
    from hazma.theory import TheoryAnn

# Both channels are open well above threshold: 2 m_pi = 279 MeV.
E_CM = 500.0  # MeV

# The muon channel integrates to within 3e-10 of one positron on the grid
# below; the pion channel falls 2.7e-6 short, the residue of the boost
# integral its kernel runs on top of the muon's. The budget covers that
# and nothing near a wrong kernel or a factor of two, both of which are
# orders of magnitude away.
RTOL = 1e-5

MODELS = [
    pytest.param(
        ScalarMediator(
            mx=250.0,
            ms=1000.0,
            gsxx=1.0,
            gsff=1e-3,
            gsGG=3e-3,
            gsFF=-5.0 / 6.0 * 1e-3,
            lam=vh,
        ),
        id="scalar-mediator",
    ),
    pytest.param(
        VectorMediator(
            mx=250.0,
            mv=1000.0,
            gvxx=1.0,
            gvuu=1.0,
            gvdd=-1.0,
            gvss=1.0,
            gvee=1.0,
            gvmumu=1.0,
        ),
        id="vector-mediator",
    ),
]


@pytest.mark.parametrize("channel", ["mu mu", "pi pi"])
@pytest.mark.parametrize("model", MODELS)
def test_channel_yields_one_positron_per_annihilation(
    model: TheoryAnn, channel: str
) -> None:
    """The channel spectrum integrates to one positron over its support.

    Each final-state particle carries ``E_CM / 2``, so the positron energy
    runs from the electron mass up to at most ``E_CM / 2``, in MeV.
    """
    e_ps = np.geomspace(electron_mass, E_CM / 2.0, 200_001)
    dnde = np.asarray(model.positron_spectrum_funcs()[channel](e_ps, E_CM))

    assert trapezoid(dnde, e_ps) == pytest.approx(1.0, rel=RTOL)
