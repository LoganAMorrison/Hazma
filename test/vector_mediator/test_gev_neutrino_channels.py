"""Neutrino yield per annihilation of ``VectorMediatorGeV``'s lepton channels.

The model counts neutrinos only, not antineutrinos, in each flavor. A
``mu+ mu-`` pair yields one electron neutrino, from the ``mu+``, and one
muon neutrino, from the ``mu-``. A ``pi+ pi-`` pair yields one electron
neutrino and, through ``pi -> mu nu``, two muon neutrinos. Counting
antineutrinos too would double every number, so the pin catches that and
a miswired kernel alike.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import trapezoid

from hazma import spectra
from hazma.vector_mediator import VectorMediatorGeV

# Both channels are open well above threshold: 2 m_pi = 279 MeV.
E_CM = 500.0  # MeV

# PDG's `BR(pi -> mu nu)`, as `hazma._core` carries it in
# `rust/src/constants.rs`. The rest goes to `pi -> e nu`.
BR_PI_TO_MU_NUMU = 0.9998770

# Neutrinos per annihilation, by channel and flavor.
EXPECTED = {
    ("mu mu", "e"): 1.0,
    ("mu mu", "mu"): 1.0,
    ("mu mu", "tau"): 0.0,
    ("pi pi", "e"): 1.0,
    ("pi pi", "mu"): 2.0 * BR_PI_TO_MU_NUMU,
    ("pi pi", "tau"): 0.0,
}

# The muon channel integrates to within 2e-9 of its count on the grid
# below; the pion channel's muon-flavored count falls 8.3e-6 short, the
# residue of the boost integral its kernel runs on top of the muon's. The
# budget covers that and nothing near a factor of two.
RTOL = 3e-5


@pytest.mark.parametrize(("channel", "flavor"), list(EXPECTED))
def test_channel_yields_neutrinos_only(channel: str, flavor: str) -> None:
    """The channel spectrum integrates to the neutrino count of its pair.

    Each final-state particle carries ``E_CM / 2``, so the neutrino energy
    runs from zero up to at most ``E_CM / 2``, in MeV; the grid starts at
    1 eV, below which the spectrum carries no measurable number.
    """
    # The cross section vanishes at ``E_CM = 2 mx``, which would zero every
    # channel, so the dark matter sits below threshold.
    model = VectorMediatorGeV(
        mx=200.0,
        mv=1000.0,
        gvxx=1.0,
        gvuu=1.0,
        gvdd=-1.0,
        gvss=1.0,
        gvee=1.0,
        gvmumu=1.0,
        gvveve=1.0,
        gvvmvm=1.0,
        gvvtvt=1.0,
    )
    e_nus = np.geomspace(1e-6, E_CM / 2.0, 200_001)
    dnde = np.asarray(model.neutrino_spectrum_funcs(flavor)[channel](e_nus, E_CM))

    expected = EXPECTED[(channel, flavor)]
    assert trapezoid(dnde, e_nus) == pytest.approx(expected, rel=RTOL, abs=1e-300)


@pytest.mark.parametrize("flavor", ["e", "mu", "tau"])
def test_kaon_channel_matches_one_charged_kaon(flavor: str) -> None:
    """The ``k k`` channel yields the neutrinos of one charged kaon.

    By CP, the neutrinos of a given flavor from a ``K+ K-`` pair equal the
    neutrino and antineutrino count of a single charged kaon carrying
    ``E_CM / 2``, in MeV. The channel is compared with that kernel on a
    common grid, so a dropped or doubled factor shows as a ratio of 2 or
    1/2. The counts are about 1.11 (e), 1.49 (mu) and 0 (tau).
    """
    e_cm = 1500.0  # MeV; above 2 m_K = 987 MeV
    model = VectorMediatorGeV(
        mx=200.0,
        mv=1000.0,
        gvxx=1.0,
        gvuu=1.0,
        gvdd=-1.0,
        gvss=1.0,
        gvee=1.0,
        gvmumu=1.0,
        gvveve=1.0,
        gvvmvm=1.0,
        gvvtvt=1.0,
    )
    e_nus = np.geomspace(1e-6, e_cm / 2.0, 200_001)
    dnde = np.asarray(model.neutrino_spectrum_funcs(flavor)["k k"](e_nus, e_cm))
    kaon = spectra.dnde_neutrino_charged_kaon(e_nus, e_cm / 2.0, flavor=flavor)

    # Both sides evaluate the same kernel on the same grid, so they agree
    # to rounding; 1e-9 leaves room for that and nothing near a factor of two.
    assert trapezoid(dnde, e_nus) == pytest.approx(
        trapezoid(kaon, e_nus), rel=1e-9, abs=1e-300
    )
