"""CMB energy-deposition contract of :class:`hazma.theory.TheoryCMB`.

A stub model injects a spectrum whose deposition efficiency has a closed
form: a monoenergetic line folds to one table value, and a flat positron box
folds to one integral of the table. The references read the tabulated curves
of arXiv:1506.03811 through ``numpy.interp``, which interpolates linearly as
``hazma.cmb``'s ``interp1d`` does, rather than through the code under test.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hazma.cmb import f_eff_ep_data, f_eff_g_data, p_ann_planck_temp_pol, vx_cmb
from hazma.limits import CMBLimit
from hazma.parameters import electron_mass
from hazma.theory import TheoryCMB
from hazma.vector_mediator import KineticMixing

# The stub's lines sit at fixed energies and f_eff is normalized to 2 mx, so a
# line sum is exact up to rounding.
LINE_RTOL = 1e-12

# At mx = 1 MeV the table read at total energy is off by more than this.
TOTAL_ENERGY_OFFSET = 0.05

# The continuum is integrated by ``quad`` with ``epsrel=1e-3``.
CONTINUUM_RTOL = 1e-3


def _table(data: np.ndarray, energy_mev: float) -> float:
    """Tabulated efficiency at an energy in MeV (the tables are in eV)."""
    return float(np.interp(energy_mev * 1e6, data[0], data[1]))


def f_g(energy: float) -> float:
    return _table(f_eff_g_data, energy)


def f_ee(kinetic_energy: float) -> float:
    return _table(f_eff_ep_data, kinetic_energy)


class StubAnnihilation(TheoryCMB):
    """Dark matter injecting fixed lines and optional flat boxes.

    Lines map a final state to ``(energy, branching fraction)``. A box is
    ``(e_low, e_high)`` and carries one photon or one positron per
    annihilation.
    """

    def __init__(
        self,
        mx: float,
        gamma_lines: dict[str, tuple[float, float]] | None = None,
        positron_lines: dict[str, tuple[float, float]] | None = None,
        photon_box: tuple[float, float] | None = None,
        positron_box: tuple[float, float] | None = None,
    ) -> None:
        self.mx = mx
        self._gamma_lines = gamma_lines or {}
        self._positron_lines = positron_lines or {}
        self._photon_box = photon_box
        self._positron_box = positron_box

    @staticmethod
    def _lines(lines: dict[str, tuple[float, float]]) -> dict[str, dict]:
        return {fs: {"energy": e, "bf": bf} for fs, (e, bf) in lines.items()}

    def gamma_ray_lines(self, e_cm: float) -> dict[str, dict]:
        return self._lines(self._gamma_lines)

    def positron_lines(self, e_cm: float) -> dict[str, dict]:
        return self._lines(self._positron_lines)

    @staticmethod
    def _box(es: np.ndarray, box: tuple[float, float] | None) -> np.ndarray:
        if box is None:
            return np.zeros_like(es)
        lo, hi = box
        return np.where((lo <= es) & (es <= hi), 1.0 / (hi - lo), 0.0)

    def total_spectrum(self, es: np.ndarray, e_cm: float) -> np.ndarray:
        return self._box(es, self._photon_box)

    def total_positron_spectrum(self, es: np.ndarray, e_cm: float) -> np.ndarray:
        return self._box(es, self._positron_box)


class TestLines:
    def test_two_photon_line_folds_to_the_table_value(self) -> None:
        model = StubAnnihilation(100.0, gamma_lines={"g g": (100.0, 1.0)})
        assert_allclose(model.f_eff_g(), f_g(100.0), rtol=LINE_RTOL)
        assert model.f_eff_ep() == 0.0

    def test_a_line_naming_another_final_state_carries_one_photon(self) -> None:
        # pi0 g at mx = 100 MeV: one photon of 100 - m_pi0^2 / (4 mx) MeV.
        energy = 54.45
        model = StubAnnihilation(100.0, gamma_lines={"pi0 g": (energy, 0.25)})
        expected = 0.25 * energy * f_g(energy) / 200.0
        assert_allclose(model.f_eff_g(), expected, rtol=LINE_RTOL)

    def test_pair_line_reads_the_table_at_kinetic_energy(self) -> None:
        # At mx = 1 MeV the kinetic energy is half the total, so reading the
        # table at the total energy would be a distinct, wrong number.
        mx = 1.0
        model = StubAnnihilation(mx, positron_lines={"e e": (mx, 1.0)})
        expected = f_ee(mx - electron_mass)
        assert abs(f_ee(mx) / expected - 1.0) > TOTAL_ENERGY_OFFSET
        assert_allclose(model.f_eff_ep(), expected, rtol=LINE_RTOL)
        assert model.f_eff_g() == 0.0

    def test_pair_line_below_the_table_reads_its_first_point(self) -> None:
        # Kinetic energy 1 keV, below the table's first point at 5.02 keV,
        # where the table is flat.
        mx = electron_mass + 1e-3
        model = StubAnnihilation(mx, positron_lines={"e e": (mx, 1.0)})
        assert_allclose(model.f_eff_ep(), f_eff_ep_data[1][0], rtol=LINE_RTOL)

    def test_pair_line_at_rest_is_counted(self) -> None:
        # A pair at rest has zero kinetic energy, the bottom of the range the
        # table's first point covers.
        model = StubAnnihilation(
            electron_mass, positron_lines={"e e": (electron_mass, 1.0)}
        )
        assert_allclose(model.f_eff_ep(), f_eff_ep_data[1][0], rtol=LINE_RTOL)
        assert np.isfinite(model.cmb_limit())

    def test_photon_line_at_the_table_edge_is_counted(self) -> None:
        energy = f_eff_g_data[0][0] / 1e6
        model = StubAnnihilation(1.0, gamma_lines={"pi0 g": (energy, 1.0)})
        expected = energy * f_eff_g_data[1][0] / 2.0
        assert_allclose(model.f_eff_g(), expected, rtol=LINE_RTOL)

    def test_below_the_electron_mass_the_pair_channel_is_empty(self) -> None:
        # Photon-only annihilation: the pair continuum must not reach the
        # table at negative kinetic energy.
        model = StubAnnihilation(0.3, gamma_lines={"g g": (0.3, 1.0)})
        assert model.f_eff_ep() == 0.0
        assert_allclose(model.f_eff(), f_g(0.3), rtol=LINE_RTOL)


class TestBoxes:
    def test_photon_box_folds_the_photon_spectrum(self) -> None:
        mx, lo, hi = 100.0, 0.1, 90.0
        model = StubAnnihilation(mx, photon_box=(lo, hi))
        es = np.linspace(lo, hi, 400_001)
        table = np.interp(es * 1e6, *f_eff_g_data)
        expected = np.trapezoid(es / (hi - lo) * table, es) / (2.0 * mx)
        assert_allclose(model.f_eff_g(), expected, rtol=CONTINUUM_RTOL)

    @pytest.mark.parametrize(
        ("mx", "box"),
        [(50.0, (1.0, 40.0)), (5.0, (0.52, 4.5)), (1.0, (electron_mass, 0.6))],
        ids=str,
    )
    def test_box_folds_the_charge_summed_spectrum(
        self, mx: float, box: tuple[float, float]
    ) -> None:
        lo, hi = box
        model = StubAnnihilation(mx, positron_box=box)
        # Two particles (e+ and e-) per annihilation, flat in total energy,
        # each weighted by its total energy and the table at its kinetic
        # energy, which ``numpy.interp`` holds at the first table point below
        # it. The integrand is piecewise quadratic, so a fine trapezoid
        # converges far below CONTINUUM_RTOL.
        es = np.linspace(lo, hi, 400_001)
        table = np.interp((es - electron_mass) * 1e6, *f_eff_ep_data)
        expected = np.trapezoid(es * 2.0 / (hi - lo) * table, es) / (2.0 * mx)
        assert_allclose(model.f_eff_ep(), expected, rtol=CONTINUUM_RTOL)


class TestCMBLimit:
    def test_limit_carries_the_dirac_population_factor(self) -> None:
        model = StubAnnihilation(100.0, gamma_lines={"g g": (100.0, 1.0)})
        expected = 2.0 * 100.0 * p_ann_planck_temp_pol / f_g(100.0)
        assert_allclose(model.cmb_limit(), expected, rtol=LINE_RTOL)

    def test_limit_is_infinite_without_deposition(self) -> None:
        # Below the electron mass with no photons, nothing is deposited.
        assert StubAnnihilation(0.3).cmb_limit() == np.inf

    def test_limit_class_agrees_with_the_model(self) -> None:
        model = StubAnnihilation(100.0, gamma_lines={"g g": (100.0, 1.0)})
        limit = CMBLimit(x_kd=1e-3, p_ann=3.2e-31)
        assert_allclose(
            limit.constrain([model]),
            [model.cmb_limit(x_kd=1e-3, p_ann=3.2e-31)],
            rtol=0.0,
        )
        assert limit.f_eff(model) == model.f_eff(x_kd=1e-3)


def test_kinetic_mixing_positron_line_counts_both_charges() -> None:
    # Below the muon threshold the only positrons are the e+ e- line, so
    # f_eff_ep is that line's electron and positron folded once each.
    model = KineticMixing(mx=50.0, mv=200.0, gvxx=1.0, eps=1e-3)
    # The branching fraction is 0 / 0 at e_cm = 2 mx exactly, so evaluate it
    # at the recombination energy f_eff_ep uses.
    e_cm = 2.0 * model.mx * (1.0 + 0.5 * vx_cmb(model.mx, 1e-4) ** 2)
    line = model.positron_lines(e_cm)["e e"]
    expected = line["bf"] * f_ee(model.mx - electron_mass)
    # The model's line sits at e_cm / 2, above mx by v^2 / 2 ~ 4e-11.
    assert_allclose(model.f_eff_ep(), expected, rtol=1e-9)
