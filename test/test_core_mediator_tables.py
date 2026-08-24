""":mod:`hazma._core.mediator_tables` against NumPy and the live Cython.

The four mediator-spectrum ``.pyx`` — ``scalar_mediator_decay_spectrum``,
``vector_mediator_decay_spectrum`` and the two ``*_positron_spec`` — each
rebuild a 500-point log-spaced rest-frame table, interpolate it, and
re-dispatch a mode string inside their integrand. cython-to-rust Task 6.1
factors those three things into ``rust/src/kernels/mediator_tables.rs``;
Tasks 6.2 and 6.3 build the entry points on it. This module is the gate
on the factored part, before any entry point moves.

Why the oracles live here rather than in ``cargo test``
-------------------------------------------------------
Two of Task 6.1's claims are about agreement with something outside the
crate, so ``cargo`` cannot state them:

* **the grid is ``numpy.logspace``.** ``cargo`` pins the algorithm — the
  unfused ``i * step + start``, the last point substituted from ``stop``
  — but agreement with NumPy rests on Rust's ``log10``/``powf`` and
  NumPy's ``power`` loop reaching the same libm. Hard-coding one
  platform's bits into a ``cargo`` test would turn a Linux CI job red for
  a libm difference rather than a defect, the failure mode Phase 04
  learnings §4 records twice. Here the comparison re-derives on whatever
  platform the suite runs, and if a platform ever *does* disagree this
  module is where it says so.
* **the tables hold the Phase 04 kernels themselves**, called natively
  rather than through Python. Asserting that against
  ``hazma._core.photon`` / ``hazma._core.positron`` — the same kernels
  through their public entry points — is a real check; a re-derivation
  inside the crate would only compare the kernel to itself.

The mode oracles are stronger still, because the Cython twins are all
four alive until Tasks 6.2-6.4 delete them: an unrecognised mode string
can be put through the shipped ``.pyx`` and the Rust parser side by side.

The tables are the *legacy* constants
-------------------------------------
All four ``.pyx`` ``include "../_utils/legacy_parameters.pxd"``, so the
positron grid starts at ``0.510998928`` MeV and not at
``hazma.parameters.electron_mass``. ``rules.md`` rule 4 keeps it that
way; :class:`TestPositronTables` asserts the difference rather than
letting a later cleanup pass silently unify them.
"""

import math
from typing import ClassVar

import numpy as np
import pytest

from hazma._core import mediator_tables as tables
from hazma._core import photon as core_photon
from hazma._core import positron as core_positron

#: The mediator masses ``test/parity/cases.py`` samples, in MeV.
CORPUS_MASSES = (250.0, 550.0, 900.0)

#: ``MASS_E`` in ``hazma/_utils/legacy_parameters.pxd``, MeV. The value the
#: positron grid starts at; deliberately not ``hazma.parameters``'s.
LEGACY_ELECTRON_MASS = 0.510998928

#: Points in every mediator-spectrum table (``n_interp_pts`` in all four
#: ``.pyx``).
N_INTERP_PTS = 500

#: The decay modules' lower grid endpoint, as the base-10 exponent both
#: files write (``np.logspace(-1.0, ...)``).
PHOTON_LOG10_START = -1.0

#: The energy that exponent names, MeV -- the tail threshold the decay
#: modules compare against (``if eng_gam < 10**-1``).
PHOTON_GRID_FIRST_ENERGY = 0.1

#: The mediator mass the ``lookup`` and last-point tests work at, MeV,
#: and the daughter energy it implies.
REFERENCE_MASS = 550.0
REFERENCE_DAUGHTER_ENERGY = REFERENCE_MASS / 2.0

#: Every bit set, i.e. the scalar decay module's default mode list.
ALL_SCALAR_PHOTON_BITS = 127


def bits(array: np.ndarray) -> np.ndarray:
    """Reinterpret a ``float64`` array as its bit patterns."""
    return np.asarray(array, dtype=np.float64).view(np.int64)


def assert_bit_equal(actual: np.ndarray, expected: np.ndarray, what: str) -> None:
    """Assert two ``float64`` arrays agree in every bit."""
    differing = np.flatnonzero(bits(actual) != bits(expected))
    assert differing.size == 0, (
        f"{what}: {differing.size} of {np.size(expected)} values differ; "
        f"first at index {differing[0] if differing.size else -1} "
        f"({actual[differing[0]]!r} != {expected[differing[0]]!r})"
    )


class TestLogspace:
    """``mediator_tables.logspace`` against ``numpy.logspace``."""

    @pytest.mark.parametrize("mass", CORPUS_MASSES)
    @pytest.mark.parametrize(
        "start", [PHOTON_LOG10_START, math.log10(LEGACY_ELECTRON_MASS)]
    )
    def test_matches_numpy_on_the_grids_the_pyx_build(
        self, mass: float, start: float
    ) -> None:
        stop = math.log10(mass / 2.0)
        assert_bit_equal(
            tables.logspace(start, stop, N_INTERP_PTS),
            np.logspace(start, stop, num=N_INTERP_PTS),
            f"logspace({start}, {stop}, {N_INTERP_PTS})",
        )

    def test_matches_numpy_across_a_wide_mass_sweep(self) -> None:
        # 4,001 masses rather than the corpus's three: the last-point
        # substitution NumPy performs is worth one ulp at about 9% of
        # masses and at none of the corpus's, so a three-mass sweep would
        # not reach it.
        for mass in np.linspace(1.0, 2000.0, 4001):
            for start in (PHOTON_LOG10_START, math.log10(LEGACY_ELECTRON_MASS)):
                stop = math.log10(mass / 2.0)
                assert_bit_equal(
                    tables.logspace(start, stop, N_INTERP_PTS),
                    np.logspace(start, stop, num=N_INTERP_PTS),
                    f"logspace at m = {mass}, start = {start}",
                )

    def test_the_last_point_is_ten_to_the_stop_exponent(self) -> None:
        # Not ``mass / 2``: ``10 ** log10(275)`` is 275.0000000000001, and
        # a table whose last abscissa were exactly 275.0 would be a
        # different table.
        stop = math.log10(REFERENCE_DAUGHTER_ENERGY)
        grid = tables.logspace(PHOTON_LOG10_START, stop, N_INTERP_PTS)
        assert grid[-1] == 10.0**stop
        assert grid[-1] != REFERENCE_DAUGHTER_ENERGY

    def test_rejects_a_degenerate_point_count(self) -> None:
        with pytest.raises(ValueError, match="num must be at least 2"):
            tables.logspace(-1.0, 1.0, 1)


class TestPhotonTables:
    """The decay modules' tables: grid, values, and their provenance."""

    @pytest.mark.parametrize("mass", CORPUS_MASSES)
    def test_grid_is_the_pyx_grid(self, mass: float) -> None:
        energies, _, _ = tables.photon_tables(mass)
        assert_bit_equal(
            energies,
            np.logspace(PHOTON_LOG10_START, math.log10(mass / 2.0), num=N_INTERP_PTS),
            f"photon grid at m = {mass}",
        )
        assert energies.size == N_INTERP_PTS

    @pytest.mark.parametrize("mass", CORPUS_MASSES)
    def test_values_are_the_phase_04_kernels(self, mass: float) -> None:
        # The exit criterion "built by calling the Phase 04 kernel fns
        # natively" made checkable: the tabulated columns must be exactly
        # what the public entry points return on the same abscissae.
        energies, charged_pion, muon = tables.photon_tables(mass)
        daughter = mass / 2.0
        assert_bit_equal(
            charged_pion,
            core_photon.dnde_photon_charged_pion(energies, daughter),
            f"charged-pion photon column at m = {mass}",
        )
        assert_bit_equal(
            muon,
            core_photon.dnde_photon_muon(energies, daughter),
            f"muon photon column at m = {mass}",
        )

    def test_is_memoized_and_re_keys_on_the_mass(self) -> None:
        first, _, _ = tables.photon_tables(550.0)
        again, _, _ = tables.photon_tables(550.0)
        assert_bit_equal(again, first, "photon grid on a cache hit")
        other, _, _ = tables.photon_tables(900.0)
        assert not np.array_equal(bits(other), bits(first))
        back, _, _ = tables.photon_tables(550.0)
        assert_bit_equal(back, first, "photon grid after a re-key")


class TestPositronTables:
    """The positron modules' tables."""

    @pytest.mark.parametrize("mass", CORPUS_MASSES)
    def test_grid_starts_at_the_legacy_electron_mass(self, mass: float) -> None:
        from hazma.parameters import electron_mass  # noqa: PLC0415

        energies, _, _ = tables.positron_tables(mass)
        assert_bit_equal(
            energies,
            np.logspace(
                math.log10(LEGACY_ELECTRON_MASS),
                math.log10(mass / 2.0),
                num=N_INTERP_PTS,
            ),
            f"positron grid at m = {mass}",
        )
        # rules.md rule 4: the mediator `.pyx` include the legacy header,
        # so this is *not* the PDG value hazma.parameters exposes.
        assert energies[0] == LEGACY_ELECTRON_MASS
        assert energies[0] != electron_mass

    @pytest.mark.parametrize("mass", CORPUS_MASSES)
    def test_values_are_the_phase_04_kernels(self, mass: float) -> None:
        energies, charged_pion, muon = tables.positron_tables(mass)
        daughter = mass / 2.0
        assert_bit_equal(
            charged_pion,
            core_positron.dnde_positron_charged_pion(energies, daughter),
            f"charged-pion positron column at m = {mass}",
        )
        assert_bit_equal(
            muon,
            core_positron.dnde_positron_muon(energies, daughter),
            f"muon positron column at m = {mass}",
        )


class TestLookup:
    """The two below-grid policies, against the ``.pyx`` they come from."""

    @staticmethod
    def _table() -> tuple[np.ndarray, np.ndarray]:
        energies, charged_pion, _ = tables.photon_tables(REFERENCE_MASS)
        return energies, charged_pion

    def test_clamped_lookup_is_numpy_interp(self) -> None:
        energies, values = self._table()
        # Interior points, both endpoints, and both outsides -- the whole
        # domain `numpy.interp` defines.
        probes = np.concatenate(
            [
                [1e-6, 1e-3, energies[0], energies[-1], 1e3, 1e6],
                np.geomspace(energies[0], energies[-1], 4001),
                energies,
            ]
        )
        assert_bit_equal(
            tables.lookup(probes, energies, values, False),
            np.interp(probes, energies, values),
            "clamped lookup",
        )

    def test_inverse_energy_tail_is_the_pyx_branch(self) -> None:
        # `scalar_mediator_decay_spectrum.pyx:55-56`: below 10**-1 the
        # Cython returns `spec_cp[0] * e_gams[0] / eng_gam` and otherwise
        # defers to np.interp.
        energies, values = self._table()
        probes = np.concatenate(
            [
                np.geomspace(1e-8, energies[0], 2001),
                np.geomspace(energies[0], energies[-1], 2001),
                [1e3, 1e6],
            ]
        )
        expected = np.where(
            probes < PHOTON_GRID_FIRST_ENERGY,
            values[0] * energies[0] / probes,
            np.interp(probes, energies, values),
        )
        assert_bit_equal(
            tables.lookup(probes, energies, values, True),
            expected,
            "1/E-tail lookup",
        )

    def test_the_tail_threshold_is_the_grids_own_first_point(self) -> None:
        # The Cython compares against the literal `10**-1` rather than
        # against `e_gams[0]`; the two are the same double, so the branch
        # opens exactly at the grid's lower endpoint and introduces no
        # jump there.
        energies, values = self._table()
        assert energies[0] == PHOTON_GRID_FIRST_ENERGY
        just_below = np.nextafter(energies[0], 0.0)
        assert tables.lookup(just_below, energies, values, True) == pytest.approx(
            values[0], rel=1e-15
        )
        assert tables.lookup(energies[0], energies, values, True) == values[0]

    def test_both_policies_clamp_above_the_grid(self) -> None:
        # Neither `.pyx` guards the upper side, so both inherit NumPy's.
        energies, values = self._table()
        for tail in (True, False):
            assert tables.lookup(1e12, energies, values, tail) == values[-1]

    def test_nan_propagates(self) -> None:
        energies, values = self._table()
        for tail in (True, False):
            assert math.isnan(tables.lookup(float("nan"), energies, values, tail))

    def test_rejects_mismatched_columns(self) -> None:
        energies, values = self._table()
        with pytest.raises(ValueError, match="not of the same length"):
            tables.lookup(1.0, energies, values[:-1], False)
        with pytest.raises(ValueError, match="empty"):
            tables.lookup(1.0, energies[:0], values[:0], False)


class TestPhotonMode:
    """The vector decay module's ``mode`` argument."""

    #: Every string `vector_mediator_decay_spectrum.pyx:166-178` compares
    #: against, with whether `:223` adds the pi0 line for it.
    ACCEPTED: ClassVar[dict[str, tuple[str, bool]]] = {
        "total": ("Total", True),
        "e e g": ("ElectronFsr", False),
        "pi pi g": ("ChargedPionFsr", False),
        "pi pi": ("ChargedPionDecay", False),
        "pi0 g": ("NeutralPionLine", True),
        "mu mu g": ("MuonFsr", False),
        "mu mu": ("MuonDecay", False),
    }

    @pytest.mark.parametrize("mode", sorted(ACCEPTED))
    def test_accepts_the_pyx_strings(self, mode: str) -> None:
        assert tables.photon_mode(mode) == self.ACCEPTED[mode]

    @pytest.mark.parametrize(
        "mode", ["", " ", "Total", "total ", "pi0g", "e e", "pi0 pi0", "g g"]
    )
    def test_rejects_everything_else(self, mode: str) -> None:
        assert tables.photon_mode(mode) is None

    def test_the_rejected_set_is_what_the_cython_answers_with_zero(self) -> None:
        # The strongest oracle available while the twin is alive: a mode
        # the parser rejects is one the shipped `.pyx` returns 0.0 for --
        # its integrand falls off the end of a `cdef double` -- so the
        # port must not tighten it into a raise.
        from hazma.vector_mediator.vector_mediator_decay_spectrum import (  # noqa: PLC0415
            dnde_decay_v_pt,
        )

        pws = np.array([0.25, 0.25, 0.25, 0.25])
        for mode in ["", "Total", "pi0g", "g g", "not a mode"]:
            assert tables.photon_mode(mode) is None
            assert dnde_decay_v_pt(30.0, 600.0, 550.0, pws, mode) == 0.0


class TestPositronMode:
    """The positron modules' ``fs`` argument."""

    ACCEPTED: ClassVar[dict[str, str]] = {
        "total": "Total",
        "e e": "ElectronLine",
        "mu mu": "MuonDecay",
        "pi pi": "ChargedPionDecay",
    }

    @pytest.mark.parametrize("fs", sorted(ACCEPTED))
    def test_accepts_the_pyx_strings(self, fs: str) -> None:
        assert tables.positron_mode(fs) == self.ACCEPTED[fs]

    @pytest.mark.parametrize("fs", ["", "e e g", "pi0 g", "ee", "TOTAL", "pi pi g"])
    def test_rejects_everything_else(self, fs: str) -> None:
        assert tables.positron_mode(fs) is None

    def test_the_rejected_set_is_what_the_cython_answers_with_zero(self) -> None:
        from hazma.scalar_mediator.scalar_mediator_positron_spec import (  # noqa: PLC0415
            dnde_decay_s_pt,
        )

        pws = np.array([0.4, 0.3, 0.3])
        for fs in ["", "e e g", "pi0 g", "not a mode"]:
            assert tables.positron_mode(fs) is None
            assert dnde_decay_s_pt(30.0, 600.0, 550.0, pws, fs) == 0.0


class TestScalarPhotonModes:
    """The scalar decay module's mode *list*, folded to a bitflag."""

    #: `scalar_mediator_decay_spectrum.pyx:16-22`.
    BITS: ClassVar[dict[str, int]] = {
        "pi pi": 1,
        "mu mu": 2,
        "pi0 pi0": 4,
        "g g": 8,
        "e e g": 16,
        "pi pi g": 32,
        "mu mu g": 64,
    }

    #: The list `_scalar_mediator_spectra.py` passes by default.
    DEFAULT: ClassVar[list[str]] = [
        "pi pi",
        "mu mu",
        "pi0 pi0",
        "g g",
        "e e g",
        "pi pi g",
        "mu mu g",
    ]

    @pytest.mark.parametrize("mode", sorted(BITS))
    def test_each_mode_sets_its_own_bit(self, mode: str) -> None:
        assert tables.scalar_photon_mode_bits([mode]) == self.BITS[mode]

    def test_the_default_list_sets_every_bit(self) -> None:
        assert tables.scalar_photon_mode_bits(self.DEFAULT) == ALL_SCALAR_PHOTON_BITS

    def test_unknown_and_repeated_names_are_ignored(self) -> None:
        # The `.pyx` writes `if "mu mu" in modes: bitflag += BITFLAG_MM`,
        # so `in` is tested once per mode name: a duplicate cannot double
        # a flag into its neighbour's bit, and an unknown name adds
        # nothing and raises nothing.
        muon_decay = self.BITS["mu mu"]
        assert (
            tables.scalar_photon_mode_bits(["mu mu", "mu mu", "junk", ""]) == muon_decay
        )
        assert tables.scalar_photon_mode_bits([]) == 0

    def test_a_repeated_mode_does_not_change_the_cython_result(self) -> None:
        # The same claim against the shipped entry point, which is what
        # actually consumes the bitflag.
        from hazma.scalar_mediator.scalar_mediator_decay_spectrum import (  # noqa: PLC0415
            scalar_mediator_decay_spectrum,
        )

        pws = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        once = scalar_mediator_decay_spectrum(30.0, 600.0, 550.0, pws, ["mu mu"])
        twice = scalar_mediator_decay_spectrum(
            30.0, 600.0, 550.0, pws, ["mu mu", "mu mu", "junk"]
        )
        assert once == twice
