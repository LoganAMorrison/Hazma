""":mod:`hazma._core.mediator_tables` against NumPy and the mediator spectra.

The four mediator-spectrum ``.pyx`` — ``scalar_mediator_decay_spectrum``,
``vector_mediator_decay_spectrum`` and the two ``*_positron_spec`` — each
rebuilt a 500-point log-spaced rest-frame table, interpolated it, and
re-dispatched a mode string inside their integrand. cython-to-rust Task
6.1 factored those three things into
``rust/src/kernels/mediator_tables.rs``; Tasks 6.2 and 6.3 build the entry
points on it. This module is the gate on the factored part.

**All four are gone.** Task 6.2 deleted both decay modules and Task 6.3
both positron ones, so every mode oracle below that used to call a
shipped ``.pyx`` now calls the port that replaced it. What was measured
against the Cython while it was alive is recorded in
``projects/cython-to-rust/task-notes/phase-06/task-6.2-decay-spectra.md``
and ``task-6.3-positron-spectra.md``; what those tests still buy is that
the parser's verdict and the entry point's behaviour stay coupled, which
is the half a later edit could break.

Why the oracles live here rather than in ``cargo test``
-------------------------------------------------------
Two of Task 6.1's claims are about agreement with something outside the
crate, so ``cargo`` cannot state them:

* **the grid is ``numpy.logspace``.** ``cargo`` pins the algorithm — the
  unfused ``i * step + start``, the last point substituted from ``stop``
  — but agreement with NumPy rests on Rust's ``powf`` and NumPy's
  ``power`` loop, which are not the same code. Hard-coding one platform's
  bits into a ``cargo`` test would turn a Linux CI job red for a libm
  difference rather than a defect, the failure mode Phase 04 learnings §4
  records twice. Here the comparison re-derives wherever the suite runs.
* **the tables hold the Phase 04 kernels themselves**, called natively
  rather than through Python. Asserting that against
  ``hazma._core.photon`` / ``hazma._core.positron`` — the same kernels
  through their public entry points — is a real check; a re-derivation
  inside the crate would only compare the kernel to itself. That claim is
  Rust against Rust and holds bit-for-bit on every platform.

The mode oracles were stronger still while the Cython twins were alive:
an unrecognised mode string could be put through the shipped ``.pyx`` and
the Rust parser side by side. Task 6.2 spent that oracle for the two
decay modules and Task 6.3 spends the rest; each records its measurement
in its own task note before deleting the file.

Two comparisons are platform-scoped, and both were measured
------------------------------------------------------------
The first CI round on this module was green on macOS/arm64 and red on all
five Linux jobs, in exactly the fifteen tests that compare against NumPy
and in none of the fifty that do not (run 32681245809). The port is not
what differs — ``hazma._core`` reproduces the Phase 04 kernels on both —
so the mode is **declared from the platform**, the way
``test/test_core_interp.py`` declares it, rather than detected by a
probe that Phase 04 learnings §4 records as unsound twice over.

**The grid.** ``numpy.logspace`` evaluates ``10 ** y`` through NumPy's
own ``power`` loop, which is vectorised on x86-64; ``rust/src/kernels/
mediator_tables.rs`` calls ``f64::powf``, which goes to the platform
libm. On macOS/arm64 the two agree bit-for-bit at every abscissa. On
Linux/x86-64 about 5% of the 500 points disagree — 19 to 31 per grid —
and **every disagreement measured was exactly one ulp**, worst relative
2.16e-16 across the nine pairs the failing run reported. That is the most
two implementations of ``10 ** y`` can differ if both round correctly, so
:func:`assert_matches_numpy_grid` allows one ulp off-platform and nothing
on it.

**The interpolation.** ``TestLookup`` compares ``hazma._core``'s
``np.interp`` port against NumPy's on a grid both sides share, so the only
divergence is whether ``slope * (x - xp[j]) + fp[j]`` is contracted — the
same question ``test/test_core_interp.py`` answers at length. This module
reuses that module's derived figure rather than inventing one:
:data:`OFF_PLATFORM_BUDGET` is ``1e-12`` of the compared array's **peak**,
4.6e3x the worst peak-relative disagreement measured on linux/amd64.
Peak-scaled and not pointwise, because the pointwise relative reading
reaches 4e-2 at cancellation points; see that module's docstring for the
measurement table.

What neither budget may absorb is a *support* change — an abscissa moving
to a different cell, or a value that was exactly zero becoming nonzero.
The tail-branch test asserts the branch structure separately from the
values for that reason.

The tables are the *legacy* constants
-------------------------------------
All four ``.pyx`` ``include "../_utils/legacy_parameters.pxd"``, so the
positron grid starts at ``0.510998928`` MeV and not at
``hazma.parameters.electron_mass``. ``rules.md`` rule 4 keeps it that
way; :class:`TestPositronTables` asserts the difference rather than
letting a later cleanup pass silently unify them.
"""

import json
import math
import platform
import sys
from pathlib import Path
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
#: ``.pyx``, two of which Task 6.2 deleted).
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


#: The machine the parity corpus was captured on, read from its own
#: manifest so the two cannot drift apart -- the same source
#: ``test/test_core_interp.py`` reads.
CAPTURE_MACHINE = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "test"
        / "parity"
        / "data"
        / "manifest.json"
    ).read_text()
)["environment"]["machine"]

ON_THE_CAPTURING_PLATFORM = (
    sys.platform == "darwin" and platform.machine() == CAPTURE_MACHINE
)

#: What ``lookup`` may differ from ``numpy.interp`` by off the capturing
#: platform, as a fraction of the compared array's peak. Taken verbatim
#: from ``test/test_core_interp.py``, which derived it from a linux/amd64
#: build over 1.15 million abscissae; the mechanism here is identical (the
#: same port, the same question about a contracted multiply-add) and the
#: grid is shared by both sides, so a second derivation would measure the
#: same thing.
OFF_PLATFORM_BUDGET = 1e-12


def bits(array: np.ndarray) -> np.ndarray:
    """Reinterpret a ``float64`` array as its bit patterns."""
    return np.asarray(array, dtype=np.float64).view(np.int64)


def ulp_distance(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """How many representable doubles separate each pair.

    Valid for the same-sign, finite values these grids hold; a sign
    change or a NaN is not something a one-ulp budget should absorb, and
    the callers below never produce one.
    """
    return np.abs(bits(actual).astype(np.int64) - bits(expected).astype(np.int64))


def assert_bit_equal(actual: np.ndarray, expected: np.ndarray, what: str) -> None:
    """Assert two ``float64`` arrays agree in every bit, on any platform.

    For Rust-against-Rust claims, which carry no libm question.
    """
    differing = np.flatnonzero(bits(actual) != bits(expected))
    assert differing.size == 0, (
        f"{what}: {differing.size} of {np.size(expected)} values differ; "
        f"first at index {differing[0] if differing.size else -1} "
        f"({actual[differing[0]]!r} != {expected[differing[0]]!r})"
    )


def assert_within_one_ulp(actual: np.ndarray, expected: np.ndarray, what: str) -> None:
    """The off-platform half of :func:`assert_matches_numpy_grid`.

    A separate function so :class:`TestTheOffPlatformBudgets` can
    exercise it on every platform, including the one where the caller
    below never reaches it. A budget only the CI machines run is a budget
    nobody has checked.
    """
    distance = ulp_distance(actual, expected)
    worst = int(distance.max(initial=0))
    assert worst <= 1, (
        f"{what}: {int(np.count_nonzero(distance))} of {np.size(expected)} "
        f"values differ from numpy.logspace by up to {worst} ulp, off the "
        f"platform the corpus was captured on where one ulp is the most two "
        f"correctly-rounded implementations of 10**y can differ"
    )


def assert_within_peak_budget(
    actual: np.ndarray, expected: np.ndarray, what: str
) -> None:
    """The off-platform half of :func:`assert_matches_numpy_interp`.

    Split out for the same reason as :func:`assert_within_one_ulp`.
    """
    peak = float(np.max(np.abs(expected), initial=0.0))
    tolerance = OFF_PLATFORM_BUDGET * max(peak, 1.0)
    np.testing.assert_allclose(
        actual, expected, rtol=OFF_PLATFORM_BUDGET, atol=tolerance, err_msg=what
    )


def assert_matches_numpy_grid(
    actual: np.ndarray, expected: np.ndarray, what: str
) -> None:
    """Assert a grid matches ``numpy.logspace`` to within one ulp.

    One ulp everywhere, including where the corpus was captured. This
    comparison used to be bit-for-bit there, on the reasoning that the
    port was written against *this* build's arithmetic and reproduced it
    exactly. That was true of the build the reasoning was written
    against and not of the build it named: exactness here turns out to
    depend on the **cargo profile**, not only on the platform. Under
    `[profile.release]`'s `lto = true` and `codegen-units = 1` the
    generated grid moves by one ulp at 4 of 500 abscissae at m = 550 MeV
    and 1 of 500 at m = 900 MeV, and it did so before this comparison was
    written -- every published wheel has always been a release build. The
    bit-equal branch was green only because the editable install the dev
    loop and CI both use was, under setuptools-rust, a **debug** build;
    the maturin cutover made it release, which is what surfaced this.

    So one ulp is the honest budget, and it is the one this module
    already derived: off the capturing platform the comparison measures
    NumPy's ``power`` loop against the platform libm, and one ulp is the
    most two correctly-rounded implementations of ``10**y`` can differ.
    Nothing about the values users receive changed here -- what changed
    is that the developer's build now computes the same ones. The one-ulp
    grid difference does not reach a published number either: a
    16-function, 7206-value sweep of the public spectra and cross sections
    is bit-equal between the two profiles, so this module reading
    ``mediator_tables`` directly is the only place it is visible.

    :func:`assert_matches_numpy_interp` keeps its platform split: the
    interp comparison is a different mechanism and is unaffected by the
    profile (measured, same run).
    """
    assert_within_one_ulp(actual, expected, what)


def assert_matches_numpy_interp(
    actual: np.ndarray, expected: np.ndarray, what: str
) -> None:
    """Assert a lookup matches ``numpy.interp``.

    Bit-for-bit on the capturing platform; :data:`OFF_PLATFORM_BUDGET` of
    the compared array's peak elsewhere, scaled to the peak rather than
    applied pointwise for the reason ``test/test_core_interp.py``
    documents -- the pointwise relative reading reaches 4e-2 at
    cancellation points and cannot set an honest figure.
    """
    if ON_THE_CAPTURING_PLATFORM:
        assert_bit_equal(actual, expected, what)
    else:
        assert_within_peak_budget(actual, expected, what)


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
        assert_matches_numpy_grid(
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
                assert_matches_numpy_grid(
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


class TestTheOffPlatformBudgets:
    """The two off-platform comparators reject what they must reject.

    They are called **directly** rather than through
    :func:`assert_matches_numpy_grid` / :func:`assert_matches_numpy_interp`,
    so they run on every platform including the one where the scoped
    callers never reach them. A budget only the CI machines execute is a
    budget nobody has checked -- which is how
    ``test/test_core_interp.py``'s probe silently voided nine claims
    before its 2026-08-12 rewrite.
    """

    @staticmethod
    def _grid() -> np.ndarray:
        return np.logspace(PHOTON_LOG10_START, math.log10(275.0), num=N_INTERP_PTS)

    def test_the_grid_budget_accepts_one_ulp(self) -> None:
        grid = self._grid()
        nudged = np.nextafter(grid, np.inf)
        assert np.all(ulp_distance(nudged, grid) == 1)
        assert_within_one_ulp(nudged, grid, "one ulp")

    def test_the_grid_budget_rejects_two_ulp(self) -> None:
        grid = self._grid()
        nudged = np.nextafter(np.nextafter(grid, np.inf), np.inf)
        with pytest.raises(AssertionError, match="by up to 2 ulp"):
            assert_within_one_ulp(nudged, grid, "two ulp")

    def test_the_interp_budget_accepts_a_last_bit_difference(self) -> None:
        values = self._grid()
        assert_within_peak_budget(np.nextafter(values, np.inf), values, "one ulp")

    def test_the_interp_budget_rejects_a_visible_error(self) -> None:
        # 1e-8 of the peak: 1e4x the budget, and still far too small to
        # see in a plot -- the probe size test_core_interp.py uses.
        values = self._grid()
        perturbed = values + 1e-8 * float(np.max(values))
        with pytest.raises(AssertionError):
            assert_within_peak_budget(perturbed, values, "visible error")

    def test_the_grid_comparator_is_budgeted_on_every_platform(self) -> None:
        # One ulp is accepted and two are not, here and everywhere: the
        # grid comparison no longer has a platform-scoped exact branch.
        # Asserted rather than assumed, so a comparator that silently
        # tightened or loosened would say so.
        grid = self._grid()
        nudged = np.nextafter(grid, np.inf)
        assert_matches_numpy_grid(nudged, grid, "one ulp")
        with pytest.raises(AssertionError, match="up to 2 ulp"):
            assert_matches_numpy_grid(np.nextafter(nudged, np.inf), grid, "two ulp")

    def test_the_interp_comparator_is_exact_here_and_budgeted_elsewhere(self) -> None:
        # The interp half keeps the platform split the grid half gave up;
        # which branch this machine takes, asserted rather than assumed.
        values = self._grid()
        nudged = np.nextafter(values, np.inf)
        if ON_THE_CAPTURING_PLATFORM:
            with pytest.raises(AssertionError, match="values differ"):
                assert_matches_numpy_interp(nudged, values, "one ulp")
        else:
            assert_matches_numpy_interp(nudged, values, "one ulp")


class TestPhotonTables:
    """The decay modules' tables: grid, values, and their provenance."""

    @pytest.mark.parametrize("mass", CORPUS_MASSES)
    def test_grid_is_the_pyx_grid(self, mass: float) -> None:
        energies, _, _ = tables.photon_tables(mass)
        assert_matches_numpy_grid(
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
        assert_matches_numpy_grid(
            energies,
            np.logspace(
                math.log10(LEGACY_ELECTRON_MASS),
                math.log10(mass / 2.0),
                num=N_INTERP_PTS,
            ),
            f"positron grid at m = {mass}",
        )
        # rules.md rule 4: the mediator `.pyx` include the legacy header,
        # so this is *not* the PDG value hazma.parameters exposes. The
        # first abscissa is `10 ** log10(m_e)`, which round-trips exactly
        # on the capturing platform and is allowed one ulp elsewhere for
        # the same reason the grid is; the gap to the PDG mass is 6e-9
        # relative, seven decades outside that, so the second assertion
        # needs no scoping.
        assert_matches_numpy_grid(
            np.array([energies[0]]),
            np.array([LEGACY_ELECTRON_MASS]),
            "positron grid start",
        )
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
        assert_matches_numpy_interp(
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
        got = tables.lookup(probes, energies, values, True)
        # The branch structure first, and exactly, on every platform: a
        # tolerance may absorb a rounding difference but must never absorb
        # a probe taking the wrong branch.
        below = probes < PHOTON_GRID_FIRST_ENERGY
        assert_bit_equal(
            got[below],
            (values[0] * energies[0] / probes)[below],
            "1/E tail below the threshold",
        )
        assert_matches_numpy_interp(got[~below], expected[~below], "1/E-tail lookup")

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

    def test_the_rejected_set_is_what_the_entry_point_answers_with_zero(
        self,
    ) -> None:
        # A mode the parser rejects is one the entry point returns 0.0 for,
        # because the `.pyx`'s integrand fell off the end of a `cdef
        # double`. This was the shipped Cython until Task 6.2 deleted it
        # and is the port now, so it pins the *coupling* rather than the
        # behaviour -- the behaviour itself is pinned against the
        # pre-deletion measurement in
        # `test/test_core_mediator_decay_photon.py`.
        from hazma._core.vector_mediator import dnde_decay_v_pt  # noqa: PLC0415

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

    def test_the_rejected_set_reaches_the_entry_point_as_zero(self) -> None:
        """A rejected string is `0.0` at the entry point, not an error.

        Until Task 6.3 the call below went to
        ``scalar_mediator_positron_spec.pyx``, so this was an independent
        oracle: every ``cdef double`` integrand there ends in an
        ``if``-chain with no ``else``, and a C function that falls off
        its end returns zero. That ``.pyx`` is gone and the call now goes
        to the port, so what this pins is the port's promise to keep
        answering the way the Cython did — transcription with its
        provenance recorded, the same standing
        ``cython_dispatch_messages()`` has had since Task 6.2.

        The measurement behind the transcription is in
        ``docs/followups/todo/mediator-spectra-accept-unknown-mode-strings.md``,
        which is also where the case for *changing* it lives.
        """
        from hazma.scalar_mediator._scalar_mediator_positron_spectra import (  # noqa: PLC0415
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

    def test_a_repeated_mode_does_not_change_the_entry_point_result(self) -> None:
        # The same claim against the entry point, which is what actually
        # consumes the bitflag. The shipped `.pyx` until Task 6.2 deleted
        # it; the port since.
        from hazma._core.scalar_mediator import (  # noqa: PLC0415
            scalar_mediator_decay_spectrum,
        )

        pws = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        once = scalar_mediator_decay_spectrum(30.0, 600.0, 550.0, pws, ["mu mu"])
        twice = scalar_mediator_decay_spectrum(
            30.0, 600.0, 550.0, pws, ["mu mu", "mu mu", "junk"]
        )
        assert once == twice
