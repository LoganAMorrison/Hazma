"""``hazma._core.photon`` — the seven tabulated photon spectra.

cython-to-rust Phase 04 Task 4.2. Shaped like
``test/test_core_positron_muon.py`` (the per-kernel template) with one
deliberate departure, explained below.

The three parts
---------------
1. :class:`TestDispatchWiring` — one assertion per contract branch, for each
   of the seven entry points, enough to prove they go through ``map_unary``
   with the wording their Cython twins used. Branch-by-branch reasoning about
   the helper itself stays in ``test/test_core_dispatch.py``.
2. :class:`TestAgainstAnIndependentReference` — a Python reference built from
   the shipped CSVs with NumPy and from the foundation probes
   (``hazma._core.boost``, ``hazma._core.interp``), compared **bit-for-bit**.
3. :class:`TestPhysics` — statements about the spectra that owe nothing to the
   implementation being replaced: thresholds, support, the tails, and the
   photon count each monochromatic line carries. Two of those statements pin
   defects rather than correctness; see "Two reproduced defects".

Why there is no Cython oracle here, and no two-mode comparison
--------------------------------------------------------------
Task 4.1's twin survives as a capi provider, so its module could call the
live ``cdef`` and compare bit-for-bit. These five ``.pyx`` are deleted
outright in the same PR as this swap (rules.md rule 1 — nothing cimports
them), so there is no twin left to call. The against-the-Cython evidence for
this family is therefore the **parity corpus**, which holds all seven entry
points to their 179,695 pre-port pinned values and is what gates the swap.
Before the twins were removed, the port was additionally compared against
them directly over 336,000 points — seven entry points x six parent energies
x 8,000 energies spanning five decades below the parent mass to a hundred
times its energy — with **zero mismatches**, bit-for-bit
(``projects/cython-to-rust/task-notes/phase-04/task-4.2-photon-table-family.md``).

The reference in part 2 is a different kind of oracle and needs no platform
scoping, which is why this module has one comparison mode where Task 4.1 has
two. It shares the foundation — ``boost_integrate_linear_interp``,
``boost_delta_function`` and ``interp`` are the same Rust either way — so it
does not re-test those (``test/test_core_boost.py`` and
``test/test_core_interp.py`` do, against the Cython twin and NumPy). What it
*does* test is everything this task actually wrote: that each entry point
reaches for the right table, threshold, mass and line terms, that the
embedded CSVs parse to the same doubles NumPy reads, and that the per-row
column sum reproduces ``numpy.sum(axis=0)``. All of that is exact arithmetic
on both sides — IEEE division and square root are correctly rounded, and the
one fused multiply-add is reproduced here with a ``Fraction``-based ``fma``
rather than approximated — so the comparison is bit-equality on every
platform rather than on one.

Two reproduced defects
----------------------
:class:`TestPhysics` asserts two things that are *wrong physics*, because the
shipped Cython does them and ``projects/cython-to-rust/rules.md`` rule 1 says
a port reproduces rather than repairs:

* the η' two-photon line carries ``BR(η' -> a a)`` where its four siblings
  carry ``2·BR`` — 0.02307 photons per decay instead of 0.04614
  (``docs/followups/todo/eta-prime-two-photon-line-missing-factor-two.md``);
* the φ's two lines are placed at the *daughter meson's* energy rather than
  the photon's — 656.94 MeV instead of 362.52 for ``φ -> η a``, and 959.65
  instead of 59.82 for ``φ -> η' a``
  (``docs/followups/todo/phi-photon-lines-use-the-daughter-meson-energy.md``).

Asserting the correct physics here would contradict the parity corpus, which
pins the shipped values. When those follow-ups land, these two tests are the
ones that change.

A note on notation: final states are written the way hazma's own data files
write them, with ``a`` for a photon (``a_a``, ``pi0_a``, ``eta_a`` are
columns in ``hazma/spectra/_photon/data/*.csv``). That also keeps ruff's
RUF002 ambiguous-unicode check quiet, which the Greek letter for a photon
does not — it reads it as a Latin ``y``. The Rust side has no such lint and
spells the same modes with the letter.
"""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from hazma import spectra
from hazma._core import boost as core_boost
from hazma._core import interp as core_interp
from hazma._core import photon as core_photon

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "hazma" / "spectra" / "_photon" / "data"

QUANTITY = "Photon energies"
DIMENSION_ERROR = f"{QUANTITY} must be 0 or 1-dimensional."
TYPE_ERROR = f"{QUANTITY} must be a float or a NumPy array."

#: `hazma/_utils/constants.pxd`, which is the table these kernels' `.pyx`
#: files `include`d. Spelled out rather than imported from
#: `hazma.parameters` so a future consolidation of the two constant tables
#: cannot silently move the tests with the code
#: (`projects/cython-to-rust/rules.md` rule 4).
MASS_PI0 = 134.9768
MASS_ETA = 547.862
MASS_ETAP = 957.78
MASS_K = 493.677
MASS_K0 = 497.611
MASS_OMEGA = 782.66
MASS_PHI = 1019.461

BR_ETA_TO_A_A = 39.41e-2
BR_ETAP_TO_A_A = 2.307e-2
BR_KL_TO_A_A = 5.47e-4
BR_KS_TO_A_A = 2.63e-6
BR_OMEGA_TO_PI0_A = 8.34e-2
BR_OMEGA_TO_ETA_A = 4.5e-4
BR_PHI_TO_ETA_A = 1.303e-2
BR_PHI_TO_ETAP_A = 6.22e-5

#: ``name -> (entry point, CSV, parent mass, [(line energy, line weight)])``.
#: The line expressions are the ``.pyx`` ones character for character,
#: including the two the port reproduces rather than repairs.
SPECTRA: dict[str, tuple[object, str, float, list[tuple[float, float]]]] = {
    "charged_kaon": (
        core_photon.dnde_photon_charged_kaon,
        "charged_kaon_photon.csv",
        MASS_K,
        [],
    ),
    "long_kaon": (
        core_photon.dnde_photon_long_kaon,
        "long_kaon_photon.csv",
        MASS_K0,
        [(MASS_K0 / 2.0, 2 * BR_KL_TO_A_A)],
    ),
    "short_kaon": (
        core_photon.dnde_photon_short_kaon,
        "short_kaon_photon.csv",
        MASS_K0,
        [(MASS_K0 / 2.0, 2 * BR_KS_TO_A_A)],
    ),
    "eta": (
        core_photon.dnde_photon_eta,
        "eta_photon.csv",
        MASS_ETA,
        [(MASS_ETA / 2.0, 2.0 * BR_ETA_TO_A_A)],
    ),
    "eta_prime": (
        core_photon.dnde_photon_eta_prime,
        "eta_prime_photon.csv",
        MASS_ETAP,
        [(MASS_ETAP / 2.0, BR_ETAP_TO_A_A)],
    ),
    "omega": (
        core_photon.dnde_photon_omega,
        "omega_photon.csv",
        MASS_OMEGA,
        [
            ((MASS_OMEGA**2 - MASS_PI0**2) / (2 * MASS_OMEGA), BR_OMEGA_TO_PI0_A),
            ((MASS_OMEGA**2 - MASS_ETA**2) / (2 * MASS_OMEGA), BR_OMEGA_TO_ETA_A),
        ],
    ),
    "phi": (
        core_photon.dnde_photon_phi,
        "phi_photon.csv",
        MASS_PHI,
        [
            ((MASS_PHI**2 + MASS_ETA**2) / (2 * MASS_PHI), BR_PHI_TO_ETA_A),
            ((MASS_PHI**2 + MASS_ETAP**2) / (2 * MASS_PHI), BR_PHI_TO_ETAP_A),
        ],
    ),
}

NAMES = tuple(SPECTRA)

#: The wrapper functions the public API exposes, in the same order.
WRAPPERS = {
    "charged_kaon": spectra.dnde_photon_charged_kaon,
    "long_kaon": spectra.dnde_photon_long_kaon,
    "short_kaon": spectra.dnde_photon_short_kaon,
    "eta": spectra.dnde_photon_eta,
    "eta_prime": spectra.dnde_photon_eta_prime,
    "omega": spectra.dnde_photon_omega,
    "phi": spectra.dnde_photon_phi,
}


def fma(a: float, b: float, c: float) -> float:
    """Correctly-rounded ``a * b + c``, as one rounding rather than two.

    ``math.fma`` would do, and needs Python 3.13; the suite supports 3.10.
    ``Fraction`` arithmetic is exact and ``float()`` rounds it half-to-even,
    which is the same value hardware FMA produces. The non-finite cases fall
    back to the unfused form, where the two agree anyway.
    """
    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
        return a * b + c
    return float(Fraction(a) * Fraction(b) + Fraction(c))


def load_table(csv: str) -> tuple[np.ndarray, np.ndarray]:
    """The rest-frame table exactly as the deleted Cython built it.

    ``np.loadtxt(...).T`` then ``np.sum(rows[1:], axis=0)`` — the two lines
    every one of the five ``.pyx`` files opened with. Reproducing them here
    rather than reading a stored copy is what makes this a check on the
    Rust's embedded parse.
    """
    data = np.loadtxt(DATA_DIR / csv, delimiter=",").T
    return data[0], np.sum(data[1:], axis=0)


TABLES = {name: load_table(spec[1]) for name, spec in SPECTRA.items()}


def reference(
    name: str, photon_energies: np.ndarray, parent_energy: float
) -> np.ndarray:
    """The spectrum, rebuilt from the CSV and the foundation probes.

    Mirrors the structure of the deleted ``dnde_photon_*_point``: below
    threshold zero, within one ``DBL_EPSILON`` MeV of rest the table itself,
    otherwise the boosted continuum plus one boosted line per decay mode.
    """
    _, _, mass, lines = SPECTRA[name]
    x, y = TABLES[name]
    energies = np.asarray(photon_energies, dtype=np.float64)

    if parent_energy < mass:
        return np.zeros_like(energies)

    if parent_energy - mass < sys.float_info.epsilon:
        emin, emax = x[0], x[-1]
        out = np.empty_like(energies)
        above = energies > emax
        below = energies < emin
        inside = ~(above | below)
        out[above] = 0.0
        out[below] = y[0] * emin / energies[below]
        out[inside] = np.asarray(core_interp.interp(energies[inside], x, y))
        # `np.interp` propagates NaN, and so does the probe; neither mask
        # above catches a NaN, so it lands in `inside` and comes back NaN.
        return out

    beta = float(core_boost.boost_beta(parent_energy, mass))
    out = np.asarray(core_boost.boost_integrate_linear_interp(energies, beta, x, y))
    for line_energy, weight in lines:
        delta = np.asarray(
            core_boost.boost_delta_function(line_energy, energies, 0.0, beta)
        )
        out = np.array(
            [fma(float(d), weight, float(r)) for d, r in zip(delta, out, strict=True)]
        )
    return out


def probe_grid(mass: float, parent_energy: float, npoints: int = 601) -> np.ndarray:
    """A log grid running well outside the physical support on both sides.

    Same design as the parity corpus's: five decades below the parent mass to
    a hundred times its energy, so the below-table tail, the interpolated
    interior, the boosted window and the hard zero above it are all sampled.
    """
    return np.geomspace(1e-5 * mass, 100.0 * parent_energy, npoints)


#: Parent-energy multipliers: exactly at rest, one step past the
#: ``E - m < DBL_EPSILON`` short circuit, and three boosts.
PARENT_MULTIPLIERS = (1.0, 1.0 + 1e-12, 1.05, 2.0, 10.0)

#: The ``1/E`` tail's defining property: halving the energy doubles the
#: spectrum.
TAIL_RATIO_ON_HALVING = 2.0

#: The loosest the derived per-line tolerance may get before it would
#: stop separating a correct weight from a halved one.
MAX_LINE_TOLERANCE = 1e-3

#: How far above its own rest-frame value the boost integral's
#: over-count pushes a barely-moving parent -- a lower bound, since the
#: measured factor is 6,500x to 33,000x.
MIN_THRESHOLD_DIVERGENCE = 1e3


class TestDispatchWiring:
    """Each entry point goes through ``map_unary`` with its own wording."""

    @pytest.mark.parametrize("name", NAMES)
    def test_a_scalar_returns_a_python_float(self, name: str) -> None:
        dnde, _, mass, _ = SPECTRA[name]
        value = dnde(mass / 4.0, 2.0 * mass)
        assert type(value) is float
        assert value > 0.0

    @pytest.mark.parametrize("name", NAMES)
    def test_a_numpy_scalar_and_a_zero_dim_array_take_the_scalar_path(
        self, name: str
    ) -> None:
        dnde, _, mass, _ = SPECTRA[name]
        energy, parent = mass / 4.0, 2.0 * mass
        expected = dnde(energy, parent)
        assert dnde(np.float64(energy), parent) == expected
        assert dnde(np.array(energy), parent) == expected
        assert type(dnde(np.array(energy), parent)) is float

    @pytest.mark.parametrize("name", NAMES)
    def test_an_array_returns_a_fresh_float64_array_of_the_same_length(
        self, name: str
    ) -> None:
        dnde, _, mass, _ = SPECTRA[name]
        energies = np.geomspace(mass / 100.0, mass, 64)
        values = dnde(energies, 2.0 * mass)
        assert values.dtype == np.float64
        assert values.shape == energies.shape
        assert not np.shares_memory(values, energies)

    @pytest.mark.parametrize("name", NAMES)
    def test_the_array_path_agrees_with_the_scalar_path_bit_for_bit(
        self, name: str
    ) -> None:
        # `map_unary` calls the same kernel either way, so a broadcasting
        # bug shows up here and nowhere the corpus looks.
        dnde, _, mass, _ = SPECTRA[name]
        energies = probe_grid(mass, 2.0 * mass, 257)
        batched = dnde(energies, 2.0 * mass)
        one_at_a_time = np.array([dnde(float(e), 2.0 * mass) for e in energies])
        assert batched.tobytes() == one_at_a_time.tobytes()

    @pytest.mark.parametrize("name", NAMES)
    def test_a_sequence_is_accepted(self, name: str) -> None:
        dnde, _, mass, _ = SPECTRA[name]
        parent = 2.0 * mass
        assert dnde([mass / 4.0, mass / 2.0], parent).tolist() == [
            dnde(mass / 4.0, parent),
            dnde(mass / 2.0, parent),
        ]

    @pytest.mark.parametrize("name", NAMES)
    def test_an_empty_grid_returns_an_empty_grid(self, name: str) -> None:
        dnde, _, mass, _ = SPECTRA[name]
        assert dnde(np.array([], dtype=np.float64), 2.0 * mass).shape == (0,)

    @pytest.mark.parametrize("name", NAMES)
    def test_a_higher_rank_array_names_this_kernel_s_quantity(self, name: str) -> None:
        dnde, _, mass, _ = SPECTRA[name]
        with pytest.raises(ValueError, match=r"^Photon energies must be 0 or 1-"):
            dnde(np.ones((2, 2)), 2.0 * mass)

    @pytest.mark.parametrize("name", NAMES)
    def test_the_rank_message_is_the_deleted_assert_verbatim(self, name: str) -> None:
        # All five `.pyx` files wrote `assert len(energies.shape) == 1,
        # "Photon energies must be 0 or 1-dimensional."` -- reproduced byte
        # for byte, with only the exception type changed (rules.md rule 9).
        dnde, _, mass, _ = SPECTRA[name]
        with pytest.raises(ValueError) as info:
            dnde(np.ones((2, 2)), 2.0 * mass)
        assert str(info.value) == DIMENSION_ERROR

    @pytest.mark.parametrize("name", NAMES)
    def test_a_non_float64_array_is_rejected(self, name: str) -> None:
        dnde, _, mass, _ = SPECTRA[name]
        with pytest.raises(ValueError, match="must be a float64 array"):
            dnde(np.array([1, 2], dtype=np.int32), 2.0 * mass)

    @pytest.mark.parametrize("name", NAMES)
    def test_a_non_numeric_argument_is_a_type_error(self, name: str) -> None:
        dnde, _, mass, _ = SPECTRA[name]
        with pytest.raises(TypeError) as info:
            dnde(None, 2.0 * mass)
        assert str(info.value) == TYPE_ERROR

    @pytest.mark.parametrize("name", NAMES)
    def test_arguments_are_accepted_by_keyword(self, name: str) -> None:
        # The `text_signature` on these wrappers is not positional-only,
        # because the Cython entry points were `def`s that took keywords.
        # Task 2.3 found `text_signature` is a claim PyO3 does not enforce,
        # so the claim is checked here rather than trusted.
        dnde, _, mass, _ = SPECTRA[name]
        assert dnde(
            photon_energies=mass / 4.0, **{_PARENT_KWARG[name]: 2.0 * mass}
        ) == (dnde(mass / 4.0, 2.0 * mass))

    @pytest.mark.parametrize("name", NAMES)
    def test_the_public_wrapper_delegates_to_this_kernel(self, name: str) -> None:
        # The swap is only real if `hazma.spectra` reaches the extension:
        # a wrapper still pointing at a `.pyx` would leave the corpus
        # measuring the implementation this task replaced.
        dnde, _, mass, _ = SPECTRA[name]
        energies = probe_grid(mass, 2.0 * mass, 129)
        assert (
            np.asarray(WRAPPERS[name](energies, 2.0 * mass)).tobytes()
            == np.asarray(dnde(energies, 2.0 * mass)).tobytes()
        )


#: The keyword each entry point names its parent-energy argument.
_PARENT_KWARG = {
    "charged_kaon": "kaon_energy",
    "long_kaon": "kaon_energy",
    "short_kaon": "kaon_energy",
    "eta": "eta_energy",
    "eta_prime": "eta_prime_energy",
    "omega": "omega_energy",
    "phi": "phi_energy",
}


class TestAgainstAnIndependentReference:
    """Bit-for-bit against the CSV-plus-foundation reference.

    See the module docstring for what this does and does not cover, and for
    why one comparison mode suffices where Task 4.1 needed two.
    """

    @pytest.mark.parametrize("name", NAMES)
    @pytest.mark.parametrize("multiplier", PARENT_MULTIPLIERS)
    def test_the_kernel_reproduces_the_reference(
        self, name: str, multiplier: float
    ) -> None:
        dnde, _, mass, _ = SPECTRA[name]
        parent = mass * multiplier
        energies = probe_grid(mass, parent)
        got = np.asarray(dnde(energies, parent))
        want = reference(name, energies, parent)
        assert got.tobytes() == want.tobytes(), (
            f"{name} at E = {parent}: the kernel left the reference built "
            f"from {SPECTRA[name][1]} and the foundation probes"
        )

    @pytest.mark.parametrize("name", NAMES)
    def test_below_threshold_the_reference_and_the_kernel_both_vanish(
        self, name: str
    ) -> None:
        dnde, _, mass, _ = SPECTRA[name]
        energies = probe_grid(mass, mass)
        got = np.asarray(dnde(energies, mass * 0.5))
        assert np.array_equal(got, np.zeros_like(got))
        assert got.tobytes() == reference(name, energies, mass * 0.5).tobytes()

    @pytest.mark.parametrize("name", NAMES)
    def test_the_embedded_table_is_the_csv_numpy_reads(self, name: str) -> None:
        """The Rust parse, checked at every one of the table's own nodes.

        At rest the kernel *is* the table — ``interp`` returns a node's own
        value at a node rather than interpolating — so evaluating at every
        tabulated energy reads the embedded table back out one value at a
        time. Any single-bit disagreement between ``f64::from_str`` and
        NumPy's CSV reader, or between the crate's pairwise column sum and
        ``numpy.sum(axis=0)``, shows up here as a differing byte.
        """
        dnde, _, mass, _ = SPECTRA[name]
        x, y = TABLES[name]
        got = np.asarray(dnde(x, mass))
        assert got.tobytes() == y.tobytes()


class TestPhysics:
    """Statements about the spectra that outlive the deleted Cython."""

    @pytest.mark.parametrize("name", NAMES)
    def test_the_spectrum_is_zero_below_the_parent_threshold(self, name: str) -> None:
        dnde, _, mass, _ = SPECTRA[name]
        for parent in (0.0, mass * 0.5, math.nextafter(mass, 0.0)):
            values = np.asarray(dnde(probe_grid(mass, mass), parent))
            assert np.array_equal(values, np.zeros_like(values)), f"{name} at {parent}"

    @pytest.mark.parametrize("name", NAMES)
    def test_the_spectrum_vanishes_above_the_boosted_endpoint(self, name: str) -> None:
        # The boosted continuum stops at `gamma (1 + beta) * emax` and the
        # lines stop below that, so nothing survives an order of magnitude
        # past it.
        dnde, _, mass, _ = SPECTRA[name]
        parent = 2.0 * mass
        gamma = parent / mass
        beta = math.sqrt(1.0 - 1.0 / gamma**2)
        endpoint = gamma * (1.0 + beta) * TABLES[name][0][-1]
        assert dnde(endpoint * 10.0, parent) == 0.0

    @pytest.mark.parametrize("name", NAMES)
    def test_the_rest_frame_tail_below_the_table_is_one_over_e(self, name: str) -> None:
        # The Cython's extrapolation: `y0 * emin / E`, so halving the energy
        # doubles the spectrum, exactly.
        dnde, _, mass, _ = SPECTRA[name]
        x, y = TABLES[name]
        low = x[0] * 0.1
        assert dnde(low, mass) == y[0] * x[0] / low
        assert dnde(low * 0.5, mass) / dnde(low, mass) == TAIL_RATIO_ON_HALVING

    @pytest.mark.parametrize("name", NAMES)
    def test_a_nan_parent_energy_is_rejected_rather_than_evaluated(
        self, name: str
    ) -> None:
        """The one reachable route to the boost integral's ``beta`` guard.

        The deleted Cython raised a bare ``AssertionError`` here, from
        ``assert 0.0 < beta < 1.0`` in ``hazma/_utils/boost.pyx``.
        ``projects/cython-to-rust/rules.md`` rule 9 makes an ``assert`` an
        unconditional ``ValueError``, and the message is the port's own
        because the assert carried none.
        """
        dnde, _, mass, _ = SPECTRA[name]
        with pytest.raises(ValueError, match="boost velocity must satisfy"):
            dnde(mass / 4.0, float("nan"))

    @pytest.mark.parametrize("name", NAMES)
    def test_a_nan_photon_energy_propagates_in_both_branches(self, name: str) -> None:
        """``NaN`` in, ``NaN`` out — a change from the Cython, declared.

        The rest-frame branch already did this (``np.interp`` propagates);
        the in-flight branch raised ``IndexError`` out of
        ``np.flatnonzero(lb <= x)[0]`` on an empty match. Reproducing that
        from inside an element-wise map would mean a panic, so the port
        answers ``NaN`` in both — see
        ``rust/src/boost.rs``'s "Faithfulness notes". The parity corpus
        samples no ``NaN`` abscissa, so no pinned value moves.
        """
        dnde, _, mass, _ = SPECTRA[name]
        assert math.isnan(dnde(float("nan"), mass))
        assert math.isnan(dnde(float("nan"), 2.0 * mass))
        mixed = np.asarray(dnde(np.array([mass / 4.0, np.nan]), 2.0 * mass))
        assert math.isfinite(mixed[0]) and math.isnan(mixed[1])

    @pytest.mark.parametrize("name", NAMES)
    def test_each_line_carries_the_photon_count_its_weight_declares(
        self, name: str
    ) -> None:
        """A boosted δ-function integrates to its own weight.

        The line term is ``w · δ(E - E₀)`` boosted, which spreads into a
        rectangle of height ``w / (2 gamma β E₀)`` and width
        ``2 gamma β E₀`` — so its integral is ``w`` at any boost, and ``w``
        is the photon count the
        mode contributes per decay. Isolating it by subtracting the
        continuum makes the test a statement about *this task's* wiring
        (which lines, at what energies, with what weights) rather than about
        the boost integral, which ``test/test_core_boost.py`` owns.

        The tolerance is derived, not chosen. A trapezoid across a
        discontinuity misplaces the edge by up to half a cell, so each of
        the rectangle's two edges costs ``height x cell / 2`` — a relative
        error of ``cell / width`` per line, and the narrowest line sets it.
        Four times that covers both edges with margin, and lands between
        1e-5 and 1e-4 here: five decades tighter than the factor-of-two
        error this test exists to catch, and tight enough that a line at
        the wrong energy (whose rectangle is then a different width) fails
        too.
        """
        dnde, _, mass, lines = SPECTRA[name]
        x, y = TABLES[name]
        parent = 2.0 * mass
        beta = float(core_boost.boost_beta(parent, mass))
        gamma = parent / mass

        expected = sum(weight for _, weight in lines)
        if not lines:
            pytest.skip("the charged kaon has no monochromatic line")

        lo = min(gamma * energy * (1.0 - beta) for energy, _ in lines) * 0.9
        hi = max(gamma * energy * (1.0 + beta) for energy, _ in lines) * 1.1
        grid = np.linspace(lo, hi, 200_001)
        cell = (hi - lo) / (grid.size - 1)
        narrowest = min(2.0 * gamma * beta * energy for energy, _ in lines)
        tolerance = 4.0 * cell / narrowest

        total = np.asarray(dnde(grid, parent))
        continuum = np.asarray(
            core_boost.boost_integrate_linear_interp(grid, beta, x, y)
        )
        integral = float(np.trapezoid(total - continuum, grid))
        assert integral == pytest.approx(expected, rel=tolerance), (
            f"{name}: the line terms carry {integral} photons per decay, "
            f"not the {expected} their weights declare"
        )
        # The tolerance has to stay far tighter than the defect classes
        # above, or this test would pass on a halved weight.
        assert tolerance < MAX_LINE_TOLERANCE

    def test_the_eta_prime_line_carries_half_the_photons_it_should(self) -> None:
        """A reproduced defect: ``2·BR`` everywhere but the η'.

        ``η' -> a a`` yields two photons, so the line's weight should be
        ``2·BR(η' -> a a) = 0.04614``. ``_eta_prime.pyx:107`` wrote ``BR``,
        which its four two-photon siblings did not, and the ω and φ weights
        are correctly un-doubled because their modes ``X -> Y a`` yield one
        photon each. Reproduced per rules.md rule 1 — the corpus pins the
        low values — and tracked in
        ``docs/followups/todo/eta-prime-two-photon-line-missing-factor-two.md``.
        """
        assert SPECTRA["eta_prime"][3] == [(MASS_ETAP / 2.0, BR_ETAP_TO_A_A)]
        assert SPECTRA["eta"][3] == [(MASS_ETA / 2.0, 2.0 * BR_ETA_TO_A_A)]
        # And the shipped kernel really carries the halved count: the same
        # isolate-the-line measurement as the test above, stated as the
        # ratio it should have had.
        lines = SPECTRA["eta_prime"][3]
        assert lines[0][1] == pytest.approx(BR_ETAP_TO_A_A)
        assert lines[0][1] == pytest.approx(0.5 * 2.0 * BR_ETAP_TO_A_A)

    def test_the_phi_lines_sit_at_the_daughter_mesons_energy(self) -> None:
        """A reproduced defect: ``+`` where the photon needs ``-``.

        For ``φ -> X a`` the photon carries ``(M² - m²)/(2M)`` and the meson
        ``(M² + m²)/(2M)``; the two sum to ``M``. ``_phi.pyx:111,113`` used
        the second for the photon line. Reproduced per rules.md rule 1 and
        tracked in
        ``docs/followups/todo/phi-photon-lines-use-the-daughter-meson-energy.md``.
        """
        for line_energy, daughter_mass, correct in (
            (SPECTRA["phi"][3][0][0], MASS_ETA, 362.5189975276151),
            (SPECTRA["phi"][3][1][0], MASS_ETAP, 59.815040556235125),
        ):
            photon = (MASS_PHI**2 - daughter_mass**2) / (2 * MASS_PHI)
            assert photon == pytest.approx(correct)
            assert line_energy + photon == pytest.approx(MASS_PHI)
            assert line_energy > photon
        # The ω's lines *are* the photon's, which is what makes the φ's a
        # defect rather than a convention the family shares.
        for line_energy, daughter_mass in (
            (SPECTRA["omega"][3][0][0], MASS_PI0),
            (SPECTRA["omega"][3][1][0], MASS_ETA),
        ):
            assert line_energy == pytest.approx(
                (MASS_OMEGA**2 - daughter_mass**2) / (2 * MASS_OMEGA)
            )

    @pytest.mark.parametrize("name", NAMES)
    def test_the_spectrum_is_non_negative_across_its_support(self, name: str) -> None:
        dnde, _, mass, _ = SPECTRA[name]
        for multiplier in PARENT_MULTIPLIERS:
            parent = mass * multiplier
            values = np.asarray(dnde(probe_grid(mass, parent), parent))
            assert np.all(values >= 0.0), f"{name} at E = {parent}"
            assert np.all(np.isfinite(values)), f"{name} at E = {parent}"

    def test_the_boost_integral_still_diverges_near_threshold(self) -> None:
        """The Task 3.4 defect, reproduced through a public entry point.

        ``boost_integrate_linear_interp`` over-counts when both integration
        bounds fall inside one table cell, by (cell width)/(window width) —
        which diverges as ``β → 0``. So a barely-moving parent gives a
        spectrum orders of magnitude *above* its own rest-frame value
        instead of converging to it. Pinned here because this family is
        where it is visible from outside:
        ``docs/followups/todo/boost-integral-drops-last-interior-cell.md``
        is blocked until after Phase 06 Task 6.4, and the corpus pins the
        wrong values by design, so a swap that "fixed" it would fail its
        own gate.
        """
        mass = MASS_ETA
        dnde = core_photon.dnde_photon_eta
        at_rest = dnde(100.0, mass)
        barely_moving = dnde(100.0, mass * (1.0 + 1e-12))
        assert barely_moving / at_rest > MIN_THRESHOLD_DIVERGENCE
