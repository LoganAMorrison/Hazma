""":mod:`hazma._core` — the two mediator decay *positron* spectra.

cython-to-rust Phase 06 Task 6.3. Covers
``hazma._core.scalar_mediator.dnde_positron_decay_s`` / ``_pt`` and
``hazma._core.vector_mediator.dnde_positron_decay_v`` / ``_pt``, which
replace ``hazma/{scalar,vector}_mediator/*_mediator_positron_spec.pyx`` —
deleted in the same PR, as ``projects/cython-to-rust/rules.md`` rule 1
requires. Both wrappers re-export them under the ``.pyx``'s own
``dnde_decay_s``/``dnde_decay_v`` names, which is the spelling every
caller outside this file uses.

One module for both, as for the decay pair — but here the clone-pair is
literal. Normalise one ``.pyx`` against the other by rewriting ``s``/``v``,
``ms``/``mv``, ``eng_s``/``eng_v`` and "scalar"/"vector" and ``diff``
reports nothing but those substitutions and the order of two ``import``
lines, so the port serves all four entry points from one Rust kernel and
:class:`TestAgainstAnIndependentReference` asserts that the two models
agree bit-for-bit rather than merely closely.

The five parts
--------------
1. :class:`TestDispatchWiring` — the array-only contract and the
   exception wordings. Reasoning about the helpers themselves stays in
   ``test/test_core_dispatch.py``.
2. :class:`TestAgainstAnIndependentReference` — the ``.pyx`` body
   re-transcribed in NumPy and ``scipy.integrate.quad``
   (:func:`reference`), compared at a stated budget.
3. :class:`TestPhysics` — statements that owe nothing to the
   implementation being replaced: thresholds, the line's positron count,
   additivity over channels, support, and broadcasting.
4. :class:`TestErrorPaths` — every documented failure mode, including the
   two the port reproduces rather than repairs.
5. :class:`TestTheThresholdSingularity` — the one value this swap
   deliberately moves, and the neighbours it does not.

Why there is no Cython oracle here
----------------------------------
Both twins are deleted in this PR, so there is no ``def`` and no ``cdef``
left to call. The against-the-Cython evidence is the **parity corpus**,
which pins all four of these entry points to their pre-port values and is
what gates the swap; the drift it measures is in
``projects/cython-to-rust/task-notes/phase-06/task-6.3-positron-spectra.md``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from hazma import spectra
from hazma._core import scalar_mediator as core_scalar
from hazma._core import vector_mediator as core_vector

#: The four entry points, under the names the deleted `.pyx` gave them.
#: Bound here rather than imported directly so the import block matches
#: `test/test_core_mediator_decay_photon.py`'s, which is the shape the
#: repo's `isort` and `ruff` configurations agree on.
dnde_positron_decay_s = core_scalar.dnde_positron_decay_s
dnde_positron_decay_s_pt = core_scalar.dnde_positron_decay_s_pt
dnde_positron_decay_v = core_vector.dnde_positron_decay_v
dnde_positron_decay_v_pt = core_vector.dnde_positron_decay_v_pt

REPO_ROOT = Path(__file__).resolve().parents[1]

#: `hazma/_utils/legacy_parameters.pxd:18`, which both `.pyx`
#: `include`d. Spelled out rather than imported from `hazma.parameters`,
#: whose `electron_mass` is the *modern* `0.5109989461`, so that a future
#: consolidation of the two tables cannot silently move these tests with
#: the code (`projects/cython-to-rust/rules.md` rule 4).
LEGACY_MASS_E = 0.510998928

#: `scalar_mediator_positron_spec.pyx:26` — the rest-frame table size.
N_INTERP_PTS = 500

#: The four final-state strings both `.pyx` compared against
#: (`:150-161` and the vector clone's `:151-162`).
MODES = ["total", "e e", "mu mu", "pi pi"]

#: `[e e, mu mu, pi pi]`, the order both `.pyx` indexed `pws` in
#: (`:135-136`, `:203`). Deliberately not normalised to one: nothing in
#: the kernel requires it, and a non-unit sum makes an accidental
#: renormalisation visible.
PWS = np.array([0.31, 0.44, 0.17])

#: `(mediator mass, mediator energy)` in MeV. The masses straddle the
#: two rest-frame thresholds: at 125 MeV the daughter energy `m/2` is
#: below both `m_mu` and `m_pi`, so both tables are identically zero and
#: the spectrum is its line term alone; at 600 MeV both channels are
#: open. The energies are barely-boosted and hard-boosted. `energy ==
#: mass` is excluded on purpose — there the line term divides by
#: `beta == 0`.
CONFIGS = [(125.0, 200.0), (125.0, 1000.0), (600.0, 700.0), (600.0, 3000.0)]

#: The budget :func:`reference` is compared at. The reference integrates
#: with scipy's QUADPACK binding and the port with the in-tree port of
#: the same algorithm, and the reference's arithmetic is unfused where
#: the `.pyx`'s C tree fused. 1e-9 is
#: `test/parity/tolerances.PORTED_NESTED_RTOL`, the figure Task 4.5
#: established for exactly this "nested quadrature, ported integrator"
#: shape and the one all four corpus cases now hold; the worst difference
#: measured over every (model, config, mode, energy) below is 1.6e-14, at
#: scalar/`mass=600`/`energy=3000`/`"mu mu"`/`eng_p=100`.
REFERENCE_RTOL = 1e-9


# ===========================================================================
# ---- The independent reference --------------------------------------------
# ===========================================================================


def _tables(mass: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(energies, charged_pion, muon)`` as ``__set_spectra`` built them.

    ``:72-74``: a 500-point ``numpy.logspace`` from the **legacy** ``m_e``
    to ``m/2``, evaluated at the daughter energy ``m/2``. The ``.pyx``
    reached the two kernels through a ``cimport`` of their ``cdef``
    ``*_array`` twins; the public wrappers are the same kernels, and
    using them keeps this reference free of anything Task 6.3 wrote.
    """
    energies = np.logspace(
        math.log10(LEGACY_MASS_E), math.log10(mass / 2.0), num=N_INTERP_PTS
    )
    daughter = mass / 2.0
    return (
        energies,
        np.asarray(spectra.dnde_positron_charged_pion(energies, daughter), float),
        np.asarray(spectra.dnde_positron_muon(energies, daughter), float),
    )


def reference(
    eng_p: float, energy: float, mass: float, pws: np.ndarray, fs: str
) -> float:
    """The deleted ``.pyx`` body, re-derived in NumPy and scipy.

    ``__dnde_decay_s`` at ``:166-215`` over ``__integrand`` at
    ``:106-161``, transcribed from the source rather than from the port.
    Note ``np.interp`` and not the decay modules' ``1/E`` tail: the
    positron modules clamp below the grid (``:97-99``), which is
    ``np.interp``'s own behaviour and the whole difference in below-grid
    policy between the two clone-pairs.
    """
    if energy < mass:
        return 0.0

    beta = math.sqrt(1.0 - (mass / energy) ** 2)
    gamma = energy / mass
    r = math.sqrt(1.0 - 4.0 * LEGACY_MASS_E * LEGACY_MASS_E / (mass * mass))
    eplus = energy * (1.0 + r * beta) / 2.0
    eminus = energy * (1.0 - r * beta) / 2.0

    lines_contrib = 0.0
    if eminus <= eng_p <= eplus:
        lines_contrib = pws[0] / (energy * beta)

    if fs == "e e":
        return lines_contrib
    if fs not in {"total", "pi pi", "mu mu"}:
        return 0.0

    energies, cp_dnde, mu_dnde = _tables(mass)

    def integrand(cl: float) -> float:
        if eng_p < LEGACY_MASS_E:
            return 0.0
        p = math.sqrt(max(eng_p * eng_p - LEGACY_MASS_E * LEGACY_MASS_E, 0.0))
        rest_frame = gamma * (eng_p - p * beta * cl)
        jac = p / (
            2.0
            * math.sqrt(
                (1.0 + (beta * cl) ** 2) * eng_p * eng_p
                - (1.0 + beta * beta * (-1.0 + cl * cl)) * LEGACY_MASS_E * LEGACY_MASS_E
                - 2.0 * beta * cl * eng_p * p
            )
            * gamma
        )
        dnde = 0.0
        if fs in {"total", "pi pi"}:
            dnde += pws[2] * float(np.interp(rest_frame, energies, cp_dnde))
        if fs in {"total", "mu mu"}:
            dnde += pws[1] * float(np.interp(rest_frame, energies, mu_dnde))
        return jac * dnde

    value = quad(integrand, -1.0, 1.0, points=[-1.0, 1.0], epsabs=1e-10, epsrel=1e-5)[0]
    return value + lines_contrib


#: ``(label, array entry point, pointwise entry point)`` for the two
#: models. The port serves both from one kernel, so every parameterised
#: test below runs twice and :meth:`the_two_models_agree_bit_for_bit`
#: asserts that is not a coincidence.
MODELS = [
    ("scalar", dnde_positron_decay_s, dnde_positron_decay_s_pt),
    ("vector", dnde_positron_decay_v, dnde_positron_decay_v_pt),
]
MODEL_IDS = [name for name, _, _ in MODELS]


# ===========================================================================
# ---- Part 1: the dispatch contract ----------------------------------------
# ===========================================================================


class TestDispatchWiring:
    """The array-only contract both ``.pyx`` signatures imposed."""

    @pytest.mark.parametrize(("_name", "array_fn", "point_fn"), MODELS, ids=MODEL_IDS)
    def test_the_array_entry_point_returns_a_fresh_float64_array(
        self, _name: str, array_fn: object, point_fn: object
    ) -> None:
        grid = np.array([1.0, 10.0, 100.0])
        out = array_fn(grid, 200.0, 125.0, PWS, "total")
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float64
        assert out.shape == grid.shape
        assert out is not grid
        # And the pointwise twin agrees element by element.
        assert list(out) == [point_fn(e, 200.0, 125.0, PWS, "total") for e in grid]

    @pytest.mark.parametrize(("_name", "array_fn", "_point"), MODELS, ids=MODEL_IDS)
    def test_a_list_is_accepted_where_the_pyx_refused_one(
        self, _name: str, array_fn: object, _point: object
    ) -> None:
        # A declared widening: `np.ndarray[double]` refused a list, and
        # `require_vector` accepts any sequence that converts. Same
        # divergence Task 6.2 declared for the photon pair.
        assert array_fn([1.0, 10.0], 200.0, 125.0, PWS, "total") == pytest.approx(
            array_fn(np.array([1.0, 10.0]), 200.0, 125.0, PWS, "total")
        )

    @pytest.mark.parametrize(("_name", "array_fn", "_point"), MODELS, ids=MODEL_IDS)
    def test_a_scalar_energy_is_refused_by_the_array_entry_point(
        self, _name: str, array_fn: object, _point: object
    ) -> None:
        # The other declared divergence: the `.pyx` raised `TypeError`
        # from the buffer cast, the port raises `ValueError` from its own
        # check. The refusal itself is what the contract owes.
        with pytest.raises(
            ValueError, match=r"Positron energies must be a list or array\."
        ):
            array_fn(10.0, 200.0, 125.0, PWS, "total")

    @pytest.mark.parametrize(("_name", "array_fn", "_point"), MODELS, ids=MODEL_IDS)
    def test_a_rank_error_names_the_quantity(
        self, _name: str, array_fn: object, _point: object
    ) -> None:
        with pytest.raises(
            ValueError, match=r"Positron energies must be 1-dimensional\."
        ):
            array_fn(np.ones((2, 2)), 200.0, 125.0, PWS, "total")
        with pytest.raises(ValueError, match=r"Partial widths must be 1-dimensional\."):
            array_fn(np.array([1.0]), 200.0, 125.0, np.ones((2, 2)), "total")

    @pytest.mark.parametrize(("_name", "array_fn", "_point"), MODELS, ids=MODEL_IDS)
    def test_an_empty_grid_returns_an_empty_grid(
        self, _name: str, array_fn: object, _point: object
    ) -> None:
        out = array_fn(np.array([]), 200.0, 125.0, PWS, "total")
        assert isinstance(out, np.ndarray)
        assert out.shape == (0,)

    @pytest.mark.parametrize(("_name", "_array", "point_fn"), MODELS, ids=MODEL_IDS)
    def test_the_signatures_accept_keywords(
        self, _name: str, _array: object, point_fn: object
    ) -> None:
        assert (
            point_fn(eng_p=100.0, eng_s=200.0, ms=125.0, pws=PWS, fs="total")
            if _name == "scalar"
            else point_fn(eng_p=100.0, eng_v=200.0, mv=125.0, pws=PWS, fs="total")
        )


# ===========================================================================
# ---- Part 2: against an independent reference ------------------------------
# ===========================================================================


class TestAgainstAnIndependentReference:
    """The port reproduces :func:`reference` inside :data:`REFERENCE_RTOL`.

    The reference is the deleted ``.pyx`` body re-transcribed from source
    into NumPy and ``scipy.integrate.quad``. It shares no code with the
    port except the Phase 04 positron kernels the ``.pyx`` itself
    cimported, which are what the tables are made of on both sides.
    """

    @pytest.mark.parametrize(("mass", "energy"), CONFIGS)
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize(("_name", "_array", "point_fn"), MODELS, ids=MODEL_IDS)
    def test_the_spectrum_matches_channel_by_channel(
        self,
        _name: str,
        _array: object,
        point_fn: object,
        mass: float,
        energy: float,
        mode: str,
    ) -> None:
        for eng_p in (1.0, 20.0, 100.0, 0.4 * energy):
            want = reference(eng_p, energy, mass, PWS, mode)
            got = point_fn(eng_p, energy, mass, PWS, mode)
            assert got == pytest.approx(
                want, rel=REFERENCE_RTOL, abs=0.0
            ), f"{mode} at eng_p={eng_p}, mass={mass}, energy={energy}"

    @pytest.mark.parametrize(("mass", "energy"), CONFIGS)
    @pytest.mark.parametrize("mode", MODES)
    def test_the_two_models_agree_bit_for_bit(
        self, mass: float, energy: float, mode: str
    ) -> None:
        # The two `.pyx` were the same text, so the port serves both from
        # one kernel. This is what makes that claim falsifiable: any
        # future edit that gives one model its own arithmetic fails here
        # before it reaches the corpus.
        grid = np.logspace(0.0, math.log10(0.4 * energy), 23)
        scalar = dnde_positron_decay_s(grid, energy, mass, PWS, mode)
        vector = dnde_positron_decay_v(grid, energy, mass, PWS, mode)
        assert scalar.tobytes() == vector.tobytes()

    @pytest.mark.parametrize(("mass", "energy"), CONFIGS)
    @pytest.mark.parametrize(("_name", "array_fn", "point_fn"), MODELS, ids=MODEL_IDS)
    def test_the_two_entry_points_agree_bit_for_bit(
        self,
        _name: str,
        array_fn: object,
        point_fn: object,
        mass: float,
        energy: float,
    ) -> None:
        grid = np.logspace(0.0, math.log10(0.4 * energy), 23)
        swept = array_fn(grid, energy, mass, PWS, "total")
        one_at_a_time = np.array(
            [point_fn(e, energy, mass, PWS, "total") for e in grid]
        )
        assert swept.tobytes() == one_at_a_time.tobytes()


# ===========================================================================
# ---- Part 3: physics -------------------------------------------------------
# ===========================================================================


class TestPhysics:
    """Statements about the spectrum, not about the code it replaced."""

    @pytest.mark.parametrize(("_name", "_array", "point_fn"), MODELS, ids=MODEL_IDS)
    def test_a_mediator_below_its_own_mass_contributes_nothing(
        self, _name: str, _array: object, point_fn: object
    ) -> None:
        assert point_fn(100.0, 124.0, 125.0, PWS, "total") == 0.0

    @pytest.mark.parametrize(("_name", "array_fn", "_point"), MODELS, ids=MODEL_IDS)
    def test_the_electron_line_carries_its_own_positron_count(
        self, _name: str, array_fn: object, _point: object
    ) -> None:
        # `S/V -> e+ e-` is a line at `m/2` in the rest frame, boosted to
        # a flat box between `eminus` and `eplus`. The box is `E r beta`
        # wide and `pw_ee / (E beta)` tall (`:197-203`), so it integrates
        # to `pw_ee * r` — *not* to `pw_ee`, which is what one positron
        # per decay weighted by its branching fraction would give.
        #
        # The missing `1/r` is a defect in the code this port replaces,
        # reproduced here rather than repaired because `rules.md` rule 1
        # forbids a physics change inside a swap. It is worth 3.3e-5 at
        # this mass and diverges as `m -> 2 m_e`. Filed as
        # `docs/followups/todo/mediator-positron-line-misses-the-electron-velocity.md`,
        # which is also what flips this assertion back.
        mass, energy = 125.0, 200.0
        beta = math.sqrt(1.0 - (mass / energy) ** 2)
        r = math.sqrt(1.0 - 4.0 * LEGACY_MASS_E**2 / mass**2)
        eminus, eplus = energy * (1.0 - r * beta) / 2.0, energy * (1.0 + r * beta) / 2.0
        grid = np.linspace(eminus, eplus, 4001)
        box = array_fn(grid, energy, mass, PWS, "e e")
        # The trapezoid is exact on a constant and the box is one, so the
        # only error is the half-cell at each end; 1e-9 is far inside it.
        assert np.trapezoid(box, grid) == pytest.approx(PWS[0] * r, rel=1e-9)
        # And the deficit really is the `r`, not a coincidence of scale.
        assert np.trapezoid(box, grid) < PWS[0]

    @pytest.mark.parametrize(("mass", "energy"), CONFIGS)
    @pytest.mark.parametrize(("_name", "_array", "point_fn"), MODELS, ids=MODEL_IDS)
    def test_the_channels_are_additive(
        self,
        _name: str,
        _array: object,
        point_fn: object,
        mass: float,
        energy: float,
    ) -> None:
        # Every recognised mode carries the line, so the two continua add
        # to the total once the line each of them repeats is removed.
        for eng_p in (20.0, 100.0, 0.4 * energy):

            def at(mode: str, eng_p: float = eng_p) -> float:
                return point_fn(eng_p, energy, mass, PWS, mode)

            line = at("e e")
            residual = (
                (at("mu mu") - line) + (at("pi pi") - line) - (at("total") - line)
            )
            # The integrator's own `epsrel`, not the algebra's: the three
            # calls subdivide independently, so they converge to
            # different node sets.
            assert abs(residual) <= 1e-5 * abs(at("total") - line) + 1e-300

    @pytest.mark.parametrize(("_name", "array_fn", "_point"), MODELS, ids=MODEL_IDS)
    def test_the_spectrum_is_non_negative_everywhere_it_is_sampled(
        self, _name: str, array_fn: object, _point: object
    ) -> None:
        grid = np.logspace(-2, 3.5, 400)
        for mass, energy in CONFIGS:
            values = array_fn(grid, energy, mass, PWS, "total")
            assert np.all(np.isfinite(values))
            assert np.all(values >= 0.0)

    @pytest.mark.parametrize(("_name", "array_fn", "_point"), MODELS, ids=MODEL_IDS)
    def test_zero_partial_widths_give_a_zero_spectrum(
        self, _name: str, array_fn: object, _point: object
    ) -> None:
        grid = np.logspace(0.0, 2.5, 64)
        values = array_fn(grid, 700.0, 600.0, np.zeros(3), "total")
        assert np.count_nonzero(values) == 0

    @pytest.mark.parametrize(("_name", "array_fn", "_point"), MODELS, ids=MODEL_IDS)
    def test_a_dark_continuum_leaves_only_the_line(
        self, _name: str, array_fn: object, _point: object
    ) -> None:
        # At `mass = 125` the daughter energy is `62.5` MeV, below both
        # `m_mu` and `m_pi`, so neither table has a non-zero entry and the
        # continuum modes reduce to the line exactly.
        grid = np.logspace(0.0, 2.0, 64)
        line = array_fn(grid, 200.0, 125.0, PWS, "e e")
        for mode in ("total", "mu mu", "pi pi"):
            assert array_fn(grid, 200.0, 125.0, PWS, mode).tobytes() == line.tobytes()


# ===========================================================================
# ---- Part 4: error paths ---------------------------------------------------
# ===========================================================================


class TestErrorPaths:
    """Every documented failure mode, and the two reproduced rather than fixed."""

    @pytest.mark.parametrize("length", [0, 1, 2])
    @pytest.mark.parametrize(("_name", "_array", "point_fn"), MODELS, ids=MODEL_IDS)
    def test_a_short_partial_width_buffer_raises_index_error(
        self, _name: str, _array: object, point_fn: object, length: int
    ) -> None:
        # Inside the line window, so `pws[0]` is read too and every
        # length below three is short for something.
        with pytest.raises(IndexError, match="Out of bounds on buffer access"):
            point_fn(100.0, 200.0, 125.0, np.zeros(length), "total")

    @pytest.mark.parametrize(("_name", "_array", "point_fn"), MODELS, ids=MODEL_IDS)
    def test_the_first_width_is_read_only_inside_the_line_window(
        self, _name: str, _array: object, point_fn: object
    ) -> None:
        # Reproduced, not repaired: `pws[0]` is read under the window
        # test alone (`:202-203`), so an empty buffer legitimately
        # succeeds outside it. A port that validated the buffer up front
        # would have broken a working call — the same finding Task 6.2
        # recorded for the photon pair's `pws[4]`.
        empty = np.array([])
        assert point_fn(1.0, 200.0, 125.0, empty, "e e") == 0.0
        with pytest.raises(IndexError, match="Out of bounds on buffer access"):
            point_fn(100.0, 200.0, 125.0, empty, "e e")

    @pytest.mark.parametrize(("_name", "_array", "point_fn"), MODELS, ids=MODEL_IDS)
    def test_the_electron_line_never_reaches_the_other_two_widths(
        self, _name: str, _array: object, point_fn: object
    ) -> None:
        # `"e e"` returns before the integral (`:205-206`), so `pws[1]`
        # and `pws[2]` are never read and a one-element buffer suffices.
        assert point_fn(100.0, 200.0, 125.0, np.array([0.31]), "e e") > 0.0

    @pytest.mark.parametrize("mode", ["", "e e g", "pi0 g", "not a mode", None])
    @pytest.mark.parametrize(("_name", "_array", "point_fn"), MODELS, ids=MODEL_IDS)
    def test_an_unrecognised_mode_returns_zero(
        self, _name: str, _array: object, point_fn: object, mode: object
    ) -> None:
        # Reproduced, not repaired: every `cdef double` integrand ends in
        # an `if`-chain with no `else`, and a C function that falls off
        # its end returns zero. The line term is computed and discarded,
        # so even inside the window the answer is `0.0`. Filed as
        # `docs/followups/todo/mediator-spectra-accept-unknown-mode-strings.md`.
        assert point_fn(100.0, 200.0, 125.0, PWS, mode) == 0.0

    @pytest.mark.parametrize(("_name", "_array", "point_fn"), MODELS, ids=MODEL_IDS)
    def test_an_unrecognised_mode_still_reads_the_first_width(
        self, _name: str, _array: object, point_fn: object
    ) -> None:
        # The discarding happens *after* the window test, so a buffer too
        # short for `pws[0]` raises even though the answer would be zero.
        with pytest.raises(IndexError, match="Out of bounds on buffer access"):
            point_fn(100.0, 200.0, 125.0, np.array([]), "not a mode")

    @pytest.mark.parametrize(("_name", "_array", "point_fn"), MODELS, ids=MODEL_IDS)
    def test_a_mediator_below_its_mass_does_not_read_the_widths(
        self, _name: str, _array: object, point_fn: object
    ) -> None:
        # `:194-195` returns before touching `pws`.
        assert point_fn(100.0, 124.0, 125.0, np.array([]), "total") == 0.0


# ===========================================================================
# ---- Part 5: the threshold singularity -------------------------------------
# ===========================================================================


class TestTheThresholdSingularity:
    """The one value this swap moves, and the neighbours it does not.

    At exactly the legacy ``m_e`` the shipped extension returned ``nan``
    from every continuum mode, because clang contracted
    ``sqrt(eng_p * eng_p - me * me)`` into an FMA whose radicand is the
    rounding of ``me * me`` — negative, by ``1.45e-17``. The momentum at
    the threshold is zero, so the port clamps the radicand and answers
    ``0.0``, which is also the limit from both sides.

    Chosen over consolidating the two ``MASS_E`` tables, which would have
    moved published spectra near threshold and relocated the singular
    point; see
    ``docs/followups/done/positron-spectrum-nan-at-legacy-electron-mass.md``.
    """

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize(("_name", "array_fn", "_point"), MODELS, ids=MODEL_IDS)
    def test_the_legacy_electron_mass_is_finite(
        self, _name: str, array_fn: object, _point: object, mode: str
    ) -> None:
        value = array_fn(np.array([LEGACY_MASS_E]), 250.0, 125.0, PWS, mode)[0]
        assert value == 0.0

    @pytest.mark.parametrize(("_name", "array_fn", "_point"), MODELS, ids=MODEL_IDS)
    def test_the_neighbouring_doubles_are_untouched(
        self, _name: str, array_fn: object, _point: object
    ) -> None:
        # Both already answered `0.0` before the clamp, and the point is
        # that the clamp did not widen: the fix is one double wide.
        neighbours = np.array(
            [
                math.nextafter(LEGACY_MASS_E, 0.0),
                math.nextafter(LEGACY_MASS_E, math.inf),
            ]
        )
        assert list(array_fn(neighbours, 250.0, 125.0, PWS, "total")) == [0.0, 0.0]

    @pytest.mark.parametrize(("_name", "array_fn", "_point"), MODELS, ids=MODEL_IDS)
    def test_no_energy_near_the_threshold_is_a_nan(
        self, _name: str, array_fn: object, _point: object
    ) -> None:
        # The follow-up's own sweep found exactly one `nan` in
        # `[0.5109988, 0.5109990]`, at the threshold. A coarser sweep of
        # the same interval is enough to catch a clamp that missed.
        grid = np.linspace(0.5109988, 0.5109990, 20001)
        assert np.all(np.isfinite(array_fn(grid, 250.0, 125.0, PWS, "total")))


# ===========================================================================
# ---- The twins are gone ----------------------------------------------------
# ===========================================================================


class TestTheCythonTwinsAreGone:
    """``rules.md`` rule 1: no second reachable implementation.

    Asserted on the sources and on ``setup.py`` rather than by importing,
    because a built ``.so`` outlives its deleted ``.pyx`` — see
    ``docs/agents/environment.md``, and
    ``test/test_core_photon_rho.py`` for the worked example.
    """

    def test_neither_pyx_is_on_disk(self) -> None:
        for path in (
            "hazma/scalar_mediator/scalar_mediator_positron_spec.pyx",
            "hazma/vector_mediator/vector_mediator_positron_spec.pyx",
            "hazma/vector_mediator/vector_mediator_positron_spec.pyi",
        ):
            assert not (REPO_ROOT / path).exists(), path

    def test_setup_py_builds_neither(self) -> None:
        # The quoted form, which is how `make_extension` names a module.
        # `setup.py` still *mentions* both in the comment recording where
        # they went, and that comment is not a build instruction.
        setup = (REPO_ROOT / "setup.py").read_text()
        assert '"scalar_mediator_positron_spec"' not in setup
        assert '"vector_mediator_positron_spec"' not in setup

    def test_neither_mediator_package_builds_any_extension(self) -> None:
        # Task 6.3 took the last one from each, which is what made Task
        # 6.4's sweep over the four capi survivors possible. Kept as a
        # per-package guard now that `test/test_no_cython_remains.py`
        # carries the tree-wide claim.
        for package in ("scalar_mediator", "vector_mediator"):
            assert not list((REPO_ROOT / "hazma" / package).glob("*.pyx"))
            assert not list((REPO_ROOT / "hazma" / package).glob("*.pxd"))

    def test_the_wrappers_re_export_the_cython_names(self) -> None:
        from hazma.scalar_mediator import (  # noqa: PLC0415
            _scalar_mediator_positron_spectra as scalar_wrapper,
        )
        from hazma.vector_mediator import (  # noqa: PLC0415
            _vector_mediator_positron_spectra as vector_wrapper,
        )

        assert scalar_wrapper.dnde_decay_s is dnde_positron_decay_s
        assert scalar_wrapper.dnde_decay_s_pt is dnde_positron_decay_s_pt
        assert vector_wrapper.dnde_decay_v is dnde_positron_decay_v
        assert vector_wrapper.dnde_decay_v_pt is dnde_positron_decay_v_pt
