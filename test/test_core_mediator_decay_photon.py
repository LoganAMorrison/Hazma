"""``hazma._core`` — the two mediator decay *photon* spectra.

cython-to-rust Phase 06 Task 6.2. Covers
``hazma._core.scalar_mediator.scalar_mediator_decay_spectrum`` and
``hazma._core.vector_mediator.dnde_decay_v`` / ``dnde_decay_v_pt``, which
replace ``hazma/{scalar,vector}_mediator/*_mediator_decay_spectrum.pyx``
— deleted in the same PR, as ``projects/cython-to-rust/rules.md`` rule 1
requires.

One module for both because they are a clone-pair: the same 500-point
log-spaced rest-frame table, the same ``1/E`` tail below ``10**-1`` MeV,
the same ``cos(theta)`` boost integral with the same quadrature settings,
and the same monochromatic line added outside it. Only the channel list,
the FSR formulae and the selector type differ, so the reference in
:class:`TestAgainstAnIndependentReference` is written once and
parameterised — the shape ``test/test_core_photon_tables.py`` uses for
its own family of near-copies.

The four parts
--------------
1. :class:`TestDispatchWiring` — one assertion per contract branch, plus
   the exception wordings the ``.pyx`` gave each argument. Reasoning about
   the helpers themselves stays in ``test/test_core_dispatch.py``.
2. :class:`TestAgainstAnIndependentReference` — the ``.pyx`` bodies
   re-transcribed in NumPy and ``scipy.integrate.quad`` (:func:`reference`
   below), compared at a stated budget.
3. :class:`TestPhysics` — statements that owe nothing to the
   implementation being replaced: thresholds, support, the line's photon
   count, additivity over channels, and broadcasting.
4. :class:`TestErrorPaths` — every documented failure mode, including the
   two the port reproduces rather than repairs.

Why there is no Cython oracle here
----------------------------------
Both twins are deleted in this PR, so there is no ``cdef`` left to call —
the same situation ``test/test_core_photon_tables.py`` and
``test/test_core_neutrino.py`` are in, and the same answer: the
against-the-Cython evidence is the **parity corpus**, which pins all
three of these entry points to their pre-port values and is what gates
the swap.

Before the twins were removed the port was additionally compared against
them directly, over 5,325 points — five ``(mass, energy)`` configurations
crossed with every mode of both entry points and 71 photon energies
spanning six decades — giving **71.6% bit-equal and a worst relative
difference of 2.2e-12**, at
``scalar_mediator_decay_spectrum(..., modes=["pi0 pi0"])`` where the
integrand is the neutral pion's discontinuous box and the adaptive
subdivision amplifies a last-bit disagreement. The residual is the
quadrature port's, not the transliteration's: at ``eng_s == ms`` the boost
integrand is a constant and every channel agrees to within one ulp, while
``crate::quad`` is already known not to be bit-equal to scipy's QUADPACK
(``PORTED_QUAD_RTOL`` exists for that reason). Full evidence:
``projects/cython-to-rust/task-notes/phase-06/task-6.2-decay-spectra.md``.

Why the reference is not compared bit-for-bit
---------------------------------------------
:func:`reference` integrates with ``scipy.integrate.quad`` where the port
integrates with ``crate::quad``, and writes its arithmetic unfused where
the ``.pyx``'s C tree fused thirty-seven multiply-adds. Both differences
are real and neither is a defect, so the comparison carries
:data:`REFERENCE_RTOL`. That is the *point* of the reference: it re-derives
the algorithm from the deleted source without inheriting the port's
choices, and a transliteration error large enough to matter would not fit
inside a budget three decades under the corpus's own.
"""

from __future__ import annotations

import inspect
import math
import re
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.integrate import quad

from hazma import spectra
from hazma._core import scalar_mediator as core_scalar
from hazma._core import vector_mediator as core_vector

if TYPE_CHECKING:
    from collections.abc import Callable

#: What :func:`reference` selects with: a list of mode names for the
#: scalar entry point, one ``mode`` string for the vector ones. Each
#: source's own argument type, kept distinct so the two halves of the
#: reference cannot be called with the wrong one by accident.
Selector = list[str] | str | None

scalar_spectrum = core_scalar.scalar_mediator_decay_spectrum
dnde_decay_v = core_vector.dnde_decay_v
dnde_decay_v_pt = core_vector.dnde_decay_v_pt

# ===========================================================================
# ---- Constants, transcribed from the deleted sources ----------------------
# ===========================================================================

#: ``hazma/_utils/legacy_parameters.pxd``, which all four mediator spectrum
#: ``.pyx`` files ``include``\ d -- the two this task deleted among them.
#: Spelled out rather than imported from ``hazma.parameters`` so a future
#: consolidation of the two constant tables cannot silently move these
#: tests with the code (``projects/cython-to-rust/rules.md`` rule 4).
#: These are *not* the PDG values ``hazma/_utils/constants.pxd`` carries.
MASS_E = 0.510998928
MASS_MU = 105.6583715
MASS_PI0 = 134.9766
MASS_PI = 139.57018
ALPHA_EM = 1.0 / 137.0

#: ``qe = sqrt(4 pi alpha)``, the module-level ``cdef double`` both
#: ``.pyx`` files declared.
QE = math.sqrt(4.0 * math.pi * ALPHA_EM)

#: Points in the rest-frame interpolation table — ``n_interp_pts`` in both
#: sources.
N_INTERP_PTS = 500

#: The decay modules' lower grid endpoint, written as the literal exponent
#: ``-1.0`` and reused as the threshold of the ``1/E`` tail below it.
GRID_LOG10_START = -1.0

#: The quadrature keywords both entry points pass
#: (``scalar_mediator_decay_spectrum.pyx:184-186``,
#: ``vector_mediator_decay_spectrum.pyx:219-221``). ``points`` selects
#: QAGP even though scipy discards both entries as non-interior.
QUAD_KWARGS = {"points": [-1.0, 1.0], "epsabs": 1e-10, "epsrel": 1e-5}

#: The scalar entry point's default ``modes``, in source order.
SCALAR_MODES = ["pi pi", "mu mu", "pi0 pi0", "g g", "e e g", "pi pi g", "mu mu g"]

#: Every ``mode`` string ``vector_mediator_decay_spectrum.pyx:166-178``
#: compares against, in source order.
VECTOR_MODES = ["total", "e e g", "pi pi g", "pi pi", "pi0 g", "mu mu g", "mu mu"]

#: Normalised partial widths for the scalar entry point, indexed
#: ``[e e, mu mu, pi0 pi0, pi pi, g g]``
#: (``hazma/scalar_mediator/_scalar_mediator_spectra.py:74-78``). All five
#: distinct, so a channel reading the wrong slot cannot pass unnoticed.
SCALAR_PWS = np.array([0.31, 0.17, 0.23, 0.11, 0.05])

#: Normalised partial widths for the vector entry points, indexed
#: ``[e e, mu mu, pi0 g, pi pi]``
#: (``hazma/vector_mediator/_vector_mediator_spectra.py:87-90``).
VECTOR_PWS = np.array([0.31, 0.17, 0.11, 0.23])

#: ``(mediator mass, mediator energy)`` in MeV: at rest, barely boosted,
#: and hard-boosted. The rest case is the one where the boost integrand is
#: a constant, which is what isolates the integrand from the integrator.
CONFIGS = [(550.0, 550.0), (550.0, 600.0), (550.0, 1500.0)]

#: The budget :func:`reference` is compared at. The reference integrates
#: with scipy's QUADPACK binding and the port with the in-tree port of the
#: same algorithm, and the reference's arithmetic is unfused where the
#: ``.pyx``'s C tree fused; 1e-9 is
#: ``test/parity/tolerances.PORTED_NESTED_RTOL``, the figure Task 4.5
#: established for exactly this "nested quadrature, ported integrator"
#: shape, and the worst difference measured here is 3.5e-12.
REFERENCE_RTOL = 1e-9

#: The additivity budget. Each single-channel call is its own adaptive
#: quadrature, so the sum of the channels is not the integral of the sum;
#: the honest bound is the integrator's own relative tolerance.
ADDITIVITY_RTOL = 1e-5

#: The exception wordings the ``.pyx`` files carried, transcribed here
#: because the sources they came from are deleted in this PR. Before that
#: deletion ``test/test_core_dispatch.py::TestCythonMessageParity`` read
#: them out of the tree; nothing in the tree spells them now.
#:
#: * ``scalar_mediator_decay_spectrum.pyx:270`` --
#:   ``assert len(energies.shape) == 1, "Photon energies must be 0 or
#:   1-dimensional."``
#: * ``:249`` -- ``raise ValueError("Partial widths must be a list or
#:   array.")``
#: * ``:251`` -- ``assert len(pws.shape) == 1, "Partial widths must be
#:   1-dimensional."``
RANK_MESSAGE = "Photon energies must be 0 or 1-dimensional."
WIDTHS_MISSING_MESSAGE = "Partial widths must be a list or array."
WIDTHS_RANK_MESSAGE = "Partial widths must be 1-dimensional."

#: Cython's own ``boundscheck(True)`` wording, measured against the
#: shipped 2.1.0 extension rather than read off the generated C.
OUT_OF_BOUNDS_MESSAGE = "Out of bounds on buffer access (axis 0)"


# ===========================================================================
# ---- The independent reference --------------------------------------------
# ===========================================================================


def _grid(mass: float) -> np.ndarray:
    """The rest-frame abscissae, ``numpy.logspace`` as the ``.pyx`` built it."""
    return np.logspace(GRID_LOG10_START, np.log10(mass / 2.0), num=N_INTERP_PTS)


def _tabulate(
    mass: float, kernel: Callable[[np.ndarray, float], np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """``(energies, dnde)`` from a public Phase 04 entry point.

    The ``.pyx`` called the ``cdef`` twin of these through a ``cimport``;
    the public wrapper is the same kernel, and using it keeps the
    reference free of anything this task wrote.
    """
    energies = _grid(mass)
    return energies, np.asarray(kernel(energies, mass / 2.0), dtype=float)


def _interp_with_tail(energy: float, energies: np.ndarray, dnde: np.ndarray) -> float:
    """``np.interp``, with the decay modules' ``1/E`` tail below the grid.

    ``scalar_mediator_decay_spectrum.pyx:55-56`` and
    ``vector_mediator_decay_spectrum.pyx:49-56`` compare against the
    literal ``10**-1`` rather than against ``e_gams[0]``; the two are the
    same double.
    """
    if energy < 10**GRID_LOG10_START:
        return dnde[0] * energies[0] / energy
    return float(np.interp(energy, energies, dnde))


def _fsr_cp_scalar(egam: float, ms: float) -> float:
    """``dnde_fsr_cp_srf`` -- ``scalar_mediator_decay_spectrum.pyx:63-84``."""
    mupi = MASS_PI / ms
    x = 2.0 * egam / ms
    xmax = 1 - 4.0 * mupi**2
    if x < 0.0 or x > xmax:
        return 0.0
    root = math.sqrt(1 - x) * math.sqrt(1 - 4 * mupi**2 - x)
    dynamic = (
        -2 * math.sqrt(1 - x) * math.sqrt(1 - 4 * mupi**2 - x)
        + (-1 + 2 * mupi**2 + x) * math.log((1 - x - root) ** 2 / (-1 + x - root) ** 2)
    ) / x
    coeff = QE**2 / (8.0 * math.sqrt(1 - 4 * mupi**2) * math.pi**2)
    return 2 * (dynamic * coeff) / ms


def _fsr_l_scalar(egam: float, ml: float, ms: float) -> float:
    """``dnde_fsr_l_srf`` -- ``scalar_mediator_decay_spectrum.pyx:90-115``."""
    mul = ml / ms
    x = 2.0 * egam / ms
    xmax = 1 - 4.0 * mul**2
    if x < 0.0 or x > xmax:
        return 0.0
    root = math.sqrt((-1 + x) * (-1 + 4 * mul**2 + x))
    dynamic = (
        4 * (-1 + 4 * mul**2) * math.sqrt(1 - x) * math.sqrt(1 - 4 * mul**2 - x)
        + (2 - 12 * mul**2 + 16 * mul**4 - 2 * x + 8 * mul**2 * x + x**2)
        * math.log((1 - x + root) ** 2 / (-1 + x + root) ** 2)
    ) / x
    coeff = QE**2 / (16.0 * (1 - 4 * mul**2) ** 1.5 * math.pi**2)
    return 2 * (dynamic * coeff) / ms


def _fsr_cp_vector(egam: float, mv: float) -> float:
    """``__dnde_fsr_cp_vrf`` -- ``vector_mediator_decay_spectrum.pyx:61-83``."""
    mupi = MASS_PI / mv
    x = 2.0 * egam / mv
    xmax = 1 - 4.0 * mupi**2
    if x < 0.0 or x > xmax:
        return 0.0
    coeff = QE**2 / (4.0 * (1 - 4 * mupi**2) ** 1.5 * math.pi**2)
    root = math.sqrt(1 - x) * math.sqrt(1 - 4 * mupi**2 - x)
    dynamic = (
        2
        * math.sqrt(1 - 4 * mupi**2 - x)
        * (-1 - 4 * mupi**2 * (-1 + x) + x + x**2)
        / math.sqrt(1 - x)
        + (-1 + 4 * mupi**2)
        * (-1 + 2 * mupi**2 + x)
        * math.log((1 + root - x) ** 2 / (-1 + root + x) ** 2)
    ) / x
    return 2 * (dynamic * coeff) / mv


def _fsr_l_vector(egam: float, ml: float, mv: float) -> float:
    """``__dnde_fsr_l_vrf`` -- ``vector_mediator_decay_spectrum.pyx:86-110``."""
    mul = ml / mv
    x = 2.0 * egam / mv
    xmax = 1 - 4.0 * mul**2
    if x < 0.0 or x > xmax:
        return 0.0
    coeff = -(QE**2) / (8.0 * math.sqrt(1 - 4 * mul**2) * (1 + 2 * mul**2) * math.pi**2)
    root = math.sqrt(1 - x) * math.sqrt(1 - 4 * mul**2 - x)
    dynamic = (
        2
        * math.sqrt(1 - 4 * mul**2 - x)
        * (2 - 4 * mul**2 * (-1 + x) - 2 * x + x**2)
        / math.sqrt(1 - x)
        + (2 - 8 * mul**4 - 4 * mul**2 * x + (-2 + x) * x)
        * math.log((-1 + root + x) ** 2 / (1 + root - x) ** 2)
    ) / x
    return 2 * (dynamic * coeff) / mv


def reference(  # noqa: PLR0913 -- one argument per `.pyx` parameter
    egam: float,
    energy: float,
    mass: float,
    pws: np.ndarray,
    selector: Selector,
    *,
    vector: bool,
) -> float:
    """The deleted ``.pyx`` body, re-derived in NumPy and scipy.

    ``selector`` is a list of mode names for the scalar entry point and a
    single ``mode`` string for the vector ones, matching each source's own
    argument.
    """
    if energy < mass:
        return 0.0

    beta = math.sqrt(1.0 - (mass / energy) ** 2)
    gamma = energy / mass
    eplus = energy * (1.0 + beta) / 2.0
    eminus = energy * (1.0 - beta) / 2.0

    cp_energies, cp_dnde = _tabulate(mass, spectra.dnde_photon_charged_pion)
    if vector:
        mu_energies, mu_dnde = _tabulate(mass, spectra.dnde_photon_muon)

    def integrand(cl: float) -> float:
        jac = 1.0 / (2.0 * gamma * abs(1.0 - beta * cl))
        erf = egam * gamma * (1.0 - beta * cl)
        if vector:
            e_pi0 = 0.5 * (MASS_PI0**2 + mass**2) / mass
            components = {
                "e e g": pws[0] * _fsr_l_vector(erf, MASS_E, mass),
                "mu mu g": pws[1] * _fsr_l_vector(erf, MASS_MU, mass),
                "pi pi g": pws[3] * _fsr_cp_vector(erf, mass),
                "pi pi": 2.0 * pws[3] * _interp_with_tail(erf, cp_energies, cp_dnde),
                "pi0 g": pws[2] * spectra.dnde_photon_neutral_pion(erf, e_pi0),
                "mu mu": 2.0 * pws[1] * _interp_with_tail(erf, mu_energies, mu_dnde),
            }
            if selector == "total":
                return jac * sum(components.values())
            return jac * components.get(selector, 0.0)

        result = 0.0
        if "e e g" in selector:
            result += pws[0] * _fsr_l_scalar(erf, MASS_E, mass)
        if "pi pi g" in selector:
            result += pws[3] * _fsr_cp_scalar(erf, mass)
        if "pi pi" in selector:
            result += 2.0 * pws[3] * _interp_with_tail(erf, cp_energies, cp_dnde)
        if "pi0 pi0" in selector:
            result += 2.0 * pws[2] * spectra.dnde_photon_neutral_pion(erf, mass / 2.0)
        if "mu mu g" in selector:
            result += pws[1] * _fsr_l_scalar(erf, MASS_MU, mass)
        if "mu mu" in selector:
            result += 2.0 * pws[1] * spectra.dnde_photon_muon(erf, mass / 2.0)
        return jac * result

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = quad(integrand, -1.0, 1.0, **QUAD_KWARGS)[0]

    in_window = eminus <= egam <= eplus
    if vector:
        if selector in ("pi0 g", "total") and in_window:
            result += pws[2] / (energy * beta)
    elif "g g" in selector and in_window:
        result += pws[4] / (energy * beta)
    return result


def scalar_call(
    egam: object,
    energy: float,
    mass: float,
    pws: object = None,
    modes: object = None,
) -> object:
    """``scalar_mediator_decay_spectrum`` with this module's defaults."""
    pws = SCALAR_PWS if pws is None else pws
    if modes is None:
        return scalar_spectrum(egam, energy, mass, pws)
    return scalar_spectrum(egam, energy, mass, pws, modes)


def vector_call(
    egam: object,
    energy: float,
    mass: float,
    pws: object = None,
    mode: str | None = "total",
) -> float:
    """``dnde_decay_v_pt`` with this module's defaults."""
    pws = VECTOR_PWS if pws is None else pws
    return dnde_decay_v_pt(egam, energy, mass, pws, mode)


# ===========================================================================
# ---- Part 1: the dispatch contract ----------------------------------------
# ===========================================================================


class TestDispatchWiring:
    """Each entry point reaches ``crate::dispatch`` with the right wording.

    ``scalar_mediator_decay_spectrum`` dispatches its first argument the
    way every ``hazma/spectra/**`` entry point did — scalar or 1-D array
    in, the same out — so it goes through ``map_unary_try``. The two
    vector entry points do not: the ``.pyx`` declared
    ``np.ndarray[double] eng_gam`` on one and ``double eng_gam`` on the
    other, so ``dnde_decay_v`` takes ``require_vector`` and
    ``dnde_decay_v_pt`` takes PyO3's own scalar extraction.
    """

    def test_a_float_returns_a_float(self) -> None:
        assert type(scalar_call(30.0, 600.0, 550.0)) is float
        assert type(vector_call(30.0, 600.0, 550.0)) is float

    def test_a_grid_returns_a_fresh_float64_array(self) -> None:
        energies = np.array([30.0, 40.0, 50.0])
        for got in (
            scalar_call(energies, 600.0, 550.0),
            dnde_decay_v(energies, 600.0, 550.0, VECTOR_PWS, "total"),
        ):
            assert isinstance(got, np.ndarray)
            assert got.dtype == np.float64
            assert got.shape == energies.shape
            assert got is not energies

    def test_a_sequence_is_accepted(self) -> None:
        # The widening `crate::dispatch` declares for every entry point:
        # the scalar `.pyx` already accepted a list (it called `np.array`),
        # and `dnde_decay_v` did not.
        assert np.asarray(scalar_call([30.0, 40.0], 600.0, 550.0)).shape == (2,)
        assert dnde_decay_v([30.0, 40.0], 600.0, 550.0, VECTOR_PWS, "total").shape == (
            2,
        )

    def test_a_zero_dimensional_array_takes_the_scalar_path(self) -> None:
        assert type(scalar_call(np.array(30.0), 600.0, 550.0)) is float

    def test_a_rank_error_names_the_quantity(self) -> None:
        with pytest.raises(ValueError, match=re.escape(RANK_MESSAGE)):
            scalar_call(np.ones((2, 2)), 600.0, 550.0)

    def test_a_dtype_error_names_the_dtype(self) -> None:
        with pytest.raises(ValueError, match="float64 array; got dtype float32"):
            scalar_call(np.ones(3, dtype=np.float32), 600.0, 550.0)
        with pytest.raises(ValueError, match="float64 array; got dtype float32"):
            dnde_decay_v(
                np.ones(3, dtype=np.float32), 600.0, 550.0, VECTOR_PWS, "total"
            )

    def test_a_non_number_is_a_type_error(self) -> None:
        with pytest.raises(TypeError):
            scalar_call(object(), 600.0, 550.0)
        with pytest.raises(TypeError):
            vector_call(object(), 600.0, 550.0)

    @pytest.mark.parametrize(
        ("widths", "message"),
        [
            (1.0, WIDTHS_MISSING_MESSAGE),
            (np.zeros((2, 2)), WIDTHS_RANK_MESSAGE),
        ],
    )
    def test_the_partial_width_messages_are_the_pyx_s(
        self, widths: object, message: str
    ) -> None:
        # Both wordings were that call site's own text, and both are
        # reproduced verbatim -- the `raise ValueError` keeps its type and
        # the `assert` is promoted to one (rules.md rule 9).
        with pytest.raises(ValueError, match=re.escape(message)):
            scalar_call(30.0, 600.0, 550.0, pws=widths)
        with pytest.raises(ValueError, match=re.escape(message)):
            vector_call(30.0, 600.0, 550.0, pws=widths)

    def test_a_scalar_energy_is_refused_by_the_array_entry_point(self) -> None:
        # A declared divergence. The `.pyx` raised `TypeError` here
        # ("Argument 'eng_gam' has incorrect type"); `require_vector`
        # raises `ValueError` with the quantity's own wording. No working
        # call reaches it -- the Python wrapper picks `_pt` for scalars.
        with pytest.raises(
            ValueError, match=re.escape("Photon energies must be a list or array.")
        ):
            dnde_decay_v(30.0, 600.0, 550.0, VECTOR_PWS, "total")

    def test_the_signatures_are_introspectable_and_accept_keywords(self) -> None:
        # The `.pyx` entry points were `def`s, so every argument was
        # accepted by keyword; a positional-only claim here would narrow
        # the public API.
        assert (
            str(inspect.signature(dnde_decay_v_pt)) == "(eng_gam, eng_v, mv, pws, mode)"
        )
        assert vector_call(30.0, 600.0, 550.0) == dnde_decay_v_pt(
            eng_gam=30.0, eng_v=600.0, mv=550.0, pws=VECTOR_PWS, mode="total"
        )
        assert scalar_call(30.0, 600.0, 550.0) == scalar_spectrum(
            photon_energies=30.0,
            sm_energy=600.0,
            sm_mass=550.0,
            partial_widths=SCALAR_PWS,
        )


# ===========================================================================
# ---- Part 2: the independent reference ------------------------------------
# ===========================================================================


class TestAgainstAnIndependentReference:
    """The port reproduces :func:`reference` inside :data:`REFERENCE_RTOL`.

    The reference is the deleted ``.pyx`` bodies re-transcribed from source
    into NumPy and ``scipy.integrate.quad``. It shares no code with the
    port except the Phase 04 photon kernels the ``.pyx`` itself cimported,
    which are what the tables are made of on both sides.
    """

    @pytest.mark.parametrize(("mass", "energy"), CONFIGS)
    @pytest.mark.parametrize("mode", SCALAR_MODES)
    def test_the_scalar_spectrum_matches_channel_by_channel(
        self, mass: float, energy: float, mode: str
    ) -> None:
        for egam in (0.05, 1.0, 30.0, 300.0, 0.9 * energy):
            want = reference(egam, energy, mass, SCALAR_PWS, [mode], vector=False)
            got = scalar_call(egam, energy, mass, modes=[mode])
            assert got == pytest.approx(
                want, rel=REFERENCE_RTOL, abs=0.0
            ), f"{mode} at egam={egam}"

    @pytest.mark.parametrize(("mass", "energy"), CONFIGS)
    def test_the_scalar_spectrum_matches_with_every_channel_open(
        self, mass: float, energy: float
    ) -> None:
        egams = np.logspace(-2, np.log10(0.9 * energy), 17)
        want = np.array(
            [
                reference(e, energy, mass, SCALAR_PWS, SCALAR_MODES, vector=False)
                for e in egams
            ]
        )
        got = np.asarray(scalar_call(egams, energy, mass))
        np.testing.assert_allclose(got, want, rtol=REFERENCE_RTOL, atol=0.0)

    @pytest.mark.parametrize(("mass", "energy"), CONFIGS)
    @pytest.mark.parametrize("mode", VECTOR_MODES)
    def test_the_vector_spectrum_matches_channel_by_channel(
        self, mass: float, energy: float, mode: str
    ) -> None:
        egams = np.logspace(-2, np.log10(0.9 * energy), 11)
        want = np.array(
            [reference(e, energy, mass, VECTOR_PWS, mode, vector=True) for e in egams]
        )
        got = np.asarray(dnde_decay_v(egams, energy, mass, VECTOR_PWS, mode))
        np.testing.assert_allclose(got, want, rtol=REFERENCE_RTOL, atol=0.0)

    @pytest.mark.parametrize("mode", VECTOR_MODES)
    def test_the_two_vector_entry_points_agree_bit_for_bit(self, mode: str) -> None:
        # They are the same kernel behind two dispatch shapes, so this is
        # bit-equality and not a tolerance question.
        egams = np.logspace(-2, 3, 23)
        array = np.asarray(dnde_decay_v(egams, 600.0, 550.0, VECTOR_PWS, mode))
        pointwise = np.array([vector_call(e, 600.0, 550.0, mode=mode) for e in egams])
        assert array.tobytes() == pointwise.tobytes()


# ===========================================================================
# ---- Part 3: physics ------------------------------------------------------
# ===========================================================================


class TestPhysics:
    """Statements that owe nothing to the implementation being replaced."""

    @pytest.mark.parametrize("energy", [0.0, 100.0, 549.999])
    def test_a_mediator_below_its_own_mass_contributes_nothing(
        self, energy: float
    ) -> None:
        assert scalar_call(30.0, energy, 550.0) == 0.0
        assert vector_call(30.0, energy, 550.0) == 0.0

    def test_the_two_photon_line_carries_its_own_photon_count(self) -> None:
        # `s -> gamma gamma` is a flat box of height `pw/(E_s beta)` over
        # `[E_-, E_+]`, whose width is `E_s beta`. So the line integrates
        # to exactly `pw` photons -- one per decay, weighted by the
        # branching fraction, independent of the boost.
        energy, mass = 1500.0, 550.0
        beta = math.sqrt(1.0 - (mass / energy) ** 2)
        eminus, eplus = energy * (1 - beta) / 2, energy * (1 + beta) / 2
        egams = np.linspace(eminus, eplus, 4001)
        heights = np.asarray(scalar_call(egams, energy, mass, modes=["g g"]))
        assert np.ptp(heights) == 0.0
        assert np.trapezoid(heights, egams) == pytest.approx(SCALAR_PWS[4], rel=1e-3)

    def test_the_pi0_gamma_line_carries_its_own_photon_count(self) -> None:
        # The vector's line is the photon of `V -> pi0 gamma`, one per
        # decay. It is a flat floor of height `pw/(E_v beta)` across a
        # window of width `E_v beta`, so it integrates to exactly `pw` --
        # asserted as that arithmetic rather than by quadrature, because
        # the mode also carries the `pi0` *continuum* on top of the floor
        # and there is no mode that isolates the continuum.
        energy, mass = 1500.0, 550.0
        beta = math.sqrt(1.0 - (mass / energy) ** 2)
        eminus, eplus = energy * (1 - beta) / 2, energy * (1 + beta) / 2
        step = VECTOR_PWS[2] / (energy * beta)
        assert step * (eplus - eminus) == pytest.approx(VECTOR_PWS[2], rel=1e-14)

        egams = np.linspace(eminus, eplus, 4001)
        with_line = np.asarray(dnde_decay_v(egams, energy, mass, VECTOR_PWS, "pi0 g"))
        # The floor is a floor: never below it inside the window. It is
        # reached *exactly* at the top, where the boosted `pi0` continuum
        # has already ended -- which is why this is `>=` and not `>`.
        assert np.all(with_line >= step)
        assert with_line[-1] == step
        assert with_line[0] > step
        assert np.trapezoid(with_line - step, egams) > 0.0

        # And there is no floor outside the window.
        assert dnde_decay_v_pt(1.001 * eplus, energy, mass, VECTOR_PWS, "pi0 g") < step

    def test_the_spectrum_vanishes_above_the_endpoint(self) -> None:
        # Every channel's rest-frame support ends at `m/2`, so the boosted
        # support ends at the maximally forward-boosted `m/2`.
        energy, mass = 1500.0, 550.0
        gamma = energy / mass
        beta = math.sqrt(1.0 - 1.0 / gamma**2)
        endpoint = (mass / 2.0) * gamma * (1.0 + beta)
        assert scalar_call(10.0 * endpoint, energy, mass) == 0.0
        assert vector_call(10.0 * endpoint, energy, mass) == 0.0

    @pytest.mark.parametrize(("mass", "energy"), CONFIGS)
    def test_the_scalar_channels_are_additive(self, mass: float, energy: float) -> None:
        # Each channel enters the boost integral linearly, so asking for
        # all seven must give the same answer as summing seven
        # single-channel calls -- to the integrator's own tolerance,
        # since each call subdivides independently.
        for egam in (1.0, 30.0, 200.0):
            total = scalar_call(egam, energy, mass)
            summed = sum(
                scalar_call(egam, energy, mass, modes=[mode]) for mode in SCALAR_MODES
            )
            assert total == pytest.approx(summed, rel=ADDITIVITY_RTOL, abs=0.0)

    @pytest.mark.parametrize(("mass", "energy"), CONFIGS)
    def test_the_vector_channels_are_additive(self, mass: float, energy: float) -> None:
        for egam in (1.0, 30.0, 200.0):
            total = vector_call(egam, energy, mass)
            summed = sum(
                vector_call(egam, energy, mass, mode=mode)
                for mode in VECTOR_MODES
                if mode != "total"
            )
            assert total == pytest.approx(summed, rel=ADDITIVITY_RTOL, abs=0.0)

    def test_the_spectrum_is_positive_where_it_is_supported(self) -> None:
        egams = np.logspace(-2, 2.5, 41)
        assert np.all(np.asarray(scalar_call(egams, 600.0, 550.0)) > 0.0)
        assert np.all(
            np.asarray(dnde_decay_v(egams, 600.0, 550.0, VECTOR_PWS, "total")) > 0.0
        )

    def test_zero_partial_widths_give_a_zero_spectrum(self) -> None:
        # Not a tolerance question: every channel is multiplied by its
        # width, so the integrand is exactly zero and QUADPACK sums exact
        # zeros.
        egams = np.logspace(-2, 2.5, 41)
        assert np.all(
            np.asarray(scalar_call(egams, 600.0, 550.0, pws=np.zeros(5))) == 0.0
        )
        assert np.all(
            np.asarray(dnde_decay_v(egams, 600.0, 550.0, np.zeros(4), "total")) == 0.0
        )

    def test_a_scalar_argument_and_a_one_element_grid_agree_bit_for_bit(self) -> None:
        for egam in (0.05, 30.0, 300.0):
            grid = np.asarray(scalar_call(np.array([egam]), 600.0, 550.0))
            assert grid[0] == scalar_call(egam, 600.0, 550.0)

    def test_an_empty_grid_returns_an_empty_grid(self) -> None:
        empty = np.array([], dtype=float)
        assert np.asarray(scalar_call(empty, 600.0, 550.0)).shape == (0,)
        assert dnde_decay_v(empty, 600.0, 550.0, VECTOR_PWS, "total").shape == (0,)


# ===========================================================================
# ---- Part 4: error paths and reproduced quirks ----------------------------
# ===========================================================================


class TestErrorPaths:
    """Every documented failure mode, plus the two quirks the port keeps."""

    @pytest.mark.parametrize("length", [0, 1, 2, 3])
    def test_a_short_partial_width_buffer_raises_index_error(self, length: int) -> None:
        # `boundscheck(True)` on both integrands means the first four
        # entries are read at every quadrature node, whatever the mode.
        pws = np.zeros(length)
        with pytest.raises(IndexError, match=re.escape(OUT_OF_BOUNDS_MESSAGE)):
            scalar_call(30.0, 600.0, 550.0, pws=pws)
        with pytest.raises(IndexError, match=re.escape(OUT_OF_BOUNDS_MESSAGE)):
            vector_call(30.0, 600.0, 550.0, pws=pws)

    def test_the_fifth_scalar_width_is_read_only_inside_the_line_window(self) -> None:
        # Measured against the shipped 2.1.0 extension: a four-element
        # `pws` returns for a photon outside the `g g` window and raises
        # inside it. A port that validated the length up front would have
        # broken the working half.
        four = SCALAR_PWS[:4]
        assert scalar_call(30.0, 600.0, 550.0, pws=four) == pytest.approx(
            scalar_call(
                30.0, 600.0, 550.0, pws=four, modes=SCALAR_MODES[:4] + SCALAR_MODES[4:]
            )
        )
        with pytest.raises(IndexError):
            scalar_call(300.0, 600.0, 550.0, pws=four)

    def test_a_mediator_below_its_mass_does_not_read_the_widths(self) -> None:
        # Both `.pyx` return before touching the buffer.
        assert scalar_call(30.0, 100.0, 550.0, pws=np.array([])) == 0.0
        assert vector_call(30.0, 100.0, 550.0, pws=np.array([])) == 0.0

    @pytest.mark.parametrize("mode", ["zzz", "", "PI PI", None])
    def test_an_unrecognised_vector_mode_returns_zero(self, mode: object) -> None:
        # Reproduced, not repaired. Every `cdef double` integrand ends in
        # an `if`-chain with no `else`, and a C function that falls off its
        # end returns zero -- so a typo'd mode integrates a zero integrand
        # and the entry point returns `0.0` rather than raising. Filed as
        # `docs/followups/todo/mediator-spectra-accept-unknown-mode-strings.md`;
        # this test is what changes when that lands.
        assert vector_call(30.0, 600.0, 550.0, mode=mode) == 0.0
        assert np.all(
            dnde_decay_v(np.array([30.0, 40.0]), 600.0, 550.0, VECTOR_PWS, mode) == 0.0
        )

    def test_an_unrecognised_vector_mode_still_reads_the_widths(self) -> None:
        # The buffer reads precede the mode chain in the integrand, so the
        # `0.0` above is not a short circuit.
        with pytest.raises(IndexError):
            vector_call(30.0, 600.0, 550.0, pws=np.zeros(3), mode="zzz")

    def test_an_unrecognised_scalar_mode_sets_no_bit(self) -> None:
        # Same defect through the other route: the fold tests `"pi pi" in
        # modes` seven times and an unknown entry simply sets nothing, so
        # `modes=["bogus"]` is `modes=[]` is `0.0`.
        assert scalar_call(30.0, 600.0, 550.0, modes=["bogus"]) == 0.0
        assert scalar_call(30.0, 600.0, 550.0, modes=[]) == 0.0

    def test_the_scalar_modes_argument_uses_python_membership(self) -> None:
        # `"pi pi" in modes` accepts anything with a `__contains__`, and a
        # `str` is the live example: `modes="pi pi g"` sets the `"pi pi"`
        # *and* `"pi pi g"` bits by substring. Reproduced because the port
        # asks Python rather than comparing lists.
        by_string = scalar_call(30.0, 600.0, 550.0, modes="pi pi g")
        by_list = scalar_call(30.0, 600.0, 550.0, modes=["pi pi", "pi pi g"])
        assert by_string == by_list
        assert scalar_call(30.0, 600.0, 550.0, modes=("mu mu",)) == scalar_call(
            30.0, 600.0, 550.0, modes=["mu mu"]
        )
        assert scalar_call(30.0, 600.0, 550.0, modes={"mu mu"}) == scalar_call(
            30.0, 600.0, 550.0, modes=["mu mu"]
        )

    def test_a_repeated_mode_is_not_counted_twice(self) -> None:
        assert scalar_call(30.0, 600.0, 550.0, modes=["mu mu", "mu mu"]) == scalar_call(
            30.0, 600.0, 550.0, modes=["mu mu"]
        )

    def test_a_modes_object_whose_membership_raises_propagates(self) -> None:
        class Hostile:
            def __contains__(self, item: object) -> bool:
                raise KeyError(item)

        with pytest.raises(KeyError, match="pi pi"):
            scalar_call(30.0, 600.0, 550.0, modes=Hostile())

    def test_the_complex_coefficient_raises_at_the_degenerate_mass(self) -> None:
        # `__Pyx_SoftComplexToDouble` raised `TypeError` where the `**1.5`
        # coefficient's denominator vanishes, and the port keeps the type.
        # The scalar's is the *lepton* coefficient and the vector's the
        # *charged pion*'s, because the two `.pyx` put the 1.5 exponent on
        # different factors -- and only `egam = 0` gets past the
        # `x > xmax` guard to see it.
        ms = 2.0 * MASS_MU
        with pytest.raises(TypeError, match="complex at this mediator mass"):
            scalar_call(0.0, ms, ms, modes=["mu mu g"])
        assert scalar_call(1.0, ms, ms, modes=["mu mu g"]) == 0.0

        mv = 2.0 * MASS_PI
        with pytest.raises(TypeError, match="complex at this mediator mass"):
            vector_call(0.0, mv, mv, mode="pi pi g")
        assert vector_call(1.0, mv, mv, mode="pi pi g") == 0.0

    def test_a_single_vector_channel_still_pays_for_the_pion_coefficient(self) -> None:
        # The `.pyx` computes all six components before selecting one, so
        # a mode that names none of the charged-pion FSR still raises where
        # that coefficient does. A lazy port would return a number.
        mv = 2.0 * MASS_PI
        with pytest.raises(TypeError, match="complex at this mediator mass"):
            vector_call(0.0, mv, mv, mode="e e g")

    def test_the_scalar_integrand_is_lazy_where_the_pyx_is(self) -> None:
        # The mirror of the test above, and the reason the two ports differ
        # in structure: the scalar `.pyx` guards each channel with a
        # bitflag `if`, so a mode that excludes the lepton FSR never
        # evaluates it and cannot raise.
        ms = 2.0 * MASS_MU
        # `nan`, not a number: at `E_gamma = 0` the charged pion's `1/E`
        # tail below the grid divides by zero and the boost integral of an
        # infinity is undefined. What matters here is that it does not
        # *raise* -- the lepton coefficient that would have is never
        # evaluated -- and :func:`reference` agrees, because the `.pyx`
        # took the same tail.
        assert math.isnan(scalar_call(0.0, ms, ms, modes=["pi pi"]))
        assert math.isnan(reference(0.0, ms, ms, SCALAR_PWS, ["pi pi"], vector=False))

    def test_a_nan_energy_propagates(self) -> None:
        assert math.isnan(scalar_call(float("nan"), 600.0, 550.0))
        assert math.isnan(vector_call(float("nan"), 600.0, 550.0))
