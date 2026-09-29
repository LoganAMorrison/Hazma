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
   below), compared at a stated budget. The scalar FSR transcriptions
   carry :data:`PAIR_NORMALIZATION`, a deliberate departure from the
   ``.pyx`` that :class:`TestPhysics` justifies; the clipped ``cos theta``
   window is the other, and :class:`TestTheBoostedTail` checks it against
   a quadrature in the energy variable (:func:`energy_reference`) and
   against the boost's photon-energy identity.
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
from scipy.integrate import IntegrationWarning, quad

from hazma import parameters, spectra
from hazma._core import scalar_mediator as core_scalar
from hazma._core import vector_mediator as core_vector
from hazma.scalar_mediator import ScalarMediator
from hazma.vector_mediator import VectorMediator

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

#: The factor ``scalar_mediator_decay_spectrum.pyx``'s two rest-frame FSR
#: coefficients were missing. The ``.pyx`` returned half the pair-summed
#: spectrum in every FSR channel; ``rust/src/kernels/scalar_decay_photon.rs``
#: restores the factor as its ``PAIR_NORMALIZATION`` and the transcriptions
#: below apply the same one, so the reference describes the repaired kernel
#: rather than the deleted source. :class:`TestPhysics` pins the corrected
#: size against the annihilation-side matrix elements and the collinear
#: limit; ``docs/followups/done/scalar-decay-fsr-half-normalized.md`` has
#: the measurement.
PAIR_NORMALIZATION = 2.0

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
#: shape, and the worst difference measured here is 2.4e-11, at
#: ``"e e g"``, ``mass = 550``, ``energy = 1500``, ``egam = 414`` MeV.
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
    """``dnde_fsr_cp_srf`` -- ``scalar_mediator_decay_spectrum.pyx:63-84``, times :data:`PAIR_NORMALIZATION`."""
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
    return PAIR_NORMALIZATION * (2 * (dynamic * coeff) / ms)


def _fsr_l_scalar(egam: float, ml: float, ms: float) -> float:
    """``dnde_fsr_l_srf`` -- ``scalar_mediator_decay_spectrum.pyx:90-115``, times :data:`PAIR_NORMALIZATION`."""
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
    return PAIR_NORMALIZATION * (2 * (dynamic * coeff) / ms)


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


def _table_edge(energies: np.ndarray, dnde: np.ndarray) -> float:
    """The energy, MeV, at and above which a rest-frame table interpolates to zero.

    The first abscissa after the last non-zero entry: ``np.interp`` carries
    that entry down to zero across the next cell. Infinite if the last entry
    is non-zero, because ``np.interp`` clamps to it above the grid, and
    minus infinity for a table that is zero throughout.
    """
    nonzero = np.flatnonzero(dnde)
    if nonzero.size == 0:
        return -math.inf
    if nonzero[-1] + 1 == energies.size:
        return math.inf
    return float(energies[nonzero[-1] + 1])


def _neutral_pion_box_top(epi: float) -> float:
    """The top of the ``pi0 -> gamma gamma`` box at pion energy ``epi``, MeV.

    ``E_pi (1 + beta) / 2``, with ``beta`` rounded to ``float32`` as the
    public kernel declares it; minus infinity below the pion mass.
    """
    if epi < parameters.neutral_pion_mass:
        return -math.inf
    ratio = parameters.neutral_pion_mass / epi
    beta = float(np.float32(math.sqrt(1.0 - ratio * ratio)))
    return epi * (1.0 + beta) / 2.0


def _muon_photon_endpoint(emu: float) -> float:
    """The highest photon energy ``spectra.dnde_photon_muon`` reaches, MeV.

    The radiative muon decay's rest-frame edge ``(1 - r) m_mu / 2``, with
    ``r = (m_e / m_mu)**2``, boosted fully forward; minus infinity below the
    muon mass. The public kernel takes PDG masses, not the legacy ones above.
    """
    mmu = parameters.muon_mass
    if emu < mmu:
        return -math.inf
    r = (parameters.electron_mass / mmu) ** 2
    beta = math.sqrt(1.0 - (mmu / emu) ** 2)
    return (1.0 - r) * emu * (1.0 + beta) / 2.0


def _channel_endpoints(
    mass: float, *, vector: bool, tables: dict[str, float]
) -> dict[str, float]:
    """Each channel's rest-frame photon endpoint, MeV, keyed by its mode.

    ``tables`` holds the two interpolated tables' edges, keyed ``"pi pi"``
    and ``"mu mu"``. The FSR channels end at ``x = 1 - 4 mu**2``; the
    neutral pion at the top of its box; the scalar's muon, which is not
    tabulated, at its forward-cone edge. Lines ride outside the integral
    and have no endpoint here.
    """

    def fsr(radiator: float) -> float:
        return (1.0 - 4.0 * (radiator / mass) ** 2) * mass / 2.0

    if vector:
        e_pi0 = 0.5 * (MASS_PI0**2 + mass**2) / mass
        return {
            "e e g": fsr(MASS_E),
            "mu mu g": fsr(MASS_MU),
            "pi pi g": fsr(MASS_PI),
            "pi pi": tables["pi pi"],
            "pi0 g": _neutral_pion_box_top(e_pi0),
            "mu mu": tables["mu mu"],
        }
    return {
        "e e g": fsr(MASS_E),
        "pi pi g": fsr(MASS_PI),
        "pi pi": tables["pi pi"],
        "pi0 pi0": _neutral_pion_box_top(mass / 2.0),
        "mu mu g": fsr(MASS_MU),
        "mu mu": _muon_photon_endpoint(mass / 2.0),
    }


def rest_frame_spectrum(
    mass: float, pws: np.ndarray, selector: Selector, *, vector: bool
) -> tuple[Callable[[float], float], float, list[float]]:
    """The boost integrand's rest-frame spectrum, its endpoint and its kinks.

    Returns ``(spectrum, endpoint, kinks)``: ``spectrum(E')`` is ``dN/dE'``
    in MeV^-1 summed over the channels ``selector`` opens, lines excluded;
    ``endpoint`` in MeV is the widest of those channels' endpoints, above
    which ``spectrum`` is zero; ``kinks`` are every rest-frame energy, MeV,
    where some channel starts, stops or changes form -- break points for a
    quadrature over ``E'``.
    """
    cp_energies, cp_dnde = _tabulate(mass, spectra.dnde_photon_charged_pion)
    mu_energies, mu_dnde = _tabulate(mass, spectra.dnde_photon_muon)
    tables = {
        "pi pi": _table_edge(cp_energies, cp_dnde),
        "mu mu": _table_edge(mu_energies, mu_dnde),
    }
    edges = _channel_endpoints(mass, vector=vector, tables=tables)
    if vector:
        selected = list(edges) if selector == "total" else [selector]
    else:
        selected = list(selector)
    endpoint = max(
        (edges[mode] for mode in selected if mode in edges), default=-math.inf
    )
    e_pi0 = 0.5 * (MASS_PI0**2 + mass**2) / mass if vector else mass / 2.0
    box_bottom = e_pi0 - (_neutral_pion_box_top(e_pi0) - e_pi0)
    kinks = [10**GRID_LOG10_START, box_bottom, *edges.values()]

    def spectrum(erf: float) -> float:
        if vector:
            components = {
                "e e g": pws[0] * _fsr_l_vector(erf, MASS_E, mass),
                "mu mu g": pws[1] * _fsr_l_vector(erf, MASS_MU, mass),
                "pi pi g": pws[3] * _fsr_cp_vector(erf, mass),
                "pi pi": 2.0 * pws[3] * _interp_with_tail(erf, cp_energies, cp_dnde),
                "pi0 g": pws[2] * spectra.dnde_photon_neutral_pion(erf, e_pi0),
                "mu mu": 2.0 * pws[1] * _interp_with_tail(erf, mu_energies, mu_dnde),
            }
            if selector == "total":
                return sum(components.values())
            return components.get(selector, 0.0)

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
        return result

    return spectrum, endpoint, kinks


def _line(  # noqa: PLR0913 -- one argument per `.pyx` parameter
    egam: float,
    energy: float,
    mass: float,
    pws: np.ndarray,
    selector: Selector,
    *,
    vector: bool,
) -> float:
    """The monochromatic line both sources add outside the integral, MeV^-1.

    The vector's ``pi0 gamma`` line for ``"pi0 g"`` and ``"total"``, the
    scalar's ``gamma gamma`` line when ``"g g"`` is selected: a box of
    height ``pw / (E beta)`` over ``E (1 -+ beta) / 2``. At rest the box
    is the single point ``m / 2`` and its height is IEEE ``pw / 0``, as the
    kernels divide it.
    """
    beta = math.sqrt(1.0 - (mass / energy) ** 2)
    if not energy * (1.0 - beta) / 2.0 <= egam <= energy * (1.0 + beta) / 2.0:
        return 0.0
    if vector:
        if selector not in ("pi0 g", "total"):
            return 0.0
        width = pws[2]
    elif "g g" in selector:
        width = pws[4]
    else:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.divide(width, energy * beta))


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

    One departure from the source, the one the kernel makes: the ``cos
    theta`` integral starts where the rest-frame energy falls to the
    selected channels' endpoint rather than at ``-1``. The ``.pyx``
    integrated the whole range, and at a large boost its quadrature never
    sampled the support near ``cos theta = 1``;
    :class:`TestTheBoostedTail` checks the clipped integral against one in
    the energy variable, which has no such window to miss.
    """
    if energy < mass:
        return 0.0

    beta = math.sqrt(1.0 - (mass / energy) ** 2)
    gamma = energy / mass
    spectrum, endpoint, _ = rest_frame_spectrum(mass, pws, selector, vector=vector)
    lower = -1.0
    if beta > 0.0 and egam > 0.0:
        lower = max(-1.0, (1.0 - endpoint / (gamma * egam)) / beta)

    def integrand(cl: float) -> float:
        jac = 1.0 / (2.0 * gamma * abs(1.0 - beta * cl))
        return jac * spectrum(egam * gamma * (1.0 - beta * cl))

    result = 0.0
    if lower < 1.0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = quad(integrand, lower, 1.0, **QUAD_KWARGS)[0]
    return result + _line(egam, energy, mass, pws, selector, vector=vector)


def energy_reference(  # noqa: PLR0913 -- one argument per `.pyx` parameter
    egam: float,
    energy: float,
    mass: float,
    pws: np.ndarray,
    selector: Selector,
    *,
    vector: bool,
) -> float:
    """The boosted spectrum integrated over rest-frame energy, MeV^-1.

    For an isotropic massless daughter, ``E' = gamma E (1 - beta cos
    theta)`` turns the ``cos theta`` integral into

        dN/dE = 1 / (2 beta gamma) int dE' f(E') / E',
                gamma E (1 - beta) <= E' <= gamma E (1 + beta),

    which has no angular window to miss: the support is the part of this
    range below the endpoint, and the channels' kinks are break points.
    Integrated over ``u = ln E'``, in which ``f(E') dE' / E'`` is
    ``f(e^u) du``, at ``epsrel = 1e-8`` -- three decades under the kernel's
    own, and the tightest scipy reaches without reporting roundoff. Shares only the rest-frame spectrum with :func:`reference`.
    """
    beta = math.sqrt(1.0 - (mass / energy) ** 2)
    gamma = energy / mass
    spectrum, _, kinks = rest_frame_spectrum(mass, pws, selector, vector=vector)
    lower = math.log(gamma * egam * (1.0 - beta))
    upper = math.log(gamma * egam * (1.0 + beta))
    points = sorted(
        {math.log(k) for k in kinks if k > 0.0 and lower < math.log(k) < upper}
    )
    # The tabulated channels are piecewise linear over 500 knots, and scipy
    # reports each unresolved kink as roundoff at this tolerance. The value
    # is what the tests check, against the kernel and the energy identity.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        value = quad(
            lambda u: spectrum(math.exp(u)),
            lower,
            upper,
            points=points or None,
            epsabs=0.0,
            epsrel=1e-8,
            limit=500,
        )[0]
    return value / (2.0 * beta * gamma) + _line(
        egam, energy, mass, pws, selector, vector=vector
    )


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


#: The scalar selectors :class:`TestTheBoostedTail` samples: each channel
#: inside the integral alone, then the entry point's default list.
TAIL_SCALAR_SELECTORS = [
    ["pi pi"],
    ["mu mu"],
    ["pi0 pi0"],
    ["e e g"],
    ["pi pi g"],
    ["mu mu g"],
    SCALAR_MODES,
]

#: Mediator boosts ``gamma = E / m`` for :class:`TestTheBoostedTail`. At
#: ``gamma = 2`` 2.3.0 already lost the top of the electron FSR; by 30 it
#: returned zero over most of every channel's range.
TAIL_BOOSTS = [2.0, 10.0, 30.0]

#: The budget against :func:`energy_reference`. The kernel converges to
#: ``epsrel = 1e-5`` or ``epsabs = 1e-10`` MeV^-1, whichever is looser, so in
#: the far tail, where ``dN/dE`` falls to 1e-12, the absolute bound governs
#: and the relative error grows: measured up to 7.6e-4 at 0.889 of the lab
#: endpoint of ``"pi pi"`` on a 60-point sweep, and 6.9e-5 worst on the
#: 13-point grid below, at 550 MeV. The defect this guards is a relative
#: error of exactly 1.
TAIL_RTOL = 1e-3

#: The budget on ``int E dN/dE dE = gamma int E' f(E') dE'``. Measured
#: within 8.5e-6 of one at every boost below, which is the trapezoid rule's
#: error on 4,001 log-spaced energies; 2.3.0 carried 0.081 of the scalar's
#: photon energy at ``gamma = 30``.
ENERGY_IDENTITY_TOL = 3e-5


def _lab_endpoint(
    mass: float, energy: float, pws: np.ndarray, selector: Selector, *, vector: bool
) -> float:
    """The highest lab photon energy the boost integral reaches, MeV.

    The rest-frame endpoint boosted fully forward, ``gamma E'_max (1 + beta)``.
    """
    _, endpoint, _ = rest_frame_spectrum(mass, pws, selector, vector=vector)
    beta = math.sqrt(1.0 - (mass / energy) ** 2)
    return energy / mass * endpoint * (1.0 + beta)


class TestTheBoostedTail:
    """The boost integral keeps its support when the mediator is fast.

    ``E' = gamma E (1 - beta cos theta)`` puts the support of a channel whose
    rest-frame spectrum ends at ``E'_max`` at ``cos theta >= (1 - E'_max /
    (gamma E)) / beta``: near the lab endpoint, a cone ``1 / (2 gamma**2)``
    wide. Over the whole of ``[-1, 1]`` the first 21-point rule sampled
    only zeros there and QUADPACK accepted ``0.0``, so 2.3.0 lost the top
    of every channel's spectrum once the mediator moved. The kernels now
    start the integral at the cone's edge; these tests check the result
    against a quadrature that never had the window to miss.
    """

    @pytest.mark.parametrize("gamma", TAIL_BOOSTS)
    @pytest.mark.parametrize("selector", TAIL_SCALAR_SELECTORS, ids=str)
    def test_the_scalar_spectrum_matches_the_energy_integral(
        self, gamma: float, selector: list[str]
    ) -> None:
        self._compare(gamma, SCALAR_PWS, selector, vector=False)

    @pytest.mark.parametrize("gamma", TAIL_BOOSTS)
    @pytest.mark.parametrize("mode", VECTOR_MODES)
    def test_the_vector_spectrum_matches_the_energy_integral(
        self, gamma: float, mode: str
    ) -> None:
        self._compare(gamma, VECTOR_PWS, mode, vector=True)

    @staticmethod
    def _compare(
        gamma: float, pws: np.ndarray, selector: Selector, *, vector: bool
    ) -> None:
        mass = 550.0
        energy = gamma * mass
        top = _lab_endpoint(mass, energy, pws, selector, vector=vector)
        call = vector_call if vector else scalar_call
        for egam in top * np.geomspace(1e-3, 0.999, 13):
            got = call(float(egam), energy, mass, pws, selector)
            want = energy_reference(egam, energy, mass, pws, selector, vector=vector)
            assert got == pytest.approx(
                want, rel=TAIL_RTOL, abs=0.0
            ), f"{selector} at {egam / top:.4f} of the {top:.1f} MeV endpoint"

    @pytest.mark.parametrize("gamma", TAIL_BOOSTS)
    @pytest.mark.parametrize(
        ("selector", "vector"),
        [
            (["pi pi", "mu mu", "pi0 pi0", "e e g", "pi pi g", "mu mu g"], False),
            ("e e g", True),
            ("pi pi", True),
            ("mu mu", True),
        ],
        ids=["scalar-continua", "vector-e_e_g", "vector-pi_pi", "vector-mu_mu"],
    )
    def test_the_boost_carries_the_rest_frame_photon_energy(
        self, gamma: float, selector: Selector, vector: bool
    ) -> None:
        """``int E dN/dE dE`` in the lab is ``gamma`` times the rest frame's.

        A statement about any isotropic source that owes nothing to either
        integrator: the mean of ``E = gamma E' (1 + beta cos theta*)`` over
        the decay angle is ``gamma E'``. The lab side is a trapezoid over
        ``ln E`` from ``1e-7`` of the endpoint up to it, the rest-frame side
        scipy over ``ln E'`` from ``1e-7`` of its endpoint, so each drops the
        same negligible sliver at the bottom. Lines are excluded, since they
        ride outside the integral.
        """
        mass = 550.0
        energy = gamma * mass
        pws = VECTOR_PWS if vector else SCALAR_PWS
        spectrum, endpoint, kinks = rest_frame_spectrum(
            mass, pws, selector, vector=vector
        )
        floor = 1e-7 * endpoint
        points = sorted({math.log(k) for k in kinks if floor < k < endpoint})
        # Roundoff at the tables' knots, as in `energy_reference`.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IntegrationWarning)
            rest_frame = quad(
                lambda u: math.exp(2.0 * u) * spectrum(math.exp(u)),
                math.log(floor),
                math.log(endpoint),
                points=points,
                limit=500,
                epsrel=1e-8,
            )[0]
        top = _lab_endpoint(mass, energy, pws, selector, vector=vector)
        energies = top * np.geomspace(1e-7, 1.0, 4001)
        if vector:
            lab = np.asarray(dnde_decay_v(energies, energy, mass, pws, selector))
        else:
            lab = np.asarray(scalar_spectrum(energies, energy, mass, pws, selector))
        carried = np.trapezoid(energies * energies * lab, np.log(energies))
        assert carried / (gamma * rest_frame) == pytest.approx(
            1.0, abs=ENERGY_IDENTITY_TOL
        )


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

    # ---- the FSR normalization ------------------------------------------
    #
    # `chi chi -> S* -> f fbar gamma` at `sqrt(s) = m_s` and `S -> f fbar
    # gamma` at rest share a matrix element: the dark-matter current
    # factorizes out of the normalized photon spectrum (the vector case by
    # current conservation). So the annihilation-side closed forms in
    # `hazma.scalar_mediator` / `hazma.vector_mediator`, which reproduce
    # the pair-summed collinear limit of arXiv:1907.11846 Eq. 4.6 to under
    # a percent, are an oracle for the decay kernels' FSR that shares no
    # code with them. The kernels evaluate at the legacy constant table
    # (`ALPHA_EM = 1/137`, the masses above); the models at
    # `hazma.parameters`. The alpha ratio is applied explicitly and the
    # mass differences (3e-8 for the leptons, 1.5e-6 for the pion) set the
    # budgets.

    #: `alpha_legacy / alpha_PDG`: what separates a kernel from a model
    #: evaluating the same expression.
    ALPHA_RATIO = ALPHA_EM / parameters.alpha_em

    @staticmethod
    def _fsr_grid(mass: float, mode: str) -> np.ndarray:
        # Forty photon energies from the soft end to 98% of the channel's
        # own endpoint `x_max = 1 - 4 (m_f / m_s)^2`, so every point is
        # inside the support of both sides.
        m_f = {
            "e e g": parameters.electron_mass,
            "mu mu g": parameters.muon_mass,
            "pi pi g": parameters.charged_pion_mass,
        }[mode]
        x_max = 1.0 - 4.0 * (m_f / mass) ** 2
        return np.geomspace(1e-2, 0.98 * x_max * mass / 2.0, 40)

    @pytest.mark.parametrize(
        ("mode", "index", "rtol"),
        [("e e g", 0, 1e-6), ("mu mu g", 1, 1e-6), ("pi pi g", 3, 1e-4)],
    )
    def test_the_scalar_fsr_is_the_annihilation_matrix_element_at_rest(
        self, mode: str, index: int, rtol: float
    ) -> None:
        # The pre-repair kernel sat at exactly 0.5 here in all three
        # channels (`docs/followups/done/scalar-decay-fsr-half-normalized.md`).
        mass = 550.0
        model = ScalarMediator(
            mx=1e-3, ms=1e3, gsxx=1.0, gsff=1.0, gsGG=0.0, gsFF=0.0, lam=1e5
        )
        egams = self._fsr_grid(mass, mode)
        pws = np.zeros(5)
        pws[index] = 1.0
        if mode == "pi pi g":
            want = model.dnde_xx_to_s_to_pipig(egams, mass)
        else:
            lepton = (
                parameters.electron_mass if mode == "e e g" else parameters.muon_mass
            )
            want = model.dnde_xx_to_s_to_ffg(egams, mass, lepton)
        got = np.asarray(scalar_call(egams, mass, mass, pws=pws, modes=[mode]))
        assert np.all(want > 0.0)
        np.testing.assert_allclose(got / self.ALPHA_RATIO, want, rtol=rtol, atol=0.0)

    @pytest.mark.parametrize(
        ("mode", "index", "rtol"),
        [("e e g", 0, 1e-6), ("mu mu g", 1, 1e-6), ("pi pi g", 3, 1e-4)],
    )
    def test_the_vector_fsr_is_the_annihilation_matrix_element_at_rest(
        self, mode: str, index: int, rtol: float
    ) -> None:
        # The vector twin never had the defect; this is the same statement
        # so that the two kernels are held to one normalization.
        mass = 550.0
        model = VectorMediator(
            mx=1e-3,
            mv=1e3,
            gvxx=1.0,
            gvuu=1.0,
            gvdd=-1.0,
            gvss=0.0,
            gvee=1.0,
            gvmumu=1.0,
        )
        egams = self._fsr_grid(mass, mode)
        pws = np.zeros(4)
        pws[index] = 1.0
        if mode == "pi pi g":
            want = model.dnde_xx_to_v_to_pipig(egams, mass)
        else:
            want = model.dnde_xx_to_v_to_ffg(
                egams, mass, "e" if mode == "e e g" else "mu"
            )
        got = np.array([vector_call(e, mass, mass, pws=pws, mode=mode) for e in egams])
        assert np.all(want > 0.0)
        np.testing.assert_allclose(got / self.ALPHA_RATIO, want, rtol=rtol, atol=0.0)

    def test_the_scalar_lepton_fsr_reproduces_the_collinear_limit(self) -> None:
        # Independent of both models: at `m_e / m_s = 5e-4` and `x = 0.02`
        # the exact `S -> e+ e- gamma` spectrum is the pair-summed
        # Altarelli-Parisi form (Eq. 4.6 of arXiv:1907.11846, twice the
        # per-leg `dnde_photon_ap_fermion`) to 1.4e-5. A 1e-3 budget is
        # fifty times that and five hundred times tighter than the factor
        # of two the repair removed.
        mass = 1000.0
        egam = 0.01 * mass
        limit = (
            2.0
            * spectra.dnde_photon_ap_fermion(
                np.array([egam]), mass**2, parameters.electron_mass
            )[0]
        )
        got = scalar_call(
            egam, mass, mass, pws=np.array([1.0, 0, 0, 0, 0]), modes=["e e g"]
        )
        assert got / self.ALPHA_RATIO == pytest.approx(limit, rel=1e-3, abs=0.0)


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
