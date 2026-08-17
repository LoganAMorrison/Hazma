r"""Specification of the golden parity corpus.

This module is the *specification* half of Hazma's parity corpus: it
declares which compiled entry points are pinned, on which arguments, and
how each one is invoked. `generate.py` evaluates the specification and
writes the reference arrays; the parity test suite re-evaluates it
against whatever implementation is live and compares.

Nothing here imports the reference data, and nothing here asserts a
number. Keeping the specification separate from both means the corpus
can be regenerated and re-checked from one description.

Coverage
--------
The corpus covers the **41 consumed** public ``def``s of the surviving
compiled layer, enumerated in
``projects/cython-to-rust/references/cython-inventory.md`` ("Entry
points by module"): 16 spectra kernels, 12 scalar-mediator and 6
vector-mediator cross-section entry points, and 7 mediator-spectrum
entry points. The two ``sigma_xx_to_all`` exports are excluded because
nothing imports them; `assert_unconsumed_exports_are_unimported` proves
that at generation time rather than trusting the inventory snapshot.

Grid design
-----------
Every swept argument is a sorted union of

* a broad **log-spaced base grid**, deliberately running well below and
  well above the scales of the problem, so that the below-threshold
  zeros, the above-endpoint zeros and the NaN/negative-prone regions are
  all sampled without having to be predicted; and
* **anchors** — energies where the integrand or the branch structure
  changes — each replicated at :math:`a`, :math:`a(1 \pm 10^{-9})` and
  :math:`a(1 \pm 10^{-6})` so a port that moves a branch boundary by a
  representable amount is caught rather than stepped over.

Anchors are never clipped to the base range. A few legitimately fall
outside it — the tabulated photon spectra start their energy tables
around :math:`M / 10^{6}` — and there the grid simply extends past the
base grid's ends.

For the spectra the anchors are the rest-frame kinematic edges boosted
into the lab frame, :math:`\gamma e_{\rm rf} (1 \pm \beta)`. Three
families of rest-frame edge are declared:

* ``M / 2`` for every parent — the two-body line energy for
  :math:`X \to \gamma\gamma` and the natural half-mass scale the phase
  file names explicitly (``E -> m/2``);
* the endpoint constants the kernels themselves define, cited to their
  source line (``hazma/spectra/_photon/_pion.pyx:16-18``);
* the **table edges** of the tabulated photon spectra, read from the
  CSVs in ``hazma/spectra/_photon/data/`` at generation time rather
  than transcribed, since those are exactly where
  ``dnde_photon_*_rest_frame`` switches between the ``1/E`` tail, the
  interpolant and the hard zero (e.g.
  ``hazma/spectra/_photon/_eta.pyx:38-44``).

Parent energies
---------------
Each spectrum is captured at five parent energies: exactly at rest
(``E = M``), at rest plus one ulp-scale increment (``E = M(1 + 1e-12)``,
which straddles the ``E - M < DBL_EPSILON`` rest-frame short-circuit),
just above rest (``1.05 M``), mildly boosted (``2 M``) and strongly
boosted (``10 M``). That is one more than the four the phase file
requires; the extra one is the short-circuit boundary, which is a
branch every ported kernel has to reproduce.

Citations
---------
Every ``file:line`` citation in this module was read against the
pre-port tree at commit ``f025448``. Phases 04-06 delete the files they
point into; the line numbers are historical evidence, not live
references.

Values are stored exactly as the library returns them, including
``nan``, ``inf`` and negative entries: edge behavior is part of the
contract being pinned, not noise to be filtered.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import re
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hazma import parameters as params

#: A live compiled entry point.
EntryPoint = Callable[..., Any]
#: An imported Python module. `types.ModuleType` would be exact, but naming
#: it costs an import for a value only ever introspected for ``__file__``.
Module = Any
#: What an entry point returns: a float, a 1-D array, or -- for the neutrino
#: kernels -- a 3-tuple or a ``(3, N)`` array.
Result = Any
#: A mediator model instance -- ``HiggsPortal`` or ``KineticMixing``. Typed
#: loosely on purpose: naming the classes here would mean importing them at
#: module scope, which pulls the compiled cross-section extensions in and
#: breaks ``generate.py --check`` on an unbuilt tree.
Model = Any
#: ``(label, model)`` pairs, as produced by the ``*_model_points`` factories.
ModelPoints = Sequence[tuple[str, Model]]

REPO_ROOT = Path(__file__).resolve().parents[2]
PHOTON_DATA_DIR = REPO_ROOT / "hazma" / "spectra" / "_photon" / "data"

# Relative offsets applied around every anchor energy. 1e-9 and 1e-6 are
# well above float64 resolution (~2e-16 relative) so the offset points are
# distinct from the anchor, and well below any physical scale in the
# problem so they stay on the intended side of the branch.
ANCHOR_OFFSETS = (-1e-6, -1e-9, 0.0, 1e-9, 1e-6)

#: Points in the broad log-spaced base grid underlying every swept argument.
BASE_GRID_POINTS = 240

#: Points in the ``x = mx / T`` grid for the thermally averaged cross sections.
#: Each point is a QAGP call, so this grid is coarser than the others.
THERMAL_GRID_POINTS = 60

#: Number of grid points re-evaluated through the scalar (float in, float
#: out) branch of an entry point that accepts both. Small because the
#: scalar path pins dispatch, not extra physics.
SCALAR_PROBE_POINTS = 8


# ===========================================================================
# ---- Specification types --------------------------------------------------
# ===========================================================================


@dataclass(frozen=True)
class Block:
    """One captured evaluation: a swept grid at one fixed argument set.

    Parameters
    ----------
    label : str
        Identifier, unique within its `Case`. Used as the npz key prefix
        and as the pytest parameter id.
    params : dict
        The fixed (non-swept) arguments, in JSON-serializable form. These
        are recorded in the manifest so a reader can reconstruct the call
        without reading this module.
    grid : numpy.ndarray
        The swept argument, ascending, float64.
    array_call : callable
        ``(fn, grid) -> ndarray``. Invokes the entry point over the whole
        grid. For entry points that take only a scalar, this loops.
    scalar_call : callable, optional
        ``(fn, x) -> float or tuple``. Invokes the entry point's scalar
        branch at a single grid value. ``None`` when the entry point has
        no separate scalar branch.
    """

    label: str
    params: dict[str, Any]
    grid: np.ndarray
    array_call: Callable[[Callable[..., Any], np.ndarray], Any]
    scalar_call: Callable[[Callable[..., Any], float], Any] | None = None

    @property
    def scalar_probe(self) -> np.ndarray:
        """The subset of `grid` re-evaluated through the scalar branch.

        Evenly spaced across the grid so the probe spans the full range
        rather than clustering at one end.
        """
        if self.scalar_call is None:
            return np.empty(0, dtype=np.float64)
        idx = np.linspace(0, self.grid.size - 1, SCALAR_PROBE_POINTS)
        return self.grid[np.unique(idx.astype(int))]


@dataclass(frozen=True)
class Case:
    """One compiled entry point and every block captured for it."""

    name: str
    module: str
    function: str
    summary: str
    blocks: list[Block] = field(default_factory=list)

    @property
    def entry_point(self) -> str:
        """``module:function``, the form recorded in the manifest."""
        return f"{self.module}:{self.function}"

    def resolve(self) -> Callable[..., Any]:
        """Import and return the live entry point.

        The module's file is checked to live inside `REPO_ROOT` before the
        entry point is handed back. Import resolution follows `sys.path`,
        which need not agree with the tree this module measures: a
        site-packages install shadows the checkout depending on cwd and how
        the environment was set up (`docs/agents/environment.md`, "You may
        be importing an installed hazma, not the worktree"). Without this
        check the corpus could be captured from one build while
        `kernel_digest` described another, and the manifest would record
        that falsehood as provenance.
        """
        module = importlib.import_module(self.module)
        assert_module_is_repo_tree(module)
        return getattr(module, self.function)


# ===========================================================================
# ---- Grid construction ----------------------------------------------------
# ===========================================================================


def log_grid(
    lo: float, hi: float, anchors: Sequence[float], npoints: int = BASE_GRID_POINTS
) -> np.ndarray:
    """Build a log-spaced grid over ``[lo, hi]`` with anchors woven in.

    Parameters
    ----------
    lo, hi : float
        Ends of the base grid. Both must be strictly positive.
    anchors : sequence of float
        Energies to sample densely. Each contributes the five points
        ``a * (1 + delta)`` for ``delta`` in `ANCHOR_OFFSETS`. Anchors
        outside ``[lo, hi]`` are **kept**, and extend the returned grid
        past the base range — an anchor marks a branch boundary, so
        clipping it would drop exactly the point worth sampling.
        Non-positive and non-finite anchors are dropped (they arise
        naturally, e.g. a two-body edge for a closed channel).
    npoints : int
        Size of the base grid.

    Returns
    -------
    numpy.ndarray
        Ascending float64 grid. Duplicates are removed so the grid is a
        set, which keeps the size stable under anchor collisions.
    """
    if not (lo > 0.0 and hi > lo):
        raise ValueError(f"log_grid needs 0 < lo < hi, got lo={lo}, hi={hi}")

    points = [np.geomspace(lo, hi, npoints)]
    for anchor in anchors:
        if not np.isfinite(anchor) or anchor <= 0.0:
            continue
        points.append(np.array([anchor * (1.0 + d) for d in ANCHOR_OFFSETS]))

    return np.unique(np.concatenate(points))


def boosted_edges(
    rest_frame_edges: Sequence[float], energy: float, mass: float
) -> list[float]:
    """Map rest-frame edge energies into the lab frame.

    A massless (or ultrarelativistic) product emitted at a rest-frame
    energy ``e`` populates the lab-frame interval
    ``[gamma * e * (1 - beta), gamma * e * (1 + beta)]``; both ends are
    branch boundaries for the boost integral. At rest (``energy == mass``)
    this collapses to ``e`` itself.

    Parameters
    ----------
    rest_frame_edges : sequence of float
        Rest-frame energies, MeV.
    energy, mass : float
        Parent energy and mass, MeV.

    Returns
    -------
    list of float
        Lab-frame edge energies, MeV.
    """
    gamma = energy / mass
    beta = np.sqrt(max(1.0 - 1.0 / (gamma * gamma), 0.0))
    edges: list[float] = []
    for e in rest_frame_edges:
        edges.extend([e, gamma * e * (1.0 - beta), gamma * e * (1.0 + beta)])
    return edges


def parent_energies(mass: float) -> list[tuple[str, float]]:
    """The five parent energies every spectrum is captured at.

    See the module docstring for why the ``1 + 1e-12`` entry is there.
    """
    return [
        ("rest", mass),
        ("rest_plus_eps", mass * (1.0 + 1e-12)),
        ("near_rest", mass * 1.05),
        ("boosted_mild", mass * 2.0),
        ("boosted_strong", mass * 10.0),
    ]


def table_edges(csv_name: str) -> list[float]:
    """Read the energy-grid ends of a tabulated photon spectrum.

    The tabulated kernels (``_kaon``, ``_eta``, ``_eta_prime``,
    ``_omega``, ``_phi``) load these CSVs at import, transpose them, and
    treat column 0 as the energy grid; below ``energies[0]`` they switch
    to a ``1/E`` tail and above ``energies[-1]`` they return zero — see
    ``hazma/spectra/_photon/_eta.pyx:38-44``. Reading the ends here keeps
    the anchors tied to the shipped data instead of to a transcription.
    """
    energies = np.loadtxt(PHOTON_DATA_DIR / csv_name, delimiter=",").T[0]
    return [float(energies[0]), float(energies[-1])]


# ===========================================================================
# ---- Rest-frame kinematic edges -------------------------------------------
# ===========================================================================

# Endpoint constants the kernels define for themselves, cited to source so
# a reviewer can check the transcription rather than the physics:
#   ENG_GAM_MAX_MURF -- hazma/spectra/_photon/_pion.pyx:16
#   ENG_GAM_MAX_PIRG -- hazma/spectra/_photon/_pion.pyx:17
#   ENG_MU_PIRF      -- hazma/spectra/_photon/_pion.pyx:18
ENG_GAM_MAX_MURF = 52.82795006985128
ENG_GAM_MAX_PIRG = 69.78345771948752
ENG_MU_PIRF = 109.77820123634007


def _half_mass(mass: float) -> list[float]:
    """The ``E -> m/2`` anchor the phase file names, for every parent."""
    return [mass / 2.0]


# ===========================================================================
# ---- Spectra entry points -------------------------------------------------
# ===========================================================================

#: ``(case name, module, function, parent mass, extra rest-frame edges)``.
#: The parent masses are `hazma.parameters` values, which agree with the
#: compile-time ``MASS_*`` in ``hazma/_utils/constants.pxd`` for every
#: parent listed here.
_SPECTRA: list[tuple[str, str, str, float, list[float]]] = [
    (
        # Ported: cython-to-rust Task 4.3. The module is the *wrapper*,
        # not the `.pyx`, because that is where the value now comes from
        # — see `PORTED_ENTRY_POINTS`.
        "photon.muon",
        "hazma.spectra._photon",
        "dnde_photon_muon",
        params.muon_mass,
        [ENG_GAM_MAX_MURF],
    ),
    (
        "photon.charged_pion",
        "hazma.spectra._photon._pion",
        "dnde_photon_charged_pion",
        params.charged_pion_mass,
        [ENG_GAM_MAX_PIRG, ENG_GAM_MAX_MURF, ENG_MU_PIRF],
    ),
    (
        "photon.neutral_pion",
        "hazma.spectra._photon._pion",
        "dnde_photon_neutral_pion",
        params.neutral_pion_mass,
        [],
    ),
    (
        "photon.charged_rho",
        "hazma.spectra._photon._rho",
        "dnde_photon_charged_rho",
        params.rho_mass,
        [ENG_GAM_MAX_PIRG, params.neutral_pion_mass / 2.0],
    ),
    (
        "photon.neutral_rho",
        "hazma.spectra._photon._rho",
        "dnde_photon_neutral_rho",
        params.rho_mass,
        [ENG_GAM_MAX_PIRG, params.neutral_pion_mass / 2.0],
    ),
    (
        # Ported: cython-to-rust Task 4.2 -- this row and the six below.
        # The module is the *wrapper* for the same reason the positron
        # muon's is; see `PORTED_ENTRY_POINTS`.
        "photon.charged_kaon",
        "hazma.spectra._photon",
        "dnde_photon_charged_kaon",
        params.charged_kaon_mass,
        table_edges("charged_kaon_photon.csv"),
    ),
    (
        "photon.long_kaon",
        "hazma.spectra._photon",
        "dnde_photon_long_kaon",
        params.long_kaon_mass,
        table_edges("long_kaon_photon.csv"),
    ),
    (
        "photon.short_kaon",
        "hazma.spectra._photon",
        "dnde_photon_short_kaon",
        params.short_kaon_mass,
        table_edges("short_kaon_photon.csv"),
    ),
    (
        "photon.eta",
        "hazma.spectra._photon",
        "dnde_photon_eta",
        params.eta_mass,
        table_edges("eta_photon.csv"),
    ),
    (
        "photon.eta_prime",
        "hazma.spectra._photon",
        "dnde_photon_eta_prime",
        params.eta_prime_mass,
        table_edges("eta_prime_photon.csv"),
    ),
    (
        "photon.omega",
        "hazma.spectra._photon",
        "dnde_photon_omega",
        params.omega_mass,
        table_edges("omega_photon.csv"),
    ),
    (
        "photon.phi",
        "hazma.spectra._photon",
        "dnde_photon_phi",
        params.phi_mass,
        table_edges("phi_photon.csv"),
    ),
    (
        # Ported: cython-to-rust Task 4.1. The module is the *wrapper*,
        # not the `.pyx`, because that is where the value now comes from
        # — see `PORTED_ENTRY_POINTS` for why the origin is still
        # recorded and what `assert_full_coverage` does with it.
        "positron.muon",
        "hazma.spectra._positron",
        "dnde_positron_muon",
        params.muon_mass,
        [params.electron_mass],
    ),
    (
        "positron.charged_pion",
        "hazma.spectra._positron._pion",
        "dnde_positron_charged_pion",
        params.charged_pion_mass,
        [params.electron_mass, params.muon_mass / 2.0, ENG_MU_PIRF],
    ),
    (
        "neutrino.muon",
        "hazma.spectra._neutrino._muon",
        "dnde_neutrino_muon",
        params.muon_mass,
        [],
    ),
    (
        "neutrino.charged_pion",
        "hazma.spectra._neutrino._pion",
        "dnde_neutrino_charged_pion",
        params.charged_pion_mass,
        [ENG_MU_PIRF, params.muon_mass / 2.0],
    ),
]


def _spectrum_case(
    name: str,
    module: str,
    function: str,
    mass: float,
    extra_edges: Sequence[float],
) -> Case:
    """Build the `Case` for one ``dnde_*(product_energy, parent_energy)``."""
    rest_edges = list(_half_mass(mass)) + list(extra_edges)
    blocks = []
    for label, energy in parent_energies(mass):
        # Base grid: five decades below the parent mass to a hundred times
        # the parent energy, which puts the zero (or ``1/E`` tail) regions
        # on both sides of the physical support into the corpus rather than
        # only the support itself. Anchors are added whether or not they
        # land inside that range, and some do not: the tabulated photon
        # spectra start their tables around ``M / 1e6`` (the charged-kaon
        # table's first energy is 4.936770e-04 MeV), so those anchors
        # extend the grid downward. That is intended — the table's lower
        # end is a branch boundary and has to be sampled.
        anchors = boosted_edges(rest_edges, energy, mass)
        blocks.append(
            Block(
                label=label,
                params={"parent_energy": energy, "parent_mass": mass},
                grid=log_grid(1e-5 * mass, 100.0 * energy, anchors),
                array_call=lambda fn, grid, e=energy: fn(grid, e),
                scalar_call=lambda fn, x, e=energy: fn(float(x), e),
            )
        )
    return Case(
        name=f"spectra.{name}",
        module=module,
        function=function,
        summary=f"dN/dE from {name.replace('.', ' ')} decay, MeV^-1",
        blocks=blocks,
    )


# ===========================================================================
# ---- Mediator model parameter points --------------------------------------
# ===========================================================================


def _scalar_model_points() -> ModelPoints:
    """Three `HiggsPortal` configurations for the scalar-mediator cases.

    Chosen so the three differ in the feature that dominates the
    cross-section shape:

    ``open_resonance``
        ``ms > 2 mx``, moderate mixing — the s-channel pole at
        ``e_cm = ms`` sits well above the ``2 mx`` threshold and is
        broad.
    ``narrow_resonance``
        the near-resonance configuration the phase file requires:
        ``stheta = 1e-4`` makes ``width_s`` tiny, so the pole is a
        spike that a quadrature or propagator difference will show up
        against. Grid anchors are placed at ``ms`` and at
        ``ms +- width_s``.
    ``closed_resonance``
        ``ms < 2 mx`` — the pole is below threshold and never sampled
        on the physical side, which exercises the opposite branch.
    """
    from hazma.scalar_mediator import HiggsPortal  # noqa: PLC0415 (see below)

    return [
        ("open_resonance", HiggsPortal(mx=100.0, ms=300.0, gsxx=1.0, stheta=1e-1)),
        ("narrow_resonance", HiggsPortal(mx=200.0, ms=550.0, gsxx=1.0, stheta=1e-4)),
        ("closed_resonance", HiggsPortal(mx=300.0, ms=200.0, gsxx=1.0, stheta=1e-2)),
    ]


def _vector_model_points() -> ModelPoints:
    """Three `KineticMixing` configurations, mirroring `_scalar_model_points`."""
    from hazma.vector_mediator import KineticMixing  # noqa: PLC0415 (see below)

    return [
        ("open_resonance", KineticMixing(mx=100.0, mv=300.0, gvxx=1.0, eps=1e-1)),
        ("narrow_resonance", KineticMixing(mx=200.0, mv=550.0, gvxx=1.0, eps=1e-4)),
        ("closed_resonance", KineticMixing(mx=300.0, mv=200.0, gvxx=1.0, eps=1e-2)),
    ]


def _scalar_args(model: Model) -> list[float]:
    """The nine leading arguments every scalar cross-section entry takes."""
    return [
        model.mx,
        model.ms,
        model.gsxx,
        model.gsff,
        model.gsGG,
        model.gsFF,
        model.lam,
        model.width_s,
        model.vs,
    ]


def _vector_args(model: Model) -> list[float]:
    """The eight leading arguments the multi-coupling vector entries take."""
    return [
        model.mx,
        model.mv,
        model.gvxx,
        model.gvuu,
        model.gvdd,
        model.gvss,
        model.gvee,
        model.gvmumu,
        model.width_v,
    ]


def _cross_section_anchors(mx: float, mmed: float, width: float) -> list[float]:
    """Thresholds and poles reachable by any cross-section entry point.

    One anchor list serves every entry point in a model: annihilation
    thresholds (``2 mx``, ``2 m_f``), the s-channel pole (``mmed``) and
    its half-width shoulders, the ``ss -> xx`` threshold (``2 mmed``),
    and the elastic-scattering thresholds (``mx + m_target``). Anchors
    that no given entry point cares about cost only a few extra grid
    points.
    """
    targets = [
        params.electron_mass,
        params.muon_mass,
        params.charged_pion_mass,
        params.neutral_pion_mass,
        mmed,
    ]
    anchors = [mx, 2.0 * mx, mmed, 2.0 * mmed, mmed + width, mmed - width]
    anchors += [2.0 * m for m in targets]
    anchors += [mx + m for m in targets]
    return anchors


def _cross_section_blocks(
    model_points: ModelPoints,
    args_fn: Callable[[Model], list[float]],
    med_attr: str,
    width_attr: str,
    extra: Sequence[float] = (),
) -> list[Block]:
    """Blocks sweeping ``e_cm`` for one cross-section entry point.

    Parameters
    ----------
    model_points : sequence
        ``(label, model)`` pairs from `_scalar_model_points` or
        `_vector_model_points`.
    args_fn : callable
        Maps a model to the fixed argument list following ``e_cms``.
    med_attr, width_attr : str
        Attribute names of the mediator mass and width on the model.
    extra : sequence of float
        Trailing arguments after `args_fn`'s, e.g. the final-state
        fermion mass.
    """
    blocks = []
    for label, model in model_points:
        mmed = getattr(model, med_attr)
        width = getattr(model, width_attr)
        anchors = _cross_section_anchors(model.mx, mmed, width)
        args = args_fn(model) + list(extra)
        blocks.append(
            Block(
                label=label,
                params={"args": args, "mx": model.mx, "m_med": mmed, "width": width},
                # From a decade below the lowest threshold to two decades
                # above the highest scale: both the identically-zero
                # sub-threshold region and the asymptotic tail are pinned.
                grid=log_grid(0.1 * min(anchors), 100.0 * max(anchors), anchors),
                array_call=lambda fn, grid, a=args: fn(grid, *a),
                scalar_call=lambda fn, x, a=args: fn(float(x), *a),
            )
        )
    return blocks


def _thermal_blocks(
    model_points: ModelPoints,
    args_fn: Callable[[Model], list[float]],
    med_attr: str,
) -> list[Block]:
    r"""Blocks sweeping ``x = mx / T`` for a thermally averaged cross section.

    The grid spans ``x`` from 0.1 (relativistic, well before freeze-out)
    to 1000 (long after), bracketing the ``x ~ 20`` freeze-out region.
    Anchors:

    * ``x = 20`` -- the conventional freeze-out value;
    * ``x = 0.5`` -- where the integration upper bound switches from the
      constant floor to ``50 / x``
      (``max(50.0 / x, 100.0)`` at
      ``hazma/scalar_mediator/_c_scalar_mediator_cross_sections.pyx:1412``;
      the vector floor is 150
      (``hazma/vector_mediator/_c_vector_mediator_cross_sections.pyx:657``),
      giving ``x = 1/3``);
    * ``x = 1/3`` -- the vector model's equivalent switch;
    * ``x = m_med / mx`` and ``2 m_med / mx`` -- the QAGP breakpoints,
      which are exactly where a breakpoint-handling difference shows;
    * ``x = 300`` -- the low-temperature cutoff, where the two models
      **disagree with each other**: the scalar returns ``0.0`` outright
      (``if x > 300: return 0.0``,
      ``hazma/scalar_mediator/_c_scalar_mediator_cross_sections.pyx:1401-1402``)
      while the vector saturates, clipping to ``xnew = 300`` and
      continuing to return the value there
      (``hazma/vector_mediator/_c_vector_mediator_cross_sections.pyx:649``).
      A port that unifies the two would move published numbers above
      ``x = 300``, so the corpus pins both behaviors.

    ``thermal_cross_section`` takes a scalar ``x`` only, so the "array"
    call loops; there is no separate scalar branch to probe.
    """
    blocks = []
    for label, model in model_points:
        mmed = getattr(model, med_attr)
        ratio = mmed / model.mx
        anchors = [20.0, 0.5, 1.0 / 3.0, 1.0, 300.0, ratio, 2.0 * ratio]
        args = args_fn(model)
        blocks.append(
            Block(
                label=label,
                params={"args": args, "mx": model.mx, "m_med": mmed},
                grid=log_grid(0.1, 1000.0, anchors, npoints=THERMAL_GRID_POINTS),
                array_call=lambda fn, grid, a=args: np.array(
                    [fn(float(x), *a) for x in grid]
                ),
                scalar_call=None,
            )
        )
    return blocks


# ===========================================================================
# ---- Mediator spectrum entry points ---------------------------------------
# ===========================================================================


def _scalar_spectrum_models() -> ModelPoints:
    """Three `HiggsPortal` points for the mediator *spectrum* entry points.

    Here the pole structure is irrelevant — what matters is that the
    mediator actually decays visibly, so the partial-width vector the
    kernels consume is not identically zero. Setting ``mx = ms`` closes
    ``s -> x x`` (which needs ``ms > 2 mx``) and leaves the visible modes
    carrying the whole width. The three masses straddle the ``2 m_mu``,
    ``2 m_pi`` and ``2 m_pi0`` thresholds so different mode sets are open
    in each.
    """
    from hazma.scalar_mediator import HiggsPortal  # noqa: PLC0415 (see below)

    return [
        (f"ms_{int(ms)}", HiggsPortal(mx=ms, ms=ms, gsxx=1.0, stheta=1e-1))
        for ms in (250.0, 550.0, 900.0)
    ]


def _vector_spectrum_models() -> ModelPoints:
    """Three `KineticMixing` points, mirroring `_scalar_spectrum_models`."""
    from hazma.vector_mediator import KineticMixing  # noqa: PLC0415 (see below)

    return [
        (f"mv_{int(mv)}", KineticMixing(mx=mv, mv=mv, gvxx=1.0, eps=1e-1))
        for mv in (250.0, 550.0, 900.0)
    ]


def _normalized_widths(model: Model, keys: Sequence[str]) -> np.ndarray:
    """Branching ratios in the order a mediator-spectrum kernel expects.

    Mirrors the Python wrappers, which divide each partial width by the
    total (`hazma/scalar_mediator/_scalar_mediator_positron_spectra.py:69-70`,
    `hazma/vector_mediator/_vector_mediator_spectra.py:94-97`). Missing
    keys become zero, which is what a closed channel contributes.
    """
    widths = model.partial_widths()
    total = widths["total"]
    return np.array([widths.get(key, 0.0) / total for key in keys], dtype=np.float64)


def _array_caller(args: tuple) -> Callable[[EntryPoint, np.ndarray], Any]:
    """Call an entry point that takes the whole energy grid at once."""

    def call(fn: EntryPoint, grid: np.ndarray) -> Result:
        return fn(grid, *args)

    return call


def _pointwise_caller(args: tuple) -> Callable[[EntryPoint, np.ndarray], Any]:
    """Sweep an entry point that only accepts a scalar energy (``*_pt``)."""

    def call(fn: EntryPoint, grid: np.ndarray) -> Result:
        return np.array([fn(float(x), *args) for x in grid])

    return call


def _mediator_spectrum_blocks(
    models: ModelPoints,
    mass_attr: str,
    width_keys: Sequence[str],
    modes: Sequence[str],
    pointwise: bool,
) -> list[Block]:
    """Blocks for a mediator decay/positron spectrum entry point.

    Sweeps the product energy at each of the five parent energies of
    `parent_energies`, for each of three mediator masses and each mode
    string — the mediator mass plays the role the parent mass plays for
    the spectra kernels.

    Parameters
    ----------
    models : sequence
        ``(label, model)`` pairs.
    mass_attr : str
        ``"ms"`` or ``"mv"``.
    width_keys : sequence of str
        `partial_widths` keys, in the order the kernel indexes them.
    modes : sequence of str
        Mode/final-state selector strings to capture.
    pointwise : bool
        True for the ``*_pt`` entry points, which take a scalar product
        energy; the sweep then loops instead of passing the array.
    """
    blocks = []
    for model_label, model in models:
        mass = getattr(model, mass_attr)
        pws = _normalized_widths(model, width_keys)
        for energy_label, energy in parent_energies(mass):
            anchors = boosted_edges(
                [mass / 2.0, params.muon_mass / 2.0, params.charged_pion_mass / 2.0],
                energy,
                mass,
            )
            grid = log_grid(1e-5 * mass, 100.0 * energy, anchors)
            for mode in modes:
                mode_label = re.sub(r"\W+", "_", mode)
                args = (energy, mass, pws, mode)
                array_call = (
                    _pointwise_caller(args) if pointwise else _array_caller(args)
                )
                blocks.append(
                    Block(
                        label=f"{model_label}.{energy_label}.{mode_label}",
                        params={
                            "mediator_energy": energy,
                            "mediator_mass": mass,
                            "partial_widths": pws.tolist(),
                            "width_keys": list(width_keys),
                            "mode": mode,
                        },
                        grid=grid,
                        array_call=array_call,
                        # Every mediator-spectrum entry point has a fixed
                        # arity: the ``*_pt`` variants take a scalar, the
                        # others an array. Neither dispatches, so there is
                        # no second branch to probe.
                        scalar_call=None,
                    )
                )
    return blocks


def _scalar_decay_spectrum_blocks(models: ModelPoints) -> list[Block]:
    """Blocks for ``scalar_mediator_decay_spectrum``.

    Its signature differs from the other six: the selector is a *list* of
    modes reduced to a bitflag, not a single string
    (``hazma/scalar_mediator/scalar_mediator_decay_spectrum.pyx:253-266``).
    Two mode sets are captured — the full default list the wrapper passes
    (`hazma/scalar_mediator/_scalar_mediator_spectra.py:11`) and a
    single-mode list, so the bitflag reduction is pinned rather than only
    its all-bits-set value.
    """
    all_modes = ["pi pi", "mu mu", "pi0 pi0", "g g", "e e g", "pi pi g", "mu mu g"]
    mode_sets = [("default", all_modes), ("mu_mu_only", ["mu mu"])]
    width_keys = ["e e", "mu mu", "pi0 pi0", "pi pi", "g g"]

    blocks = []
    for model_label, model in models:
        mass = model.ms
        pws = _normalized_widths(model, width_keys)
        for energy_label, energy in parent_energies(mass):
            anchors = boosted_edges(
                [mass / 2.0, params.muon_mass / 2.0, params.charged_pion_mass / 2.0],
                energy,
                mass,
            )
            grid = log_grid(1e-5 * mass, 100.0 * energy, anchors)
            for mode_label, modes in mode_sets:
                args = (energy, mass, pws, list(modes))
                blocks.append(
                    Block(
                        label=f"{model_label}.{energy_label}.{mode_label}",
                        params={
                            "mediator_energy": energy,
                            "mediator_mass": mass,
                            "partial_widths": pws.tolist(),
                            "width_keys": width_keys,
                            "modes": list(modes),
                        },
                        grid=grid,
                        array_call=lambda fn, g, a=args: fn(g, a[0], a[1], a[2], a[3]),
                        scalar_call=lambda fn, x, a=args: fn(
                            float(x), a[0], a[1], a[2], a[3]
                        ),
                    )
                )
    return blocks


# ===========================================================================
# ---- The corpus -----------------------------------------------------------
# ===========================================================================

#: Entry points that exist in the compiled layer but that nothing imports.
#: `assert_unconsumed_exports_are_unimported` re-derives this at generation
#: time; the plan drops them in Phase 05 rather than porting them.
UNCONSUMED_EXPORTS = {
    "hazma.scalar_mediator._c_scalar_mediator_cross_sections": "sigma_xx_to_all",
    "hazma.vector_mediator._c_vector_mediator_cross_sections": "sigma_xx_to_all",
}

_SCALAR_XS_MODULE = "hazma.scalar_mediator._c_scalar_mediator_cross_sections"
_VECTOR_XS_MODULE = "hazma.vector_mediator._c_vector_mediator_cross_sections"


def build_cases() -> dict[str, Case]:
    """Construct the whole corpus specification.

    Built lazily rather than at import so that constructing the mediator
    models — which solve for ``vs`` and integrate partial widths — is not
    paid by a reader who only wants the module's constants.

    Returns
    -------
    dict
        Case name to `Case`, insertion-ordered: spectra, then scalar
        cross sections, then vector cross sections, then the mediator
        spectra.
    """
    cases: list[Case] = [_spectrum_case(*spec) for spec in _SPECTRA]

    scalar_models = _scalar_model_points()
    vector_models = _vector_model_points()

    def scalar_xs(name: str, extra: Sequence[float] = (), summary: str = "") -> Case:
        return Case(
            name=f"cross_sections.scalar.{name}",
            module=_SCALAR_XS_MODULE,
            function=name,
            summary=summary or f"scalar-mediator {name}, MeV^-2",
            blocks=_cross_section_blocks(
                scalar_models, _scalar_args, "ms", "width_s", extra
            ),
        )

    def vector_xs(
        name: str,
        args_fn: Callable[[Model], list[float]],
        extra: Sequence[float] = (),
    ) -> Case:
        return Case(
            name=f"cross_sections.vector.{name}",
            module=_VECTOR_XS_MODULE,
            function=name,
            summary=f"vector-mediator {name}, MeV^-2",
            blocks=_cross_section_blocks(
                vector_models, args_fn, "mv", "width_v", extra
            ),
        )

    # -- scalar mediator: 12 consumed entry points --------------------------
    # sigma_xx_to_s_to_ff and sigma_xl_to_xl take a trailing fermion mass;
    # the wrappers only ever pass the electron or the muon
    # (hazma/scalar_mediator/_scalar_mediator_cross_sections.py:42-43), so
    # both are captured under one case with distinct blocks.
    cases.append(
        Case(
            name="cross_sections.scalar.sigma_xx_to_s_to_ff",
            module=_SCALAR_XS_MODULE,
            function="sigma_xx_to_s_to_ff",
            summary="scalar-mediator xx -> s* -> f fbar, MeV^-2",
            blocks=[
                Block(
                    label=f"{block.label}.{fermion}",
                    params={**block.params, "fermion": fermion},
                    grid=block.grid,
                    array_call=block.array_call,
                    scalar_call=block.scalar_call,
                )
                for fermion, mf in (
                    ("e", params.electron_mass),
                    ("mu", params.muon_mass),
                )
                for block in _cross_section_blocks(
                    scalar_models, _scalar_args, "ms", "width_s", (mf,)
                )
            ],
        )
    )
    cases.append(
        Case(
            name="cross_sections.scalar.sigma_xl_to_xl",
            module=_SCALAR_XS_MODULE,
            function="sigma_xl_to_xl",
            summary="scalar-mediator x l -> x l elastic scattering, MeV^-2",
            blocks=[
                Block(
                    label=f"{block.label}.{fermion}",
                    params={**block.params, "fermion": fermion},
                    grid=block.grid,
                    array_call=block.array_call,
                    scalar_call=block.scalar_call,
                )
                for fermion, mf in (
                    ("e", params.electron_mass),
                    ("mu", params.muon_mass),
                )
                for block in _cross_section_blocks(
                    scalar_models, _scalar_args, "ms", "width_s", (mf,)
                )
            ],
        )
    )
    for name in (
        "sigma_xx_to_s_to_gg",
        "sigma_xx_to_s_to_pi0pi0",
        "sigma_xx_to_s_to_pipi",
        "sigma_xx_to_ss",
        "sigma_ss_to_xx",
        "sigma_xpi_to_xpi",
        "sigma_xpi0_to_xpi0",
        "sigma_xg_to_xg",
        "sigma_xs_to_xs",
    ):
        cases.append(scalar_xs(name))
    cases.append(
        Case(
            name="cross_sections.scalar.thermal_cross_section",
            module=_SCALAR_XS_MODULE,
            function="thermal_cross_section",
            summary="scalar-mediator thermally averaged <sigma v>, MeV^-2",
            blocks=_thermal_blocks(scalar_models, _scalar_args, "ms"),
        )
    )

    # -- vector mediator: 6 consumed entry points ---------------------------
    # sigma_xx_to_v_to_ff takes (gvll, width_v, ml) rather than the full
    # coupling vector (hazma/vector_mediator/_c_vector_mediator_cross_sections.pyx:299),
    # and the wrapper picks gvll by final state
    # (hazma/vector_mediator/_vector_mediator_cross_sections.py:54).
    cases.append(
        Case(
            name="cross_sections.vector.sigma_xx_to_v_to_ff",
            module=_VECTOR_XS_MODULE,
            function="sigma_xx_to_v_to_ff",
            summary="vector-mediator xx -> v* -> l lbar, MeV^-2",
            blocks=[
                Block(
                    label=f"{block.label}.{lepton}",
                    params={**block.params, "lepton": lepton},
                    grid=block.grid,
                    array_call=block.array_call,
                    scalar_call=block.scalar_call,
                )
                for lepton, coupling_attr, ml in (
                    ("e", "gvee", params.electron_mass),
                    ("mu", "gvmumu", params.muon_mass),
                )
                for block in _cross_section_blocks(
                    vector_models,
                    lambda m, a=coupling_attr: [
                        m.mx,
                        m.mv,
                        m.gvxx,
                        getattr(m, a),
                        m.width_v,
                    ],
                    "mv",
                    "width_v",
                    (ml,),
                )
            ],
        )
    )
    for name in (
        "sigma_xx_to_v_to_pipi",
        "sigma_xx_to_v_to_pi0g",
        "sigma_xx_to_v_to_pi0v",
        "sigma_xx_to_vv",
    ):
        cases.append(vector_xs(name, _vector_args))
    cases.append(
        Case(
            name="cross_sections.vector.thermal_cross_section",
            module=_VECTOR_XS_MODULE,
            function="thermal_cross_section",
            summary="vector-mediator thermally averaged <sigma v>, MeV^-2",
            blocks=_thermal_blocks(vector_models, _vector_args, "mv"),
        )
    )

    # -- mediator spectra: 7 consumed entry points --------------------------
    scalar_spec_models = _scalar_spectrum_models()
    vector_spec_models = _vector_spectrum_models()

    cases.append(
        Case(
            name="mediator_spectra.scalar.photon.scalar_mediator_decay_spectrum",
            module="hazma.scalar_mediator.scalar_mediator_decay_spectrum",
            function="scalar_mediator_decay_spectrum",
            summary="photon dN/dE from scalar-mediator decay, MeV^-1",
            blocks=_scalar_decay_spectrum_blocks(scalar_spec_models),
        )
    )

    # The positron modules index pws as [e e, mu mu, pi pi]
    # (hazma/scalar_mediator/scalar_mediator_positron_spec.pyx:232-234).
    positron_keys = ["e e", "mu mu", "pi pi"]
    positron_modes = ["total", "e e", "mu mu", "pi pi"]
    for function, pointwise in (("dnde_decay_s", False), ("dnde_decay_s_pt", True)):
        cases.append(
            Case(
                name=f"mediator_spectra.scalar.positron.{function}",
                module="hazma.scalar_mediator.scalar_mediator_positron_spec",
                function=function,
                summary="positron dN/dE from scalar-mediator decay, MeV^-1",
                blocks=_mediator_spectrum_blocks(
                    scalar_spec_models, "ms", positron_keys, positron_modes, pointwise
                ),
            )
        )

    # The vector photon module indexes pws as [e e, mu mu, pi0 g, pi pi]
    # (hazma/vector_mediator/_vector_mediator_spectra.py:94-97) and selects
    # with the mode strings at
    # hazma/vector_mediator/vector_mediator_decay_spectrum.pyx:166-178.
    vector_photon_keys = ["e e", "mu mu", "pi0 g", "pi pi"]
    vector_photon_modes = [
        "total",
        "e e g",
        "mu mu",
        "mu mu g",
        "pi pi",
        "pi pi g",
        "pi0 g",
    ]
    for function, pointwise in (("dnde_decay_v", False), ("dnde_decay_v_pt", True)):
        cases.append(
            Case(
                name=f"mediator_spectra.vector.photon.{function}",
                module="hazma.vector_mediator.vector_mediator_decay_spectrum",
                function=function,
                summary="photon dN/dE from vector-mediator decay, MeV^-1",
                blocks=_mediator_spectrum_blocks(
                    vector_spec_models,
                    "mv",
                    vector_photon_keys,
                    vector_photon_modes,
                    pointwise,
                ),
            )
        )
    for function, pointwise in (("dnde_decay_v", False), ("dnde_decay_v_pt", True)):
        cases.append(
            Case(
                name=f"mediator_spectra.vector.positron.{function}",
                module="hazma.vector_mediator.vector_mediator_positron_spec",
                function=function,
                summary="positron dN/dE from vector-mediator decay, MeV^-1",
                blocks=_mediator_spectrum_blocks(
                    vector_spec_models, "mv", positron_keys, positron_modes, pointwise
                ),
            )
        )

    by_name = {case.name: case for case in cases}
    if len(by_name) != len(cases):
        raise RuntimeError("duplicate case names in the corpus specification")
    return by_name


# ===========================================================================
# ---- Coverage and provenance guards ---------------------------------------
# ===========================================================================


def assert_module_is_repo_tree(module: Module) -> None:
    """Fail unless `module` was loaded from inside `REPO_ROOT`.

    Parameters
    ----------
    module : module
        Any imported module. One without a ``__file__`` (a namespace
        package) is rejected too — that is what a broken or partial
        install looks like.
    """
    path = getattr(module, "__file__", None)
    if path is None:
        raise RuntimeError(
            f"{module.__name__} has no __file__, so it cannot be shown to "
            f"come from {REPO_ROOT}. Refusing to use it for the parity "
            "corpus."
        )
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        raise RuntimeError(
            f"{module.__name__} resolves to {resolved}, outside the "
            f"repository at {REPO_ROOT}. The parity corpus would then be "
            "captured from a different build than the one `kernel_digest` "
            "describes. Install this checkout (`pip install -e .`) or run "
            "from a shell whose sys.path reaches it."
        )


def hazma_package_path() -> Path:
    """Where `hazma` actually resolves from, for the manifest record.

    Imported here rather than at module scope for the same reason the
    mediator models are: `generate.py --check` must keep working on an
    unbuilt tree, and it never calls this.
    """
    import hazma  # noqa: PLC0415 (see docstring)

    assert_module_is_repo_tree(hazma)
    return Path(hazma.__file__).resolve().parent


def rust_core_available() -> bool:
    """Whether this tree has a Rust extension at all.

    True from cython-to-rust Phase 02 on, when the crate is scaffolded.
    It is **not** the question the corpus cares about — an extension that
    serves no kernel changes no value. Use `rust_core_kernels` for that.
    """
    return importlib.util.find_spec("hazma._core") is not None


#: Public names on ``hazma._core`` that are scaffolding rather than ported
#: kernels: nothing in `hazma/` calls them and they compute no physics.
#: `roundtrip` is Phase 02 Task 2.1's plumbing probe. Anything else public
#: on the extension is a kernel by definition — a Phase 04-06 swap adds it
#: precisely so a wrapper can call it. See also
#: :data:`_CORE_TEST_ONLY_MODULES`, which is the same exemption a whole
#: submodule at a time.
_CORE_SCAFFOLD_NAMES = frozenset({"roundtrip"})

#: Submodules of ``hazma._core`` that exist only so a test can reach the
#: Rust side. Their contents compute real numbers but serve no wrapper,
#: so counting them would flip the corpus out of bit-equality mode for
#: the rest of the port with nothing turning red — the failure Task 2.1
#: fixed once already (``docs/agents/lessons.md``,
#: ``[gate-disabled-stays-green]``).
#:
#: ``hazma._core.special`` is Phase 03 Task 3.2's ``spence``/``k1``/``kn``
#: shim, exposed to Python only so ``test/test_core_special.py`` can sweep
#: it against scipy; ``hazma._core.quad`` is Task 3.3's QUADPACK port,
#: exposed so ``test/test_core_quad.py`` can put one Python integrand
#: through both it and ``scipy.integrate.quad``; ``hazma._core.interp``
#: and ``hazma._core.boost`` are Task 3.4's interpolation and boost
#: foundation, exposed so ``test/test_core_interp.py`` can sweep against
#: ``np.interp`` and ``test/test_core_boost.py`` against the Cython twin
#: itself through ``hazma._utils.boost.__pyx_capi__``;
#: ``hazma._core.dispatch`` is Task 3.5's argument-and-error layer,
#: exposed so ``test/test_core_dispatch.py`` can render every error
#: message with a caller-chosen quantity and compare it byte for byte
#: against the strings extracted from the ``.pyx`` sources. In every case the
#: kernels that will use them call the Rust side directly, in Rust, and
#: never through Python. What makes the exemption safe rather than
#: convenient is that no module under `hazma/` may import these — asserted
#: by ``test_test_only_core_submodules_have_no_importer`` in
#: ``test_parity.py``, not left to this comment. **Do not add a submodule
#: here to quiet a failing mode check**: a submodule a wrapper imports is
#: a served kernel, whatever it is named.
_CORE_TEST_ONLY_MODULES = frozenset(
    {
        "hazma._core.special",
        "hazma._core.quad",
        "hazma._core.interp",
        "hazma._core.boost",
        "hazma._core.dispatch",
    }
)


def rust_core_kernels() -> list[str]:
    """Fully-qualified names of the kernels ``hazma._core`` serves today.

    Empty while the extension is a bare scaffold, non-empty from the
    first Phase 04 swap. Discovered by walking the module rather than
    from a hardcoded list, so a kernel added to a submodule this file has
    never heard of still counts.

    Returns
    -------
    list of str
        Sorted, e.g. ``["hazma._core.photon.dnde_photon_muon"]``.
    """
    if not rust_core_available():
        return []

    core = importlib.import_module("hazma._core")
    found: list[str] = []
    seen: set[int] = set()

    def walk(module: types.ModuleType, prefix: str) -> None:
        if id(module) in seen:
            return
        seen.add(id(module))
        for name in dir(module):
            if name.startswith("_"):
                continue
            member = getattr(module, name)
            if inspect.ismodule(member):
                # Only into our own subtree: a submodule that imports
                # numpy must not make numpy look like a ported kernel.
                # And not into the test-only submodules, which are Rust
                # that no wrapper calls.
                qualified = getattr(member, "__name__", "")
                if (
                    qualified.startswith("hazma._core")
                    and qualified not in _CORE_TEST_ONLY_MODULES
                ):
                    walk(member, f"{prefix}.{name}")
            elif callable(member) and name not in _CORE_SCAFFOLD_NAMES:
                found.append(f"{prefix}.{name}")

    walk(core, "hazma._core")
    return sorted(found)


def assert_no_rust_core() -> None:
    """Refuse to touch the corpus if any kernel already runs on Rust.

    ``rules.md`` rule 2: reference arrays are generated only from
    pre-port Cython, or a regenerated corpus would pin the port against
    itself. The test is whether a kernel is *served*, which is what rule 2
    says — the mere existence of the extension is not, and has not been
    since Phase 02 scaffolded it.
    """
    served = rust_core_kernels()
    if served:
        raise RuntimeError(
            f"hazma._core serves {len(served)} kernel(s) "
            f"({', '.join(served[:3])}...): this tree runs Rust kernels. "
            "The parity corpus must only ever be generated from pre-port "
            "Cython (projects/cython-to-rust/rules.md, rule 2)."
        )


def assert_unconsumed_exports_are_unimported() -> None:
    """Prove the excluded entry points still have no importers.

    The corpus deliberately omits the two ``sigma_xx_to_all`` exports
    because nothing in `hazma` imports them, so there is no consumed
    behavior to pin. That is a property of the tree, not a fact to
    inherit from the inventory snapshot, so it is re-derived here: if
    either name ever acquires an importer, generation fails and the
    corpus has to grow.
    """
    package = REPO_ROOT / "hazma"
    # A name can be defined in more than one module (both cross-section
    # modules export ``sigma_xx_to_all``), so every defining file is
    # exempt from every name's search, not just its own.
    defining_files = {
        package.joinpath(*module.split(".")[1:]).with_suffix(".pyx")
        for module in UNCONSUMED_EXPORTS
    }
    sources = sorted(set(package.rglob("*.py")) | set(package.rglob("*.pyx")))
    for module, name in UNCONSUMED_EXPORTS.items():
        importers = [
            str(path.relative_to(REPO_ROOT))
            for path in sources
            if path not in defining_files
            and re.search(rf"\b{re.escape(name)}\b", path.read_text())
        ]
        if importers:
            raise RuntimeError(
                f"{module}:{name} was excluded from the parity corpus as "
                f"unconsumed, but is now referenced by: {', '.join(importers)}. "
                "Either add it to the corpus or drop the reference."
            )


#: Case name -> the ``.pyx`` ``module:function`` the corpus values were
#: captured from, for entry points now served by ``hazma._core``.
#:
#: A swap moves a `Case`'s `module` off the ``.pyx`` and onto the
#: pure-Python wrapper, because the wrapper is where the value the user
#: gets now comes from — pointing the case at the twin would leave the
#: gate measuring the implementation the swap replaced. That move breaks
#: the identity `assert_full_coverage` compares on, so the origin is
#: recorded here instead of being lost: the walk still knows that
#: ``spectra.positron.muon`` is the pin for what
#: ``hazma/spectra/_positron/_muon.pyx`` used to export.
#:
#: One row per swapped entry point, added by the swapping task. Rows for
#: a capi survivor (`hazma/spectra/_positron/_muon.pyx`, whose ``cdef``s
#: outlive its ``def``) and for a fully deleted twin look the same; the
#: difference is only whether the ``.pyx`` is still on disk.
PORTED_ENTRY_POINTS: dict[str, tuple[str, str]] = {
    # cython-to-rust Task 4.1.
    "spectra.positron.muon": ("hazma.spectra._positron._muon", "dnde_positron_muon"),
    # cython-to-rust Task 4.2 -- the tabulated photon family. Unlike the
    # entry above, these five ``.pyx`` are gone from the tree entirely
    # rather than surviving as capi providers, so their rows here are the
    # only record of where the pinned values came from.
    "spectra.photon.charged_kaon": (
        "hazma.spectra._photon._kaon",
        "dnde_photon_charged_kaon",
    ),
    "spectra.photon.long_kaon": (
        "hazma.spectra._photon._kaon",
        "dnde_photon_long_kaon",
    ),
    "spectra.photon.short_kaon": (
        "hazma.spectra._photon._kaon",
        "dnde_photon_short_kaon",
    ),
    "spectra.photon.eta": ("hazma.spectra._photon._eta", "dnde_photon_eta"),
    "spectra.photon.eta_prime": (
        "hazma.spectra._photon._eta_prime",
        "dnde_photon_eta_prime",
    ),
    "spectra.photon.omega": ("hazma.spectra._photon._omega", "dnde_photon_omega"),
    "spectra.photon.phi": ("hazma.spectra._photon._phi", "dnde_photon_phi"),
    # cython-to-rust Task 4.3. Like the positron muon above and unlike
    # the tabulated family, this ``.pyx`` survives as a capi provider —
    # its ``def`` is gone and its two ``cdef``s are not.
    "spectra.photon.muon": ("hazma.spectra._photon._muon", "dnde_photon"),
}


def assert_full_coverage(cases: dict[str, Case]) -> None:
    """Check the corpus covers every consumed public ``def`` in the tree.

    Walks the surviving ``.pyx`` for top-level ``def``s, subtracts
    `UNCONSUMED_EXPORTS`, and compares against the entry points the
    corpus declares. This is what keeps the corpus honest as the port
    deletes Cython modules: a `Case` naming a module that no longer
    exists, or a ``def`` nobody pinned, both fail here.

    A case listed in `PORTED_ENTRY_POINTS` is compared on its recorded
    ``.pyx`` origin rather than on its live module, so a swap neither
    reads as a lost pin nor as a case pointing at nothing. It stays an
    error for the origin to still export the ``def``: a swap that
    repoints the wrapper but leaves the Cython entry point in place has
    left two implementations reachable, which is the drift window
    ``projects/cython-to-rust/rules.md`` rule 1 exists to close.
    """
    covered: set[tuple[str, str]] = set()
    for name, case in cases.items():
        covered.add(PORTED_ENTRY_POINTS.get(name, (case.module, case.function)))

    declared: set[tuple[str, str]] = set()
    for path in sorted((REPO_ROOT / "hazma").rglob("*.pyx")):
        module = ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)
        for match in re.finditer(r"^def\s+(\w+)\s*\(", path.read_text(), re.MULTILINE):
            declared.add((module, match.group(1)))

    surviving_twins = sorted(set(PORTED_ENTRY_POINTS.values()) & declared)
    if surviving_twins:
        raise RuntimeError(
            "these entry points are served by hazma._core but their Cython "
            f"def is still exported: {surviving_twins}"
        )

    excluded = set(UNCONSUMED_EXPORTS.items())
    missing = declared - covered - excluded
    stale = covered - declared - set(PORTED_ENTRY_POINTS.values())
    if missing or stale:
        raise RuntimeError(
            "parity corpus coverage mismatch.\n"
            f"  public defs with no corpus case: {sorted(missing)}\n"
            f"  corpus cases with no public def: {sorted(stale)}"
        )
