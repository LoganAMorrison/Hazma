"""``hazma._core.neutrino`` — both spectra, and the ``(3, N)`` return shape.

cython-to-rust Phase 04 Task 4.6, which closes the phase. This is the one
module in the phase whose kernels return **three** numbers per energy, so
it is also the only end-to-end exercise of
``crate::dispatch::map_flavors`` outside ``test/test_core_dispatch.py``'s
``roundtrip_flavors`` probe.

Shaped after ``test/test_core_photon_rho.py`` rather than after
``test/test_core_positron_pion.py``, and the choice is forced: nothing
outside ``hazma/spectra/_neutrino/`` cimported ``_muon.pyx``, ``_pion.pyx``
or ``_neutrino.pyx``, so ``projects/cython-to-rust/rules.md`` rule 1
applies without its capi exception and **all three files went in this
swap's PR**. There is no twin left to call, so the against-the-Cython
evidence is the parity corpus plus the direct comparison run *before* the
deletion — see "What replaced the Cython oracle" below.

Five parts:

1. :class:`TestDispatchWiring` — one assertion per contract branch, for
   both entry points, including the two shapes only these kernels have.
2. :class:`TestFlavorSelection` — the wrapper's ``flavor=`` argument,
   which is the only place the row order is user-visible.
3. :class:`TestWrapperAndPublicApi` — the swap wired out to what users
   import, and the twins gone from the tree.
4. :class:`TestAgainstAnIndependentReference` — both spectra recomputed
   in Python: the muon's closed forms transcribed from
   ``hep-ph/9909265``-era Michel algebra, and the pion's boost integral
   with ``scipy.integrate.quad`` over the *ported* muon kernel. A genuine
   second opinion, using a different QUADPACK binding for the pion.
5. :class:`TestPhysics` — statements about the spectra that outlive the
   Cython, including the two declared defects.

What replaced the Cython oracle
-------------------------------
Measured against the live twins on this tree before the three ``.pyx``
were deleted, at the parent energies the corpus samples and at eight it
does not:

* ``dnde_neutrino_muon`` was **bit-equal at every one of the 3,795 values
  the corpus pins**, and at all 9,600 points of a denser off-corpus sweep.
  It is the phase's second `EXACT`-class kernel after the positron muon,
  and ``test/parity/tolerances.py`` keeps it at ``rtol = 0``;
* ``dnde_neutrino_charged_pion`` moved by at most **9.7e-16** relative
  over its 4,185 pinned values, 3,793 of them bit-equal, and by
  **2.3e-14** on the off-corpus sweep. Task 4.6 tightened its budget from
  ``QUAD_RTOL`` (1e-8) to ``PORTED_QUAD_RTOL`` (1e-12) on those numbers.

Two declared defects
--------------------
Both are live in hazma 2.1.0, both are reproduced on purpose (rule 1: the
corpus pins them), and both are asserted below rather than described.

* **The ``pi -> e nu`` line is counted twice.** ``_pion.pyx`` sums
  ``c_dnde_mu_numu_point`` and ``c_dnde_e_nue_point``, and *both* add the
  boosted electron-neutrino line. The overweight is one extra ``BR_e``
  (1.23e-4) per pion, landing as a 0.03-0.06% excess on the
  electron-neutrino plateau where the line sits. Tracked in
  ``docs/followups/todo/neutrino-pion-electron-line-counted-twice.md``.
* **A pion at rest loses both prompt lines.** The ``E - m < DBL_EPSILON``
  branch returns only the muon-decay continuum, because a delta function
  in the rest frame has no representation here. Not filed separately: it
  is the same "the rest-frame branch is not the limit of the boosted one"
  family as the rho's, and unlike the rho's it does not change units.

What is **not** a defect, and must not be "fixed": ``_neutrino/_muon.pyx``
applies the Michel normalization the right way round, so both its rows
integrate to exactly one neutrino. Its ``_positron/_muon.pyx`` sibling
divides where it should multiply and is low by 0.0374%
(``docs/followups/todo/positron-muon-spectrum-normalization-inverted.md``).
The two files really do disagree and only one of them is wrong;
:class:`TestPhysics` pins both sides of that.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.integrate import quad

from hazma import spectra
from hazma._core import neutrino as core_neutrino
from hazma.spectra import _nbody
from hazma.spectra import _neutrino as wrapper

if TYPE_CHECKING:
    from collections.abc import Callable

    #: What the `entry` fixture hands a dispatch test: one of the two
    #: `hazma._core.neutrino` entry points, and a parent energy in support.
    EntryPoint = tuple[Callable[..., object], float]

REPO_ROOT = Path(__file__).resolve().parents[1]

dnde_muon = core_neutrino.dnde_neutrino_muon
dnde_pion = core_neutrino.dnde_neutrino_charged_pion

QUANTITY = "Neutrino energies"
DIMENSION_ERROR = f"{QUANTITY} must be 0 or 1-dimensional."
TYPE_ERROR = f"{QUANTITY} must be a float or a NumPy array."

#: `hazma/_utils/constants.pxd`, the table these kernels' `.pyx` `include`d.
#: Spelled out rather than imported from `hazma.parameters` so a future
#: consolidation of the two tables cannot silently move the tests with the
#: code (`projects/cython-to-rust/rules.md` rule 4).
MASS_E = 0.5109989461
MASS_MU = 105.6583745
MASS_PI = 139.57039
BR_PI_TO_MU_NUMU = 0.9998770
BR_PI_TO_E_NUE = 1.230e-4

#: The Michel normalization, `1 / (1 - 8r^2 + 8r^6 - r^8 - 12 r^4 ln(r^2))`.
#: `_neutrino/_muon.pyx` multiplies by it, which is correct.
R = MASS_E / MASS_MU
R_FACTOR = 1.0001870858234163

#: Neutrino flavors, in the row order the `(3, N)` return shape publishes.
FLAVORS = ("e", "mu", "tau")
N_FLAVORS = len(FLAVORS)


def two_body_energy(q: float, m1: float, m2: float) -> float:
    """``hazma/_utils/kinematics.pxd``'s helper, in Python.

    ``E1 = (q^2 + m1^2 - m2^2) / (2 q)`` — the energy of particle 1 in the
    rest frame of a parent of mass ``q`` decaying to ``m1 + m2``, MeV.
    """
    return (q * q + m1 * m1 - m2 * m2) / (2.0 * q)


#: The muon's energy in the charged-pion rest frame, MeV, and the second
#: argument the pion's boost integrand passes to the muon spectrum.
ENG_MU_PI_RF = two_body_energy(MASS_PI, MASS_MU, 0.0)
#: The prompt `pi -> e nu_e` neutrino energy in the pion rest frame, MeV.
ENU_E_PI_RF = two_body_energy(MASS_PI, 0.0, MASS_E)
#: The prompt `pi -> mu nu_mu` neutrino energy in the pion rest frame, MeV.
ENU_MU_PI_RF = two_body_energy(MASS_PI, 0.0, MASS_MU)

#: Muon and pion energies the sweeps use: rest, one step off rest, and
#: increasing boosts.
MUON_ENERGIES = (MASS_MU, MASS_MU + 1e-9, 110.0, 150.0, 500.0, 1500.0)
PION_ENERGIES = (MASS_PI, MASS_PI + 1e-9, 145.0, 200.0, 500.0, 1500.0)

#: How far the port may sit from the independent Python reference below.
#:
#: For the muon that is a *transcription* comparison over the same closed
#: forms in the same order, so the only slack is the FMAs the shipped
#: Cython contracts and the reference cannot spell: measured at 3e-16
#: relative on this tree, and 1e-13 is the same figure
#: ``test/parity/tolerances.py`` gives its `SPECFUN` class for the same
#: kind of last-bit difference.
#:
#: For the pion it is a genuinely different integrator — scipy's QUADPACK
#: binding against the in-tree port — over the same integrand. Phase 03
#: Task 3.3's envelope for two converged QUADPACK runs is 8.2e-11
#: relative; 1e-9 sits an order of magnitude outside it, which is what
#: leaves the bound a statement about the algorithm rather than about one
#: platform's bisection decisions. The coarsest thing that can go wrong
#: here — a swapped flavor row, a dropped branching fraction — moves the
#: answer by O(1). Measured on these grids: 2.5e-15 worst, so the slack
#: is deliberate rather than needed.
MUON_REFERENCE_BUDGET = 1e-13
PION_REFERENCE_BUDGET = 1e-9

#: The muon budget for a parent within one part in 1e8 of rest, where the
#: closed form is ill-conditioned rather than the port inaccurate.
#:
#: The boosted expression carries a ``1/(2 beta)`` prefactor against a
#: bracket that vanishes like ``beta``, so a last-bit difference anywhere
#: inside arrives amplified by ``1/(2 beta)``. At
#: ``emu = m_mu + 1e-9`` MeV that is ``beta = 1.4e-6`` and an amplification
#: of 3.6e5, i.e. ``2.2e-16 x 3.6e5 = 7.9e-11`` of headroom needed on
#: arithmetic grounds alone. Measured here at **4.0e-12**, so 1e-10 sits
#: 25x above the measurement and inside the derivation. Every other parent
#: energy in :data:`MUON_ENERGIES` needs 6e-15 or less.
#:
#: The same amplification is why ``test/parity/tolerances.py`` carries
#: `PLATFORM_SPECFUN_RTOL`, and why
#: ``test/test_core_positron_muon.py``'s off-platform table peaks at its
#: own just-off-rest row.
NEAR_REST_BUDGET = 1e-10

#: The parents :data:`NEAR_REST_BUDGET` applies to: those within
#: 1e-8 MeV of the muon rest mass.
NEAR_REST_TOLERANCE = 1e-8


def muon_budget(emu: float) -> float:
    """The budget appropriate to this parent energy; see above."""
    if MASS_MU < emu < MASS_MU + NEAR_REST_TOLERANCE:
        return NEAR_REST_BUDGET
    return MUON_REFERENCE_BUDGET


def reference_dnde_neutrino_muon(enu: float, emu: float) -> np.ndarray:
    """An independent transcription of the muon neutrino spectra.

    Written from the Michel algebra rather than from the Rust: the
    rest-frame ``dN/dx`` for each flavor, and in flight the same
    polynomials integrated over the boost cone in closed form between
    ``x_-`` and ``x_+``. Returns the three flavors in row order, MeV^-1.
    """
    zero = np.zeros(N_FLAVORS)
    if emu < MASS_MU:
        return zero

    r2 = R * R
    r4, r6 = r2 * r2, r2 * r2 * r2
    xmax = 1.0 - r2

    if emu - MASS_MU < np.finfo(np.float64).eps:
        pre = 2.0 / MASS_MU
        x = pre * enu
        if x <= 0.0 or x >= xmax:
            return zero
        xm = 1.0 - x
        common = R_FACTOR * x * x * (xmax - x) ** 2 / xm
        dndxe = 12.0 * common
        dndxm = (
            2.0 * common * (3.0 + r2 * (3.0 - x) - 5.0 * x + 2.0 * x * x) / (xm * xm)
        )
        return np.array([pre * dndxe, pre * dndxm, 0.0])

    e_to_x = 2.0 / emu
    x = e_to_x * enu
    gam = emu / MASS_MU
    beta = math.sqrt(1.0 - (MASS_MU / emu) ** 2)
    pre = R_FACTOR * e_to_x / (2.0 * beta)

    if x <= 0.0 or (1.0 + beta) * xmax <= x:
        return zero

    xm = gam**2 * x * (1.0 - beta)
    xp = min(xmax, gam**2 * x * (1.0 + beta))
    xmm, xpm = 1.0 - xm, 1.0 - xp
    log = math.log(xpm / xmm)

    electron = (
        2.0
        * pre
        * (
            (xm - xp)
            * (
                -3.0 * (xm + xp)
                + 2.0 * (3.0 * r4 + xm**2 + xm * xp + xp**2 + 3.0 * r2 * (xm + xp))
            )
            - 6.0 * r4 * log
        )
    )
    muon = pre * (
        3.0 * r2 * (xm - xp) * (xm + xp)
        + (xm**2 * (-9.0 + 4.0 * xm) + (9.0 - 4.0 * xp) * xp**2) / 3.0
        + r6 * ((-2.0 * xm) / xmm**2 + (2.0 * xp) / xpm**2)
        + 6.0 * r4 * (1.0 / xmm - 1.0 / xpm)
        + 2.0 * r4 * (-3.0 + r2) * log
    )
    return np.array([electron, muon, 0.0])


def reference_dnde_neutrino_charged_pion(enu: float, epi: float) -> np.ndarray:
    """An independent recomputation of the pion neutrino spectra.

    The boost integral is redone with ``scipy.integrate.quad`` — a
    different QUADPACK binding — over the **ported** muon kernel, which
    :class:`TestAgainstAnIndependentReference` has already checked against
    its own reference. The prompt lines are recomputed from the flat-boost
    closed form rather than from ``boost_delta_function``.

    Reproduces the double-counted ``pi -> e nu`` line on purpose: the
    point of this reference is to check the *boost*, not to disagree with
    the shipped physics, which :class:`TestPhysics` covers separately.
    """
    zero = np.zeros(N_FLAVORS)
    if epi < MASS_PI:
        return zero

    if epi - MASS_PI < np.finfo(np.float64).eps:
        rest = reference_dnde_neutrino_muon(enu, ENG_MU_PI_RF)
        return np.array([rest[0] * BR_PI_TO_MU_NUMU, rest[1] * BR_PI_TO_MU_NUMU, 0.0])

    beta = math.sqrt(1.0 - (MASS_PI / epi) ** 2)
    gamma = 1.0 / math.sqrt(1.0 - beta**2)

    def line(e0: float) -> float:
        """A massless rest-frame line at ``e0``, boosted to ``enu``."""
        lo, hi = gamma * enu * (1.0 - beta), gamma * enu * (1.0 + beta)
        return 1.0 / (2.0 * gamma * beta * e0) if lo < e0 < hi else 0.0

    emin, emax = max(0.0, enu * gamma * (1.0 - beta)), enu * gamma * (1.0 + beta)
    pre = 0.5 / (gamma * beta) * BR_PI_TO_MU_NUMU

    def continuum(row: int) -> float:
        if emin == emax:
            return 0.0
        return (
            pre
            * quad(
                lambda e: reference_dnde_neutrino_muon(e, ENG_MU_PI_RF)[row] / e,
                emin,
                emax,
            )[0]
        )

    # The `pi -> e nu` line appears once from each half of the `.pyx`.
    electron = 2.0 * BR_PI_TO_E_NUE * line(ENU_E_PI_RF) + continuum(0)
    muon = BR_PI_TO_MU_NUMU * line(ENU_MU_PI_RF) + continuum(1)
    return np.array([electron, muon, 0.0])


def reference_pion_continuum(enu: float, epi: float, row: int) -> float:
    """The pion's muon-decay continuum alone, without either prompt line.

    Recomputed with ``scipy.integrate.quad`` over the ported muon kernel,
    so subtracting it from the shipped spectrum isolates the lines using
    an integrator the code under test does not share. ``row`` is 0 for
    electron neutrinos and 1 for muon neutrinos.
    """
    beta = math.sqrt(1.0 - (MASS_PI / epi) ** 2)
    gamma = 1.0 / math.sqrt(1.0 - beta**2)
    emin, emax = max(0.0, enu * gamma * (1.0 - beta)), enu * gamma * (1.0 + beta)
    if emin == emax:
        return 0.0
    pre = 0.5 / (gamma * beta) * BR_PI_TO_MU_NUMU
    return (
        pre
        * quad(
            lambda e: reference_dnde_neutrino_muon(e, ENG_MU_PI_RF)[row] / e, emin, emax
        )[0]
    )


def reference_spectrum(
    reference: Callable[[float, float], np.ndarray],
    parent: float,
    energies: np.ndarray,
) -> np.ndarray:
    """``reference`` evaluated pointwise into a ``(3, N)`` array."""
    return np.stack([reference(float(e), parent) for e in energies], axis=1)


def assert_matches_the_reference(
    got: np.ndarray, want: np.ndarray, budget: float, context: str
) -> None:
    """Assert two ``(3, N)`` spectra agree to ``budget`` of the peak.

    ``atol`` is scaled by the peak rather than left at zero because both
    spectra pass through zero at their endpoints and at every line edge,
    and a relative bound alone is unbounded at a cancellation.
    """
    finite = np.isfinite(want)
    peak = float(np.abs(want[finite]).max()) if finite.any() else 0.0
    np.testing.assert_allclose(
        got,
        want,
        rtol=budget,
        atol=budget * peak,
        err_msg=(
            f"{context}: the port left the independent reference's budget "
            f"of {budget:.0e} x the spectrum peak ({peak:.6e})."
        ),
    )


class TestDispatchWiring:
    """Both entry points go through ``map_flavors`` with their wording.

    The branch-by-branch argument about ``map_flavors`` itself lives in
    ``test/test_core_dispatch.py``; what is specific to these kernels is
    that they reached that helper at all, with the quantity string the
    port declared, and that the two return shapes are the published ones.
    """

    @pytest.fixture(params=["muon", "charged_pion"])
    def entry(self, request: pytest.FixtureRequest) -> EntryPoint:
        """One entry point and a parent energy comfortably in support."""
        if request.param == "muon":
            return dnde_muon, 500.0
        return dnde_pion, 500.0

    def test_a_scalar_returns_a_three_tuple_of_floats(self, entry: EntryPoint) -> None:
        dnde, parent = entry
        value = dnde(20.0, parent)
        assert type(value) is tuple
        assert len(value) == N_FLAVORS
        assert all(type(component) is float for component in value)
        assert value[0] > 0.0 and value[1] > 0.0

    def test_a_numpy_scalar_and_a_zero_dim_array_take_the_scalar_path(
        self, entry: EntryPoint
    ) -> None:
        dnde, parent = entry
        expected = dnde(20.0, parent)
        assert dnde(np.float64(20.0), parent) == expected
        assert dnde(np.array(20.0), parent) == expected
        assert type(dnde(np.array(20.0), parent)) is tuple

    def test_an_array_returns_a_fresh_three_by_n_float64_array(
        self, entry: EntryPoint
    ) -> None:
        dnde, parent = entry
        energies = np.geomspace(1.0, 400.0, 64)
        values = dnde(energies, parent)
        assert values.dtype == np.float64
        assert values.shape == (N_FLAVORS, energies.size)
        assert not np.shares_memory(values, energies)

    def test_the_array_path_agrees_with_the_scalar_path_bit_for_bit(
        self, entry: EntryPoint
    ) -> None:
        # `map_flavors` calls the same kernel either way, so a
        # broadcasting or a transposition bug shows up here and nowhere
        # else the corpus looks.
        dnde, parent = entry
        energies = np.geomspace(0.5, 600.0, 97)
        batched = dnde(energies, parent)
        one_at_a_time = np.array([dnde(float(e), parent) for e in energies]).T
        assert batched.tobytes() == one_at_a_time.tobytes()

    def test_a_sequence_is_accepted(self, entry: EntryPoint) -> None:
        dnde, parent = entry
        rows = zip(dnde(20.0, parent), dnde(30.0, parent), strict=True)
        assert dnde([20.0, 30.0], parent).tolist() == [list(row) for row in rows]

    def test_an_empty_grid_returns_three_empty_rows(self, entry: EntryPoint) -> None:
        dnde, parent = entry
        values = dnde(np.array([], dtype=np.float64), parent)
        assert values.shape == (N_FLAVORS, 0)

    def test_the_rank_message_names_the_neutrino_quantity(
        self, entry: EntryPoint
    ) -> None:
        # `hazma/spectra/_neutrino/_muon.pyx:205` said "Photon energies",
        # a copy-paste defect its `_pion.pyx` sibling did not share.
        # Task 3.5 decided the port says "Neutrino energies" for both, and
        # this is where that decision is pinned now that the `.pyx` is
        # gone -- `test/test_core_dispatch.py`'s roster no longer carries
        # either wording.
        dnde, parent = entry
        with pytest.raises(ValueError) as excinfo:
            dnde(np.ones((2, 2)), parent)
        assert str(excinfo.value) == DIMENSION_ERROR

    def test_a_non_float64_array_is_a_value_error(self, entry: EntryPoint) -> None:
        dnde, parent = entry
        with pytest.raises(ValueError, match="must be a float64 array"):
            dnde(np.arange(4), parent)

    def test_a_non_number_is_a_type_error(self, entry: EntryPoint) -> None:
        dnde, parent = entry
        with pytest.raises(TypeError) as excinfo:
            dnde(None, parent)
        assert str(excinfo.value) == TYPE_ERROR

    def test_both_arguments_are_accepted_by_keyword(self) -> None:
        # The Cython twins were `def`s and took keywords; a
        # positional-only port would be a silent public-API narrowing.
        assert dnde_muon(neutrino_energies=20.0, muon_energy=500.0) == dnde_muon(
            20.0, 500.0
        )
        assert dnde_pion(neutrino_energies=20.0, pion_energy=500.0) == dnde_pion(
            20.0, 500.0
        )


class TestFlavorSelection:
    """The wrapper's ``flavor=`` argument, and the row order behind it.

    This is the only place the ``(3, N)`` row order is user-visible, so a
    transposition or a permutation inside ``map_flavors`` surfaces here.
    """

    @pytest.mark.parametrize(("index", "flavor"), list(enumerate(FLAVORS)))
    def test_each_flavor_selects_its_own_row(self, index: int, flavor: str) -> None:
        energies = np.geomspace(1.0, 400.0, 32)
        both = wrapper.dnde_neutrino_charged_pion(energies, 500.0)
        one = wrapper.dnde_neutrino_charged_pion(energies, 500.0, flavor=flavor)
        assert one.tobytes() == both[index].tobytes()

    def test_the_rows_are_pairwise_distinguishable(self) -> None:
        # With equal rows a permutation would pass every assertion above.
        energies = np.geomspace(1.0, 400.0, 32)
        electron, muon, tau = wrapper.dnde_neutrino_charged_pion(energies, 500.0)
        assert not np.array_equal(electron, muon)
        assert not np.array_equal(electron, tau)
        assert np.array_equal(tau, np.zeros_like(tau))

    def test_an_unknown_flavor_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid flavor"):
            wrapper.dnde_neutrino_muon(20.0, 500.0, flavor="s")
        with pytest.raises(ValueError, match="Invalid flavor"):
            wrapper.dnde_neutrino_charged_pion(20.0, 500.0, flavor="s")


class TestWrapperAndPublicApi:
    """The swap is wired all the way out, and the twins are gone."""

    def test_the_private_wrappers_return_the_core_kernels_values(self) -> None:
        energies = np.geomspace(1.0, 400.0, 32)
        assert wrapper.dnde_neutrino_muon(energies, 500.0).tobytes() == (
            dnde_muon(energies, 500.0).tobytes()
        )
        assert wrapper.dnde_neutrino_charged_pion(energies, 500.0).tobytes() == (
            dnde_pion(energies, 500.0).tobytes()
        )

    def test_the_public_spectra_names_resolve_to_the_same_functions(self) -> None:
        assert spectra.dnde_neutrino_muon(20.0, 500.0) == dnde_muon(20.0, 500.0)
        assert spectra.dnde_neutrino_charged_pion(20.0, 500.0) == dnde_pion(20.0, 500.0)

    def test_the_cython_twins_are_gone_from_the_tree(self) -> None:
        """rules.md rule 1, in its unqualified form.

        Nothing outside the package cimported these three, so they were
        not capi survivors and the swap PR removed the modules outright
        rather than only their ``def``s. The four that did survive that
        way went in Phase 06 Task 6.4.

        Asserted against the **source tree and the build declaration**,
        not against importability. A stale ``_muon.cpython-*.so`` from an
        older build sits in the package directory until someone cleans it
        and imports perfectly well (``docs/agents/environment.md``), so an
        ``ImportError`` assertion tests whoever last ran ``pip install``
        rather than this change.
        """
        package = REPO_ROOT / "hazma" / "spectra" / "_neutrino"
        for stem in ("_muon", "_pion", "_neutrino"):
            for suffix in (".pyx", ".pxd", ".pyi"):
                assert not (package / f"{stem}{suffix}").exists(), f"{stem}{suffix}"
        # And they are out of the build, so no future `pip install -e .`
        # brings the extensions back.
        setup = (REPO_ROOT / "setup.py").read_text()
        assert '"spectra", "_neutrino"' not in setup

    def test_the_nbody_dispatch_table_reaches_the_ported_entry_points(self) -> None:
        # `_nbody.py` maps final-state names to spectrum functions; a swap
        # that repointed the wrappers but not this table would leave the
        # N-body path on the implementations the swap replaced.
        assert _nbody._dnde_neutrino_dict["mu"] is wrapper.dnde_neutrino_muon
        assert _nbody._dnde_neutrino_dict["pi"] is wrapper.dnde_neutrino_charged_pion

    def test_the_package_still_serves_its_tabulated_siblings(self) -> None:
        # The `_neutrino` package keeps its CSV-driven entry points, which
        # are pure Python and were never Cython. Deleting three
        # extensions from it must not have disturbed them.
        assert wrapper.dnde_neutrino_charged_kaon(20.0, 600.0).shape == (N_FLAVORS,)


class TestAgainstAnIndependentReference:
    """Both spectra, recomputed in Python.

    Not the Cython — that is gone. The muon's reference is a transcription
    of the same closed forms; the pion's redoes the boost integral with
    scipy's QUADPACK binding over the ported muon kernel, which is a
    genuinely different integrator over the same integrand.
    """

    @pytest.mark.parametrize("emu", MUON_ENERGIES)
    def test_the_muon_spectra_match_the_transcription(self, emu: float) -> None:
        energies = np.geomspace(1e-3, emu * 1.2, 401)
        assert_matches_the_reference(
            dnde_muon(energies, emu),
            reference_spectrum(reference_dnde_neutrino_muon, emu, energies),
            muon_budget(emu),
            f"muon swept grid, {emu=}",
        )

    @pytest.mark.parametrize("epi", PION_ENERGIES)
    def test_the_pion_spectra_match_the_independent_boost(self, epi: float) -> None:
        energies = np.geomspace(1e-2, epi * 1.2, 61)
        assert_matches_the_reference(
            dnde_pion(energies, epi),
            reference_spectrum(reference_dnde_neutrino_charged_pion, epi, energies),
            PION_REFERENCE_BUDGET,
            f"pion swept grid, {epi=}",
        )

    def test_the_support_is_identical_to_the_reference(self) -> None:
        """Which energies are *zero* is structural, so no budget excuses it.

        The bounds above are statements about rounding; this is the
        statement rounding cannot excuse. A port that moved a threshold or
        a boost limit by one grid point turns this red.
        """
        for emu in MUON_ENERGIES:
            energies = np.geomspace(1e-3, emu * 1.2, 401)
            got = dnde_muon(energies, emu)
            want = reference_spectrum(reference_dnde_neutrino_muon, emu, energies)
            assert np.array_equal(got == 0.0, want == 0.0), f"muon support, {emu=}"

    def test_the_reference_budgets_reject_a_real_error(self) -> None:
        """Neither bound is vacuous."""
        energies = np.geomspace(1e-3, 600.0, 201)
        want = reference_spectrum(reference_dnde_neutrino_muon, 500.0, energies)
        nudged = want.copy()
        nudged[0, np.abs(want[0]).argmax()] += 1e-10 * np.abs(want).max()
        assert_matches_the_reference(want, want, MUON_REFERENCE_BUDGET, "unperturbed")
        with pytest.raises(AssertionError):
            assert_matches_the_reference(
                nudged, want, MUON_REFERENCE_BUDGET, "perturbed"
            )

    def test_a_permuted_row_order_is_caught(self) -> None:
        """The comparison would not pass a transposed result.

        Worth stating because the ``(3, N)`` shape is the one place a
        silent permutation could hide: the two non-zero rows have the same
        support and the same order of magnitude.
        """
        energies = np.geomspace(1e-3, 600.0, 201)
        want = reference_spectrum(reference_dnde_neutrino_muon, 500.0, energies)
        swapped = want[[1, 0, 2]]
        with pytest.raises(AssertionError):
            assert_matches_the_reference(
                swapped, want, MUON_REFERENCE_BUDGET, "swapped"
            )


class TestPhysics:
    """Statements about the spectra, not about the code they replaced."""

    def test_the_spectra_vanish_below_their_parent_thresholds(self) -> None:
        assert dnde_muon(20.0, MASS_MU * 0.999) == (0.0, 0.0, 0.0)
        assert dnde_pion(20.0, MASS_PI * 0.999) == (0.0, 0.0, 0.0)

    def test_no_kernel_ever_makes_a_tau_neutrino(self) -> None:
        # The row exists because the return shape is `(3, N)`, not because
        # a pion or a muon makes tau neutrinos at these energies.
        for dnde, parent in ((dnde_muon, 500.0), (dnde_pion, 500.0)):
            energies = np.geomspace(1e-3, 2.0 * parent, 501)
            assert np.array_equal(dnde(energies, parent)[2], np.zeros(501))

    @pytest.mark.parametrize("emu", [MASS_MU, 150.0, 500.0, 1500.0])
    def test_the_muon_emits_exactly_one_neutrino_of_each_flavor(
        self, emu: float
    ) -> None:
        """``int dN/dE dE = 1`` for both rows, at every parent energy.

        Two statements at once. That the integral does not depend on
        ``emu`` is the physics — the boost moves neutrinos around in
        energy but creates none. That it is **1** rather than ``1/N**2`` is
        the thing a reader coming from ``_positron/_muon.pyx`` must not
        "fix": that file divides by the Michel normalization where it
        should multiply and is low by 0.0374%, and this one multiplies.

        Trapezoid on 200_001 points. 1e-5 relative: the in-flight spectrum
        has a kink where ``x_+`` meets its ``1 - r^2`` clip, and a
        composite rule of this order gets no closer — measured, not
        chosen. The positron file's deficit is 3.7e-4, so the bound
        separates the two by a factor of 37.
        """
        beta = math.sqrt(1.0 - (MASS_MU / emu) ** 2) if emu > MASS_MU else 0.0
        endpoint = (1.0 + beta) * (1.0 - R * R) * emu / 2.0
        energies = np.linspace(0.0, endpoint, 200_001)
        values = dnde_muon(energies, emu)
        for row, flavor in enumerate(("electron", "muon")):
            integral = np.trapezoid(values[row], energies)
            assert integral == pytest.approx(1.0, rel=1e-5), flavor
        # And the positron sibling's defect is well outside that bound.
        assert abs(1.0 - 1.0 / R_FACTOR**2) > 30.0 * 1e-5

    def test_the_pion_yields_one_muon_neutrino_from_each_of_two_sources(self) -> None:
        """The muon-neutrino row integrates to ``2 BR_mu``.

        A charged pion makes a prompt ``nu_mu`` and then the muon makes
        another, so the row carries two per pion. The electron-neutrino
        row carries one from the muon plus the doubled prompt line, which
        the next test isolates.

        Trapezoid on 20_001 points to past the highest endpoint. Each
        point costs two adaptive quadratures, so the grid is short and the
        budget is 1e-3 relative: the spectrum has step discontinuities
        where each line's window opens and closes, which a composite rule
        resolves at ``O(h)``.
        """
        epi = 400.0
        gamma = epi / MASS_PI
        beta = math.sqrt(1.0 - (MASS_PI / epi) ** 2)
        energies = np.linspace(0.0, gamma * (1.0 + beta) * ENU_E_PI_RF * 1.02, 20_001)
        values = dnde_pion(energies, epi)
        assert np.trapezoid(values[1], energies) == pytest.approx(
            2.0 * BR_PI_TO_MU_NUMU, rel=1e-3
        )
        assert np.trapezoid(values[0], energies) == pytest.approx(
            BR_PI_TO_MU_NUMU + 2.0 * BR_PI_TO_E_NUE, rel=1e-3
        )

    def test_the_electron_line_is_counted_twice_and_the_muon_line_once(self) -> None:
        """The declared defect, asserted rather than described.

        Each boosted prompt line is a flat plateau of height
        ``BR / (2 gamma beta E_rf)`` across the lab energies whose boost
        window straddles ``E_rf``. Subtracting the muon-decay continuum —
        recomputed here with scipy over the ported muon kernel, so the
        subtraction owes nothing to the code under test — leaves exactly
        the plateau.

        What comes out is **2.0000** copies of the ``pi -> e nu`` line and
        **1.0000** of the ``pi -> mu nu`` one, at every energy inside both
        windows. The electron excess is the defect: ``_pion.pyx`` sums
        ``c_dnde_mu_numu_point`` and ``c_dnde_e_nue_point`` and both add
        it. The muon row has no second copy because ``c_dnde_e_nue_point``
        writes nothing there, which is what makes the pair of ratios a
        discriminating test rather than a scale check.

        1e-6 relative, which is what the subtraction supports: the
        continuum is recomputed to scipy's default 1.49e-8 absolute.
        """
        epi = 400.0
        gamma = epi / MASS_PI
        beta = math.sqrt(1.0 - (MASS_PI / epi) ** 2)
        electron_line = BR_PI_TO_E_NUE / (2.0 * gamma * beta * ENU_E_PI_RF)
        muon_line = BR_PI_TO_MU_NUMU / (2.0 * gamma * beta * ENU_MU_PI_RF)

        # Inside both lines' windows. A lab energy `E` sees a rest-frame
        # line at `E_rf` when `gamma E (1-beta) < E_rf < gamma E (1+beta)`,
        # which at `E_pi = 400` MeV is `E in (12.6, 408)` for the electron
        # line and `E in (5.4, 174)` for the muon one; the three below sit
        # in the overlap.
        for enu in (20.0, 30.0, 50.0):
            total = dnde_pion(enu, epi)
            assert total[0] - reference_pion_continuum(enu, epi, 0) == pytest.approx(
                2.0 * electron_line, rel=1e-6
            ), f"the pi -> e nu line is not doubled at {enu=}"
            assert total[1] - reference_pion_continuum(enu, epi, 1) == pytest.approx(
                muon_line, rel=1e-6
            ), f"the pi -> mu nu line is not single at {enu=}"

    def test_a_pion_at_rest_keeps_only_the_muon_continuum(self) -> None:
        """The second declared defect.

        At ``E_pi = m_pi`` the two prompt lines are delta functions with
        no rest-frame representation, so the branch returns only the
        muon-decay continuum weighted by ``BR(pi -> mu nu)``. The muon
        row therefore integrates to ``BR_mu`` rather than to ``2 BR_mu``,
        which is the discontinuity in the parent energy this records.
        """
        energies = np.linspace(0.0, ENG_MU_PI_RF, 200_001)
        at_rest = dnde_pion(energies, MASS_PI)
        assert np.trapezoid(at_rest[1], energies) == pytest.approx(
            BR_PI_TO_MU_NUMU, rel=1e-5
        )
        # The electron row loses the prompt line too, so it carries only
        # the muon's nu_e_bar rather than that plus two copies of it.
        assert np.trapezoid(at_rest[0], energies) == pytest.approx(
            BR_PI_TO_MU_NUMU, rel=1e-5
        )
        # A boosted pion carries twice as many muon neutrinos, which
        # `test_the_pion_yields_one_muon_neutrino_from_each_of_two_sources`
        # pins -- so the at-rest branch is a step of a whole prompt line,
        # not a rounding. The discontinuity is not observable by
        # integrating a *nearly*-at-rest pion: the prompt lines are then
        # plateaus a few times `beta E` wide -- 1e-6 MeV at one ulp above
        # `m_pi` -- and no practical grid resolves them.
        assert np.trapezoid(at_rest[1], energies) < 0.6 * (2.0 * BR_PI_TO_MU_NUMU)

    def test_a_zero_energy_neutrino_is_zero_rather_than_nan(self) -> None:
        """The boost window collapses onto the origin, where ``0/0`` lives.

        ``scipy.integrate.quad`` short-circuits ``a == b`` to zero without
        evaluating the integrand, so the Cython never met that ``NaN``.
        Task 4.6 added the same short circuit to ``crate::quad::quad``;
        this is its live call site, and no corpus grid samples it.
        """
        assert dnde_pion(0.0, 400.0) == (0.0, 0.0, 0.0)
        assert dnde_muon(0.0, 400.0) == (0.0, 0.0, 0.0)

    @pytest.mark.parametrize("parent", [150.0, 500.0, 1500.0])
    def test_both_spectra_are_finite_and_non_negative(self, parent: float) -> None:
        energies = np.geomspace(1e-3, parent * 2.0, 601)
        for values in (dnde_muon(energies, parent), dnde_pion(energies, parent)):
            assert np.all(np.isfinite(values))
            assert np.all(values >= 0.0)

    def test_a_faster_parent_spreads_the_same_neutrinos_over_more_energy(self) -> None:
        """The peak falls as the parent is boosted, at fixed total.

        A statement the corpus cannot make, because it compares each
        parent energy only against itself.
        """
        peaks = [
            float(dnde_muon(np.geomspace(1e-3, 2.0 * emu, 2001), emu)[1].max())
            for emu in (150.0, 500.0, 1500.0)
        ]
        assert peaks[0] > peaks[1] > peaks[2]
