"""Model-layer aggregation contract for the concrete ``Theory`` models.

The parity corpus under ``test/parity/`` pins the *compiled* layer: the 41
``.pyx`` entry points, at bit-equality against the capturing tree. It stops
where Cython stops. Everything above it — assembling per-channel cross
sections into a dict with a ``"total"``, dividing that into branching
fractions, weighting each spectrum by its branching fraction, and attaching a
branching fraction to a line — is pure Python in
:mod:`hazma.theory` and the two mediator packages, and no corpus case reaches
it.

This module covers that layer, and it covers it with *identities* rather than
stored reference arrays. Each assertion below is a relation the aggregation
must satisfy for any model at any energy (``total`` is the channel sum;
a branching fraction is a cross-section ratio; a spectrum is its branching
fraction times its channel spectrum), plus three two-body kinematic
quantities pinned to their closed forms. Identities need no golden data, so
they cannot rot the way the ``.npy`` corpora they replace did, and they hold
bit-for-bit on every platform — unlike the parity corpus, which is scoped to
its capturing platform (see ``test/parity/README.md``).

What this deliberately does *not* do is re-pin the numbers the kernels
produce. That is the corpus's job, and duplicating it at a loose tolerance
would only create a second, weaker gate to keep in sync.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from hazma.parameters import neutral_pion_mass, vh
from hazma.scalar_mediator import HeavyQuark, ScalarMediator
from hazma.vector_mediator import KineticMixing, QuarksOnly, VectorMediator

if TYPE_CHECKING:
    from _pytest.mark.structures import ParameterSet

    from hazma.theory import TheoryAnn

# A dark-matter pair at rest annihilates at e_cm = 2 mx (1 + v_rel^2 / 2);
# v_rel = 1e-3 is the Milky Way halo value the model docs use throughout.
V_REL = 1e-3

# Higgs-portal couplings for the scalar mediator: gsff = sin(theta),
# gsGG = 3 sin(theta), gsFF = -5/6 sin(theta), lam = v_h.
S_THETA = 1e-3


def e_cm_halo(mx: float) -> float:
    """Center-of-mass energy in MeV for a halo-velocity annihilation."""
    return 2.0 * mx * (1.0 + 0.5 * V_REL**2)


def _higgs_portal(mx: float, ms: float) -> ScalarMediator:
    return ScalarMediator(
        mx=mx,
        ms=ms,
        gsxx=1.0,
        gsff=S_THETA,
        gsGG=3.0 * S_THETA,
        gsFF=-5.0 / 6.0 * S_THETA,
        lam=vh,
    )


def _models() -> list[ParameterSet]:
    """The four model points this module runs every identity against.

    Two per model class, straddling the mediator threshold: one with the
    mediator light enough that the ``s s`` / ``v v`` channel is open (so the
    branching fractions are dominated by it), one with it closed (so the
    open channels carry the whole spectrum). The vector pair also flips
    ``gvdd``, which changes the sign of the isospin combination feeding
    ``pi pi``.
    """
    return [
        pytest.param(_higgs_portal(250.0, 125.0), id="scalar-mediator-open"),
        pytest.param(_higgs_portal(250.0, 550.0), id="scalar-mediator-closed"),
        pytest.param(
            KineticMixing(mx=125.0, mv=125.0, gvxx=1.0, eps=0.1),
            id="vector-mediator-open",
        ),
        pytest.param(
            VectorMediator(
                mx=125.0,
                mv=550.0,
                gvxx=1.0,
                gvuu=1.0,
                gvdd=-1.0,
                gvss=1.0,
                gvee=1.0,
                gvmumu=1.0,
            ),
            id="vector-mediator-closed",
        ),
    ]


MODELS = _models()


def photon_energies(model: TheoryAnn) -> np.ndarray:
    """Photon energies spanning 1 MeV to the annihilation energy."""
    return np.geomspace(1.0, e_cm_halo(model.mx), 10)


def positron_energies(model: TheoryAnn) -> np.ndarray:
    """Positron energies spanning 1 MeV to the annihilation energy.

    The lower edge is 1 MeV rather than the electron mass on purpose. Both
    mediator positron kernels return ``nan`` at exactly ``0.510998928`` — the
    legacy ``MASS_E`` in ``hazma/_utils/legacy_parameters.pxd``, against
    ``0.5109989461`` everywhere else — and that kernel edge belongs to
    ``docs/followups/todo/positron-spectrum-nan-at-legacy-electron-mass.md``,
    not to this module, which tests the aggregation above it.
    """
    return np.geomspace(1.0, e_cm_halo(model.mx), 10)


def channels(spectra_like: dict[str, object]) -> list[str]:
    """The per-channel keys of a ``spectra``-shaped dict, without ``total``."""
    return [key for key in spectra_like if key != "total"]


# --------------------------------------------------------------------------
# Cross sections and branching fractions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
def test_cross_section_keys_are_the_final_states_plus_total(model: TheoryAnn) -> None:
    """``annihilation_cross_sections`` covers exactly the advertised states."""
    e_cm = e_cm_halo(model.mx)
    sigmas = model.annihilation_cross_sections(e_cm)

    assert list(sigmas) == model.list_annihilation_final_states() + ["total"]


@pytest.mark.parametrize("model", MODELS)
def test_total_cross_section_is_the_channel_sum(model: TheoryAnn) -> None:
    """``total`` is the sum of the channels, to the bit.

    ``Theory.annihilation_cross_sections`` builds ``total`` with ``sum()`` over
    the same dict in the same order, so this is exact, not approximate. A
    channel that stops being summed in — the failure this guards against
    during the Phase 04-06 swaps — moves ``total`` by far more than a ulp.
    """
    e_cm = e_cm_halo(model.mx)
    sigmas = model.annihilation_cross_sections(e_cm)

    assert sigmas["total"] == sum(sigmas[fs] for fs in channels(sigmas))


@pytest.mark.parametrize("model", MODELS)
def test_branching_fractions_are_cross_section_ratios(model: TheoryAnn) -> None:
    """Each branching fraction is its channel's share of the total.

    Exact: ``annihilation_branching_fractions`` performs this very division,
    so any difference means the two methods disagree about the cross sections
    themselves.
    """
    e_cm = e_cm_halo(model.mx)
    sigmas = model.annihilation_cross_sections(e_cm)
    bfs = model.annihilation_branching_fractions(e_cm)

    assert list(bfs) == model.list_annihilation_final_states()
    for fs, bf in bfs.items():
        assert bf == sigmas[fs] / sigmas["total"], fs


@pytest.mark.parametrize("model", MODELS)
def test_branching_fractions_sum_to_one(model: TheoryAnn) -> None:
    """The branching fractions are normalized.

    ``rtol=1e-15`` — a handful of ulp (2.2e-16 each) is the most that summing
    six ratios in a different order than the total was accumulated can cost.
    Anything larger is a normalization bug, not rounding.
    """
    e_cm = e_cm_halo(model.mx)
    bfs = model.annihilation_branching_fractions(e_cm)

    assert_allclose(sum(bfs.values()), 1.0, rtol=1e-15, atol=0.0)


@pytest.mark.parametrize("model", MODELS)
def test_branching_fractions_vanish_when_every_channel_is_closed(
    model: TheoryAnn,
) -> None:
    """Below every threshold the total vanishes and the ratio is not taken.

    The zero-total branch of ``annihilation_branching_fractions`` is the one
    place the method can raise instead of returning, so it gets its own test.
    """
    e_cm = 1e-3  # MeV: below twice the electron mass, so nothing is open.
    sigmas = model.annihilation_cross_sections(e_cm)
    bfs = model.annihilation_branching_fractions(e_cm)

    assert sigmas["total"] == 0.0
    assert list(bfs) == model.list_annihilation_final_states()
    assert set(bfs.values()) == {0.0}


@pytest.mark.parametrize("model", MODELS)
def test_partial_widths_total_is_the_channel_sum(model: TheoryAnn) -> None:
    """The mediator's total width is the sum of its partial widths.

    ``rtol=1e-15`` rather than exact equality: ``partial_widths`` accumulates
    ``total`` with a ``+`` chain while this test uses ``sum()`` from ``0``,
    and the two orders differ by one ulp (measured: 1.4e-16 for the
    ``ms = 550 MeV`` point).
    """
    widths = model.partial_widths()

    assert_allclose(
        widths["total"],
        sum(widths[key] for key in channels(widths)),
        rtol=1e-15,
        atol=0.0,
    )


# --------------------------------------------------------------------------
# Spectra
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
def test_spectra_are_branching_fraction_weighted(model: TheoryAnn) -> None:
    """Each channel's spectrum is its branching fraction times its kernel.

    This is the identity that ties the pure-Python aggregation to the
    compiled kernels the parity corpus pins: get the weighting wrong during a
    Phase 04-06 swap and the corpus still passes while every published
    spectrum moves.
    """
    e_cm = e_cm_halo(model.mx)
    e_gams = photon_energies(model)
    bfs = model.annihilation_branching_fractions(e_cm)
    specs = model.spectra(e_gams, e_cm)
    dnde_fns = model.spectrum_funcs()

    for fs in channels(specs):
        expected = bfs[fs] * np.asarray(dnde_fns[fs](e_gams, e_cm))
        assert_array_equal(np.asarray(specs[fs]), expected, err_msg=fs)


@pytest.mark.parametrize("model", MODELS)
def test_spectra_total_is_the_channel_sum(model: TheoryAnn) -> None:
    """The total photon spectrum is the sum of the channel spectra, to the bit."""
    e_cm = e_cm_halo(model.mx)
    e_gams = photon_energies(model)
    specs = model.spectra(e_gams, e_cm)

    expected = sum(np.asarray(specs[fs]) for fs in channels(specs))
    assert_array_equal(np.asarray(specs["total"]), expected)


@pytest.mark.parametrize("model", MODELS)
def test_total_spectrum_matches_the_spectra_total(model: TheoryAnn) -> None:
    """``total_spectrum`` is the ``"total"`` entry of ``spectra``."""
    e_cm = e_cm_halo(model.mx)
    e_gams = photon_energies(model)

    assert_array_equal(
        np.asarray(model.total_spectrum(e_gams, e_cm)),
        np.asarray(model.spectra(e_gams, e_cm)["total"]),
    )


@pytest.mark.parametrize("model", MODELS)
def test_closed_channels_contribute_no_spectrum(model: TheoryAnn) -> None:
    """A channel with a vanishing branching fraction contributes exactly zero.

    ``spectrum_funcs`` wraps each kernel so it short-circuits to zeros when
    the channel's cross section is not positive; ``spectra`` then skips it
    entirely. Both paths must produce zeros of the right shape rather than
    ``nan`` from a kernel evaluated outside its support.
    """
    e_cm = e_cm_halo(model.mx)
    e_gams = photon_energies(model)
    bfs = model.annihilation_branching_fractions(e_cm)
    specs = model.spectra(e_gams, e_cm)

    closed = [fs for fs in channels(specs) if bfs[fs] == 0.0]
    for fs in closed:
        assert_array_equal(np.asarray(specs[fs]), np.zeros(e_gams.shape), err_msg=fs)


@pytest.mark.parametrize("model", MODELS)
def test_positron_spectra_are_branching_fraction_weighted(model: TheoryAnn) -> None:
    """The positron counterpart of the photon weighting identity.

    It needs its own test: the total-is-the-sum check below survives dropping
    the weight from *every* channel at once, which is exactly what a swap that
    loses the branching fraction would do.
    """
    e_cm = e_cm_halo(model.mx)
    e_ps = positron_energies(model)
    bfs = model.annihilation_branching_fractions(e_cm)
    specs = model.positron_spectra(e_ps, e_cm)
    dnde_fns = model.positron_spectrum_funcs()

    for fs in channels(specs):
        expected = bfs[fs] * np.asarray(dnde_fns[fs](e_ps, e_cm))
        assert_array_equal(np.asarray(specs[fs]), expected, err_msg=fs)


@pytest.mark.parametrize("model", MODELS)
def test_positron_spectra_total_is_the_channel_sum(model: TheoryAnn) -> None:
    """The total positron spectrum is the sum of the channel spectra."""
    e_cm = e_cm_halo(model.mx)
    e_ps = positron_energies(model)
    specs = model.positron_spectra(e_ps, e_cm)

    expected = sum(np.asarray(specs[fs]) for fs in channels(specs))
    assert_array_equal(np.asarray(specs["total"]), expected)


@pytest.mark.parametrize("model", MODELS)
def test_total_positron_spectrum_matches_the_positron_spectra_total(
    model: TheoryAnn,
) -> None:
    """``total_positron_spectrum`` is the ``"total"`` entry of ``positron_spectra``."""
    e_cm = e_cm_halo(model.mx)
    e_ps = positron_energies(model)

    assert_array_equal(
        np.asarray(model.total_positron_spectrum(e_ps, e_cm)),
        np.asarray(model.positron_spectra(e_ps, e_cm)["total"]),
    )


# --------------------------------------------------------------------------
# Lines
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
def test_lines_carry_their_channel_branching_fraction(model: TheoryAnn) -> None:
    """Every line's ``bf`` is that final state's annihilation branching fraction."""
    e_cm = e_cm_halo(model.mx)
    bfs = model.annihilation_branching_fractions(e_cm)

    lines = dict(model.gamma_ray_lines(e_cm))
    lines.update(model.positron_lines(e_cm))
    assert lines, "every model here has at least one line"

    for fs, line in lines.items():
        assert line["bf"] == bfs[fs], fs


@pytest.mark.parametrize("model", MODELS)
def test_positron_line_sits_at_half_the_annihilation_energy(model: TheoryAnn) -> None:
    """``e e`` is two equal-mass bodies, so each positron carries ``e_cm / 2``."""
    e_cm = e_cm_halo(model.mx)

    assert model.positron_lines(e_cm)["e e"]["energy"] == e_cm / 2.0


def test_scalar_gamma_ray_line_sits_at_half_the_annihilation_energy() -> None:
    """``g g`` is two massless bodies, so each photon carries ``e_cm / 2``."""
    model = _higgs_portal(250.0, 125.0)
    e_cm = e_cm_halo(model.mx)

    assert model.gamma_ray_lines(e_cm)["g g"]["energy"] == e_cm / 2.0


def test_vector_gamma_ray_line_sits_at_the_two_body_photon_energy() -> None:
    """``pi0 g`` puts the photon at ``(e_cm^2 - m_pi0^2) / (2 e_cm)``.

    The closed form for the massless body in a two-body final state. Asserted
    exactly rather than to a tolerance: the equality is not delicate here, and
    two rearrangements of the same expression (``e_cm / 2 - m^2 / (2 e_cm)``
    and ``(e_cm - m)(e_cm + m) / (2 e_cm)``) were both measured bit-identical
    at this point, so an exact assertion is not a hostage to how the
    implementation groups its terms.
    """
    model = KineticMixing(mx=125.0, mv=125.0, gvxx=1.0, eps=0.1)
    e_cm = e_cm_halo(model.mx)

    expected = (e_cm**2 - neutral_pion_mass**2) / (2.0 * e_cm)
    assert model.gamma_ray_lines(e_cm)["pi0 g"]["energy"] == expected


# --------------------------------------------------------------------------
# Final-state advertisement
# --------------------------------------------------------------------------


def test_scalar_final_states() -> None:
    """``ScalarMediator``'s six annihilation channels, and ``HeavyQuark``'s four.

    ``HeavyQuark`` drops the two that need a light-quark or lepton coupling.
    """
    assert ScalarMediator.list_annihilation_final_states() == [
        "mu mu",
        "e e",
        "g g",
        "pi0 pi0",
        "pi pi",
        "s s",
    ]
    assert HeavyQuark.list_annihilation_final_states() == [
        "g g",
        "pi0 pi0",
        "pi pi",
        "s s",
    ]


def test_vector_final_states() -> None:
    """``VectorMediator``'s six annihilation channels, and ``QuarksOnly``'s four.

    ``QuarksOnly`` drops the two leptonic ones.
    """
    assert VectorMediator.list_annihilation_final_states() == [
        "mu mu",
        "e e",
        "pi pi",
        "pi0 g",
        "pi0 v",
        "v v",
    ]
    assert KineticMixing.list_annihilation_final_states() == (
        VectorMediator.list_annihilation_final_states()
    )
    assert QuarksOnly.list_annihilation_final_states() == [
        "pi pi",
        "pi0 g",
        "pi0 v",
        "v v",
    ]


@pytest.mark.parametrize("model", MODELS)
def test_instance_and_class_final_states_agree(model: TheoryAnn) -> None:
    """The final-state list does not depend on the parameter point."""
    assert model.list_annihilation_final_states() == (
        type(model).list_annihilation_final_states()
    )


def test_scalar_vev_is_the_documented_zero_approximation() -> None:
    """``compute_vs`` returns zero, which is the approximation it documents.

    Not a derived result: ``ScalarMediator.compute_vs`` hardcodes ``0.0`` on
    both branches, with the tadpole solution left commented out and a
    docstring warning that says so. Pinned anyway, because ``vs`` is written
    back onto the model by every coupling setter and feeds ``compute_width_s``
    — so switching the solution back on moves published widths and spectra and
    is a declared numerical change, not an implementation detail.
    """
    for ms in (125.0, 550.0):
        model = _higgs_portal(250.0, ms)
        assert model.compute_vs() == 0.0
        assert model.vs == 0.0
