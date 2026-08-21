r"""``hazma._core.vector_mediator`` — the six vector cross sections.

The Cython twin,
``hazma/vector_mediator/_c_vector_mediator_cross_sections.pyx``, was
deleted in the same change that added the port (cython-to-rust Task 5.1,
``rules.md`` rule 1), and it left no ``.pxd`` and no ``__pyx_capi__``
capsules behind — so unlike ``test_core_positron_muon.py`` there is no
twin left to compare against here. What replaces it is three separate
oracles, each named where it is used:

``the parity corpus``
    ``test/parity`` pins what the Cython returned at 5,667 sampled
    energies across three mediator model points, and this port reproduces
    every one of them **bit-for-bit** — the ``EXACT`` budget, ``rtol =
    0``. That is the value gate and it is not repeated here. Numbers
    measured against the twin *before* its deletion are quoted in the
    module below where they bear on a claim.

``an independent Python implementation``
    ``ReferenceCrossSections`` re-derives all five closed forms from the
    formulas, with no shared code, and is compared at a loose tolerance.
    It exists to catch a transcription error the corpus could not: the
    corpus only says "the same as before", and the same-as-before could
    have been wrong.

``the argument surface``
    The dispatch contract, the keyword names, and the wrapper's
    re-exports — none of which the corpus drives, because it calls
    positionally with float64 arrays.

The physics claims that owe nothing to any oracle — thresholds, the
high-energy limit, the resonance shape, the saturation above ``x = 300``
— live in ``rust/src/kernels/vector_xs.rs``'s own ``cargo test`` module,
where they can be stated against the kernels directly. This file covers
what only Python can see.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from hazma._core import vector_mediator as core_vector
from hazma.vector_mediator import _vector_mediator_cross_sections as wrapper

#: The six module-level ``cdef double``\ s of the deleted ``.pyx``
#: (``:9-14``). Four coincide with ``hazma/_utils/legacy_parameters.pxd``
#: and two do not; none of them is ``hazma.parameters``' value for the
#: same particle, which is why they are written out rather than imported.
ME = 0.510998928
MMU = 105.6583715
MPI0 = 134.9766
MPI = 139.57018
FPI = 92.2138
ALPHA_EM = 1.0 / 137.04

#: A model point well above every threshold, so all six channels are open
#: and no comparison is testing a shared ``0.0``. Not a corpus point --
#: the corpus is the parity gate and this file is not trying to be a
#: second copy of it.
MX = 40.0
MV = 150.0
COUPLINGS = dict(
    gvxx=1.3,
    gvuu=0.37,
    gvdd=-0.19,
    gvss=0.11,
    gvee=0.23,
    gvmumu=-0.41,
    width_v=3.7,
)
#: ``(gvxx, gvuu, gvdd, gvss, gvee, gvmumu, width_v)`` — the trailing
#: seven arguments in the order every entry point but
#: ``sigma_xx_to_v_to_ff`` takes them.
REST = tuple(COUPLINGS.values())

#: Energies at which every channel is open, spanning four decades above
#: the highest threshold (``2 m_v = 300`` MeV).
OPEN_ENERGIES = [310.0, 400.0, 750.0, 1500.0, 1e4, 1e5]

#: The subset ``sigma_xx_to_vv`` is compared on. It stops at 1500 MeV, and
#: not because the port is wrong above it: the `V V` channel is the only
#: one whose expression contains a catastrophic cancellation, and above
#: about ``10 max(m_v, m_x)`` that cancellation dominates any statement
#: about the formula. `TestTheMediatorPairConditioning` is where the rest
#: of the range is covered, by asserting the cancellation law instead of a
#: fixed budget.
VV_WELL_CONDITIONED_ENERGIES = [310.0, 400.0, 750.0, 1500.0]


class ReferenceCrossSections:
    """The five closed forms, re-derived from the physics expressions.

    Deliberately *not* a transcription of the port: written with plain
    ``**`` and ``/``, no ``math.fma``, no ``__divdc3`` shim, and grouped
    the way the formula reads rather than the way the compiler emitted
    it. Agreement is therefore a statement about the formula, not about
    the arithmetic — which is why the tolerance below is 1e-13 and not
    zero.

    Every method takes ``(e_cm, mx, mv, *args)`` with ``args`` the
    trailing coupling block, in the order the entry point it mirrors
    takes it — which is also how the tests call both sides, so a
    mis-ordered argument cannot cancel out between them.
    """

    @staticmethod
    def propagator(mv: float, s: float, width_v: float) -> float:
        """``(m_v^2 - s)^2 + m_v^2 Gamma_v^2``, shared by four channels."""
        return (mv**2 - s) ** 2 + mv**2 * width_v**2

    @staticmethod
    def ff(e_cm: float, mx: float, mv: float, *args: float) -> float:
        """``x xbar -> V* -> f fbar``; ``args`` is ``(gvxx, gvll, width_v, mf)``."""
        gvxx, gvll, width_v, mf = args
        if e_cm < 2.0 * mf or e_cm < 2.0 * mx:
            return 0.0
        s = e_cm**2
        return (
            gvll**2
            * gvxx**2
            * (2.0 * mf**2 + s)
            * (2.0 * mx**2 + s)
            * math.sqrt((s - 4.0 * mf**2) / (s - 4.0 * mx**2))
        ) / (12.0 * math.pi * s * ReferenceCrossSections.propagator(mv, s, width_v))

    @staticmethod
    def pipi(e_cm: float, mx: float, mv: float, *args: float) -> float:
        """``x xbar -> V* -> pi+ pi-``; ``args`` is the seven-coupling block."""
        gvxx, gvuu, gvdd, _gvss, _gvee, _gvmumu, width_v = args
        if e_cm < 2.0 * mx or e_cm < 2.0 * MPI:
            return 0.0
        s = e_cm**2
        return (
            (gvdd - gvuu) ** 2 * gvxx**2 * (s - 4.0 * MPI**2) ** 1.5 * (2.0 * mx**2 + s)
        ) / (
            48.0
            * math.pi
            * s
            * math.sqrt(s - 4.0 * mx**2)
            * ReferenceCrossSections.propagator(mv, s, width_v)
        )

    @staticmethod
    def pi0g(e_cm: float, mx: float, mv: float, *args: float) -> float:
        """``x xbar -> V* -> pi0 gamma``; ``args`` is the seven-coupling block."""
        gvxx, gvuu, gvdd, _gvss, _gvee, _gvmumu, width_v = args
        if e_cm < MPI0 or e_cm < 2.0 * mx:
            return 0.0
        s = e_cm**2
        return (
            ALPHA_EM
            * (gvdd + 2.0 * gvuu) ** 2
            * gvxx**2
            * (s - MPI0**2) ** 3
            * (2.0 * mx**2 + s)
        ) / (
            3456.0
            * FPI**2
            * math.pi**4
            * e_cm**3
            * math.sqrt(s - 4.0 * mx**2)
            * ReferenceCrossSections.propagator(mv, s, width_v)
        )

    @staticmethod
    def pi0v(e_cm: float, mx: float, mv: float, *args: float) -> float:
        """``x xbar -> V* -> pi0 V``; ``args`` is the seven-coupling block."""
        gvxx, gvuu, gvdd, _gvss, _gvee, _gvmumu, width_v = args
        if e_cm < MPI0 + mv or e_cm < 2.0 * mx:
            return 0.0
        s = e_cm**2
        kallen = (
            (MPI0 - mv - e_cm)
            * (MPI0 + mv - e_cm)
            * (MPI0 - mv + e_cm)
            * (MPI0 + mv + e_cm)
        )
        return (
            (gvdd - gvuu) ** 2
            * (gvdd + gvuu) ** 2
            * gvxx**2
            * kallen**1.5
            * (2.0 * mx**2 + s)
        ) / (
            1536.0
            * FPI**2
            * math.pi**5
            * e_cm**3
            * math.sqrt(s - 4.0 * mx**2)
            * ReferenceCrossSections.propagator(mv, s, width_v)
        )

    @staticmethod
    def vv(e_cm: float, mx: float, mv: float, *args: float) -> float:
        """``x xbar -> V V``; only ``args[0]`` (``gvxx``) reaches this channel."""
        gvxx = args[0]
        if e_cm < 2.0 * mv or e_cm < 2.0 * mx:
            return 0.0
        s = e_cm**2
        root_v = math.sqrt(s - 4.0 * mv**2)
        root_x = math.sqrt(s - 4.0 * mx**2)
        t_channel = (
            -2.0 * root_v * root_x * (2.0 * mv**4 + 4.0 * mx**4 + mx**2 * s)
        ) / (mv**4 - 4.0 * mv**2 * mx**2 + mx**2 * s)
        shifted = s - 2.0 * mv**2
        s_channel = (
            2.0
            * (4.0 * mv**4 - 8.0 * mv**2 * mx**2 - 8.0 * mx**4 + 4.0 * mx**2 * s + s**2)
            * math.log((shifted + root_v * root_x) / (shifted - root_v * root_x))
        ) / shifted
        return (gvxx**4 * (t_channel + s_channel)) / (
            16.0 * math.pi * s * (s - 4.0 * mx**2)
        )


#: How far the port may sit from `ReferenceCrossSections`.
#:
#: Not a parity budget -- parity is `test/parity`'s job, at ``rtol = 0``.
#: This bounds the difference between two *spellings* of the same
#: formula, one of which uses fused multiply-adds, a `cpow`-equivalent
#: `exp(1.5 log t)` and a scaled complex division where the other uses
#: plain operators. Each of those costs a few ulp, and the expressions
#: chain a dozen of them, so a few hundred ulp is the honest bound:
#: 1e-13 is ~450 ulp and still ten decades away from any error that
#: would mean a wrong coefficient, power or sign.
#:
#: Every comparison passes ``abs=0.0`` alongside it. ``pytest.approx``
#: defaults to ``abs=1e-12`` and takes the **looser** of the two, and
#: these cross sections run from 1e-8 down to 1e-20 MeV^-2 — so without
#: that argument the absolute floor would swallow the relative budget
#: whole and every assertion in this class would pass against any
#: implementation at all.
REFERENCE_RTOL = 1e-13


class TestAgainstAnIndependentImplementation:
    """Each closed form, against a separate derivation of the same physics."""

    @pytest.mark.parametrize("e_cm", OPEN_ENERGIES)
    @pytest.mark.parametrize("lepton", ["e", "mu"])
    def test_lepton_channel(self, e_cm: float, lepton: str) -> None:
        coupling = COUPLINGS["gvee"] if lepton == "e" else COUPLINGS["gvmumu"]
        mass = ME if lepton == "e" else MMU
        got = core_vector.sigma_xx_to_v_to_ff(
            e_cm, MX, MV, COUPLINGS["gvxx"], coupling, COUPLINGS["width_v"], mass
        )
        want = ReferenceCrossSections.ff(
            e_cm, MX, MV, COUPLINGS["gvxx"], coupling, COUPLINGS["width_v"], mass
        )
        assert want > 0.0
        assert got == pytest.approx(want, rel=REFERENCE_RTOL, abs=0.0)

    @pytest.mark.parametrize("e_cm", OPEN_ENERGIES)
    @pytest.mark.parametrize(
        "channel",
        ["sigma_xx_to_v_to_pipi", "sigma_xx_to_v_to_pi0g", "sigma_xx_to_v_to_pi0v"],
    )
    def test_hadronic_channels(self, e_cm: float, channel: str) -> None:
        reference = {
            "sigma_xx_to_v_to_pipi": ReferenceCrossSections.pipi,
            "sigma_xx_to_v_to_pi0g": ReferenceCrossSections.pi0g,
            "sigma_xx_to_v_to_pi0v": ReferenceCrossSections.pi0v,
        }[channel]
        got = getattr(core_vector, channel)(e_cm, MX, MV, *REST)
        want = reference(e_cm, MX, MV, *REST)
        assert want > 0.0
        assert got == pytest.approx(want, rel=REFERENCE_RTOL, abs=0.0)

    @pytest.mark.parametrize("e_cm", VV_WELL_CONDITIONED_ENERGIES)
    def test_mediator_pair_channel(self, e_cm: float) -> None:
        got = core_vector.sigma_xx_to_vv(e_cm, MX, MV, *REST)
        want = ReferenceCrossSections.vv(e_cm, MX, MV, *REST)
        assert want > 0.0
        assert got == pytest.approx(want, rel=REFERENCE_RTOL, abs=0.0)

    def test_the_reference_would_notice_a_wrong_coefficient(self) -> None:
        """The tolerance is tight enough to be a test.

        A guard on the guard: `REFERENCE_RTOL` is loose enough to absorb
        fused multiply-adds, and this shows it is not loose enough to
        absorb an error. Perturbing one coupling by a part in 1e9 --
        far below any plausible transcription slip -- already separates
        the two by four decades more than the budget.
        """
        e_cm = 400.0
        want = ReferenceCrossSections.pipi(e_cm, MX, MV, *REST)
        nudged = list(REST)
        nudged[1] *= 1.0 + 1e-9
        got = core_vector.sigma_xx_to_v_to_pipi(e_cm, MX, MV, *nudged)
        assert got != pytest.approx(want, rel=REFERENCE_RTOL, abs=0.0)


class TestTheMediatorPairConditioning:
    r"""``sigma_xx_to_vv`` loses digits to a cancellation.

    And it loses them at a rate the expression predicts. The `V V`
    amplitude's logarithm is

    .. code-block:: text

        log( (s - 2 m_v^2 + R) / (s - 2 m_v^2 - R) ),
        R = sqrt(s - 4 m_v^2) sqrt(s - 4 m_x^2)

    and at high `s` the denominator is a difference of two nearly equal
    numbers: `R -> s - 2 m_v^2 - 2 m_x^2`, so the subtraction keeps only
    the `2 m_x^2` and throws away everything above it. The surviving
    fraction

    .. code-block:: text

        c(s) = |s - 2 m_v^2 - R| / (s - 2 m_v^2)

    falls like `1/s`, and any two spellings of the expression must
    therefore separate like `eps / c(s)` however carefully each is
    written. Measured against `ReferenceCrossSections`, which differs from
    the port only in where it fuses:

    .. code-block:: text

        e_cm      c(s)       relative difference
        1500     1.6e-3      1.7e-15
        1e4      3.2e-5      9.5e-14
        1e5      3.2e-7      9.0e-12
        1e6      3.2e-9      1.7e-10

    — four decades of `c`, four decades of difference, at a ratio of
    about `3e-17`, i.e. an eighth of an ulp per unit of lost
    cancellation. That is the whole story, and it is a property of the
    formula rather than of either implementation. Asserting it is
    stronger than asserting a tolerance: a genuine transcription error
    would not scale with `c(s)`.

    This is not a defect the port introduced or could remove without
    rewriting the expression, which `rules.md` rule 1 forbids. It is
    recorded here so that a later reader who measures a 1e-10
    disagreement at 1e6 MeV knows what they are looking at.
    """

    #: The proportionality constant above, with room: measured 3.0e-17,
    #: allowed 3e-16. Loose by a decade because the constant is an average
    #: over a chain of roundings, not a bound on one.
    CONDITIONING_CONSTANT = 3e-16

    @staticmethod
    def surviving_fraction(e_cm: float) -> float:
        """`c(s)`: what is left of the denominator after the subtraction."""
        s = e_cm**2
        shifted = s - 2.0 * MV**2
        product = math.sqrt(s - 4.0 * MV**2) * math.sqrt(s - 4.0 * MX**2)
        return abs(shifted - product) / shifted

    @pytest.mark.parametrize("e_cm", [1500.0, 1e4, 1e5, 1e6])
    def test_the_disagreement_tracks_the_cancellation(self, e_cm: float) -> None:
        got = core_vector.sigma_xx_to_vv(e_cm, MX, MV, *REST)
        want = ReferenceCrossSections.vv(e_cm, MX, MV, *REST)
        assert want > 0.0
        difference = abs(got - want) / want
        allowed = self.CONDITIONING_CONSTANT / self.surviving_fraction(e_cm)
        assert difference <= allowed, (
            f"e_cm={e_cm}: {difference:.3e} exceeds {allowed:.3e}, so the "
            "disagreement is no longer explained by the cancellation"
        )

    def test_the_cancellation_really_does_worsen(self) -> None:
        """Guard the guard: the bound above tightens as the energy falls."""
        # Five decades of loss across the four energies, which is what
        # makes the bound above non-vacuous at the top end.
        decades_lost = 1e5
        fractions = [self.surviving_fraction(e) for e in (1500.0, 1e4, 1e5, 1e6)]
        assert fractions == sorted(fractions, reverse=True)
        assert fractions[0] / fractions[-1] > decades_lost

    def test_the_channel_is_still_accurate_where_it_is_conditioned(self) -> None:
        """The physical range is unaffected.

        hazma's mediators are hundreds of MeV, so `e_cm` a few times
        `m_v` is where a sub-GeV model actually samples, and there the
        two spellings agree to a couple of ulp.
        """
        for e_cm in VV_WELL_CONDITIONED_ENERGIES:
            got = core_vector.sigma_xx_to_vv(e_cm, MX, MV, *REST)
            want = ReferenceCrossSections.vv(e_cm, MX, MV, *REST)
            assert got == pytest.approx(want, rel=1e-14, abs=0.0)


class TestTheUnusedCouplings:
    """The three unused couplings reach no channel that ignores them.

    ``gvss`` reaches none of the five at all, and neither lepton coupling
    reaches a hadronic one. The `.pyx` marked ``gvss``, ``gvee`` and ``gvmumu`` ``CYTHON_UNUSED``
    in four of its five kernels, and the port drops them from those
    kernels' Rust signatures entirely — so this is the layer that can
    still check the claim, because the PyO3 wrapper is where they are
    still accepted. If a future edit wires one of them in, the public
    signature would not change and only this test would notice.
    """

    @pytest.mark.parametrize(
        "channel",
        [
            "sigma_xx_to_v_to_pipi",
            "sigma_xx_to_v_to_pi0g",
            "sigma_xx_to_v_to_pi0v",
            "sigma_xx_to_vv",
        ],
    )
    @pytest.mark.parametrize("unused", ["gvss", "gvee", "gvmumu"])
    def test_no_hadronic_channel_reads_them(self, channel: str, unused: str) -> None:
        e_cm = 400.0
        baseline = getattr(core_vector, channel)(e_cm, MX, MV, *REST)
        assert baseline > 0.0

        bumped = dict(COUPLINGS)
        bumped[unused] = bumped[unused] * 7.0 + 1.0
        got = getattr(core_vector, channel)(e_cm, MX, MV, *bumped.values())
        assert got.hex() == baseline.hex(), f"{channel} moved when {unused} changed"

    def test_the_quark_couplings_do_reach_them(self) -> None:
        """The negative above is only meaningful beside this positive."""
        e_cm = 400.0
        for moved in ("gvuu", "gvdd"):
            bumped = dict(COUPLINGS)
            bumped[moved] = bumped[moved] * 7.0 + 1.0
            for channel in ("sigma_xx_to_v_to_pipi", "sigma_xx_to_v_to_pi0g"):
                baseline = getattr(core_vector, channel)(e_cm, MX, MV, *REST)
                got = getattr(core_vector, channel)(e_cm, MX, MV, *bumped.values())
                assert got != baseline, f"{channel} ignored {moved}"


class TestTheDispatchContract:
    """Scalar in / float out, array in / fresh array out, errors between.

    ``crate::dispatch`` implements this once and
    ``test/test_core_dispatch.py`` sweeps it exhaustively; what is checked
    here is that these six entry points are wired to it, and that the
    array path agrees with the scalar path element by element — which the
    Cython's separate ``__vec_*`` loop made a real question and the port's
    shared kernel makes a cheap one.
    """

    def test_a_scalar_returns_a_float(self) -> None:
        got = core_vector.sigma_xx_to_v_to_pipi(400.0, MX, MV, *REST)
        assert type(got) is float

    def test_an_array_returns_a_fresh_array(self) -> None:
        grid = np.array(OPEN_ENERGIES, dtype=np.float64)
        got = core_vector.sigma_xx_to_vv(grid, MX, MV, *REST)
        assert isinstance(got, np.ndarray)
        assert got.dtype == np.float64
        assert got.shape == grid.shape
        assert got is not grid

    @pytest.mark.parametrize(
        "channel",
        [
            "sigma_xx_to_v_to_pipi",
            "sigma_xx_to_v_to_pi0g",
            "sigma_xx_to_v_to_pi0v",
            "sigma_xx_to_vv",
        ],
    )
    def test_the_array_path_matches_the_scalar_path(self, channel: str) -> None:
        grid = np.array(OPEN_ENERGIES, dtype=np.float64)
        batched = getattr(core_vector, channel)(grid, MX, MV, *REST)
        one_by_one = [
            getattr(core_vector, channel)(float(e), MX, MV, *REST) for e in grid
        ]
        # Bit-equality, not approximate: it is the same kernel called the
        # same way, so any difference would be a dispatch bug.
        assert [v.hex() for v in batched.tolist()] == [v.hex() for v in one_by_one]

    def test_a_zero_dimensional_array_takes_the_scalar_path(self) -> None:
        got = core_vector.sigma_xx_to_vv(np.float64(400.0), MX, MV, *REST)
        assert type(got) is float

    def test_a_two_dimensional_array_is_a_value_error(self) -> None:
        grid = np.ones((2, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="0 or 1-dimensional"):
            core_vector.sigma_xx_to_vv(grid, MX, MV, *REST)

    def test_a_non_number_is_a_type_error(self) -> None:
        with pytest.raises(TypeError):
            core_vector.sigma_xx_to_vv(None, MX, MV, *REST)

    def test_every_argument_is_accepted_by_keyword(self) -> None:
        """Every argument is accepted by keyword, as the Cython's were.

        A ``text_signature`` is a claim PyO3 does not enforce; this is
        the check. Narrowing any of these to positional-only would be a
        silent break of the public API.
        """
        positional = core_vector.sigma_xx_to_v_to_pipi(400.0, MX, MV, *REST)
        by_keyword = core_vector.sigma_xx_to_v_to_pipi(
            e_cms=400.0, mx=MX, mv=MV, **COUPLINGS
        )
        assert by_keyword.hex() == positional.hex()

        thermal_positional = core_vector.thermal_cross_section(20.0, MX, MV, *REST)
        thermal_keyword = core_vector.thermal_cross_section(
            x=20.0, mx=MX, mv=MV, **COUPLINGS
        )
        assert thermal_keyword.hex() == thermal_positional.hex()


class TestTheThresholdRaise:
    """``e_cm = 2 m_x`` raises in two channels and not in the other four.

    The parity corpus pins this in three blocks
    (``test/parity/data/manifest.json``, ``raises``), because the Cython's
    ``**`` operator compiled to complex arithmetic and
    ``__Pyx_SoftComplexToDouble`` rejects a non-zero imaginary part. The
    port reproduces the *type*, not the wording — see
    ``rust/src/vector_mediator.rs``. It is a defect rather than a design:
    ``docs/followups/todo/vector-cross-sections-raise-at-the-two-mx-threshold.md``.
    """

    #: A dark matter mass whose ``2 m_x`` clears both complex channels'
    #: own thresholds, so the raise is not masked by an early ``0.0``.
    HEAVY_MX = 400.0

    @pytest.mark.parametrize(
        "channel", ["sigma_xx_to_v_to_pipi", "sigma_xx_to_v_to_pi0v"]
    )
    def test_the_complex_channels_raise(self, channel: str) -> None:
        with pytest.raises(TypeError):
            getattr(core_vector, channel)(2.0 * self.HEAVY_MX, self.HEAVY_MX, MV, *REST)

    @pytest.mark.parametrize(
        ("channel", "kind"),
        [
            ("sigma_xx_to_v_to_ff", "inf"),
            ("sigma_xx_to_v_to_pi0g", "inf"),
            ("sigma_xx_to_vv", "nan"),
        ],
    )
    def test_the_real_channels_return_a_non_finite_number(
        self, channel: str, kind: str
    ) -> None:
        """No raise here, but no usable answer either.

        And the three real channels do not even agree on which non-finite
        value. `ff` and `pi0g` divide a finite numerator by the vanishing
        `sqrt(e_cm^2 - 4 mx^2)` and get an infinity; `vv` divides by
        `e_cm^2 - 4 mx^2` *and* multiplies by that same root, so it gets
        `0 * inf` and a NaN. That inconsistency across four channels at
        one kinematic point is the substance of the follow-up.
        """
        if channel == "sigma_xx_to_v_to_ff":
            got = core_vector.sigma_xx_to_v_to_ff(
                2.0 * self.HEAVY_MX,
                self.HEAVY_MX,
                MV,
                COUPLINGS["gvxx"],
                COUPLINGS["gvee"],
                COUPLINGS["width_v"],
                ME,
            )
        else:
            got = getattr(core_vector, channel)(
                2.0 * self.HEAVY_MX, self.HEAVY_MX, MV, *REST
            )
        assert not math.isfinite(got)
        assert math.isinf(got) if kind == "inf" else math.isnan(got)

    def test_one_bad_element_takes_the_whole_array_down(self) -> None:
        """One bad element takes the whole array down.

        The Cython's ``__vec_*`` loop jumped to its error label on the
        first failing index rather than filling that slot, so an array
        containing the threshold raises instead of returning a partly-nan
        result. ``dispatch::map_unary_try`` does the same.
        """
        grid = np.array(
            [2.0 * self.HEAVY_MX - 1.0, 2.0 * self.HEAVY_MX, 2.0 * self.HEAVY_MX + 1.0],
            dtype=np.float64,
        )
        with pytest.raises(TypeError):
            core_vector.sigma_xx_to_v_to_pipi(grid, self.HEAVY_MX, MV, *REST)

    def test_the_neighbouring_doubles_are_ordinary(self) -> None:
        """The neighbouring doubles are ordinary numbers.

        One ulp away it is finite, which is what makes this a point
        defect rather than a region.
        """
        e_cm = 2.0 * self.HEAVY_MX
        for offset in (1, 2, 3):
            neighbour = math.nextafter(e_cm, math.inf)
            for _ in range(offset - 1):
                neighbour = math.nextafter(neighbour, math.inf)
            got = core_vector.sigma_xx_to_v_to_pipi(neighbour, self.HEAVY_MX, MV, *REST)
            assert math.isfinite(got)


class TestTheWrapperReExports:
    """The wrapper binds each short alias to the kernel it stands for.

    Its mixin methods call ``sig_ff``, ``sig_pipi`` and friends,
    and the parity corpus drives the canonical names beside them — so a
    mis-wired alias would ship green. Six identity checks close that.
    """

    @pytest.mark.parametrize(
        ("alias", "kernel"),
        [
            ("sig_ff", "sigma_xx_to_v_to_ff"),
            ("sig_pipi", "sigma_xx_to_v_to_pipi"),
            ("sig_pi0g", "sigma_xx_to_v_to_pi0g"),
            ("sig_pi0v", "sigma_xx_to_v_to_pi0v"),
            ("sig_vv", "sigma_xx_to_vv"),
            ("tcs", "thermal_cross_section"),
        ],
    )
    def test_each_alias_is_its_kernel(self, alias: str, kernel: str) -> None:
        assert getattr(wrapper, alias) is getattr(core_vector, kernel)

    def test_the_cython_twin_is_gone(self) -> None:
        """The twin is gone, asserted on the source files.

        Not on an ``ImportError``: a built ``.so`` and its generated ``.c`` sit beside a deleted
        ``.pyx``, gitignored, and neither ``git checkout`` nor ``git
        stash`` removes them (Phase 04's learnings, §3) — so an import
        check would pass on a stale tree and say nothing.
        """
        repo = Path(__file__).resolve().parent.parent
        pyx = repo / "hazma" / "vector_mediator"
        assert not (pyx / "_c_vector_mediator_cross_sections.pyx").exists()
        # The `setup.py` check is on the *extension list*, not on the
        # whole file: the entry there is what would rebuild the module,
        # and the file also carries a comment naming it.
        sources = (repo / "setup.py").read_text()
        assert '"_c_vector_mediator_cross_sections",' not in sources
