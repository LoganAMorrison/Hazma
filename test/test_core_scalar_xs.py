"""`hazma._core.scalar_mediator` — the twelve scalar cross sections.

Companion to `test/test_core_vector_xs.py`, for the twin ported one task
later (cython-to-rust Task 5.2). The division of labour is the same and
worth restating, because it decides what belongs here:

* **the parity corpus** (`test/parity/`) compares every entry point
  against 12,440 values captured from the pre-port Cython, at `rtol = 0`
  for the eleven closed forms. That is the numerical gate, and nothing
  in this file duplicates it;
* **`rust/src/kernels/scalar_xs.rs`'s unit tests** assert the analytic
  statements the closed forms make — thresholds, the resonance peak, the
  sum-of-channels identity, the high-energy limit — with no Python in
  the way;
* **this file** covers what only the PyO3 boundary can see: the argument
  surface, the keyword names, the wrapper's aliases, and an independent
  Python implementation of the four simplest channels.

The independent reference matters because a transliterated Mathematica
dump can be transcribed wrongly in a way that is *self-consistent*: the
corpus would catch it only if the original were right, and the Rust unit
tests share the formula. `ReferenceCrossSections` below shares neither.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from hazma._core import scalar_mediator as core_scalar
from hazma.parameters import electron_mass, muon_mass
from hazma.scalar_mediator import HiggsPortal
from hazma.scalar_mediator import _scalar_mediator_cross_sections as wrapper

#: The nine module constants `_c_scalar_mediator_cross_sections.pyx:9-17`
#: declared for itself, written out because the `.pyx` is gone. They are
#: *not* `hazma.parameters`' values -- `alpha_em` there is 1/137.036 and
#: here it is 1/137.04 (`rust/src/constants.rs`, "the two tables
#: disagree"), and `rust/src/kernels/scalar_xs.rs`'s constants are these
#: numbers, which `the_module_constants_are_the_pyx_values` pins on the
#: Rust side.
VH = 246.22795e3
ALPHA_EM = 1.0 / 137.04
ME = 0.510998928
MMU = 105.6583715
MPI0 = 134.9766
MPI = 139.57018

#: The corpus's `open_resonance` model point, as a positional argument
#: block: `(mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs)`. The
#: derived couplings are rounded -- nothing here needs the solver's exact
#: `vs`, only a point where every channel is open and well conditioned.
MX = 100.0
MS = 300.0
ARGS = (MX, MS, 1.0, 0.1, 0.1, 0.1, 1.0e5, 2.5, 1.0)

#: Every entry point that takes `(e_cms, *ARGS)` and nothing else.
PLAIN = [
    "sigma_xx_to_s_to_gg",
    "sigma_xx_to_s_to_pi0pi0",
    "sigma_xx_to_s_to_pipi",
    "sigma_xx_to_ss",
    "sigma_ss_to_xx",
    "sigma_xpi_to_xpi",
    "sigma_xpi0_to_xpi0",
    "sigma_xg_to_xg",
    "sigma_xs_to_xs",
]

#: The two that take a trailing fermion mass.
WITH_FERMION = ["sigma_xx_to_s_to_ff", "sigma_xl_to_xl"]


class ReferenceCrossSections:
    """The four simplest channels, re-derived from the physics.

    Written from the same published expressions the `.pyx` holds but with
    no shared code, no shared association order and no FMAs, so it agrees
    with the port only to about a dozen ulp -- which is exactly what makes
    it able to catch a wrong *coefficient*, the failure a bit-equality
    corpus cannot distinguish from a correct one.

    Only the four channels whose closed forms fit in a few readable lines
    are here. The two 90-line elastic expressions are not re-derivable at
    this length, and the corpus is what covers them.
    """

    @staticmethod
    def propagator(ms: float, e_cm: float, width_s: float) -> float:
        """`(ms^2 - s)^2 + ms^2 width^2`, the Breit-Wigner denominator."""
        return (ms**2 - e_cm**2) ** 2 + ms**2 * width_s**2

    @classmethod
    def ff(cls, e_cm: float, mf: float) -> float:
        """`xx -> S* -> f fbar`, MeV^-2."""
        mx, ms, gsxx, gsff, _gsGG, _gsFF, _lam, width_s, _vs = ARGS
        if e_cm < 2.0 * mf or e_cm < 2.0 * mx:
            return 0.0
        beta_f = math.sqrt(e_cm**2 - 4.0 * mf**2)
        beta_x = math.sqrt(e_cm**2 - 4.0 * mx**2)
        num = gsff**2 * gsxx**2 * mf**2 * beta_f**3 * beta_x
        return num / (
            16.0 * math.pi * e_cm**2 * VH**2 * cls.propagator(ms, e_cm, width_s)
        )

    @classmethod
    def gg(cls, e_cm: float) -> float:
        """`xx -> S* -> gamma gamma`, MeV^-2."""
        mx, ms, gsxx, _gsff, _gsGG, gsFF, lam, width_s, _vs = ARGS
        if e_cm < 2.0 * mx:
            return 0.0
        beta_x = math.sqrt(e_cm**2 - 4.0 * mx**2)
        num = ALPHA_EM**2 * gsFF**2 * gsxx**2 * e_cm**3 * beta_x
        return num / (128.0 * lam**2 * math.pi**3 * cls.propagator(ms, e_cm, width_s))

    @classmethod
    def two_pion(cls, e_cm: float, mpi: float, identical: bool) -> float:
        """`xx -> S* -> pi pi`, MeV^-2, for either pion.

        The hadronic vertex is the same bracket in both channels; the only
        differences are the pion mass and the factor 2 for identical
        particles in the final state.
        """
        mx, ms, gsxx, gsff, gsGG, _gsFF, lam, width_s, vs = ARGS
        if e_cm < 2.0 * mpi or e_cm < 2.0 * mx:
            return 0.0
        b0 = 2654.082197477761
        quarks = 4.8 + 2.3
        gluon = 162.0 * gsGG * lam**3 * (2.0 * mpi**2 - e_cm**2) * VH**2
        quark = (
            b0
            * quarks
            * (9.0 * lam + 4.0 * gsGG * vs)
            * (-3.0 * lam * VH + 3.0 * gsff * lam * vs + 2.0 * gsGG * VH * vs)
            * (
                2.0 * gsGG * VH * (9.0 * lam - 4.0 * gsGG * vs)
                + 9.0 * gsff * lam * (3.0 * lam + 4.0 * gsGG * vs)
            )
        )
        phase_space = math.sqrt((e_cm**2 - 4.0 * mpi**2) * (e_cm**2 - 4.0 * mx**2))
        num = gsxx**2 * phase_space * (gluon + quark) ** 2
        den = (
            (419904.0 if identical else 209952.0)
            * lam**6
            * math.pi
            * e_cm**2
            * VH**4
            * (9.0 * lam + 4.0 * gsGG * vs) ** 2
            * cls.propagator(ms, e_cm, width_s)
        )
        return num / den


#: Energies spanning four decades, all above every threshold in `ARGS`
#: (the highest is `2 ms = 600 MeV`) and off resonance.
SPANNING = [700.0, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7]

#: How far the reference may sit from the port. The two evaluate the same
#: closed forms in different association orders, with FMAs on one side and
#: not the other, so the gap is accumulated last-bit arithmetic over
#: expressions ~20 operations deep. 1e-13 is a few dozen ulp; the measured
#: worst over `SPANNING` is under 1e-14.
REFERENCE_RTOL = 1e-13


class TestAgainstAnIndependentImplementation:
    """The four re-derivable channels, against `ReferenceCrossSections`."""

    @pytest.mark.parametrize("e_cm", SPANNING)
    @pytest.mark.parametrize("lepton", ["e", "mu"])
    def test_fermion_channel(self, e_cm: float, lepton: str) -> None:
        mf = ME if lepton == "e" else MMU
        got = core_scalar.sigma_xx_to_s_to_ff(e_cm, *ARGS, mf)
        assert got == pytest.approx(
            ReferenceCrossSections.ff(e_cm, mf), rel=REFERENCE_RTOL
        )

    @pytest.mark.parametrize("e_cm", SPANNING)
    def test_photon_channel(self, e_cm: float) -> None:
        got = core_scalar.sigma_xx_to_s_to_gg(e_cm, *ARGS)
        assert got == pytest.approx(ReferenceCrossSections.gg(e_cm), rel=REFERENCE_RTOL)

    @pytest.mark.parametrize("e_cm", SPANNING)
    @pytest.mark.parametrize("channel", ["pi0pi0", "pipi"])
    def test_pion_channels(self, e_cm: float, channel: str) -> None:
        if channel == "pi0pi0":
            got = core_scalar.sigma_xx_to_s_to_pi0pi0(e_cm, *ARGS)
            want = ReferenceCrossSections.two_pion(e_cm, MPI0, identical=True)
        else:
            got = core_scalar.sigma_xx_to_s_to_pipi(e_cm, *ARGS)
            want = ReferenceCrossSections.two_pion(e_cm, MPI, identical=False)
        assert got == pytest.approx(want, rel=REFERENCE_RTOL)

    def test_the_reference_would_notice_a_wrong_coefficient(self) -> None:
        """The reference is sharp enough to be worth running.

        `REFERENCE_RTOL` is 1e-13, so a mistyped coefficient -- the
        transcription error a 90-line Mathematica dump invites -- lands
        many decades outside it. Asserted rather than assumed: a 1-in-10^6
        perturbation of the `.pyx`'s 419904 is still 10^7 times the
        tolerance.
        """
        e_cm = 1.0e4
        exact = ReferenceCrossSections.two_pion(e_cm, MPI0, identical=True)
        perturbed = exact * 419904.0 / 419903.58
        assert abs(perturbed / exact - 1.0) > 1.0e6 * REFERENCE_RTOL


class TestTheDispatchContract:
    """Scalar in / scalar out, array in / fresh array out, and the errors.

    The shape `crate::dispatch::map_unary` guarantees, checked at the one
    layer that can see it. `thermal_cross_section` is excluded throughout:
    the Cython declared its `x` a `double`, so it never had array
    dispatch.
    """

    def test_a_scalar_returns_a_float(self) -> None:
        got = core_scalar.sigma_xx_to_s_to_gg(700.0, *ARGS)
        assert isinstance(got, float)

    def test_an_array_returns_a_fresh_array(self) -> None:
        grid = np.array([700.0, 800.0, 900.0])
        got = core_scalar.sigma_xx_to_s_to_gg(grid, *ARGS)
        assert isinstance(got, np.ndarray)
        assert got is not grid
        assert got.shape == grid.shape
        assert got.dtype == np.float64

    @pytest.mark.parametrize("name", PLAIN)
    def test_the_array_path_matches_the_scalar_path(self, name: str) -> None:
        """Bit-for-bit, not approximately: it is the same kernel."""
        fn = getattr(core_scalar, name)
        grid = np.geomspace(1.0, 1.0e5, 64)
        batched = fn(grid, *ARGS)
        one_at_a_time = np.array([fn(float(e), *ARGS) for e in grid])
        assert batched.tobytes() == one_at_a_time.tobytes()

    @pytest.mark.parametrize("name", WITH_FERMION)
    def test_the_array_path_matches_the_scalar_path_with_a_fermion(
        self, name: str
    ) -> None:
        fn = getattr(core_scalar, name)
        grid = np.geomspace(1.0, 1.0e5, 64)
        batched = fn(grid, *ARGS, MMU)
        one_at_a_time = np.array([fn(float(e), *ARGS, MMU) for e in grid])
        assert batched.tobytes() == one_at_a_time.tobytes()

    def test_a_zero_dimensional_array_takes_the_scalar_path(self) -> None:
        got = core_scalar.sigma_xx_to_s_to_gg(np.float64(700.0), *ARGS)
        assert isinstance(got, float)
        assert got == core_scalar.sigma_xx_to_s_to_gg(700.0, *ARGS)

    def test_a_two_dimensional_array_is_a_value_error(self) -> None:
        with pytest.raises(ValueError, match="Center of mass energies"):
            core_scalar.sigma_xx_to_s_to_gg(np.zeros((2, 2)) + 700.0, *ARGS)

    def test_a_non_number_is_a_type_error(self) -> None:
        with pytest.raises(TypeError):
            core_scalar.sigma_xx_to_s_to_gg(None, *ARGS)

    def test_a_string_array_is_a_value_error(self) -> None:
        """A `str` converts to a `<U` array, so it fails on dtype.

        Worth pinning apart from the `None` case above: the two take
        different branches of `crate::dispatch` and give different
        exception types, and a user passing `"700"` sees the second.
        """
        with pytest.raises(ValueError, match="must be a float64 array"):
            core_scalar.sigma_xx_to_s_to_gg("700", *ARGS)

    def test_every_argument_is_accepted_by_keyword(self) -> None:
        """The Cython entry points were `def`s, so every name is public.

        Including the ones the kernel does not read: `sigma_xx_to_ss`
        takes only `gsxx`, but the wrapper passes all nine positionally
        and a user may pass any of them by name.
        """
        keywords = dict(
            zip(
                ("mx", "ms", "gsxx", "gsff", "gsGG", "gsFF", "lam", "width_s", "vs"),
                ARGS,
                strict=True,
            )
        )
        for name in PLAIN:
            fn = getattr(core_scalar, name)
            assert fn(e_cms=700.0, **keywords) == fn(700.0, *ARGS)
        assert core_scalar.sigma_xx_to_s_to_ff(
            e_cms=700.0, mf=MMU, **keywords
        ) == core_scalar.sigma_xx_to_s_to_ff(700.0, *ARGS, MMU)
        assert core_scalar.thermal_cross_section(
            x=20.0, **keywords
        ) == core_scalar.thermal_cross_section(20.0, *ARGS)


class TestTheUnusedCouplings:
    """A channel ignores exactly the couplings its `.pyx` marked unused.

    `CYTHON_UNUSED` in the generated C is the source of truth, and the
    Rust kernels simply do not take those arguments — so this is the only
    layer that can check it, and the check is that varying an ignored
    coupling changes nothing while varying a read one changes something.
    """

    #: `(entry point, coupling index in ARGS, read?)`. Indices are into
    #: `ARGS`, i.e. 3 = gsff, 4 = gsGG, 5 = gsFF.
    CASES: ClassVar[list[tuple[str, int, bool]]] = [
        ("sigma_xx_to_ss", 3, False),
        ("sigma_xx_to_ss", 4, False),
        ("sigma_xx_to_ss", 5, False),
        ("sigma_ss_to_xx", 3, False),
        ("sigma_xs_to_xs", 4, False),
        ("sigma_xx_to_s_to_gg", 3, False),
        ("sigma_xx_to_s_to_gg", 5, True),
        ("sigma_xx_to_s_to_pi0pi0", 5, False),
        ("sigma_xx_to_s_to_pi0pi0", 4, True),
        ("sigma_xx_to_s_to_pipi", 5, False),
        ("sigma_xpi_to_xpi", 5, False),
        ("sigma_xpi_to_xpi", 4, True),
        ("sigma_xpi0_to_xpi0", 5, False),
        ("sigma_xg_to_xg", 3, False),
        ("sigma_xg_to_xg", 5, True),
    ]

    @pytest.mark.parametrize(("name", "index", "read"), CASES)
    def test_a_coupling_matters_exactly_when_the_pyx_read_it(
        self, name: str, index: int, read: bool
    ) -> None:
        fn = getattr(core_scalar, name)
        e_cm = 700.0
        base = fn(e_cm, *ARGS)
        varied = list(ARGS)
        varied[index] = varied[index] * 3.0 + 0.7
        got = fn(e_cm, *varied)
        if read:
            assert got != base
        else:
            assert got == base


class TestTheWrapperReExports:
    """The wrapper binds each short alias to the kernel it stands for.

    Its mixin methods call `sig_ff`, `sig_gg` and friends, and the parity
    corpus drives those same aliases -- so a mis-wired alias would fail
    the corpus too. What these add is the *identity*: they say which
    kernel each alias is, which is what makes the corpus's coverage
    legible rather than coincidental.
    """

    ALIASES: ClassVar[list[tuple[str, str]]] = [
        ("sig_ff", "sigma_xx_to_s_to_ff"),
        ("sig_gg", "sigma_xx_to_s_to_gg"),
        ("sig_pi0pi0", "sigma_xx_to_s_to_pi0pi0"),
        ("sig_pipi", "sigma_xx_to_s_to_pipi"),
        ("sig_ss", "sigma_xx_to_ss"),
        ("sig_ss_to_xx", "sigma_ss_to_xx"),
        ("sig_xl", "sigma_xl_to_xl"),
        ("sig_xpi", "sigma_xpi_to_xpi"),
        ("sig_xpi0", "sigma_xpi0_to_xpi0"),
        ("sig_xg", "sigma_xg_to_xg"),
        ("sig_xs", "sigma_xs_to_xs"),
        ("tcs", "thermal_cross_section"),
    ]

    @pytest.mark.parametrize(("alias", "kernel"), ALIASES)
    def test_each_alias_is_its_kernel(self, alias: str, kernel: str) -> None:
        assert getattr(wrapper, alias) is getattr(core_scalar, kernel)

    #: Everything `hazma._core.scalar_mediator` serves that is *not* a
    #: cross section, and which task put it there. The submodule is
    #: per-model rather than per-`.pyx`, so Phase 06 lands the mediator
    #: spectra alongside these twelve; each is pinned in its own test
    #: module (`test/test_core_mediator_decay_photon.py` for the one
    #: below) rather than here. Task 6.3 adds `dnde_decay_s` and
    #: `dnde_decay_s_pt`.
    NON_CROSS_SECTIONS: ClassVar[set[str]] = {
        "scalar_mediator_decay_spectrum",  # cython-to-rust Task 6.2
    }

    def test_the_alias_table_is_the_whole_served_cross_section_roster(self) -> None:
        """Twelve aliases, twelve kernels, nothing left over either way.

        The equality is against the aliases *plus*
        :data:`NON_CROSS_SECTIONS`, so a kernel that appears on this
        submodule without a home still fails -- what the roster no longer
        claims is that cross sections are all this submodule holds.
        """
        served = {name for name in dir(core_scalar) if not name.startswith("_")}
        expected = {kernel for _, kernel in self.ALIASES} | self.NON_CROSS_SECTIONS
        assert served == expected
        # And the two sets really are disjoint, so the union above cannot
        # be hiding a missing alias.
        assert not self.NON_CROSS_SECTIONS & {kernel for _, kernel in self.ALIASES}

    def test_the_cython_twin_is_gone(self) -> None:
        """The twin is gone, asserted on the source files.

        Not on an `ImportError`: a built `.so` and its generated `.c` sit
        beside a deleted `.pyx`, gitignored, and neither `git checkout`
        nor `git stash` removes them (Phase 04's learnings, §3) -- so an
        import check would pass on a stale tree and say nothing.
        """
        repo = Path(__file__).resolve().parent.parent
        package = repo / "hazma" / "scalar_mediator"
        assert not (package / "_c_scalar_mediator_cross_sections.pyx").exists()
        # The `setup.py` check is on the *extension list*, not on the
        # whole file: the entry there is what would rebuild the module,
        # and the file also carries a comment naming it.
        sources = (repo / "setup.py").read_text()
        assert '"_c_scalar_mediator_cross_sections",' not in sources


class TestTheModelLayerStillWorks:
    """The mixin methods reach the kernels with the model's own arguments.

    The wrapper's job is to unpack `self` into the nine-argument block,
    and nothing above it would notice an argument passed in the wrong
    order -- every one is a float. So each method is checked against the
    kernel called directly with the model's attributes.
    """

    def test_each_method_matches_the_kernel(self) -> None:
        model = HiggsPortal(mx=100.0, ms=300.0, gsxx=1.0, stheta=1e-1)
        args = (
            model.mx,
            model.ms,
            model.gsxx,
            model.gsff,
            model.gsGG,
            model.gsFF,
            model.lam,
            model.width_s,
            model.vs,
        )
        e_cm = 700.0
        pairs = [
            (model.sigma_xx_to_s_to_gg(e_cm), core_scalar.sigma_xx_to_s_to_gg),
            (model.sigma_xx_to_s_to_pi0pi0(e_cm), core_scalar.sigma_xx_to_s_to_pi0pi0),
            (model.sigma_xx_to_s_to_pipi(e_cm), core_scalar.sigma_xx_to_s_to_pipi),
            (model.sigma_xx_to_ss(e_cm), core_scalar.sigma_xx_to_ss),
            (model.sigma_ss_to_xx(e_cm), core_scalar.sigma_ss_to_xx),
            (model.sigma_xpi_to_xpi(e_cm), core_scalar.sigma_xpi_to_xpi),
            (model.sigma_xpi0_to_xpi0(e_cm), core_scalar.sigma_xpi0_to_xpi0),
            (model.sigma_xg_to_xg(e_cm), core_scalar.sigma_xg_to_xg),
            (model.sigma_xs_to_xs(e_cm), core_scalar.sigma_xs_to_xs),
        ]
        for got, kernel in pairs:
            assert got == kernel(e_cm, *args)

        # The wrapper passes `hazma.parameters`' lepton masses, not the
        # `.pyx`'s own -- 0.51099895 rather than 0.510998928 for the
        # electron. That was true before the port too (the Cython wrapper
        # is unchanged here), and it is why the module constants above
        # are *not* what a mixin method reaches the kernel with. The two
        # tables disagreeing is `rules.md` rule 4's territory, not this
        # task's.
        for tag, mf in (("e", electron_mass), ("mu", muon_mass)):
            assert model.sigma_xx_to_s_to_ff(e_cm, tag) == (
                core_scalar.sigma_xx_to_s_to_ff(e_cm, *args, mf)
            )
            assert model.sigma_xl_to_xl(e_cm, tag) == (
                core_scalar.sigma_xl_to_xl(e_cm, *args, mf)
            )
        assert model.thermal_cross_section(20.0) == (
            core_scalar.thermal_cross_section(20.0, *args)
        )
