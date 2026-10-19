"""The axial-vector mediator model and its annihilation rates."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import NoReturn

import numpy as np

from hazma import parameters
from hazma._core import axial_vector_mediator as _rates
from hazma.form_factors.axial import (
    MIXING_GUARD,
    AxialCurrent,
    AxialFormFactorGammaGamma,
    AxialFormFactorPi0Pi0Pi0,
    AxialFormFactorPiPiPi0,
    InvisibleCut,
    TriangleCouplings,
)
from hazma.form_factors.axial._sources import _finite_real
from hazma.form_factors.axial._three_pi import _real_array
from hazma.theory import TheoryAnn
from hazma.utils import RealOrRealArray  # noqa: TC001 - annotations resolve at runtime

from ._couplings import LeptonCouplings

_FERMION_PAIRS = ("e e", "mu mu", "tau tau", "ve ve", "vm vm", "vt vt")
_THREE_PIONS = ("pi pi pi0", "pi0 pi0 pi0")
#: Final states with no transverse density, which reach the dark matter only
#: through the axial contact: "g g" by Landau-Yang, and the LO three neutral
#: pions because only the pion pole produces them.
_LONGITUDINAL_ONLY = ("pi0 pi0 pi0", "g g")


def _scalar_or_array(value: np.ndarray) -> float | np.ndarray:
    return value.item() if value.ndim == 0 else value


def _parameter(name: str, doc: str, *, positive: bool = False) -> property:
    """A validated scalar parameter whose change recomputes the width."""
    attribute = f"_{name}"

    def get(self: AxialVectorMediator) -> float:
        return getattr(self, attribute)

    def set_(self: AxialVectorMediator, value: float) -> None:
        self._assign(attribute, _finite_real(value, name, positive=positive))

    return property(get, set_, doc=doc)


class AxialVectorMediator(TheoryAnn):
    r"""Spin-one mediator with vector and axial couplings to Dirac dark matter.

    The mediator couples as
    :math:`Z'_\mu\bar\chi\gamma^\mu(g_V^\chi + g_A^\chi\gamma_5)\chi`
    to the dark matter, through the axial light-quark current of ``quarks``
    to hadrons, and through ``leptons`` to the leptons.

    Parameters
    ----------
    mx : float
        Dark-matter mass in MeV, positive.
    mv : float
        Mediator mass in MeV, positive.
    gvxx, gaxx : float
        Dimensionless dark-matter vector and axial couplings.
    quarks : AxialCurrent
        Axial light-quark source :math:`G_A`. The mediator has no vector
        coupling to quarks.
    leptons : LeptonCouplings
        Charged-lepton and neutrino couplings.
    gacc : float
        Dimensionless charm axial coupling, which enters only the
        :math:`\gamma\gamma` triangle.
    theory : {"chpt"}
        Hadronic theory: ``"chpt"`` is U(3) LO chiral perturbation theory
        with the WZW action. ``"rcht"`` (HLS with an :math:`a_1` nonet) is
        a declared theory without an implementation and raises
        `NotImplementedError`. One instance uses exactly one theory.
    width : float, optional
        Total mediator width in MeV that replaces the sum of
        `partial_widths`, nonnegative. Required when a mediator decay this
        model does not compute is open.

    Notes
    -----
    Every rate follows the master formula of the mediator note
    (``eq:ax-master``),

    .. math::

        \sigma v = \frac{[g_V^{\chi2}(s + 2m_\chi^2)
        + g_A^{\chi2}(s - 4m_\chi^2)]A_T(s)}{s|s - M^2 + iM\Gamma|^2}
        + \frac{2m_\chi^2g_A^{\chi2}A_L(s)}{sM^4},

    with the transverse and source-longitudinal densities :math:`A_T`,
    :math:`A_L` of each final state in :math:`\mathrm{MeV}^2`. The width
    dresses only the transverse pole; the :math:`A_L` term is a
    non-resonant contact. A partial width is :math:`A_T(M^2)/(2M)`.

    :math:`\sigma v` is :math:`\sigma v_\mathrm{Mol}` per :math:`\chi\bar\chi`
    pair, and the cross section is :math:`\sigma = \sigma v/v_\mathrm{Mol}`
    with the centre-of-mass :math:`v_\mathrm{Mol} = 2\sqrt{1 - 4m_\chi^2/s}`.
    Neither carries a dark-matter population factor, such as the one a
    Dirac flux needs, nor the multiplicity of a particle in its final
    state, such as the two photons of ``"g g"``; the consumers apply those.

    The three-pion channels read only the isovector component of
    ``quarks``: in the isospin limit G parity forbids the octet and singlet
    directions from reaching three pions. Their LO action has one common
    pion mass, :data:`hazma.parameters.neutral_pion_mass`, so both open at
    three times it. They are declared up to 600 MeV, and ``"g g"`` up to
    1600 MeV; a rate past either raises, unless its couplings make it
    identically zero. With an axial dark-matter coupling, every rate raises
    at an energy where a meson-mixing parameter of
    `AxialMesonPoles.mixing_expansion` exceeds `MIXING_GUARD`, outside the
    first-order treatment of the poles, and a rate raises on an exact pole
    of zero width. Totals raise while a final state this model does not
    compute is open: the mediator pair ``"v v"`` and, for a source with
    octet or singlet directions, ``eta pi0 pi0`` and ``eta pi pi``.
    """

    def __init__(  # noqa: PLR0913 - the model's parameters
        self,
        mx: float,
        mv: float,
        gvxx: float,
        gaxx: float,
        *,
        quarks: AxialCurrent,
        leptons: LeptonCouplings,
        gacc: float,
        theory: str,
        width: float | None = None,
    ) -> None:
        if theory == "rcht":
            raise NotImplementedError(
                "The rcht hadronic theory has no implementation; use theory='chpt'."
            )
        if theory != "chpt":
            raise ValueError("theory must be 'chpt' or 'rcht'.")
        self._theory = theory
        self._mx = _finite_real(mx, "mx", positive=True)
        self._mv = _finite_real(mv, "mv", positive=True)
        self._gvxx = _finite_real(gvxx, "gvxx")
        self._gaxx = _finite_real(gaxx, "gaxx")
        self._gacc = _finite_real(gacc, "gacc")
        self._quarks = self._checked_quarks(quarks)
        self._leptons = self._checked_leptons(leptons)
        if width is not None:
            width = _finite_real(width, "width")
            if width < 0.0:
                raise ValueError("width must be nonnegative.")
        self._width_override = width
        self._three_pion_form_factors = {
            "pi pi pi0": AxialFormFactorPiPiPi0(),
            "pi0 pi0 pi0": AxialFormFactorPi0Pi0Pi0(),
        }
        self._gamma_gamma = AxialFormFactorGammaGamma()
        self._update_width()

    def __repr__(self) -> str:
        return (
            f"AxialVectorMediator(mx={self.mx} MeV, mv={self.mv} MeV, "
            f"gvxx={self.gvxx}, gaxx={self.gaxx}, quarks={self.quarks!r}, "
            f"leptons={self.leptons!r}, gacc={self.gacc}, theory={self.theory!r}, "
            f"width_v={self.width_v} MeV)"
        )

    mx = _parameter("mx", "Dark-matter mass in MeV.", positive=True)
    mv = _parameter("mv", "Mediator mass in MeV.", positive=True)
    gvxx = _parameter("gvxx", "Dimensionless dark-matter vector coupling.")
    gaxx = _parameter("gaxx", "Dimensionless dark-matter axial coupling.")
    gacc = _parameter("gacc", "Dimensionless charm axial coupling.")

    @property
    def quarks(self) -> AxialCurrent:
        """Axial light-quark source."""
        return self._quarks

    @quarks.setter
    def quarks(self, quarks: AxialCurrent) -> None:
        self._assign("_quarks", self._checked_quarks(quarks))

    @property
    def leptons(self) -> LeptonCouplings:
        """Charged-lepton and neutrino couplings."""
        return self._leptons

    @leptons.setter
    def leptons(self, leptons: LeptonCouplings) -> None:
        self._assign("_leptons", self._checked_leptons(leptons))

    @property
    def theory(self) -> str:
        """Hadronic theory, fixed at construction."""
        return self._theory

    @property
    def width_v(self) -> float:
        """Total mediator width in MeV: the override if given, else computed."""
        return self._width_v

    @staticmethod
    def _checked_quarks(quarks: AxialCurrent) -> AxialCurrent:
        if not isinstance(quarks, AxialCurrent):
            raise TypeError("quarks must be an AxialCurrent.")
        return quarks

    @staticmethod
    def _checked_leptons(leptons: LeptonCouplings) -> LeptonCouplings:
        if not isinstance(leptons, LeptonCouplings):
            raise TypeError("leptons must be a LeptonCouplings.")
        return leptons

    def _assign(self, attribute: str, value: object) -> None:
        """Set a parameter and recompute the width, or change neither."""
        previous = getattr(self, attribute)
        setattr(self, attribute, value)
        try:
            self._update_width()
        except ValueError:
            setattr(self, attribute, previous)
            raise

    def _update_width(self) -> None:
        if self._width_override is None:
            self._width_v = self.partial_widths()["total"]
        else:
            self._width_v = self._width_override

    def _fermion_pair(self, final_state: str) -> tuple[float, float, float]:
        """Return ``(mass in MeV, g_V, g_A)`` of a fermion-pair final state."""
        c = self._leptons
        return {
            "e e": (parameters.electron_mass, c.gvee, c.gaee),
            "mu mu": (parameters.muon_mass, c.gvmumu, c.gamumu),
            "tau tau": (parameters.tau_mass, c.gvtautau, c.gatautau),
            "ve ve": (0.0, 0.5 * c.gnue, -0.5 * c.gnue),
            "vm vm": (0.0, 0.5 * c.gnumu, -0.5 * c.gnumu),
            "vt vt": (0.0, 0.5 * c.gnutau, -0.5 * c.gnutau),
        }[final_state]

    def _isovector_quarks(self) -> AxialCurrent:
        """Return the T3 component of ``quarks``, the only one three pions see."""
        c3, _, _ = self._quarks.generator_coefficients
        return AxialCurrent.from_generator_coefficients(
            c3,
            0.0,
            0.0,
            qcd_scheme=self._quarks.qcd_scheme,
            scale_mev=self._quarks.scale_mev,
            provenance=f"isovector component of: {self._quarks.provenance}",
        )

    @staticmethod
    def _check_support(final_state: str, q: np.ndarray, q_max: float) -> None:
        if np.any(q > q_max):
            raise ValueError(
                f"The {final_state} rate is declared only up to {q_max} MeV."
            )

    def _densities(
        self, final_state: str, q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(A_T, A_L)`` in MeV² of a final state at 1-D energies ``q``."""
        if self._densities_vanish(final_state):
            return np.zeros_like(q), np.zeros_like(q)
        if final_state in _FERMION_PAIRS:
            mass, vector, axial = self._fermion_pair(final_state)
            return _rates.fermion_pair_densities(q, mass, vector, axial, 1.0)
        if final_state in _THREE_PIONS:
            form_factor = self._three_pion_form_factors[final_state]
            self._check_support(final_state, q, form_factor.q_max)
            result = form_factor.spectral_functions(q, self._isovector_quarks())
        else:
            self._check_support(final_state, q, self._gamma_gamma.q_max)
            result = self._gamma_gamma.spectral_functions(
                q, self._quarks, self._triangles(), invisible=self._invisible_cut()
            )
        return (
            np.ascontiguousarray(result.transverse, dtype=float),
            np.ascontiguousarray(result.source_longitudinal, dtype=float),
        )

    def _densities_vanish(self, final_state: str) -> bool:
        """Whether the Standard Model couplings give a final state no density.

        Such a channel is an exact zero at every energy, so it is neither
        range-checked nor listed among the thresholds.
        """
        if final_state in _FERMION_PAIRS:
            _, vector, axial = self._fermion_pair(final_state)
            return vector == 0.0 and axial == 0.0
        if final_state in _THREE_PIONS:
            return self._isovector_quarks().gau == 0.0
        c = self._leptons
        quarks = (self._quarks.gau, self._quarks.gad, self._quarks.gas)
        return not any((*quarks, c.gaee, c.gamumu, c.gatautau, self._gacc))

    def _rate_vanishes(self, final_state: str) -> bool:
        """Whether a final state's annihilation rate is identically zero."""
        needs_axial = final_state in _LONGITUDINAL_ONLY
        decoupled = self._gaxx == 0.0 and (needs_axial or self._gvxx == 0.0)
        return decoupled or self._densities_vanish(final_state)

    def _triangles(self) -> TriangleCouplings:
        c = self._leptons
        return TriangleCouplings(
            electron=c.gaee, muon=c.gamumu, tau=c.gatautau, charm=self._gacc
        )

    def _invisible_cut(self) -> InvisibleCut:
        # Annihilation needs s >= 4 mx², so the dark-matter pair cut is
        # always open beneath the poles that the rate probes.
        return InvisibleCut(
            dm_mass=self._mx, dm_axial=self._gaxx, mediator_mass=self._mv
        )

    def _check_mixing(self, q: np.ndarray) -> None:
        epsilon = self._gamma_gamma.poles.mixing_expansion(
            q, self._quarks, mediator_mass=self._mv
        )
        if np.any(np.max(epsilon, axis=0) > MIXING_GUARD):
            raise ValueError(
                "The meson mixing parameter exceeds MIXING_GUARD at this energy, "
                "outside the first-order treatment of the poles."
            )

    @staticmethod
    def list_annihilation_final_states() -> list[str]:
        """Return the annihilation final states this model computes."""
        return [*_FERMION_PAIRS, *_THREE_PIONS, "g g"]

    def _sigma_v(self, final_state: str, e_cm: RealOrRealArray) -> float | np.ndarray:
        energies = _real_array(e_cm, "e_cm")
        if np.any(energies < 0.0):
            raise ValueError("e_cm must be nonnegative.")
        flat = energies.reshape(-1)
        rate = np.zeros_like(flat)
        open_ = flat >= 2.0 * self._mx
        if np.any(open_) and not self._rate_vanishes(final_state):
            q = np.ascontiguousarray(flat[open_])
            if self._gaxx != 0.0:
                # The dropped second mixing insertion rescales only the
                # longitudinal contact, which needs an axial coupling.
                self._check_mixing(q)
            transverse, longitudinal = self._densities(final_state, q)
            dark = (self._mx, self._gvxx, self._gaxx, self._mv, self._width_v)
            values = _rates.sigma_v(q, transverse, longitudinal, dark)
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"The {final_state} rate is singular at an exact pole of zero "
                    "width."
                )
            rate[open_] = values
        return _scalar_or_array(rate.reshape(energies.shape))

    def _cross_section(
        self, final_state: str, e_cm: RealOrRealArray
    ) -> float | np.ndarray:
        sigma_v = np.asarray(self._sigma_v(final_state, e_cm), dtype=float)
        energies = np.asarray(e_cm, dtype=float)
        value = _rates.cross_section(
            energies.reshape(-1), sigma_v.reshape(-1), self._mx
        )
        return _scalar_or_array(value.reshape(energies.shape))

    def annihilation_sigma_v_funcs(
        self,
    ) -> dict[str, Callable[[RealOrRealArray], float | np.ndarray]]:
        r"""Return functions of ``e_cm`` (MeV) giving :math:`\sigma v_\mathrm{Mol}`.

        Each takes a scalar or array center-of-mass energy in MeV and returns
        :math:`\sigma v_\mathrm{Mol}` per :math:`\chi\bar\chi` pair in
        :math:`\mathrm{MeV}^{-2}` with its shape, zero below
        :math:`2m_\chi` and finite at threshold.
        """
        return {
            fs: partial(self._sigma_v, fs)
            for fs in self.list_annihilation_final_states()
        }

    def annihilation_cross_section_funcs(
        self,
    ) -> dict[str, Callable[[RealOrRealArray], float | np.ndarray]]:
        r"""Return functions of ``e_cm`` (MeV) giving the cross section.

        Each takes a scalar or array center-of-mass energy in MeV and returns
        :math:`\sigma = \sigma v_\mathrm{Mol}/v_\mathrm{Mol}` in
        :math:`\mathrm{MeV}^{-2}` with its shape: zero below :math:`2m_\chi`,
        and infinite at exactly :math:`2m_\chi` for a nonzero s-wave.
        """
        return {
            fs: partial(self._cross_section, fs)
            for fs in self.list_annihilation_final_states()
        }

    def _unmodeled_thresholds(self) -> dict[str, float]:
        """Return the opening energies in MeV of final states without a rate."""
        thresholds = {"v v": 2.0 * self._mv}
        _, c8, c0 = self._quarks.generator_coefficients
        if c8 != 0.0 or c0 != 0.0:
            eta = parameters.eta_mass
            thresholds["eta pi0 pi0"] = eta + 2.0 * parameters.neutral_pion_mass
            thresholds["eta pi pi"] = eta + 2.0 * parameters.charged_pion_mass
        return thresholds

    def _require_complete_annihilation(self, e_cm: RealOrRealArray) -> None:
        energies = _real_array(e_cm, "e_cm")
        physical = energies[energies >= 2.0 * self._mx]
        unmodeled = [
            fs
            for fs, threshold in self._unmodeled_thresholds().items()
            if np.any(physical > threshold)
        ]
        if unmodeled:
            raise ValueError(
                "The total annihilation rate is undefined at this energy: "
                f"{', '.join(unmodeled)} is open and this model has no rate for it."
            )

    def annihilation_cross_sections(
        self, e_cm: RealOrRealArray
    ) -> dict[str, float | np.ndarray]:
        r"""Return each cross section and their ``"total"``, in MeV⁻².

        Parameters
        ----------
        e_cm : float or array-like
            Center-of-mass energy in MeV.

        Raises
        ------
        ValueError
            If a final state this model does not compute is open, since the
            total would omit it.
        """
        self._require_complete_annihilation(e_cm)
        return super().annihilation_cross_sections(e_cm)

    def annihilation_sigma_v(
        self, e_cm: RealOrRealArray
    ) -> dict[str, float | np.ndarray]:
        r"""Return each :math:`\sigma v_\mathrm{Mol}` and their ``"total"``, in MeV⁻².

        Parameters
        ----------
        e_cm : float or array-like
            Center-of-mass energy in MeV.

        Raises
        ------
        ValueError
            If a final state this model does not compute is open, since the
            total would omit it.
        """
        self._require_complete_annihilation(e_cm)
        rates = {fs: fn(e_cm) for fs, fn in self.annihilation_sigma_v_funcs().items()}
        rates["total"] = sum(rates.values())
        return rates

    def annihilation_branching_fractions(self, e_cm: float) -> dict[str, float]:
        r"""Return each final state's share of the annihilation rate.

        Computed from :math:`\sigma v_\mathrm{Mol}`, whose ratios equal the
        cross sections' and stay finite at exactly ``2 mx``.

        Raises
        ------
        ValueError
            If a final state this model does not compute is open.
        """
        rates = self.annihilation_sigma_v(e_cm)
        total = rates.pop("total")
        if total == 0.0:
            return {fs: 0.0 for fs in rates}
        return {fs: rate / total for fs, rate in rates.items()}

    def annihilation_thresholds(self) -> dict[str, float]:
        """Return the center-of-mass energy in MeV at which each final state opens.

        A final state whose rate vanishes identically for these couplings
        never opens and is omitted.
        """
        thresholds = {fs: 2.0 * self._fermion_pair(fs)[0] for fs in _FERMION_PAIRS}
        for fs, form_factor in self._three_pion_form_factors.items():
            thresholds[fs] = form_factor.threshold
        thresholds["g g"] = 0.0
        return {
            fs: threshold
            for fs, threshold in thresholds.items()
            if not self._rate_vanishes(fs)
        }

    def annihilation_resonances(self) -> list[tuple[float, float]]:
        """Return ``(mass, width)`` in MeV of each annihilation resonance.

        The mediator, and each of the pi0, eta and eta' poles that the
        ``"g g"`` rate passes through at ``e_cm`` equal to its mass, if the
        quark source couples to it and that rate does not vanish.
        """
        if self._rate_vanishes("g g"):
            return [(self._mv, self._width_v)]
        poles = self._gamma_gamma.poles
        couplings = poles.source_couplings(self._quarks)
        mesons = [
            (poles.pion_mass, poles.pion_width),
            (poles.eta_mass, poles.eta_width),
            (poles.eta_prime_mass, poles.eta_prime_width),
        ]
        return [(self._mv, self._width_v)] + [
            meson for meson, c in zip(mesons, couplings, strict=True) if c != 0.0
        ]

    def partial_widths(self) -> dict[str, float]:
        r"""Return the mediator partial widths and their ``"total"``, in MeV.

        Each is :math:`A_T(M^2)/(2M)`, with ``"x x"`` the dark-matter pair.
        ``"g g"`` is absent because Landau-Yang forbids it.

        Raises
        ------
        ValueError
            If a decay this model does not compute is open, or the mediator
            mass is past a channel's declared range. A model in that case
            needs the ``width`` argument.
        """
        mv = self._mv
        unmodeled = [
            fs
            for fs, threshold in self._unmodeled_thresholds().items()
            if fs != "v v" and mv > threshold
        ]
        if unmodeled:
            raise ValueError(
                f"The total width is undefined: {', '.join(unmodeled)} is open "
                "and this model has no rate for it; pass width."
            )
        q = np.array([mv])
        widths = {
            fs: float(self._densities(fs, q)[0][0]) / (2.0 * mv)
            for fs in (*_FERMION_PAIRS, *_THREE_PIONS)
        }
        dm_pair, _ = _rates.fermion_pair_densities(
            q, self._mx, self._gvxx, self._gaxx, 1.0
        )
        widths["x x"] = float(dm_pair[0]) / (2.0 * mv)
        widths["total"] = sum(widths.values())
        return widths

    def _spectrum_funcs(self) -> NoReturn:
        raise NotImplementedError(
            "AxialVectorMediator does not compute continuum photon spectra."
        )

    def _positron_spectrum_funcs(self) -> NoReturn:
        raise NotImplementedError(
            "AxialVectorMediator does not compute continuum positron spectra."
        )

    def _gamma_ray_line_energies(self, e_cm: float) -> dict[str, float]:
        """Return the photon energy in MeV of each line final state."""
        return {"g g": e_cm / 2.0}

    def _positron_line_energies(self, e_cm: float) -> dict[str, float]:
        """Return the positron energy in MeV of each line final state."""
        return {"e e": e_cm / 2.0}

    def constraints(self) -> dict:
        """Return the model's constraints: none are declared."""
        return {}
