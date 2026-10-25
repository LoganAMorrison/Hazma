"""The axial-vector mediator model, its annihilation rates and its yields."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

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
_PRODUCTS = ("photon", "positron", "electron_positron", "neutrino")
_NEUTRINO_FLAVORS = {"ve ve": 0, "vm vm": 1, "vt vt": 2}
#: The fermion-pair decays with a cascade kernel, by the kernel's lepton name.
_CASCADE_LEPTONS = {
    "e e": "e",
    "mu mu": "mu",
    "ve ve": "ve",
    "vm vm": "vm",
    "vt vt": "vt",
}
#: The lepton pairs whose prompt photons take the Altarelli-Parisi
#: approximation; ``"pi pi pi0"`` radiates exactly.
_RADIATING_PAIRS = ("e e", "mu mu")
#: The π⁰ and π± masses of the LO three-pion current, one common mass.
_PION_MASSES = (parameters.neutral_pion_mass, parameters.neutral_pion_mass)
#: Absolute tolerance of a helicity density's Hermiticity, trace and spectrum.
_DENSITY_ATOL = 1e-12


def _line_multiplicity(final_state: str, product: str) -> float | np.ndarray | None:
    """Return a final state's primary-line multiplicity of ``product``, if any."""
    if product == "photon" and final_state == "g g":
        return 2.0
    if product == "positron" and final_state == "e e":
        return 1.0
    if product == "electron_positron" and final_state == "e e":
        return 2.0
    if product == "neutrino" and final_state in _NEUTRINO_FLAVORS:
        multiplicity = np.zeros(3)
        multiplicity[_NEUTRINO_FLAVORS[final_state]] = 2.0
        return multiplicity
    return None


def _scalar_or_array(value: np.ndarray) -> float | np.ndarray:
    return value.item() if value.ndim == 0 else value


def _helicity_fractions(density: np.ndarray) -> tuple[float, float, float]:
    """Validate a mediator's helicity density and return its diagonal.

    The density is a 3x3 matrix in the helicity basis (+1, 0, -1) along the
    mediator's flight direction; the diagonal is ``(F₊, F₀, F₋)``.
    """
    rho = np.asarray(density)
    if rho.shape != (3, 3) or not np.issubdtype(rho.dtype, np.number):
        raise ValueError("density must be a 3x3 numeric matrix.")
    rho = rho.astype(complex)
    if not np.all(np.isfinite(rho)):
        raise ValueError("density must be finite.")
    if np.max(np.abs(rho - rho.conj().T)) > _DENSITY_ATOL:
        raise ValueError("density must be Hermitian.")
    if abs(np.trace(rho) - 1.0) > _DENSITY_ATOL:
        raise ValueError("density must have unit trace.")
    if np.linalg.eigvalsh(rho).min() < -_DENSITY_ATOL:
        raise ValueError("density must be positive semidefinite.")
    plus, zero, minus = np.clip(np.diag(rho).real, 0.0, 1.0)
    return float(plus), float(zero), float(minus)


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

    `rest_frame_yields`, `rest_frame_counts` and `rest_frame_lines` give the
    photon, e⁺, e± and neutrino yields of each final state in the
    centre-of-mass frame, and `spectra` and `positron_spectra` weight their
    photon and e⁺ continua by the branching fractions.
    `mediator_decay_yields` and `mediator_decay_counts` give the same
    products of a moving mediator's decays from its helicity density, and
    `mediator_pair_yields` and `mediator_pair_counts` the inclusive yield of
    a mediator pair. None of them holds prompt radiation:
    `rest_frame_resolved_photons`, `mediator_decay_resolved_photons` and
    their counts give the real-emission photons above an infrared cut.
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
        unmodeled = [
            fs
            for fs, threshold in self._unmodeled_thresholds().items()
            if fs != "v v" and self._mv > threshold
        ]
        if unmodeled:
            raise ValueError(
                f"The total width is undefined: {', '.join(unmodeled)} is open "
                "and this model has no rate for it; pass width."
            )
        widths = self._visible_widths()
        dm_pair, _ = _rates.fermion_pair_densities(
            np.array([self._mv]), self._mx, self._gvxx, self._gaxx, 1.0
        )
        widths["x x"] = float(dm_pair[0]) / (2.0 * self._mv)
        widths["total"] = sum(widths.values())
        return widths

    def _visible_widths(self) -> dict[str, float]:
        """Return the partial widths in MeV of the decays into Standard Model states."""
        mv = self._mv
        q = np.array([mv])
        return {
            fs: float(self._densities(fs, q)[0][0]) / (2.0 * mv)
            for fs in (*_FERMION_PAIRS, *_THREE_PIONS)
        }

    def _rate_parts(self, final_state: str, e_cm: float) -> tuple[float, float]:
        r"""Return the transverse and longitudinal rates of an open final state.

        Both are :math:`\sigma v` in MeV⁻² at ``e_cm`` in MeV, and sum to its rate.
        """
        q = np.array([e_cm])
        if self._gaxx != 0.0:
            self._check_mixing(q)
        transverse, longitudinal = self._densities(final_state, q)
        dark = (self._mx, self._gvxx, self._gaxx, self._mv, self._width_v)
        zero = np.zeros_like(q)
        return (
            float(_rates.sigma_v(q, transverse, zero, dark)[0]),
            float(_rates.sigma_v(q, zero, longitudinal, dark)[0]),
        )

    def _muon_polarization(self, e_cm: float) -> float:
        """Return the effective polarization of each muon of ``"mu mu"``.

        It is the μ⁻ helicity, the helicity-weighted rate over the rate,
        and equals minus the μ⁺ helicity (see `rest_frame_yields`).
        """
        q = np.array([e_cm])
        mass, vector, axial = self._fermion_pair("mu mu")
        helicity = _rates.fermion_pair_helicity_density(q, mass, vector, axial, 1.0)
        dark = (self._mx, self._gvxx, self._gaxx, self._mv, self._width_v)
        weighted = float(_rates.sigma_v(q, helicity, np.zeros_like(q), dark)[0])
        return weighted / sum(self._rate_parts("mu mu", e_cm))

    def _open_final_states(self, e_cm: float) -> list[str]:
        """Return the final states with a nonzero rate at ``e_cm`` in MeV."""
        return [
            fs
            for fs in self.list_annihilation_final_states()
            if self._sigma_v(fs, e_cm) > 0.0
        ]

    def _continuum(
        self,
        final_state: str,
        product: str,
        points: np.ndarray,
        e_cm: float,
        *,
        binned: bool,
    ) -> np.ndarray:
        """Return one open final state's continuum yields as rows.

        The rows are ``[e⁺, e⁻]`` for ``"charged_lepton"``, the flavors for
        ``"neutrino"`` and one row for ``"photon"``, with a column per
        energy in ``points``, or per bin between its edges when ``binned``.
        """
        rows = {"photon": 1, "charged_lepton": 2, "neutrino": 3}[product]
        columns = points.size - 1 if binned else points.size
        if final_state == "tau tau":
            raise ValueError(
                "The tau tau yields are undefined: this model has no tau decay "
                "kernels."
            )
        if final_state == "mu mu":
            polarization = self._muon_polarization(e_cm)
            return _rates.muon_pair_yields(
                product, points, binned, e_cm / 2.0, polarization
            )
        if final_state in _THREE_PIONS:
            transverse, longitudinal = self._rate_parts(final_state, e_cm)
            parents = (
                final_state == "pi0 pi0 pi0",
                e_cm,
                parameters.neutral_pion_mass,
                transverse / (transverse + longitudinal),
            )
            return _rates.three_pion_yields(product, points, binned, parents)
        # Lepton and neutrino pairs and "g g" are primary lines alone.
        return np.zeros((rows, columns))

    @staticmethod
    def _product_rows(product: str, rows: np.ndarray) -> np.ndarray:
        """Select ``product``'s yield from a final state's continuum rows."""
        if product == "positron":
            return rows[0]
        if product == "electron_positron":
            return rows[0] + rows[1]
        if product == "photon":
            return rows[0]
        return rows

    @staticmethod
    def _kernel_product(product: str) -> str:
        if product not in _PRODUCTS:
            raise ValueError(f"product must be one of {', '.join(_PRODUCTS)}.")
        if product in ("positron", "electron_positron"):
            return "charged_lepton"
        return product

    @staticmethod
    def _check_energies(points: np.ndarray, product: str, name: str) -> None:
        if product == "photon" and np.any(points <= 0.0):
            raise ValueError(f"Photon {name} must be positive.")
        if np.any(points < 0.0):
            raise ValueError(f"{name} must be nonnegative.")

    @classmethod
    def _checked_edges(cls, edges: np.ndarray, product: str) -> np.ndarray:
        """Validate bin edges for ``product`` and return them contiguous."""
        points = _real_array(edges, "edges")
        if (
            points.ndim != 1
            or points.size < 2  # noqa: PLR2004 - a bin needs two edges
            or np.any(np.diff(points) <= 0.0)
        ):
            raise ValueError(
                "edges must be a strictly increasing 1-D array of two or more."
            )
        cls._check_energies(points, product, "edges")
        return np.ascontiguousarray(points)

    def rest_frame_yields(
        self, energies: RealOrRealArray, e_cm: float, *, product: str
    ) -> dict[str, float | np.ndarray]:
        r"""Return each open final state's continuum yield :math:`dN/dE`.

        The yield is per annihilation into that final state, in the
        :math:`\chi\bar\chi` centre-of-mass frame, with every particle of
        the state and of its decays counted and no branching fraction
        applied; `annihilation_branching_fractions` weights the states.
        Primary lines are excluded and given by `rest_frame_lines`;
        `rest_frame_counts` bins both.

        Parameters
        ----------
        energies : float or array_like
            Particle energies in MeV: total energies for e±, positive for
            photons and nonnegative otherwise.
        e_cm : float
            Center-of-mass energy in MeV.
        product : {"photon", "positron", "electron_positron", "neutrino"}
            ``"positron"`` counts e⁺ only, and ``"electron_positron"`` is the
            charge-summed e⁺ + e⁻ injection. ``"neutrino"`` is
            :math:`\nu + \bar\nu` per flavor (e, μ, τ).

        Returns
        -------
        dict of str to float or numpy.ndarray
            :math:`dN/dE` in MeV⁻¹ for each final state with a nonzero rate
            at ``e_cm``, shaped like ``energies``, with a leading flavor axis
            of length 3 for neutrinos.

        Raises
        ------
        ValueError
            If ``"tau tau"`` is open, since this model has no tau decay
            kernels, or if a rate raises at ``e_cm``.

        Notes
        -----
        Muons and charged pions decay through the Born V-A muon decay with
        a finite electron mass and the muon spin: fully polarized from
        :math:`\pi\to\mu\nu`, and in a ``"mu mu"`` pair with the μ⁻
        helicity of the chiral couplings, which CP gives the μ⁺ as minus
        its helicity, so the pair's e⁺ and e⁻ yields agree. The three
        pions of the LO currents share one mass,
        :data:`hazma.parameters.neutral_pion_mass`, and each decays at its
        production Lorentz factor with its physical mass. Their energy
        spectra mix the transverse and longitudinal Dalitz weights in
        proportion to each one's :math:`\sigma v`. Photons are
        :math:`\pi^0\to\gamma\gamma` and Hazma's unpolarized radiative
        muon and charged-pion decays; the :math:`\pi^0` Dalitz decay is not
        included, and the prompt radiation is `rest_frame_resolved_photons`.
        """
        kernel_product = self._kernel_product(product)
        values = _real_array(energies, "energies")
        self._check_energies(values, product, "energies")
        e_cm = _finite_real(e_cm, "e_cm", positive=True)
        points = np.ascontiguousarray(values.reshape(-1))
        yields = {}
        for fs in self._open_final_states(e_cm):
            rows = self._continuum(fs, kernel_product, points, e_cm, binned=False)
            selected = self._product_rows(product, rows)
            shape = (*selected.shape[:-1], *values.shape)
            yields[fs] = _scalar_or_array(selected.reshape(shape))
        return yields

    def rest_frame_lines(
        self, e_cm: float, *, product: str
    ) -> dict[str, dict[str, float | np.ndarray]]:
        """Return each open final state's primary lines.

        Parameters
        ----------
        e_cm : float
            Center-of-mass energy in MeV.
        product : {"photon", "positron", "electron_positron", "neutrino"}
            As in `rest_frame_yields`.

        Returns
        -------
        dict of str to dict
            For each final state with a line and a nonzero rate,
            ``"energy"`` in MeV and ``"multiplicity"``, the particles per
            annihilation into that state, an array over the flavors
            (e, μ, τ) for neutrinos.
        """
        self._kernel_product(product)
        e_cm = _finite_real(e_cm, "e_cm", positive=True)
        lines = {}
        for fs in self._open_final_states(e_cm):
            multiplicity = _line_multiplicity(fs, product)
            if multiplicity is not None:
                lines[fs] = {"energy": e_cm / 2.0, "multiplicity": multiplicity}
        return lines

    def rest_frame_counts(
        self, edges: np.ndarray, e_cm: float, *, product: str
    ) -> dict[str, np.ndarray]:
        """Return each open final state's yield in energy bins, lines included.

        Bin ``i`` is ``[edges[i], edges[i + 1])``, and a line exactly at the
        last edge belongs to the last bin. The continuum of a bin of width
        ``ΔE`` tends to ``rest_frame_yields(E) ΔE`` as it shrinks.

        Parameters
        ----------
        edges : array_like
            Strictly increasing bin edges in MeV, at least two: total
            energies for e±, positive for photons and nonnegative otherwise.
        e_cm : float
            Center-of-mass energy in MeV.
        product : {"photon", "positron", "electron_positron", "neutrino"}
            As in `rest_frame_yields`.

        Returns
        -------
        dict of str to numpy.ndarray
            Particles per annihilation into each final state with a nonzero
            rate, one per bin, with a leading flavor axis of length 3 for
            neutrinos.

        Raises
        ------
        ValueError
            As `rest_frame_yields` does.
        """
        kernel_product = self._kernel_product(product)
        points = self._checked_edges(edges, product)
        e_cm = _finite_real(e_cm, "e_cm", positive=True)
        counts = {}
        for fs in self._open_final_states(e_cm):
            rows = self._continuum(fs, kernel_product, points, e_cm, binned=True)
            selected = np.array(self._product_rows(product, rows))
            multiplicity = _line_multiplicity(fs, product)
            if multiplicity is not None:
                index = np.searchsorted(points, e_cm / 2.0, side="right") - 1
                if e_cm / 2.0 == points[-1]:
                    index = points.size - 2
                if 0 <= index < points.size - 1:
                    selected[..., index] += multiplicity
            counts[fs] = selected
        return counts

    def _radiation(
        self,
        final_state: str,
        points: np.ndarray,
        e_cm: float,
        cut: float,
        *,
        binned: bool,
    ) -> np.ndarray:
        """Return one open final state's resolved photons, as `_continuum` does."""
        emission = (parameters.alpha_em, cut)
        if final_state == "tau tau":
            raise ValueError(
                "The tau tau yields are undefined: this model has no tau decay "
                "kernels."
            )
        if final_state == "pi pi pi0":
            transverse, longitudinal = self._rate_parts(final_state, e_cm)
            fraction = transverse / (transverse + longitudinal)
            return _rates.three_pion_radiation(
                points, binned, e_cm, _PION_MASSES, fraction, emission
            )
        if final_state in _RADIATING_PAIRS:
            mass = self._fermion_pair(final_state)[0]
            return _rates.pair_radiation(points, binned, e_cm, mass, emission)
        return np.zeros(points.size - 1 if binned else points.size)

    def rest_frame_resolved_photons(
        self, energies: RealOrRealArray, e_cm: float, *, cut: float
    ) -> dict[str, float | np.ndarray]:
        r"""Return each open final state's resolved prompt photons :math:`dN/dE`.

        The yield is the real emission at :math:`O(\alpha)` per Born
        annihilation into that final state, in the :math:`\chi\bar\chi`
        centre-of-mass frame, of photons above ``cut`` in that frame. It adds
        to the photons of `rest_frame_yields`, which hold no prompt
        radiation, and like them carries no branching fraction.

        Parameters
        ----------
        energies : float or array_like
            Positive photon energies in MeV.
        e_cm : float
            Center-of-mass energy in MeV.
        cut : float
            Infrared cut on the photon energy in MeV, positive.

        Returns
        -------
        dict of str to float or numpy.ndarray
            :math:`dN/dE` in MeV⁻¹ for each final state with a nonzero rate
            at ``e_cm``, shaped like ``energies``, and zero for the states
            without charged particles.

        Raises
        ------
        ValueError
            As `rest_frame_yields` does, or if ``cut`` is not positive.

        Notes
        -----
        ``"pi pi pi0"`` is exact at tree level within the LO current. In
        sigma-model coordinates the photon couples only to the external charged
        pions, whose Born vertex sees only their total momentum, so the
        amplitude is the Born current at :math:`\tilde s = (q - p_0)^2`
        times :math:`e[p_+\cdot\epsilon^*/(p_+\cdot k)
        - p_-\cdot\epsilon^*/(p_-\cdot k)]`, for the transverse and the
        source-longitudinal densities alike and at any photon energy. The
        pions share :data:`hazma.parameters.neutral_pion_mass`, as in
        `rest_frame_yields`, and near threshold the yield is sensitive to
        that choice. ``"e e"`` and ``"mu mu"`` take the Altarelli-Parisi
        spectrum of `hazma.spectra.altarelli_parisi.dnde_photon_ap_fermion`
        for each lepton, a leading-log approximation. Every final state
        takes exactly one of the two, and Hazma's generic final-state
        radiation is never added, so no emission is counted twice. The
        virtual corrections, which cancel the logarithm of ``cut``, and the
        :math:`O(\alpha)` recoil and virtual corrections to the decay
        yields are not included.
        """
        values = _real_array(energies, "energies")
        self._check_energies(values, "photon", "energies")
        e_cm = _finite_real(e_cm, "e_cm", positive=True)
        cut = _finite_real(cut, "cut", positive=True)
        points = np.ascontiguousarray(values.reshape(-1))
        return {
            fs: _scalar_or_array(
                self._radiation(fs, points, e_cm, cut, binned=False).reshape(
                    values.shape
                )
            )
            for fs in self._open_final_states(e_cm)
        }

    def rest_frame_resolved_photon_counts(
        self, edges: np.ndarray, e_cm: float, *, cut: float
    ) -> dict[str, np.ndarray]:
        """Return each open final state's resolved prompt photons in energy bins.

        Bin ``i`` is ``[edges[i], edges[i + 1])``, and a bin of width ``ΔE``
        tends to ``rest_frame_resolved_photons(E) ΔE`` as it shrinks.

        Parameters
        ----------
        edges : array_like
            Strictly increasing positive photon bin edges in MeV, at least
            two.
        e_cm : float
            Center-of-mass energy in MeV.
        cut : float
            Infrared cut on the photon energy in MeV, positive.

        Returns
        -------
        dict of str to numpy.ndarray
            Photons per Born annihilation into each final state with a
            nonzero rate, one per bin.

        Raises
        ------
        ValueError
            As `rest_frame_resolved_photons` does.
        """
        points = self._checked_edges(edges, "photon")
        e_cm = _finite_real(e_cm, "e_cm", positive=True)
        cut = _finite_real(cut, "cut", positive=True)
        return {
            fs: self._radiation(fs, points, e_cm, cut, binned=True)
            for fs in self._open_final_states(e_cm)
        }

    def _mediator_decays(
        self, mediator_energy: float, density: np.ndarray
    ) -> tuple[tuple[float, float], tuple[float, float, float], list[str]]:
        """Return ``(mass, energy)``, the helicity fractions and the open decays.

        The open decays are those into Standard Model states with a positive
        partial width; the dark-matter pair leaves nothing visible.
        """
        energy = _finite_real(mediator_energy, "mediator_energy", positive=True)
        if energy <= self._mv:
            raise ValueError(
                "mediator_energy must exceed the mediator mass: a mediator at rest "
                "decays into lines."
            )
        helicities = _helicity_fractions(density)
        decays = [fs for fs, width in self._visible_widths().items() if width > 0.0]
        if "tau tau" in decays:
            raise ValueError(
                "The tau tau cascade is undefined: this model has no tau decay "
                "kernels."
            )
        return (self._mv, energy), helicities, decays

    def _cascade(
        self,
        final_state: str,
        product: str,
        points: np.ndarray,
        state: tuple[tuple[float, float], tuple[float, float, float]],
        *,
        binned: bool,
    ) -> np.ndarray:
        """Return one decay's yields of ``product`` as rows, as `_continuum` does."""
        mediator, helicities = state
        if final_state == "pi pi pi0":
            return _rates.three_pion_cascade_yields(
                product,
                points,
                binned,
                parameters.neutral_pion_mass,
                mediator,
                helicities,
            )
        return _rates.pair_cascade_yields(
            product,
            points,
            binned,
            _CASCADE_LEPTONS[final_state],
            self._fermion_pair(final_state),
            mediator,
            helicities,
        )

    def mediator_decay_yields(
        self,
        energies: RealOrRealArray,
        mediator_energy: float,
        *,
        product: str,
        density: np.ndarray,
    ) -> dict[str, float | np.ndarray]:
        r"""Return the lab yield :math:`dN/dE` of each open mediator decay.

        The yield is per decay of one mediator of lab energy
        ``mediator_energy`` into that final state, with every particle of
        the state and of its decays counted and no branching fraction
        applied; `mediator_pair_yields` gives the inclusive yield of a
        pair. Each decay is boosted from the mediator's rest frame with the
        angular distribution its helicity density gives each daughter.

        Parameters
        ----------
        energies : float or array_like
            Lab particle energies in MeV: total energies for e±, positive for
            photons and nonnegative otherwise.
        mediator_energy : float
            The mediator's lab energy in MeV, above its mass.
        product : {"photon", "positron", "electron_positron", "neutrino"}
            As in `rest_frame_yields`.
        density : array_like
            The mediator's 3x3 helicity density in the basis (+1, 0, -1)
            along its flight direction: Hermitian, positive semidefinite and
            of unit trace. ``numpy.eye(3) / 3`` is the unpolarized mediator.

        Returns
        -------
        dict of str to float or numpy.ndarray
            :math:`dN/dE` in MeV⁻¹ for each decay with a positive partial
            width, shaped like ``energies``, with a leading flavor axis of
            length 3 for neutrinos.

        Raises
        ------
        ValueError
            If ``density`` is not a helicity density, ``mediator_energy`` is
            not above the mass, ``"tau tau"`` is open, or a partial width
            raises.

        Notes
        -----
        The yields depend on the density only through its diagonal, the
        helicity fractions :math:`F_\lambda`, because the off-diagonal
        elements carry the daughters' azimuth about the flight direction.
        A daughter's angular density about that direction is
        :math:`[1 + g(F_+ - F_-)P_1 + h(3F_0 - 1)P_2]/2`, so
        :math:`F_0 = 1/3` with :math:`F_+ = F_-` boosts an isotropic decay;
        pair production at threshold is transverse, :math:`F_0 = 0`. The
        fermion pairs take their analyzing powers and each fermion's spin
        from the tree-level helicity amplitudes, and a muon decays with its
        lab helicity, which the boost mixes with its spin in the decay
        plane. ``"pi pi pi0"`` comes from the transverse LO current alone,
        with every pion at :data:`hazma.parameters.neutral_pion_mass` as in
        `rest_frame_yields`. Radiative muon and charged-pion photons stay
        unpolarized, the :math:`\pi^0` Dalitz decay is not included, and
        the prompt radiation is `mediator_decay_resolved_photons`.
        """
        kernel_product = self._kernel_product(product)
        values = _real_array(energies, "energies")
        self._check_energies(values, product, "energies")
        mediator, helicities, decays = self._mediator_decays(mediator_energy, density)
        points = np.ascontiguousarray(values.reshape(-1))
        yields = {}
        for fs in decays:
            rows = self._cascade(
                fs, kernel_product, points, (mediator, helicities), binned=False
            )
            selected = self._product_rows(product, rows)
            shape = (*selected.shape[:-1], *values.shape)
            yields[fs] = _scalar_or_array(selected.reshape(shape))
        return yields

    def mediator_decay_counts(
        self,
        edges: np.ndarray,
        mediator_energy: float,
        *,
        product: str,
        density: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return each open mediator decay's lab yield in energy bins.

        Bin ``i`` is ``[edges[i], edges[i + 1])``. A moving mediator has no
        lines, and a bin of width ``ΔE`` tends to
        ``mediator_decay_yields(E) ΔE`` as it shrinks.

        Parameters
        ----------
        edges : array_like
            Strictly increasing lab bin edges in MeV, at least two: total
            energies for e±, positive for photons and nonnegative otherwise.
        mediator_energy : float
            The mediator's lab energy in MeV, above its mass.
        product : {"photon", "positron", "electron_positron", "neutrino"}
            As in `rest_frame_yields`.
        density : array_like
            The mediator's helicity density, as in `mediator_decay_yields`.

        Returns
        -------
        dict of str to numpy.ndarray
            Particles per decay into each decay with a positive partial
            width, one per bin, with a leading flavor axis of length 3 for
            neutrinos.

        Raises
        ------
        ValueError
            As `mediator_decay_yields` does.
        """
        kernel_product = self._kernel_product(product)
        points = self._checked_edges(edges, product)
        mediator, helicities, decays = self._mediator_decays(mediator_energy, density)
        return {
            fs: self._product_rows(
                product,
                self._cascade(
                    fs, kernel_product, points, (mediator, helicities), binned=True
                ),
            )
            for fs in decays
        }

    def _cascade_radiation(
        self,
        final_state: str,
        points: np.ndarray,
        state: tuple[tuple[float, float], tuple[float, float, float]],
        cut: float,
        *,
        binned: bool,
    ) -> np.ndarray:
        """Return one decay's resolved photons, as `_cascade` does."""
        mediator, helicities = state
        emission = (parameters.alpha_em, cut)
        if final_state == "pi pi pi0":
            return _rates.three_pion_radiation_cascade(
                points, binned, _PION_MASSES, emission, mediator, helicities
            )
        if final_state in _RADIATING_PAIRS:
            return _rates.pair_radiation_cascade(
                points,
                binned,
                self._fermion_pair(final_state),
                emission,
                mediator,
                helicities,
            )
        return np.zeros(points.size - 1 if binned else points.size)

    def mediator_decay_resolved_photons(
        self,
        energies: RealOrRealArray,
        mediator_energy: float,
        *,
        density: np.ndarray,
        cut: float,
    ) -> dict[str, float | np.ndarray]:
        r"""Return the lab resolved prompt photons :math:`dN/dE` of each open decay.

        The yield is the real emission at :math:`O(\alpha)` per Born decay of
        one mediator of lab energy ``mediator_energy`` into that final
        state, of photons above ``cut`` in the mediator's rest frame. It adds
        to the photons of `mediator_decay_yields`.

        Parameters
        ----------
        energies : float or array_like
            Positive lab photon energies in MeV.
        mediator_energy : float
            The mediator's lab energy in MeV, above its mass.
        density : array_like
            The mediator's helicity density, as in `mediator_decay_yields`.
        cut : float
            Infrared cut on the rest-frame photon energy in MeV, positive.

        Returns
        -------
        dict of str to float or numpy.ndarray
            :math:`dN/dE` in MeV⁻¹ for each decay with a positive partial
            width, shaped like ``energies``, and zero for the decays without
            charged particles.

        Raises
        ------
        ValueError
            As `mediator_decay_yields` does, or if ``cut`` is not positive.

        Notes
        -----
        The emission is that of `rest_frame_resolved_photons` at the
        mediator mass. ``"pi pi pi0"`` comes from the transverse current an
        on-shell mediator has, and a photon of rest-frame energy ω has the
        angular density :math:`[f + (3F_0 - 1)h(\omega)P_2(\cos\theta)]/2`
        about the flight direction, with no :math:`P_1` term because the
        current is a polar vector. The Altarelli-Parisi photons of ``"e e"``
        and ``"mu mu"`` are collinear with their leptons and take the
        pair's angular density, whose :math:`P_1` terms cancel between the
        two leptons; the spin correlations of the emission itself are not
        modeled.
        """
        values = _real_array(energies, "energies")
        self._check_energies(values, "photon", "energies")
        cut = _finite_real(cut, "cut", positive=True)
        mediator, helicities, decays = self._mediator_decays(mediator_energy, density)
        points = np.ascontiguousarray(values.reshape(-1))
        state = (mediator, helicities)
        return {
            fs: _scalar_or_array(
                self._cascade_radiation(fs, points, state, cut, binned=False).reshape(
                    values.shape
                )
            )
            for fs in decays
        }

    def mediator_decay_resolved_photon_counts(
        self,
        edges: np.ndarray,
        mediator_energy: float,
        *,
        density: np.ndarray,
        cut: float,
    ) -> dict[str, np.ndarray]:
        """Return each open decay's lab resolved prompt photons in energy bins.

        Bin ``i`` is ``[edges[i], edges[i + 1])``, and a bin of width ``ΔE``
        tends to ``mediator_decay_resolved_photons(E) ΔE`` as it shrinks.

        Parameters
        ----------
        edges : array_like
            Strictly increasing positive lab photon bin edges in MeV, at
            least two.
        mediator_energy : float
            The mediator's lab energy in MeV, above its mass.
        density : array_like
            The mediator's helicity density, as in `mediator_decay_yields`.
        cut : float
            Infrared cut on the rest-frame photon energy in MeV, positive.

        Returns
        -------
        dict of str to numpy.ndarray
            Photons per Born decay into each decay with a positive partial
            width, one per bin.

        Raises
        ------
        ValueError
            As `mediator_decay_resolved_photons` does.
        """
        points = self._checked_edges(edges, "photon")
        cut = _finite_real(cut, "cut", positive=True)
        mediator, helicities, decays = self._mediator_decays(mediator_energy, density)
        state = (mediator, helicities)
        return {
            fs: self._cascade_radiation(fs, points, state, cut, binned=True)
            for fs in decays
        }

    def _pair_branching_fractions(self) -> dict[str, float]:
        """Return each open decay's partial width over the total `width_v`."""
        widths = {
            fs: width
            for fs, width in self.partial_widths().items()
            if fs != "total" and width > 0.0
        }
        if widths and self.width_v <= 0.0:
            raise ValueError(
                "The pair yield needs a positive total width, but width is zero "
                "while decays are open."
            )
        return {fs: width / self.width_v for fs, width in widths.items()}

    @staticmethod
    def _inclusive_pair(
        per_decay: dict[str, float | np.ndarray],
        branching: dict[str, float],
        shape: tuple[int, ...],
    ) -> np.ndarray:
        """Return ``2 Σ_f B_f K_f`` over the per-decay yields ``K_f``."""
        total = np.zeros(shape)
        for fs, value in per_decay.items():
            total += 2.0 * branching[fs] * np.asarray(value)
        return total

    def mediator_pair_yields(
        self,
        energies: RealOrRealArray,
        mediator_energy: float,
        *,
        product: str,
        density: np.ndarray,
    ) -> float | np.ndarray:
        r"""Return the inclusive lab yield :math:`dN/dE` of a mediator pair.

        It is :math:`2\sum_f B_f K_f`, with :math:`K_f` the
        `mediator_decay_yields` of decay :math:`f` and :math:`B_f` its
        branching fraction :math:`\Gamma_f` from `partial_widths` over the
        total width `width_v`, for two mediators of lab
        energy ``mediator_energy`` sharing one helicity density, such as the
        pair of :math:`\chi\bar\chi \to Z'Z'` at threshold. It counts the
        pairs in which only one mediator decays into :math:`f`; the
        probability that both do is :math:`B_f^2`, which is not this yield.

        Parameters
        ----------
        energies, mediator_energy, product, density
            As in `mediator_decay_yields`.

        Returns
        -------
        float or numpy.ndarray
            :math:`dN/dE` in MeV⁻¹ per pair, shaped like ``energies``, with a
            leading flavor axis of length 3 for neutrinos.

        Raises
        ------
        ValueError
            As `mediator_decay_yields` does, when `partial_widths` raises
            because a decay this model does not compute is open, even when a
            ``width`` override supplies the total, and when a ``width`` of
            zero leaves open decays without a branching fraction.
        """
        branching = self._pair_branching_fractions()
        per_decay = self.mediator_decay_yields(
            energies, mediator_energy, product=product, density=density
        )
        shape = np.shape(energies)
        if product == "neutrino":
            shape = (3, *shape)
        return _scalar_or_array(self._inclusive_pair(per_decay, branching, shape))

    def mediator_pair_counts(
        self,
        edges: np.ndarray,
        mediator_energy: float,
        *,
        product: str,
        density: np.ndarray,
    ) -> np.ndarray:
        """Return the inclusive lab yield of a mediator pair in energy bins.

        It weights `mediator_decay_counts` as `mediator_pair_yields` weights
        `mediator_decay_yields`.

        Parameters
        ----------
        edges, mediator_energy, product, density
            As in `mediator_decay_counts`.

        Returns
        -------
        numpy.ndarray
            Particles per pair, one per bin, with a leading flavor axis of
            length 3 for neutrinos.

        Raises
        ------
        ValueError
            As `mediator_pair_yields` does.
        """
        branching = self._pair_branching_fractions()
        per_decay = self.mediator_decay_counts(
            edges, mediator_energy, product=product, density=density
        )
        bins = np.size(edges) - 1
        shape = (3, bins) if product == "neutrino" else (bins,)
        return self._inclusive_pair(per_decay, branching, shape)

    def _spectrum_funcs(
        self,
    ) -> dict[str, Callable[[RealOrRealArray, float], float | np.ndarray]]:
        return {
            fs: partial(self._final_state_yield, fs, "photon")
            for fs in self.list_annihilation_final_states()
        }

    def _positron_spectrum_funcs(
        self,
    ) -> dict[str, Callable[[RealOrRealArray, float], float | np.ndarray]]:
        return {
            fs: partial(self._final_state_yield, fs, "positron")
            for fs in self.list_annihilation_final_states()
        }

    def _final_state_yield(
        self, final_state: str, product: str, energies: RealOrRealArray, e_cm: float
    ) -> float | np.ndarray:
        """Return one final state's continuum yield, as `rest_frame_yields`."""
        values = _real_array(energies, "energies")
        self._check_energies(values, product, "energies")
        e_cm = _finite_real(e_cm, "e_cm", positive=True)
        points = np.ascontiguousarray(values.reshape(-1))
        rows = self._continuum(
            final_state, self._kernel_product(product), points, e_cm, binned=False
        )
        return _scalar_or_array(self._product_rows(product, rows).reshape(values.shape))

    def _gamma_ray_line_energies(self, e_cm: float) -> dict[str, float]:
        """Return the photon energy in MeV of each line final state."""
        return {"g g": e_cm / 2.0}

    def _positron_line_energies(self, e_cm: float) -> dict[str, float]:
        """Return the positron energy in MeV of each line final state."""
        return {"e e": e_cm / 2.0}

    def constraints(self) -> dict:
        """Return the model's constraints: none are declared."""
        return {}
