r"""Coherent longitudinal pi0, eta and eta' poles of the neutral axial source.

The source ``Z_mu qbar gamma^mu gamma5 G_A q`` mixes with each neutral
pseudoscalar through ``-c_P Z'.dP``, ``c_P = f Tr(G_A lambda_P)`` in MeV.
The mixing is proportional to the source momentum, so it contributes only
to the source-longitudinal density ``A_L``; through the Proca term
``k^mu k^nu/(s M**2)`` it gives the dark matter a non-resonant pseudoscalar
contact, not a spin-zero mediator pole. The three mesons are resummed
together through the inverse of their full propagator matrix, and one
insertion of the source-meson mixing is kept. The guard
``epsilon_P = |c_P**2 s/(M**2 (s - m_P**2 + i m_P Gamma_P))|`` measures the
neglected second insertion; points above :data:`MIXING_GUARD` are outside
this first-order treatment and a rate must reject them.

The physical states use the single-angle LO U(3) mixing
``lambda_eta = cos(theta_P) T8 - sin(theta_P) T0`` and
``lambda_eta' = sin(theta_P) T8 + cos(theta_P) T0`` with
``Tr(lambda_P lambda_Q) = 2 delta_PQ``, in the isospin limit. The
derivation and its conventions are the project's ADR-0003.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from hazma import parameters
from hazma._core import axial as _axial
from hazma.utils import RealOrRealArray

from ._base import AxialSpectralResult
from ._sources import AxialCurrent, SourceCoverage
from ._three_pi import _real_array, _scalar

ArrayLike = RealOrRealArray | list[float] | tuple[float, ...]

MIXING_GUARD = 0.1
"""Largest supported mixing expansion parameter ``epsilon_P``, dimensionless."""

_GAMMA_GAMMA_MAX_MEV = 1600.0


def _flavor_couplings(couplings: AxialCurrent) -> tuple[float, float, float]:
    if not isinstance(couplings, AxialCurrent):
        raise TypeError("couplings must be an AxialCurrent with source metadata.")
    return couplings.gau, couplings.gad, couplings.gas


def _energies(q: ArrayLike, q_max: float = np.inf) -> np.ndarray:
    energies = _real_array(q, "q")
    if np.any(energies < 0) or np.any(energies > q_max):
        raise ValueError("q must be nonnegative and within the declared support.")
    return energies


def _finite(value: np.ndarray, name: str) -> np.ndarray:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} is singular at an exact pole of zero width.")
    return value


def _scalar_or_array(value: np.ndarray) -> complex | float | np.ndarray:
    return value.item() if value.ndim == 0 else value


@dataclass(frozen=True, slots=True)
class InvisibleCut:
    r"""One insertion of the dark-matter pair cut in the meson self-energy.

    Parameters
    ----------
    dm_mass : float
        Dirac dark-matter mass in MeV, nonnegative.
    dm_axial : float
        Dimensionless dark-matter axial coupling ``g_A^chi``.
    mediator_mass : float
        Proca mediator mass ``M`` in MeV, positive.

    Notes
    -----
    Each meson couples to ``chibar gamma5 chi`` with ``y_P = kappa c_P``,
    ``kappa = 2 m_chi g_A^chi/M**2``. The cut's absorptive part
    ``i s beta_chi kappa**2 c c^T/(8 pi)`` is added to the full meson
    matrix, so it also couples the mesons; its dispersive part renormalizes
    the pole inputs and is not included. ``dm_mass = 0`` gives ``kappa = 0``,
    a cut that changes nothing. Within :data:`MIXING_GUARD` the width it adds
    is a subleading correction: ``Gamma_inv/Gamma_P = epsilon_P Pi/M**2`` at
    each pole, with ``Pi = beta_chi (m_chi g_A^chi)**2/(2 pi)``.
    """

    dm_mass: float
    dm_axial: float
    mediator_mass: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dm_mass", _scalar(self.dm_mass, "dm_mass", positive=False)
        )
        axial = _real_array(self.dm_axial, "dm_axial")
        if axial.ndim != 0:
            raise ValueError("dm_axial must be a real scalar.")
        object.__setattr__(self, "dm_axial", float(axial))
        object.__setattr__(
            self,
            "mediator_mass",
            _scalar(self.mediator_mass, "mediator_mass", positive=True),
        )

    @property
    def strength(self) -> float:
        """Return ``kappa = 2 m_chi g_A^chi/M**2`` in 1/MeV."""
        return 2.0 * self.dm_mass * self.dm_axial / self.mediator_mass**2

    @property
    def _kernel(self) -> tuple[float, float]:
        return self.dm_mass, self.strength


@dataclass(frozen=True, slots=True)
class TriangleCouplings:
    r"""Dimensionless axial couplings of the fermions in the gamma gamma triangles.

    Parameters
    ----------
    electron, muon, tau, charm : float
        Couplings ``g_A^f`` in ``L = Z_mu fbar gamma^mu gamma5 g_A^f f``.
        The light quarks are not listed: their anomaly is carried by the
        meson poles. All four are required, so an omitted triangle is a
        declared zero.
    """

    electron: float
    muon: float
    tau: float
    charm: float

    def __post_init__(self) -> None:
        for name in ("electron", "muon", "tau", "charm"):
            value = _real_array(getattr(self, name), name)
            if value.ndim != 0:
                raise ValueError(f"{name} must be a real scalar.")
            object.__setattr__(self, name, float(value))

    @property
    def _kernel(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        masses = (
            parameters.electron_mass,
            parameters.muon_mass,
            parameters.tau_mass,
            parameters.charm_quark_mass,
        )
        return masses, (self.electron, self.muon, self.tau, self.charm)


@dataclass(frozen=True, kw_only=True)
class AxialMesonPoles:
    r"""The LO U(3) neutral pseudoscalar block that the axial source mixes with.

    Keyword-only masses and widths are in MeV, in the order (pi0, eta,
    eta'). The defaults are the PDG 2024 pole masses and total widths
    7.8e-6, 1.31e-3 and 0.188 MeV, the single mixing angle
    ``mixing_angle_deg = -15.4`` degrees, and ``decay_constant = 92.2`` MeV
    for the common octet and singlet ``f``. Zero masses and widths are
    allowed for chiral-limit and residue checks; an exact pole of zero width
    is then singular and raises.
    """

    pion_mass: float = parameters.neutral_pion_mass
    eta_mass: float = parameters.eta_mass
    eta_prime_mass: float = parameters.eta_prime_mass
    pion_width: float = 7.8e-6
    eta_width: float = 1.31e-3
    eta_prime_width: float = 0.188
    mixing_angle_deg: float = -15.4
    decay_constant: float = 92.2

    def __post_init__(self) -> None:
        for name in (
            "pion_mass",
            "eta_mass",
            "eta_prime_mass",
            "pion_width",
            "eta_width",
            "eta_prime_width",
        ):
            object.__setattr__(
                self, name, _scalar(getattr(self, name), name, positive=False)
            )
        angle = _real_array(self.mixing_angle_deg, "mixing_angle_deg")
        if angle.ndim != 0:
            raise ValueError("mixing_angle_deg must be a real scalar.")
        object.__setattr__(self, "mixing_angle_deg", float(angle))
        object.__setattr__(
            self,
            "decay_constant",
            _scalar(self.decay_constant, "decay_constant", positive=True),
        )

    @property
    def _block(self) -> tuple[tuple[float, ...], tuple[float, ...], float, float]:
        """Masses, widths, angle in radians and f, as the kernels take them."""
        return (
            (self.pion_mass, self.eta_mass, self.eta_prime_mass),
            (self.pion_width, self.eta_width, self.eta_prime_width),
            float(np.deg2rad(self.mixing_angle_deg)),
            self.decay_constant,
        )

    def source_couplings(self, couplings: AxialCurrent) -> np.ndarray:
        """Return the mixing strengths ``c_P = f Tr(G_A lambda_P)`` in MeV.

        Parameters
        ----------
        couplings : AxialCurrent
            Dimensionless light-quark source; every flavor direction is
            supported.

        Returns
        -------
        couplings : ndarray, shape (3,)
            ``c_P`` for (pi0, eta, eta'). They are the residues of the
            longitudinal source at each pole, up to the decay amplitude.
        """
        flavors = _flavor_couplings(couplings)
        return np.array(_axial.pole_source_couplings(self._block, flavors))

    def source_self_energy(
        self,
        q: ArrayLike,
        couplings: AxialCurrent,
        *,
        invisible: InvisibleCut | None = None,
    ) -> complex | np.ndarray:
        """Return the meson-pole longitudinal self-energy ``s c^T D^-1 c`` in MeV².

        Parameters
        ----------
        q : array-like
            Nonnegative source invariant mass in MeV.
        couplings : AxialCurrent
            Dimensionless light-quark source.
        invisible : InvisibleCut, optional
            Dark-matter cut to insert once in the meson block.

        Returns
        -------
        self_energy : complex or ndarray
            Complex, with the shape of ``q``. In the chiral limit it equals
            the contact ``2 f**2 Tr(G_A**2)`` at every ``q`` (the axial Ward
            identity); divided by ``M**2`` it is the size of the second
            mixing insertion.
        """
        flavors = _flavor_couplings(couplings)
        energies = _energies(q)
        cut = None if invisible is None else invisible._kernel
        value = _axial.pole_source_self_energy(
            energies.reshape(-1), self._block, flavors, cut
        )
        return _scalar_or_array(
            _finite(value.reshape(energies.shape), "The source self-energy")
        )

    def mixing_expansion(
        self, q: ArrayLike, couplings: AxialCurrent, *, mediator_mass: float
    ) -> np.ndarray:
        """Return the first-order mixing parameters ``epsilon_P``, dimensionless.

        Parameters
        ----------
        q : array-like
            Nonnegative source invariant mass in MeV.
        couplings : AxialCurrent
            Dimensionless light-quark source.
        mediator_mass : float
            Positive Proca mediator mass ``M`` in MeV.

        Returns
        -------
        epsilon : ndarray, shape (3, ...)
            ``|c_P**2 s/(M**2 (s - m_P**2 + i m_P Gamma_P))|`` for (pi0, eta,
            eta') with the visible widths, then the shape of ``q``. A point
            whose largest entry exceeds :data:`MIXING_GUARD` is outside the
            first-order treatment. An exact pole of zero width gives inf.

        Notes
        -----
        This is the note's per-pole measure (``sec:ax-match``). The exact
        size of the second insertion, ``|source_self_energy(q)|/M**2``, is at
        most the sum of the three entries, so up to three times the largest.
        """
        flavors = _flavor_couplings(couplings)
        energies = _energies(q)
        mass = _scalar(mediator_mass, "mediator_mass", positive=True)
        value = _axial.pole_mixing_expansion(
            energies.reshape(-1), self._block, flavors, mass
        )
        return value.reshape((3,) + energies.shape)

    def invisible_widths(
        self, couplings: AxialCurrent, invisible: InvisibleCut
    ) -> np.ndarray:
        """Return ``Gamma(P -> chi chibar) = m_P beta_chi y_P**2/(8 pi)`` in MeV.

        Parameters
        ----------
        couplings : AxialCurrent
            Dimensionless light-quark source.
        invisible : InvisibleCut
            Dark-matter mass and couplings defining ``y_P = kappa c_P``.

        Returns
        -------
        widths : ndarray, shape (3,)
            Widths for (pi0, eta, eta'), zero where ``2 m_chi >= m_P``. The
            same cut, inserted in the meson block, adds exactly these widths
            at each pole.
        """
        if not isinstance(invisible, InvisibleCut):
            raise TypeError("invisible must be an InvisibleCut.")
        flavors = _flavor_couplings(couplings)
        return np.array(
            _axial.pole_invisible_widths(self._block, flavors, invisible._kernel)
        )


@dataclass(frozen=True, kw_only=True)
class AxialFormFactorGammaGamma:
    r"""Source-longitudinal gamma gamma response from meson poles and triangles.

    Keyword-only ``poles`` is the meson block, and ``q_max`` (MeV) may lower
    but not exceed the 1600 MeV support ceiling. The light-quark anomaly
    enters only through the pi0, eta and eta' poles (the Bardeen-form WZW
    has no direct source-photon-photon vertex); leptons and charm enter
    through their perturbative triangles. The transverse density is
    identically zero (Landau-Yang).
    """

    poles: AxialMesonPoles = field(default_factory=AxialMesonPoles)
    q_max: float = _GAMMA_GAMMA_MAX_MEV
    threshold: ClassVar[float] = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.poles, AxialMesonPoles):
            raise TypeError("poles must be an AxialMesonPoles.")
        object.__setattr__(self, "q_max", _scalar(self.q_max, "q_max", positive=True))
        if self.q_max > _GAMMA_GAMMA_MAX_MEV:
            raise ValueError(
                "q_max cannot exceed the declared support ceiling of 1600 MeV."
            )

    @property
    def coverage(self) -> SourceCoverage:
        """Declared LO support, not an accuracy claim."""
        return SourceCoverage(
            channel="gamma gamma",
            q_min_mev=0.0,
            q_max_mev=self.q_max,
            input_status="lo",
            provenance="U(3) O(p²) poles with single-angle eta-eta' mixing and Bardeen WZW decays, plus perturbative lepton and charm triangles; isospin limit; restricted support, not a precision bound",
        )

    def amplitude(
        self,
        q: ArrayLike,
        couplings: AxialCurrent,
        triangles: TriangleCouplings,
        *,
        invisible: InvisibleCut | None = None,
    ) -> complex | np.ndarray:
        r"""Return the dimensionless longitudinal amplitude ``A(s)``.

        Parameters
        ----------
        q : array-like
            Nonnegative invariant mass in MeV, at most ``q_max``.
        couplings : AxialCurrent
            Dimensionless light-quark source.
        triangles : TriangleCouplings
            Dimensionless lepton and charm axial couplings.
        invisible : InvisibleCut, optional
            Dark-matter cut to insert once in the meson block. Without it the
            amplitude does not depend on the mediator; with it, it depends on
            the mediator mass through the cut's ``kappa``.

        Returns
        -------
        amplitude : complex or ndarray
            ``A = (1/2) s a^T D^-1 b + sum_f N_c Q_f**2 g_A^f (1 - tau_f f(tau_f))``
            with ``a_P = Tr(G_A lambda_P)`` and ``b_P = N_c Tr(lambda_P Q**2)``,
            normalized so that ``k.H = -i (2 alpha/pi) A eps(k1, e1, k2, e2)``.
            Without triangles it tends to ``N_c sum_q g_A^q Q_q**2`` (ABJ) for
            ``s >> m_P**2``.
        """
        flavors = _flavor_couplings(couplings)
        if not isinstance(triangles, TriangleCouplings):
            raise TypeError("triangles must be a TriangleCouplings.")
        energies = _energies(q, self.q_max)
        cut = None if invisible is None else invisible._kernel
        value = _axial.gamma_gamma_amplitude(
            energies.reshape(-1), self.poles._block, flavors, triangles._kernel, cut
        )
        return _scalar_or_array(
            _finite(value.reshape(energies.shape), "The gamma gamma amplitude")
        )

    def spectral_functions(
        self,
        q: ArrayLike,
        couplings: AxialCurrent,
        triangles: TriangleCouplings,
        *,
        invisible: InvisibleCut | None = None,
    ) -> AxialSpectralResult:
        """Return ``A_T = 0`` and ``A_L = s alpha**2 |A|**2/(8 pi**3)`` in MeV².

        Parameters match :meth:`amplitude`. ``A_L`` integrates
        ``|k.H|**2/s`` over two-body massless phase space, ``1/(8 pi)``, with
        the identical-photon factor 1/2; ``alpha`` is
        :data:`hazma.parameters.alpha_em`.

        The density does not check the first-order mixing guard, which needs
        the mediator mass: a rate built on it must reject points where
        :meth:`AxialMesonPoles.mixing_expansion` exceeds :data:`MIXING_GUARD`.
        """
        flavors = _flavor_couplings(couplings)
        if not isinstance(triangles, TriangleCouplings):
            raise TypeError("triangles must be a TriangleCouplings.")
        energies = _energies(q, self.q_max)
        cut = None if invisible is None else invisible._kernel
        density = _axial.gamma_gamma_density(
            energies.reshape(-1),
            self.poles._block,
            flavors,
            triangles._kernel,
            parameters.alpha_em,
            cut,
        )
        density = _finite(density.reshape(energies.shape), "The gamma gamma density")
        return AxialSpectralResult(
            q_mev=energies,
            transverse=np.zeros_like(density),
            source_longitudinal=density,
            threshold_mev=self.threshold,
            coverage=self.coverage,
            units="MeV^2",
        )
