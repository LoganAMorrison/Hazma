r"""Exact tree-level squared matrix elements for validating `dnde_photon_fsr`.

This module is the *theoretical corpus* backing
``test/spectra/test_dnde_photon_fsr.py``: real matrix elements, computed
from the Feynman rules of the hazma mediator models, for the radiative
annihilations

* ``x xbar -> V* -> l+ l- (gamma)``  — Dirac dark matter, vector mediator
  with vector couplings ``gvxx`` (DM) and ``gvll`` (leptons);
* ``x xbar -> S* -> l+ l- (gamma)``  — Dirac dark matter, scalar mediator
  with Yukawa couplings ``gsxx`` and ``gsll``;
* ``x xbar -> V* -> pi+ pi- (gamma)`` — vector mediator coupled to the
  charged-pion current (point-like scalar QED; a hadronic form factor is
  a constant at fixed s and cancels in the FSR ratio).

Rather than transcribing hand-simplified trace formulas (easy to typo,
hard to review), the fermionic squared matrix elements are evaluated
*numerically* from explicit Dirac matrices: the amplitude of each
diagram is assembled from its propagators and vertices, spin sums become
matrix traces, and the photon polarization sum uses
:math:`\sum_\lambda \epsilon_\alpha \epsilon^*_\beta \to -g_{\alpha\beta}`
(valid because the summed amplitude obeys the Ward identity — asserted
numerically in the test suite, together with conservation of the
mediator current and soft-photon factorization).

Conventions
-----------
* Momenta arrays follow the ``dnde_photon_fsr`` contract: shape
  ``(4, n_fsp[, batch])``, rows ``(E, px, py, pz)`` in MeV, metric
  ``(+,-,-,-)``. Column order is ``(l-, l+, gamma)`` /
  ``(pi+, pi-, gamma)`` with the photon last; the non-radiative
  functions take the same layout without the photon column.
* All returned values are spin-summed over final states, averaged over
  the initial DM spins, and averaged over the beam direction. For the
  s-channel processes here the beam average is exact: the DM tensor of
  the vector current, averaged over beam directions at fixed total
  momentum :math:`P`, is

  .. math::

      \langle L^{\mu\nu} \rangle = g_{V\chi}^2\,\frac{s + 2 m_\chi^2}{3}
      \left( \frac{P^\mu P^\nu}{s} - g^{\mu\nu} \right),

  (derived by averaging :math:`p_{1,2} = P/2 \pm q` over the directions
  of the relative momentum :math:`q`), and the scalar-current DM factor
  :math:`g_{S\chi}^2 (s/2 - 2 m_\chi^2)` carries no direction at all.
* Couplings and propagators are kept explicitly — including the
  :math:`1/[(s - m_{V,S}^2)^2 + (m\,\Gamma)^2]` of the off-shell
  mediator — so the corpus doubles as a check that common factors
  cancel in the ``dnde_photon_fsr`` ratio. The electric charge is
  ``hazma.parameters.qe`` (:math:`e^2 = 4\pi\alpha`).
"""

import numpy as np

from hazma.parameters import qe
from hazma.utils import ComplexArray, RealArray, RealOrRealArray, ldot, lnorm_sqr

# ===================================================================
# ---- Numerical Dirac algebra --------------------------------------
# ===================================================================

_METRIC = np.array([1.0, -1.0, -1.0, -1.0])

_SIGMA = [
    np.array([[0, 1], [1, 0]], dtype=np.complex128),
    np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    np.array([[1, 0], [0, -1]], dtype=np.complex128),
]

_ZERO2 = np.zeros((2, 2), dtype=np.complex128)
_ID2 = np.eye(2, dtype=np.complex128)

# Dirac representation: g^0 = diag(1,1,-1,-1), g^i off-diagonal Pauli.
_GAMMA = np.stack(
    [np.block([[_ID2, _ZERO2], [_ZERO2, -_ID2]])]
    + [np.block([[_ZERO2, s], [-s, _ZERO2]]) for s in _SIGMA]
)
_G0 = _GAMMA[0]
_ID4 = np.eye(4, dtype=np.complex128)


def _ensure_batched(momenta: RealArray) -> tuple[RealArray, bool]:
    """Return (momenta with a batch axis, had_no_batch_axis)."""
    momenta = np.asarray(momenta, dtype=np.float64)
    if momenta.ndim == 2:  # noqa: PLR2004 — (4, n) without a batch axis
        return momenta[:, :, None], True
    return momenta, False


def _slash(p: RealArray) -> ComplexArray:
    r"""p-slash = p_mu gamma^mu for p of shape (4, batch) -> (4, 4, batch)."""
    return np.einsum("m,mij,m...->ij...", _METRIC, _GAMMA, p, optimize=True)


def _fermionic_emission_parts(
    p: RealArray, pbar: RealArray, k: RealArray, mass: float
) -> tuple[ComplexArray, ComplexArray]:
    r"""Propagator-dressed pieces of photon emission off a fermion pair.

    For ``current -> f(p) fbar(pbar) gamma(k)`` the two diagrams carry

    .. math::

        S_1 = \frac{\slashed{p} + \slashed{k} + m}{2 p \cdot k},
        \qquad
        S_2 = \frac{-\slashed{p}' - \slashed{k} + m}{2 p' \cdot k},

    such that the amplitude is
    ``ubar(p) [g^a S_1 V + V S_2 g^a] v(pbar)`` for a current vertex
    matrix ``V`` (photon index ``a``).
    """
    s1 = (_slash(p + k) + mass * _ID4[:, :, None]) / (2.0 * ldot(p, k))
    s2 = (-_slash(pbar + k) + mass * _ID4[:, :, None]) / (2.0 * ldot(pbar, k))
    return s1, s2


# ===================================================================
# ---- Beam-averaged dark-matter factors ----------------------------
# ===================================================================


def _dm_vector_contract(
    tensor: RealArray, total_p: RealArray, mx: float
) -> RealOrRealArray:
    r"""Contract a current tensor with the beam-averaged DM vector tensor.

    The DM tensor, couplings excluded, is
    <L>_{mu nu} / g^2 = (s + 2 mx^2)/3 * (P_mu P_nu / s - g_{mu nu}).
    """
    s = lnorm_sqr(total_p)
    p_lower = _METRIC[:, None] * total_p
    php = np.einsum("m...,mn...,n...->...", p_lower, tensor, p_lower, optimize=True)
    gt = np.einsum("m,mm...->...", _METRIC, tensor, optimize=True)
    return (s + 2.0 * mx**2) / 3.0 * (php / s - gt)


def _propagator_den(s: RealOrRealArray, mmed: float, width: float) -> RealOrRealArray:
    return (s - mmed**2) ** 2 + (mmed * width) ** 2


# ===================================================================
# ---- x xbar -> V* -> l+ l- (gamma) --------------------------------
# ===================================================================


def msqrd_xx_to_v_to_ff(  # noqa: PLR0913 — the physics parameters of the process
    momenta: RealArray,
    *,
    mf: float,
    mx: float,
    mv: float,
    gvxx: float,
    gvff: float,
    widthv: float = 0.0,
) -> RealOrRealArray:
    r"""|M|^2 for x xbar -> V* -> f fbar, spin- and beam-averaged.

    Momenta columns: (f, fbar). Equals
    ``gvxx^2 gvff^2 / D_V * (s + 2 mx^2)/3 * 4 s (1 + 2 mf^2/s)``;
    evaluated here with the numerical trace machinery so radiative and
    non-radiative corpus entries share one implementation.
    """
    momenta, single = _ensure_batched(momenta)
    p, pbar = momenta[:, 0], momenta[:, 1]
    total_p = p + pbar
    s = lnorm_sqr(total_p)

    m1 = _slash(p) + mf * _ID4[:, :, None]
    m2 = _slash(pbar) - mf * _ID4[:, :, None]

    # H0^{mu nu} = Tr[(pslash + m) g^mu (pbarslash - m) g^nu]
    tensor = np.einsum(
        "ij...,mjk...,kl...,nli...->mn...",
        m1,
        _GAMMA[:, :, :, None],
        m2,
        _GAMMA[:, :, :, None],
        optimize=True,
    ).real

    result = (
        gvxx**2
        * gvff**2
        / _propagator_den(s, mv, widthv)
        * _dm_vector_contract(tensor, total_p, mx)
    )
    return result[0] if single else result


def msqrd_xx_to_v_to_ffg(  # noqa: PLR0913 — the physics parameters of the process
    momenta: RealArray,
    *,
    mf: float,
    mx: float,
    mv: float,
    gvxx: float,
    gvff: float,
    widthv: float = 0.0,
) -> RealOrRealArray:
    r"""|M|^2 for x xbar -> V* -> f fbar gamma, spin- and beam-averaged.

    Momenta columns: (f, fbar, gamma). Photon emission off both fermion
    legs; polarization sum via -g_{alpha beta}.
    """
    momenta, single = _ensure_batched(momenta)
    p, pbar, k = momenta[:, 0], momenta[:, 1], momenta[:, 2]
    total_p = p + pbar + k
    s = lnorm_sqr(total_p)

    s1, s2 = _fermionic_emission_parts(p, pbar, k, mf)
    m1 = _slash(p) + mf * _ID4[:, :, None]
    m2 = _slash(pbar) - mf * _ID4[:, :, None]

    gam = _GAMMA[:, :, :, None]
    # A[a, mu] = g^a S1 g^mu + g^mu S2 g^a  (photon index a, V index mu)
    amp = np.einsum(
        "aij...,jk...,mkl...->amil...", gam, s1, gam, optimize=True
    ) + np.einsum("mij...,jk...,akl...->amil...", gam, s2, gam, optimize=True)
    amp_bar = np.einsum("ik,amlk...,lj->amij...", _G0, np.conj(amp), _G0, optimize=True)

    # H^{mu nu} = -g_ab Tr[(pslash+m) A^{a mu} (pbarslash-m) Abar^{b nu}]
    tensor = -np.einsum(
        "a,ij...,amjk...,kl...,anli...->mn...",
        _METRIC,
        m1,
        amp,
        m2,
        amp_bar,
        optimize=True,
    ).real

    result = (
        qe**2
        * gvxx**2
        * gvff**2
        / _propagator_den(s, mv, widthv)
        * _dm_vector_contract(tensor, total_p, mx)
    )
    return result[0] if single else result


def photon_ward_violation_v(momenta: RealArray, *, mf: float) -> RealOrRealArray:
    r"""Relative photon Ward-identity violation of the vector amplitude.

    The polarization sum replaced by k_alpha k_beta must annihilate the
    current tensor.
    """
    momenta, _ = _ensure_batched(momenta)
    p, pbar, k = momenta[:, 0], momenta[:, 1], momenta[:, 2]

    s1, s2 = _fermionic_emission_parts(p, pbar, k, mf)
    m1 = _slash(p) + mf * _ID4[:, :, None]
    m2 = _slash(pbar) - mf * _ID4[:, :, None]

    gam = _GAMMA[:, :, :, None]
    kslash = _slash(k)
    # k_alpha A^{alpha mu} = kslash S1 g^mu + g^mu S2 kslash
    amp_k = np.einsum(
        "ij...,jk...,mkl...->mil...", kslash, s1, gam, optimize=True
    ) + np.einsum("mij...,jk...,kl...->mil...", gam, s2, kslash, optimize=True)
    amp_k_bar = np.einsum(
        "ik,mlk...,lj->mij...", _G0, np.conj(amp_k), _G0, optimize=True
    )

    violation = np.einsum(
        "m,ij...,mjk...,kl...,mli...->...",
        _METRIC,
        m1,
        amp_k,
        m2,
        amp_k_bar,
        optimize=True,
    ).real

    # Scale: the same contraction with the physical -g_ab polarization sum.
    amp = np.einsum(
        "aij...,jk...,mkl...->amil...", gam, s1, gam, optimize=True
    ) + np.einsum("mij...,jk...,akl...->amil...", gam, s2, gam, optimize=True)
    amp_bar = np.einsum("ik,amlk...,lj->amij...", _G0, np.conj(amp), _G0, optimize=True)
    scale = -np.einsum(
        "a,m,ij...,amjk...,kl...,amli...->...",
        _METRIC,
        _METRIC,
        m1,
        amp,
        m2,
        amp_bar,
        optimize=True,
    ).real

    return np.abs(violation) / np.abs(scale)


# ===================================================================
# ---- x xbar -> S* -> l+ l- (gamma) --------------------------------
# ===================================================================


def _dm_scalar_factor(s: RealOrRealArray, mx: float) -> RealOrRealArray:
    r"""Spin-averaged DM factor of the scalar current, couplings excluded.

    (1/4) Tr[(p2slash - mx)(p1slash + mx)] = s/2 - 2 mx^2.
    """
    return 0.5 * s - 2.0 * mx**2


def msqrd_xx_to_s_to_ff(  # noqa: PLR0913 — the physics parameters of the process
    momenta: RealArray,
    *,
    mf: float,
    mx: float,
    ms: float,
    gsxx: float,
    gsff: float,
    widths: float = 0.0,
) -> RealOrRealArray:
    r"""|M|^2 for x xbar -> S* -> f fbar, spin-averaged.

    Momenta columns: (f, fbar). Fermionic factor
    Tr[(pslash+m)(pbarslash-m)] = 4 (p.pbar - m^2) = 2 s (1 - 4 mf^2/s).
    """
    momenta, single = _ensure_batched(momenta)
    p, pbar = momenta[:, 0], momenta[:, 1]
    s = lnorm_sqr(p + pbar)

    fermionic = 4.0 * (ldot(p, pbar) - mf**2)
    result = (
        gsxx**2
        * gsff**2
        / _propagator_den(s, ms, widths)
        * _dm_scalar_factor(s, mx)
        * fermionic
    )
    return result[0] if single else result


def msqrd_xx_to_s_to_ffg(  # noqa: PLR0913 — the physics parameters of the process
    momenta: RealArray,
    *,
    mf: float,
    mx: float,
    ms: float,
    gsxx: float,
    gsff: float,
    widths: float = 0.0,
) -> RealOrRealArray:
    r"""|M|^2 for x xbar -> S* -> f fbar gamma, spin-averaged.

    Momenta columns: (f, fbar, gamma). The scalar vertex matrix is the
    identity; photon emission off both fermion legs.
    """
    momenta, single = _ensure_batched(momenta)
    p, pbar, k = momenta[:, 0], momenta[:, 1], momenta[:, 2]
    s = lnorm_sqr(p + pbar + k)

    s1, s2 = _fermionic_emission_parts(p, pbar, k, mf)
    m1 = _slash(p) + mf * _ID4[:, :, None]
    m2 = _slash(pbar) - mf * _ID4[:, :, None]

    gam = _GAMMA[:, :, :, None]
    # A^a = g^a S1 + S2 g^a
    amp = np.einsum("aij...,jk...->aik...", gam, s1, optimize=True) + np.einsum(
        "ij...,ajk...->aik...", s2, gam, optimize=True
    )
    amp_bar = np.einsum("ik,alk...,lj->aij...", _G0, np.conj(amp), _G0, optimize=True)

    fermionic = -np.einsum(
        "a,ij...,ajk...,kl...,ali...->...", _METRIC, m1, amp, m2, amp_bar, optimize=True
    ).real

    result = (
        qe**2
        * gsxx**2
        * gsff**2
        / _propagator_den(s, ms, widths)
        * _dm_scalar_factor(s, mx)
        * fermionic
    )
    return result[0] if single else result


# ===================================================================
# ---- x xbar -> V* -> pi+ pi- (gamma), scalar QED ------------------
# ===================================================================


def msqrd_xx_to_v_to_pipi(  # noqa: PLR0913 — the physics parameters of the process
    momenta: RealArray,
    *,
    mx: float,
    mv: float,
    gvxx: float,
    gvpipi: float,
    widthv: float = 0.0,
) -> RealOrRealArray:
    r"""|M|^2 for x xbar -> V* -> pi+ pi-, spin- and beam-averaged.

    Momenta columns: (pi+, pi-). The pion current is
    ``M0^mu = (p+ - p-)^mu``.
    """
    momenta, single = _ensure_batched(momenta)
    pp, pm = momenta[:, 0], momenta[:, 1]
    total_p = pp + pm
    s = lnorm_sqr(total_p)

    current = pp - pm
    tensor = np.einsum("m...,n...->mn...", current, current, optimize=True)

    result = (
        gvxx**2
        * gvpipi**2
        / _propagator_den(s, mv, widthv)
        * _dm_vector_contract(tensor, total_p, mx)
    )
    return result[0] if single else result


def _pion_rad_amplitude(pp: RealArray, pm: RealArray, k: RealArray) -> RealArray:
    r"""Scalar-QED radiative amplitude tensor M^{mu alpha}.

    V index mu, photon index alpha, couplings excluded, for
    V* -> pi+(pp) pi-(pm) gamma(k):

    .. math::

        M^{\mu\alpha}
        = \frac{(2 p_+ + k)^\alpha (p_+ + k - p_-)^\mu}{2 p_+ \cdot k}
        - \frac{(2 p_- + k)^\alpha (p_+ - p_- - k)^\mu}{2 p_- \cdot k}
        - 2 g^{\mu\alpha}.

    The seagull coefficient is fixed by the Ward identities
    ``k_alpha M^{mu alpha} = 0`` and ``P_mu M^{mu alpha} = 0`` (both
    hold exactly; asserted in the test suite).
    """
    batch_shape = pp.shape[1:]
    metric_upper = np.broadcast_to(
        np.diag(_METRIC)[(...,) + (None,) * len(batch_shape)],
        (4, 4) + batch_shape,
    )
    return (
        np.einsum("m...,a...->ma...", pp + k - pm, (2.0 * pp + k) / (2.0 * ldot(pp, k)))
        - np.einsum(
            "m...,a...->ma...", pp - pm - k, (2.0 * pm + k) / (2.0 * ldot(pm, k))
        )
        - 2.0 * metric_upper
    )


def msqrd_xx_to_v_to_pipig(  # noqa: PLR0913 — the physics parameters of the process
    momenta: RealArray,
    *,
    mx: float,
    mv: float,
    gvxx: float,
    gvpipi: float,
    widthv: float = 0.0,
) -> RealOrRealArray:
    r"""|M|^2 for x xbar -> V* -> pi+ pi- gamma, spin- and beam-averaged.

    Momenta columns: (pi+, pi-, gamma). Point-pion scalar QED with the
    seagull required by gauge invariance.
    """
    momenta, single = _ensure_batched(momenta)
    pp, pm, k = momenta[:, 0], momenta[:, 1], momenta[:, 2]
    total_p = pp + pm + k
    s = lnorm_sqr(total_p)

    amp = _pion_rad_amplitude(pp, pm, k)
    # H^{mu nu} = -g_ab M^{mu a} M^{nu b}
    tensor = -np.einsum("a,ma...,na...->mn...", _METRIC, amp, amp, optimize=True)

    result = (
        qe**2
        * gvxx**2
        * gvpipi**2
        / _propagator_den(s, mv, widthv)
        * _dm_vector_contract(tensor, total_p, mx)
    )
    return result[0] if single else result


def pion_ward_violations(
    momenta: RealArray,
) -> tuple[RealOrRealArray, RealOrRealArray]:
    r"""Ward violations of the scalar-QED radiative amplitude.

    Returns the relative photon-current and mediator-current violations;
    both should vanish identically.
    """
    momenta, _ = _ensure_batched(momenta)
    pp, pm, k = momenta[:, 0], momenta[:, 1], momenta[:, 2]
    total_p = pp + pm + k

    amp = _pion_rad_amplitude(pp, pm, k)
    scale = np.sqrt(np.einsum("ma...,ma...->...", amp, amp, optimize=True))

    photon = np.einsum("a,a...,ma...->m...", _METRIC, k, amp, optimize=True)
    mediator = np.einsum("m,m...,ma...->a...", _METRIC, total_p, amp, optimize=True)

    photon_violation = (
        np.sqrt(np.einsum("m...,m...->...", photon, photon, optimize=True)) / scale
    )
    mediator_violation = (
        np.sqrt(np.einsum("a...,a...->...", mediator, mediator, optimize=True)) / scale
    )
    return photon_violation, mediator_violation


# ===================================================================
# ---- Soft-photon (eikonal) factor ---------------------------------
# ===================================================================


def eikonal_factor(p: RealArray, pbar: RealArray, k: RealArray) -> RealOrRealArray:
    r"""Leading soft-photon factor for emission off a charge +1/-1 pair.

    Defined as

    .. math::

        e^2 \left[ \frac{2 p \cdot \bar{p}}{(p \cdot k)(\bar{p} \cdot k)}
        - \frac{p^2}{(p \cdot k)^2}
        - \frac{\bar{p}^2}{(\bar{p} \cdot k)^2} \right],

    such that ``|M_rad|^2 -> eikonal_factor * |M_nonrad|^2`` as the
    photon momentum ``k -> 0`` (universal for fermions and scalars).
    """
    return qe**2 * (
        2.0 * ldot(p, pbar) / (ldot(p, k) * ldot(pbar, k))
        - lnorm_sqr(p) / ldot(p, k) ** 2
        - lnorm_sqr(pbar) / ldot(pbar, k) ** 2
    )
