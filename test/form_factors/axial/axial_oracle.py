"""NumPy reference implementation of the axial tensor, three-pion and pole numerics.

The tensor and three-pion functions are the NumPy functions
`hazma.form_factors.axial` evaluated before the numerics moved into the
`hazma._core.axial` kernels, kept unchanged as the oracle `test_oracle.py`
compares the Rust-backed public functions against. The tensor functions keep
their validation so a test can call them with the same arguments as the
public ones; the three-pion functions take arguments the public wrappers have
already validated.

The pole functions never had a NumPy predecessor. They are an independent
formulation for `test_poles.py`: the full quadratic kernel of the
longitudinal source and the octet-singlet mesons, inverted whole, where the
kernel resums only the meson block in the physical basis.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np

ArrayLike = float | complex | Sequence[object] | np.ndarray
_LORENTZ_DIMENSION = 4
_TENSOR_RANK = 2


def _four_vector(value: ArrayLike, name: str, *, real: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if real and np.iscomplexobj(array):
        raise ValueError(f"{name} must be real.")
    array = np.asarray(array, dtype=np.float64 if real else np.complex128)
    if array.ndim < 1 or array.shape[0] != _LORENTZ_DIMENSION:
        raise ValueError(f"{name} must have shape (4, ...).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array


def _broadcast_lorentz(
    array: np.ndarray, shape: tuple[int, ...], rank: int
) -> np.ndarray:
    padding = (1,) * (len(shape) - (array.ndim - rank))
    return np.broadcast_to(
        array.reshape(array.shape[:rank] + padding + array.shape[rank:]),
        array.shape[:rank] + shape,
    )


def _momentum(q: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    momentum = _four_vector(q, "q", real=True)
    with np.errstate(over="ignore", invalid="ignore"):
        invariant = momentum[0] ** 2 - np.sum(momentum[1:] ** 2, axis=0)
    if (
        not np.all(np.isfinite(invariant))
        or np.any(invariant <= 0)
        or np.any(momentum[0] <= 0)
    ):
        raise ValueError(
            "q must be finite, future-directed and timelike (q squared > 0)."
        )
    return momentum, invariant


def _finite_result(value: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(value)):
        raise ValueError("The tensor calculation overflowed; rescale the inputs.")
    return value


def hadronic_tensor(current: ArrayLike) -> np.ndarray:
    r"""Return the Hermitian contravariant tensor H^mu H^{nu*}.

    Parameters
    ----------
    current : array-like, shape (4, ...)
        Contravariant current; complex phases and interference are retained.
        Its units are those of the supplied channel current.

    Returns
    -------
    tensor : complex ndarray, shape (4, 4, ...)
        Tensor in the same batch shape, with units ``abs(current units)**2``.
    """
    amplitude = _four_vector(current, "current")
    with np.errstate(over="ignore", invalid="ignore"):
        return _finite_result(amplitude[:, None] * amplitude.conj()[None, :])


def current_projections(
    current: ArrayLike, q: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return transverse and source-longitudinal current densities.

    Parameters
    ----------
    current : array-like, shape (4, ...)
        Complex contravariant current in its channel's declared units.
    q : array-like, shape (4, ...)
        Real future-directed timelike four-momentum in MeV. Batch dimensions
        broadcast with the current's batch dimensions.

    Returns
    -------
    transverse, source_longitudinal : ndarray
        Batch densities in squared current units: ``|H_CM spatial|**2 / 3``
        and ``|H_CM time|**2``. No phase-space measure is included.

    Notes
    -----
    With s = q squared and metric +---, the invariant definitions are
    ``T = (|q.dot(H)|**2 / s - H.dot(H.conj())) / 3`` and
    ``L = |q.dot(H)|**2 / s``. A boost to the q rest frame evaluates the same
    projectors as positive norms, avoiding cancellation of large norms.
    """
    amplitude = _four_vector(current, "current")
    momentum, invariant = _momentum(q)
    shape = np.broadcast_shapes(amplitude.shape[1:], momentum.shape[1:])
    amplitude = _broadcast_lorentz(amplitude, shape, 1)
    momentum = _broadcast_lorentz(momentum, shape, 1)
    mass = np.sqrt(np.broadcast_to(invariant, shape))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        spatial_dot = np.sum(momentum[1:] * amplitude[1:], axis=0)
        time = (momentum[0] * amplitude[0] - spatial_dot) / mass
        spatial = amplitude[1:] + momentum[1:] * (
            spatial_dot / (mass * (momentum[0] + mass)) - amplitude[0] / mass
        )
        transverse = np.sum(np.abs(spatial) ** 2, axis=0) / 3.0
        longitudinal = np.abs(time) ** 2
    return _finite_result(transverse), _finite_result(longitudinal)


def tensor_projections(
    tensor: ArrayLike, q: ArrayLike, *, rtol: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    r"""Project a Hermitian hadronic tensor using invariant projectors.

    Parameters
    ----------
    tensor : array-like, shape (4, 4, ...)
        Contravariant Hermitian tensor, including a spin sum or phase-space
        integration if desired. Units are declared by its provider.
    q : array-like, shape (4, ...)
        Real future-directed timelike four-momentum in MeV.
    rtol : float, default 1e-6
        Largest accepted roundoff bound, relative to the rest-frame norm
        ``source_longitudinal + 3 * transverse`` of each batch element.
        Must lie in (0, 1).

    Returns
    -------
    transverse, source_longitudinal : ndarray
        ``(-g_mu_nu + q_mu q_nu / s) W^mu_nu / 3`` and
        ``q_mu q_nu W^mu_nu / s``, in the supplied tensor's units.

    Raises
    ------
    ValueError
        For invalid shapes, nonfinite inputs/results, non-Hermitian tensors,
        invalid momentum, materially negative projected densities, or a
        roundoff bound above ``rtol`` times the rest-frame norm.

    Notes
    -----
    The contraction cancels terms of order ``gamma**4`` times the result,
    where ``gamma = q[0] / sqrt(s)``. The rounding already present in the
    tensor entries then limits the projections to a relative accuracy of
    about ``gamma**4`` times machine epsilon, and no reordering of the sum
    recovers it. With the default ``rtol`` the projection is rejected from
    about ``gamma = 50``. Project the amplitude with
    :func:`current_projections`, whose error grows only like
    ``gamma**2``, or supply the tensor in the q rest frame.
    """
    if (
        isinstance(rtol, bool)
        or not isinstance(rtol, (int, float, np.floating, np.integer))
        or not np.isfinite(rtol)
        or not 0 < rtol < 1
    ):
        raise ValueError("rtol must be a finite number in (0, 1).")
    density = np.asarray(tensor, dtype=np.complex128)
    if density.ndim < _TENSOR_RANK or density.shape[:2] != (4, 4):
        raise ValueError("tensor must have shape (4, 4, ...).")
    if not np.all(np.isfinite(density)):
        raise ValueError("tensor must be finite.")
    # A norm-relative roundoff budget, without an arbitrary absolute floor.
    scale = np.max(np.abs(density), axis=(0, 1))
    tolerance = 64 * np.finfo(float).eps * scale
    if np.any(np.abs(density - density.swapaxes(0, 1).conj()) > tolerance):
        raise ValueError("tensor must be Hermitian in its Lorentz indices.")
    momentum, invariant = _momentum(q)
    shape = np.broadcast_shapes(density.shape[2:], momentum.shape[1:])
    density = _broadcast_lorentz(density, shape, 2)
    momentum = _broadcast_lorentz(momentum, shape, 1)
    invariant = np.broadcast_to(invariant, shape)
    covariant = momentum * np.array([1.0, -1.0, -1.0, -1.0]).reshape(
        (4,) + (1,) * len(shape)
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        terms = covariant[:, None] * covariant[None, :] * density / invariant
        longitudinal = np.sum(terms, axis=(0, 1)).real
        trace = density[0, 0].real - sum(density[i, i].real for i in range(1, 4))
        transverse = (longitudinal - trace) / 3.0
        budget = (
            128
            * np.finfo(float).eps
            * (
                np.sum(np.abs(terms), axis=(0, 1))
                + np.sum(np.abs(np.diagonal(density, axis1=0, axis2=1)), axis=-1)
            )
        )
    _finite_result(budget)
    for value in (transverse, longitudinal):
        _finite_result(value)
        if np.any(value < -budget):
            raise ValueError("tensor has a negative projected density.")
    transverse = np.maximum(transverse, 0.0)
    longitudinal = np.maximum(longitudinal, 0.0)
    # The budget also bounds positive errors; compare it with the rest-frame
    # norm, which dominates both projections.
    if np.any(budget > rtol * (longitudinal + 3.0 * transverse)):
        raise ValueError(
            "tensor projection is ill-conditioned for this q: its roundoff "
            "bound exceeds rtol times the rest-frame norm. Project the "
            "current with current_projections or supply the tensor in the "
            "q rest frame."
        )
    return transverse, longitudinal


def integrate_spectral_functions(
    current: ArrayLike, q: ArrayLike, weights: ArrayLike, *, sample_axis: int = -1
) -> tuple[np.ndarray, np.ndarray]:
    r"""Integrate current projections with a supplied phase-space measure.

    Parameters
    ----------
    current : array-like, shape (4, ...)
        Current samples in their channel's declared units.
    q : array-like, shape (4, ...)
        Timelike momentum in MeV, broadcast over current samples.
    weights : array-like
        Nonnegative finite integration weights, broadcast over batch/sample
        dimensions. Include the quadrature/sample normalization and any final
        state symmetry factor exactly once. These weights represent dPhi,
        without a factor 1/2 from an optical-theorem Gram cut.
    sample_axis : int, default -1
        Axis of the broadcast densities to sum, excluding the Lorentz index.

    Returns
    -------
    transverse, source_longitudinal : ndarray
        ``A_T = integral dPhi |H_CM spatial|**2 / 3`` and
        ``A_L = integral dPhi |H_CM time|**2``. Units are squared current
        units times measure units. For a dimensionless three-pion current
        and dPhi_3 in MeV squared, these are MeV squared.
    """
    measure = np.asarray(weights)
    if np.iscomplexobj(measure):
        raise ValueError("weights must be real.")
    measure = np.asarray(measure, dtype=np.float64)
    if not np.all(np.isfinite(measure)) or np.any(measure < 0):
        raise ValueError("weights must be finite and nonnegative.")
    transverse, longitudinal = current_projections(current, q)
    transverse, longitudinal, measure = np.broadcast_arrays(
        transverse, longitudinal, measure
    )
    if isinstance(sample_axis, bool) or not isinstance(sample_axis, (int, np.integer)):
        raise ValueError("sample_axis must be an integer density axis.")
    if transverse.ndim == 0 or not -transverse.ndim <= sample_axis < transverse.ndim:
        raise ValueError("sample_axis is outside the density batch dimensions.")
    with np.errstate(over="ignore", invalid="ignore"):
        result_t = np.sum(transverse * measure, axis=sample_axis)
        result_l = np.sum(longitudinal * measure, axis=sample_axis)
    return _finite_result(result_t), _finite_result(result_l)


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[0] * b[0] - np.sum(a[1:] * b[1:], axis=0)


@lru_cache(maxsize=8)
def _quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    theta = (nodes + 1) * np.pi / 2
    return theta, weights * np.pi / 2


@np.errstate(over="ignore", invalid="ignore", divide="ignore")
def three_pi_current(
    ps: np.ndarray, g3: float, pion_mass: float, decay_constant: float, neutral: bool
) -> np.ndarray:
    """Return the LO current H^mu of momenta ``ps`` of shape (4, 3, ...)."""
    q = np.sum(ps, axis=1)
    q2 = _dot(q, q)
    m2 = np.square(pion_mass)
    if g3 == 0 or (neutral and pion_mass == 0):
        return np.zeros_like(q)
    if neutral:
        return -2 * g3 / decay_constant * q * (m2 / (q2 - m2))
    pair = ps[:, 0] + ps[:, 1]
    s = _dot(pair, pair)
    return 2 * g3 / decay_constant * (-2 * ps[:, 2] + q * ((q2 - s) / (q2 - m2)))


@np.errstate(over="ignore", invalid="ignore", divide="ignore")
def three_pi_differential_weight(
    ps: np.ndarray, g3: float, pion_mass: float, decay_constant: float, neutral: bool
) -> np.ndarray:
    """Return the spin-averaged |epsilon.H|^2 of momenta ``ps``."""
    q = np.sum(ps, axis=1)
    q2 = _dot(q, q)
    if neutral:
        return np.zeros_like(q2)
    pcm2 = np.maximum(_dot(q, ps[:, 2]) ** 2 / q2 - np.square(pion_mass), 0.0)
    return 16 * g3**2 * pcm2 / (3 * np.square(decay_constant))


@np.errstate(over="ignore", invalid="ignore", divide="ignore")
def three_pi_spectral_functions(  # noqa: PLR0913 - the model parameters plus the rule
    energies: np.ndarray,
    g3: float,
    pion_mass: float,
    decay_constant: float,
    neutral: bool,
    *,
    quadrature_order: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (A_T, A_L) in MeV^2 at invariant energies in MeV."""
    energies = np.asarray(energies, dtype=float)
    transverse = np.zeros_like(energies)
    longitudinal = np.zeros_like(energies)
    mask = energies > 3 * pion_mass
    if pion_mass == 0:
        if not neutral:
            transverse = (
                (g3 * energies / decay_constant) ** 2 * energies**2 / (384 * np.pi**3)
            )
        mask = np.zeros_like(mask)
    if np.any(mask):
        energy = energies[mask, None]
        ratio = pion_mass / energy
        ratio2 = ratio**2
        theta, weights = _quadrature(int(quadrature_order))
        span = np.maximum((1 - 3 * ratio) * (1 + ratio), 0.0)
        s = 4 * ratio2 + span * (1 - np.cos(theta)) / 2
        jac = span * np.sin(theta) / 2
        lam = ((1 - ratio) ** 2 - s) * ((1 + ratio) ** 2 - s)
        measure = (
            energy**2
            * weights
            * jac
            * np.sqrt(np.maximum(lam, 0))
            * np.sqrt(np.maximum(1 - 4 * ratio2 / s, 0))
            / (128 * np.pi**3)
        )
        prefactor = 4 * (g3 * energy / decay_constant) ** 2
        if neutral:
            value_l = prefactor * ratio2**2 / (1 - ratio2) ** 2
            longitudinal[mask] = np.sum(measure * value_l, axis=-1) / 6
        else:
            value_t = prefactor * lam / 3
            value_l = prefactor * ratio2**2 * (s - ratio2) ** 2 / (1 - ratio2) ** 2
            transverse[mask] = np.sum(measure * value_t, axis=-1)
            longitudinal[mask] = np.sum(measure * value_l, axis=-1)
    return transverse, longitudinal


# Generators in flavor order (u, d, s), normalized so Tr(T_a T_b) = 2 delta_ab.
OCTET_SINGLET = np.array(
    [[1.0, -1.0, 0.0], np.array([1.0, 1.0, -2.0]) / np.sqrt(3.0), [np.sqrt(2 / 3)] * 3]
)
_QUARK_CHARGES_SQUARED = np.array([4.0, 1.0, 1.0]) / 9.0


def physical_rotation(mixing_angle: float) -> np.ndarray:
    """Rows (pi0, eta, eta') in terms of the generators (T3, T8, T0)."""
    cos, sin = np.cos(mixing_angle), np.sin(mixing_angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])


def longitudinal_kernel(  # noqa: PLR0913, PLR0917 - the meson block plus the source
    q: float,
    couplings: np.ndarray,
    masses: np.ndarray,
    widths: np.ndarray,
    mixing_angle: float,
    decay_constant: float,
    mediator_mass: float,
    meson_insertion: np.ndarray | None = None,
) -> np.ndarray:
    """Return the 4x4 kernel K over (a, pi3, eta8, eta0) in MeV², at s = q².

    ``a`` is the source-longitudinal Proca mode, ``Z'_mu = (k_mu/sqrt(s)) a``,
    whose propagator is ``i/M²``, so ``K_aa = M²``. The derivative mixing
    ``-c_i Z'.d phi_i`` gives ``K_ai = i sqrt(s) c_i`` and ``K_ia = -i sqrt(s)
    c_i``, Hermitian for real ``s``. The meson block is the physical
    ``diag(s - m² + i m Gamma)`` rotated to the generator basis, plus an
    optional complex matrix in that basis.
    """
    s = q * q
    rotation = physical_rotation(mixing_angle)
    meson = rotation.T @ np.diag(s - masses**2 + 1j * masses * widths) @ rotation
    if meson_insertion is not None:
        meson = meson + meson_insertion
    c = decay_constant * (OCTET_SINGLET @ couplings)
    kernel = np.zeros((4, 4), dtype=np.complex128)
    kernel[0, 0] = mediator_mass**2
    kernel[0, 1:] = 1j * q * c
    kernel[1:, 0] = -1j * q * c
    kernel[1:, 1:] = meson
    return kernel


def schur_source_amplitude(kernel: np.ndarray, h_mesons: np.ndarray) -> complex:
    """Return ``h_eff = -K_am D^-1 h_m`` from the full inverse of ``kernel``.

    Block inversion gives ``(K^-1 h)_a = h_eff/D_a`` and ``(K^-1)_aa = 1/D_a``
    with ``D_a = K_aa - K_am D^-1 K_ma``, so their ratio is the one-insertion
    source amplitude for any ``K_aa``.
    """
    h = np.concatenate([[0.0], h_mesons]).astype(np.complex128)
    column = np.linalg.solve(kernel, h)
    unit = np.linalg.solve(kernel, np.eye(4, dtype=np.complex128)[:, 0])
    return column[0] / unit[0]


def source_self_energy(kernel: np.ndarray) -> complex:
    """Return ``s c^T D^-1 c = K_am D^-1 K_ma``, in MeV², from the full inverse.

    It is minus the one-insertion amplitude of the source's own mixing
    column. Taking it as that ratio, rather than as ``K_aa - D_a``, avoids
    cancelling ``M²`` against itself when the mediator is heavy.
    """
    return -schur_source_amplitude(kernel, kernel[1:, 0])


def photon_traces() -> np.ndarray:
    """Return ``N_c Tr(T_i Q²)`` for (T3, T8, T0), dimensionless."""
    return 3.0 * OCTET_SINGLET @ _QUARK_CHARGES_SQUARED


def triangle_factor(q: float, mass: float) -> complex:
    """Return ``1 - tau f(tau)``, ``tau = 4 m²/q²``, from the textbook ``f``."""
    tau = 4 * mass * mass / (q * q)
    if tau >= 1:
        return complex(1 - tau * np.arcsin(1 / np.sqrt(tau)) ** 2)
    r = np.sqrt(1 - tau)
    return complex(1 + 0.25 * tau * (np.log((1 + r) / (1 - r)) - 1j * np.pi) ** 2)


_GAMMA0 = np.diag([1.0, 1.0, -1.0, -1.0]).astype(np.complex128)
_PAULI = [
    np.array([[0, 1], [1, 0]], dtype=np.complex128),
    np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    np.array([[1, 0], [0, -1]], dtype=np.complex128),
]
_GAMMA = [_GAMMA0] + [
    np.block([[np.zeros((2, 2)), sigma], [-sigma, np.zeros((2, 2))]])
    for sigma in _PAULI
]
_GAMMA5 = 1j * _GAMMA[0] @ _GAMMA[1] @ _GAMMA[2] @ _GAMMA[3]
_METRIC = np.array([1.0, -1.0, -1.0, -1.0])


def _slash(p: np.ndarray) -> np.ndarray:
    return sum(_METRIC[mu] * p[mu] * _GAMMA[mu] for mu in range(4))


def longitudinal_pair_cut(
    q: float, dm_mass: float, dm_vector: float, dm_axial: float
) -> float:
    """Return Im Pi_aa in MeV², the dark-matter pair cut of the Proca mode ``a``.

    ``(1/2) integral dPhi_2 sum |ubar eps_L.gamma (g_V + g_A gamma5) v|²`` with
    ``eps_L = k/sqrt(s)``, from explicit Dirac traces; the integrand is
    isotropic, so the phase space contributes its volume ``beta/(8 pi)``.
    """
    s = q * q
    if s <= 4 * dm_mass**2:
        return 0.0
    momentum = np.sqrt(s / 4 - dm_mass**2)
    p1 = np.array([q / 2, 0.0, 0.0, momentum])
    p2 = np.array([q / 2, 0.0, 0.0, -momentum])
    vertex = _slash((p1 + p2) / q) @ (dm_vector * np.eye(4) + dm_axial * _GAMMA5)
    conjugate = _GAMMA0 @ vertex.conj().T @ _GAMMA0
    spin_sum = np.trace(
        (_slash(p1) + dm_mass * np.eye(4))
        @ vertex
        @ (_slash(p2) - dm_mass * np.eye(4))
        @ conjugate
    )
    beta = 2 * momentum / q
    return float(0.5 * spin_sum.real * beta / (8 * np.pi))
