"""Axial current tensors with metric ``diag(1, -1, -1, -1)``.

Lorentz indices precede all batch indices. The source-longitudinal direction
is proportional to q; it is distinct from the helicity-zero polarization of
an on-shell spin-one particle, which obeys q.dot(epsilon) = 0.

The functions here validate and broadcast their arguments; the contractions
run in the ``hazma._core.axial`` kernels, which take each batch
component-major, as ``(W, N)``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from hazma._core import axial as _axial

# Keep runtime-resolvable annotations, as required by AGENTS.md.
from hazma.utils import ComplexArray, RealArray  # noqa: TC001

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


def _components(array: np.ndarray, shape: tuple[int, ...], rank: int) -> np.ndarray:
    """Broadcast ``array`` to batch ``shape`` as a C-contiguous ``(W, N)`` array.

    W is the number of Lorentz components in the ``rank`` leading axes and N
    the batch size: the layout every ``hazma._core.axial`` kernel takes, and
    the one a kernel result is reshaped back from.
    """
    batch = np.ascontiguousarray(_broadcast_lorentz(array, shape, rank))
    return batch.reshape(math.prod(array.shape[:rank]), math.prod(shape))


def _momentum(q: ArrayLike) -> tuple[RealArray, RealArray]:
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


def hadronic_tensor(current: ArrayLike) -> ComplexArray:
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
    shape = amplitude.shape[1:]
    tensor = _axial.hadronic_tensor(_components(amplitude, shape, 1))
    return _finite_result(tensor.reshape((4, 4) + shape))


def current_projections(
    current: ArrayLike, q: ArrayLike
) -> tuple[RealArray, RealArray]:
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
    momentum, _ = _momentum(q)
    shape = np.broadcast_shapes(amplitude.shape[1:], momentum.shape[1:])
    transverse, longitudinal = _axial.current_projections(
        _components(amplitude, shape, 1), _components(momentum, shape, 1)
    )
    return (
        _finite_result(transverse.reshape(shape)[()]),
        _finite_result(longitudinal.reshape(shape)[()]),
    )


def tensor_projections(
    tensor: ArrayLike, q: ArrayLike, *, rtol: float = 1e-6
) -> tuple[RealArray, RealArray]:
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
    momentum, _ = _momentum(q)
    shape = np.broadcast_shapes(density.shape[2:], momentum.shape[1:])
    transverse, longitudinal = _axial.tensor_projections(
        _components(density, shape, 2), _components(momentum, shape, 1), float(rtol)
    )
    return transverse.reshape(shape)[()], longitudinal.reshape(shape)[()]


def integrate_spectral_functions(
    current: ArrayLike, q: ArrayLike, weights: ArrayLike, *, sample_axis: int = -1
) -> tuple[RealArray, RealArray]:
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
    # One row of samples per batch element, summed by the kernel.
    densities = [
        np.ascontiguousarray(np.moveaxis(value, sample_axis, -1))
        for value in (transverse, longitudinal, measure)
    ]
    batch = densities[0].shape[:-1]
    rows = (math.prod(batch), densities[0].shape[-1])
    result_t, result_l = _axial.integrate_spectral_functions(
        *(value.reshape(rows) for value in densities)
    )
    return (
        _finite_result(result_t.reshape(batch)[()]),
        _finite_result(result_l.reshape(batch)[()]),
    )
