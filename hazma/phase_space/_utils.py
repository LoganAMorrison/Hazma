from typing import Sequence, Dict, Tuple, List
import logging
import itertools

import numpy as np

from hazma.utils import kallen_lambda


def normalize_distribution(probabilities, edges):
    norm = np.sum([p * (edges[i + 1] - edges[i]) for i, p in enumerate(probabilities)])
    if norm <= 0.0:
        if np.min(probabilities) < 0.0:
            logging.warning(f"Negative probabilities encountered: {probabilities}")
            return np.ones_like(probabilities) * np.nan
        return probabilities
    return probabilities / norm


def energy_limits(q: float, masses: Sequence[float]) -> List[Tuple[float, float]]:
    r"""Compute the limits on the energies of each of final-state particles.

    Parameters
    ----------
    q: float
        Center-of-mass energy.
    masses: Sequence[float]
        Masses of the final-state particles.

    Returns
    -------
    limits: list[(float, float)]
        List containing the limits on the energies of each of the final-state
        particles.
    """
    n = len(masses)
    msum = sum(masses)

    def lims(i):
        m = masses[i]
        emin = m
        emax = (q**2 + m**2 - (msum - m) ** 2) / (2 * q)
        return (emin, emax)

    return [lims(i) for i in range(n)]


def invariant_mass_limits(
    q: float, masses: Sequence[float]
) -> Dict[Tuple[int, int], Tuple[float, float]]:
    r"""Compute the limits on the invariant masses of each pair of final-state
    particles.

    Parameters
    ----------
    q: float
        Center-of-mass energy.
    masses: Sequence[float]
        Masses of the final-state particles.

    Returns
    -------
    limits: dict[(int,int), (float,float)]
        Dictionary containing the limits for each pair of final-state
        particles.
    """
    n = len(masses)

    idxs = {i for i in range(n)}
    pairs = itertools.combinations(idxs, 2)
    limits = dict()

    for i, j in pairs:
        msum = sum(masses[k] for k in idxs.symmetric_difference({i, j}))
        mmin = masses[i] + masses[j]
        mmax = q - msum
        limits[(i, j)] = (mmin, mmax)

    return limits


def two_body_phase_space_prefactor(q, m1: float, m2: float):
    r"""Compute the pre-factor of the two-body final-state phase-space integral.

    Parameters
    ----------
    q: float or array-like
        Center-of-mass energy.
    m1, m2: float
        Masses of the final state particles.

    Returns
    -------
    pre: float or array-like
        The prefactor with same shape as `q`.
    """
    s = q**2
    return np.sqrt(np.clip(kallen_lambda(s, m1**2, m2**2), 0.0, None)) / (8 * np.pi * s)
