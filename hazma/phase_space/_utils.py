from typing import Sequence, Dict, Tuple, List
import logging
import itertools

import numpy as np


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

    for (i, j) in pairs:
        msum = sum(masses[k] for k in idxs.symmetric_difference({i, j}))
        mmin = masses[i] + masses[j]
        mmax = q - msum
        limits[(i, j)] = (mmin, mmax)

    return limits
