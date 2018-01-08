"""
Generate histograms of the energies of particles.

* Author - Logan A. Morrison and Adam Coogan
* Date - December 2017
"""

import numpy as np
cimport numpy as np


cdef point_to_energy_list(double[:] point):

    cdef double[:] engs = point[0:4]
    cdef double weight = point[len(point)]

    return engs, weight


def space_to_energy_hist(np.ndarray pts, int num_ps_pts, int num_fsp,
                         int num_bins):
    """
    """
    cdef int i, j
    cdef np.ndarray weights = np.zeros(num_ps_pts, dtype=np.float64)
    cdef np.ndarray hist = np.zeros((num_fsp, num_bins), dtype=np.float64)
    cdef np.ndarray bins = np.zeros((num_fsp, num_bins + 1), dtype=np.float64)
    cdef np.ndarray engs = np.zeros((num_fsp, num_bins), dtype=np.float64)
    cdef np.ndarray probs = np.zeros((num_fsp, 2, num_bins), dtype=np.float64)

    cdef double tot_weight = 0.0

    weights = pts[:, 4 * num_fsp]

    for i in range(num_ps_pts):
        tot_weight += weights[i]

    for i in range(num_ps_pts):
        weights[i] = weights[i] / tot_weight

    for i in range(num_fsp):
        hist[i, :], bins[i, :] = np.histogram(pts[:, 4 * i], bins=num_bins,
                                              weights=weights)
    engs = (bins[:, :-1] + bins[:, 1:]) / 2

    for i in range(num_fsp):
        for j in range(num_bins):
            probs[i, 0, j] = engs[i, j]
            probs[i, 1, j] = hist[i, j]

    return probs
