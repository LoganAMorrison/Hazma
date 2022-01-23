"""
Generate histograms of the energies of particles.

* Author - Logan A. Morrison and Adam Coogan
* Date - December 2017
"""

import numpy as np
cimport numpy as np
import cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def space_to_energy_hist(np.ndarray[np.float64_t, ndim=2] pts, int num_ps_pts,
                         int num_fsp, int num_bins, density=False):
    """
    """
    cdef int i, j
    cdef int weight_index = 4 * num_fsp

    # Define arrays for various information.
    cdef np.ndarray[np.float64_t, ndim=1] weights
    cdef np.ndarray[np.float64_t, ndim=2] bins
    cdef np.ndarray[np.float64_t, ndim=2] engs
    cdef np.ndarray[np.float64_t, ndim=3] probs

    cdef np.ndarray[np.float64_t, ndim=2] means
    cdef np.ndarray[np.float64_t, ndim=2] vars
    cdef np.ndarray[np.float64_t, ndim=2] errs

    # Allocate arrays to store various information.
    weights = np.zeros(num_ps_pts, dtype=np.float64)
    bins = np.zeros((num_fsp, num_bins + 1), dtype=np.float64)
    engs = np.zeros((num_fsp, num_bins), dtype=np.float64)
    probs = np.zeros((num_fsp, 2, num_bins), dtype=np.float64)

    means = np.zeros((num_fsp, num_bins), dtype=np.float64)
    vars = np.zeros((num_fsp, num_bins), dtype=np.float64)
    errs = np.zeros((num_fsp, num_bins), dtype=np.float64)

    # Extract the weights from phase space points.
    for i in range(num_ps_pts):
        weights[i] = pts[i, weight_index]

    for i in range(num_fsp):

        means[i, :], bins[i, :] = np.histogram(pts[:, 4 * i], bins=num_bins,
                                               weights=weights, density=density)

        vars[i, :] = np.histogram(pts[:, 4 * i], bins=num_bins,
                                  weights=weights * weights, density=density)[0]

        errs[i, :] = np.sqrt((vars[i, :] - means[i, :]**2))

        # Normalize the distributions by the number of phase space points,
        # but only if density is False, otherwise the density is True case
        # will not be normalized to 1.
        if density is False:
            means[i, :] = means[i, :] * (1.0 / num_ps_pts)
            vars[i, :] = vars[i, :] * (1.0 / num_ps_pts)
            errs[i, :] = errs[i, :] * (1.0 / num_ps_pts)

    for i in range(num_fsp):
        for j in range(num_bins):
            engs[i, j] = (bins[i, j+1] + bins[i, j]) / 2

    for i in range(num_fsp):
        for j in range(num_bins):
            probs[i, 0, j] = engs[i, j]
            if density is True:
                probs[i, 1, j] = means[i, j]
            else:
                probs[i, 1, j] = means[i, j] / (bins[i, j+1] - bins[i, j])

    return probs, errs
