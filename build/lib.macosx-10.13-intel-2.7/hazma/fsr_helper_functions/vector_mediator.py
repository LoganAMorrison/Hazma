import numpy as np

alpha = 1.0 / 137.0


def fermion(egam, Q, mf):
    val = 0.0

    e, m = egam / Q, mf / Q

    if 0 < e and e < 0.5 * (1.0 - 2 * m**2):

        pre_factor = alpha / (4 * e *
                              np.sqrt(1 - 4 * m**2) * (1 + 2 * m**2) *
                              np.pi * np.sqrt(Q * (Q - 2 * e * Q)))

        terms = np.array([
            2 * np.sqrt(1 - 2 * e - 4 * m**2) *
            (1 + 2 * m**2 + 2 * e * (-1 + e - 2 * m**2)),
            -2 * np.sqrt(1 - 2 * e) *
            (1 + 2 * (-1 + e) * e - 4 * e * m**2 - 4 * m**4) * np.arctanh(
                np.sqrt(1 - 2 * e - 4 * m**2) /
                np.sqrt(1 - 2 * e)),
            np.sqrt(1 - 2 * e) *
            (1 + 2 * (-1 + e) * e - 4 * e * m**2 - 4 * m**4) *
            np.log(1 + np.sqrt(1 - 2 * e - 4 * m**2) /
                   np.sqrt(1 - 2 * e)),
            - (np.sqrt(1 - 2 * e) *
               (1 + 2 * (-1 + e) * e - 4 * e * m**2 - 4 * m**4) *
               np.log(1 - np.sqrt(1 - 2 * e - 4 * m**2) /
                      np.sqrt(1 - 2 * e)))
        ])

        val = np.real(pre_factor * np.sum(terms))

    return val
