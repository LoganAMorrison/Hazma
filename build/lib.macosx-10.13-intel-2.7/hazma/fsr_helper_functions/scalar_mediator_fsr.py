import numpy as np

alpha = 1.0 / 137.0


def fermion(egam, Q, mf):
    val = 0.0

    if 0 < egam and egam < (Q**2 - 2 * mf**2) / (2 * Q):
        e, m = egam / Q, mf / Q

        prefac = (4 * alpha) / (e * (1 - 4 * m**2)**1.5 * np.pi * Q)

        terms = np.array([
            2 * (-1 + 4 * m**2) *
            np.sqrt((1 - 2 * e) * (1 - 2 * e - 4 * m**2)),
            2 * (1 + 2 * (-1 + e) * e - 6 * m**2 + 8 * e * m**2 + 8 *
                 m**4) * np.arctanh(np.sqrt(1 - (4 * m**2) / (1 - 2 * e))),
            (1 + 2 * (-1 + e) * e - 6 * m**2 + 8 * e * m**2 + 8 * m**4) *
            np.log(1 + np.sqrt(1 - (4 * m**2) / (1 - 2 * e))),
            (-1 - 2 * (-1 + e) * e + 6 * m**2 - 8 * e * m**2 - 8 * m**4) *
            np.log(1 - np.sqrt(1 - (4 * m**2) / (1 - 2 * e)))
        ])

        val = np.real(prefac * np.sum(terms))

    return val
