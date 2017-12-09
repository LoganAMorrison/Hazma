
import numpy as np

alpha = 1.0 / 137.0


def fermion(egam, Q, mf):
    val = 0.0

    if 0 < egam and egam < (Q**2 - 2 * mf**2) / (2 * Q):
        e, m = egam / Q, mf / Q

        prefac = alpha / (4 * e * (1 + 2 * m**2) * np.pi *
                          np.sqrt((Q - 2 * e * Q) * (Q - 4 * m**2 * Q)))
        term0 = 2 * np.sqrt(1 - 2 * e - 4 * m**2) * \
            (1 + 2 * m**2 + 2 * e * (-1 + e - 2 * m**2))
        term1 = np.sqrt(1 - 2 * e) * (1 + 2 * (-1 + e) *
                                      e - 4 * e * m**2 - 4 * m**4)
        term2 = -2 * np.arctan(np.sqrt(1 + (4 * m**2) / (-1 + 2 * e)))

        term3 = np.log(np.abs(-1 - np.sqrt(1 + (4 * m**2) / (-1 + 2 * e))))

        term4 = np.log(np.abs(-1 + np.sqrt(1 + (4 * m**2) / (-1 + 2 * e))))

        val = prefac * (term0 + term1 * (term2 + term3 + term4))

    return -val
