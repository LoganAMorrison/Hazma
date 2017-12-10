import numpy as np

alpha = 1.0 / 137.0


def fermion(egam, Q, mf):
    val = 0.0

    if 0 < egam and egam < (Q**2 - 2 * mf**2) / (2 * Q):
        e, b = egam / Q, mf / Q

        prefac = 4 * alpha / ((1 - 4 * b**2)**1.5 * e * np.pi * Q)
        term1 = 2 * (-1 + 4 * b**2) * np.sqrt(1 - 2 * e) \
            * np.sqrt(1 - 4 * b**2 - 2 * e)
        term2 = 2 * (1 + 8 * b**4 + 2 * (-1 + e) * e + b**2 * (-6 + 8 * e))\
            * np.arctanh(np.sqrt(1 + (4 * b**2) / (-1 + 2 * e)))

        term3 = (1 + 8 * b**4 + 2 * (-1 + e) * e + b**2 * (-6 + 8 * e)) \
            * np.log(np.abs((-1 - np.sqrt(1 + (4 * b**2) / (-1 + 2 * e)))))

        term4 = (1 + 8 * b**4 + 2 * (-1 + e) * e + b**2 * (-6 + 8 * e)) \
            * np.log(np.abs(-1 + np.sqrt(1 + (4 * b**2) / (-1 + 2 * e))))

        val = prefac * (term1 + term2 + term3 + term4)

    return -val
